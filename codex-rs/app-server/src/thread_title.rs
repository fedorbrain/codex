//! Server-side auto-generation of a thread title on the first completed user turn.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;

use codex_app_server_protocol::{ServerNotification, ThreadNameUpdatedNotification};
use codex_core::StartThreadOptions;
use codex_core::ThreadManager;
use codex_core::TurnInputRequest;
use codex_core::CodexThread;
use codex_core::config::Config;
use codex_features::Feature;
use codex_protocol::ThreadId;
use codex_protocol::config_types::WebSearchMode;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::Op;
use codex_protocol::protocol::TurnEnvironmentRequest;
use codex_protocol::user_input::UserInput;
use codex_thread_store::ReadThreadParams;
use codex_thread_store::ThreadMetadataPatch as StoreThreadMetadataPatch;
use serde_json::json;

use crate::outgoing_message::ThreadScopedOutgoingMessageSender;

const THREAD_TITLE_MAX_CHARS: usize = 36;

/// Upper bound for one title-generation turn. The turn is a single structured completion, so
/// this only bounds a model that never answers; tool misuse is cut off earlier instead.
const THREAD_TITLE_TIMEOUT: Duration = Duration::from_secs(/*secs*/ 120);

/// Shutdown bound for the throwaway thread once its turn finished or was abandoned.
const THREAD_TITLE_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(/*secs*/ 10);

/// Tools a title-generation turn never needs.
///
/// The generator thread inherits the parent thread config, so without this list a model is
/// offered the whole production tool surface and starts "researching" its own title with
/// `exec_command` instead of answering once. Disabling a pinned feature is a no-op.
const THREAD_TITLE_DISABLED_FEATURES: &[Feature] = &[
    Feature::ShellTool,
    Feature::UnifiedExec,
    Feature::UnifiedExecTty,
    Feature::ShellZshFork,
    Feature::UnifiedExecZshFork,
    Feature::StableEnvironmentTools,
    Feature::ShellSnapshot,
    Feature::ShellSnapshotV2,
    Feature::DeferredExecutor,
    Feature::ExecutorCapabilityDiscovery,
    Feature::ViewImage,
    Feature::SleepTool,
    Feature::RequestPermissionsTool,
    Feature::ExecPermissionApprovals,
    Feature::WriteStdinApproval,
    Feature::ApplyPatchStreamingEvents,
    Feature::ApplyPatchFreeform,
    Feature::SearchTool,
    Feature::JsRepl,
    Feature::JsReplToolsOnly,
    Feature::CodeMode,
    Feature::CodeModeToolSearch,
    Feature::CodeModeHost,
    Feature::CodeModePrewarm,
    Feature::CodeModeInterrupt,
    Feature::CodeModeOnly,
    Feature::CodeModeOnlyStrictThirdPartyTools,
    Feature::Plugins,
    Feature::PluginHooks,
    Feature::RecommendedPlugins,
    Feature::RemotePlugin,
    Feature::PluginSharing,
    Feature::Apps,
    Feature::EnableMcpApps,
    Feature::ToolSearch,
    Feature::ToolSearchAlwaysDeferMcpTools,
    Feature::ToolSuggest,
    Feature::SkillSearch,
    Feature::SkillMcpDependencyInstall,
    Feature::SkillEnvVarDependencyPrompt,
    Feature::Collab,
    Feature::MultiAgentV2,
    Feature::MultiAgentV2DynamicTools,
    Feature::MultiAgentMode,
    Feature::CollaborationModes,
    Feature::AgentMessageBoard,
    Feature::SpawnCsv,
    Feature::SendAsyncMessage,
    Feature::SendMessageToUserAsync,
    Feature::DefaultModeRequestUserInput,
    Feature::Steer,
    Feature::Goals,
    Feature::TokenBudget,
    Feature::ContextManagement,
    Feature::RolloutBudget,
    Feature::CurrentTimeReminder,
    Feature::Personality,
    Feature::Artifact,
    Feature::ImageGeneration,
    Feature::UnifiedImageBudget,
    Feature::ImageResizeNotice,
    Feature::ImageDetailOriginal,
    Feature::ResizeAllImages,
    Feature::MemoryTool,
    Feature::Chronicle,
    Feature::ExternalAgentMemoryImport,
    Feature::Worktrees,
    Feature::NetworkProxy,
    Feature::WorkspaceDependencies,
    Feature::InAppBrowser,
    Feature::BrowserUse,
    Feature::BrowserUseExternal,
    Feature::BrowserUseFullCdpAccess,
    Feature::BrowserAnnotationApi,
    Feature::ComputerUse,
    Feature::InAppChat,
    Feature::InAppVoice,
    Feature::InAppDictation,
    Feature::InAppLocalAutomation,
    Feature::InAppUpdates,
    Feature::WebSearchRequest,
    Feature::WebSearchCached,
    Feature::StandaloneWebSearch,
    Feature::UnavailableDummyTools,
    Feature::CodexHooks,
];

/// Global semaphore limiting concurrent title generations.
static THREAD_LIST_STATE_PERMIT: OnceLock<Arc<Semaphore>> = OnceLock::new();

fn thread_list_state_permit() -> &'static Arc<Semaphore> {
    THREAD_LIST_STATE_PERMIT
        .get_or_init(|| Arc::new(Semaphore::new(4)))
}

fn thread_title_output_schema() -> serde_json::Value {
    json!({
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "minLength": 1,
                "maxLength": THREAD_TITLE_MAX_CHARS,
            },
        },
        "required": ["title"],
        "additionalProperties": false,
    })
}

fn thread_title_prompt(user_message: &str) -> String {
    let instructions = format!(
        "Generate a concise, single-line task title of at most {THREAD_TITLE_MAX_CHARS} characters. Start with an imperative verb. Write in the user language. Do not answer the request."
    );
    let truncated: String = user_message.chars().take(500).collect();
    format!("{}\n{}", instructions, truncated)
}

/// Narrow an inherited thread config down to a bare structured completion.
///
/// Only the provider, model, and reasoning settings are meant to be inherited. Everything that
/// advertises a tool is dropped so the generator cannot turn a one-line title into a tool loop.
fn isolate_title_generation_config(config: &mut Config) {
    for feature in THREAD_TITLE_DISABLED_FEATURES {
        let _ = config.features.disable(*feature);
    }
    config.update_plan_enabled = false;
    config.experimental_request_user_input_enabled = false;
    // Prompt blocks that only describe tools this thread no longer has.
    config.include_skill_instructions = false;
    config.include_apps_instructions = false;
    config.include_collaboration_mode_instructions = false;
    config.include_permissions_instructions = false;
    config.cloud_skill_enabled = false;
    config.orchestrator_mcp_enabled = false;
    // Plugin and user MCP servers contribute most of the advertised tools, and every one of
    // them would be started for this throwaway thread.
    let _ = config.mcp_servers.set(HashMap::new());
    let _ = config.web_search_mode.set(WebSearchMode::Disabled);
    config.web_search_config = None;
}

pub(crate) fn spawn_thread_title_generation(
    conversation_id: ThreadId,
    conversation: Arc<CodexThread>,
    thread_manager: Arc<ThreadManager>,
    outgoing: ThreadScopedOutgoingMessageSender,
) {
    tokio::spawn(async move {
        if let Err(err) =
            generate_and_apply_title(conversation_id, conversation, thread_manager, outgoing).await
        {
            tracing::debug!(?err, "thread title generation skipped");
        }
    });
}

async fn generate_and_apply_title(
    conversation_id: ThreadId,
    conversation: Arc<CodexThread>,
    thread_manager: Arc<ThreadManager>,
    outgoing: ThreadScopedOutgoingMessageSender,
) -> Result<(), String> {

    let stored = thread_manager
        .read_stored_thread(ReadThreadParams {
            thread_id: conversation_id,
            include_archived: true,
            include_history: false,
        })
        .await
        .map_err(|e| {
            format!("read_stored_thread: {e}")
        })?;

    let first_user = stored
        .first_user_message
        .clone()
        .unwrap_or_else(|| stored.preview.trim().to_string());
    if first_user.is_empty() {
        return Ok(());
    }

    let mut config = (*conversation.config().await).clone();
    config.ephemeral = true;
    // Generate the title with the thread's own provider and model. A hardcoded alias here
    // breaks whenever the configured provider no longer serves that alias (for example an
    // Ollama-only alias while the active provider is a local server), and the title then
    // silently never appears.
    isolate_title_generation_config(&mut config);

    let mut start_options = StartThreadOptions::new(config);
    // Without an environment no environment-backed tool can be advertised at all.
    start_options.environments = Some(Vec::<TurnEnvironmentRequest>::new());
    start_options.dynamic_tools = Vec::new();

    let new_thread = thread_manager
        .start_thread(start_options)
        .await
        .map_err(|e| {
            format!("start_thread: {e}")
        })?;

    new_thread
        .thread
        .start_or_steer_turn(
            TurnInputRequest::user_input(vec![UserInput::Text {
                text: thread_title_prompt(&first_user),
                text_elements: Vec::new(),
            }])
            .on_start(codex_core::TurnStartOptions {
                final_output_json_schema: Some(thread_title_output_schema()),
                ..Default::default()
            }),
        )
        .await
        .map_err(|e| {
            format!("start_turn: {e}")
        })?;

    let candidate = collect_title_candidate(&new_thread.thread).await;

    // Removing the thread from the registry alone leaves a still-running turn submitting model
    // requests forever, so always interrupt and shut the throwaway session down.
    let _ = new_thread.thread.submit(Op::Interrupt).await;
    let _ = thread_manager
        .remove_thread_for_client(&new_thread.thread_id)
        .await;
    let _ = tokio::time::timeout(
        THREAD_TITLE_SHUTDOWN_TIMEOUT,
        new_thread.thread.shutdown_and_wait(),
    )
    .await;

    let Some(raw) = candidate else {
        return Ok(());
    };

    let trimmed = raw.trim();
    let name: Option<String> = if trimmed.starts_with("{") {
        serde_json::from_str::<serde_json::Value>(trimmed)
            .ok()
            .and_then(|v| v.get("title").and_then(|t| t.as_str()).map(|s| s.to_string()))
    } else if !trimmed.is_empty() && trimmed != "null" {
        Some(trimmed.to_string())
    } else {
        None
    };

    let Some(name) = name else {
        return Ok(());
    };
    if name.trim().is_empty() || name.trim() == first_user.trim() {
        return Ok(());
    }

    let _permit = thread_list_state_permit().clone()
        .acquire_owned()
        .await
        .map_err(|e| {
            format!("permit: {e}")
        })?;

    thread_manager
        .update_thread_metadata(
            conversation_id,
            StoreThreadMetadataPatch {
                name: Some(Some(name.clone())),
                ..Default::default()
            },
            false,
        )
        .await
        .map_err(|e| {
            format!("update_metadata: {e}")
        })?;

    outgoing
        .send_server_notification(ServerNotification::ThreadNameUpdated(
            ThreadNameUpdatedNotification {
                thread_id: conversation_id.to_string(),
                thread_name: Some(name.clone()),
            },
        ))
        .await;

    Ok(())
}

/// Wait for the title answer, abandoning the turn as soon as it starts using tools.
///
/// A title turn has no tools to call, so the first tool event means the model went off the
/// rails; keeping it alive is what produced the endless `exec_command` loop.
async fn collect_title_candidate(thread: &Arc<CodexThread>) -> Option<String> {
    let deadline = Instant::now() + THREAD_TITLE_TIMEOUT;
    let mut candidate: Option<String> = None;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            break;
        }
        let ev = match tokio::time::timeout(remaining, thread.next_event()).await {
            Ok(Ok(ev)) => ev,
            Ok(Err(_)) | Err(_) => break,
        };
        match &ev.msg {
            EventMsg::ItemCompleted(item) => match &item.item {
                codex_protocol::items::TurnItem::AgentMessage(msg) => {
                    if let Some(content) = msg.content.first()
                        && let codex_protocol::items::AgentMessageContent::Text { text } = content
                    {
                        candidate = Some(text.clone());
                        break;
                    }
                }
                codex_protocol::items::TurnItem::CommandExecution(_)
                | codex_protocol::items::TurnItem::McpToolCall(_)
                | codex_protocol::items::TurnItem::DynamicToolCall(_)
                | codex_protocol::items::TurnItem::CollabAgentToolCall(_)
                | codex_protocol::items::TurnItem::SubAgentActivity(_)
                | codex_protocol::items::TurnItem::WebSearch(_)
                | codex_protocol::items::TurnItem::ImageGeneration(_)
                | codex_protocol::items::TurnItem::ImageView(_)
                | codex_protocol::items::TurnItem::FileChange(_) => break,
                _ => {}
            },
            EventMsg::AgentMessage(ev) => {
                candidate = Some(ev.message.clone());
                break;
            }
            EventMsg::ExecCommandBegin(_)
            | EventMsg::McpToolCallBegin(_)
            | EventMsg::WebSearchBegin(_)
            | EventMsg::ImageGenerationBegin(_)
            | EventMsg::PatchApplyBegin(_)
            | EventMsg::CollabAgentSpawnBegin(_)
            | EventMsg::CollabAgentInteractionBegin(_) => break,
            EventMsg::TurnComplete(_) => break,
            EventMsg::TurnAborted(_) => break,
            _ => {}
        }
    }
    candidate
}