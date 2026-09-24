//! Server-side auto-generation of a thread title on the first completed user turn.

use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;

use codex_app_server_protocol::{ServerNotification, ThreadNameUpdatedNotification};
use codex_core::{CodexThread, StartThreadOptions, ThreadManager, TurnInputRequest};
use codex_protocol::ThreadId;
use codex_protocol::protocol::EventMsg;
use codex_protocol::user_input::UserInput;
use codex_thread_store::ReadThreadParams;
use codex_thread_store::ThreadMetadataPatch as StoreThreadMetadataPatch;
use serde_json::json;

use crate::outgoing_message::ThreadScopedOutgoingMessageSender;

const THREAD_TITLE_MAX_CHARS: usize = 36;

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
    config.model = Some("qwen38-fast".to_string());

    let new_thread = thread_manager
        .start_thread(StartThreadOptions::new(config))
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

    let deadline = Instant::now() + Duration::from_secs(120);
    let mut candidate: Option<String> = None;
    while Instant::now() < deadline {
        let ev = new_thread
            .thread
            .next_event()
            .await
            .map_err(|e| {
                e.to_string()
            })?;
        match &ev.msg {
            EventMsg::ItemCompleted(item) => {
                if let codex_protocol::items::TurnItem::AgentMessage(msg) = &item.item
                    && let Some(c) = msg.content.first()
                    && let codex_protocol::items::AgentMessageContent::Text { text } = c
                {
                    candidate = Some(text.clone());
                }
            }
            EventMsg::AgentMessage(ev) => {
                candidate = Some(ev.message.clone());
            }
            EventMsg::TurnComplete(_) => break,
            EventMsg::TurnAborted(_) => break,
            _ => {}
        }
    }

    let _ = thread_manager.remove_thread_for_client(&new_thread.thread_id);

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

    let _permit = thread_list_state_permit()
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
