//! A chat request as the worker's OpenAI API wants it, and its server-sent
//! events turned back into ordered [`Event`]s.

use nyx_ai_proto::{ChatRequest, Content, Event, ResponseFormat, ToolCallDelta, Usage};
use serde_json::{Value, json};

/// The worker's request body. Only fields AI Core checked go through.
pub fn body(req: &ChatRequest, stream: bool) -> Value {
    let messages: Vec<Value> = req
        .messages
        .iter()
        .map(|m| {
            let mut item = json!({"role": m.role, "content": match &m.content {
                Content::Text(text) => json!(text),
                Content::Parts(parts) => json!(parts),
            }});
            if !m.tool_calls.is_empty() {
                item["tool_calls"] = json!(m.tool_calls);
            }
            if let Some(id) = &m.tool_call_id {
                item["tool_call_id"] = json!(id);
            }
            item
        })
        .collect();
    let mut body = json!({"model": req.model, "messages": messages, "stream": stream});
    if stream {
        body["stream_options"] = json!({"include_usage": true});
    }
    if !req.tools.is_empty() {
        body["tools"] = json!(req.tools);
    }
    if let Some(choice) = &req.tool_choice {
        body["tool_choice"] = choice.clone();
    }
    if let Some(grammar) = &req.grammar {
        body["grammar"] = json!(grammar);
    }
    match &req.response_format {
        Some(ResponseFormat::JsonObject) => body["response_format"] = json!({"type": "json_object"}),
        Some(ResponseFormat::JsonSchema { schema }) => body["response_format"] = json!({"type": "json_schema", "schema": schema}),
        Some(ResponseFormat::Text) | None => {}
    }
    let s = &req.sampling;
    let set = |body: &mut Value, key: &str, value: Option<Value>| {
        if let Some(v) = value {
            body[key] = v;
        }
    };
    set(&mut body, "temperature", s.temperature.map(|v| json!(v)));
    set(&mut body, "top_p", s.top_p.map(|v| json!(v)));
    set(&mut body, "top_k", s.top_k.map(|v| json!(v)));
    set(&mut body, "min_p", s.min_p.map(|v| json!(v)));
    set(&mut body, "repeat_penalty", s.repeat_penalty.map(|v| json!(v)));
    set(&mut body, "presence_penalty", s.presence_penalty.map(|v| json!(v)));
    set(&mut body, "frequency_penalty", s.frequency_penalty.map(|v| json!(v)));
    set(&mut body, "seed", s.seed.map(|v| json!(v)));
    set(&mut body, "max_tokens", s.max_tokens.map(|v| json!(v)));
    if !s.stop.is_empty() {
        body["stop"] = json!(s.stop);
    }
    if let Some(effort) = &req.reasoning_effort
        && req.thinking != Some(false)
    {
        body["reasoning_effort"] = json!(effort);
    }
    if let Some(thinking) = req.thinking {
        body["chat_template_kwargs"] = json!({"enable_thinking": thinking});
    }
    body
}

#[derive(Debug, PartialEq)]
pub enum Item {
    Delta(Event),
    Error(String),
    Done,
}

/// Collects what the stream says about the reply as a whole.
#[derive(Debug, Default)]
pub struct Sse {
    buffer: Vec<u8>,
    pub finish_reason: String,
    pub usage: Usage,
    /// The worker's own timings: (prompt ms, tokens per second).
    pub timings: Option<(f64, f64)>,
    pub tokens: u64,
}

impl Sse {
    pub fn push(&mut self, chunk: &[u8]) -> Vec<Item> {
        self.buffer.extend_from_slice(chunk);
        let mut out = Vec::new();
        while let Some(end) = self.buffer.iter().position(|b| *b == b'\n') {
            let line: Vec<u8> = self.buffer.drain(..=end).collect();
            let line = String::from_utf8_lossy(&line);
            let Some(data) = line.trim().strip_prefix("data:").map(str::trim) else { continue };
            if data == "[DONE]" {
                out.push(Item::Done);
                continue;
            }
            let Ok(value) = serde_json::from_str::<Value>(data) else { continue };
            if let Some(error) = value.get("error") {
                let message = error["message"].as_str().or(error.as_str()).unwrap_or("generation failed");
                out.push(Item::Error(message.chars().take(300).collect()));
                continue;
            }
            let choice = &value["choices"][0];
            let delta = &choice["delta"];
            let content = delta["content"].as_str().unwrap_or("").to_string();
            let reasoning = delta["reasoning_content"].as_str().unwrap_or("").to_string();
            let tool_calls: Vec<ToolCallDelta> = delta["tool_calls"]
                .as_array()
                .into_iter()
                .flatten()
                .map(|part| ToolCallDelta {
                    index: part["index"].as_u64().unwrap_or(0) as u32,
                    id: part["id"].as_str().map(String::from),
                    name: part["function"]["name"].as_str().map(String::from),
                    arguments: part["function"]["arguments"].as_str().map(String::from),
                })
                .collect();
            if !content.is_empty() || !reasoning.is_empty() || !tool_calls.is_empty() {
                self.tokens += 1;
                out.push(Item::Delta(Event::Delta { content, reasoning, tool_calls }));
            }
            if let Some(reason) = choice["finish_reason"].as_str() {
                self.finish_reason = reason.into();
            }
            if let Some(usage) = value["usage"].as_object() {
                self.usage.prompt_tokens = usage.get("prompt_tokens").and_then(Value::as_u64).unwrap_or(0);
                self.usage.completion_tokens = usage.get("completion_tokens").and_then(Value::as_u64).unwrap_or(0);
            }
            if let Some(t) = value["timings"].as_object() {
                let prompt_ms = t.get("prompt_ms").and_then(Value::as_f64).unwrap_or(0.0);
                let per_second = t.get("predicted_per_second").and_then(Value::as_f64).unwrap_or(0.0);
                self.timings = Some((prompt_ms, per_second));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nyx_ai_proto::{Message, Part, Role, Sampling};

    #[test]
    fn body_carries_checked_fields() {
        let req = ChatRequest {
            model: "m".into(),
            messages: vec![
                Message::text(Role::User, "hi"),
                Message { role: Role::User, content: Content::Parts(vec![Part::Text { text: "x".into() }]), tool_calls: vec![], tool_call_id: None },
            ],
            response_format: Some(ResponseFormat::JsonObject),
            sampling: Sampling { temperature: Some(0.5), stop: vec!["END".into()], ..Default::default() },
            reasoning_effort: Some("high".into()),
            thinking: Some(false),
            ..Default::default()
        };
        let b = body(&req, true);
        assert_eq!(b["messages"][0], json!({"role": "user", "content": "hi"}));
        assert_eq!(b["messages"][1]["content"][0], json!({"type": "text", "text": "x"}));
        assert_eq!(b["response_format"]["type"], "json_object");
        assert_eq!(b["temperature"], 0.5);
        assert!(b.get("reasoning_effort").is_none());
        assert_eq!(b["chat_template_kwargs"]["enable_thinking"], false);
        assert_eq!(b["stream_options"]["include_usage"], true);
        assert!(b.get("grammar").is_none());
        let grammar = ChatRequest { model: "m".into(), messages: vec![Message::text(Role::User, "hi")], grammar: Some("root ::= \"a\"".into()), ..Default::default() };
        assert_eq!(body(&grammar, false)["grammar"], "root ::= \"a\"");
    }

    #[test]
    fn parses_split_events() {
        let mut sse = Sse::default();
        let a = "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"hm\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"con";
        let b = "tent\":\"Hi\",\"tool_calls\":[{\"index\":0,\"id\":\"c\",\"function\":{\"name\":\"calc\",\"arguments\":\"{}\"}}]}}]}\n\n";
        let c = "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":9,\"completion_tokens\":3},\"timings\":{\"prompt_ms\":12.5,\"predicted_per_second\":40.0}}\n\ndata: [DONE]\n\n";
        let mut items = sse.push(a.as_bytes());
        items.extend(sse.push(b.as_bytes()));
        items.extend(sse.push(c.as_bytes()));
        assert_eq!(items.len(), 3);
        assert!(matches!(&items[0], Item::Delta(Event::Delta { reasoning, .. }) if reasoning == "hm"));
        assert!(matches!(&items[1], Item::Delta(Event::Delta { content, tool_calls, .. }) if content == "Hi" && tool_calls[0].name.as_deref() == Some("calc")));
        assert_eq!(items[2], Item::Done);
        assert_eq!(sse.finish_reason, "tool_calls");
        assert_eq!(sse.usage.completion_tokens, 3);
        assert_eq!(sse.timings, Some((12.5, 40.0)));
        assert_eq!(sse.push(b"data: {\"error\":{\"message\":\"context full\"}}\n"), [Item::Error("context full".into())]);
    }
}
