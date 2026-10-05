//! R95 (opt-in tooling): per-agent TOKEN LEDGER. Two strictly separate bases so nothing is mislabelled:
//! * `counted`: API-reported `usage` of recorded model calls (input / cache-write / cache-read / output tokens), e.g. from Claude Code transcripts (`*.jsonl`, one JSON object per line);
//! * `proxy`: estimates (e.g. 1.33 x words) that the mesh computes itself for routes it never sent to a model.
//! Only counts are kept: `parse_transcript` reads `type == "assistant"` lines' `message.id` + `message.usage` and never stores prompt or tool content.
//! Price-weighted units use caller-supplied ratios (`Pricing::default()` = input 1, cache-write 1.25, cache-read 0.1, output 5: an ASSUMPTION about billing, not a measurement).
//! RD-LOG R87 (python) / R95 (this module) report the recorded-session numbers.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

/// token counts of ONE model call (API `usage`)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub input: u64,
    pub cache_write: u64,
    pub cache_read: u64,
    pub output: u64,
}
impl Usage {
    /// context the call read or wrote (everything but the output)
    pub fn context(&self) -> u64 {
        self.input + self.cache_write + self.cache_read
    }
    pub fn total(&self) -> u64 {
        self.context() + self.output
    }
}

/// price ratios (relative to one fresh input token)
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Pricing {
    pub input: f64,
    pub cache_write: f64,
    pub cache_read: f64,
    pub output: f64,
}
impl Default for Pricing {
    fn default() -> Self {
        Pricing { input: 1.0, cache_write: 1.25, cache_read: 0.1, output: 5.0 }
    }
}
impl Pricing {
    pub fn units(&self, u: &Usage) -> f64 {
        self.input * u.input as f64 + self.cache_write * u.cache_write as f64 + self.cache_read * u.cache_read as f64 + self.output * u.output as f64
    }
}

/// aggregate for one agent label
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AgentTokens {
    /// sessions / spawns recorded under this label
    pub runs: u64,
    pub calls: u64,
    pub usage: Usage,
    /// context size (input + cache) of the FIRST call of each run: the fixed per-agent prefix
    pub first_ctx: Vec<u64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Ledger {
    /// API-counted tokens by agent label
    pub counted: BTreeMap<String, AgentTokens>,
    /// mesh-side ESTIMATES by "agent/route" label (never mixed into `counted`)
    pub proxy: BTreeMap<String, u64>,
}

impl Ledger {
    pub fn new() -> Self {
        Ledger::default()
    }
    /// record one run (a session or a spawned sub-agent) whose model calls are `calls` in time order
    pub fn add_run(&mut self, agent: &str, calls: &[Usage]) {
        if calls.is_empty() {
            return;
        }
        let a = self.counted.entry(agent.to_string()).or_default();
        a.runs += 1;
        a.calls += calls.len() as u64;
        a.first_ctx.push(calls[0].context());
        for c in calls {
            a.usage.input += c.input;
            a.usage.cache_write += c.cache_write;
            a.usage.cache_read += c.cache_read;
            a.usage.output += c.output;
        }
    }
    /// record a mesh-side estimate (e.g. tokens NOT sent to a model because a local tool answered)
    pub fn add_proxy(&mut self, label: &str, tokens: u64) {
        *self.proxy.entry(label.to_string()).or_insert(0) += tokens;
    }
    /// union of two ledgers (sums; first-call samples are concatenated)
    pub fn merge(&mut self, other: &Ledger) {
        for (k, v) in &other.counted {
            let a = self.counted.entry(k.clone()).or_default();
            a.runs += v.runs;
            a.calls += v.calls;
            a.usage.input += v.usage.input;
            a.usage.cache_write += v.usage.cache_write;
            a.usage.cache_read += v.usage.cache_read;
            a.usage.output += v.usage.output;
            a.first_ctx.extend_from_slice(&v.first_ctx);
        }
        for (k, v) in &other.proxy {
            *self.proxy.entry(k.clone()).or_insert(0) += v;
        }
    }
    pub fn total_counted(&self) -> Usage {
        let mut t = Usage::default();
        for a in self.counted.values() {
            t.input += a.usage.input;
            t.cache_write += a.usage.cache_write;
            t.cache_read += a.usage.cache_read;
            t.output += a.usage.output;
        }
        t
    }
    /// price-weighted units per agent label
    pub fn units(&self, p: &Pricing) -> BTreeMap<String, f64> {
        self.counted.iter().map(|(k, v)| (k.clone(), p.units(&v.usage))).collect()
    }
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }
    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }
}

/// median of the recorded first-call contexts (None when empty)
pub fn median_first_ctx(a: &AgentTokens) -> Option<u64> {
    if a.first_ctx.is_empty() {
        return None;
    }
    let mut v = a.first_ctx.clone();
    v.sort_unstable();
    Some(v[v.len() / 2])
}

/// parse one transcript (JSONL text) into per-call usages in time order. Streaming chunks of the same API message share `message.id`: the chunk with the largest `output_tokens` wins.
/// Lines that are not assistant messages with usage are ignored; no content is retained.
pub fn parse_transcript(text: &str) -> Vec<Usage> {
    let mut by_id: HashMap<String, (String, Usage)> = HashMap::new();
    let mut order = 0usize;
    for line in text.lines() {
        if !line.contains("\"usage\"") {
            continue;
        }
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else { continue };
        if v.get("type").and_then(|x| x.as_str()) != Some("assistant") {
            continue;
        }
        let Some(m) = v.get("message") else { continue };
        let Some(u) = m.get("usage") else { continue };
        let g = |k: &str| u.get(k).and_then(|x| x.as_u64()).unwrap_or(0);
        let usage = Usage { input: g("input_tokens"), cache_write: g("cache_creation_input_tokens"), cache_read: g("cache_read_input_tokens"), output: g("output_tokens") };
        let id = m.get("id").and_then(|x| x.as_str()).map(|s| s.to_string()).or_else(|| v.get("uuid").and_then(|x| x.as_str()).map(|s| s.to_string())).unwrap_or_else(|| {
            order += 1;
            format!("anon{order}")
        });
        let ts = v.get("timestamp").and_then(|x| x.as_str()).unwrap_or("").to_string();
        match by_id.get_mut(&id) {
            None => {
                by_id.insert(id, (ts, usage));
            }
            Some(prev) => {
                if usage.output >= prev.1.output {
                    prev.1 = usage;
                }
            }
        }
    }
    let mut v: Vec<(String, Usage)> = by_id.into_values().collect();
    v.sort_by(|a, b| a.0.cmp(&b.0));
    v.into_iter().map(|x| x.1).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_dedupes_streaming_chunks_orders_by_time_and_keeps_no_content() {
        let t = r#"{"type":"assistant","timestamp":"2026-01-01T00:00:02Z","message":{"id":"b","content":[{"type":"text","text":"SECRET"}],"usage":{"input_tokens":2,"cache_creation_input_tokens":10,"cache_read_input_tokens":100,"output_tokens":5}}}
{"type":"assistant","timestamp":"2026-01-01T00:00:01Z","message":{"id":"a","usage":{"input_tokens":1,"cache_creation_input_tokens":0,"cache_read_input_tokens":50,"output_tokens":3}}}
{"type":"assistant","timestamp":"2026-01-01T00:00:01Z","message":{"id":"a","usage":{"input_tokens":1,"cache_creation_input_tokens":0,"cache_read_input_tokens":50,"output_tokens":9}}}
{"type":"user","message":{"usage":{"input_tokens":999}}}
not json "usage"
"#;
        let c = parse_transcript(t);
        assert_eq!(c, vec![Usage { input: 1, cache_write: 0, cache_read: 50, output: 9 }, Usage { input: 2, cache_write: 10, cache_read: 100, output: 5 }]);
        let mut l = Ledger::new();
        l.add_run("general-purpose", &c);
        l.add_run("general-purpose", &c);
        l.add_proxy("router/calc", 40);
        let a = &l.counted["general-purpose"];
        assert_eq!((a.runs, a.calls, a.usage.output, a.usage.cache_read), (2, 4, 28, 300));
        assert_eq!(median_first_ctx(a), Some(51));
        // price units: 1*6 + 1.25*20 + 0.1*300 + 5*28 = 201
        assert!((l.units(&Pricing::default())["general-purpose"] - 201.0).abs() < 1e-9);
        // proxy tokens never leak into the counted totals
        assert_eq!(l.total_counted().total(), 6 + 20 + 300 + 28);
        let mut m = Ledger::new();
        m.merge(&l);
        m.merge(&l);
        assert_eq!(m.counted["general-purpose"].calls, 8);
        assert_eq!(m.proxy["router/calc"], 80);
        assert_eq!(Ledger::from_json(&m.to_json()).unwrap(), m);
        assert!(!l.to_json().contains("SECRET"));
    }
}
