//! Shared helpers for the `neural-mesh serve` integration tests.
#![allow(dead_code)]

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, ExitStatus, Stdio};

use serde_json::Value;

pub const BIN: &str = env!("CARGO_BIN_EXE_neural-mesh");

pub fn tmp_state(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("nm-serve-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir.join("state.json")
}

/// One-shot CLI call; returns the parsed stdout JSON.
pub fn cli(args: &[&str]) -> Value {
    let out = Command::new(BIN).args(args).output().unwrap();
    serde_json::from_slice(&out.stdout).unwrap_or_else(|e| panic!("bad CLI output {:?}: {e}", String::from_utf8_lossy(&out.stdout)))
}

pub fn cli_stats(p: &Path) -> Value {
    cli(&["stats", "--state", p.to_str().unwrap()])
}

pub fn cli_feedback(p: &Path, agent: &str, reward: &str) -> Value {
    cli(&["feedback", "--state", p.to_str().unwrap(), "--features", "x", "--agent", agent, "--reward", reward])
}

pub struct Srv {
    pub child: Child,
    pub stdin: Option<ChildStdin>,
    pub out: BufReader<ChildStdout>,
}

impl Srv {
    pub fn spawn(state: &Path, extra: &[&str]) -> Srv {
        let mut child = Command::new(BIN)
            .args(["serve", "--state", state.to_str().unwrap()])
            .args(extra)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let stdin = child.stdin.take();
        let out = BufReader::new(child.stdout.take().unwrap());
        Srv { child, stdin, out }
    }

    pub fn stdio(state: &Path, extra: &[&str]) -> Srv {
        let mut a = vec!["--stdio"];
        a.extend_from_slice(extra);
        Srv::spawn(state, &a)
    }

    pub fn send_raw(&mut self, bytes: &[u8]) {
        let w = self.stdin.as_mut().unwrap();
        w.write_all(bytes).unwrap();
        w.flush().unwrap();
    }

    pub fn read(&mut self) -> Value {
        let mut line = String::new();
        self.out.read_line(&mut line).unwrap();
        serde_json::from_str(&line).unwrap_or_else(|e| panic!("bad server line {line:?}: {e}"))
    }

    /// Send one request line and read one response line.
    pub fn req(&mut self, line: &str) -> Value {
        self.send_raw(format!("{line}\n").as_bytes());
        self.read()
    }

    pub fn feedback(&mut self, agent: &str, reward: f64) -> Value {
        self.req(&format!(r#"{{"cmd":"feedback","features":"x","agent":"{agent}","reward":{reward}}}"#))
    }

    /// Close stdin (EOF) and wait for exit.
    pub fn close(mut self) -> ExitStatus {
        drop(self.stdin.take());
        self.child.wait().unwrap()
    }
}
