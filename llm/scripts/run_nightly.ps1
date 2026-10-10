# Nightly round, installed as the Windows scheduled task "neural-mesh-ingest-nightly" (daily 03:30):
#   1) ingest catch-up (ingest\scripts\run_ingest.ps1 -NoMirror; waits for a SessionEnd ingest still running)
#   2) python -m llm nightly: preflight abandoned-sweep -> train -> eval -> promote under ONE llm/models/nightly.lock
#      (skips its llm stages itself on battery power; RunRecord -> llm/models/runs.jsonl)
#   3) mirror: python -m ingest mirror (db: -MirrorDb, default <repo>\.swarm\memory.db; outside the repo = refused, exit 5)
# A failed ingest skips the llm round but the mirror still runs. llm exit 75 (another llm run holds the lock) is logged as
# "busy" and counts as 0. Exit code = the worst stage exit code (0 ok, 3 a stage failed, 4 mirror stored nothing, 5 mirror
# refused); an unexpected PowerShell error writes FATAL to the log and exits 1. Logs rotate above 5 MB (.1 .. .3).
#   -JudgeBudget N   LOCAL (Ollama, free, ~10 s/call) judge calls for the ingestor: SFT candidates round-robin over sessions, then a decoy call each
#                    (ingest/schedule.py). 800 clears the first backlog of ~270 candidates + decoys in one night; later nights need a few dozen.
#   -PairBudget N    PAID Claude Haiku pairwise-judge calls for the llm eval gate (default 120). Before R122 -JudgeBudget was passed to both, i.e. 600 paid calls.
#   -Python P        llm venv python (default llm\.venv\Scripts\python.exe)
#   -IngestPython P  python for ingest/mirror (default `python` on PATH)
#   -NoMirror        skip stage 3
#   -Log P           log file (default llm\models\nightly.log)
param([int]$IngestWaitSeconds = 3600, [int]$JudgeBudget = 800, [int]$PairBudget = 120, [string]$MirrorDb = '', [string]$Python = '',
      [string]$IngestPython = 'python', [switch]$NoMirror, [string]$Log = '')
$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $Log) { $Log = Join-Path $repo 'llm\models\nightly.log' }
if (-not $Python) { $Python = Join-Path $repo 'llm\.venv\Scripts\python.exe' }

function Invoke-Logged([string]$Exe, [string[]]$Arguments, [string]$LogFile) {
    # native stderr must never become a terminating error (PS 5.1 + EAP Stop); every line goes to the log as text
    $ErrorActionPreference = 'Continue'
    & $Exe @Arguments 2>&1 | ForEach-Object { "$_" } | Out-File -Append -Encoding utf8 $LogFile
    return $LASTEXITCODE
}

function Rotate-Log([string]$Path, [long]$MaxBytes = 5MB, [int]$Keep = 3) {
    if (-not (Test-Path -LiteralPath $Path)) { return }
    if ((Get-Item -LiteralPath $Path).Length -le $MaxBytes) { return }
    for ($i = $Keep; $i -ge 1; $i--) {
        $src = if ($i -eq 1) { $Path } else { "$Path.$($i - 1)" }
        if (Test-Path -LiteralPath $src) { Move-Item -LiteralPath $src -Destination "$Path.$i" -Force }
    }
}

function Write-Log([string]$Text) { "$(Get-Date -Format o) $Text" | Out-File -Append -Encoding utf8 $Log }

try {
    New-Item -ItemType Directory -Force -Path (Split-Path $Log) | Out-Null
    Rotate-Log $Log
    Push-Location $repo
    try {
        Write-Log '=== nightly start'
        $worst = 0
        # 1) ingest (its own log; -NoMirror: the mirror is stage 3 here)
        $ingestArgs = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $repo 'ingest\scripts\run_ingest.ps1'),
                        '-WaitSeconds', "$IngestWaitSeconds", '-JudgeBudget', "$JudgeBudget", '-Python', $IngestPython, '-NoMirror')
        if ($PSBoundParameters.ContainsKey('Log')) { $ingestArgs += @('-Log', "$Log.ingest") }   # tests: keep the real logs clean
        $irc = Invoke-Logged 'powershell' $ingestArgs $Log
        Write-Log "ingest exit $irc"
        # 1b) scorecard (cross-session metrics; reads transcripts + outcomes.jsonl; never fails the round)
        $src = Invoke-Logged $IngestPython @('-m', 'ingest', 'scorecard') $Log
        Write-Log "scorecard exit $src"
        if ($irc -ne 0) {
            $worst = $irc
            Write-Log 'ingest failed; skipping the llm round'
        } else {
            # 2) the llm round under one lock
            $lrc = Invoke-Logged $Python @('-u', '-m', 'llm', 'nightly', '--judge-budget', "$PairBudget") $Log
            if ($lrc -eq 75) { Write-Log 'llm busy (nightly.lock held by another llm run); skipped'; $lrc = 0 }
            else { Write-Log "llm nightly exit $lrc" }
            if ($lrc -gt $worst) { $worst = $lrc }
        }
        # 3) mirror
        if (-not $NoMirror) {
            $margs = @('-m', 'ingest', 'mirror', '--wait', "$IngestWaitSeconds")
            if ($MirrorDb) { $margs += @('--db-path', $MirrorDb) }
            $mrc = Invoke-Logged $IngestPython $margs $Log
            switch ($mrc) {
                0 { Write-Log 'mirror ok' }
                5 { Write-Log 'mirror REFUSED (db outside the repo)' }
                75 { Write-Log 'mirror skipped: ingest lock busy'; $mrc = 0 }
                default { Write-Log "mirror exit $mrc" }
            }
            if ($mrc -gt $worst) { $worst = $mrc }
        }
        # 4) codemem: refresh the BM25 code/doc index (incremental, ~0.4 s when nothing changed; never fails the round)
        $cm = Join-Path $repo 'data\codemem\bin\codemem.exe'
        if (Test-Path -LiteralPath $cm) { $crc = Invoke-Logged $cm @('update') $Log; Write-Log "codemem update exit $crc" }
        Write-Log "=== nightly done (exit $worst)"
        exit $worst
    } finally { Pop-Location }
} catch {
    try { "$(Get-Date -Format o) FATAL: $_" | Out-File -Append -Encoding utf8 $Log } catch { }
    exit 1
}
