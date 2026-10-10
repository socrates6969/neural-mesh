# Nightly local-LLM round, chained AFTER the ingestor: ingest (wait for any running ingest) -> train -> eval -> promote.
# Each `python -m llm <cmd>` takes llm/models/nightly.lock (exit 75 if another llm run holds it; a lock older than 12 h is
# treated as stale). A step that finds nothing to do exits 0, so the chain is safe to run every night.
# Provided, NOT installed. Register with (one line, run once as the user):
#   schtasks /Create /TN "neural-mesh-llm-nightly" /SC DAILY /ST 03:30 /RL LIMITED /F /TR "powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mariu\neural-mesh\llm\scripts\run_nightly.ps1"
param([int]$IngestWaitSeconds = 3600, [int]$JudgeBudget = 600, [string]$MirrorDb = '')
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$py = Join-Path $repo 'llm\.venv\Scripts\python.exe'
$log = Join-Path $repo 'llm\models\nightly.log'
New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null
Push-Location $repo
try {
    "=== $(Get-Date -Format o) nightly start" | Out-File -Append -Encoding utf8 $log
    # 1) the ingestor's own nightly catch-up (system python, stdlib); waits for a SessionEnd ingest still running
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $repo 'ingest\scripts\run_ingest.ps1') -WaitSeconds $IngestWaitSeconds -JudgeBudget $JudgeBudget -MirrorDb $MirrorDb
    if ($LASTEXITCODE -ne 0) { "ingest exit $LASTEXITCODE; skipping llm round" | Out-File -Append -Encoding utf8 $log; exit $LASTEXITCODE }
    # 2) python -m llm train && python -m llm eval && python -m llm promote
    foreach ($cmd in 'train', 'eval', 'promote') {
        & $py -m llm $cmd *>> $log
        if ($LASTEXITCODE -ne 0) { "llm $cmd exit $LASTEXITCODE; stopping" | Out-File -Append -Encoding utf8 $log; exit $LASTEXITCODE }
    }
    "=== $(Get-Date -Format o) nightly done" | Out-File -Append -Encoding utf8 $log
} finally { Pop-Location }
