# Runs the lifelong ingestor from the neural-mesh repo root. Concurrency-safe: `python -m ingest run` holds an OS file lock
# (data/lifelong/.ingest.lock, released automatically if the process dies); a second run either skips (-WaitSeconds 0) or waits.
#   -Detach        start in the background and return at once (for the Claude Code SessionEnd hook, which must not block exit)
#   -WaitSeconds N wait up to N seconds for a running ingest to finish (nightly catch-up)
#   -JudgeBudget N max new local-LLM judge calls this run
param([switch]$Detach, [int]$WaitSeconds = 0, [int]$JudgeBudget = 300)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$log = Join-Path $repo 'data\lifelong\ingest-runs.log'
New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null
$argList = @('-m', 'ingest', 'run', '--quiet', '--wait', "$WaitSeconds", '--judge-budget', "$JudgeBudget")
if ($Detach) {
    Start-Process -FilePath 'python' -ArgumentList $argList -WorkingDirectory $repo -WindowStyle Hidden `
        -RedirectStandardOutput $log -RedirectStandardError ($log + '.err') | Out-Null
} else {
    Push-Location $repo
    try { & python @argList *>> $log } finally { Pop-Location }
}
