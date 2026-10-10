# Runs the lifelong ingestor from the neural-mesh repo root. Concurrency-safe: `python -m ingest run` holds an OS file lock
# (data/lifelong/.ingest.lock, released automatically if the process dies); a second run either skips (-WaitSeconds 0) or waits.
#   -Detach        start in the background and return at once (for the Claude Code SessionEnd hook, which must not block exit)
#   -WaitSeconds N wait up to N seconds for a running ingest to finish (nightly catch-up)
#   -JudgeBudget N max new local-LLM judge calls this run
#   -MirrorDb P    after a foreground run, mirror scrubbed memories/outcomes into this LOCAL ruflo memory.db
param([switch]$Detach, [int]$WaitSeconds = 0, [int]$JudgeBudget = 300, [string]$MirrorDb = '')
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
    try {
        & python @argList *>> $log
        $ingestExit = $LASTEXITCODE
        # The user approved mirroring into this specific external ruflo db (2026-10-10), hence --allow-external-db.
        # Mirror failures are logged but never fail the run, so they cannot stop the nightly LLM chain.
        if ($MirrorDb -and $ingestExit -eq 0) {
            & python -m ingest mirror --db-path $MirrorDb --allow-external-db *>> $log
            "mirror exit $LASTEXITCODE" *>> $log
        }
        exit $ingestExit
    } finally { Pop-Location }
}
