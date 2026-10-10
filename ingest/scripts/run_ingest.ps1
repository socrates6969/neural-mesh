# Runs the lifelong ingestor from the neural-mesh repo root. Concurrency-safe: `python -m ingest run` holds an OS file lock
# (data/lifelong/.ingest.lock, released automatically if the process dies); a second run either skips (-WaitSeconds 0) or waits.
#   -Detach        start in the background and return at once (for the Claude Code SessionEnd hook, which must not block exit)
#   -WaitSeconds N wait up to N seconds for a running ingest to finish (nightly catch-up)
#   -JudgeBudget N max new local-LLM judge calls this run
#   -Python P      python executable (default: `python` on PATH; stdlib + numpy)
#   -NoMirror      foreground run: skip the mirror step (the nightly does its own mirror stage after the llm round)
#   -MirrorDb P    mirror target; default empty = <repo>\.swarm\memory.db (a db outside the repo is refused by the mirror, exit 5)
#   -Log P         log file (default data\lifelong\ingest-runs.log; rotated above 5 MB to .1 .. .3)
# Exit code: the worst of ingest / mirror (mirror: 0 ok, 3 partial, 4 nothing stored, 5 refused; 75 busy is logged as 0).
param([switch]$Detach, [int]$WaitSeconds = 0, [int]$JudgeBudget = 300, [string]$MirrorDb = '', [string]$Python = 'python',
      [switch]$NoMirror, [string]$Log = '')
$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $Log) { $Log = Join-Path $repo 'data\lifelong\ingest-runs.log' }

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
    $argList = @('-m', 'ingest', 'run', '--quiet', '--wait', "$WaitSeconds", '--judge-budget', "$JudgeBudget")
    if ($Detach) {
        Start-Process -FilePath $Python -ArgumentList $argList -WorkingDirectory $repo -WindowStyle Hidden `
            -RedirectStandardOutput $Log -RedirectStandardError ($Log + '.err') | Out-Null
        exit 0
    }
    Push-Location $repo
    try {
        $worst = 0
        $rc = Invoke-Logged $Python $argList $Log
        Write-Log "ingest run exit $rc"
        if ($rc -ne 0) { $worst = $rc }
        if (-not $NoMirror) {
            $margs = @('-m', 'ingest', 'mirror', '--wait', "$WaitSeconds")
            if ($MirrorDb) { $margs += @('--db-path', $MirrorDb) }
            $mrc = Invoke-Logged $Python $margs $Log
            switch ($mrc) {
                0 { Write-Log 'mirror ok' }
                5 { Write-Log 'mirror REFUSED (db outside the repo; see --allow-external-db)' }
                75 { Write-Log 'mirror skipped: ingest lock busy'; $mrc = 0 }
                default { Write-Log "mirror exit $mrc" }
            }
            if ($mrc -gt $worst) { $worst = $mrc }
        }
        exit $worst
    } finally { Pop-Location }
} catch {
    try { "$(Get-Date -Format o) FATAL: $_" | Out-File -Append -Encoding utf8 $Log } catch { }
    exit 1
}
