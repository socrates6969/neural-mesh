<#
.SYNOPSIS
  Register (or remove) the "neural-mesh-dashboard" scheduled task: every 15 min it regenerates the mesh-ops
  dashboard and runs one alert pass (ops/run_all.py) with Python 3.12, windowless (pythonw).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File ops\install_schedule.ps1            # install / update
  powershell -ExecutionPolicy Bypass -File ops\install_schedule.ps1 -Uninstall # remove
  powershell -ExecutionPolicy Bypass -File ops\install_schedule.ps1 -WhatIf    # show what would be registered

.NOTES
  The task runs as the current user, only while that user is logged on (toasts need the interactive session).
  Output: C:\Users\mariu\ruv\hive\runs\dashboard\ (index.html, status.json, alerts.log, alert_state.json,
  ops_error.log on failure). Override with -OutDir (sets OPS_OUT_DIR for the task).
#>
[CmdletBinding(SupportsShouldProcess)]
param(
    [switch]$Uninstall,
    [int]$IntervalMinutes = 15,
    [string]$Python = '',
    [string]$OutDir = '',
    [string]$TaskName = 'neural-mesh-dashboard'
)
$ErrorActionPreference = 'Stop'

if ($Uninstall) {
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        if ($PSCmdlet.ShouldProcess($TaskName, 'Unregister scheduled task')) {
            Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
            Write-Output "removed scheduled task $TaskName"
        }
    } else {
        Write-Output "scheduled task $TaskName is not registered"
    }
    return
}

$repo = Split-Path -Parent $PSScriptRoot
$entry = Join-Path $PSScriptRoot 'run_all.py'
if (-not (Test-Path $entry)) { throw "missing $entry" }

if (-not $Python) {
    $exe = (& py -3.12 -c 'import sys; print(sys.executable)' 2>$null)
    if (-not $exe) { throw 'Python 3.12 not found via "py -3.12"; pass -Python <path to pythonw.exe>' }
    $Python = Join-Path (Split-Path -Parent $exe.Trim()) 'pythonw.exe'
}
if (-not (Test-Path $Python)) { throw "python not found: $Python" }

# pythonw directly (no console window). With -OutDir, go through cmd to set OPS_OUT_DIR for the task only.
if ($OutDir) {
    $cmdLine = 'cmd.exe /c set "OPS_OUT_DIR=' + $OutDir + '" && "' + $Python + '" "' + $entry + '"'
    $action = New-ScheduledTaskAction -Execute 'cmd.exe' -Argument $cmdLine.Substring(8) -WorkingDirectory $repo
} else {
    $cmdLine = '"' + $Python + '" "' + $entry + '"'
    $action = New-ScheduledTaskAction -Execute $Python -Argument ('"' + $entry + '"') -WorkingDirectory $repo
}
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Minutes $IntervalMinutes)
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 5) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$principal = New-ScheduledTaskPrincipal -UserId ([Security.Principal.WindowsIdentity]::GetCurrent().Name) `
    -LogonType Interactive -RunLevel Limited

if ($PSCmdlet.ShouldProcess($TaskName, "Register: $cmdLine every $IntervalMinutes min")) {
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
        -Principal $principal -Description 'neural-mesh ops: dashboard + alerts (ops/run_all.py)' -Force | Out-Null
    Write-Output "registered $TaskName every $IntervalMinutes min: $cmdLine"
    Write-Output "run now:  Start-ScheduledTask -TaskName $TaskName"
}