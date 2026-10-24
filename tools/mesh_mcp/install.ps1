# Registers the mesh-tools MCP server for the current user in Claude Code (user scope, all projects).
# Usage:  powershell -File tools\mesh_mcp\install.ps1 [-RepoRoot <dir>] [-Python <exe>] [-Image <tag>] [-DryRun]
# RepoRoot defaults to the checkout this script lives in; after exp/mesh-tools is merged, pass C:\Users\mariu\neural-mesh.
param(
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path,
    [string]$Python = "C:\Users\mariu\AppData\Local\Programs\Python\Python312\python.exe",
    [string]$Image = "meshtrain-grader:2",
    [switch]$DryRun
)
$ErrorActionPreference = "Stop"
if (-not (Test-Path (Join-Path $RepoRoot "tools\mesh_mcp\server.py"))) { throw "server.py not found under $RepoRoot" }
if (-not (Test-Path $Python)) { throw "Python not found: $Python" }
$cfg = [ordered]@{
    type    = "stdio"
    command = $Python
    args    = @("-m", "tools.mesh_mcp.server")
    env     = [ordered]@{ PYTHONPATH = $RepoRoot; MESH_TOOLS_IMAGE = $Image; PYTHONIOENCODING = "utf-8" }
}
$json = $cfg | ConvertTo-Json -Compress -Depth 4
Write-Host "claude mcp add-json --scope user mesh-tools $json"
if ($DryRun) { return }
# Windows PowerShell 5.1 drops embedded double quotes when calling native programs, so escape them there.
$arg = if ($PSVersionTable.PSVersion.Major -lt 7) { $json.Replace('"', '\"') } else { $json }
& claude mcp add-json --scope user mesh-tools $arg
if ($LASTEXITCODE -ne 0) { throw "claude mcp add-json failed ($LASTEXITCODE)" }
& claude mcp get mesh-tools
