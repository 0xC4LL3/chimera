param(
    [switch]$Build,
    [string]$Engine = 'C:\Program Files\Epic Games\UE_5.8'
)
$ErrorActionPreference = 'Stop'
if ($PSScriptRoot.StartsWith('\\')) {
    $forward = @('-Engine', $Engine)
    if ($Build) { $forward += '-Build' }
    & (Join-Path $PSScriptRoot 'launch.cmd') @forward
    exit $LASTEXITCODE
}
$editor = Join-Path $Engine 'Engine\Binaries\Win64\UnrealEditor.exe'
$arguments = @((Join-Path $PSScriptRoot 'ChimeraDemo.uproject'), '-DDC=Chimera', '-NoSplash', '-NoSound', '-log')
if ($Build) { $arguments += '/Engine/Maps/Entry' }
$script = if ($Build) { 'Scripts\build_stage.py' } else { 'Scripts\preview.py' }
$arguments += '-ExecutePythonScript=' + (Join-Path $PSScriptRoot $script)
Push-Location $PSScriptRoot
try { & $editor @arguments | Out-Host }
finally { Pop-Location }
