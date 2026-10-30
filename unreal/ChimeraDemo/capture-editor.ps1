param([Parameter(Mandatory=$true)][int]$EditorPid)
$ErrorActionPreference = 'Stop'
$process = Get-Process -Id $EditorPid
if ($process.ProcessName -ne 'UnrealEditor') { throw 'The selected process is not Unreal Editor.' }
$saved = Join-Path $PSScriptRoot 'Saved'
$env:TEMP = Join-Path $saved 'Temp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force $env:TEMP | Out-Null
Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName UIAutomationClient
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class ChimeraWindow {
    [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
    [StructLayout(LayoutKind.Sequential)]
    public struct Rect { public int Left, Top, Right, Bottom; }
    [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr window, out Rect rect);
    [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr window, int command);
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr window);
    [DllImport("user32.dll")] public static extern bool MoveWindow(IntPtr window, int x, int y, int width, int height, bool repaint);
}
'@
$element = [System.Windows.Automation.AutomationElement]::RootElement.FindAll(
    [System.Windows.Automation.TreeScope]::Children, [System.Windows.Automation.Condition]::TrueCondition) |
    Where-Object { $_.Current.ProcessId -eq $EditorPid -and $_.Current.Name -like '*Unreal Editor*' } |
    Select-Object -First 1
if (-not $element) { throw 'The editor window is not available.' }
$window = [IntPtr]$element.Current.NativeWindowHandle
[ChimeraWindow]::SetProcessDPIAware() | Out-Null
[ChimeraWindow]::ShowWindow($window, 9) | Out-Null
[ChimeraWindow]::MoveWindow($window, 100, 100, 1920, 1080, $true) | Out-Null
[ChimeraWindow]::SetForegroundWindow($window) | Out-Null
Start-Sleep -Milliseconds 800
$rect = New-Object ChimeraWindow+Rect
[ChimeraWindow]::GetWindowRect($window, [ref]$rect) | Out-Null
$bitmap = New-Object System.Drawing.Bitmap(($rect.Right - $rect.Left), ($rect.Bottom - $rect.Top))
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
$graphics.CopyFromScreen($rect.Left, $rect.Top, 0, 0, $bitmap.Size)
$path = Join-Path $saved 'editor-proof.png'
$bitmap.Save($path, [System.Drawing.Imaging.ImageFormat]::Png)
$graphics.Dispose()
$bitmap.Dispose()
Write-Output $path
