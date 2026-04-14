$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$req = Join-Path $scriptDir "requirements.txt"

Write-Host "[comfyui_dwg_exporter] Installing Python dependencies..."
python -m pip install -r $req

$odaCandidates = @(
    "C:\Program Files\ODA\ODAFileConverter\ODAFileConverter.exe",
    "C:\Program Files\ODA\ODAFileConverter 25.12.0\ODAFileConverter.exe",
    "C:\Program Files\ODA\ODAFileConverter 25.6.0\ODAFileConverter.exe"
)

$odaPath = $odaCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if ($odaPath) {
    [Environment]::SetEnvironmentVariable("ODAFC_PATH", $odaPath, "User")
    Write-Host "[comfyui_dwg_exporter] ODAFC_PATH configured: $odaPath"
} else {
    Write-Warning "[comfyui_dwg_exporter] ODA File Converter not found. Install it manually:"
    Write-Host "https://www.opendesign.com/guestfiles/oda_file_converter"
}

Write-Host "[comfyui_dwg_exporter] Done. Restart ComfyUI."
