# Yerel mock TEKNOFEST sunucusu + PyQt arayuzu (Windows)
# Kullanim: HY klasorunde: powershell -ExecutionPolicy Bypass -File .\start_local_demo.ps1

$Root = $PSScriptRoot
Set-Location $Root

Write-Host "Installing dependencies..."
python -m pip install -q --user flask pillow
Set-Location (Join-Path $Root "teknofest_ai_system")
python -m pip install -q --user -r requirements.txt
Set-Location $Root

Write-Host "Starting mock server (new window)..."
Start-Process python -ArgumentList @("mock_evaluation_server.py", "--host", "127.0.0.1", "--port", "5000") -WorkingDirectory $Root

Start-Sleep -Seconds 2

Write-Host "Starting TEKNOFEST AI GUI (new window)..."
Start-Process python -ArgumentList "main.py" -WorkingDirectory (Join-Path $Root "teknofest_ai_system")

Write-Host ""
Write-Host "Server: http://127.0.0.1:5000/  user/pass: demo / demo"
Write-Host "In the app: Sunucu tab -> Connect with these credentials."
