# Baseline run of NO-FC on Windows + Ollama. Usage: powershell -ExecutionPolicy Bypass -File scripts\run_baseline.ps1
$ErrorActionPreference = "Continue"
$date = Get-Date -Format "yyyy-MM-dd"
try { Invoke-WebRequest -UseBasicParsing http://localhost:11434/api/version | Out-Null }
catch { Start-Process -WindowStyle Hidden "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" -ArgumentList "serve"; Start-Sleep 8 }
$models = @("qwen3:8b", "hf.co/ltg/normistral-11b-thinking-gguf:Q4_K_M")
New-Item -ItemType Directory -Force results | Out-Null
foreach ($m in $models) {
  ollama pull $m
  $safe = ($m -replace '[^A-Za-z0-9._-]', '_')
  python -m nofc.evaluate --base-url http://localhost:11434/v1 --model $m --out "results\$safe-$date.jsonl" 2>"results\$safe-$date.log" | Out-File -Encoding utf8 "results\$safe-$date.json"
  Write-Host "== $m"; Get-Content "results\$safe-$date.json"
}
