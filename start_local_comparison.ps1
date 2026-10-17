$ErrorActionPreference = 'Stop'
$comparisonRoot = Join-Path $PSScriptRoot 'outputs\eeg-002\context_comparison'
New-Item -ItemType Directory -Path $comparisonRoot -Force | Out-Null
$processRecord = Join-Path $comparisonRoot 'process.json'
if (Test-Path -LiteralPath $processRecord) {
    $previousRun = Get-Content -LiteralPath $processRecord -Raw | ConvertFrom-Json
    $previousProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$($previousRun.pid)" -ErrorAction SilentlyContinue
    if ($previousProcess -and $previousProcess.CommandLine -match 'brain2qwerty_v1.scripts.local_comparison') {
        throw "The comparison queue is already running with PID $($previousRun.pid)."
    }
}
$comparisonProcess = Start-Process -FilePath (Join-Path $PSScriptRoot '.venv\Scripts\python.exe') `
    -ArgumentList @('-u', '-m', 'brain2qwerty_v1.scripts.local_comparison') `
    -WorkingDirectory $PSScriptRoot -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $comparisonRoot 'queue.stdout.log') `
    -RedirectStandardError (Join-Path $comparisonRoot 'queue.stderr.log') -PassThru
@{pid=$comparisonProcess.Id; started_at=(Get-Date).ToString('o')} |
    ConvertTo-Json | Set-Content -LiteralPath $processRecord -Encoding utf8
Write-Output "Local-context comparison started with PID $($comparisonProcess.Id)."
Write-Output "Progress: $comparisonRoot\status.json"
