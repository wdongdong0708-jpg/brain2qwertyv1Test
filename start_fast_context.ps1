$ErrorActionPreference = 'Stop'
$fastRoot = Join-Path $PSScriptRoot 'outputs\eeg-002\fast_context_seed33'
New-Item -ItemType Directory -Path $fastRoot -Force | Out-Null
$recordPath = Join-Path $fastRoot 'process.json'
if (Test-Path -LiteralPath $recordPath) {
    $previousFast = Get-Content -LiteralPath $recordPath -Raw | ConvertFrom-Json
    $previousProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$($previousFast.pid)" -ErrorAction SilentlyContinue
    if ($previousProcess -and $previousProcess.CommandLine -match 'brain2qwerty_v1.scripts.fast_context_run') {
        throw "Fast context queue is already running with PID $($previousFast.pid)."
    }
}
$fastProcess = Start-Process -FilePath (Join-Path $PSScriptRoot '.venv\Scripts\python.exe') `
    -ArgumentList @('-u', '-m', 'brain2qwerty_v1.scripts.fast_context_run') `
    -WorkingDirectory $PSScriptRoot -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $fastRoot 'queue.stdout.log') `
    -RedirectStandardError (Join-Path $fastRoot 'queue.stderr.log') -PassThru
@{pid=$fastProcess.Id; started_at=(Get-Date).ToString('o'); seed=33} |
    ConvertTo-Json | Set-Content -LiteralPath $recordPath -Encoding utf8
Write-Output "Fast seed33 context queue started with PID $($fastProcess.Id)."
