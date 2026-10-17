$ErrorActionPreference = 'Stop'
$jointRoot = Join-Path $PSScriptRoot 'outputs\eeg-002\small_joint_seed33'
New-Item -ItemType Directory -Path $jointRoot -Force | Out-Null
$recordPath = Join-Path $jointRoot 'process.json'
if (Test-Path -LiteralPath $recordPath) {
    $previousJoint = Get-Content -LiteralPath $recordPath -Raw | ConvertFrom-Json
    $previousProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$($previousJoint.pid)" -ErrorAction SilentlyContinue
    if ($previousProcess -and $previousProcess.CommandLine -match 'brain2qwerty_v1.scripts.fast_context_run') {
        throw "Small joint queue is already running with PID $($previousJoint.pid)."
    }
}
$jointProcess = Start-Process -FilePath (Join-Path $PSScriptRoot '.venv\Scripts\python.exe') `
    -ArgumentList @('-u', '-m', 'brain2qwerty_v1.scripts.fast_context_run', '--train-encoder') `
    -WorkingDirectory $PSScriptRoot -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $jointRoot 'queue.stdout.log') `
    -RedirectStandardError (Join-Path $jointRoot 'queue.stderr.log') -PassThru
@{pid=$jointProcess.Id; started_at=(Get-Date).ToString('o'); seed=33} |
    ConvertTo-Json | Set-Content -LiteralPath $recordPath -Encoding utf8
Write-Output "Small joint seed33 queue started with PID $($jointProcess.Id)."
