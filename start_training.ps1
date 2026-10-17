$ErrorActionPreference = 'Stop'
$trainingRoot = Join-Path $PSScriptRoot 'outputs\eeg-002'
$runPath = Join-Path $trainingRoot 'train'
$processRecord = Join-Path $trainingRoot 'training_process.json'
if (Test-Path -LiteralPath $processRecord) {
    $previousRun = Get-Content -LiteralPath $processRecord -Raw | ConvertFrom-Json
    $previousProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$($previousRun.pid)" -ErrorAction SilentlyContinue
    if ($previousProcess -and $previousProcess.CommandLine -match 'brain2qwerty_v1.scripts.single_subject') {
        throw "Training is already running with PID $($previousRun.pid)."
    }
}
New-Item -ItemType Directory -Path $trainingRoot -Force | Out-Null
if (Test-Path -LiteralPath (Join-Path $runPath 'best.ckpt')) {
    throw 'This run already has a checkpoint. Resume it with run_subject.ps1 train --resume instead.'
}
$trainingProcess = Start-Process -FilePath (Join-Path $PSScriptRoot '.venv\Scripts\python.exe') `
    -ArgumentList @('-u', '-m', 'brain2qwerty_v1.scripts.single_subject', 'train') `
    -WorkingDirectory $PSScriptRoot -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $trainingRoot 'train.stdout.log') `
    -RedirectStandardError (Join-Path $trainingRoot 'train.stderr.log') -PassThru
@{
    pid = $trainingProcess.Id
    started_at = (Get-Date).ToString('o')
    subject = '002'
    modality = 'eeg'
    run_dir = $runPath
} | ConvertTo-Json | Set-Content -LiteralPath $processRecord -Encoding utf8
Write-Output "Training started with PID $($trainingProcess.Id)."
Write-Output "Progress: $runPath\status.json"
