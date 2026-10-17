$ErrorActionPreference = 'Stop'
$researchRoot = Join-Path $PSScriptRoot 'outputs\eeg-002\context_research_seed33'
New-Item -ItemType Directory -Path $researchRoot -Force | Out-Null
$processRecord = Join-Path $researchRoot 'process.json'
if (Test-Path -LiteralPath $processRecord) {
    $previousResearch = Get-Content -LiteralPath $processRecord -Raw | ConvertFrom-Json
    $previousProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$($previousResearch.pid)" -ErrorAction SilentlyContinue
    if ($previousProcess -and $previousProcess.CommandLine -match 'brain2qwerty_v1.scripts.research_queue') {
        throw "The seed33 queue is already running with PID $($previousResearch.pid)."
    }
}
$researchProcess = Start-Process -FilePath (Join-Path $PSScriptRoot '.venv\Scripts\python.exe') `
    -ArgumentList @('-u', '-m', 'brain2qwerty_v1.scripts.research_queue') `
    -WorkingDirectory $PSScriptRoot -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $researchRoot 'queue.stdout.log') `
    -RedirectStandardError (Join-Path $researchRoot 'queue.stderr.log') -PassThru
@{pid=$researchProcess.Id; started_at=(Get-Date).ToString('o'); seed=33} |
    ConvertTo-Json | Set-Content -LiteralPath $processRecord -Encoding utf8
Write-Output "Seed33 context research started with PID $($researchProcess.Id)."
Write-Output "Progress: $researchRoot\status.json"
