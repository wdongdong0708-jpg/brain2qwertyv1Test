$ErrorActionPreference = 'Stop'
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw 'Project environment is missing. Follow SINGLE_SUBJECT_zh.md first.'
}
Push-Location -LiteralPath $PSScriptRoot
try {
    & $projectPython -u -m brain2qwerty_v1.scripts.single_subject @args
    $experimentExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $experimentExitCode
