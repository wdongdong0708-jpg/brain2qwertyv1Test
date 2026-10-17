param([string]$BasePython = 'D:\soft\miniconda\envs\E1\python.exe')

$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath $PSScriptRoot
try {
    & $BasePython -c "import sys, torch; assert sys.version_info >= (3, 12); assert torch.cuda.is_available(); print(sys.version); print(torch.__version__, torch.cuda.get_device_name(0))"
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.12+ with working CUDA PyTorch is required.' }
    & $BasePython -m venv --system-site-packages '.venv'
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the project environment.' }
    & '.\.venv\Scripts\python.exe' -m pip install --index-url https://pypi.org/simple --cache-dir '.setup\pip-cache' -r 'requirements-v1-windows.txt'
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
    Write-Output 'Ready. Run .\run_subject.ps1 inspect'
}
finally {
    Pop-Location
}
