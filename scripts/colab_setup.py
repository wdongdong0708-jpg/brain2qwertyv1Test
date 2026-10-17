import argparse
import importlib.metadata
import importlib.util
import os
import subprocess
import sys
import tempfile
import venv
from pathlib import Path
from urllib.request import urlopen


def pip_available(python: Path) -> bool:
    result = subprocess.run(
        [str(python), "-m", "pip", "--version"], capture_output=True, text=True
    )
    return result.returncode == 0


def create_virtualenv(environment: Path) -> None:
    """Seed pip without invoking the runtime's optional ensurepip module."""
    if importlib.util.find_spec("virtualenv") is None:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "virtualenv"], check=True
        )
    subprocess.run(
        [sys.executable, "-m", "virtualenv", "--system-site-packages", str(environment)],
        check=True,
    )


def ensure_pip(python: Path, environment: Path, *, try_virtualenv: bool = True) -> None:
    """Repair new or partially created Colab environments without ensurepip."""
    if pip_available(python):
        return
    if try_virtualenv:
        print("Venv has no usable pip; bootstrapping with virtualenv...", flush=True)
        try:
            create_virtualenv(environment)
        except (OSError, subprocess.CalledProcessError) as exc:
            print(
                f"virtualenv bootstrap failed ({exc}); trying get-pip.py...", flush=True
            )
    if pip_available(python):
        return
    with tempfile.TemporaryDirectory(prefix="b2q-pip-") as folder:
        bootstrap = Path(folder) / "get-pip.py"
        with urlopen("https://bootstrap.pypa.io/get-pip.py", timeout=60) as response:
            bootstrap.write_bytes(response.read())
        subprocess.run([str(python), str(bootstrap)], check=True)
    if not pip_available(python):
        raise RuntimeError(f"pip is still unavailable in {python}")


def prepare_environment(environment: Path) -> Path:
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.exists():
        try:
            create_virtualenv(environment)
        except (OSError, subprocess.CalledProcessError) as exc:
            print(
                f"virtualenv creation failed ({exc}); using venv + get-pip.py...",
                flush=True,
            )
            venv.EnvBuilder(with_pip=False, system_site_packages=True).create(environment)
            ensure_pip(python, environment, try_virtualenv=False)
            return python
    ensure_pip(python, environment)
    return python


def main():
    p = argparse.ArgumentParser(
        description="Install V1 while preserving the runtime's PyTorch/CUDA packages"
    )
    p.add_argument("--venv", type=Path, default=Path("/content/b2q-venv"))
    args = p.parse_args()
    if sys.version_info < (3, 12):
        raise RuntimeError(
            "This project requires Python >= 3.12; the validated version is 3.12"
        )
    constraints = []
    for package in ("torch", "torchvision", "torchaudio"):
        try:
            constraints.append(f"{package}=={importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            if package == "torch":
                raise RuntimeError(
                    "Install a compatible PyTorch runtime before V1 dependencies"
                ) from None
    python = prepare_environment(args.venv)
    constraint_file = args.venv / "runtime-constraints.txt"
    constraint_file.write_text("\n".join(constraints) + "\n", encoding="utf-8")
    requirements = Path(__file__).resolve().parents[1] / "requirements-v1-colab.txt"
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "-r",
            str(requirements),
            "-c",
            str(constraint_file),
        ],
        check=True,
    )
    print(f"V1 ready: {python}", flush=True)


if __name__ == "__main__":
    main()
