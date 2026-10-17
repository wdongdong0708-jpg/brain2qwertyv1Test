import argparse
import importlib.metadata
import subprocess
import sys
import venv
from pathlib import Path


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
    if not (args.venv / "bin/python").exists():
        venv.EnvBuilder(with_pip=True, system_site_packages=True).create(args.venv)
    constraint_file = args.venv / "runtime-constraints.txt"
    constraint_file.write_text("\n".join(constraints) + "\n", encoding="utf-8")
    requirements = Path(__file__).resolve().parents[1] / "requirements-v1-colab.txt"
    subprocess.run(
        [
            str(args.venv / "bin/python"),
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
    print(f"V1 ready: {args.venv / 'bin/python'}", flush=True)


if __name__ == "__main__":
    main()
