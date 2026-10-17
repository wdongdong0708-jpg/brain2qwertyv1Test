import argparse
import hashlib
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

from brain2qwerty_v1.scripts.colab_data import (
    check_brainvision_files,
    safe_path,
    select_files,
)


def stage_subject(source: Path, destination: Path, subject="002") -> dict:
    source, destination = source.resolve(), destination.resolve()
    if source == destination or destination.is_relative_to(source):
        raise ValueError("Destination must be outside the existing raw data directory")
    entries = [
        SimpleNamespace(path=p.relative_to(source).as_posix(), size=p.stat().st_size)
        for p in (source / "EEG").rglob("*")
        if p.is_file()
    ]
    selected = select_files(entries, subject)
    linked = copied = existing = 0
    for entry in selected:
        src, dst = safe_path(source, entry["path"]), safe_path(destination, entry["path"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            if not os.path.samefile(src, dst):
                with src.open("rb") as left, dst.open("rb") as right:
                    if (
                        hashlib.file_digest(left, "sha256").digest()
                        != hashlib.file_digest(right, "sha256").digest()
                    ):
                        raise FileExistsError(f"Existing file differs: {dst}")
            existing += 1
            continue
        try:
            os.link(src, dst)
            linked += 1
        except OSError:
            shutil.copy2(src, dst)
            copied += 1
    recordings = check_brainvision_files(destination, subject)
    return {
        "recordings": recordings,
        "files": len(selected),
        "hardlinked": linked,
        "copied": copied,
        "existing": existing,
        "destination": str(destination),
    }


def main():
    p = argparse.ArgumentParser(
        description="Stage only one EEG subject, using hard links on the same disk"
    )
    p.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Existing SpanishBCBL directory containing EEG/",
    )
    p.add_argument(
        "--destination",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "workspace/data/Pinet2024Eeg",
    )
    p.add_argument("--subject", default="002")
    args = p.parse_args()
    print(stage_subject(args.source, args.destination, args.subject), flush=True)


if __name__ == "__main__":
    main()
