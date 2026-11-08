import argparse
import hashlib
import os
import shutil
from pathlib import Path, PurePosixPath
from types import SimpleNamespace


def validate_subject(subject):
    number = int(subject)
    if number < 1:
        raise ValueError("Subject must be positive")
    if number == 1:
        raise ValueError("The upstream EEG reader excludes subject 001. Use EEG 002.")
    return f"{number:03d}"


def select_files(entries, subject):
    subject = validate_subject(subject)
    selected = []
    for entry in entries:
        path = entry.path
        name = PurePosixPath(path).name
        signal = (
            path.startswith(f"EEG/EEG/{subject}_")
            and "_task" in name
            and name.endswith((".eeg", ".vhdr", ".vmrk"))
        )
        log = (
            path.startswith("EEG/logs/")
            and name.startswith(f"S{int(subject)}_session")
            and "_block" in name
            and name.endswith(".mat")
        )
        if signal or log:
            selected.append({"path": path, "bytes": int(entry.size)})
    if not any(row["path"].endswith(".vhdr") for row in selected):
        raise RuntimeError("No EEG headers match the selected subject")
    if not any(row["path"].endswith(".mat") for row in selected):
        raise RuntimeError("No MATLAB keystroke logs match the selected subject")
    return sorted(selected, key=lambda row: row["path"])


def safe_path(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Dataset path escapes the requested directory")
    return path


def check_brainvision_files(root, subject):
    folder = Path(root) / "EEG/EEG"
    headers = sorted(folder.glob(f"{subject}_*task*.vhdr"))
    if not headers:
        raise RuntimeError("EEG header files are missing")
    for header in headers:
        references = {}
        for line in header.read_text(encoding="utf-8-sig", errors="replace").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                if key.strip().lower() in {"datafile", "markerfile"}:
                    references[key.strip().lower()] = value.strip()
        for key in ("datafile", "markerfile"):
            if key not in references:
                raise RuntimeError(f"{header.name} has no {key} reference")
            target = safe_path(header.parent, references[key])
            if not target.is_file():
                raise RuntimeError(f"Missing {references[key]} required by {header.name}")
    return len(headers)


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
