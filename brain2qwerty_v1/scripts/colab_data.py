"""Prepare one EEG participant, including the matching MATLAB keystroke logs."""

import argparse
import json
import os
import shutil
from pathlib import Path, PurePosixPath

from .research_queue import read_json, write_json


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


def copy_atomic(source, destination):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.copying")
    try:
        shutil.copyfile(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def manifest_is_complete(root, manifest):
    return bool(manifest.get("files")) and all(
        safe_path(root, row["path"]).is_file()
        and safe_path(root, row["path"]).stat().st_size == row["bytes"]
        for row in manifest["files"]
    )


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


def prepare(settings):
    subject = validate_subject(settings["subject"])
    persistent = Path(settings["drive_data"])
    local = Path(settings["local_data"])
    persistent.mkdir(parents=True, exist_ok=True)
    local.mkdir(parents=True, exist_ok=True)
    manifest_path = persistent / f"manifest_eeg{subject}.json"
    manifest = read_json(manifest_path)
    if manifest.get("subject") != subject or not manifest_is_complete(persistent, manifest):
        from huggingface_hub import HfApi, snapshot_download
        from huggingface_hub.hf_api import RepoFile

        token = os.environ.get("HF_TOKEN") or False
        repo = "bcbl190626/SpanishBCBL"
        api = HfApi(token=token)
        revision = api.dataset_info(repo).sha
        print("Listing EEG files and matching MATLAB logs ...", flush=True)
        entries = api.list_repo_tree(
            repo, path_in_repo="EEG", recursive=True, repo_type="dataset", revision=revision
        )
        selected = select_files((entry for entry in entries if isinstance(entry, RepoFile)), subject)
        total = sum(row["bytes"] for row in selected)
        print(f"EEG {subject}: {len(selected)} files, {total / 1024**3:.2f} GiB", flush=True)
        snapshot_download(
            repo_id=repo, repo_type="dataset", revision=revision, token=token,
            local_dir=str(persistent), allow_patterns=[row["path"] for row in selected],
            max_workers=2,
        )
        manifest = {"subject": subject, "repo": repo, "revision": revision, "files": selected}
        if not manifest_is_complete(persistent, manifest):
            raise RuntimeError("Download incomplete: missing files or unexpected file sizes")
        write_json(manifest_path, manifest)
    else:
        print("Drive download already complete; reusing it.", flush=True)
    print("Copying this subject to the Colab local disk ...", flush=True)
    for row in manifest["files"]:
        source, target = safe_path(persistent, row["path"]), safe_path(local, row["path"])
        if not target.is_file() or target.stat().st_size != row["bytes"]:
            print(f"  {row['path']}", flush=True)
            copy_atomic(source, target)
    if not manifest_is_complete(local, manifest):
        raise RuntimeError("Local data copy is incomplete")
    count = check_brainvision_files(local, subject)
    write_json(local / "download_manifest.json", manifest)
    print(f"Ready: {count} EEG recordings with signal, marker and log files.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", required=True, type=Path)
    args = parser.parse_args()
    prepare(json.loads(args.settings.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
