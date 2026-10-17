import argparse
import os
import tarfile
from pathlib import Path


def pack(cache: Path, archive: Path):
    cache, archive = cache.resolve(), archive.resolve()
    if not cache.is_dir() or archive.is_relative_to(cache):
        raise ValueError("Cache must exist and archive must be outside it")
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_name(archive.name + ".tmp")
    try:
        with tarfile.open(temporary, "w") as target:
            for root, dirs, files in os.walk(cache, followlinks=False):
                dirs[:] = sorted(d for d in dirs if not (Path(root) / d).is_symlink())
                for name in sorted(files):
                    path = Path(root) / name
                    if not path.is_symlink() and not name.endswith((".lock", ".tmp")):
                        target.add(
                            path,
                            arcname=path.relative_to(cache).as_posix(),
                            recursive=False,
                        )
        temporary.replace(archive)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Archived {archive.stat().st_size / 1024**3:.3f} GiB to {archive}", flush=True)


def unpack(archive: Path, cache: Path):
    archive, cache = archive.resolve(), cache.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as source:
        # The data filter also rejects escaping paths and unsafe file types.
        for member in source:
            if not (member.isfile() or member.isdir()):
                raise ValueError(
                    f"Cache archives must contain regular files only: {member.name}"
                )
            destination = (cache / member.name).resolve()
            if not destination.is_relative_to(cache):
                raise ValueError(f"Archive path escapes cache: {member.name}")
            source.extract(member, path=cache, filter="data")
    print(f"Cache extracted to {cache}", flush=True)


def main():
    p = argparse.ArgumentParser(
        description="Archive CPU features; restore onto the GPU runtime's local disk"
    )
    p.add_argument("mode", choices=["pack", "unpack"])
    p.add_argument("--cache-root", required=True, type=Path)
    p.add_argument("--archive", required=True, type=Path)
    args = p.parse_args()
    if args.mode == "pack":
        pack(args.cache_root, args.archive)
    else:
        unpack(args.archive, args.cache_root)


if __name__ == "__main__":
    main()
