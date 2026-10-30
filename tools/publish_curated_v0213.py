#!/usr/bin/env python3
"""Safely stage/push only the exact reviewed Wakeel 2.0 public source ZIP.

Requires a clean local checkout of chore/v0213-baseline-import-p0 and an
authenticated Git installation. The archive must be outside the repository.
Default operation STAGES only; --push explicitly commits and pushes.
"""
import argparse
import hashlib
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

BRANCH = "chore/v0213-baseline-import-p0"
ZIP_SHA = "77da30249ce9d4baa51d17e8166ae4efdf66c03604f64352578b2270f2c217e6"
MANIFEST_SHA = "919b552c0427740849483e5fe8e47edc638a80bfcb351f4a3beefcb251468cb9"
PREFIX = "baseline/v0.21.3/"
REQUIRED_COUNT = 76
MAX_UNCOMPRESSED = 12 * 1024 * 1024

def sha(data):
    return hashlib.sha256(data).hexdigest()

def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()

def validated_entries(archive):
    if sha(archive.read_bytes()) != ZIP_SHA:
        raise ValueError("Curated ZIP SHA-256 mismatch")
    with zipfile.ZipFile(archive) as z:
        entries = z.infolist()
        names = [i.filename for i in entries]
        if len(names) != REQUIRED_COUNT or len(set(names)) != REQUIRED_COUNT:
            raise ValueError("Wrong entry count or duplicates")
        if sum(i.file_size for i in entries) > MAX_UNCOMPRESSED:
            raise ValueError("Excessive uncompressed size")
        for i in entries:
            name = i.filename
            relative = name.removeprefix(PREFIX)
            if (not name.startswith(PREFIX) or not relative
                    or any(segment in ("", ".", "..") for segment in relative.split("/"))
                    or "\\" in name or i.is_dir()
                    or ((i.external_attr >> 16) & 0o170000) == 0o120000):
                raise ValueError("Unsafe or non-baseline member: " + name)
        manifest_bytes = z.read(PREFIX + "PUBLIC-MANIFEST.json")
        if sha(manifest_bytes) != MANIFEST_SHA:
            raise ValueError("Reviewed public manifest SHA mismatch")
        manifest = json.loads(manifest_bytes)
        if manifest.get("canonical_source_sha256") != "a97899ca74ec4a269b9a677c9618b0901de07dcbb9fc65b97059b2bd9eac6659":
            raise ValueError("Wrong frozen original source identity")
        records = manifest["files"]
        actual_paths = {n.removeprefix(PREFIX) for n in names} - {"PUBLIC-MANIFEST.json"}
        if len(records) != 75 or set(records) != actual_paths:
            raise ValueError("75-file manifest coverage mismatch")
        for filename, expected in records.items():
            if not re.fullmatch(r"[a-f0-9]{64}", expected):
                raise ValueError("Invalid digest format: " + filename)
            if sha(z.read(PREFIX + filename)) != expected:
                raise ValueError("Changed approved file: " + filename)
        return {n.removeprefix(PREFIX): z.read(n) for n in names}

def stage(repo, archive):
    if git(repo, "rev-parse", "--abbrev-ref", "HEAD") != BRANCH:
        raise ValueError("Wrong checkout: switch to " + BRANCH)
    remote = git(repo, "remote", "get-url", "origin")
    if not re.fullmatch(r"(?:https://github[.]com/|git@github[.]com:|ssh://git@github[.]com/)(?:ERLab-ai/Wakeel)(?:[.]git)?/?", remote):
        raise ValueError("Unexpected GitHub origin: " + remote)
    if git(repo, "status", "--porcelain", "--untracked-files=normal"):
        raise ValueError("Working tree is not clean: commit/stash local changes first")
    destination = repo / "baseline" / "v0.21.3"
    if destination.exists():
        raise ValueError("Refusing to overwrite an existing scientific baseline")
    contents = validated_entries(archive)  # No writes before ALL digest checks pass
    with tempfile.TemporaryDirectory(prefix="wakeel-verified-") as td:
        temp = pathlib.Path(td)
        for relative, data in contents.items():
            target = temp.joinpath(*relative.split("/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        sys.path.insert(0, str(repo / "tools"))
        from verify_public_v0213 import verify
        verify(temp)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(temp, destination)
    verify(destination)
    git(repo, "add", "--", PREFIX.rstrip("/"))
    staged = git(repo, "diff", "--cached", "--name-only").splitlines()
    if len(staged) != REQUIRED_COUNT or any(not path.startswith(PREFIX) for path in staged):
        git(repo, "reset", "--", PREFIX.rstrip("/"))
        raise ValueError("Unexpected staged paths or count; commit refused")
    print(f"Verified and staged exactly {len(staged)} paths under {PREFIX}")
    return staged

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", required=True, type=pathlib.Path,
                   help="Owner-verified curated ZIP located OUTSIDE Git checkout")
    p.add_argument("--repo", type=pathlib.Path, default=pathlib.Path.cwd())
    p.add_argument("--push", action="store_true",
                   help="After staging, make a commit and push the P0 branch to origin")
    opts = p.parse_args()
    try:
        repo = opts.repo.resolve()
        archive = opts.archive.resolve()
        if repo in archive.parents:
            raise ValueError("Store the curated ZIP outside the Git checkout")
        staged = stage(repo, archive)
        if opts.push:
            subprocess.run(["git", "-C", str(repo), "commit", "-m",
                            "import(p0): stage 75 original-byte v0.21.3 source files"], check=True)
            subprocess.run(["git", "-C", str(repo), "push", "origin", "HEAD:refs/heads/" + BRANCH], check=True)
            print("PUSH COMPLETE: confirm scientific CI before closing P0")
        else:
            print("STAGED ONLY: review with 'git diff --cached --stat'; rerun with --push "
                  "only on a CLEAN checkout (or commit and push manually).")
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError, zipfile.BadZipFile) as exc:
        print("REFUSED: " + str(exc), file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
