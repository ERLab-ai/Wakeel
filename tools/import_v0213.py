#!/usr/bin/env python3
"""Verify and extract the signed-off Wakeel 2.0 v0.21.3 source without mutation."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import zipfile

ARCHIVE_SHA256 = "a97899ca74ec4a269b9a677c9618b0901de07dcbb9fc65b97059b2bd9eac6659"
EXPECTED_PACK = "WAKEEL-2-V0.21.3-SOURCE-PACK"

def digest(data):
    return hashlib.sha256(data).hexdigest()

def safe_path(s):
    return bool(s and "\\" not in s and not s.startswith("/") and
                not re.match(r"^[A-Za-z]:", s) and
                all(part not in (".", "..", "") for part in s.split("/")) and
                not PurePosixPath(s).is_absolute())

def verify(source):
    actual = digest(source.read_bytes())
    if actual != ARCHIVE_SHA256:
        raise ValueError("Archive SHA-256 mismatch: " + actual)
    with zipfile.ZipFile(source) as z:
        items = z.infolist()
        names = [x.filename for x in items]
        if len(names) != 149 or len(set(names)) != 149:
            raise ValueError("Archive entry count / duplicates")
        if any(x.is_dir() or not safe_path(x.filename) or
               stat.S_ISLNK((x.external_attr >> 16) & 0xffff) for x in items):
            raise ValueError("Unsafe archive member")
        if sum(x.file_size for x in items) > 12 * 1024 * 1024:
            raise ValueError("Uncompressed size exceeds limit")
        m = json.loads(z.read("MANIFEST.sha256.json"))
        if m["pack"] != EXPECTED_PACK:
            raise ValueError("Pack identifier mismatch")
        expected = m["files"]
        if len(expected) != 148 or set(expected) != set(names) - {"MANIFEST.sha256.json"}:
            raise ValueError("Manifest coverage mismatch")
        for name, sha in expected.items():
            if digest(z.read(name)) != sha:
                raise ValueError("Content hash mismatch: " + name)
        return {"archive_sha256": actual, "entries": len(names),
                "verified": len(expected), "uncompressed_bytes": sum(x.file_size for x in items)}

def extract(source, destination):
    result = verify(source)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Refusing nonempty destination")
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as z:
        for name in z.namelist():
            path = destination.joinpath(*PurePosixPath(name).parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(z.read(name))
        for name in z.namelist():
            if digest((destination / name).read_bytes()) != digest(z.read(name)):
                raise ValueError("Staging mismatch: " + name)
    return result

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", required=True, type=Path)
    p.add_argument("--destination", type=Path)
    p.add_argument("--report", type=Path)
    args = p.parse_args()
    try:
        r = extract(args.archive, args.destination) if args.destination else verify(args.archive)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(r, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(r, indent=2))
        print("VERIFIED; NOT PUBLISHED OR COMMITTED")
        return 0
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as e:
        print("IMPORT REFUSED: " + str(e), file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
