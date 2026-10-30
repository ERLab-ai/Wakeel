#!/usr/bin/env python3
"""Fail closed on missing, changed, or unexpected public scientific source files."""
import hashlib, json, pathlib, sys

CANONICAL = "a97899ca74ec4a269b9a677c9618b0901de07dcbb9fc65b97059b2bd9eac6659"
APPROVED_PUBLIC_MANIFEST_SHA256 = "919b552c0427740849483e5fe8e47edc638a80bfcb351f4a3beefcb251468cb9"
EVIDENCE = {
 "_ch9_p0/golden_ch9_p0.json",
 "content/chapter-08/runs.log",
 "content/chapter-09/runs.log",
 "content/chapter-09/archive/v1.0/results.json",
}

def permitted(name):
    if name in EVIDENCE:
        return True
    if name.startswith(("_ch9_p0/", "archive/", "review-")) or "/archive/" in name or "/review-" in name:
        return False
    if name.startswith("content/"):
        import re
        return bool(re.fullmatch(r"content/chapter-0[1-9]/(?:ar\.md|en\.md|outline\.md|examples\.json|results\.json|factorial_isolation\.json|golden_v1_2\.json)",name))
    return name=="requirements.txt" or name.startswith(("data/", "schema/", "templates/", "tools/"))

def verify(root):
    root=pathlib.Path(root)
    raw=(root/"PUBLIC-MANIFEST.json").read_bytes()
    if hashlib.sha256(raw).hexdigest()!=APPROVED_PUBLIC_MANIFEST_SHA256:
        raise ValueError("Public manifest digest differs from the reviewed 75-file original-byte allowlist")
    meta=json.loads(raw)
    files=meta.get("files",{})
    if meta.get("canonical_source_sha256")!=CANONICAL or len(files)!=75:
        raise ValueError("Missing or wrong canonical archive identity / count")
    if not EVIDENCE.issubset(files):
        raise ValueError("Missing required reproducibility evidence")
    if any(not permitted(n) or n.startswith("/") or ".." in pathlib.PurePosixPath(n).parts for n in files):
        raise ValueError("Disallowed source file in public manifest")
    found={p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and not p.relative_to(root).as_posix().startswith(("out/", "__pycache__/")) and "__pycache__" not in p.parts}
    found.discard("PUBLIC-MANIFEST.json")
    if set(files)!=found:
        raise ValueError(f"Public source unexpected/missing files: extra={sorted(found-set(files))}, missing={sorted(set(files)-found)}")
    for name,sha in files.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=sha:
            raise ValueError("Changed scientific source bytes: "+name)
    return {"valid":True,"files":len(files),"canonical_source_sha256":CANONICAL}

if __name__=="__main__":
    root=sys.argv[1] if len(sys.argv)>1 else "baseline/v0.21.3"
    try: print(json.dumps(verify(root),indent=2))
    except (OSError,ValueError,KeyError) as exc:
        print("PUBLIC SOURCE CHECK FAILED: "+str(exc),file=sys.stderr);sys.exit(1)
