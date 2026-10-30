#!/usr/bin/env python3
"""Build a curated 75-file source candidate from the exact frozen original ZIP."""
import argparse, hashlib, json, pathlib, zipfile
from import_v0213 import verify

CANONICAL="a97899ca74ec4a269b9a677c9618b0901de07dcbb9fc65b97059b2bd9eac6659"
CLASSIFICATION="305af558ed5fc584787217dcf2b2470a682a690ae781841252a611d5b130425b"
EVIDENCE={"_ch9_p0/golden_ch9_p0.json","content/chapter-08/runs.log","content/chapter-09/runs.log","content/chapter-09/archive/v1.0/results.json"}
CATEGORIES={"educational-scientific-source","source-tooling"}
def digest(b):return hashlib.sha256(b).hexdigest()

def prepare(archive,classification,output):
    verify(archive)
    raw=classification.read_bytes()
    if digest(raw)!=CLASSIFICATION:
        raise ValueError("Unreviewed source classification")
    report=json.loads(raw)
    with zipfile.ZipFile(archive) as z:
        if set(report["files"])!=set(z.namelist()):
            raise ValueError("Classification / canonical source file list differs")
        files=sorted(n for n,v in report["files"].items() if v["category"] in CATEGORIES or n in EVIDENCE)
        if len(files)!=75 or not EVIDENCE.issubset(files):
            raise ValueError("Incorrect public file count or evidence set")
        data={}
        for name in files:
            b=z.read(name)
            if digest(b)!=report["files"][name]["sha256"]:
                raise ValueError("File digest mismatch: "+name)
            data[name]=b
    from verify_public_v0213 import permitted
    if any(not permitted(name) for name in data):
        raise ValueError("Disallowed path in public-source candidate")
    manifest={"schema":"wakeel2-public-source-manifest/v1","source_version":"v0.21.3",
              "canonical_source_sha256":CANONICAL,"classification_report_sha256":CLASSIFICATION,
              "count_original_source_files":149,"count_public_source_files":75,
              "scope":"Source for public Git repository, NOT a website deployment allowlist",
              "evidence_required_by_readonly_scientific_tests":sorted(EVIDENCE),
              "files":{n:digest(b) for n,b in sorted(data.items())}}
    if output.exists():raise FileExistsError("Refusing to overwrite existing ZIP")
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        data["PUBLIC-MANIFEST.json"]=(json.dumps(manifest,ensure_ascii=False,sort_keys=True,indent=2)+"\n").encode()
        for name, b in sorted(data.items()):
            f=zipfile.ZipInfo("baseline/v0.21.3/"+name,date_time=(2026,10,30,0,0,0))
            f.compress_type=zipfile.ZIP_DEFLATED;f.external_attr=0o644<<16
            z.writestr(f,b,compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
    print(json.dumps({"public_source_files":75,"source_zip_sha256":CANONICAL,"candidate_sha256":digest(output.read_bytes())},indent=2))

if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--archive",required=True,type=pathlib.Path)
    a.add_argument("--classification",required=True,type=pathlib.Path)
    a.add_argument("--output",required=True,type=pathlib.Path)
    x=a.parse_args()
    prepare(x.archive,x.classification,x.output)
