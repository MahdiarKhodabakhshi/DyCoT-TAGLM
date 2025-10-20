from __future__ import annotations
import argparse, hashlib, json, tarfile, zipfile
from pathlib import Path
from typing import Dict, Any
import urllib.request

import yaml

def sha256sum(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _download(url: str, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as r, out.open("wb") as f:
        f.write(r.read())

def _extract(archive: Path, target_dir: Path, strip: int = 0):
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for m in z.infolist():
                name = "/".join(m.filename.split("/")[strip:])
                if not name:
                    continue
                dest = target_dir / name
                if m.is_dir():
                    dest.mkdir(parents=True, exist_ok=True)
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(m) as src, dest.open("wb") as dst:
                        dst.write(src.read())
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as t:
            for m in t.getmembers():
                name = "/".join(m.name.split("/")[strip:])
                if not name:
                    continue
                dest = target_dir / name
                if m.isdir():
                    dest.mkdir(parents=True, exist_ok=True)
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with t.extractfile(m) as src, dest.open("wb") as dst:
                        if src: dst.write(src.read())
    else:
        raise ValueError(f"Not an archive: {archive}")

def download_from_yaml(config_path: str, only: str | None = None):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    root = Path(cfg.get("data_root", "data"))

    datasets: Dict[str, Any] = cfg.get("datasets", {})
    if only:
        datasets = {only: datasets[only]}

    for name, meta in datasets.items():
        tgt = root / meta["target_dir"]
        tgt.mkdir(parents=True, exist_ok=True)
        print(f"==> {name} -> {tgt}")

        for fmeta in meta.get("files", []):
            fname   = fmeta["name"]
            url     = fmeta["url"]
            sha     = fmeta.get("sha256")
            archive = fmeta.get("archive", False)
            strip   = int(fmeta.get("strip", 0))
            outpath = tgt / fname

            if outpath.exists():
                print(f"    exists: {fname}")
            else:
                print(f"    downloading: {fname}")
                _download(url, outpath)

            if sha:
                got = sha256sum(outpath)
                if got != sha:
                    raise RuntimeError(f"Checksum mismatch for {fname}: {got} != {sha}")
                else:
                    print(f"    checksum ok")

            if archive:
                print(f"    extracting: {fname}")
                _extract(outpath, tgt, strip=strip)
                # optional: delete archive after extracting
                # outpath.unlink(missing_ok=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/datasets.yaml")
    ap.add_argument("--only", choices=["qald9", "lcquad", "vquanda"], help="download a single dataset")
    args = ap.parse_args()
    download_from_yaml(args.config, args.only)