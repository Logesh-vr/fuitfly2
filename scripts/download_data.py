"""Download published data without executing downloads or deserializing pickle files."""
from pathlib import Path
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "91bdd1e7dcf193f3e7ca5a8933497fcef63b7960"
BASE = f"https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/{COMMIT}"
FILES = {
    "vendor/shiu/model.py": f"{BASE}/model.py",
    "vendor/shiu/LICENSE": f"{BASE}/LICENSE",
    "data/flywire783/Completeness_783.csv": f"{BASE}/Completeness_783.csv",
    "data/flywire783/Connectivity_783.parquet": f"{BASE}/Connectivity_783.parquet",
    "data/flywire783/classification.csv.gz": "https://storage.googleapis.com/flywire-data/codex/data/fafb/783/classification.csv.gz",
}


def main():
    manifest_path = ROOT / "data_manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest = {}
    for relative, url in FILES.items():
        path = ROOT / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            print(f"Downloading {relative}", flush=True)
            temporary = path.with_suffix(path.suffix + ".part")
            urllib.request.urlretrieve(url, temporary)
            temporary.replace(path)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if relative in previous and sha != previous[relative]["sha256"]:
            raise RuntimeError(f"Checksum mismatch for {relative}; refusing to silently change dataset")
        manifest[relative] = {"url": url, "sha256": sha, "bytes": path.stat().st_size}
        print(f"Verified {relative}: {sha}", flush=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
