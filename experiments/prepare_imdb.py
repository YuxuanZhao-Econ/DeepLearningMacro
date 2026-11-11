"""Download public IMDb data and write a lossless, Julia-readable transport format.

All tokenization, splitting, vocabulary fitting and neural training happen in Julia.
Python/Arrow are used only to decode the public Parquet download.
"""
from pathlib import Path
import hashlib
import json
import urllib.request
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reference/datasets/imdb"
OUT.mkdir(parents=True, exist_ok=True)
BASE = "https://huggingface.co/datasets/stanfordnlp/imdb"

def download(url, path):
    if not path.exists():
        partial = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(url, timeout=90) as response, partial.open("wb") as target:
            while chunk := response.read(1024 * 1024):
                target.write(chunk)
        partial.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    manifest = {"source": BASE, "label_mapping": {"0": "negative", "1": "positive"},
                "transport": "source row index, label, UTF-8 text hex; tab-separated",
                "license": "other; see upstream dataset card; not represented as public domain",
                "splits": {}}
    download(BASE + "/raw/main/README.md", OUT / "UPSTREAM_DATASET_CARD.md")
    for split in ("train", "test"):
        name = f"{split}-00000-of-00001.parquet"
        path = OUT / name
        url = BASE + "/resolve/main/plain_text/" + name
        digest = download(url, path)
        rows = pq.read_table(path).to_pylist()
        assert len(rows) == 25000
        dest = OUT / (split + ".tsv")
        with dest.open("w", encoding="ascii", newline="\n") as f:
            for i, row in enumerate(rows):
                assert row["label"] in (0, 1)
                f.write(f'{i}\t{row["label"]}\t{row["text"].encode("utf-8").hex()}\n')
        manifest["splits"][split] = {
            "url": url, "rows": len(rows), "parquet_sha256": digest,
            "tsv_sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
            "positive": sum(row["label"] for row in rows)}
        print(f"{split}: {len(rows)} reviews, SHA256 {digest}", flush=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
