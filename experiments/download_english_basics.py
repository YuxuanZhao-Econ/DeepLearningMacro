"""Bounded public-text downloads; never downloads model weights."""
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reference/datasets/english_basics"
BASE = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
OUT.mkdir(parents=True, exist_ok=True)
records = {}
for name, limit in [("TinyStoriesV2-GPT4-train.txt",64*1024**2),
                    ("TinyStoriesV2-GPT4-valid.txt",4*1024**2)]:
    dest = OUT / name
    if not dest.exists():
        request = urllib.request.Request(BASE+name,headers={"Range":f"bytes=0-{limit-1}","User-Agent":"local-nlp-teaching/1.0"})
        with urllib.request.urlopen(request, timeout=90) as response:
            chunks, total = [], 0
            while total < limit:
                chunk = response.read(min(1024**2,limit-total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
            raw = b"".join(chunks)
            end = raw.rfind(b"<|endoftext|>")
            assert end > 1024**2, "Too little complete text downloaded."
            raw = raw[:end+len(b"<|endoftext|>")]
            dest.write_bytes(raw)
    raw = dest.read_bytes()
    records[name] = {"url":BASE+name,"download":"bounded prefix ending at last complete document",
                     "max_bytes":limit,"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}
    print(name,len(raw),"bytes",flush=True)
notice = OUT / "TINYSTORIES_UPSTREAM_README.md"
if not notice.exists():
    notice.write_bytes(urllib.request.urlopen(BASE+"README.md",timeout=30).read())
(OUT/"download_manifest.json").write_text(json.dumps(records,indent=2)+"\n")
