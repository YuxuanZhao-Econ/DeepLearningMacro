"""TinyStories-only foundation experiment; learn the tokenizer from training text.

Uses tokenizers==0.22.2 from the isolated local temporary package directory.
Training uses public text, not pretrained tokenizer/model parameters.
"""
import array
import hashlib
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tmp/english_bpe_python_deps"))
from tokenizers import Tokenizer, Regex, models, trainers, decoders, pre_tokenizers
OUT = ROOT/"reference/datasets/english_basics"
SPECIAL = ["<unk>","<eos>","<bos>"]
SEED = 20261028

def normalize(text):
    text = text.translate(str.maketrans({"\u2019":"'","\u2018":"'","\u201c":'"',"\u201d":'"',
                                      "\u2013":"-","\u2014":"-","\u2026":"..."}))
    text = unicodedata.normalize("NFKD",text).encode("ascii","ignore").decode()
    return re.sub(r"\s+"," ",text).strip()

def load_stories(path):
    rows, seen = [], set()
    for s in path.read_text(encoding="utf-8").split("<|endoftext|>"):
        s = normalize(s)
        fingerprint = hashlib.sha256(s.encode()).hexdigest()
        if 200 <= len(s) <= 7000 and fingerprint not in seen:
            seen.add(fingerprint)
            rows.append(s)
    return rows

def main():
    train = load_stories(OUT/"TinyStoriesV2-GPT4-train.txt")
    train_hashes = {hashlib.sha256(s.encode()).hexdigest() for s in train}
    held = [s for s in load_stories(OUT/"TinyStoriesV2-GPT4-valid.txt")
            if hashlib.sha256(s.encode()).hexdigest() not in train_hashes]
    random.Random(SEED).shuffle(held)
    split = len(held)//2
    corpora = {"train":train,"validation":held[:split],"test":held[split:]}
    fit, used = [], 0
    for s in train:
        fit.append(s)
        used += len(s)
        if used >= 16*1024**2:
            break
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Split(
        Regex(r" ?[A-Za-z]+| ?[0-9]+| ?[^A-Za-z0-9\s]+|\s+"),behavior="isolated")
    tokenizer.decoder = decoders.Fuse()
    trainer = trainers.BpeTrainer(vocab_size=2048,min_frequency=5,max_token_length=24,
        initial_alphabet=["\n"]+[chr(i) for i in range(32,127)],special_tokens=SPECIAL,show_progress=False)
    print(f"Fitting a new 2048-token BPE on {used:,} training characters.",flush=True)
    tokenizer.train_from_iterator(fit,trainer=trainer)
    tokenizer.save(str(OUT/"tokenizer.json"))
    tok = json.loads((OUT/"tokenizer.json").read_text())
    vocabulary = tok["model"]["vocab"]
    ordered = sorted(vocabulary,key=vocabulary.get)
    (OUT/"tokens_hex.txt").write_text("\n".join(s.encode().hex() for s in ordered)+"\n")
    with (OUT/"merges.tsv").open("w") as f:
        for left,right in tok["model"]["merges"]:
            f.write(f"{vocabulary[left]+1}\t{vocabulary[right]+1}\t{vocabulary[left+right]+1}\n")
    manifest = {"seed":SEED,"vocab_size":len(vocabulary),"bpe_fit_training_characters":used,
        "special_tokens":{s:vocabulary[s]+1 for s in SPECIAL},
        "source":"https://huggingface.co/datasets/roneneldan/TinyStories",
        "source_description":"Public GPT-4-generated simple English stories, bounded file prefixes; no model weights downloaded",
        "license":"CDLA-Sharing-1.0","tokenizer":"2048-token BPE with word/punctuation pre-tokenization; training text only",
        "split":"Official train prefix for training; official valid prefix deduplicated against train, shuffled and halved into local validation/test",
        "deduplication":"Exact normalized story hashes; semantic or paraphrased duplicates may remain",
        "corpora":{}}
    for name,rows in corpora.items():
        stream = array.array("H")
        for start in range(0,len(rows),256):
            for encoded in tokenizer.encode_batch(rows[start:start+256]):
                assert vocabulary["<unk>"] not in encoded.ids
                stream.append(vocabulary["<bos>"]+1)
                stream.extend(i+1 for i in encoded.ids)
                stream.append(vocabulary["<eos>"]+1)
        if sys.byteorder != "little":
            stream.byteswap()
        path = OUT/f"{name}.bin"
        path.write_bytes(stream.tobytes())
        (OUT/f"{name}.txt").write_bytes(("\n".join(rows)+"\n").encode("ascii"))
        manifest["corpora"][name] = {"documents":len(rows),"characters":sum(map(len,rows)),
            "tokens":len(stream),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
        print(name,manifest["corpora"][name],flush=True)
    probes = ["Once upon a time, there was a little", "The cat looked at the dog and said,",
              '"Hello," said Tom. "How are you?', "One day, a little girl found a", "a\na\n a"]
    probes += train[:3]+corpora["validation"][:3]
    with (OUT/"tokenizer_probes.tsv").open("w") as f:
        for s in probes:
            f.write(s.encode().hex()+"\t"+",".join(str(i+1) for i in tokenizer.encode(s).ids)+"\n")
    manifest["tokenizer_sha256"] = hashlib.sha256((OUT/"tokenizer.json").read_bytes()).hexdigest()
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print("TinyStories-only corpus and new tokenizer saved.",flush=True)

if __name__ == "__main__":
    main()
