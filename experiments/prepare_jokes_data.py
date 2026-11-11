"""Prepare a small English question-and-answer jokes corpus using only stdlib.

Run with any Python 3: python experiments/prepare_jokes_data.py
The source CSV and upstream notices live in reference/datasets/short_jokes.
"""
import collections
import csv
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "reference/datasets/short_jokes"
SOURCE = "https://raw.githubusercontent.com/amoudgl/short-jokes-dataset/master/shortjokes.csv"
EXPECTED_SHA256 = "e6337cc85c3a1110d1f666ff9a7bffd4e2f56014bf3168ae50b7ea480f469573"

# A conservative topic/word filter for this ordinary-humor teaching experiment.
# This is a heuristic, not a semantic guarantee or an upstream content label.
BLOCK = re.compile(
    r"\b(?:fuck\w*|shit\w*|bitch\w*|cunt\w*|cock\w*|dick\w*|pussy\w*|"
    r"sex\w*|porn\w*|naked|nude\w*|orgasm\w*|penis\w*|vagina\w*|boob\w*|"
    r"breast\w*|anus|anal|rape\w*|molest\w*|incest\w*|pedoph\w*|"
    r"nigg\w*|fag\w*|retard\w*|trann\w*|whore\w*|slut\w*|"
    r"jew\w*|muslim\w*|islam\w*|christ\w*|black|white|asian\w*|chinese|"
    r"indian\w*|mexican\w*|african\w*|arab\w*|blonde\w*|gay\w*|lesbian\w*|"
    r"transgender\w*|women|woman|wife|wives|girlfriend\w*|husband\w*|"
    r"kill\w*|murder\w*|suicid\w*|cancer\w*|aids|dead|death|died|"
    r"hitler|nazi\w*|holocaust|terror\w*|trump|obama|clinton|"
    r"abuse\w*|divorc\w*|drug\w*|cocaine|heroin|alcohol\w*|drunk\w*|"
    r"abortion\w*|prostitut\w*|cum|ass|arse\w*|ball\w*|screw\w*|"
    r"strip\w*|priest\w*|child\w*|kid\w*|boy\w*|girl\w*|man|men|"
    r"father\w*|mother\w*|dad\w*|mom\w*|orphan\w*|eth\w*|drown\w*|"
    r"lynch\w*|slave\w*|hang\w*|shoot\w*|gun\w*|bomb\w*|bleed\w*|"
    r"piss\w*|poop\w*|fart\w*|toilet\w*|butt\w*|virgin\w*|hell|"
    r"crippl\w*|disab\w*|deaf|blind|fat|ugly|stupid|dumb|satan\w*)\b",
    re.I,
)

# Require an ordinary object/animal/science/food subject in the question.
# This intentionally sacrifices size for a narrower teaching corpus.
TOPICS = (r"computers?|programmers?|robots?|laptops?|keyboards?|mice|mouse|"
    r"mathematicians?|scientists?|physicists?|chemists?|biologists?|"
    r"atoms?|electrons?|protons?|neutrons?|photons?|numbers?|triangles?|circles?|"
    r"cats?|dogs?|chickens?|ducks?|cows?|sheep|goats?|pigs?|horses?|fish|"
    r"bears?|penguins?|elephants?|giraffes?|lions?|tigers?|rabbits?|bunnies|"
    r"frogs?|turtles?|snakes?|spiders?|bees?|birds?|parrots?|owls?|crabs?|"
    r"lobsters?|sharks?|whales?|dolphins?|dinosaurs?|squirrels?|pandas?|"
    r"tomatoes|tomato|potatoes|potato|bananas?|oranges?|apples?|grapes?|"
    r"lemons?|carrots?|onions?|mushrooms?|peppers?|cucumbers?|peas?|"
    r"bread|cookies?|biscuits?|cakes?|pizzas?|cheese|eggs?|sausages?|"
    r"coffee|tea|milk|noodles?|spaghetti|pasta|rice|pancakes?|waffles?|"
    r"books?|pencils?|pens?|erasers?|rulers?|clocks?|watches|watch|"
    r"bicycles?|bikes?|cars?|trains?|boats?|ships?|airplanes?|planes?|"
    r"trees?|flowers?|plants?|leaves|leaf|grass|rocks?|stones?|mountains?|"
    r"clouds?|sun|moon|stars?|planets?|earth|snow|rain|wind|"
    r"guitars?|pianos?|drums?|trumpets?|violins?|scarecrows?|snowmen|snowman")
SUBJECT = re.compile(
    r"^(?:q\s*[:.]\s*)?(?:(?:why|how|what|where|when) "
    r"(?:did|does|do|is|are|was|were|can|could|would|will|should|can't|couldn't|"
    r"didn't|don't|doesn't|isn't|wasn't) |what do you (?:call|get) )"
    r"(?:(?:a|an|the|two|three|some) )?(?:" + TOPICS + r")\b", re.I)


def normalize(text):
    text = text.translate(str.maketrans({"\u2019": "'", "\u2018": "'", "\u201c": '"',
        "\u201d": '"', "\u2014": "-", "\u2013": "-", "\u2026": "..."}))
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", text).strip()


def main():
    raw_sha = hashlib.sha256((DATA / "shortjokes.csv").read_bytes()).hexdigest()
    assert raw_sha == EXPECTED_SHA256, "Source CSV differs from the inspected version."
    groups = collections.defaultdict(list)
    seen = set()
    counts = collections.Counter()
    with (DATA / "shortjokes.csv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            counts["raw_rows"] += 1
            joke = normalize(row["Joke"])
            if not (35 <= len(joke) <= 180) or "?" not in joke:
                continue
            if BLOCK.search(joke) or re.search(r"https?://|www\.|[<>]|\br/", joke, re.I):
                continue
            question, answer = joke.split("?", 1)
            if len(question.split()) < 4 or len(re.findall(r"[a-zA-Z]", answer)) < 8:
                continue
            if not SUBJECT.search(question):
                continue
            # Group equal questions even when the supplied punchlines differ.
            # Punctuation/case differences cannot put the same question in two splits.
            key = re.sub(r"[^a-z0-9]", "", question.lower())
            fingerprint = re.sub(r"[^a-z0-9]", "", joke.lower())
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            groups[key].append({"source_id": row["ID"], "text": joke})
    keys = sorted(groups)
    random.Random(20261028).shuffle(keys)
    n = len(keys)
    boundaries = (int(.90 * n), int(.95 * n))
    split_keys = {"train": keys[:boundaries[0]], "validation": keys[boundaries[0]:boundaries[1]],
                  "test": keys[boundaries[1]:]}
    manifest = {"source_url": SOURCE, "source_sha256": raw_sha, "raw_rows": counts["raw_rows"],
        "seed": 20261028, "selection": "ASCII Q/A jokes, 35-180 characters; allowlisted ordinary subjects and heuristic exclusion words",
        "split_method": "90/5/5 by normalized question group, before concatenation/token windows",
        "near_duplicate_limit": "Paraphrased or semantically similar questions may remain across splits.",
        "splits": {}}
    train_chars = None
    for name, group_keys in split_keys.items():
        rows = [row for key in group_keys for row in groups[key]]
        text = "\n" + "\n".join(row["text"] for row in rows) + "\n"
        path = DATA / f"{name}.txt"
        path.write_bytes(text.encode("ascii"))
        with (DATA / f"{name}_records.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")
        if name == "train":
            train_chars = set(text)
        manifest["splits"][name] = {"jokes": len(rows), "question_groups": len(group_keys),
            "characters": len(text), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    # Fix a readable character alphabet in advance; no validation text is used to fit it.
    alphabet = "\n" + "".join(chr(c) for c in range(32, 127))
    (DATA / "vocab.txt").write_bytes(alphabet.encode("ascii"))
    manifest["vocabulary_size"] = len(alphabet)
    manifest["vocabulary"] = "newline plus 95 printable ASCII characters, fixed before training"
    assert train_chars <= set(alphabet)
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
