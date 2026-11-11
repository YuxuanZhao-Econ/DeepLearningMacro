# Short Jokes: local teaching subset

Source: [amoudgl/short-jokes-dataset](https://github.com/amoudgl/short-jokes-dataset),
by Abhinav Moudgil. Downloaded 2026-10-28 from the upstream `master/shortjokes.csv`.
The original CSV contains 231,657 rows and is preserved without modification.

SHA-256: `e6337cc85c3a1110d1f666ff9a7bffd4e2f56014bf3168ae50b7ea480f469573`.
The downloaded CSV is 24,085,786 bytes. Original construction details and the upstream
repository's GPL-2.0 license are preserved in `UPSTREAM_README.md` and `UPSTREAM_LICENSE`.
The corpus was scraped from websites and Reddit; repository licensing does not establish
ownership of all underlying joke texts. We do not substitute a mirror's license label for
the original source notice.

`experiments/prepare_jokes_data.py` uses only Python standard libraries and performs:

1. Unicode punctuation/ASCII normalization and whitespace folding.
2. Selection of question-and-answer texts with 35–180 characters and ordinary subject
   words concerning animals, objects, food, or science.
3. A conservative keyword/topic exclusion filter. This is heuristic and cannot certify
   the meaning or suitability of every retained text.
4. Deduplication ignoring case and non-alphanumeric characters.
5. Grouping by the normalized question, then seeded 90/5/5 splitting of question groups.
   Different answers to an equal question stay together. Semantic near-duplicates or
   paraphrased questions may still cross splits.
6. Concatenation within each split, with newline separators. Token windows are drawn
   only after splitting. The fixed vocabulary is newline plus 95 printable ASCII characters.

The inspected subset contains 2,675 training, 145 validation, and 177 test jokes.
`manifest.json` records the exact sizes, hashes, and seed. Each `*_records.jsonl` retains
the original source row IDs. The `*.txt` streams are consumed by the Julia trainer.

This is a narrow joke-completion experiment, with no chat roles or conversational
instruction targets. It should not be described as a general English chatbot.
