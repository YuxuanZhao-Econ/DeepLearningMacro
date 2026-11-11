# Stopped tokenizer pilot, not the final English model

This first from-scratch run used unrestricted BPE merges, including trailing spaces.
Inspection showed prompt endings could be tokenized very differently from the same
words in training (for example `little ` was a token while `little` was not).
We stopped it, added word/punctuation pre-tokenization, regenerated a new tokenizer,
and initialized the final model from random weights again.

These checkpoints are retained for provenance only. No pilot weights initialize the
final model. The pilot's last logged evaluation was step 14,000 at 505.6 seconds,
and it was interrupted shortly after that. It did not perform a final test evaluation.
We conservatively charge ten minutes to this pilot in the combined GPU time budget;
the final fresh run has a forty-minute process budget, leaving the total below one hour.

Saved tokenizer files identify this stopped pilot's token mapping. The current dataset
directory contains the corrected tokenizer and re-encoded streams instead.
