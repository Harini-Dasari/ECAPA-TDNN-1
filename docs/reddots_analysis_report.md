# RedDots Evaluation Analysis

## Short answer
Yes, you can evaluate on the RedDots corpus, but not with the current Datasets/VeriTest setup unchanged. The existing evaluation pipeline was built around VoxCeleb-style trial lists and WAV reading, while RedDots uses its own `.trn` / `.ndx` protocols and raw `.pcm` audio files.

## What changes for RedDots
- RedDots uses text-dependent protocols with separate parts and trial lists.
- The corpus stores audio as raw PCM, not standard WAV files.
- The current `final_evaluate.py` and `trainECAPAModel.py` evaluation flow expects a VoxCeleb-style `eval_list` and `soundfile.read`-friendly audio.

## Can threshold `0.31` be reused?
- As a quick baseline, yes, you can try `0.31`.
- As a final threshold, no, it should not be assumed to transfer from VoxCeleb text-independent evaluation to RedDots text-dependent evaluation.
- RedDots thresholds should be calibrated on RedDots development trials, and often separated by part / protocol / gender.

## Practical recommendation
1. Convert or load RedDots `.pcm` files correctly.
2. Build a RedDots-specific trial list from the `.trn` and `.ndx` files.
3. Run score extraction on a RedDots dev set.
4. Tune the threshold from RedDots scores.
5. Use the tuned threshold for the final RedDots test report.

## Conclusion
- **Possible to evaluate RedDots?** Yes.
- **Possible to use the exact same pipeline unchanged?** No.
- **Possible to keep threshold 0.31 as the final threshold?** Not recommended.
- **Best practice:** calibrate a RedDots-specific threshold before final reporting.
