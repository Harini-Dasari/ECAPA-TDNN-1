# RedDots infos — brief explanations

This file summarizes the purpose and expected format of the files in reddots/infos.

- **f_imposter.txt / m_imposter.txt**: Imposter trial lists for female (`f_`) and male (`m_`) speakers. Typically each line encodes a trial (enroll-id, test-file, label=0). Use these for negative (non-target) trial evaluation.
- **f_target.txt / m_target.txt**: Target trial lists (positive trials) for female and male speakers. Each line is a target trial (enroll-id, test-file, label=1). Use these to compute true accepts.
- **script.txt**: Mapping of textual prompts or utterance scripts used in the dataset. Useful for text-dependent analysis and for filtering trials by prompt.
- **stat_reddots_r2015q4_v1.txt**: Dataset statistics and metadata (counts by partition, duration summaries, speaker counts). Inspect this file to understand train/dev/test splits and trial sizes.
- **reddots/readme.txt**: Dataset readme with dataset-level details and licensing/usage notes.
- **reddots/ndx/**: Index files (trial definitions). These `.ndx` files list enroll/test utterance pairs for evaluation protocol(s). They are canonical for producing scores in the RedDots benchmark.
- **reddots/pcm/**: Raw audio files in PCM format (likely 16-bit, little-endian). These are raw samples (no WAV header). To read a `.pcm` file into numpy use:

```python
import numpy as np
with open('some_file.pcm','rb') as f:
    audio = np.frombuffer(f.read(), dtype=np.int16).astype(np.float32) / 32768.0

# If sample rate is 16000 Hz, use that when downstream processing (feature extraction).
```

Notes and recommendations:
- RedDots is primarily text-dependent; do not assume speaker-independent protocols map directly to thresholds chosen on text-independent datasets (like VoxCeleb). Always tune a threshold on RedDots dev trials before applying to test.
- If you want a full line-by-line expansion (showing sample lines and exact field meanings), tell me and I'll parse the files and produce the expanded doc.

Created by the evaluation helper; next step can be a per-file line-by-line expansion.
