# RedDots Threshold Sweep Report

This report records the completed RedDots sweep for `m_part_01.ndx` and `m_part_01.trn` using the pretrained ECAPA-TDNN checkpoint.

## Run settings

- Model: `exps/pretrain.model`
- Trial list: `reddots/ndx/m_part_01.ndx`
- Enrollment list: `reddots/ndx/m_part_01.trn`
- Audio root: `reddots/pcm`
- Threshold range: `-1.0` to `1.0`
- Threshold step: `0.001`
- Output directory: `exps/red-dot`

## Output files

- `exps/red-dot/m_part_01_threshold_sweep.csv`
- `exps/red-dot/m_part_01_threshold_summary.json`
- `exps/red-dot/m_part_01_trial_scores.csv`

## Final summary

- EER: `2.54%`
- minDCF: `0.3739`
- Best accuracy threshold: `0.754`

## Notes

- The sweep uses RedDots protocol parsing from the `.trn` and `.ndx` files and raw PCM audio loading.
- The script treats `target-correct` as the positive class and all other RedDots protocol categories as negative trials for the threshold sweep.
- The sweep was run in the WSL `ecapa` environment, which has the required PyTorch stack for this repository.