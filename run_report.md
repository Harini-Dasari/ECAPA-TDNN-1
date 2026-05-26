# Run Report: RedDots ECAPA-TDNN Experiments

This file is a reproducible log of the RedDots work completed in this workspace. It records the exact commands, the observed console summaries, and the saved CSV / JSON / PNG artifacts.

## Environment

- Workspace: `C:\Users\Harini\Documents\GitHub\ECAPA-TDNN-1`
- Runtime: WSL Ubuntu 22.04
- Conda environment: `ecapa`
- Model checkpoint: [exps/pretrain.model](exps/pretrain.model)
- RedDots protocol files: [reddots/ndx/m_part_01.ndx](reddots/ndx/m_part_01.ndx), [reddots/ndx/m_part_01.trn](reddots/ndx/m_part_01.trn)
- RedDots PCM root: [reddots/pcm](reddots/pcm)
- Phrase metadata: [reddots/infos/script.txt](reddots/infos/script.txt)

## Run Log

| Step | Purpose | Command | Main result | Saved outputs |
|---|---|---|---|---|
| 1 | Baseline RedDots sweep | `wsl -d Ubuntu-22.04 -- bash -lc "source /home/harini/miniconda3/etc/profile.d/conda.sh ; conda activate ecapa ; cd /mnt/c/Users/Harini/Documents/GitHub/ECAPA-TDNN-1 ; python reddots_threshold_sweep.py --initial_model exps/pretrain.model --ndx reddots/ndx/m_part_01.ndx --trn reddots/ndx/m_part_01.trn --pcm_root reddots/pcm --output_dir exps/red-dot --output_csv exps/red-dot/m_part_01_threshold_sweep.csv --summary_json exps/red-dot/m_part_01_threshold_summary.json"` | `EER 2.54%, minDCF 0.3739, best accuracy threshold 0.754` | [m_part_01_threshold_sweep.csv](exps/red-dot/m_part_01_threshold_sweep.csv), [m_part_01_threshold_summary.json](exps/red-dot/m_part_01_threshold_summary.json), [m_part_01_trial_scores.csv](exps/red-dot/m_part_01_trial_scores.csv) |
| 2 | Focused sweep from 0.3 to 0.4 | `wsl -d Ubuntu-22.04 -- bash -lc "source /home/harini/miniconda3/etc/profile.d/conda.sh ; conda activate ecapa ; cd /mnt/c/Users/Harini/Documents/GitHub/ECAPA-TDNN-1 ; python reddots_threshold_sweep.py --initial_model exps/pretrain.model --ndx reddots/ndx/m_part_01.ndx --trn reddots/ndx/m_part_01.trn --pcm_root reddots/pcm --output_dir exps/red-dot --output_csv exps/red-dot/reddot_0.3_to_0.4.csv --summary_json exps/red-dot/reddot_0.3_to_0.4.json --threshold_start 0.3 --threshold_end 0.4 --threshold_step 0.01"` | `EER 2.65%, minDCF 0.3739, best accuracy threshold 0.400` | [reddot_0.3_to_0.4.csv](exps/red-dot/reddot_0.3_to_0.4.csv), [reddot_0.3_to_0.4.json](exps/red-dot/reddot_0.3_to_0.4.json), [m_part_01_trial_scores.csv](exps/red-dot/m_part_01_trial_scores.csv) |
| 3 | Phrase-aware high-resolution sweep from 0.28 to 0.30 | `wsl -d Ubuntu-22.04 -- bash -lc "source /home/harini/miniconda3/etc/profile.d/conda.sh ; conda activate ecapa ; cd /mnt/c/Users/Harini/Documents/GitHub/ECAPA-TDNN-1 ; python3 reddots_threshold_sweep.py --initial_model exps/pretrain.model --ndx reddots/ndx/m_part_01.ndx --trn reddots/ndx/m_part_01.trn --script_txt reddots/infos/script.txt --phrase_gate --pcm_root reddots/pcm --output_dir exps/red-dot --output_csv exps/red-dot/reddot_0.28_to_0.30_phrase.csv --summary_json exps/red-dot/reddot_0.28_to_0.30_phrase.json --threshold_start 0.28 --threshold_end 0.30 --threshold_step 0.001"` | `EER 0.62%, minDCF 0.3739, best accuracy threshold 0.300` | [reddot_0.28_to_0.30_phrase.csv](exps/red-dot/reddot_0.28_to_0.30_phrase.csv), [reddot_0.28_to_0.30_phrase.json](exps/red-dot/reddot_0.28_to_0.30_phrase.json), [m_part_01_trial_scores.csv](exps/red-dot/m_part_01_trial_scores.csv), [phrase_0.28_to_0.30.png](exps/red-dot/phrase_0.28_to_0.30.png) |
| 4 | Final fixed-threshold phrase-aware run at 0.285 | `wsl -d Ubuntu-22.04 -- bash -lc "source /home/harini/miniconda3/etc/profile.d/conda.sh ; conda activate ecapa ; cd /mnt/c/Users/Harini/Documents/GitHub/ECAPA-TDNN-1 ; python3 reddots_threshold_sweep.py --initial_model exps/pretrain.model --ndx reddots/ndx/m_part_01.ndx --trn reddots/ndx/m_part_01.trn --script_txt reddots/infos/script.txt --phrase_gate --pcm_root reddots/pcm --output_dir exps/red-dot --output_csv exps/red-dot/reddot_fixed_0.285_phrase.csv --summary_json exps/red-dot/reddot_fixed_0.285_phrase.json --threshold_start 0.285 --threshold_end 0.285 --threshold_step 0.001"` | `EER 0.62%, minDCF 0.3739, best accuracy threshold 0.285` | [reddot_fixed_0.285_phrase.csv](exps/red-dot/reddot_fixed_0.285_phrase.csv), [reddot_fixed_0.285_phrase.json](exps/red-dot/reddot_fixed_0.285_phrase.json), [m_part_01_trial_scores.csv](exps/red-dot/m_part_01_trial_scores.csv) |
| 5 | Final decision CSV | `python3 tools/generate_decisions.py --scores exps/red-dot/m_part_01_trial_scores.csv --threshold 0.285 --out exps/red-dot/m_0.285_phrase_threshold.csv` | Decision table written at threshold 0.285 | [m_0.285_phrase_threshold.csv](exps/red-dot/m_0.285_phrase_threshold.csv) |

## Plot Commands

These commands regenerate the main figures from the saved sweep files.

```bash
python3 plot_reddots_threshold_curves.py --csv exps/red-dot/m_part_01_threshold_sweep.csv --output exps/red-dot/m_part_01_far_frr_vs_threshold.png
```

```bash
C:/Users/Harini/Documents/GitHub/ECAPA-TDNN-1/.venv/Scripts/python.exe plot_reddots_threshold_curves.py --csv exps/red-dot/reddot_0.3_to_0.4.csv --output exps/red-dot/graph_0.3_to_0.4.png
```

```bash
python3 plot_reddots_compare_thresholds.py --speaker_csv exps/red-dot/m_part_01_threshold_sweep.csv --phrase_csv exps/red-dot/m_part_01_phrase_gate_threshold_sweep.csv --out exps/red-dot/compare_far_frr.png
```

```bash
python3 plot_reddots_single_thresholds.py --csv exps/red-dot/reddot_0.28_to_0.30_phrase.csv --out exps/red-dot/phrase_0.28_to_0.30.png
```

## Main Figures

- [exps/red-dot/m_part_01_far_frr_vs_threshold.png](exps/red-dot/m_part_01_far_frr_vs_threshold.png)
- [exps/red-dot/m_part_01_far_frr_vs_threshold1.svg](exps/red-dot/m_part_01_far_frr_vs_threshold1.svg)
- [exps/red-dot/graph_0.3_to_0.4.png](exps/red-dot/graph_0.3_to_0.4.png)
- [exps/red-dot/graph_0.3_to_0.4.svg](exps/red-dot/graph_0.3_to_0.4.svg)
- [exps/red-dot/compare_far_frr.png](exps/red-dot/compare_far_frr.png)
- [exps/red-dot/compare_0.25_to_0.40.png](exps/red-dot/compare_0.25_to_0.40.png)
- [exps/red-dot/phrase_0.28_to_0.30.png](exps/red-dot/phrase_0.28_to_0.30.png)

## Other Key Artifacts

- [exps/red-dot/reddots_embeddings_20c54fd2f3ecd2f8_m_part_01_m_part_01.pt](exps/red-dot/reddots_embeddings_20c54fd2f3ecd2f8_m_part_01_m_part_01.pt)
- [exps/red-dot/m_part_01_phrase_gate_threshold_sweep.csv](exps/red-dot/m_part_01_phrase_gate_threshold_sweep.csv)
- [exps/red-dot/m_part_01_phrase_gate_summary.json](exps/red-dot/m_part_01_phrase_gate_summary.json)
- [exps/red-dot/reddot_0.25_to_0.4_phrase.csv](exps/red-dot/reddot_0.25_to_0.4_phrase.csv)
- [exps/red-dot/reddot_0.25_to_0.4_phrase.json](exps/red-dot/reddot_0.25_to_0.4_phrase.json)

## Related VoxCeleb Outputs

These files were also produced in the workspace and belong to the same project log:

- [exps/final_eval/veri_test2_threshold_310_summary.csv](exps/final_eval/veri_test2_threshold_310_summary.csv)
- [exps/final_eval/veri_test2_score_distributions.png](exps/final_eval/veri_test2_score_distributions.png)
- [exps/final_eval/veri_test2_far_frr_curve.png](exps/final_eval/veri_test2_far_frr_curve.png)
- [exps/final_eval/final_eval_summary.csv](exps/final_eval/final_eval_summary.csv)

## Short Interpretation

- Baseline RedDots EER was about 2.54%.
- Phrase-aware gating reduced the reported EER to about 0.62% in the focused sweep.
- The final fixed threshold used for the decision CSV was 0.285.

## Re-run Checklist

1. Run the baseline full sweep.
2. Run the focused 0.3 to 0.4 sweep.
3. Run the phrase-aware 0.28 to 0.30 sweep.
4. Run the fixed 0.285 phrase-aware sweep.
5. Generate the final decision CSV.
6. Regenerate the plots from the saved sweep outputs.
