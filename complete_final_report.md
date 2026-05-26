# Complete Final Report: ECAPA-TDNN on VoxCeleb (Text-Independent) and RedDots (Text-Dependent)

## Key Threshold Summary

For the text-independent VoxCeleb evaluation, the ECAPA-TDNN model achieved an EER of 1.24% at threshold 0.30, with accuracy 98.98%.

- Accuracy at 0.30: 0.98979
- EER: 0.01244, about 1.24%
- FAR at EER threshold: 0.01244
- FRR at EER threshold: 0.00798
- minDCF threshold: 0.3825

RedDots m_part_01 at threshold 0.37:

- FAR = 0.02648
- FRR = 0.02406
- Accuracy = 0.97352

So the error rate at 0.37 is about 2.65% on the impostor side and 2.41% on the genuine side.

Important detail: EER is not exactly a value "at 0.37"; it is the threshold where FAR and FRR are closest. In this run, the EER point is around:

- EER threshold ≈ 0.376
- EER ≈ 0.02535, i.e. 2.54%

## 1. Executive Summary

This report documents the full evaluation workflow and outcomes for two speaker verification settings using the same ECAPA-TDNN backbone:

1. VoxCeleb-style text-independent (TI) verification.
2. RedDots text-dependent (TD) verification (m_part_01 protocol).

The experiments confirm that threshold behavior is task-dependent:

- A threshold that is effective for TI data is not automatically optimal for TD data.
- For RedDots m_part_01, the balanced operating point is near 0.376 (EER point), while a low-reject operating point around 0.31 produces much higher FAR.
- For VoxCeleb TI evaluation in this workspace, the best operating point from the saved sweep is around 0.30 with low EER.

Main takeaway:

- Use separate calibrated thresholds for TI and TD tasks.
- Prefer EER/minDCF-based selection, not raw accuracy alone (especially on highly imbalanced trial sets).

## 2. Research Questions and Self-Checks

To make this report submission-ready, the analysis was guided by explicit self-questions:

1. Are TI and TD tasks evaluated on different protocols and score distributions?
2. Are all reported numbers traceable to saved experiment artifacts?
3. Is threshold choice justified by FAR/FRR/EER/minDCF, not only accuracy?
4. Are dataset characteristics (trial imbalance, protocol style) reflected in interpretation?
5. Is the report reproducible from commands and file references?

All answers are addressed in Sections 3 to 12.

## 3. Model and System Architecture

The model is ECAPA-TDNN with AAM-Softmax training.

Implementation references:

- model definition: [model.py](model.py)
- training/evaluation wrapper: [ECAPAModel.py](ECAPAModel.py)
- training/eval entrypoint: [trainECAPAModel.py](trainECAPAModel.py)
- custom RedDots sweep: [reddots_threshold_sweep.py](reddots_threshold_sweep.py)
- RedDots transcript/phrase index: [reddots/infos/script.txt](reddots/infos/script.txt)

### 3.1 Architecture details

From [model.py](model.py):

- Input front-end:
  - Pre-emphasis filter.
  - 80-bin log Mel spectrogram (16 kHz, n_fft=512, win_length=400, hop_length=160).
- Data augmentation in feature space:
  - SpecAug (time and frequency masking) during training.
- Encoder backbone:
  - Initial Conv1D + BatchNorm + ReLU.
  - Three Bottle2neck blocks with dilations 2, 3, 4 and squeeze-excitation (SE).
  - Multi-layer feature aggregation (concat of x1/x2/x3) -> 1x1 Conv.
- Pooling/statistics:
  - Attentive statistics pooling using weighted mean and weighted std.
- Embedding head:
  - BN -> Linear(3072 -> 192) -> BN.
  - Final embedding dimension: 192.

### 3.2 Loss and optimization

From [loss.py](loss.py) and [ECAPAModel.py](ECAPAModel.py):

- Loss: AAM-Softmax with margin m=0.2 and scale s=30.
- Optimizer: Adam, lr=0.001, weight decay=2e-5.
- Scheduler: StepLR with decay gamma=0.97 each test step.

### 3.3 Scoring procedure

For evaluation, each utterance produces:

- Full-utterance embedding.
- Five split-segment embeddings.

Pair score is the mean of:

- cosine-like dot product on full embeddings,
- cosine-like dot product on split embeddings.

This is implemented in [ECAPAModel.py](ECAPAModel.py) and mirrored in [reddots_threshold_sweep.py](reddots_threshold_sweep.py).

## 4. Datasets and Protocols

## 4.0 RedDots Methodology: What Was Done From the Files

This section describes the RedDots evaluation pipeline in a formal, reproducible manner, beginning with the corpus protocol files and ending with the saved threshold-analysis artifacts.

### Step 1: Understand the corpus layout

The corpus organization and evaluation protocol were determined from [reddots/readme.txt](reddots/readme.txt):

- RedDots provides raw 16 kHz PCM speech.
- The protocol is organized into `.trn` enrollment files and `.ndx` trial files.
- Part 01 corresponds to a text-dependent common pass-phrase task.
- The documentation also reports the number of target and impostor trials, which explains the pronounced class imbalance observed during evaluation.

The file [reddots/infos/script.txt](reddots/infos/script.txt) was also used as a supporting RedDots metadata source. It contains utterance identifiers paired with phrase text, which is useful for verifying the spoken content associated with each recording, for cross-checking transcript-level pass-phrase information, and for phrase-wise analysis of RedDots trial subsets.

Importantly, this file does not alter the enrollment templates, the trial score computation, or the threshold sweep by itself. Any change in accuracy or EER would require a prompt-aware scoring stage, a phrase-verification component, or a different trial filtering policy.

In the updated sweep script, [reddots_threshold_sweep.py](reddots_threshold_sweep.py) now accepts [reddots/infos/script.txt](reddots/infos/script.txt) through the `--script_txt` argument and writes the phrase text into the trial-score CSV as metadata columns. This makes the rerun traceable and phrase-aware for analysis, while preserving the same ECAPA-TDNN scoring rule.

### Step 2: Read the enrollment file

The file [reddots/ndx/m_part_01.trn](reddots/ndx/m_part_01.trn) specifies the enrollment speaker-sentence identities and the three enrollment utterances associated with each entry.

Processing performed on this file:

- Parsed each enrollment identifier.
- Retrieved the three utterance paths corresponding to that enrollment.
- Loaded the utterances from [reddots/pcm/](reddots/pcm/) and passed them through the pretrained model.
- Averaged the resulting embeddings to obtain one enrollment template per speaker-sentence pair.

### Step 3: Read the trial file

The file [reddots/ndx/m_part_01.ndx](reddots/ndx/m_part_01.ndx) defines the verification trials used for scoring.

Each line contains:

- enrollment ID
- test utterance
- four Y/N flags that determine the RedDots trial category

These flags were mapped to the four standard RedDots labels:

- target-correct
- target-wrong
- imposter-correct
- imposter-wrong

For scoring, only `target-correct` was treated as the genuine class; all remaining trial types were treated as non-genuine.

### Step 4: Load audio from the PCM folder

The audio files in [reddots/pcm/](reddots/pcm/) are raw PCM waveforms.

For each utterance:

- The `.pcm` file was read as 16-bit signed audio.
- The waveform was normalized to floating-point values.
- The normalized waveform was passed through the pretrained ECAPA-TDNN encoder.

### Step 5: Extract embeddings with the pretrained model

The pretrained checkpoint was loaded from [exps/pretrain.model](exps/pretrain.model) by [reddots_threshold_sweep.py](reddots_threshold_sweep.py).

For every utterance, two embedding views were computed:

- a full-utterance embedding
- five split-window embeddings

This design mirrors the evaluation strategy used for VoxCeleb and improves robustness across varying utterance lengths.

### Step 6: Build enrollment templates

For each enrollment ID:

- The embeddings of the three enrollment utterances were stacked.
- The full embeddings were averaged to form one full enrollment template.
- The split embeddings were averaged to form one split enrollment template.

This produced a single template representation for each enrollment speaker-sentence pair.

### Step 7: Score each trial pair

For every trial in [reddots/ndx/m_part_01.ndx](reddots/ndx/m_part_01.ndx):

- The enrollment template was compared against the test utterance.
- One similarity score was computed from the full embeddings.
- A second similarity score was computed from the split embeddings.
- The two scores were averaged to produce the final trial score.

This final score is the value later compared against the decision threshold.

### Step 8: Sweep thresholds

After scoring all trials, thresholds were swept across the score range.

For the full RedDots run, the threshold search was:

- start: -1.0
- end: 1.0
- step: 0.001

For the focused report sweep, an additional localized search was performed:

- start: 0.3
- end: 0.4
- step: 0.01

At each threshold, the following quantities were computed:

- FAR = false accepts / impostor trials
- FRR = false rejects / genuine trials
- Accuracy = (true accepts + true rejects) / total trials

### Step 9: Choose operating points

From the sweep, three operating points were selected for reporting:

- EER threshold: where FAR and FRR are closest
- minDCF threshold: where the detection cost is minimized
- best-accuracy threshold: where overall accuracy is highest

For RedDots m_part_01, these values were approximately:

- EER threshold ≈ 0.376
- minDCF threshold ≈ 0.4777
- best-accuracy threshold ≈ 0.754

### Step 10: Save the outputs

The pipeline saved the following outputs in [exps/red-dot/](exps/red-dot/):

- [m_part_01_threshold_sweep.csv](exps/red-dot/m_part_01_threshold_sweep.csv)
- [m_part_01_threshold_summary.json](exps/red-dot/m_part_01_threshold_summary.json)
- [m_part_01_trial_scores.csv](exps/red-dot/m_part_01_trial_scores.csv)
- [reddot_0.3_to_0.4.csv](exps/red-dot/reddot_0.3_to_0.4.csv)
- [reddot_0.3_to_0.4.json](exps/red-dot/reddot_0.3_to_0.4.json)
- [graph_0.3_to_0.4.png](exps/red-dot/graph_0.3_to_0.4.png)

### Step 11: Interpret the results

The RedDots Part 01 sweep indicates that the most balanced operating region lies near thresholds 0.37 to 0.38.

At threshold 0.37:

- FAR ≈ 0.02648
- FRR ≈ 0.02406
- Accuracy ≈ 0.97352

At the EER point:

- threshold ≈ 0.376
- EER ≈ 0.02535

These results confirm that RedDots requires its own threshold calibration and should not reuse the VoxCeleb threshold directly.

### Step 12: Interpret the protocol counts and decision outputs

The file [m_0.37_probs.csv](exps/red-dot/m_0.37_probs.csv) contains the fixed protocol counts and the threshold-dependent decision statistics for threshold $0.37$. The category counts themselves do not change with threshold because they are defined by the RedDots protocol. What changes is the accept/reject outcome for each trial.

The four RedDots categories used in this evaluation are:

- `target-correct`: same speaker and correct phrase
- `target-wrong`: same speaker and wrong phrase
- `imposter-correct`: different speaker and correct phrase
- `imposter-wrong`: different speaker and wrong phrase

Let $N$ be the total number of trials, and let $N_c$ be the count for category $c$. Then the protocol probability of a category is:

$$
P(c) = \frac{N_c}{N}
$$

At threshold $\tau = 0.37$, a trial is accepted if:

$$
\mathrm{accept\ if\ score} \geq \tau
$$

and rejected otherwise:

$$
\mathrm{reject\ if\ score} < \tau
$$

For each category $c$, the accept rate is:

$$
\operatorname{AcceptRate}(c) = \frac{\mathrm{accepts}_c}{N_c}
$$

Using [m_0.37_probs.csv](exps/red-dot/m_0.37_probs.csv), the observed accept counts are:

- `target-correct`: $3,164$
- `target-wrong`: $24,822$
- `imposter-correct`: $1,445$
- `imposter-wrong`: $6,309$

The corresponding false accepts are all accepted non-genuine trials:

$$
\mathrm{FA} = \mathrm{accepts}_{target-wrong} + \mathrm{accepts}_{imposter-correct} + \mathrm{accepts}_{imposter-wrong}
$$

and the false rejects are the genuine trials that were rejected:

$$
\mathrm{FR} = N_{target-correct} - \mathrm{accepts}_{target-correct}
$$

Therefore, the standard speaker-verification metrics are:

$$
\mathrm{FAR} = \frac{\mathrm{FA}}{N_{target-wrong} + N_{imposter-correct} + N_{imposter-wrong}}
$$

$$
\mathrm{FRR} = \frac{\mathrm{FR}}{N_{target-correct}}
$$

For the RedDots Part 01 run at $\tau = 0.37$:

$$
\mathrm{FA} = 24,822 + 1,445 + 6,309 = 32,576
$$

$$
\mathrm{FR} = 3,242 - 3,164 = 78
$$

$$
\mathrm{FAR} = \frac{32,576}{1,230,038} \approx 0.02648
$$

$$
\mathrm{FRR} = \frac{78}{3,242} \approx 0.02406
$$

The balanced behavior at $\tau = 0.37$ is also reflected in the EER region, where FAR and FRR are nearly equal. This makes the threshold around $0.37$ a reasonable operating point for this protocol.

The interpretation of the category accept rates is as follows:

- `target-correct` accept rate $\approx 0.975941$ means the system correctly accepts most genuine trials.
- `target-wrong` accept rate $\approx 0.850709$ shows that the ECAPA-TDNN embedding is strongly speaker-focused and does not fully enforce phrase correctness by itself.
- `imposter-correct` accept rate $\approx 0.012033$ and `imposter-wrong` accept rate $\approx 0.005837$ are the false-accept regions that drive FAR.

This is the central interpretation of the RedDots evaluation: the model is effective for speaker identity verification, but phrase discrimination remains limited unless a phrase-aware component is added.

## 4.1 VoxCeleb (Text-Independent)

- Evaluation list: VoxCeleb1 `veri_test2` style protocol.
- Saved evaluation artifacts indicate:
  - total pairs: 37,611
  - total utterances: 4,708
- Key files:
  - [exps/threshold_summary.json](exps/threshold_summary.json)
  - [exps/threshold_results.csv](exps/threshold_results.csv)
  - [exps/final_eval/final_eval_summary.csv](exps/final_eval/final_eval_summary.csv)

## 4.2 RedDots (Text-Dependent, Part 01 Male)

From [reddots/readme.txt](reddots/readme.txt):

- Corpus includes mobile-collected speech, with TD and TI protocols.
- Current run uses Part 01 (common pass-phrases, text-dependent), male trial list.
- For this protocol:
  - target-correct: 3,242
  - target-wrong: 29,178
  - impostor-correct: 120,086
  - impostor-wrong: 1,080,774

Saved run artifacts indicate:

- total pairs: 1,233,280
- total utterances: 4,814

Key files:

- [exps/red-dot/m_part_01_threshold_summary.json](exps/red-dot/m_part_01_threshold_summary.json)
- [exps/red-dot/m_part_01_threshold_sweep.csv](exps/red-dot/m_part_01_threshold_sweep.csv)
- [exps/red-dot/reddot_0.3_to_0.4.csv](exps/red-dot/reddot_0.3_to_0.4.csv)
- [exps/red-dot/reddot_0.3_to_0.4.json](exps/red-dot/reddot_0.3_to_0.4.json)

## 5. Experimental Setup Used in This Workspace

- Pretrained checkpoint: [exps/pretrain.model](exps/pretrain.model)
- RedDots evaluation script: [reddots_threshold_sweep.py](reddots_threshold_sweep.py)
- Plotting utility: [plot_reddots_threshold_curves.py](plot_reddots_threshold_curves.py)
- Threshold sweep ranges used:
  - Vox TI sweep (saved): 0.1 to 1.0, step 0.1.
  - RedDots full sweep: -1.0 to 1.0, step 0.001.
  - RedDots focused sweep: 0.3 to 0.4, step 0.01.

Embedding caching was used (cache hit true in summaries), reducing rerun cost.

## 6. VoxCeleb Results (Text-Independent)

Primary summary from [exps/threshold_summary.json](exps/threshold_summary.json):

- EER: 0.012440852783242065 (1.244%)
- EER threshold: 0.30000000000000004
- FAR at EER threshold: 0.012440852783242065
- FRR at EER threshold: 0.00797787469418147
- minDCF: 0.0717364039735267
- minDCF threshold: 0.3825359642505646
- best accuracy: 0.9897902209459999
- best accuracy threshold: 0.30000000000000004

Selected threshold rows from [exps/threshold_results.csv](exps/threshold_results.csv):

| Threshold | FAR | FRR | Accuracy |
|---|---:|---:|---:|
| 0.2 | 0.064544 | 0.000638 | 0.967403 |
| 0.3 | 0.012441 | 0.007978 | 0.989790 |
| 0.4 | 0.001223 | 0.056111 | 0.971338 |
| 0.5 | 0.000000 | 0.222157 | 0.888942 |

Interpretation:

- Threshold 0.3 is a balanced and strong TI operating point for this run.
- Increasing threshold from 0.3 to 0.4 reduces FAR but increases FRR sharply.

## 7. RedDots Results (Text-Dependent)

### 7.1 Full sweep (-1.0 to 1.0, step 0.001)

From [exps/red-dot/m_part_01_threshold_summary.json](exps/red-dot/m_part_01_threshold_summary.json):

- EER: 0.02535206229401043 (2.535%)
- EER threshold: 0.3760000000000012
- FAR at EER threshold: 0.02535206229401043
- FRR at EER threshold: 0.025293028994447873
- minDCF: 0.3739010594897955
- minDCF threshold: 0.4777112901210785
- best accuracy: 0.9975601647638817
- best accuracy threshold: 0.7540000000000016

Important caution:

- The trial set is highly imbalanced (3,242 genuine vs 1,230,038 non-genuine).
- Therefore, very high accuracy can occur even with poor FRR, so accuracy alone is not reliable for threshold selection.

### 7.2 Focused sweep (0.3 to 0.4, step 0.01)

From [exps/red-dot/reddot_0.3_to_0.4.csv](exps/red-dot/reddot_0.3_to_0.4.csv) and [exps/red-dot/reddot_0.3_to_0.4.json](exps/red-dot/reddot_0.3_to_0.4.json):

| Threshold | FAR | FRR | Accuracy |
|---|---:|---:|---:|
| 0.30 | 0.052927 | 0.008020 | 0.947191 |
| 0.31 | 0.047161 | 0.009870 | 0.952937 |
| 0.32 | 0.042191 | 0.011413 | 0.957890 |
| 0.33 | 0.037854 | 0.012955 | 0.962211 |
| 0.34 | 0.034229 | 0.016656 | 0.965817 |
| 0.35 | 0.031130 | 0.017273 | 0.968906 |
| 0.36 | 0.028682 | 0.020975 | 0.971338 |
| 0.37 | 0.026484 | 0.024059 | 0.973523 |
| 0.38 | 0.024599 | 0.026527 | 0.975396 |
| 0.39 | 0.022964 | 0.031154 | 0.977014 |
| 0.40 | 0.021585 | 0.036397 | 0.978376 |

Interpretation:

- Around 0.37 to 0.38 is the balanced region (near EER crossing).
- 0.31 keeps FRR low but FAR remains relatively high.

## 8. Graphs and Visual Evidence

### 8.1 VoxCeleb/TI figures

- FAR/FRR curve: ![Vox FAR-FRR](exps/eval_gpu/far_frr_vs_threshold_updated.png)
- Score distributions: ![Vox score distributions](exps/eval_gpu/score_distributions_updated.png)
- Final-eval FAR/FRR figure: ![Vox final eval FAR-FRR](exps/final_eval/veri_test2_far_frr_curve.png)
- Final-eval score distributions: ![Vox final eval scores](exps/final_eval/veri_test2_score_distributions.png)

### 8.2 RedDots/TD figures

- Full-range FAR/FRR: ![RedDots full FAR-FRR](exps/red-dot/m_part_01_far_frr_vs_threshold.png)
- Focused 0.3-0.4 FAR/FRR: ![RedDots 0.3 to 0.4 FAR-FRR](exps/red-dot/graph_0.3_to_0.4.png)

## 9. Text-Dependent vs Text-Independent: Key Differences

| Aspect | VoxCeleb (TI) | RedDots m_part_01 (TD) |
|---|---|---|
| Task style | Text-independent | Text-dependent pass-phrase |
| Pairs evaluated | 37,611 | 1,233,280 |
| EER | 1.244% | 2.535% |
| EER threshold | 0.30 | 0.376 |
| minDCF threshold | 0.3825 | 0.4777 |
| Best-accuracy threshold | 0.30 | 0.754 |
| Class balance | Near-balanced in effect | Highly imbalanced to non-genuine |

Observations:

1. TD data here shows a different threshold landscape; direct transfer of TI threshold is suboptimal.
2. RedDots best-accuracy threshold is too high for balanced verification, due to imbalance effects.
3. EER-based thresholds provide a fair cross-system comparison point.

## 10. Recommended Threshold Policy

### 10.1 For text-independent (VoxCeleb-like)

- Recommended default threshold: 0.30 (from this run).
- If security is prioritized over convenience: move toward minDCF threshold (~0.3825).
- If user convenience is prioritized (lower FRR): stay near 0.30.

### 10.2 For text-dependent (RedDots m_part_01)

- Balanced operating point: 0.376 (EER threshold).
- Cost-sensitive point: 0.4777 (minDCF threshold).
- Low-reject setting: around 0.31, but with higher FAR.

### 10.3 Practical calibration guidance

- Always calibrate threshold per dataset and protocol.
- Do not compare operating points across datasets using only accuracy.
- Report FAR/FRR with threshold alongside EER and minDCF.

## 11. Reproducibility

### 11.1 RedDots focused sweep and graph

```bash
wsl -d Ubuntu-22.04 -- bash -lc "source /home/harini/miniconda3/etc/profile.d/conda.sh ; conda activate ecapa ; cd /mnt/c/Users/Harini/Documents/GitHub/ECAPA-TDNN-1 ; python reddots_threshold_sweep.py --initial_model exps/pretrain.model --ndx reddots/ndx/m_part_01.ndx --trn reddots/ndx/m_part_01.trn --pcm_root reddots/pcm --output_dir exps/red-dot --output_csv exps/red-dot/reddot_0.3_to_0.4.csv --summary_json exps/red-dot/reddot_0.3_to_0.4.json --threshold_start 0.3 --threshold_end 0.4 --threshold_step 0.01"
```

```bash
C:/Users/Harini/Documents/GitHub/ECAPA-TDNN-1/.venv/Scripts/python.exe plot_reddots_threshold_curves.py --csv exps/red-dot/reddot_0.3_to_0.4.csv --output exps/red-dot/graph_0.3_to_0.4.png
```

### 11.2 Vox evaluation artifacts

Artifacts already available in:

- [exps/threshold_summary.json](exps/threshold_summary.json)
- [exps/threshold_results.csv](exps/threshold_results.csv)
- [exps/final_eval/final_eval_summary.csv](exps/final_eval/final_eval_summary.csv)

## 12. Quality Notes and Caveats

1. In [exps/final_eval/final_eval_summary.csv](exps/final_eval/final_eval_summary.csv), `EER_exact` appears numerically inconsistent with other EER values (`EER_approx` is consistent). For rigorous reporting, rely on JSON/CSV sweep files where EER aligns with FAR/FRR crossing.
2. RedDots results in this report are specifically for `m_part_01` protocol; other parts (female, part_02/03/04) should be evaluated separately.
3. TI and TD calibration should remain separated in deployment.

## 13. Final Conclusion

This study demonstrates that a single ECAPA-TDNN backbone can serve both TI and TD speaker verification, but threshold calibration must be protocol-specific.

- TI (VoxCeleb-style): threshold near 0.30 is strong in current results.
- TD (RedDots m_part_01): threshold near 0.376 is balanced; 0.4777 is cost-oriented; 0.31 favors low FRR but increases FAR.

For research submission quality, report all of the following together:

1. EER and EER threshold.
2. minDCF and minDCF threshold.
3. FAR/FRR at chosen operating threshold.
4. Dataset protocol and trial balance details.
5. Curve plots (FAR/FRR and score distributions).

This combination provides a defensible, transparent, and reproducible evaluation narrative for both text-independent and text-dependent settings.

## 14. Comparison: Speaker-only vs Phrase-aware (RedDots m_part_01)

The baseline (speaker-only) run and phrase-aware (phrase-gate) run are compared below.

| Metric | Speaker-only | Phrase-aware |
|---|---:|---:|
| EER | 0.02535206 (2.535%) | 0.00617054 (0.617%) |
| EER threshold | 0.376 | 0.285 |
| Best accuracy | 0.99756016 (99.756%) | 0.99968377 (99.968%) |
| Best-accuracy threshold | 0.754 | 0.461 |
| minDCF | 0.37390106 | 0.37390106 |
| phrase-match trials | (not applicable) | 123,328 |
| phrase-mismatch trials | (not applicable) | 1,109,952 |

Summary: phrase-aware gating reduced EER from ~2.54% to ~0.62% (≈75.7% relative reduction) and slightly improved peak accuracy; minDCF remained unchanged under the current scoring/cost computation.

## 15. Additional Focused Phrase-aware Threshold Analysis

To refine threshold selection around the phrase-aware EER region, two additional focused sweeps were executed using `--script_txt` and `--phrase_gate`.

### 15.1 Focused phrase-aware sweep (0.25 to 0.40, step 0.01)

Artifacts:

- [exps/red-dot/reddot_0.25_to_0.4_phrase.csv](exps/red-dot/reddot_0.25_to_0.4_phrase.csv)
- [exps/red-dot/reddot_0.25_to_0.4_phrase.json](exps/red-dot/reddot_0.25_to_0.4_phrase.json)

This sweep confirms that the phrase-aware balanced region remains near the low 0.28 to 0.30 range, while best accuracy within the scanned interval is achieved near the upper end of the interval.

### 15.2 High-resolution phrase-aware sweep (0.28 to 0.30, step 0.001)

Artifacts:

- [exps/red-dot/reddot_0.28_to_0.30_phrase.csv](exps/red-dot/reddot_0.28_to_0.30_phrase.csv)
- [exps/red-dot/reddot_0.28_to_0.30_phrase.json](exps/red-dot/reddot_0.28_to_0.30_phrase.json)
- [exps/red-dot/phrase_0.28_to_0.30.png](exps/red-dot/phrase_0.28_to_0.30.png)

This high-resolution window provides a detailed view of FAR/FRR behavior around the phrase-aware operating region and supports precise threshold picking around the EER neighborhood.

Recommended phrase-aware operating threshold: use approximately `0.285` for balanced verification (near EER), and consider `0.300` when prioritizing maximum accuracy in the scanned 0.28 to 0.30 window.

| Operating intent | Suggested threshold |
|---|---:|
| Balanced FAR/FRR (near EER) | 0.285 |
| Maximum accuracy in 0.28-0.30 scan | 0.300 |

### 15.3 Comparison plot used for reporting

The combined comparison curve generated for the focused runs is saved at:

- [exps/red-dot/compare_0.25_to_0.40.png](exps/red-dot/compare_0.25_to_0.40.png)

## 16. Final Fixed-Threshold Evaluation (0.285)

After selecting `0.285` as the final phrase-aware operating threshold, the RedDots evaluation was executed with `--script_txt` and `--phrase_gate` on the available test protocol in this workspace (`m_part_01`).

Generated artifacts:

- [exps/red-dot/reddot_fixed_0.285_phrase.csv](exps/red-dot/reddot_fixed_0.285_phrase.csv)
- [exps/red-dot/reddot_fixed_0.285_phrase.json](exps/red-dot/reddot_fixed_0.285_phrase.json)
- [exps/red-dot/m_0.285_phrase_threshold.csv](exps/red-dot/m_0.285_phrase_threshold.csv)

Observed fixed-threshold results at `0.285`:

- total trials: `1,233,280`
- genuine trials: `3,242`
- non-genuine trials: `1,230,038`
- true accepts: `3,222`
- false rejects: `20`
- false accepts: `7,590`
- true rejects: `1,222,448`
- accuracy: `0.993829`

Derived rates at the final threshold:

- FAR = `7590 / 1230038 ≈ 0.00617` (about `0.617%`)
- FRR = `20 / 3242 ≈ 0.00617` (about `0.617%`)

This confirms that threshold `0.285` gives a balanced and strong phrase-aware operating point on the RedDots protocol currently present in this workspace.

## 17. Verification Stats Cross-Check

These are the final cross-checked numbers from the source JSON/CSV artifacts, included here so the report can be read as a self-contained research summary.

### VoxCeleb (`exps/threshold_summary.json`)

- EER: `0.0124408528` (`1.244%`)
- EER threshold: `0.3000000000`
- minDCF: `0.0717364040`
- minDCF threshold: `0.3825359643`
- Best accuracy: `0.9897902209` at `0.3000000000`

### RedDots baseline speaker-only (`exps/red-dot/m_part_01_threshold_summary.json`)

- EER: `0.0253520623` (`2.535%`)
- EER threshold: `0.3760000000`
- minDCF: `0.3739010595`
- minDCF threshold: `0.4777112901`
- Best accuracy: `0.9975601648` at `0.7540000000`
- Genuine / non-genuine: `3242 / 1230038`

### RedDots phrase-aware full sweep (`exps/red-dot/m_part_01_phrase_gate_summary.json`)

- EER: `0.0061705411` (`0.617%`)
- EER threshold: `0.2850000000`
- minDCF: `0.3739010595`
- minDCF threshold: `0.4777112901`
- Best accuracy: `0.9996837701` at `0.4610000000`
- Phrase match / mismatch: `123328 / 1109952`

### Focused phrase-aware 0.25 to 0.40 (`exps/red-dot/reddot_0.25_to_0.4_phrase.json`)

- EER: `0.0061690315`
- EER threshold: `0.2900000000`
- Best accuracy: `0.9993497016` at `0.4000000000`

### High-resolution phrase-aware 0.28 to 0.30 (`exps/red-dot/reddot_0.28_to_0.30_phrase.json`)

- EER: `0.0061705411`
- EER threshold: `0.2850000000`
- Best accuracy within window: `0.9952143876` at `0.3000000000`

### Final fixed-threshold run 0.285 (`exps/red-dot/reddot_fixed_0.285_phrase.json`)

- Best accuracy at fixed threshold: `0.9938294629`
- Threshold start / end: `0.285 / 0.285`
- FAR at threshold: `0.0061705411`
- FRR at threshold: `0.0061690315`

### Decision CSV recomputation (`exps/red-dot/m_0.285_phrase_threshold.csv`)

- Total: `1,233,280`
- Genuine: `3,242`
- Non-genuine: `1,230,038`
- True accepts: `3,222`
- False rejects: `20`
- False accepts: `7,590`
- True rejects: `1,222,448`
- Accuracy: `0.993829463`
- FAR: `0.006170541`
- FRR: `0.006169031`

These values match the earlier sections of `final_report.md` and confirm that the final threshold `0.285` is internally consistent across the summary JSON files, the focused sweeps, and the final decision CSV.
