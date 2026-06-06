# Entropy-Derived Attention Implementation Walkthrough

> [!NOTE]
> The proposed entropy-derived attention is a post-hoc explainability measure computed from pre-softmax ECAPA attention logits. It does not modify the original ECAPA-TDNN architecture or its pretrained weights. This document details the extraction and visualization of this normalized temporal attention distribution ($\hat{\alpha}_j$) that sums to $1.0$ across the audio clip.

---

## 1. Mathematical Framework

The transition from raw confidence to normalized attention weights follows these three distinct cases:

### Case 1: Channel-Wise Probability Distribution
For each time frame $j$, we extract the pre-softmax attention logits $w$ (shape `[1536, T]`) from the bottleneck layer of the ECAPA-TDNN model. We apply a Softmax across the **channel dimension** ($dim=1$):

$$a_{ij} = \text{Softmax}(w_{ij}, \text{dim}=1)$$

Since there are $1536$ channels, the sum of all channel activations at frame $j$ is exactly $1.0$:

$$\sum_{i=1}^{1536} a_{ij} = 1.0$$

This provides a valid probability distribution over the speaker features at each frame.

### Case 2: Frame-Wise Entropy & Confidence
We compute the Shannon Entropy ($H_j$) for each frame $j$:

$$H_j = -\sum_{i=1}^{1536} a_{ij} \log(a_{ij})$$

We then normalize this entropy against the maximum possible entropy ($\log(1536)$) and invert it to get a local **Confidence Score** ($\alpha_j$):

$$\alpha_j = 1 - \frac{H_j}{\log(1536)}$$

*   **Low Entropy** $\rightarrow$ Model is focused on a few speaker-discriminative features (High Confidence $\alpha_j$).
*   **High Entropy** $\rightarrow$ Model is spread flatly across all features (Low Confidence $\alpha_j$).

These local confidence scores (e.g., $0.07$, $0.06$, $0.08$) are bounded in $[0.0, 1.0]$, but they **do not sum to 1** across time.

### Case 3: Normalized Entropy-Derived Attention ($\hat{\alpha}_j$)
To represent how the model distributes its attention across the frames of a given utterance, we normalize the confidence scores across the **time dimension**:

$$\hat{\alpha}_j = \frac{\alpha_j}{\sum_{k} \alpha_k}$$

This produces our final **Entropy-Derived Attention** weights, which sum to exactly $1.0$ across all frames:

$$\sum_{j} \hat{\alpha}_j = 1.0$$

---

## 2. Code Updates

### [extract_entropy.py](file:///c:/Users/Harini/Documents/ECAPA-TDNN-1/xai_reddots/scripts/extract_entropy.py)
We updated the extraction pass to compute and save the normalized attention profile ($\hat{\alpha}_j$) instead of raw confidence:

```python
# Step 1: Compute channel-wise distribution and entropy
a = torch.softmax(w_logits, dim=1)
H = -torch.sum(a * torch.log(a + 1e-9), dim=1)

# Step 2: Convert to confidence
C_channels = a.shape[1]
confidence = 1.0 - H / math.log(C_channels)

# Step 3: Normalize across frames to get Entropy-Derived Attention (alpha_hat)
alpha_hat = confidence / torch.sum(confidence, dim=1, keepdim=True)
entropy_profile = alpha_hat.squeeze().cpu().numpy()
```

### [generate_dashboard.py](file:///c:/Users/Harini/Documents/ECAPA-TDNN-1/xai_reddots/scripts/generate_dashboard.py)
We updated the dashboard script to:
1. Relabel all axis headers, legends, and titles to **"Attention Weight"** and **"Entropy-Derived Attention"**.
2. Use a bounding box constraint (`bbox=[0, 0, 1, 0.88]`) for the table layout to prevent overlapping with the plot titles.

---

## 3. Visual Dashboard Output

The pipeline has been executed end-to-end to generate updated, speaker-specific 6-row XAI dashboards:

*   **Row 1 & 2:** Word & Phoneme Timestamps (aligned to CMU phonemes).
*   **Row 3:** Magma Mel Spectrogram with phoneme boundaries.
*   **Row 4 & 5:** Normalized Entropy-Derived Attention Curve showing the relative focus at each frame.
*   **Row 6:** Phoneme Ranking Table sorted by mean attention weight.

The regenerated files are saved at:
👉 **[m0001_entropy_dashboard.png](file:///c:/Users/Harini/Documents/ECAPA-TDNN-1/xai_reddots/plots/m0001_entropy_dashboard.png)**
👉 **[m0002_entropy_dashboard.png](file:///c:/Users/Harini/Documents/ECAPA-TDNN-1/xai_reddots/plots/m0002_entropy_dashboard.png)**

---

## 4. Summary of Data Trace

For a single representative trial sequence:
*   **Attention Vector Shape:** `[1, 1536, T]`
*   **Normalized Attention Weights ($\hat{\alpha}_j$):** Sum to exactly `1.0` across the `T` frames.
*   **Scale Range:** Individual frame values range from `0.002` to `0.007`.

---

## 5. Multi-Speaker Comparison Insights (m0001 vs m0002)

Comparing the top-ranked phonemes between different speakers reveals that the ECAPA-TDNN model dynamically relies on different phonetic regions to classify different voices:

### Top Phoneme Rankings (by Mean Attention Weight)

| Rank | Speaker `m0001` (Top Focus) | Speaker `m0002` (Top Focus) |
| :--- | :--- | :--- |
| **1** | `/IH/` in "is" (`0.005400`) | `/S/` in "voice" (`0.006445`) |
| **2** | `/Z/` in "is" (`0.005200`) | `/OY/` in "voice" (`0.006320`) |
| **3** | `/S/` in "voice" (`0.005164`) | `/V/` in "voice" (`0.006282`) |
| **4** | `/AE/` in "password" (`0.005156`) | `/IH/` in "is" (`0.006280`) |
| **5** | `/W/` in "password" (`0.005056`) | `/M/` in "my" (`0.005900`) |

### Key Scientific Takeaway:
*   **Feature Concentration for `m0001`:** The entropy-derived attention assigns the highest frame importance to phoneme regions aligned with the word **"is"** (`/IH/` and `/Z/`).
*   **Feature Concentration for `m0002`:** The entropy-derived attention assigns the highest frame importance to phoneme regions aligned with the word **"voice"** (`/S/`, `/OY/`, `/V/`).
*   **Preliminary Comparison:** These results suggest speaker-specific differences in which acoustic regions exhibit the highest feature concentration (and therefore highest information selectivity). Proving systematic dynamic shifting of attention would require broader cohort testing and statistical significance verification.

---

## 6. Future Work: Single-Channel Temporal Attention

A proposed future research direction is to simplify the attention model by reducing the channel dimension in the attention layer from $1536$ down to $1$:

```python
self.attention = nn.Sequential(
    nn.Conv1d(4608, 256, kernel_size=1),
    nn.ReLU(),
    nn.BatchNorm1d(256),
    nn.Tanh(),
    nn.Conv1d(256, 1, kernel_size=1),  # Replaced 1536 with 1 channel
    nn.Softmax(dim=2),
)
```

### Implementation via Transfer Learning (Recommended)
Because this modification changes the attention pooling output dimension, loading the original pretrained weights directly will cause a shape mismatch error in PyTorch (`[1536, 256, 1]` vs. `[1, 256, 1]`). 

To address this without the overhead of training from scratch, we can utilize **Transfer Learning**:
1. Load the pretrained weights for all other ECAPA layers (Fbank, Conv1, SE-Res2 blocks, MFA layer, and FC layers).
2. Initialize the final `Conv1d(256, 1, kernel_size=1)` attention layer with random weights.
3. Fine-tune the network on the target dataset for a few epochs. This enables the model to adapt its feature classification to the new global pooling constraint with minimal training overhead.

### The Research Trade-Off: Interpretability vs. Performance

This modification introduces an important research balance between explainability and verification accuracy:

*   **Advantages:**
    *   **Direct native explainability:** Generates a single, direct temporal attention curve ($\alpha_t$) that represents frame-level importance out-of-the-box, eliminating the need for post-hoc channel entropy calculations.
    *   **Complexity reduction:** Removes approximately **394k parameters** from the pooling block.
*   **Risks:**
    *   **Lower speaker discrimination (EER):** Standard ECAPA uses 1536 channels because different speech features (like pitch, formants, and vocal timbre) may require different temporal focus. Forcing a single global attention curve ($\alpha_t$) across all features reduces the model's expressiveness.

### Strategic Conclusion:
*   **Current Approach (Safer):** Our current dashboard uses **post-hoc entropy analysis** on the existing pretrained model. This allows us to explain the model's decisions *without* performing any retraining or fine-tuning.
*   **Future Extension:** The single-channel temporal attention proposal serves as a valuable future research step to evaluate the performance impact of forcing native, single-channel explainability.
