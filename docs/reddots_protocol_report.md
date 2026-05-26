# RedDots Protocol Report

This report explains the RedDots testing fields and how they should be interpreted when you evaluate a speaker verification system.

---

## 1. What the columns mean

For a RedDots trial row, the fields are:

- `spk-sent ID`
- `test utterance`
- `is target-correct`
- `is target-wrong`
- `is imposter-correct`
- `is imposter-wrong`

These are **not probabilities**. They are **trial categories / counts** that describe how the protocol is organized for evaluation.

---

## 2. Meaning of each field

### `spk-sent ID`
This is the enrollment identity for the trial. In RedDots it usually combines:
- the speaker ID
- the sentence / phrase ID

Example style:
- `m0001_31`
- `f0002_36`

It tells you which speaker-sentence enrollment model is being tested.

### `test utterance`
This is the utterance or recording used for evaluation against the enrollment model. It is the trial audio that will be scored by the speaker verification system.

### `is target-correct`
This counts **correct target trials**.
- The speaker is correct.
- The sentence / prompt is also correct.
- The system should ideally accept these.

### `is target-wrong`
This counts **wrong target trials**.
- The speaker may be the same, but the target condition is not correct.
- This is a mismatch inside the target class.
- The system should ideally reject these.

### `is imposter-correct`
This counts **correct impostor trials**.
- The trial is from a non-matching speaker or non-matching condition.
- The system should reject it.
- If it rejects correctly, it is counted here.

### `is imposter-wrong`
This counts **wrong impostor trials**.
- The trial is from a non-matching speaker / impostor.
- The system accepts it by mistake.
- This is a false accept.

---

## 3. Important correction: these are not probabilities

The values in those columns are not probabilities by default.
They are counts or trial categories from the corpus protocol.

If you want probabilities, you must compute them from the counts.

For a given set of trials:

$$
\text{total} = \text{target-correct} + \text{target-wrong} + \text{imposter-correct} + \text{imposter-wrong}
$$

Then the probability of each category is:

$$
P(\text{target-correct}) = \frac{\text{target-correct}}{\text{total}}
$$

$$
P(\text{target-wrong}) = \frac{\text{target-wrong}}{\text{total}}
$$

$$
P(\text{imposter-correct}) = \frac{\text{imposter-correct}}{\text{total}}
$$

$$
P(\text{imposter-wrong}) = \frac{\text{imposter-wrong}}{\text{total}}
$$

---

## 4. How testing works with these fields

For speaker verification testing, the workflow is:

1. Read the enrollment ID and test utterance.
2. Extract embeddings from the audio.
3. Compute a similarity score.
4. Compare the score with a threshold.
5. Accept or reject the test utterance.

Decision rule:

$$
\text{accept if score} \geq \text{threshold}
$$

$$
\text{reject if score} < \text{threshold}
$$

The RedDots protocol then tells you whether that decision was:
- a true accept
- a false accept
- a true reject
- a false reject

---

## 5. What each evaluation outcome means

### True accept
- The trial is genuine.
- The system accepts it.
- This is good.

### False accept
- The trial is an impostor.
- The system accepts it by mistake.
- This is bad.

### True reject
- The trial is an impostor.
- The system rejects it correctly.
- This is good.

### False reject
- The trial is genuine.
- The system rejects it by mistake.
- This is bad.

---

## 6. How this relates to FAR, FRR, and EER

From these trial outcomes, you compute the usual speaker verification metrics:

### FAR
False Acceptance Rate:

$$
FAR = \frac{\text{false accepts}}{\text{total impostor trials}}
$$

### FRR
False Rejection Rate:

$$
FRR = \frac{\text{false rejects}}{\text{total genuine trials}}
$$

### EER
Equal Error Rate is the point where FAR and FRR are approximately equal.

In practice, you sweep thresholds and find the threshold where the two error rates balance.

---

## 7. Can the same threshold 0.31 be reused?

### Short answer
- **As a baseline:** yes, you can test it.
- **As the final threshold:** no, not recommended.

### Why not?
The threshold `0.31` was tuned on the VoxCeleb-style evaluation you already ran.
RedDots is a different corpus and is text-dependent, so the score distribution can shift.

That means the best threshold on RedDots may be different.

### Best practice
- Use `0.31` only as a temporary baseline.
- Tune the threshold on RedDots development trials.
- Then use the tuned value for final RedDots reporting.

---

## 8. RedDots structure reminder

From the RedDots documentation, the corpus includes several parts:
- Part 01: Common Pass-phrases Text-Dependent
- Part 02: Unique Pass-phrases Text-Dependent
- Part 03: Free-choice Pass-phrases Text-Dependent
- Part 04: Text-Prompted

Some parts support text-dependent enrollment and some parts also allow text-independent enrollment in the prompted case.

This means RedDots should be evaluated with its own protocol parsing, not with the VoxCeleb `veri_test2.txt` style list.

---

## 9. Practical recommendation for your project

If you want to evaluate RedDots properly:

1. Read the RedDots `.trn` and `.ndx` files.
2. Parse the PCM audio correctly.
3. Extract embeddings with the ECAPA-TDNN encoder.
4. Score all RedDots trial pairs.
5. Sweep thresholds on RedDots dev data.
6. Pick the best threshold from RedDots itself.
7. Report final FAR, FRR, EER, and minDCF for RedDots.

---

## 10. Final conclusion

The RedDots columns describe **trial types and counts**, not probabilities.
They tell you how the evaluation protocol is structured and how to count correct and incorrect decisions.
For testing, you still need to score the audio and compare against a threshold.

The current threshold `0.31` is fine as a **starting baseline**, but for RedDots final evaluation it should be **re-tuned** using RedDots development trials.
