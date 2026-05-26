# RedDots Probability Report

This report converts the RedDots trial-count summary in `reddots/readme.txt` into probabilities.

The protocol fields `spk-sent ID` and `test utterance` are identifiers, not probabilistic categories. The probabilities below are computed from the aggregated trial outcomes in each part:

Probability = category count / total trials in that section

## Part 01: Common Pass-phrases

### Male

| Category | Count | Probability |
| --- | ---: | ---: |
| target-correct | 3,242 | 0.00262876 |
| target-wrong | 29,178 | 0.02365886 |
| imposter-correct | 120,086 | 0.09737124 |
| imposter-wrong | 1,080,774 | 0.87634114 |
| Total | 1,233,280 | 1.00000000 |

### Female

| Category | Count | Probability |
| --- | ---: | ---: |
| target-correct | 634 | 0.01253559 |
| target-wrong | 5,706 | 0.11282031 |
| imposter-correct | 4,438 | 0.08774913 |
| imposter-wrong | 39,798 | 0.78689497 |
| Total | 50,576 | 1.00000000 |

### Combined

| Category | Count | Probability |
| --- | ---: | ---: |
| target-correct | 3,876 | 0.00301903 |
| target-wrong | 34,884 | 0.02717127 |
| imposter-correct | 124,524 | 0.09699219 |
| imposter-wrong | 1,120,572 | 0.87281751 |
| Total | 1,283,856 | 1.00000000 |

## Part 02: Unique Pass-phrases

### Male

| Category | Count | Probability |
| --- | ---: | ---: |
| target-correct | 3,192 | 0.00267857 |
| target-wrong | 28,728 | 0.02410714 |
| imposter-correct | 0 | 0.00000000 |
| imposter-wrong | 1,159,760 | 0.97321429 |
| Total | 1,191,680 | 1.00000000 |

### Female

| Category | Count | Probability |
| --- | ---: | ---: |
| target-correct | 582 | 0.01367738 |
| target-wrong | 5,238 | 0.12309645 |
| imposter-correct | 0 | 0.00000000 |
| imposter-wrong | 36,732 | 0.86322617 |
| Total | 42,552 | 1.00000000 |

### Combined

| Category | Count | Probability |
| --- | ---: | ---: |
| target-correct | 3,774 | 0.00305777 |
| target-wrong | 33,966 | 0.02751995 |
| imposter-correct | 0 | 0.00000000 |
| imposter-wrong | 1,196,492 | 0.96942228 |
| Total | 1,234,232 | 1.00000000 |

## Part 03: Free-choice Pass-phrases

### Male

| Category | Count | Probability |
| --- | ---: | ---: |
| target-correct | 618 | 0.01315565 |
| target-wrong | 618 | 0.01315565 |
| imposter-correct | 0 | 0.00000000 |
| imposter-wrong | 45,740 | 0.97368869 |
| Total | 46,976 | 1.00000000 |

### Female

| Category | Count | Probability |
| --- | ---: | ---: |
| target-correct | 132 | 0.07142857 |
| target-wrong | 132 | 0.07142857 |
| imposter-correct | 0 | 0.00000000 |
| imposter-wrong | 1,584 | 0.85714286 |
| Total | 1,848 | 1.00000000 |

### Combined

| Category | Count | Probability |
| --- | ---: | ---: |
| target-correct | 750 | 0.01536130 |
| target-wrong | 750 | 0.01536130 |
| imposter-correct | 0 | 0.00000000 |
| imposter-wrong | 47,324 | 0.96927740 |
| Total | 48,824 | 1.00000000 |

## Part 04: Text-Prompted

### Male

| Category | Count | Probability |
| --- | ---: | ---: |
| target-correct | 5,696 | 0.00108793 |
| target-wrong | 131,002 | 0.02502116 |
| imposter-correct | 99,264 | 0.01895926 |
| imposter-wrong | 4,999,686 | 0.95493165 |
| Total | 5,235,648 | 1.00000000 |

### Female

| Category | Count | Probability |
| --- | ---: | ---: |
| target-correct | 1,122 | 0.00531009 |
| target-wrong | 25,806 | 0.12213199 |
| imposter-correct | 3,906 | 0.01848592 |
| imposter-wrong | 180,462 | 0.85407201 |
| Total | 211,296 | 1.00000000 |

### Combined

| Category | Count | Probability |
| --- | ---: | ---: |
| target-correct | 6,818 | 0.00125171 |
| target-wrong | 156,808 | 0.02878825 |
| imposter-correct | 103,170 | 0.01894090 |
| imposter-wrong | 5,180,148 | 0.95101914 |
| Total | 5,446,944 | 1.00000000 |

## Notes

- The counts come from the RedDots corpus summary in `reddots/readme.txt`.
- These probabilities describe the distribution of trial outcomes in each part, not speaker identity or utterance identity.
- `spk-sent ID` and `test utterance` remain categorical identifiers in the `.ndx` files.