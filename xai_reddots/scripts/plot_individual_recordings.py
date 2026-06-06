"""
plot_individual_recordings.py
==============================
Generates one XAI final_figure for EACH individual recording of a given speaker+phrase.

ROOT CAUSE FIX:
  The MFA timeline was built by averaging 6 recordings with different lead-in silence.
  Simple duration-scaling can't fix this — boundaries shift uniformly but the SPEECH
  START varies per recording. 

SOLUTION: 
  1. Detect the true speech_start and speech_end in each individual recording
     using RMS energy thresholding.
  2. Detect the MFA timeline's speech span (first phoneme start → last phoneme end).
  3. Linearly remap every phoneme boundary from [mfa_speech_start, mfa_speech_end]
     → [this_recording_speech_start, this_recording_speech_end].
  This perfectly anchors the phoneme grid to the actual acoustic events in each file.
"""

import os, sys, json, csv, copy
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())

from xai_reddots.scripts.batch_pipeline import PHRASES, pcm_duration
from xai_reddots.scripts.final_figure import (
    load_model, extract_attention, rms_envelope, mel_spec, build_figure,
    N_MELS, HOP, WIN, NFFT, SR
)

# ─────────────────────────────────────────────────────────────
# CONFIGURATION  ← change these two lines for other phrases
# ─────────────────────────────────────────────────────────────
SPEAKER   = 'm0004'
PHRASE_KEY = 'my_voice_is_my_password'
PHRASE_IDX = 1     # folder name prefix (phrase1_, phrase2_, …)

PHRASE_DEF  = next(p for p in PHRASES if p['key'] == PHRASE_KEY)
CSV_META    = f'xai_reddots/metadata/separated_phrases/{PHRASE_KEY}.csv'
MFA_TIMELINE = f'xai_reddots/metadata/{SPEAKER}_{PHRASE_KEY}_timeline.json'
OUTPUT_DIR  = f'xai_reddots/results/phrase{PHRASE_IDX}_{PHRASE_KEY}/plots_individual'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# SPEECH ONSET / OFFSET DETECTION
# ─────────────────────────────────────────────────────────────
def detect_speech_span(audio, sr=SR, hop=HOP,
                        noise_percentile=15,
                        onset_multiplier=8.0,
                        offset_multiplier=3.0,
                        min_silence_frames=5,
                        artifact_skip_sec=0.30):
    """
    Return (speech_start_sec, speech_end_sec) using a robust two-threshold VAD.

    Algorithm:
      1. Skip the first `artifact_skip_sec` (0.30s) to avoid DC/transient spikes.
      2. Apply a 5-frame median filter to the RMS to suppress remaining spikes.
      3. Estimate the noise floor as the Nth percentile of the ENTIRE smoothed
         signal AFTER the skip window (so even if speech starts early, the
         percentile of a mix of speech+silence gives a stable noise estimate).
      4. ONSET threshold  = noise_floor × onset_multiplier  (strict — avoids FP)
      5. OFFSET threshold = noise_floor × offset_multiplier (loose — catches trailing
         quiet consonants like /d/ at end of "password")
      6. Onset  = first frame (after skip) that crosses onset_threshold AND is
         preceded by min_silence_frames quiet frames.
      7. Offset = last frame that crosses offset_threshold.
    """
    from scipy.ndimage import median_filter

    rms = rms_envelope(audio)
    rms_smooth = median_filter(rms.astype(float), size=5)

    skip_frames = int(artifact_skip_sec * sr / hop)

    # ── Noise floor: low percentile of post-skip signal ───────────────
    post_skip = rms_smooth[skip_frames:]
    if len(post_skip) == 0:
        return 0.0, len(audio) / sr

    noise_floor = np.percentile(post_skip, noise_percentile)

    # Thresholds
    onset_thresh  = noise_floor * onset_multiplier
    offset_thresh = noise_floor * offset_multiplier

    # Safety cap: onset must be at least 4% of signal peak
    onset_thresh = max(onset_thresh, np.max(rms_smooth) * 0.04)

    # ── Find ONSET (after skip window) ───────────────────────────────
    search = rms_smooth[skip_frames:]
    onset_idxs = np.where(search > onset_thresh)[0]

    onset_frame = None
    if len(onset_idxs) > 0:
        for fi in onset_idxs:
            abs_fi = fi + skip_frames
            sil_start = max(0, abs_fi - min_silence_frames)
            if np.all(rms_smooth[sil_start:abs_fi] <= onset_thresh):
                onset_frame = abs_fi
                break
        if onset_frame is None:
            onset_frame = onset_idxs[0] + skip_frames
    else:
        # Fallback: no onset found — start at beginning
        onset_frame = skip_frames

    # ── Find OFFSET (full signal, loose threshold) ────────────────────
    all_speech = np.where(rms_smooth > offset_thresh)[0]
    offset_frame = all_speech[-1] if len(all_speech) > 0 else onset_frame

    onset_sec  = onset_frame  * hop / sr
    offset_sec = (offset_frame + 1) * hop / sr

    return onset_sec, offset_sec


# ─────────────────────────────────────────────────────────────
# TIMELINE REMAPPING
# ─────────────────────────────────────────────────────────────
def remap_timeline(mfa_tdata, rec_speech_start, rec_speech_end):
    """
    Take the MFA timeline and remap every phoneme boundary so that
    the speech span [mfa_speech_start, mfa_speech_end] maps to
    [rec_speech_start, rec_speech_end].

    This is a linear (affine) transformation — no warping needed.
    """
    tdata = copy.deepcopy(mfa_tdata)

    phonemes = tdata['phonemes']
    if not phonemes:
        return tdata

    mfa_speech_start = phonemes[0]['start']
    mfa_speech_end   = phonemes[-1]['end']
    mfa_span = mfa_speech_end - mfa_speech_start
    rec_span = rec_speech_end  - rec_speech_start

    if mfa_span < 1e-6:
        return tdata

    def remap(t_mfa):
        """Linear remap from MFA time to recording time."""
        frac = (t_mfa - mfa_speech_start) / mfa_span
        return rec_speech_start + frac * rec_span

    for ph in phonemes:
        ph['start']      = remap(ph['start'])
        ph['end']        = remap(ph['end'])
        ph['word_start'] = remap(ph['word_start'])
        ph['word_end']   = remap(ph['word_end'])

    return tdata


# ─────────────────────────────────────────────────────────────
# SINGLE-RECORDING AGGREGATOR
# ─────────────────────────────────────────────────────────────
def aggregate_single(pcm_path, model, tdata):
    """
    Runs exactly like final_figure.aggregate() but for one file only.
    The tdata passed in already has remapped phoneme boundaries.
    """
    import soundfile as sf
    audio, _ = sf.read(pcm_path, channels=1, samplerate=SR,
                        subtype='PCM_16', format='RAW')
    if len(audio) < 400:
        audio = np.pad(audio, (0, 400 - len(audio)))

    dur  = len(audio) / SR
    avg_dur = float(tdata.get('avg_duration_sec', dur))

    prof, _ = extract_attention(model, pcm_path)
    P       = len(prof)
    prof_t  = np.linspace(0, avg_dur, P)

    # --- Resample everything onto prof_t ---
    prof_resampled = np.interp(prof_t, np.linspace(0, dur, len(prof)), prof)

    r = rms_envelope(audio)
    rms_resampled = np.interp(prof_t, np.linspace(0, dur, len(r)), r)

    an = audio / (np.max(np.abs(audio)) + 1e-9)
    mP = mel_spec(an)
    bx = np.linspace(0, dur, mP.shape[1])
    mel_resampled = np.zeros((N_MELS, P))
    for fi in range(N_MELS):
        mel_resampled[fi] = np.interp(np.linspace(0, avg_dur, P), bx, mP[fi])

    all_profs = prof_resampled[np.newaxis, :]   # shape (1, P)
    mean_prof = prof_resampled
    std_prof  = np.zeros_like(mean_prof)
    mean_rms  = rms_resampled
    rms_norm  = mean_rms / (np.max(mean_rms) + 1e-9)
    mel_db    = 10 * np.log10(mel_resampled + 1e-10)

    phonemes = tdata.get('phonemes', [])
    occ = {ph['phoneme']: sum(1 for p in phonemes if p['phoneme'] == ph['phoneme'])
           for ph in phonemes}

    ph_data = []
    for ph in phonemes:
        s, e = float(ph['start']), float(ph['end'])
        mask = (prof_t >= s) & (prof_t <= e)
        if not np.any(mask):
            ci = np.argmin(np.abs(prof_t - (s + e) / 2))
            mask = np.zeros(P, bool)
            mask[ci] = True
        mean_val = float(np.mean(mean_prof[mask]))
        rms_ph   = float(np.mean(rms_norm[mask]))
        ph_data.append({
            'phoneme':    ph['phoneme'],
            'word':       ph['word'],
            'start':      s,
            'end':        e,
            'start_frame': int(round(s * 100)),
            'end_frame':   int(round(e * 100)),
            'occurrence': occ[ph['phoneme']],
            'mean':       mean_val,
            'std':        0.0,
            'rms':        rms_ph,
        })

    total = sum(p['mean'] for p in ph_data) or 1e-12
    for rank_i, p in enumerate(sorted(ph_data, key=lambda x: x['mean'], reverse=True), 1):
        p['pct']  = p['mean'] / total * 100
        p['rank'] = rank_i
    rm = {(p['phoneme'], round(p['start'], 4)): (p['pct'], p['rank']) for p in ph_data}
    for p in ph_data:
        p['pct'], p['rank'] = rm[(p['phoneme'], round(p['start'], 4))]

    top3_pct  = sum(p['pct'] for p in sorted(ph_data, key=lambda x: x['rank'])[:3])
    corr      = float(np.corrcoef(mean_rms, mean_prof)[0, 1])
    peak_idx  = int(np.argmax(mean_prof))
    peak_val  = float(mean_prof[peak_idx])
    peak_time = float(prof_t[peak_idx])
    peak_frame = int(round(peak_time * 100))

    # Word boundaries
    word_bounds = []
    curr_word = None
    w_start, w_end = 0, 0
    for ph in phonemes:
        w = ph['word']
        if curr_word is None:
            curr_word = w; w_start = ph['start']; w_end = ph['end']
        elif w == curr_word:
            w_end = max(w_end, ph['end'])
        else:
            word_bounds.append((curr_word, [w_start, w_end]))
            curr_word = w; w_start = ph['start']; w_end = ph['end']
    if curr_word is not None:
        word_bounds.append((curr_word, [w_start, w_end]))

    return dict(
        avg_dur=avg_dur, prof_t=prof_t,
        mean_prof=mean_prof, std_prof=std_prof,
        mean_rms=mean_rms, rms_norm=rms_norm, mel_db=mel_db,
        phonemes=phonemes, ph_data=ph_data, total=total,
        word_bounds=word_bounds,
        top3_pct=top3_pct, corr=corr, n_recs=1,
        peak_val=peak_val, peak_time=peak_time, peak_frame=peak_frame,
        rep_audio=audio,
    )


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    # Load the MFA timeline once
    if not os.path.exists(MFA_TIMELINE):
        raise FileNotFoundError(
            f"MFA timeline not found: {MFA_TIMELINE}\n"
            "Please set MFA_TIMELINE to the correct path."
        )
    with open(MFA_TIMELINE) as f:
        mfa_tdata = json.load(f)

    print(f"MFA timeline loaded: {len(mfa_tdata['phonemes'])} phonemes, "
          f"avg_dur={mfa_tdata['avg_duration_sec']:.3f}s")
    print(f"  Speech span in MFA: "
          f"{mfa_tdata['phonemes'][0]['start']:.3f}s → "
          f"{mfa_tdata['phonemes'][-1]['end']:.3f}s")

    print("\nLoading ECAPA-TDNN model...")
    model = load_model()

    # Collect recordings
    recs = []
    with open(CSV_META) as f:
        for row in csv.DictReader(f):
            if row['speaker_id'] == SPEAKER and os.path.exists(row['pcm_path']):
                recs.append(row['pcm_path'])

    if not recs:
        print("No recordings found!")
        return

    print(f"\nFound {len(recs)} recordings for {SPEAKER} / {PHRASE_KEY}\n")

    import soundfile as sf

    for idx, pcm_path in enumerate(recs, 1):
        print(f"── Recording {idx}/{len(recs)}: {os.path.basename(pcm_path)}")

        # 1. Load audio to measure this specific recording
        audio, _ = sf.read(pcm_path, channels=1, samplerate=SR,
                            subtype='PCM_16', format='RAW')
        dur = len(audio) / SR
        print(f"   Duration: {dur:.3f}s")

        # 2. Detect speech onset/offset using energy thresholding
        speech_start, speech_end = detect_speech_span(audio, sr=SR, hop=HOP)
        print(f"   Speech span detected: {speech_start:.3f}s → {speech_end:.3f}s")

        # 3. Remap MFA boundaries to this recording's speech span
        tdata = remap_timeline(mfa_tdata, speech_start, speech_end)
        tdata['avg_duration_sec'] = dur   # set exact duration for plotting

        ph_starts = [ph['start'] for ph in tdata['phonemes']]
        ph_ends   = [ph['end']   for ph in tdata['phonemes']]
        print(f"   Remapped phonemes: {tdata['phonemes'][0]['phoneme']} at "
              f"{ph_starts[0]:.3f}s → {tdata['phonemes'][-1]['phoneme']} at "
              f"{ph_ends[-1]:.3f}s")

        # 4. Aggregate (extract attention + signals for this single file)
        d = aggregate_single(pcm_path, model, tdata)

        # 5. Build & save figure
        phrase_title = PHRASE_DEF['display'] + f" (Rec {idx}/{len(recs)})"
        fig = build_figure(SPEAKER, d, phrase_title)

        fig_path = os.path.join(OUTPUT_DIR,
                                f"{SPEAKER}_rec{idx:02d}_final_figure.png")
        fig.savefig(fig_path, dpi=300, bbox_inches='tight',
                    facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        print(f"   ✓ Saved: {fig_path}\n")

    print("All done!")


if __name__ == '__main__':
    main()
