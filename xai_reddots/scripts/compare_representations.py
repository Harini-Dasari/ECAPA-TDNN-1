"""
compare_representations.py
===========================
For a given speaker, this script generates a side-by-side 5-panel figure
showing how the SAME speech signal can be represented as:

  Panel 1 — Raw Waveform (Amplitude vs Time)
  Panel 2 — RMS Energy Envelope  (smoothed absolute amplitude vs Time)
             + ECAPA attention overlaid  → direct comparison
  Panel 3 — Spectrogram (Frequency vs Time)  [linear STFT]
             + ECAPA attention overlaid
  Panel 4 — Mel Spectrogram (Mel Frequency vs Time)
             + ECAPA attention overlaid
  Panel 5 — ECAPA attention alone (mean ± std across all recordings)

This lets you see:
  • WHY raw waveform ≠ attention (phase cancellation)
  • WHY RMS energy is a better proxy
  • HOW the spectrogram reveals frequency-time structure
  • WHERE the model's attention peaks relative to phonemes
"""

import os
import csv
import json
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.mlab as mlab
import soundfile as sf
import torch
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator

sys.path.append(os.getcwd())
from ECAPAModel import ECAPAModel


# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
PHRASE_CLEAN  = "my_voice_is_my_password"
OUTPUT_DIR    = "xai_reddots/trail_grphs"
SR            = 16000
HOP           = 160      # 10 ms hop
WIN           = 400      # 25 ms window
NFFT          = 512
N_MELS        = 80

# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────

def load_model():
    args = type('Args', (), {})()
    args.C = 1024
    args.m = 0.2
    args.s = 30
    args.n_class = 5994
    args.lr = 0.001
    args.lr_decay = 0.97
    args.test_step = 1
    model = ECAPAModel(**vars(args))
    model.load_parameters("exps/pretrain.model")
    model.speaker_encoder.eval()
    return model


def extract_attention(model, pcm_path):
    audio, _ = sf.read(pcm_path, channels=1, samplerate=SR,
                       subtype='PCM_16', format='RAW')
    if len(audio) < 400:
        audio = np.pad(audio, (0, 400 - len(audio)))
    data = torch.FloatTensor(np.stack([audio])).cuda()
    with torch.no_grad():
        x = model.speaker_encoder.torchfbank(data) + 1e-6
        x = x.log() - torch.mean(x.log(), dim=-1, keepdim=True)
        x = model.speaker_encoder.bn1(
            model.speaker_encoder.relu(model.speaker_encoder.conv1(x)))
        x1 = model.speaker_encoder.layer1(x)
        x2 = model.speaker_encoder.layer2(x + x1)
        x3 = model.speaker_encoder.layer3(x + x1 + x2)
        x  = model.speaker_encoder.relu(
            model.speaker_encoder.layer4(torch.cat([x1, x2, x3], dim=1)))
        t  = x.size(-1)
        gx = torch.cat([x,
                        torch.mean(x, 2, keepdim=True).repeat(1,1,t),
                        torch.sqrt(torch.var(x, 2, keepdim=True)
                                   .clamp(min=1e-4)).repeat(1,1,t)], 1)
        wl = gx
        for layer in model.speaker_encoder.attention[:-1]:
            wl = layer(wl)
        a  = torch.softmax(wl, dim=1)
        H  = -torch.sum(a * torch.log(a + 1e-9), dim=1)
        C  = a.shape[1]
        conf = 1.0 - H / math.log(C)
        alpha = conf / torch.sum(conf, dim=1, keepdim=True)
    return alpha.squeeze().cpu().numpy(), audio


def rms_envelope(audio, hop=HOP, win=WIN):
    """Frame-by-frame RMS energy."""
    n_frames = len(audio) // hop
    rms = np.zeros(n_frames)
    hw  = win // 2
    for i in range(n_frames):
        c   = i * hop
        seg = audio[max(0, c-hw): min(len(audio), c+hw)]
        rms[i] = np.sqrt(np.mean(seg**2)) if len(seg) else 0.0
    return rms


def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)


def mel_filterbank(n_mels=N_MELS, n_fft=NFFT, sr=SR):
    """Simple triangular mel filterbank."""
    low_mel  = hz_to_mel(0)
    high_mel = hz_to_mel(sr / 2)
    mel_pts  = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_pts   = 700 * (10 ** (mel_pts / 2595) - 1)
    bins_hz  = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        for k in range(bins_hz[m-1], bins_hz[m]):
            fb[m-1, k] = (k - bins_hz[m-1]) / (bins_hz[m] - bins_hz[m-1] + 1e-9)
        for k in range(bins_hz[m], bins_hz[m+1]+1):
            fb[m-1, k] = (bins_hz[m+1] - k) / (bins_hz[m+1] - bins_hz[m] + 1e-9)
    return fb


def compute_mel_spectrogram(audio, n_mels=N_MELS, nfft=NFFT, hop=HOP, sr=SR):
    Pxx, freqs, bins = mlab.specgram(audio, NFFT=nfft, Fs=sr,
                                     noverlap=nfft-hop, window=mlab.window_hanning)
    fb     = mel_filterbank(n_mels, nfft, sr)
    mel_Pxx = fb @ Pxx   # (n_mels, n_time)
    return mel_Pxx, freqs, bins


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def process_speaker(speaker_id, model, ax_list, fig_title, timeline_data):
    """Fill ax_list (5 axes) with the 5-panel comparison for one speaker."""
    ax_wave, ax_rms, ax_spec, ax_mel, ax_attn = ax_list

    # ── Load recordings ──────────────────────────────────────
    sep_csv = f'xai_reddots/metadata/separated_phrases/{PHRASE_CLEAN}.csv'
    recordings = []
    with open(sep_csv) as f:
        for row in csv.DictReader(f):
            if row['speaker_id'] == speaker_id and os.path.exists(row['pcm_path']):
                recordings.append(row['pcm_path'])
    if not recordings:
        print(f"  No recordings for {speaker_id}"); return

    avg_dur = float(timeline_data.get('avg_duration_sec', 3.0))
    n_audio = int(SR * avg_dur)
    audio_t = np.linspace(0, avg_dur, n_audio)

    # Representative profile (first recording)
    rep_prof, rep_audio = extract_attention(model, recordings[0])
    profile_len  = len(rep_prof)
    profile_time = np.linspace(0, avg_dur, profile_len)

    all_profs  = []
    all_rms    = []
    all_mel    = []
    all_audio  = []   # normalised raw waveforms interpolated to common grid

    for path in recordings:
        try:
            prof, audio_i = extract_attention(model, path)
            dur_i = len(audio_i) / SR

            # Attention
            px = np.linspace(0, dur_i, len(prof))
            all_profs.append(np.interp(profile_time, px, prof))

            # RMS envelope
            r = rms_envelope(audio_i)
            rx = np.linspace(0, dur_i, len(r))
            all_rms.append(np.interp(profile_time, rx, r))

            # Normalised audio (for waveform average)
            an = audio_i / (np.max(np.abs(audio_i)) + 1e-9)
            ax = np.linspace(0, dur_i, len(an))
            all_audio.append(np.interp(audio_t, ax, an))

            # Mel spectrogram — interpolate to profile_len time bins
            mel_Pxx, mel_f, mel_bins = compute_mel_spectrogram(audio_i)
            bx     = np.linspace(0, dur_i, mel_Pxx.shape[1])
            tgt_bx = np.linspace(0, avg_dur, profile_len)  # fixed target: profile_len cols
            mel_Pxx_interp = np.zeros((mel_Pxx.shape[0], profile_len))
            for fi in range(mel_Pxx.shape[0]):
                mel_Pxx_interp[fi] = np.interp(tgt_bx, bx, mel_Pxx[fi])
            all_mel.append(mel_Pxx_interp)

        except Exception as e:
            print(f"  Skipping {path}: {e}")

    all_profs = np.vstack(all_profs)
    mean_prof = np.mean(all_profs, axis=0)
    std_prof  = np.std(all_profs, axis=0)
    mean_rms  = np.mean(all_rms, axis=0)
    mean_audio = np.mean(all_audio, axis=0)
    mean_mel   = np.mean(all_mel, axis=0)
    mean_mel_db = 10 * np.log10(mean_mel + 1e-10)

    # Also build linear STFT spectrogram of representative recording
    rep_norm = rep_audio / (np.max(np.abs(rep_audio)) + 1e-9)
    Pxx_lin, freqs_lin, bins_lin = mlab.specgram(rep_norm, NFFT=NFFT, Fs=SR,
                                                  noverlap=NFFT-HOP,
                                                  window=mlab.window_hanning)
    Pxx_lin_db = 10 * np.log10(Pxx_lin + 1e-10)

    # Load phonemes from timeline
    phonemes = timeline_data.get('phonemes', [])

    # Pearson correlation
    corr_rms = float(np.corrcoef(mean_rms, mean_prof)[0, 1])

    # ─── Colour palette ───────────────────────────────────────
    ATTN_COL = '#e67e22'
    RMS_COL  = '#27ae60'
    STD_FILL = '#fde3c8'

    def draw_phoneme_lines(ax, ypos, fontsize=6.5):
        for ph in phonemes:
            ax.axvline(ph['end'],   color='red', lw=0.7, ls='--', alpha=0.5)
            cx = (ph['start'] + ph['end']) / 2
            ax.text(cx, ypos, ph['phoneme'], color='red', ha='center',
                    va='top', fontsize=fontsize, alpha=0.75, fontweight='bold')
        if phonemes:
            ax.axvline(phonemes[0]['start'], color='red', lw=0.7, ls='--', alpha=0.5)

    def twin_attention(ax, label='ECAPA Attention'):
        ax2 = ax.twinx()
        ax2.plot(profile_time, mean_prof, color=ATTN_COL, lw=1.8,
                 label=label, alpha=0.9)
        ax2.fill_between(profile_time,
                         np.maximum(0, mean_prof - std_prof),
                         mean_prof + std_prof,
                         color=STD_FILL, alpha=0.35)
        ax2.set_ylabel("Attention", color=ATTN_COL, fontsize=8)
        ax2.tick_params(axis='y', labelcolor=ATTN_COL, labelsize=7)
        ax2.set_ylim(0, np.max(mean_prof) * 1.35)
        ax2.spines['top'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.legend(loc='upper right', fontsize=7, framealpha=0.6)
        return ax2

    def base_style(ax, ylabel, title):
        ax.set_facecolor('white')
        ax.grid(True, ls='--', color='#e0e0e0', alpha=0.7)
        ax.set_xlim(0, avg_dur)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title(title, fontsize=9, fontweight='bold', loc='left', pad=4)
        for sp in ax.spines.values():
            sp.set_color('#cccccc')

    # ─── Panel 1: Raw Waveform ─────────────────────────────────
    ax_wave.plot(audio_t, mean_audio, color='#2980b9', lw=0.6, alpha=0.85)
    base_style(ax_wave, "Amplitude", f"① Raw Waveform  (averaged, n={len(all_profs)})")
    ax_wave.axhline(0, color='gray', lw=0.5, ls=':')
    draw_phoneme_lines(ax_wave, ax_wave.get_ylim()[1] if ax_wave.get_ylim()[1] != 0 else 0.9)
    note = ("⚠ Average waveform ≈ flat line because\n"
            "phase cancels across recordings.\n"
            "Use RMS envelope instead (→ Panel ②).")
    ax_wave.text(0.98, 0.97, note, transform=ax_wave.transAxes,
                 fontsize=6.5, va='top', ha='right',
                 color='#c0392b', style='italic',
                 bbox=dict(boxstyle='round,pad=0.3', fc='#fff5f5', ec='#e74c3c', alpha=0.85))

    # ─── Panel 2: RMS envelope + attention ────────────────────
    rms_norm = mean_rms / (np.max(mean_rms) + 1e-9)
    ax_rms.plot(profile_time, rms_norm, color=RMS_COL, lw=1.6,
                label='RMS Energy (normalised)', alpha=0.9)
    ax_rms.fill_between(profile_time, 0, rms_norm,
                        color='#d5f5e3', alpha=0.35)
    base_style(ax_rms, "RMS Energy (norm.)",
               f"② RMS Envelope + ECAPA Attention  |  r = {corr_rms:+.3f}")
    ax_rms.set_ylim(0, 1.45)
    draw_phoneme_lines(ax_rms, 1.35)
    ax_rms.legend(loc='upper left', fontsize=7, framealpha=0.6)
    twin_attention(ax_rms)

    # ─── Panel 3: Linear STFT Spectrogram + attention ─────────
    vmax3 = np.percentile(Pxx_lin_db, 99.5)
    vmin3 = vmax3 - 50
    ax_spec.pcolormesh(bins_lin, freqs_lin, Pxx_lin_db,
                       cmap='gray_r', vmin=vmin3, vmax=vmax3,
                       shading='nearest')
    ax_spec.set_ylim(0, 8000)
    ax_spec.set_yticks([0, 2000, 4000, 6000, 8000])
    ax_spec.set_yticklabels(['0 Hz', '2 kHz', '4 kHz', '6 kHz', '8 kHz'], fontsize=7)
    base_style(ax_spec, "Frequency (Hz)",
               "③ Linear Spectrogram (STFT) + ECAPA Attention")
    draw_phoneme_lines(ax_spec, 7600, fontsize=6)
    twin_attention(ax_spec)

    # ─── Panel 4: Mel Spectrogram + attention ─────────────────
    mel_t  = profile_time   # already profile_len columns, aligned to avg_dur
    mel_hz = 700 * (10 ** (np.linspace(
        hz_to_mel(0), hz_to_mel(SR/2), N_MELS) / 2595) - 1)
    vmax4 = np.percentile(mean_mel_db, 99.5)
    vmin4 = vmax4 - 45
    ax_mel.pcolormesh(mel_t, mel_hz, mean_mel_db,
                      cmap='gray_r', vmin=vmin4, vmax=vmax4,
                      shading='nearest')
    ax_mel.set_ylim(0, 8000)
    ax_mel.set_yticks([0, 2000, 4000, 6000, 8000])
    ax_mel.set_yticklabels(['0 Hz', '2 kHz', '4 kHz', '6 kHz', '8 kHz'], fontsize=7)
    base_style(ax_mel, "Mel Freq (→ Hz)",
               "④ Mel Spectrogram (averaged) + ECAPA Attention")
    draw_phoneme_lines(ax_mel, 7600, fontsize=6)
    twin_attention(ax_mel)

    # ─── Panel 5: Attention alone ─────────────────────────────
    ax_attn.plot(profile_time, mean_prof, color='#7030a0', lw=2.2,
                 label='Mean ECAPA Attention')
    ax_attn.fill_between(profile_time,
                         np.maximum(0, mean_prof - std_prof),
                         mean_prof + std_prof,
                         color='#eaddf5', alpha=0.45, label='±1 std')
    base_style(ax_attn, "Attention Weight",
               "⑤ ECAPA Entropy Attention (mean ± std)")
    ax_attn.set_ylim(0, np.max(mean_prof + std_prof) * 1.3)
    draw_phoneme_lines(ax_attn, np.max(mean_prof + std_prof) * 1.2)
    ax_attn.legend(loc='upper right', fontsize=7)
    # Highlight peak attention phoneme
    peak_t = profile_time[np.argmax(mean_prof)]
    peak_v = np.max(mean_prof)
    ax_attn.annotate(f"Peak\n{peak_t:.2f}s",
                     xy=(peak_t, peak_v),
                     xytext=(peak_t + avg_dur * 0.06, peak_v * 0.95),
                     fontsize=7, color='#7030a0',
                     arrowprops=dict(arrowstyle='->', color='#7030a0', lw=1.2))


def main():
    speaker_id = sys.argv[1] if len(sys.argv) > 1 else "m0004"

    # ── Load timeline ─────────────────────────────────────────
    for tpath in [
        f'xai_reddots/timelines/{speaker_id}_{PHRASE_CLEAN}_timeline.json',
        f'xai_reddots/metadata/{speaker_id}_{PHRASE_CLEAN}_timeline.json',
        'xai_reddots/metadata/timeline.json',
    ]:
        if os.path.exists(tpath):
            with open(tpath) as f:
                tdata = json.load(f)
            break
    else:
        print("Timeline not found"); sys.exit(1)

    print(f"Loading model…")
    model = load_model()
    print(f"Generating representation comparison for {speaker_id}…")

    # ── Figure layout ─────────────────────────────────────────
    BG = '#f8f9fa'
    fig = plt.figure(figsize=(16, 26), facecolor=BG)
    fig.suptitle(
        f"Speech Signal Representation Comparison\n"
        f"Speaker: {speaker_id}  •  Phrase: \"{tdata['phrase']}\"\n"
        f"Waveform  →  RMS Energy  →  Spectrogram  →  Mel Spectrogram  →  ECAPA Attention",
        fontsize=13, fontweight='bold', y=0.99
    )

    gs = GridSpec(5, 1, figure=fig, hspace=0.55)
    axes = [fig.add_subplot(gs[i]) for i in range(5)]
    for ax in axes:
        ax.set_facecolor(BG)

    process_speaker(speaker_id, model, axes,
                    f"Speaker {speaker_id}", tdata)

    # ── Save ─────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR,
                       f"{speaker_id}_{PHRASE_CLEAN}_representation_comparison.png")
    plt.savefig(out, dpi=200, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Saved → {out}")


if __name__ == '__main__':
    main()
