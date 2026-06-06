"""
paper_figure.py
===============
Generates a clean 4-panel publication-quality figure for each speaker:

  Panel 1 — RMS Energy Envelope + ECAPA Attention  (Amplitude-proxy vs Time)
  Panel 2 — Average Mel Spectrogram + ECAPA Attention  (Frequency × Time)
  Panel 3 — ECAPA Entropy Attention + Phoneme Boundaries  (mean ± std)
  Panel 4 — Phoneme Attention Ranking  (horizontal bar chart)

Output: xai_reddots/plots/<speaker_id>_my_voice_is_my_password_paper_figure.png
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
from matplotlib.colors import Normalize

sys.path.append(os.getcwd())
from ECAPAModel import ECAPAModel

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
PHRASE_CLEAN = "my_voice_is_my_password"
OUTPUT_DIR   = "xai_reddots/plots"
SR           = 16000
HOP          = 160      # 10 ms
WIN          = 400      # 25 ms
NFFT         = 512
N_MELS       = 80

# ─────────────────────────────────────────────────────────────
# DESIGN TOKENS  (publication-grade palette)
# ─────────────────────────────────────────────────────────────
BG          = '#ffffff'
PANEL_BG    = '#fafafa'
GRID_COL    = '#e8e8e8'
ATTN_COL    = '#d35400'      # burnt orange — stands out on grayscale spec
ATTN_FILL   = '#fae0d3'
RMS_COL     = '#1a6b3c'      # dark green
RMS_FILL    = '#d6ede2'
BAR_COL     = '#4a235a'      # deep purple
PHON_COL    = '#c0392b'      # red for phoneme lines
SPINE_COL   = '#cccccc'
LABEL_FS    = 10
TICK_FS     = 8.5
TITLE_FS    = 10.5
ANNOT_FS    = 8


# ─────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────
def load_model():
    args = type('Args', (), {})()
    args.C = 1024;  args.m = 0.2;  args.s = 30
    args.n_class = 5994;  args.lr = 0.001
    args.lr_decay = 0.97;  args.test_step = 1
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
                        torch.mean(x, 2, keepdim=True).repeat(1, 1, t),
                        torch.sqrt(torch.var(x, 2, keepdim=True)
                                   .clamp(min=1e-4)).repeat(1, 1, t)], 1)
        wl = gx
        for layer in model.speaker_encoder.attention[:-1]:
            wl = layer(wl)
        a    = torch.softmax(wl, dim=1)
        H    = -torch.sum(a * torch.log(a + 1e-9), dim=1)
        conf = 1.0 - H / math.log(a.shape[1])
        alpha = conf / torch.sum(conf, dim=1, keepdim=True)
    return alpha.squeeze().cpu().numpy(), audio


# ─────────────────────────────────────────────────────────────
# SIGNAL UTILITIES
# ─────────────────────────────────────────────────────────────
def rms_envelope(audio, hop=HOP, win=WIN):
    hw = win // 2
    n  = len(audio) // hop
    out = np.zeros(n)
    for i in range(n):
        c   = i * hop
        seg = audio[max(0, c - hw): min(len(audio), c + hw)]
        out[i] = np.sqrt(np.mean(seg ** 2)) if len(seg) else 0.0
    return out


def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)


def mel_filterbank(n_mels=N_MELS, n_fft=NFFT, sr=SR):
    low  = hz_to_mel(0)
    high = hz_to_mel(sr / 2)
    pts  = np.linspace(low, high, n_mels + 2)
    hz   = 700 * (10 ** (pts / 2595) - 1)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    fb   = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        for k in range(bins[m - 1], bins[m]):
            fb[m-1, k] = (k - bins[m-1]) / (bins[m] - bins[m-1] + 1e-9)
        for k in range(bins[m], bins[m + 1] + 1):
            fb[m-1, k] = (bins[m + 1] - k) / (bins[m + 1] - bins[m] + 1e-9)
    return fb


def mel_spectrogram(audio, n_mels=N_MELS, nfft=NFFT, hop=HOP, sr=SR):
    Pxx, _, bins = mlab.specgram(audio, NFFT=nfft, Fs=sr,
                                  noverlap=nfft - hop,
                                  window=mlab.window_hanning)
    return mel_filterbank(n_mels, nfft, sr) @ Pxx, bins


# ─────────────────────────────────────────────────────────────
# AGGREGATE DATA FOR ONE SPEAKER
# ─────────────────────────────────────────────────────────────
def aggregate(speaker_id, model, timeline_data):
    sep_csv = f'xai_reddots/metadata/separated_phrases/{PHRASE_CLEAN}.csv'
    recs = []
    with open(sep_csv) as f:
        for row in csv.DictReader(f):
            if row['speaker_id'] == speaker_id and os.path.exists(row['pcm_path']):
                recs.append(row['pcm_path'])
    if not recs:
        raise RuntimeError(f"No recordings for {speaker_id}")

    avg_dur    = float(timeline_data.get('avg_duration_sec', 3.0))
    n_audio    = int(SR * avg_dur)
    audio_t    = np.linspace(0, avg_dur, n_audio)

    # First pass: establish common profile length
    rep_prof, _ = extract_attention(model, recs[0])
    P           = len(rep_prof)
    prof_t      = np.linspace(0, avg_dur, P)

    all_profs = []
    all_rms   = []
    all_mel   = []

    for path in recs:
        try:
            prof, audio_i = extract_attention(model, path)
            dur_i = len(audio_i) / SR

            # Attention → common time axis
            px = np.linspace(0, dur_i, len(prof))
            all_profs.append(np.interp(prof_t, px, prof))

            # RMS envelope → common time axis
            r  = rms_envelope(audio_i)
            rx = np.linspace(0, dur_i, len(r))
            all_rms.append(np.interp(prof_t, rx, r))

            # Mel spectrogram → common P columns
            mel_Pxx, mel_bins = mel_spectrogram(audio_i / (np.max(np.abs(audio_i)) + 1e-9))
            bx     = np.linspace(0, dur_i, mel_Pxx.shape[1])
            tgt_bx = np.linspace(0, avg_dur, P)
            mel_i  = np.zeros((mel_Pxx.shape[0], P))
            for fi in range(mel_Pxx.shape[0]):
                mel_i[fi] = np.interp(tgt_bx, bx, mel_Pxx[fi])
            all_mel.append(mel_i)

        except Exception as e:
            print(f"  skip {path}: {e}")

    all_profs = np.vstack(all_profs)
    mean_prof = np.mean(all_profs, axis=0)
    std_prof  = np.std(all_profs, axis=0)
    mean_rms  = np.mean(all_rms, axis=0)
    mean_mel  = np.mean(all_mel, axis=0)
    mel_db    = 10 * np.log10(mean_mel + 1e-10)

    # ── Phoneme attention table ──────────────────────────────
    phonemes = timeline_data.get('phonemes', [])
    ph_data  = []

    # Phoneme occurrence count (how many segments share the same symbol)
    occ_count = {}
    for ph in phonemes:
        sym = ph['phoneme']
        occ_count[sym] = occ_count.get(sym, 0) + 1

    for ph in phonemes:
        s, e = float(ph['start']), float(ph['end'])
        mask = (prof_t >= s) & (prof_t <= e)
        if not np.any(mask):
            ci   = np.argmin(np.abs(prof_t - (s + e) / 2))
            mask = np.zeros(P, dtype=bool);  mask[ci] = True
        vals = [np.mean(all_profs[r, mask]) for r in range(all_profs.shape[0])]
        vals = np.array(vals)
        ph_data.append({
            'phoneme':    ph['phoneme'],
            'word':       ph['word'],
            'label':      f"{ph['word']}: /{ph['phoneme']}/",
            'start':      s,
            'end':        e,
            'start_frame': int(round(s * 100)),   # 10 ms/frame → frame = time × 100
            'end_frame':   int(round(e * 100)),
            'occurrence': occ_count[ph['phoneme']],
            'mean':       float(np.mean(vals)),
            'std':        float(np.std(vals)),
        })

    # ── Attention percentage (sum-normalised so all phonemes → 100%) ──
    total_attn = sum(p['mean'] for p in ph_data) or 1e-12
    for rank_i, p in enumerate(
            sorted(ph_data, key=lambda x: x['mean'], reverse=True), start=1):
        p['pct']  = p['mean'] / total_attn * 100
        p['rank'] = rank_i
    # propagate rank & pct back to original (unsorted) ph_data list
    rank_map = {p['label']: (p['pct'], p['rank']) for p in ph_data}
    for p in ph_data:
        p['pct'], p['rank'] = rank_map[p['label']]

    # Attention concentration = top-3 phoneme % / 100
    top3_pct = sum(p['pct'] for p in sorted(
        ph_data, key=lambda x: x['mean'], reverse=True)[:3])

    corr = float(np.corrcoef(mean_rms, mean_prof)[0, 1])

    return dict(
        n_recs      = len(recs),
        avg_dur     = avg_dur,
        prof_t      = prof_t,
        mean_prof   = mean_prof,
        std_prof    = std_prof,
        mean_rms    = mean_rms,
        mel_db      = mel_db,
        phonemes    = phonemes,
        ph_data     = ph_data,
        corr        = corr,
        top3_pct    = top3_pct,
        total_attn  = total_attn,
    )


# ─────────────────────────────────────────────────────────────
# PLOT HELPERS
# ─────────────────────────────────────────────────────────────
def style_ax(ax, title, xlabel, ylabel, xlim):
    ax.set_facecolor(PANEL_BG)
    ax.grid(True, ls='--', color=GRID_COL, alpha=0.8, linewidth=0.7)
    ax.set_xlim(0, xlim)
    ax.set_xlabel(xlabel, fontsize=LABEL_FS)
    ax.set_ylabel(ylabel, fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS)
    ax.set_title(title, fontsize=TITLE_FS, fontweight='bold', loc='left', pad=6)
    for sp in ax.spines.values():
        sp.set_color(SPINE_COL)
        sp.set_linewidth(0.8)


def add_phoneme_lines(ax, phonemes, ypos, fontsize=7.5):
    """Dashed red lines at each phoneme boundary + label at centre."""
    if not phonemes:
        return
    ax.axvline(phonemes[0]['start'], color=PHON_COL, lw=0.8, ls='--', alpha=0.55)
    for ph in phonemes:
        ax.axvline(ph['end'], color=PHON_COL, lw=0.8, ls='--', alpha=0.55)
        cx = (ph['start'] + ph['end']) / 2
        ax.text(cx, ypos, ph['phoneme'], color=PHON_COL,
                ha='center', va='top', fontsize=fontsize,
                fontweight='bold', alpha=0.8)


def twin_attention(ax, prof_t, mean_prof, std_prof, label='ECAPA Attention'):
    ax2 = ax.twinx()
    ax2.plot(prof_t, mean_prof, color=ATTN_COL, lw=2.0, label=label, zorder=5)
    ax2.fill_between(prof_t,
                     np.maximum(0, mean_prof - std_prof),
                     mean_prof + std_prof,
                     color=ATTN_FILL, alpha=0.40, zorder=4)
    ax2.set_ylabel("Attention weight", color=ATTN_COL, fontsize=LABEL_FS)
    ax2.tick_params(axis='y', labelcolor=ATTN_COL, labelsize=TICK_FS)
    ax2.set_ylim(0, np.max(mean_prof + std_prof) * 1.4)
    for sp in ['top', 'left', 'bottom']:
        ax2.spines[sp].set_visible(False)
    ax2.spines['right'].set_color(ATTN_COL)
    ax2.legend(loc='upper right', fontsize=ANNOT_FS, framealpha=0.7,
               handlelength=1.5, borderpad=0.4)
    return ax2


# ─────────────────────────────────────────────────────────────
# BUILD FIGURE
# ─────────────────────────────────────────────────────────────
def build_figure(speaker_id, d, phrase_display):
    """
    d  — aggregated data dict from aggregate()
    Returns figure.
    """
    fig = plt.figure(figsize=(14, 22), facecolor=BG)
    fig.suptitle(
        f"ECAPA-TDNN Speaker Explainability  •  Speaker {speaker_id}\n"
        f"Phrase: \"{phrase_display}\"  |  n = {d['n_recs']} recordings",
        fontsize=13, fontweight='bold', y=0.995, color='#1a1a1a'
    )

    gs = GridSpec(4, 1, figure=fig,
                  height_ratios=[2.0, 2.8, 2.8, 4.0],
                  hspace=0.52)

    prof_t    = d['prof_t']
    mean_prof = d['mean_prof']
    std_prof  = d['std_prof']
    avg_dur   = d['avg_dur']
    phonemes  = d['phonemes']

    # ── Panel 1: RMS Energy + Attention ──────────────────────
    ax1 = fig.add_subplot(gs[0])
    rms_norm = d['mean_rms'] / (np.max(d['mean_rms']) + 1e-9)
    ax1.plot(prof_t, rms_norm, color=RMS_COL, lw=1.8,
             label='RMS Energy (norm.)', zorder=5)
    ax1.fill_between(prof_t, 0, rms_norm, color=RMS_FILL, alpha=0.45, zorder=3)
    style_ax(ax1,
             f"(a)  RMS Energy Envelope + ECAPA Attention  [r = {d['corr']:+.3f}]",
             "Time (s)", "RMS Energy (norm.)", avg_dur)
    ax1.set_ylim(0, 1.5)
    add_phoneme_lines(ax1, phonemes, 1.38, fontsize=7)
    ax1.legend(loc='upper left', fontsize=ANNOT_FS, framealpha=0.7,
               handlelength=1.5, borderpad=0.4)
    twin_attention(ax1, prof_t, mean_prof, std_prof)

    # ── Panel 2: Mel Spectrogram + Attention ──────────────────
    ax2 = fig.add_subplot(gs[1])
    mel_db  = d['mel_db']
    vmax    = np.percentile(mel_db, 99.5)
    vmin    = vmax - 45
    mel_hz  = 700 * (10 ** (np.linspace(hz_to_mel(0), hz_to_mel(SR / 2), N_MELS) / 2595) - 1)
    ax2.pcolormesh(prof_t, mel_hz, mel_db,
                   cmap='gray_r', vmin=vmin, vmax=vmax,
                   shading='nearest', rasterized=True)
    ax2.set_ylim(0, 8000)
    ax2.set_yticks([0, 2000, 4000, 6000, 8000])
    ax2.set_yticklabels(['0', '2k', '4k', '6k', '8k'], fontsize=TICK_FS)
    style_ax(ax2,
             "(b)  Averaged Mel Spectrogram + ECAPA Attention",
             "Time (s)", "Frequency (Hz)", avg_dur)
    add_phoneme_lines(ax2, phonemes, 7500, fontsize=7)
    twin_attention(ax2, prof_t, mean_prof, std_prof)

    # ── Panel 3: Attention + Phoneme Boundaries ───────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(prof_t, mean_prof, color=BAR_COL, lw=2.2,
             label='Mean attention', zorder=5)
    ax3.fill_between(prof_t,
                     np.maximum(0, mean_prof - std_prof),
                     mean_prof + std_prof,
                     color='#e0d0f0', alpha=0.50, label='±1 std', zorder=4)
    style_ax(ax3,
             "(c)  ECAPA Entropy Attention + Phoneme Boundaries",
             "Time (s)", "Attention weight", avg_dur)
    ylim3 = np.max(mean_prof + std_prof) * 1.35
    ax3.set_ylim(0, ylim3)
    add_phoneme_lines(ax3, phonemes, ylim3 * 0.96, fontsize=7.5)

    # Word-level shading bands
    word_bounds = {}
    for ph in phonemes:
        w = ph['word']
        if w not in word_bounds:
            word_bounds[w] = [ph['start'], ph['end']]
        else:
            word_bounds[w][0] = min(word_bounds[w][0], ph['start'])
            word_bounds[w][1] = max(word_bounds[w][1], ph['end'])
    wcolors = ['#e8f4fd', '#fef9e7', '#eafaf1', '#fdf2f8', '#f0f3ff']
    for wi, (w, (ws, we)) in enumerate(sorted(word_bounds.items(),
                                               key=lambda x: x[1][0])):
        ax3.axvspan(ws, we, alpha=0.18, color=wcolors[wi % len(wcolors)], zorder=2)
        ax3.text((ws + we) / 2, ylim3 * 0.88, w,
                 ha='center', fontsize=7.5, color='#555555',
                 fontweight='semibold', style='italic')

    ax3.legend(loc='upper right', fontsize=ANNOT_FS, framealpha=0.7,
               handlelength=1.5, borderpad=0.4)

    # Peak annotation
    pi     = np.argmax(mean_prof)
    peak_t = prof_t[pi]
    peak_v = mean_prof[pi]
    ax3.annotate(
        f"peak\n{peak_t:.2f}s",
        xy=(peak_t, peak_v),
        xytext=(min(peak_t + avg_dur * 0.07, avg_dur * 0.9), peak_v * 0.88),
        fontsize=ANNOT_FS, color=BAR_COL,
        arrowprops=dict(arrowstyle='->', color=BAR_COL, lw=1.2),
    )

    # ── Panel 4: Phoneme Attention % Ranking ─────────────────
    ax4 = fig.add_subplot(gs[3])
    ax4.set_facecolor(PANEL_BG)
    ax4.grid(True, ls='--', color=GRID_COL, alpha=0.8, linewidth=0.7, axis='x')
    for sp in ax4.spines.values():
        sp.set_color(SPINE_COL);  sp.set_linewidth(0.8)

    # Sort ascending for barh (lowest at bottom)
    ph_sorted = sorted(d['ph_data'], key=lambda x: x['pct'])
    labels  = [p['label']            for p in ph_sorted]
    pcts    = [p['pct']              for p in ph_sorted]
    pct_std = [p['std'] / d['total_attn'] * 100  for p in ph_sorted]  # std in % units
    ranks   = [p['rank']             for p in ph_sorted]
    occs    = [p['occurrence']       for p in ph_sorted]

    # Colour bars by quartile (based on pct)
    q75 = np.percentile(pcts, 75)
    med = np.median(pcts)
    bar_colors = []
    for pct in pcts:
        if pct >= q75:
            bar_colors.append('#6c3483')   # top quartile — deep purple
        elif pct >= med:
            bar_colors.append('#a569bd')   # above median
        else:
            bar_colors.append('#d7bde2')   # below median

    bars = ax4.barh(labels, pcts, xerr=pct_std,
                    color=bar_colors, alpha=0.90, edgecolor='#999999',
                    linewidth=0.5, height=0.65,
                    error_kw=dict(ecolor='#444444', lw=1.0, capsize=3))

    # Labels: "#rank  XX.X% ± Y.Y%  (×occurrence)"
    x_max_val = max(p + s for p, s in zip(pcts, pct_std))
    for bar, pct, std_pct, rank, occ in zip(bars, pcts, pct_std, ranks, occs):
        x_label = pct + std_pct + x_max_val * 0.012
        ax4.text(x_label, bar.get_y() + bar.get_height() / 2,
                 f"#{rank}  {pct:.1f}% ± {std_pct:.1f}%   ×{occ}",
                 va='center', ha='left', fontsize=7.5, color='#222222')
        # Rank badge inside bar
        if pct > x_max_val * 0.08:
            ax4.text(pct * 0.04, bar.get_y() + bar.get_height() / 2,
                     f"#{rank}",
                     va='center', ha='left', fontsize=6.5,
                     color='white', fontweight='bold')

    ax4.set_title(
        f"(d)  Phoneme Attention % Ranking  "
        f"[Top-3 concentration = {d['top3_pct']:.1f}%]",
        fontsize=TITLE_FS, fontweight='bold', loc='left', pad=6)
    ax4.set_xlabel("Attention share (%)", fontsize=LABEL_FS)
    ax4.set_xlim(0, x_max_val * 1.55)
    ax4.tick_params(labelsize=TICK_FS)

    # Legend
    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor='#6c3483', label='Top quartile'),
        Patch(facecolor='#a569bd', label='Above median'),
        Patch(facecolor='#d7bde2', label='Below median'),
    ]
    ax4.legend(handles=legend_els, loc='lower right', fontsize=ANNOT_FS,
               framealpha=0.75, handlelength=1.2, borderpad=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.995])
    return fig


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
def main():
    speaker_id = sys.argv[1] if len(sys.argv) > 1 else "m0004"

    # Load timeline
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

    print(f"Loading ECAPA-TDNN model…")
    model = load_model()
    print(f"Aggregating data for speaker {speaker_id} ({tdata.get('n_recordings', '?')} recordings)…")

    d = aggregate(speaker_id, model, tdata)

    # ── Save enriched CSV ─────────────────────────────────────
    csv_dir  = 'xai_reddots/csv'
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir,
                            f"{speaker_id}_{PHRASE_CLEAN}_xai_analysis.csv")
    fieldnames = ['rank', 'phoneme', 'word', 'occurrence',
                  'start', 'end', 'start_frame', 'end_frame',
                  'mean_attention', 'std_attention',
                  'attention_pct', 'attention_pct_std']
    total = d['total_attn']
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for p in sorted(d['ph_data'], key=lambda x: x['rank']):
            w.writerow({
                'rank':              p['rank'],
                'phoneme':           p['phoneme'],
                'word':              p['word'],
                'occurrence':        p['occurrence'],
                'start':             f"{p['start']:.4f}",
                'end':               f"{p['end']:.4f}",
                'start_frame':       p['start_frame'],
                'end_frame':         p['end_frame'],
                'mean_attention':    f"{p['mean']:.6f}",
                'std_attention':     f"{p['std']:.6f}",
                'attention_pct':     f"{p['pct']:.2f}",
                'attention_pct_std': f"{p['std'] / total * 100:.2f}",
            })
    print(f"   Enriched CSV → {csv_path}")

    # ── Console summary ───────────────────────────────────────
    top3 = sorted(d['ph_data'], key=lambda x: x['rank'])[:3]
    print(f"\n   Top-3 phonemes by ECAPA attention:")
    for p in top3:
        print(f"     #{p['rank']}  {p['phoneme']:>4s}  ({p['word']:>10s})"
              f"  frames {p['start_frame']:>4d}–{p['end_frame']:>4d}"
              f"  {p['pct']:.1f}%  (×{p['occurrence']})")
    print(f"   Top-3 concentration: {d['top3_pct']:.1f}%")

    phrase_display = tdata.get('phrase', 'My voice is my password')
    fig = build_figure(speaker_id, d, phrase_display)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR,
                       f"{speaker_id}_{PHRASE_CLEAN}_paper_figure.png")
    fig.savefig(out, dpi=300, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f"\n✓  Figure → {out}")
    print(f"   Pearson r (RMS vs Attention) = {d['corr']:+.4f}")
    print(f"   Recordings averaged: {d['n_recs']}")


if __name__ == '__main__':
    main()
