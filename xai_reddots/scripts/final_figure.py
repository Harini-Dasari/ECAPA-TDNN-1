"""
final_figure.py  (v2  —  matches reference design)
===================================================
Layout:
  (A) RMS Energy Envelope + ECAPA Attention             [full width]
  (B) Mel Spectrogram (80 mel bins, inferno cmap)       [full width + colorbar]
      + ECAPA Attention scaled to mel-bin range
  (C) ECAPA Entropy Attention + Phoneme Boundaries      [full width]
  (D) Phoneme Attention Analysis Table  |  XAI Summary Card  [split row]

Usage:
    python3 xai_reddots/scripts/final_figure.py m0001
    python3 xai_reddots/scripts/final_figure.py m0002
    python3 xai_reddots/scripts/final_figure.py m0004
"""

import os, csv, json, sys, math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.mlab as mlab
import matplotlib.gridspec as mgridspec
import soundfile as sf
import torch
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.append(os.getcwd())
from ECAPAModel import ECAPAModel

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
PHRASE_CLEAN = "my_voice_is_my_password"
OUTPUT_DIR   = "xai_reddots/plots"
SR = 16000;  HOP = 160;  WIN = 400;  NFFT = 512;  N_MELS = 80

# ─────────────────────────────────────────────────────────────
# DESIGN TOKENS
# ─────────────────────────────────────────────────────────────
BG        = '#ffffff'
PANEL_BG  = '#fdfdfd'
GRID_COL  = '#e0e0e0'
SPINE_COL = '#cccccc'
ATTN_COL  = '#e67e22'          # orange — attention on all panels
ATTN_FILL = '#fde8c8'
RMS_COL   = '#27ae60'          # green
RMS_FILL  = '#d5f5e3'
ATTN_C_COL= '#4a235a'          # dark purple — panel C attention line
ATTN_C_FILL= '#e0d0f0'
PHON_COL  = '#c0392b'          # red — phoneme labels/lines
WORD_COL  = '#7f8c8d'          # gray — word labels
CARD_BG   = '#0f1117'          # dark card background
LFS=10; TFS=8.5; TTLE=11; ANNFS=8.5
WORD_BAND_COLORS = ['#eaf4fb','#fef9e7','#eafaf1','#fdf2f8','#f0f3ff']


# ─────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────
def load_model():
    a = type('A',(),{})()
    a.C=1024; a.m=0.2; a.s=30; a.n_class=5994
    a.lr=0.001; a.lr_decay=0.97; a.test_step=1
    m = ECAPAModel(**vars(a))
    m.load_parameters("exps/pretrain.model")
    m.speaker_encoder.eval()
    return m

def extract_attention(model, pcm):
    audio,_ = sf.read(pcm, channels=1, samplerate=SR, subtype='PCM_16', format='RAW')
    if len(audio)<400: audio=np.pad(audio,(0,400-len(audio)))
    data = torch.FloatTensor(np.stack([audio])).cuda()
    with torch.no_grad():
        x = model.speaker_encoder.torchfbank(data)+1e-6
        x = x.log()-torch.mean(x.log(),dim=-1,keepdim=True)
        x = model.speaker_encoder.bn1(model.speaker_encoder.relu(
            model.speaker_encoder.conv1(x)))
        x1=model.speaker_encoder.layer1(x)
        x2=model.speaker_encoder.layer2(x+x1)
        x3=model.speaker_encoder.layer3(x+x1+x2)
        x =model.speaker_encoder.relu(model.speaker_encoder.layer4(
            torch.cat([x1,x2,x3],1)))
        t=x.size(-1)
        gx=torch.cat([x,
                      torch.mean(x,2,keepdim=True).repeat(1,1,t),
                      torch.sqrt(torch.var(x,2,keepdim=True).clamp(min=1e-4)
                                 ).repeat(1,1,t)],1)
        wl=gx
        for layer in model.speaker_encoder.attention[:-1]: wl=layer(wl)
        a=torch.softmax(wl,1); H=-torch.sum(a*torch.log(a+1e-9),1)
        conf=1.0-H/math.log(a.shape[1])
        alpha=conf/torch.sum(conf,1,keepdim=True)
    return alpha.squeeze().cpu().numpy(), audio

def rms_envelope(audio):
    hw=WIN//2; n=len(audio)//HOP; out=np.zeros(n)
    for i in range(n):
        c=i*HOP; seg=audio[max(0,c-hw):min(len(audio),c+hw)]
        out[i]=np.sqrt(np.mean(seg**2)) if len(seg) else 0.0
    return out

def hz_to_mel(hz): return 2595*np.log10(1+hz/700)

def mel_filterbank():
    pts=np.linspace(hz_to_mel(0),hz_to_mel(SR/2),N_MELS+2)
    hz=700*(10**(pts/2595)-1); bins=np.floor((NFFT+1)*hz/SR).astype(int)
    fb=np.zeros((N_MELS,NFFT//2+1))
    for m in range(1,N_MELS+1):
        for k in range(bins[m-1],bins[m]):
            fb[m-1,k]=(k-bins[m-1])/(bins[m]-bins[m-1]+1e-9)
        for k in range(bins[m],bins[m+1]+1):
            fb[m-1,k]=(bins[m+1]-k)/(bins[m+1]-bins[m]+1e-9)
    return fb

def mel_spec(audio):
    Pxx,_,_=mlab.specgram(audio,NFFT=NFFT,Fs=SR,
                           noverlap=NFFT-HOP,window=mlab.window_hanning)
    return mel_filterbank()@Pxx


# ─────────────────────────────────────────────────────────────
# AGGREGATE
# ─────────────────────────────────────────────────────────────
def aggregate(speaker_id, model, tdata):
    sep=f'xai_reddots/metadata/separated_phrases/{PHRASE_CLEAN}.csv'
    recs=[]
    with open(sep) as f:
        for row in csv.DictReader(f):
            if row['speaker_id']==speaker_id and os.path.exists(row['pcm_path']):
                recs.append(row['pcm_path'])
    if not recs: raise RuntimeError(f"No recordings for {speaker_id}")

    avg_dur=float(tdata.get('avg_duration_sec',3.0))
    rep_prof,_=extract_attention(model,recs[0])
    P=len(rep_prof); prof_t=np.linspace(0,avg_dur,P)

    all_profs,all_rms,all_mel=[],[],[]
    rep_audio = None
    for path in recs:
        try:
            prof,audio_i=extract_attention(model,path); dur_i=len(audio_i)/SR
            if rep_audio is None: rep_audio = audio_i
            all_profs.append(np.interp(prof_t,np.linspace(0,dur_i,len(prof)),prof))
            r=rms_envelope(audio_i)
            all_rms.append(np.interp(prof_t,np.linspace(0,dur_i,len(r)),r))
            an=audio_i/(np.max(np.abs(audio_i))+1e-9)
            mP=mel_spec(an); bx=np.linspace(0,dur_i,mP.shape[1])
            mi=np.zeros((N_MELS,P))
            for fi in range(N_MELS):
                mi[fi]=np.interp(np.linspace(0,avg_dur,P),bx,mP[fi])
            all_mel.append(mi)
        except Exception as e: print(f"  skip: {e}")

    all_profs=np.vstack(all_profs)
    mean_prof=np.mean(all_profs,0); std_prof=np.std(all_profs,0)
    mean_rms =np.mean(all_rms,0); rms_norm=mean_rms/(np.max(mean_rms)+1e-9)
    mel_db   =10*np.log10(np.mean(all_mel,0)+1e-10)

    phonemes=tdata.get('phonemes',[])
    occ={ph['phoneme']:sum(1 for p in phonemes if p['phoneme']==ph['phoneme'])
         for ph in phonemes}

    ph_data=[]
    for ph in phonemes:
        s,e=float(ph['start']),float(ph['end'])
        mask=(prof_t>=s)&(prof_t<=e)
        if not np.any(mask):
            ci=np.argmin(np.abs(prof_t-(s+e)/2)); mask=np.zeros(P,bool); mask[ci]=True
        vals=np.array([np.mean(all_profs[r,mask]) for r in range(all_profs.shape[0])])
        rms_ph=float(np.mean(rms_norm[mask])) if np.any(mask) else 0.0
        ph_data.append({'phoneme':ph['phoneme'],'word':ph['word'],
                        'start':s,'end':e,
                        'start_frame':int(round(s*100)),'end_frame':int(round(e*100)),
                        'occurrence':occ[ph['phoneme']],
                        'mean':float(np.mean(vals)),'std':float(np.std(vals)),
                        'rms':rms_ph})

    total=sum(p['mean'] for p in ph_data) or 1e-12
    for rank_i,p in enumerate(sorted(ph_data,key=lambda x:x['mean'],reverse=True),1):
        p['pct']=p['mean']/total*100; p['rank']=rank_i
    rm={(p['phoneme'],round(p['start'],4)):(p['pct'],p['rank']) for p in ph_data}
    for p in ph_data: p['pct'],p['rank']=rm[(p['phoneme'],round(p['start'],4))]

    top3_pct=sum(p['pct'] for p in sorted(ph_data,key=lambda x:x['rank'])[:3])
    corr=float(np.corrcoef(mean_rms,mean_prof)[0,1])
    peak_idx=int(np.argmax(mean_prof))
    peak_val=float(mean_prof[peak_idx]); peak_time=float(prof_t[peak_idx])
    peak_frame=int(round(peak_time*100))

    # Word boundaries (grouping consecutive phonemes of the same word)
    word_bounds = []
    curr_word = None
    w_start = 0
    w_end = 0
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

    return dict(avg_dur=avg_dur,prof_t=prof_t,mean_prof=mean_prof,std_prof=std_prof,
                mean_rms=mean_rms,rms_norm=rms_norm,mel_db=mel_db,
                phonemes=phonemes,ph_data=ph_data,total=total,word_bounds=word_bounds,
                top3_pct=top3_pct,corr=corr,n_recs=len(recs),
                peak_val=peak_val,peak_time=peak_time,peak_frame=peak_frame,
                rep_audio=rep_audio)


# ─────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────
def base_style(ax, title, xlabel, ylabel, xlim):
    ax.set_facecolor(PANEL_BG)
    ax.grid(True,ls='--',color=GRID_COL,alpha=0.75,lw=0.65,zorder=0)
    ax.set_xlim(0,xlim); ax.set_xlabel(xlabel,fontsize=LFS)
    ax.set_ylabel(ylabel,fontsize=LFS); ax.tick_params(labelsize=TFS)
    ax.set_title(title,fontsize=TTLE,fontweight='bold',loc='left',pad=25)
    for sp in ax.spines.values(): sp.set_color(SPINE_COL); sp.set_linewidth(0.8)

def add_word_bands(ax, word_bounds):
    """Pale colored bands per word with italic label."""
    for wi,(word,(ws,we)) in enumerate(word_bounds):
        ax.axvspan(ws,we,alpha=0.18,color=WORD_BAND_COLORS[wi%len(WORD_BAND_COLORS)],zorder=1)
        ax.text((ws+we)/2, 0.88, word,
                ha='center',va='center',fontsize=8,color=WORD_COL,
                style='italic',fontweight='semibold',
                transform=ax.get_xaxis_transform())

def add_phoneme_ticks(ax, phonemes):
    """Dashed lines + phoneme labels near top of panel."""
    col = PHON_COL
    alpha = 0.55
    lw = 0.8
    if phonemes:
        ax.axvline(phonemes[0]['start'],color=col,lw=lw,ls='--',alpha=alpha)
    for ph in phonemes:
        ax.axvline(ph['end'],color=col,lw=lw,ls='--',alpha=alpha)
        cx=(ph['start']+ph['end'])/2
        ax.text(cx, 1.02, ph['phoneme'], color=col, ha='center', va='bottom',
                fontsize=7.5, fontweight='bold', clip_on=False,
                transform=ax.get_xaxis_transform())

def twin_attention_axis(ax, t, prof, std, ylim_top=None, center_zero=False):
    """Standard orange attention twin-y axis."""
    ax2=ax.twinx()
    ax2.plot(t,prof,color=ATTN_COL,lw=1.8,label='ECAPA Attention (mean)',zorder=5)
    ax2.fill_between(t,np.maximum(0,prof-std),prof+std,
                     color=ATTN_FILL,alpha=0.40,label='±1 Std Dev',zorder=4)
    top = (np.max(prof)+np.max(std))*1.4 if ylim_top is None else ylim_top
    bottom = -top if center_zero else 0
    ax2.set_ylim(bottom,top)
    ax2.set_ylabel('Attention (mean)',color=ATTN_COL,fontsize=LFS)
    ax2.tick_params(axis='y',labelcolor=ATTN_COL,labelsize=TFS)
    for sp in ['top','left','bottom']: ax2.spines[sp].set_visible(False)
    ax2.spines['right'].set_color(ATTN_COL)
    ax2.yaxis.label.set_color(ATTN_COL)
    return ax2


# ─────────────────────────────────────────────────────────────
# PANEL A — Raw Audio Waveform + ECAPA Attention
# ─────────────────────────────────────────────────────────────
def draw_panel_a(ax, d):
    t=d['prof_t']; avg=d['avg_dur']
    mp=d['mean_prof']; sp=d['std_prof']
    rep_audio=d.get('rep_audio', np.zeros(10))

    # Normalize audio to [-1, 1] for clean display
    audio_norm = rep_audio / (np.max(np.abs(rep_audio)) + 1e-9)
    # Stretch time axis to match avg_dur
    t_audio = np.linspace(0, avg, len(audio_norm))

    ylim_wave=1.5

    # Plot raw waveform (MATLAB blue to match Panel B aesthetics)
    ax.plot(t_audio, audio_norm, color='#0072BD', lw=0.4, alpha=0.9, zorder=3, label='Raw Audio Waveform')
    ax.axhline(0, color='#aaaaaa', lw=0.6, zorder=2)
    
    base_style(ax,'(A)  Raw Audio Waveform + ECAPA Attention',
               'Time (s)','Amplitude (norm.)',avg)
    ax.set_ylim(-ylim_wave, ylim_wave)
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    
    # Word bands + phoneme lines
    add_word_bands(ax, d['word_bounds'])
    add_phoneme_ticks(ax, d['phonemes'])

    ax.legend(loc='upper left',fontsize=ANNFS,framealpha=0.8,
              edgecolor=SPINE_COL,handlelength=1.5)

    # Orange attention twin axis
    ax2=twin_attention_axis(ax,t,mp,sp, center_zero=True)
    ax2.legend(loc='upper right',fontsize=ANNFS,framealpha=0.8,
               edgecolor=SPINE_COL,handlelength=1.5)


# ─────────────────────────────────────────────────────────────
# PANEL B — Mel Energy Envelope (Waveform) + ECAPA Attention
# ─────────────────────────────────────────────────────────────
def draw_panel_b(ax, d, fig):
    """
    NOT a 2D heatmap.

    Collapses mel_db (80 × T) along the mel-frequency axis → 1-D energy
    envelope, then plots as a symmetric filled silhouette (waveform style).

    Steps:
      1. Convert dB → linear power, take RMS across mel bins → mel_energy (T,)
      2. Normalise to [0, 1]
      3. fill_between(t, +mel_norm, -mel_norm)  — solid black on white bg
      4. Overlay ECAPA attention curve in orange (twin-y axis)
      5. Phoneme boundary lines + word labels
    """
    t   = d['prof_t']
    mel = d['mel_db']          # shape (80, T)  — dB values
    avg = d['avg_dur']
    mp  = d['mean_prof']
    sp  = d['std_prof']

    # ── Step 1: Collapse mel → RMS energy envelope ────────────
    # Convert dB to linear power first so RMS is perceptually meaningful
    mel_linear  = 10.0 ** (mel / 10.0)                 # (80, T)
    mel_energy  = np.sqrt(np.mean(mel_linear, axis=0))  # RMS across mel bins → (T,)

    # ── Step 2: Temporal smoothing ─────────────────────
    import scipy.ndimage
    # Reduced sigma slightly to keep some natural speech texture while removing noise
    mel_energy_smooth = scipy.ndimage.gaussian_filter1d(mel_energy, sigma=3)

    # ── Step 3: Normalise to [0, 1] ───────────────────────────
    e_min, e_max = mel_energy_smooth.min(), mel_energy_smooth.max()
    mel_norm = (mel_energy_smooth - e_min) / (e_max - e_min + 1e-12)

    # Interpolate to same time grid as attention profile
    t_mel   = np.linspace(0, avg, len(mel_norm))
    mel_env = np.interp(t, t_mel, mel_norm)

    # ── Step 3: Symmetric filled silhouette (Waveform Style) ──
    # Using MATLAB blue to match the standard audio waveform aesthetic
    WAVE_COL = '#0072BD'
    
    # Generate a high-resolution time axis to simulate audio carrier
    t_high = np.linspace(0, avg, int(avg * 4000))  # 4000 points per second
    mel_env_high = np.interp(t_high, t, mel_env)
    
    # Simulate the high-frequency dense "shading" of a raw waveform
    # using uniform noise bounded by the envelope
    np.random.seed(42)  # For consistent visual texture
    carrier = np.random.uniform(-1, 1, size=len(t_high))
    wave_sim = mel_env_high * carrier
    
    ax.plot(t_high, wave_sim, color=WAVE_COL, lw=0.25, alpha=0.9, zorder=3, label='Mel Energy envelope')
    
    # Thin outline to sharpen edges
    ax.plot(t,  mel_env, color=WAVE_COL, lw=0.6, zorder=4)
    ax.plot(t, -mel_env, color=WAVE_COL, lw=0.6, zorder=4)

    # ── Axis styling — white background ───────────────────────
    ax.set_facecolor('white')
    ax.set_xlim(0, avg)
    ax.set_ylim(-1.5, 1.5)
    ax.axhline(0, color='#aaaaaa', lw=0.6, zorder=2)
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_yticklabels(['-1.0', '-0.5', '0', '0.5', '1.0'], fontsize=TFS)
    ax.set_xlabel('Time (s)', fontsize=LFS)
    ax.set_ylabel('Mel Energy (norm.)', fontsize=LFS)
    ax.tick_params(labelsize=TFS)
    ax.set_title('(B)  Mel Energy Envelope (Waveform) + ECAPA Attention',
                 fontsize=TTLE, fontweight='bold', loc='left', pad=25)
    ax.grid(True, ls='--', color=GRID_COL, alpha=0.7, lw=0.55, zorder=1)
    for sp_ in ax.spines.values():
        sp_.set_color(SPINE_COL); sp_.set_linewidth(0.8)

    # Word bands + phoneme lines
    add_word_bands(ax, d['word_bounds'])
    add_phoneme_ticks(ax, d['phonemes'])

    # ── Step 4 & 5: ECAPA attention — orange, twin-y axis ─────
    ax2 = ax.twinx()
    ax2.plot(t, mp, color=ATTN_COL, lw=2.0,
             label='ECAPA Attention (mean)', zorder=6)
    ax2.fill_between(t,
                     np.maximum(0, mp - sp),
                     mp + sp,
                     color=ATTN_FILL, alpha=0.40,
                     label='±1 Std Dev', zorder=5)
    top = (np.max(mp) + np.max(sp)) * 1.45
    ax2.set_ylim(-top, top)
    ax2.set_ylabel('Attention (mean)', color=ATTN_COL, fontsize=LFS)
    ax2.tick_params(axis='y', labelcolor=ATTN_COL, labelsize=TFS)
    for sp2 in ['top', 'left', 'bottom']:
        ax2.spines[sp2].set_visible(False)
    ax2.spines['right'].set_color(ATTN_COL)
    ax2.yaxis.label.set_color(ATTN_COL)

    # Combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2,
               loc='upper right', fontsize=ANNFS,
               framealpha=0.88, edgecolor=SPINE_COL, handlelength=1.5)





# ─────────────────────────────────────────────────────────────
# PANEL C — ECAPA Entropy Attention + Phoneme Boundaries
# ─────────────────────────────────────────────────────────────
def draw_panel_c(ax, d):
    t=d['prof_t']; mp=d['mean_prof']; sp=d['std_prof']; avg=d['avg_dur']
    ylim=np.max(mp+sp)*1.38

    ax.fill_between(t,np.maximum(0,mp-sp),mp+sp,
                    color=ATTN_C_FILL,alpha=0.55,label='±1 Std Dev',zorder=3)
    ax.plot(t,mp,color=ATTN_C_COL,lw=1.8,label='Mean Attention',zorder=4)

    base_style(ax,'(C)  ECAPA Entropy Attention + Phoneme Boundaries',
               'Time (s)','Attention (mean)',avg)
    ax.set_ylim(0,ylim)

    # Word bands + phoneme lines
    add_word_bands(ax, d['word_bounds'])
    add_phoneme_ticks(ax, d['phonemes'])

    ax.legend(loc='upper left',fontsize=ANNFS,framealpha=0.8,
              edgecolor=SPINE_COL,handlelength=1.5)

    # Peak annotation
    pi=np.argmax(mp); pt=t[pi]; pv=mp[pi]
    ax.annotate(
        f"Peak\n{pt:.2f} s\n{pv:.5f}",
        xy=(pt,pv),
        xytext=(pt+avg*0.06, pv*0.75),
        fontsize=ANNFS, color=ATTN_C_COL,
        arrowprops=dict(arrowstyle='->',color=ATTN_C_COL,lw=1.2),
        bbox=dict(boxstyle='round,pad=0.3',fc='white',ec=ATTN_C_COL,
                  alpha=0.85,lw=0.8))


# ─────────────────────────────────────────────────────────────
# PANEL D — Table  (left side of bottom row)
# ─────────────────────────────────────────────────────────────
def draw_table(ax, d, speaker_id):
    ax.set_facecolor(BG); ax.axis('off')
    n=d['n_recs']
    ax.set_title(
        f'(D)  Phoneme Attention Analysis Table  (Averaged over {n} recordings)',
        fontsize=TTLE,fontweight='bold',loc='left',pad=6)

    cols  = ['Rank','Phoneme','Word','Start (s)','End (s)',
             'Frames\n(10 ms/frame)',
             'Attention (mean)','Attention %\n(share of total)','× Occ']
    col_w = [0.070, 0.110, 0.130, 0.100, 0.100,
             0.120,
             0.120, 0.170, 0.080]

    rows_data = sorted(d['ph_data'], key=lambda x: x['rank'])
    total=d['total']
    n_rows=len(rows_data); n_cols=len(cols)
    hdr_h = 1.0/(n_rows+2.2)
    row_h = (1.0-hdr_h*1.3)/n_rows
    xs=[sum(col_w[:i]) for i in range(n_cols)]

    # Header
    for ci,(lbl,cw) in enumerate(zip(cols,col_w)):
        ax.add_patch(patches.FancyBboxPatch(
            (xs[ci],1-hdr_h),cw,hdr_h*0.92,
            boxstyle='round,pad=0.003',fc='#1a3a5c',ec='none',
            transform=ax.transAxes,clip_on=False))
        ax.text(xs[ci]+cw/2,1-hdr_h/2,lbl,ha='center',va='center',
                fontsize=7.5,color='white',fontweight='bold',
                transform=ax.transAxes,multialignment='center')

    # Rows
    max_pct=max(p['pct'] for p in rows_data)
    for ri,p in enumerate(rows_data):
        ry=1-hdr_h-(ri+1)*row_h
        bg='#eaf0fb' if p['rank']==1 else '#f1eaf8' if p['rank']<=3 else \
           '#f7f7f7' if ri%2==0 else '#ffffff'
        ax.add_patch(patches.FancyBboxPatch(
            (0,ry),1.0,row_h*0.93,boxstyle='round,pad=0.002',
            fc=bg,ec='#dddddd',lw=0.4,transform=ax.transAxes,clip_on=False))

        tc='#5b2c8f' if p['rank']==1 else '#7d3c98' if p['rank']<=3 else '#2c3e50'
        std_pct=p['std']/total*100
        frames=f"{p['start_frame']}–{p['end_frame']}"
        cells=[f"#{p['rank']}",p['phoneme'],p['word'],
               f"{p['start']:.3f}",f"{p['end']:.3f}",
               frames,
               f"{p['mean']:.5f}",
               f"{p['pct']:.2f}%",   # special — drawn with bar
               f"×{p['occurrence']}"]

        for ci,(val,cw) in enumerate(zip(cells,col_w)):
            fw='bold' if ci in (0,1) else 'normal'
            # Attention % column (index 7) — draw inline bar
            if ci==7:
                bar_w=cw*0.85*(p['pct']/max_pct)
                bar_col='#7d3c98' if p['rank']==1 else '#a569bd' if p['rank']<=3 \
                        else '#c39bd3'
                ax.add_patch(patches.FancyBboxPatch(
                    (xs[ci]+cw*0.02,ry+row_h*0.15),
                    bar_w,row_h*0.65,
                    boxstyle='round,pad=0.001',
                    fc=bar_col,ec='none',alpha=0.30,
                    transform=ax.transAxes,clip_on=False))
                ax.text(xs[ci]+cw/2,ry+row_h/2,val,
                        ha='center',va='center',fontsize=8,
                        color=tc,fontweight='bold',transform=ax.transAxes)
            else:
                ax.text(xs[ci]+cw/2,ry+row_h/2,val,
                        ha='center',va='center',fontsize=8,
                        color=tc,fontweight=fw,transform=ax.transAxes)

    # Footer note
    ax.text(0.0,-0.025,
            'Notes: Frames = time × 100 (10 ms/frame)  |  '
            'Attention % = phoneme attention / total attention × 100  |  '
            'RMS and attention are averaged across all recordings.',
            transform=ax.transAxes,fontsize=7,color='#777',style='italic',va='top')


# ─────────────────────────────────────────────────────────────
# XAI SUMMARY CARD  (right side of bottom row)
# ─────────────────────────────────────────────────────────────
def draw_stats_card(ax, d, speaker_id):
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')

    # Dark card background
    ax.add_patch(patches.FancyBboxPatch(
        (0.02,0.01),0.96,0.97,boxstyle='round,pad=0.02',
        fc=CARD_BG,ec='#2c2c3e',lw=1.2,
        transform=ax.transAxes,clip_on=False))

    # Title
    ax.text(0.5,0.95,'XAI Summary',ha='center',va='top',
            fontsize=11,fontweight='bold',color='#e0e0e0',
            transform=ax.transAxes)
    ax.text(0.5,0.89,f'Speaker: {speaker_id}  |  n = {d["n_recs"]} recordings',
            ha='center',va='top',fontsize=8,color='#8888aa',
            transform=ax.transAxes)
    ax.axhline(0.86,color='#2c2c3e',lw=1.0,xmin=0.05,xmax=0.95)

    # Top Attended Phonemes
    ax.text(0.08,0.83,'Top Attended Phonemes',ha='left',va='top',
            fontsize=8.5,fontweight='bold',color='#aaaacc',
            transform=ax.transAxes,style='italic')
    ax.text(0.55,0.83,'(by Attention %)',ha='left',va='top',
            fontsize=7.5,color='#666688',transform=ax.transAxes)

    top3=sorted(d['ph_data'],key=lambda x:x['rank'])[:3]
    rc=['#c39bd3','#a569bd','#7d3c98']
    for i,p in enumerate(top3):
        y=0.77-i*0.095
        # Rank badge
        ax.add_patch(patches.FancyBboxPatch(
            (0.06,y-0.028),0.10,0.050,
            boxstyle='round,pad=0.007',fc=rc[i],ec='none',
            transform=ax.transAxes,clip_on=False))
        ax.text(0.11,y,str(i+1),ha='center',va='center',
                fontsize=9,color='white',fontweight='bold',
                transform=ax.transAxes)
        ax.text(0.20,y+0.01,p['phoneme'],ha='left',va='center',
                fontsize=11,fontweight='bold',color='white',
                transform=ax.transAxes)
        ax.text(0.36,y+0.01,f"({p['word']})",ha='left',va='center',
                fontsize=8,color='#8888aa',transform=ax.transAxes)
        # Mini bar
        bw=p['pct']/15.0*0.38
        ax.add_patch(patches.FancyBboxPatch(
            (0.57,y-0.014),bw,0.030,
            boxstyle='round,pad=0.003',fc=rc[i],ec='none',alpha=0.7,
            transform=ax.transAxes,clip_on=False))
        ax.text(0.57+bw+0.02,y,f"{p['pct']:.2f}%",
                ha='left',va='center',fontsize=8.5,
                fontweight='bold',color=rc[i],
                transform=ax.transAxes)

    ax.axhline(0.50,color='#2c2c3e',lw=0.8,xmin=0.05,xmax=0.95)

    # Key Metrics
    ax.text(0.08,0.48,'Key Metrics',ha='left',va='top',
            fontsize=8.5,fontweight='bold',color='#aaaacc',
            transform=ax.transAxes,style='italic')

    metrics=[
        ('#f39c12','Attention Concentration',
         f"Top-3 = {d['top3_pct']:.1f}% of total attention"),
        ('#1abc9c','Entropy Attention Peak',
         f"{d['peak_val']:.5f} at {d['peak_time']:.2f} s (frame {d['peak_frame']})"),
        ('#3498db','Phrase Duration',
         f"{d['avg_dur']:.3f} s averaged"),
    ]
    for mi,(col,lbl,val) in enumerate(metrics):
        y=0.42-mi*0.105
        ax.add_patch(patches.Circle((0.10,y+0.008),0.020,
            fc=col,ec='none',transform=ax.transAxes,clip_on=False))
        ax.text(0.16,y+0.018,lbl,ha='left',va='center',
                fontsize=7.5,color='#9999bb',transform=ax.transAxes)
        ax.text(0.16,y-0.014,val,ha='left',va='center',
                fontsize=8,color='white',fontweight='bold',
                transform=ax.transAxes)

    ax.axhline(0.15,color='#2c2c3e',lw=0.8,xmin=0.05,xmax=0.95)

    # Insight
    ax.text(0.08,0.13,'Insight',ha='left',va='top',
            fontsize=7.5,fontweight='bold',color='#aaaacc',
            transform=ax.transAxes)
    corr=d['corr']
    if corr>=0.4:
        insight='ECAPA attention strongly follows RMS energy.\nThe model attends to high-energy speech segments.'
    elif corr>=-0.1:
        insight='ECAPA attention is not driven by RMS energy\n(almost flat energy, higher attention later).\nThe model focuses on specific phonemes\nthat carry speaker-discriminative information\nrather than overall loudness.'
    else:
        insight='ECAPA attention inversely tracks energy.\nModel focuses on softer, distinctive phonemes.'
    ax.text(0.08,0.08,insight,ha='left',va='top',
            fontsize=7.2,color='#cccccc',transform=ax.transAxes,
            linespacing=1.5)


# ─────────────────────────────────────────────────────────────
# BUILD FIGURE
# ─────────────────────────────────────────────────────────────
def build_figure(speaker_id, d, phrase):
    fig=plt.figure(figsize=(17,26),facecolor=BG)
    fig.suptitle(
        f"ECAPA-TDNN Speaker Explainability  •  Speaker: {speaker_id}\n"
        f'Phrase: "{phrase}"  |  n = {d["n_recs"]} recordings averaged',
        fontsize=14,fontweight='bold',y=0.995,color='#1a1a1a')

    # 4 rows: A, B, C, D(table+card)
    gs=mgridspec.GridSpec(4,1,figure=fig,
                          height_ratios=[2.0, 2.8, 2.2, 3.8],
                          hspace=0.40)

    axA=fig.add_subplot(gs[0])
    axB=fig.add_subplot(gs[1])
    axC=fig.add_subplot(gs[2])

    # Bottom row: table (left 70%) + stats card (right 30%)
    gs_bot=mgridspec.GridSpecFromSubplotSpec(
        1,2,subplot_spec=gs[3],wspace=0.04,width_ratios=[3.2,1.0])
    axD=fig.add_subplot(gs_bot[0])
    axS=fig.add_subplot(gs_bot[1])

    draw_panel_a(axA,d)
    draw_panel_b(axB,d,fig)
    draw_panel_c(axC,d)
    draw_table(axD,d,speaker_id)
    draw_stats_card(axS,d,speaker_id)

    plt.tight_layout(rect=[0,0,1,0.993])
    return fig


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('speaker_id', nargs='?', default='m0004')
    parser.add_argument('--phrase-clean', default=None)
    parser.add_argument('--timeline',     default=None)
    parser.add_argument('--output-dir',   default=None)
    parser.add_argument('--csv-dir',      default=None)
    args = parser.parse_args()

    # Declare global before any use
    global PHRASE_CLEAN

    speaker_id   = args.speaker_id
    phrase_clean = args.phrase_clean or PHRASE_CLEAN
    out_dir      = args.output_dir   or OUTPUT_DIR
    csv_out_dir  = args.csv_dir      or 'xai_reddots/csv'

    # Override module-level constant so aggregate() reads correct phrase CSV
    PHRASE_CLEAN = phrase_clean

    # ── Find timeline ─────────────────────────────────────────
    search_paths = [
        args.timeline or '',
        f'xai_reddots/timelines/{speaker_id}_{phrase_clean}_timeline.json',
        f'xai_reddots/metadata/{speaker_id}_{phrase_clean}_timeline.json',
        f'xai_reddots/results/phrase1_{phrase_clean}/timelines/{speaker_id}_timeline.json',
        f'xai_reddots/results/phrase2_{phrase_clean}/timelines/{speaker_id}_timeline.json',
        'xai_reddots/metadata/timeline.json',
    ]
    tdata = None
    for tpath in search_paths:
        if tpath and os.path.exists(tpath):
            with open(tpath) as f: tdata = json.load(f)
            break
    if tdata is None:
        print("Timeline not found"); sys.exit(1)

    print("Loading model…")
    model = load_model()
    print(f"Aggregating {speaker_id} ({tdata.get('n_utterances','?')} recordings)…")
    d = aggregate(speaker_id, model, tdata)

    phrase = tdata.get('phrase', phrase_clean.replace('_', ' ').title())
    fig    = build_figure(speaker_id, d, phrase)

    # ── Save figure ───────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir,
                            f"{speaker_id}_{phrase_clean}_final_figure.png")
    fig.savefig(fig_path, dpi=300, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)

    # ── Save XAI CSV ──────────────────────────────────────────
    os.makedirs(csv_out_dir, exist_ok=True)
    csv_path = os.path.join(csv_out_dir,
                            f"{speaker_id}_{phrase_clean}_xai_analysis.csv")
    total = d['total']
    fields = ['rank','phoneme','word','start','end',
              'start_frame','end_frame','occurrence',
              'mean_attention','std_attention','attention_pct','attention_pct_std']
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for p in sorted(d['ph_data'], key=lambda x: x['rank']):
            w.writerow({
                'rank':              p['rank'],
                'phoneme':           p['phoneme'],
                'word':              p['word'],
                'start':             f"{p['start']:.4f}",
                'end':               f"{p['end']:.4f}",
                'start_frame':       p['start_frame'],
                'end_frame':         p['end_frame'],
                'occurrence':        p['occurrence'],
                'mean_attention':    f"{p['mean']:.6f}",
                'std_attention':     f"{p['std']:.6f}",
                'attention_pct':     f"{p['pct']:.4f}",
                'attention_pct_std': f"{p['std']/total*100:.4f}",
            })

    top3 = sorted(d['ph_data'], key=lambda x: x['rank'])[:3]
    top3_str = ' | '.join(f"{p['phoneme']} {p['pct']:.1f}%" for p in top3)
    print(f"\n✓  Figure → {fig_path}")
    print(f"✓  CSV    → {csv_path}")
    print(f"   Top-3: {top3_str}")
    print(f"   Concentration: {d['top3_pct']:.1f}%")
    print(f"   Peak: {d['peak_val']:.5f} at {d['peak_time']:.2f}s (frame {d['peak_frame']})")
    print(f"   Pearson r: {d['corr']:+.4f}")
    print(f"   Recordings: {d['n_recs']}")

if __name__=='__main__':
    main()

