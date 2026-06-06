"""
individual_panels.py
====================
Generates 4 separate publication figures (each with plot + data table) for a speaker:

  panel_a_rms_attention.png        — RMS Energy + ECAPA Attention + phoneme table
  panel_b_mel_spectrogram.png      — Mel Spectrogram + ECAPA Attention + phoneme table
  panel_c_attention_boundaries.png — Attention curve + boundaries + ranking table
  panel_d_phoneme_ranking.png      — Horizontal ranking bar + full frame-level table

Usage:
    python3 xai_reddots/scripts/individual_panels.py m0004
    python3 xai_reddots/scripts/individual_panels.py m0001
    python3 xai_reddots/scripts/individual_panels.py m0002
"""

import os, csv, json, sys, math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.mlab as mlab
import soundfile as sf
import torch
from matplotlib.gridspec import GridSpec

sys.path.append(os.getcwd())
from ECAPAModel import ECAPAModel

# ── CONFIG ────────────────────────────────────────────────────
PHRASE_CLEAN = "my_voice_is_my_password"
OUTPUT_DIR   = "xai_reddots/plots/individual_panels"
SR = 16000;  HOP = 160;  WIN = 400;  NFFT = 512;  N_MELS = 80

# ── DESIGN TOKENS ─────────────────────────────────────────────
BG       = '#ffffff'
PANEL_BG = '#fafafa'
GRID_COL = '#e8e8e8'
ATTN_COL = '#d35400'
ATTN_FILL= '#fae0d3'
RMS_COL  = '#1a6b3c'
RMS_FILL = '#d6ede2'
BAR_COL  = '#4a235a'
PHON_COL = '#c0392b'
SPINE_COL= '#cccccc'
TBL_HEAD = '#2c3e50'
TBL_ODD  = '#f4f6f8'
TBL_EVEN = '#ffffff'

LFS = 11;  TFS = 9;  TTLE = 12;  ANNFS = 8.5


# ── MODEL ─────────────────────────────────────────────────────
def load_model():
    args = type('Args', (), {})()
    args.C=1024; args.m=0.2; args.s=30; args.n_class=5994
    args.lr=0.001; args.lr_decay=0.97; args.test_step=1
    m = ECAPAModel(**vars(args))
    m.load_parameters("exps/pretrain.model")
    m.speaker_encoder.eval()
    return m

def extract_attention(model, pcm):
    audio, _ = sf.read(pcm, channels=1, samplerate=SR, subtype='PCM_16', format='RAW')
    if len(audio) < 400: audio = np.pad(audio, (0, 400-len(audio)))
    data = torch.FloatTensor(np.stack([audio])).cuda()
    with torch.no_grad():
        x = model.speaker_encoder.torchfbank(data)+1e-6
        x = x.log() - torch.mean(x.log(), dim=-1, keepdim=True)
        x = model.speaker_encoder.bn1(model.speaker_encoder.relu(model.speaker_encoder.conv1(x)))
        x1=model.speaker_encoder.layer1(x); x2=model.speaker_encoder.layer2(x+x1)
        x3=model.speaker_encoder.layer3(x+x1+x2)
        x=model.speaker_encoder.relu(model.speaker_encoder.layer4(torch.cat([x1,x2,x3],1)))
        t=x.size(-1)
        gx=torch.cat([x,torch.mean(x,2,keepdim=True).repeat(1,1,t),
                      torch.sqrt(torch.var(x,2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)],1)
        wl=gx
        for layer in model.speaker_encoder.attention[:-1]: wl=layer(wl)
        a=torch.softmax(wl,dim=1); H=-torch.sum(a*torch.log(a+1e-9),dim=1)
        conf=1.0-H/math.log(a.shape[1]); alpha=conf/torch.sum(conf,dim=1,keepdim=True)
    return alpha.squeeze().cpu().numpy(), audio

def rms_envelope(audio):
    hw=WIN//2; n=len(audio)//HOP; out=np.zeros(n)
    for i in range(n):
        c=i*HOP; seg=audio[max(0,c-hw):min(len(audio),c+hw)]
        out[i]=np.sqrt(np.mean(seg**2)) if len(seg) else 0.0
    return out

def hz_to_mel(hz): return 2595*np.log10(1+hz/700)

def mel_filterbank():
    low,high=hz_to_mel(0),hz_to_mel(SR/2)
    pts=np.linspace(low,high,N_MELS+2); hz=700*(10**(pts/2595)-1)
    bins=np.floor((NFFT+1)*hz/SR).astype(int)
    fb=np.zeros((N_MELS,NFFT//2+1))
    for m in range(1,N_MELS+1):
        for k in range(bins[m-1],bins[m]):
            fb[m-1,k]=(k-bins[m-1])/(bins[m]-bins[m-1]+1e-9)
        for k in range(bins[m],bins[m+1]+1):
            fb[m-1,k]=(bins[m+1]-k)/(bins[m+1]-bins[m]+1e-9)
    return fb

def mel_spec(audio):
    Pxx,_,bins=mlab.specgram(audio,NFFT=NFFT,Fs=SR,noverlap=NFFT-HOP,window=mlab.window_hanning)
    return mel_filterbank()@Pxx, bins


# ── AGGREGATE ─────────────────────────────────────────────────
def aggregate(speaker_id, model, tdata):
    sep_csv=f'xai_reddots/metadata/separated_phrases/{PHRASE_CLEAN}.csv'
    recs=[]
    with open(sep_csv) as f:
        for row in csv.DictReader(f):
            if row['speaker_id']==speaker_id and os.path.exists(row['pcm_path']):
                recs.append(row['pcm_path'])
    if not recs: raise RuntimeError(f"No recs for {speaker_id}")

    avg_dur=float(tdata.get('avg_duration_sec',3.0))
    rep_prof,_=extract_attention(model,recs[0])
    P=len(rep_prof); prof_t=np.linspace(0,avg_dur,P)

    all_profs,all_rms,all_mel=[],[],[]
    for path in recs:
        try:
            prof,audio_i=extract_attention(model,path); dur_i=len(audio_i)/SR
            all_profs.append(np.interp(prof_t,np.linspace(0,dur_i,len(prof)),prof))
            r=rms_envelope(audio_i)
            all_rms.append(np.interp(prof_t,np.linspace(0,dur_i,len(r)),r))
            an=audio_i/(np.max(np.abs(audio_i))+1e-9)
            mP,_=mel_spec(an); bx=np.linspace(0,dur_i,mP.shape[1])
            mi=np.zeros((mP.shape[0],P))
            for fi in range(mP.shape[0]): mi[fi]=np.interp(np.linspace(0,avg_dur,P),bx,mP[fi])
            all_mel.append(mi)
        except Exception as e: print(f"  skip {path}: {e}")

    all_profs=np.vstack(all_profs)
    mean_prof=np.mean(all_profs,0); std_prof=np.std(all_profs,0)
    mean_rms=np.mean(all_rms,0)
    mel_db=10*np.log10(np.mean(all_mel,0)+1e-10)

    phonemes=tdata.get('phonemes',[])
    occ={ph['phoneme']:sum(1 for p in phonemes if p['phoneme']==ph['phoneme']) for ph in phonemes}
    ph_data=[]
    for ph in phonemes:
        s,e=float(ph['start']),float(ph['end'])
        mask=(prof_t>=s)&(prof_t<=e)
        if not np.any(mask):
            ci=np.argmin(np.abs(prof_t-(s+e)/2)); mask=np.zeros(P,bool); mask[ci]=True
        vals=np.array([np.mean(all_profs[r,mask]) for r in range(all_profs.shape[0])])
        ph_data.append({'phoneme':ph['phoneme'],'word':ph['word'],
                        'label':f"{ph['word']}: /{ph['phoneme']}/",
                        'start':s,'end':e,
                        'start_frame':int(round(s*100)),'end_frame':int(round(e*100)),
                        'occurrence':occ[ph['phoneme']],
                        'mean':float(np.mean(vals)),'std':float(np.std(vals))})

    total=sum(p['mean'] for p in ph_data) or 1e-12
    for rank_i,p in enumerate(sorted(ph_data,key=lambda x:x['mean'],reverse=True),1):
        p['pct']=p['mean']/total*100; p['rank']=rank_i
    rm={p['label']:(p['pct'],p['rank']) for p in ph_data}
    for p in ph_data: p['pct'],p['rank']=rm[p['label']]

    rms_norm=mean_rms/(np.max(mean_rms)+1e-9)
    # RMS value per phoneme (for table)
    for p in ph_data:
        mask=(prof_t>=p['start'])&(prof_t<=p['end'])
        p['rms']=float(np.mean(rms_norm[mask])) if np.any(mask) else 0.0

    top3_pct=sum(p['pct'] for p in sorted(ph_data,key=lambda x:x['mean'],reverse=True)[:3])
    corr=float(np.corrcoef(mean_rms,mean_prof)[0,1])
    n=len(recs)

    return dict(avg_dur=avg_dur,prof_t=prof_t,mean_prof=mean_prof,std_prof=std_prof,
                mean_rms=mean_rms,mel_db=mel_db,phonemes=phonemes,ph_data=ph_data,
                total=total,top3_pct=top3_pct,corr=corr,n_recs=n,
                rms_norm=mean_rms/(np.max(mean_rms)+1e-9))


# ── TABLE HELPER ──────────────────────────────────────────────
def draw_table(ax, col_labels, rows, col_widths=None):
    """Draw a clean styled table on a blank axes."""
    ax.set_facecolor(BG); ax.axis('off')
    n_cols=len(col_labels); n_rows=len(rows)
    if col_widths is None: col_widths=[1/n_cols]*n_cols

    row_h=1.0/(n_rows+1)
    xs=[sum(col_widths[:i]) for i in range(n_cols)]

    # Header
    for ci,(lbl,cw) in enumerate(zip(col_labels,col_widths)):
        ax.add_patch(patches.FancyBboxPatch((xs[ci],1-row_h),cw,row_h*0.9,
            boxstyle='round,pad=0.005',fc=TBL_HEAD,ec='none',
            transform=ax.transAxes,clip_on=False))
        ax.text(xs[ci]+cw/2,1-row_h/2,lbl,ha='center',va='center',
                fontsize=8,color='white',fontweight='bold',
                transform=ax.transAxes)

    # Rows
    for ri,row in enumerate(rows):
        ry=1-(ri+2)*row_h
        bg=TBL_ODD if ri%2==0 else TBL_EVEN
        ax.add_patch(patches.FancyBboxPatch((0,ry),1,row_h*0.9,
            boxstyle='round,pad=0.003',fc=bg,ec=SPINE_COL,lw=0.4,
            transform=ax.transAxes,clip_on=False))
        for ci,(val,cw) in enumerate(zip(row,col_widths)):
            fw='bold' if ci==0 else 'normal'
            ax.text(xs[ci]+cw/2,ry+row_h/2,str(val),ha='center',va='center',
                    fontsize=8,color='#1a1a1a',fontweight=fw,
                    transform=ax.transAxes)


def style_ax(ax,title,xlabel,ylabel,xlim):
    ax.set_facecolor(PANEL_BG)
    ax.grid(True,ls='--',color=GRID_COL,alpha=0.8,lw=0.7)
    ax.set_xlim(0,xlim); ax.set_xlabel(xlabel,fontsize=LFS)
    ax.set_ylabel(ylabel,fontsize=LFS); ax.tick_params(labelsize=TFS)
    ax.set_title(title,fontsize=TTLE,fontweight='bold',loc='left',pad=6)
    for sp in ax.spines.values(): sp.set_color(SPINE_COL); sp.set_linewidth(0.8)

def add_phoneme_lines(ax,phonemes,ypos,fs=7.5):
    if not phonemes: return
    ax.axvline(phonemes[0]['start'],color=PHON_COL,lw=0.8,ls='--',alpha=0.55)
    for ph in phonemes:
        ax.axvline(ph['end'],color=PHON_COL,lw=0.8,ls='--',alpha=0.55)
        cx=(ph['start']+ph['end'])/2
        ax.text(cx,ypos,ph['phoneme'],color=PHON_COL,ha='center',va='top',
                fontsize=fs,fontweight='bold',alpha=0.8)

def twin_attn(ax,t,prof,std):
    ax2=ax.twinx()
    ax2.plot(t,prof,color=ATTN_COL,lw=2.0,label='ECAPA Attention',zorder=5)
    ax2.fill_between(t,np.maximum(0,prof-std),prof+std,color=ATTN_FILL,alpha=0.35,zorder=4)
    ax2.set_ylabel("Attention weight",color=ATTN_COL,fontsize=LFS)
    ax2.tick_params(axis='y',labelcolor=ATTN_COL,labelsize=TFS)
    ax2.set_ylim(0,np.max(prof+std)*1.4)
    for sp in ['top','left','bottom']: ax2.spines[sp].set_visible(False)
    ax2.spines['right'].set_color(ATTN_COL)
    ax2.legend(loc='upper right',fontsize=ANNFS,framealpha=0.7)
    return ax2


# ─────────────────────────────────────────────────────────────
# PANEL A — RMS Energy + Attention
# ─────────────────────────────────────────────────────────────
def panel_a(speaker_id, d, phrase):
    fig=plt.figure(figsize=(14,11),facecolor=BG)
    fig.suptitle(f"Panel A — RMS Energy Envelope + ECAPA Attention\n"
                 f"Speaker: {speaker_id}  •  \"{phrase}\"  •  n={d['n_recs']} recordings",
                 fontsize=13,fontweight='bold',y=0.99)
    gs=GridSpec(2,1,figure=fig,height_ratios=[2.8,1.2],hspace=0.38)

    # Plot
    ax=fig.add_subplot(gs[0])
    rn=d['rms_norm']; t=d['prof_t']; avg=d['avg_dur']
    ax.plot(t,rn,color=RMS_COL,lw=1.8,label='RMS Energy (norm.)',zorder=5)
    ax.fill_between(t,0,rn,color=RMS_FILL,alpha=0.4,zorder=3)
    style_ax(ax,f"RMS Energy Envelope + ECAPA Attention  [Pearson r = {d['corr']:+.3f}]",
             "Time (s)","RMS Energy (norm.)",avg)
    ax.set_ylim(0,1.5)
    add_phoneme_lines(ax,d['phonemes'],1.38,fs=7)
    ax.legend(loc='upper left',fontsize=ANNFS,framealpha=0.7)
    twin_attn(ax,t,d['mean_prof'],d['std_prof'])

    # Table
    ax2=fig.add_subplot(gs[1])
    cols=['Phoneme','Word','Start (s)','End (s)','Start Frame','End Frame',
          'RMS (norm)','Attention %','Rank']
    cw=[0.09,0.11,0.09,0.09,0.10,0.10,0.11,0.12,0.07]
    rows=[]
    for p in sorted(d['ph_data'],key=lambda x:x['start']):
        rows.append([p['phoneme'],p['word'],f"{p['start']:.3f}",f"{p['end']:.3f}",
                     str(p['start_frame']),str(p['end_frame']),
                     f"{p['rms']:.3f}",f"{p['pct']:.1f}%",f"#{p['rank']}"])
    draw_table(ax2,cols,rows,cw)
    ax2.set_title("Phoneme-Level Data",fontsize=10,fontweight='bold',loc='left',pad=4)

    return fig


# ─────────────────────────────────────────────────────────────
# PANEL B — Mel Spectrogram + Attention
# ─────────────────────────────────────────────────────────────
def panel_b(speaker_id, d, phrase):
    fig=plt.figure(figsize=(14,11),facecolor=BG)
    fig.suptitle(f"Panel B — Averaged Mel Spectrogram + ECAPA Attention\n"
                 f"Speaker: {speaker_id}  •  \"{phrase}\"  •  n={d['n_recs']} recordings",
                 fontsize=13,fontweight='bold',y=0.99)
    gs=GridSpec(2,1,figure=fig,height_ratios=[2.8,1.2],hspace=0.38)

    ax=fig.add_subplot(gs[0])
    mel_db=d['mel_db']; t=d['prof_t']; avg=d['avg_dur']
    vmax=np.percentile(mel_db,99.5); vmin=vmax-45
    mel_hz=700*(10**(np.linspace(hz_to_mel(0),hz_to_mel(SR/2),N_MELS)/2595)-1)
    ax.pcolormesh(t,mel_hz,mel_db,cmap='gray_r',vmin=vmin,vmax=vmax,
                  shading='nearest',rasterized=True)
    ax.set_ylim(0,8000)
    ax.set_yticks([0,2000,4000,6000,8000])
    ax.set_yticklabels(['0','2k','4k','6k','8k'],fontsize=TFS)
    style_ax(ax,"Averaged Mel Spectrogram + ECAPA Attention",
             "Time (s)","Frequency (Hz)",avg)
    add_phoneme_lines(ax,d['phonemes'],7500,fs=7)
    twin_attn(ax,t,d['mean_prof'],d['std_prof'])

    # Table — frequency content per phoneme (average mel energy in key bands)
    ax2=fig.add_subplot(gs[1])
    cols=['Phoneme','Word','Start (s)','End (s)','Frames','Attention %','Rank']
    cw=[0.11,0.14,0.12,0.12,0.15,0.18,0.10]
    rows=[]
    for p in sorted(d['ph_data'],key=lambda x:x['start']):
        frames=f"{p['start_frame']}–{p['end_frame']}"
        rows.append([p['phoneme'],p['word'],f"{p['start']:.3f}",f"{p['end']:.3f}",
                     frames,f"{p['pct']:.1f}%",f"#{p['rank']}"])
    draw_table(ax2,cols,rows,cw)
    ax2.set_title("Phoneme Boundaries on Spectrogram",fontsize=10,fontweight='bold',loc='left',pad=4)

    return fig


# ─────────────────────────────────────────────────────────────
# PANEL C — Attention + Phoneme Boundaries
# ─────────────────────────────────────────────────────────────
def panel_c(speaker_id, d, phrase):
    fig=plt.figure(figsize=(14,12),facecolor=BG)
    fig.suptitle(f"Panel C — ECAPA Entropy Attention + Phoneme Boundaries\n"
                 f"Speaker: {speaker_id}  •  \"{phrase}\"  •  n={d['n_recs']} recordings",
                 fontsize=13,fontweight='bold',y=0.99)
    gs=GridSpec(2,1,figure=fig,height_ratios=[2.8,1.5],hspace=0.38)

    ax=fig.add_subplot(gs[0])
    t=d['prof_t']; mp=d['mean_prof']; sp=d['std_prof']; avg=d['avg_dur']
    ax.plot(t,mp,color=BAR_COL,lw=2.2,label='Mean attention',zorder=5)
    ax.fill_between(t,np.maximum(0,mp-sp),mp+sp,color='#e0d0f0',alpha=0.50,
                    label='±1 std',zorder=4)

    # Word shading
    wb={}
    for ph in d['phonemes']:
        w=ph['word']
        if w not in wb: wb[w]=[ph['start'],ph['end']]
        else: wb[w][0]=min(wb[w][0],ph['start']); wb[w][1]=max(wb[w][1],ph['end'])
    wc=['#e8f4fd','#fef9e7','#eafaf1','#fdf2f8','#f0f3ff']
    ylim=np.max(mp+sp)*1.35; ax.set_ylim(0,ylim)
    for wi,(w,(ws,we)) in enumerate(sorted(wb.items(),key=lambda x:x[1][0])):
        ax.axvspan(ws,we,alpha=0.18,color=wc[wi%len(wc)],zorder=2)
        ax.text((ws+we)/2,ylim*0.88,w,ha='center',fontsize=8,
                color='#555',fontweight='semibold',style='italic')

    style_ax(ax,"ECAPA Entropy Attention + Phoneme Boundaries",
             "Time (s)","Attention weight",avg)
    add_phoneme_lines(ax,d['phonemes'],ylim*0.96,fs=7.5)
    ax.legend(loc='upper right',fontsize=ANNFS,framealpha=0.7)

    pi=np.argmax(mp); peak_t=t[pi]; peak_v=mp[pi]
    ax.annotate(f"peak\n{peak_t:.2f}s",xy=(peak_t,peak_v),
                xytext=(min(peak_t+avg*0.07,avg*0.9),peak_v*0.88),
                fontsize=ANNFS,color=BAR_COL,
                arrowprops=dict(arrowstyle='->',color=BAR_COL,lw=1.2))

    # Table — ranked attention
    ax2=fig.add_subplot(gs[1])
    cols=['Rank','Phoneme','Word','Start (s)','End (s)','Attention %','±Std %','×Occ']
    cw=[0.07,0.10,0.14,0.11,0.11,0.14,0.14,0.09]
    sorted_ph=sorted(d['ph_data'],key=lambda x:x['rank'])
    rows=[]
    total=d['total']
    for p in sorted_ph:
        std_pct=p['std']/total*100
        rows.append([f"#{p['rank']}",p['phoneme'],p['word'],
                     f"{p['start']:.3f}",f"{p['end']:.3f}",
                     f"{p['pct']:.2f}%",f"±{std_pct:.2f}%",
                     f"×{p['occurrence']}"])
    draw_table(ax2,cols,rows,cw)
    ax2.set_title(f"Full Phoneme Attention Ranking  [Top-3 concentration = {d['top3_pct']:.1f}%]",
                  fontsize=10,fontweight='bold',loc='left',pad=4)

    return fig


# ─────────────────────────────────────────────────────────────
# PANEL D — Phoneme Attention % Ranking
# ─────────────────────────────────────────────────────────────
def panel_d(speaker_id, d, phrase):
    fig=plt.figure(figsize=(14,13),facecolor=BG)
    fig.suptitle(f"Panel D — Phoneme Attention % Ranking\n"
                 f"Speaker: {speaker_id}  •  \"{phrase}\"  •  n={d['n_recs']} recordings",
                 fontsize=13,fontweight='bold',y=0.99)
    gs=GridSpec(2,1,figure=fig,height_ratios=[2.2,2.2],hspace=0.42)

    # Bar chart
    ax=fig.add_subplot(gs[0])
    ax.set_facecolor(PANEL_BG)
    ax.grid(True,ls='--',color=GRID_COL,alpha=0.8,lw=0.7,axis='x')
    for sp in ax.spines.values(): sp.set_color(SPINE_COL); sp.set_linewidth(0.8)

    ph_s=sorted(d['ph_data'],key=lambda x:x['pct'])   # ascending for barh
    labels=[p['label'] for p in ph_s]
    pcts  =[p['pct']   for p in ph_s]
    total =d['total']
    pstds =[p['std']/total*100 for p in ph_s]
    ranks =[p['rank']  for p in ph_s]
    occs  =[p['occurrence'] for p in ph_s]

    q75=np.percentile(pcts,75); med=np.median(pcts)
    bc=['#6c3483' if p>=q75 else '#a569bd' if p>=med else '#d7bde2' for p in pcts]

    bars=ax.barh(labels,pcts,xerr=pstds,color=bc,alpha=0.90,
                 edgecolor='#999',lw=0.5,height=0.65,
                 error_kw=dict(ecolor='#444',lw=1.0,capsize=3))
    xmx=max(p+s for p,s in zip(pcts,pstds))
    for bar,pct,std,rank,occ in zip(bars,pcts,pstds,ranks,occs):
        ax.text(pct+std+xmx*0.012,bar.get_y()+bar.get_height()/2,
                f"#{rank}  {pct:.1f}% ± {std:.1f}%  ×{occ}",
                va='center',ha='left',fontsize=8,color='#222')
        if pct>xmx*0.08:
            ax.text(pct*0.04,bar.get_y()+bar.get_height()/2,f"#{rank}",
                    va='center',ha='left',fontsize=7,color='white',fontweight='bold')
    ax.set_title(f"Phoneme Attention % Ranking  [Top-3 = {d['top3_pct']:.1f}%]",
                 fontsize=TTLE,fontweight='bold',loc='left',pad=6)
    ax.set_xlabel("Attention share (%)",fontsize=LFS)
    ax.set_xlim(0,xmx*1.6); ax.tick_params(labelsize=TFS)

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc='#6c3483',label='Top quartile'),
                        Patch(fc='#a569bd',label='Above median'),
                        Patch(fc='#d7bde2',label='Below median')],
              loc='lower right',fontsize=ANNFS,framealpha=0.75)

    # Full frame-level evidence table
    ax2=fig.add_subplot(gs[1])
    cols=['Rank','Phoneme','Word','Start (s)','End (s)',
          'Start Frame','End Frame','Mean Attn','Attention %','±Std %','×Occ']
    cw=[0.06,0.08,0.10,0.08,0.08,0.09,0.09,0.10,0.10,0.09,0.07]
    rows=[]
    for p in sorted(d['ph_data'],key=lambda x:x['rank']):
        std_pct=p['std']/total*100
        rows.append([f"#{p['rank']}",p['phoneme'],p['word'],
                     f"{p['start']:.3f}",f"{p['end']:.3f}",
                     str(p['start_frame']),str(p['end_frame']),
                     f"{p['mean']:.5f}",f"{p['pct']:.2f}%",
                     f"±{std_pct:.2f}%",f"×{p['occurrence']}"])
    draw_table(ax2,cols,rows,cw)
    ax2.set_title("Frame-Level Evidence Table",fontsize=10,fontweight='bold',loc='left',pad=4)

    return fig


# ── MAIN ──────────────────────────────────────────────────────
def main():
    speaker_id=sys.argv[1] if len(sys.argv)>1 else "m0004"

    for tpath in [
        f'xai_reddots/timelines/{speaker_id}_{PHRASE_CLEAN}_timeline.json',
        f'xai_reddots/metadata/{speaker_id}_{PHRASE_CLEAN}_timeline.json',
        'xai_reddots/metadata/timeline.json']:
        if os.path.exists(tpath):
            with open(tpath) as f: tdata=json.load(f)
            break
    else:
        print("Timeline not found"); sys.exit(1)

    print("Loading model…"); model=load_model()
    print(f"Aggregating {speaker_id}…"); d=aggregate(speaker_id,model,tdata)

    phrase=tdata.get('phrase','My voice is my password')
    os.makedirs(OUTPUT_DIR,exist_ok=True)

    panels={'a_rms_attention':      panel_a,
            'b_mel_spectrogram':    panel_b,
            'c_attention_boundaries':panel_c,
            'd_phoneme_ranking':    panel_d}

    for name,fn in panels.items():
        fig=fn(speaker_id,d,phrase)
        out=os.path.join(OUTPUT_DIR,
                         f"{speaker_id}_{PHRASE_CLEAN}_panel_{name}.png")
        fig.savefig(out,dpi=300,bbox_inches='tight',
                    facecolor=fig.get_facecolor(),edgecolor='none')
        plt.close(fig)
        print(f"✓  {out}")

    print(f"\nAll 4 individual panels saved to: {OUTPUT_DIR}/")

if __name__=='__main__':
    main()
