"""
batch_pipeline.py
=================
Runs the full XAI pipeline for 2 phrases × 5 speakers each.

Output structure:
  xai_reddots/results/
    phrase1_my_voice_is_my_password/
      timelines/   <speaker>_timeline.json
      csv/         <speaker>_xai_analysis.csv
      plots/       <speaker>_final_figure.png
    phrase2_actions_speak_louder_than_words/
      timelines/
      csv/
      plots/

Usage:
    python3 xai_reddots/scripts/batch_pipeline.py
    python3 xai_reddots/scripts/batch_pipeline.py --dry-run
"""

import os, sys, csv, json, shutil, argparse, math, time
import numpy as np
import soundfile as sf

sys.path.append(os.getcwd())

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
SPEAKERS  = ['m0001', 'm0002', 'm0004', 'm0005', 'm0006']
SR = 16000

RESULTS_DIR  = 'xai_reddots/results'
SEP_DIR      = 'xai_reddots/metadata/separated_phrases'
META_DIR     = 'xai_reddots/metadata'

# ─────────────────────────────────────────────────────────────
# PHRASE DEFINITIONS
# ─────────────────────────────────────────────────────────────
# Each phrase has:
#   key        : filesystem-safe name
#   display    : human-readable phrase
#   words      : list of (word, [phonemes])
#   syl_weights: relative duration weight per word (syllable count)

PHRASES = [
    {
        'key'    : 'my_voice_is_my_password',
        'display': 'My voice is my password',
        'words'  : [
            ('my',       ['M', 'AY']),
            ('voice',    ['V', 'OY', 'S']),
            ('is',       ['IH', 'Z']),
            ('my',       ['M', 'AY']),
            ('password', ['P', 'AE', 'S', 'W', 'ER', 'D']),
        ],
        'syl_weights': [1, 1, 1, 1, 2],   # syllable counts
    },
    {
        'key'    : 'actions_speak_louder_than_words',
        'display': 'Actions speak louder than words',
        'words'  : [
            ('actions', ['AE', 'K', 'SH', 'AH', 'N', 'Z']),
            ('speak',   ['S',  'P', 'IY', 'K']),
            ('louder',  ['L',  'AW', 'D', 'ER']),
            ('than',    ['DH', 'AE', 'N']),
            ('words',   ['W',  'ER', 'D', 'Z']),
        ],
        'syl_weights': [2, 1, 2, 1, 1],
    },
    {
        'key'    : 'ok_google',
        'display': 'Ok Google',
        'words'  : [
            ('ok',     ['OW', 'K', 'EY']),
            ('google', ['G', 'UW', 'G', 'AH', 'L']),
        ],
        'syl_weights': [2, 2],
    },
    {
        'key'    : 'there_s_no_such_thing_as_a_free_lunch',
        'display': "There's no such thing as a free lunch",
        'words'  : [
            ("there's", ['DH', 'EH', 'R', 'Z']),
            ('no',      ['N', 'OW']),
            ('such',    ['S', 'AH', 'CH']),
            ('thing',   ['TH', 'IH', 'NG']),
            ('as',      ['AE', 'Z']),
            ('a',       ['AH']),
            ('free',    ['F', 'R', 'IY']),
            ('lunch',   ['L', 'AH', 'N', 'CH']),
        ],
        'syl_weights': [1, 1, 1, 1, 1, 1, 1, 1],
    },
    {
        'key'    : 'a_watched_pot_never_boils',
        'display': 'A watched pot never boils',
        'words'  : [
            ('a',       ['AH']),
            ('watched', ['W', 'AA', 'CH', 'T']),
            ('pot',     ['P', 'AA', 'T']),
            ('never',   ['N', 'EH', 'V', 'ER']),
            ('boils',   ['B', 'OY', 'L', 'Z']),
        ],
        'syl_weights': [1, 1, 1, 2, 1],
    },
    {
        'key'    : 'jealousy_has_twenty_twenty_vision',
        'display': 'Jealousy has twenty twenty vision',
        'words'  : [
            ('jealousy', ['JH', 'EH', 'L', 'AH', 'S', 'IY']),
            ('has',      ['HH', 'AE', 'Z']),
            ('twenty',   ['T', 'W', 'EH', 'N', 'T', 'IY']),
            ('twenty',   ['T', 'W', 'EH', 'N', 'T', 'IY']),
            ('vision',   ['V', 'IH', 'ZH', 'AH', 'N']),
        ],
        'syl_weights': [3, 1, 2, 2, 2],
    },
    {
        'key'    : 'necessity_is_the_mother_of_invention',
        'display': 'Necessity is the mother of invention',
        'words'  : [
            ('necessity', ['N', 'AH', 'S', 'EH', 'S', 'AH', 'T', 'IY']),
            ('is',        ['IH', 'Z']),
            ('the',       ['DH', 'AH']),
            ('mother',    ['M', 'AH', 'DH', 'ER']),
            ('of',        ['AH', 'V']),
            ('invention', ['IH', 'N', 'V', 'EH', 'N', 'SH', 'AH', 'N']),
        ],
        'syl_weights': [4, 1, 1, 2, 1, 3],
    },
    {
        'key'    : 'artificial_intelligence_is_for_real',
        'display': 'Artificial intelligence is for real',
        'words'  : [
            ('artificial',   ['AA', 'R', 'T', 'AH', 'F', 'IH', 'SH', 'AH', 'L']),
            ('intelligence', ['IH', 'N', 'T', 'EH', 'L', 'AH', 'JH', 'AH', 'N', 'S']),
            ('is',           ['IH', 'Z']),
            ('for',          ['F', 'AO', 'R']),
            ('real',         ['R', 'IY', 'L']),
        ],
        'syl_weights': [4, 4, 1, 1, 1],
    },
    {
        'key'    : 'only_lawyers_love_millionaires',
        'display': 'Only lawyers love millionaires',
        'words'  : [
            ('only',         ['OW', 'N', 'L', 'IY']),
            ('lawyers',      ['L', 'AO', 'Y', 'ER', 'Z']),
            ('love',         ['L', 'AH', 'V']),
            ('millionaires', ['M', 'IH', 'L', 'Y', 'AH', 'N', 'EH', 'R', 'Z']),
        ],
        'syl_weights': [2, 2, 1, 3],
    },
    {
        'key'    : 'birthday_parties_have_cupcakes_and_ice_cream',
        'display': 'Birthday parties have cupcakes and ice cream',
        'words'  : [
            ('birthday', ['B', 'ER', 'TH', 'D', 'EY']),
            ('parties',  ['P', 'AA', 'R', 'T', 'IY', 'Z']),
            ('have',     ['HH', 'AE', 'V']),
            ('cupcakes', ['K', 'AH', 'P', 'K', 'EY', 'K', 'S']),
            ('and',      ['AE', 'N', 'D']),
            ('ice',      ['AY', 'S']),
            ('cream',    ['K', 'R', 'IY', 'M']),
        ],
        'syl_weights': [2, 2, 1, 2, 1, 1, 1],
    },
]

# Existing timelines for phrase1 speakers (already aligned)
EXISTING_TIMELINES = {
    ('my_voice_is_my_password', 'm0001'): 'xai_reddots/metadata/timeline.json',
    ('my_voice_is_my_password', 'm0002'): 'xai_reddots/metadata/m0002_my_voice_is_my_password_timeline.json',
    ('my_voice_is_my_password', 'm0004'): 'xai_reddots/metadata/m0004_my_voice_is_my_password_timeline.json',
}


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def get_recordings(phrase_key, speaker_id):
    """Return list of existing PCM paths for this phrase+speaker."""
    sep_csv = os.path.join(SEP_DIR, f'{phrase_key}.csv')
    if not os.path.exists(sep_csv):
        return []
    recs = []
    with open(sep_csv) as f:
        for row in csv.DictReader(f):
            if row['speaker_id'] == speaker_id and os.path.exists(row['pcm_path']):
                recs.append(row['pcm_path'])
    return recs


def pcm_duration(path):
    """Duration in seconds for a raw PCM file at SR=16000, 16-bit."""
    try:
        audio, _ = sf.read(path, channels=1, samplerate=SR,
                           subtype='PCM_16', format='RAW')
        return len(audio) / SR
    except Exception:
        return None


def compute_avg_duration(recs, max_recs=30):
    """Average duration of recordings (seconds)."""
    durations = []
    for r in recs[:max_recs]:
        d = pcm_duration(r)
        if d and 1.0 < d < 10.0:
            durations.append(d)
    return float(np.mean(durations)) if durations else 3.0


# ─────────────────────────────────────────────────────────────
# TIMELINE GENERATION
# ─────────────────────────────────────────────────────────────
def generate_timeline(phrase_def, speaker_id, avg_dur, n_recs):
    """
    Build a timeline JSON matching the existing format.
    Phoneme timings are estimated by:
      - Speech starts at 30% of avg_dur, ends at 95%
      - Words allocated proportional to their syl_weight
      - Phonemes within each word allocated equally
    """
    speech_start = avg_dur * 0.18
    speech_end   = avg_dur * 0.93
    speech_dur   = speech_end - speech_start

    # Silence gaps between words: 5% of speech_dur / n_gaps
    words        = phrase_def['words']
    syl_w        = phrase_def['syl_weights']
    n_words      = len(words)
    gap_dur      = speech_dur * 0.04   # 4% per inter-word gap
    total_gap    = gap_dur * (n_words - 1)
    word_dur_total = speech_dur - total_gap
    total_syl    = sum(syl_w)

    phonemes_out = []
    pid = 1
    t = speech_start

    for wi, ((word, phones), sw) in enumerate(zip(words, syl_w)):
        w_dur   = word_dur_total * sw / total_syl
        ph_dur  = w_dur / len(phones)
        w_start = t
        w_end   = t + w_dur

        for pi, ph in enumerate(phones):
            ph_start = t
            ph_end   = t + ph_dur
            phonemes_out.append({
                'phoneme_id'          : f'p{pid:03d}',
                'phoneme'             : ph,
                'word'                : word,
                'word_index'          : wi,
                'phoneme_index_in_word': pi,
                'start'               : round(ph_start, 6),
                'end'                 : round(ph_end,   6),
                'word_start'          : round(w_start,  6),
                'word_end'            : round(w_end,    6),
                'word_attention'      : 0.0,
                'word_start_std'      : 0.02,
                'word_end_std'        : 0.02,
            })
            t += ph_dur
            pid += 1

        # Inter-word gap
        if wi < n_words - 1:
            t += gap_dur

    timeline = {
        'phrase'              : phrase_def['display'],
        'speaker'             : speaker_id,
        'n_utterances'        : n_recs,
        'n_valid'             : n_recs,
        'n_alignment_sources' : n_recs,
        'avg_duration_sec'    : round(avg_dur, 6),
        'transcript'          : phrase_def['display'] + '.',
        'phonemes'            : phonemes_out,
        'source'              : 'batch_pipeline_estimated',
    }
    return timeline


# ─────────────────────────────────────────────────────────────
# FOLDER STRUCTURE
# ─────────────────────────────────────────────────────────────
def make_dirs(phrase_key, phrase_idx):
    """Create structured output directories for this phrase."""
    base = os.path.join(RESULTS_DIR,
                        f'phrase{phrase_idx}_{phrase_key}')
    dirs = {
        'base'     : base,
        'timelines': os.path.join(base, 'timelines'),
        'csv'      : os.path.join(base, 'csv'),
        'plots'    : os.path.join(base, 'plots'),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────
def run(dry_run=False):
    print('='*65)
    print('  ECAPA-TDNN XAI Batch Pipeline')
    print(f'  Phrases: {len(PHRASES)}  |  Speakers: {SPEAKERS}')
    print('='*65)

    # ── Phase 1: Generate all timelines ──────────────────────
    print('\n── Phase 1: Building Timelines ──')
    timeline_map = {}   # (phrase_key, speaker_id) → path

    for pi, pdef in enumerate(PHRASES, 1):
        pkey = pdef['key']
        dirs = make_dirs(pkey, pi)

        for spk in SPEAKERS:
            out_tl = os.path.join(dirs['timelines'],
                                  f'{spk}_timeline.json')
            existing = EXISTING_TIMELINES.get((pkey, spk))

            if existing and os.path.exists(existing):
                # Copy existing aligned timeline
                shutil.copy2(existing, out_tl)
                print(f'  [copy]  phrase{pi}/{pkey} / {spk}  →  (existing alignment)')
            else:
                # Generate estimated timeline from recording durations
                recs = get_recordings(pkey, spk)
                if not recs:
                    print(f'  [SKIP]  phrase{pi}/{pkey} / {spk}  — no recordings found')
                    continue
                avg_dur = compute_avg_duration(recs)
                tl = generate_timeline(pdef, spk, avg_dur, len(recs))
                if not dry_run:
                    with open(out_tl, 'w') as f:
                        json.dump(tl, f, indent=2)
                n_ph = len(tl['phonemes'])
                print(f'  [gen]   phrase{pi}/{pkey} / {spk}'
                      f'  avg_dur={avg_dur:.2f}s  n_ph={n_ph}'
                      f'  n_recs={len(recs)}')

            timeline_map[(pkey, spk)] = out_tl

    # ── Phase 2: Run final_figure.py for each combo ───────────
    print('\n── Phase 2: Generating Figures + CSVs ──')

    total = sum(1 for (pk,spk) in timeline_map)
    done  = 0
    failed= []

    # Lazy-import to avoid loading model until needed
    import subprocess

    for pi, pdef in enumerate(PHRASES, 1):
        pkey = pdef['key']
        dirs = make_dirs(pkey, pi)

        for spk in SPEAKERS:
            csv_path = os.path.join(dirs['csv'], f'{spk}_{pkey}_xai_analysis.csv')
            fig_path = os.path.join(dirs['plots'], f'{spk}_{pkey}_final_figure.png')

            if os.path.exists(fig_path) and os.path.exists(csv_path):
                print(f"  ▶  ({done+1}/{total}) {pkey.split('_')[0]}/{spk} (already exists, skipping)")
                continue

            tl_path = timeline_map.get((pkey, spk))
            if not tl_path or (not dry_run and not os.path.exists(tl_path)):
                print(f'  [SKIP] {pkey}/{spk} — no timeline')
                continue

            done += 1
            label = f'({done}/{total}) phrase{pi}/{spk}'
            print(f'\n  ▶  {label}')

            if dry_run:
                print(f'     [DRY-RUN] would run final_figure.py'
                      f' {spk} {pkey} {tl_path} {dirs["plots"]} {dirs["csv"]}')
                continue

            # Run final_figure.py directly (already inside WSL)
            cmd = [
                'python3', 'xai_reddots/scripts/final_figure.py',
                spk,
                '--phrase-clean', pkey,
                '--timeline',     tl_path,
                '--output-dir',   dirs['plots'],
                '--csv-dir',      dirs['csv'],
            ]
            t0 = time.time()
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=os.getcwd())
            elapsed = time.time() - t0

            if result.returncode == 0:
                print(f'     ✓  Done in {elapsed:.0f}s')
                # Print the key metrics from stdout
                for line in result.stdout.strip().split('\n'):
                    if any(k in line for k in ['Top-3:', 'Concentration:', 'Peak:', 'Pearson']):
                        print(f'       {line.strip()}')
            else:
                print(f'     ✗  FAILED (rc={result.returncode})')
                print(f'       {result.stderr.strip()[-300:]}')
                failed.append((pkey, spk))

    # ── Phase 3: Summary ─────────────────────────────────────
    print('\n' + '='*65)
    print('  Pipeline complete')
    print(f'  Processed: {done} combinations')
    if failed:
        print(f'  Failed: {failed}')

    print('\n  Output structure:')
    for pi, pdef in enumerate(PHRASES, 1):
        pkey = pdef['key']
        base = os.path.join(RESULTS_DIR, f'phrase{pi}_{pkey}')
        for subdir in ['timelines', 'csv', 'plots']:
            full = os.path.join(base, subdir)
            if os.path.exists(full):
                files = os.listdir(full)
                print(f'  {full}')
                for fn in sorted(files):
                    size = os.path.getsize(os.path.join(full, fn))
                    print(f'    {fn}  ({size//1024} KB)')
    print('='*65)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without running')
    args = parser.parse_args()
    run(dry_run=args.dry_run)
