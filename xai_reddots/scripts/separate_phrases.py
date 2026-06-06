import os
import csv
import json
from collections import defaultdict

def main():
    script_path = 'Reddots/infos/script.txt'
    pcm_base = 'Reddots/pcm'
    output_dir = 'xai_reddots/metadata'
    separated_dir = os.path.join(output_dir, 'separated_phrases')
    
    os.makedirs(separated_dir, exist_ok=True)
    
    phrase_recordings = defaultdict(list)
    total_lines = 0
    
    print(f"Reading script lines from {script_path}...")
    
    with open(script_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(';')
            if len(parts) != 2:
                continue
            
            session_info, transcript = parts
            total_lines += 1
            
            # Extract speaker_id from session_info (e.g. 20150129213306229_m0001_31)
            info_parts = session_info.split('_')
            speaker_id = info_parts[1] if len(info_parts) >= 2 else "unknown"
            
            pcm_path = os.path.join(pcm_base, speaker_id, session_info + ".pcm")
            file_exists = os.path.exists(pcm_path)
            
            phrase_recordings[transcript].append({
                'recording_id': session_info,
                'speaker_id': speaker_id,
                'pcm_path': pcm_path,
                'file_exists': file_exists
            })
            
    print(f"Parsed {total_lines} total lines.")
    print(f"Found {len(phrase_recordings)} unique phrases.")
    
    # Classify phrases into Target Phrases (>= 30 recordings) and Free-Text Phrases (< 30 recordings)
    summary_data = []
    phrase_mapping = {}
    
    for phrase, recs in phrase_recordings.items():
        count = len(recs)
        phrase_type = "Target (Common)" if count >= 500 else ("Target (Minor)" if count >= 30 else "Free-Text")
        
        summary_data.append({
            'phrase': phrase,
            'recording_count': count,
            'phrase_type': phrase_type
        })
        
        phrase_mapping[phrase] = [r['recording_id'] for r in recs]
        
        # If it is a target/common phrase, output a separate CSV for it
        if count >= 30:
            # Clean filename
            safe_phrase = "".join([c if c.isalnum() else "_" for c in phrase.lower()]).strip('_')
            # Collapse multiple underscores
            while "__" in safe_phrase:
                safe_phrase = safe_phrase.replace("__", "_")
            
            csv_filename = f"{safe_phrase}.csv"
            csv_path = os.path.join(separated_dir, csv_filename)
            
            with open(csv_path, 'w', newline='', encoding='utf-8') as f_out:
                writer = csv.DictWriter(f_out, fieldnames=['recording_id', 'speaker_id', 'pcm_path', 'file_exists'])
                writer.writeheader()
                writer.writerows(recs)
                
    # Sort summary by recording count descending
    summary_data.sort(key=lambda x: x['recording_count'], reverse=True)
    
    # Save the phrase summary CSV
    summary_csv = os.path.join(output_dir, 'phrase_summary.csv')
    with open(summary_csv, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=['phrase', 'recording_count', 'phrase_type'])
        writer.writeheader()
        writer.writerows(summary_data)
        
    # Save the full mapping to JSON
    mapping_json = os.path.join(output_dir, 'phrase_to_recordings.json')
    with open(mapping_json, 'w', encoding='utf-8') as f_out:
        json.dump(phrase_mapping, f_out, indent=2)
        
    print(f"Phrase summary saved to {summary_csv}")
    print(f"Full phrase-to-recording JSON mapping saved to {mapping_json}")
    print(f"Individual CSVs for the {sum(1 for x in summary_data if x['recording_count'] >= 30)} target phrases saved under: {separated_dir}")

if __name__ == "__main__":
    main()
