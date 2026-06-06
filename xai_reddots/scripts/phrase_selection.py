import os
import csv

def main():
    script_path = 'Reddots/infos/script.txt'
    pcm_base = 'Reddots/pcm'
    output_csv = 'xai_reddots/metadata/phrase_groups.csv'
    
    import sys
    target_phrase = "My voice is my password"
    target_speaker = sys.argv[1] if len(sys.argv) > 1 else "m0002"
    
    matching_records = []
    
    print(f"Scanning {script_path} for phrase: '{target_phrase}' by speaker {target_speaker}")
    
    with open(script_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            parts = line.split(';')
            if len(parts) != 2: continue
            
            session_info, transcript = parts
            
            # Extract speaker from session_info, e.g. 20150425072659180_m0001_31
            info_parts = session_info.split('_')
            if len(info_parts) >= 2:
                speaker_id = info_parts[1]
                if speaker_id == target_speaker and transcript.lower() == target_phrase.lower():
                    # Construct probable path to PCM file.
                    # RedDots structure: Reddots/pcm/m0001/20150425072659180_m0001_31.pcm
                    pcm_path = os.path.join(pcm_base, speaker_id, session_info + ".pcm")
                    
                    # We check if it exists (though for XAI generation we might just record it)
                    if os.path.exists(pcm_path):
                        matching_records.append({
                            'recording_id': session_info,
                            'speaker_id': speaker_id,
                            'phrase': transcript,
                            'pcm_path': pcm_path
                        })

    print(f"Found {len(matching_records)} matching recordings.")
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['recording_id', 'speaker_id', 'phrase', 'pcm_path'])
        writer.writeheader()
        writer.writerows(matching_records)
        
    print(f"Saved to {output_csv}")

    # Write mock word alignments to word_alignment.csv for the selected speaker's recordings
    align_csv = 'xai_reddots/metadata/word_alignment.csv'
    mock_alignments = [
        {"word": "My", "start_time": "0.37", "end_time": "0.50"},
        {"word": "voice", "start_time": "0.57", "end_time": "0.88"},
        {"word": "is", "start_time": "0.99", "end_time": "1.08"},
        {"word": "my", "start_time": "1.14", "end_time": "1.28"},
        {"word": "password", "start_time": "1.35", "end_time": "1.88"}
    ]
    with open(align_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['recording_id', 'word', 'start_time', 'end_time'])
        writer.writeheader()
        for rec in matching_records:
            rec_id = rec['recording_id']
            for align in mock_alignments:
                writer.writerow({
                    'recording_id': rec_id,
                    'word': align['word'],
                    'start_time': align['start_time'],
                    'end_time': align['end_time']
                })
    print(f"Saved mock word alignments to {align_csv}")

if __name__ == "__main__":
    main()
