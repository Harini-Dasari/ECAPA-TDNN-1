import csv
import os

def main():
    metadata_csv = 'xai_reddots/metadata/phrase_groups.csv'
    output_csv = 'xai_reddots/metadata/word_alignment.csv'
    
    # words and rough mocked times
    # My: 0.37-0.50
    # voice: 0.57-0.88
    # is: 0.99-1.08
    # my: 1.14-1.28
    # password: 1.35-1.88
    words = [
        ('My', 0.37, 0.50),
        ('voice', 0.57, 0.88),
        ('is', 0.99, 1.08),
        ('my', 1.14, 1.28),
        ('password', 1.35, 1.88)
    ]
    
    with open(metadata_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        recordings = [row['recording_id'] for row in reader]
        
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['recording_id', 'word', 'start_time', 'end_time'])
        writer.writeheader()
        
        for rec_id in recordings:
            for w, s, e in words:
                writer.writerow({
                    'recording_id': rec_id,
                    'word': w,
                    'start_time': f"{s:.2f}",
                    'end_time': f"{e:.2f}"
                })
                
    print(f"Mock alignment generated for {len(recordings)} recordings.")

if __name__ == "__main__":
    main()
