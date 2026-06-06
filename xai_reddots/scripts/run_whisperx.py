import os
import csv
import whisperx
import subprocess

def convert_pcm_to_wav(pcm_path, wav_path):
    # WhisperX requires standard audio files, Reddots uses 16kHz 16-bit PCM
    cmd = [
        'ffmpeg', '-y', '-f', 's16le', '-ar', '16000', '-ac', '1', 
        '-i', pcm_path, wav_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def main():
    metadata_csv = 'xai_reddots/metadata/phrase_groups.csv'
    output_csv = 'xai_reddots/metadata/word_alignment.csv'
    temp_wav_dir = 'xai_reddots/temp_wavs'
    
    os.makedirs(temp_wav_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    recordings = []
    with open(metadata_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            recordings.append(row)
            
    # Load WhisperX models
    device = "cuda" 
    batch_size = 16 
    compute_type = "float16" 
    
    print("Loading WhisperX model...")
    model = whisperx.load_model("large-v2", device, compute_type=compute_type)
    model_a, metadata = whisperx.load_align_model(language_code="en", device=device)
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['recording_id', 'word', 'start_time', 'end_time'])
        writer.writeheader()
        
        for idx, rec in enumerate(recordings):
            rec_id = rec['recording_id']
            pcm_path = rec['pcm_path']
            wav_path = os.path.join(temp_wav_dir, f"{rec_id}.wav")
            
            convert_pcm_to_wav(pcm_path, wav_path)
            
            # 1. Transcribe with Whisper
            audio = whisperx.load_audio(wav_path)
            result = model.transcribe(audio, batch_size=batch_size)
            
            # 2. Align timestamps
            result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
            
            # Extract word segments
            for segment in result["segments"]:
                for word in segment["words"]:
                    # Sometimes whisperx returns words without timestamps if it's unsure
                    if 'start' in word and 'end' in word:
                        writer.writerow({
                            'recording_id': rec_id,
                            'word': word['word'].strip('.,?!'),
                            'start_time': f"{word['start']:.3f}",
                            'end_time': f"{word['end']:.3f}"
                        })
                        
            print(f"Processed {idx+1}/{len(recordings)}: {rec_id}")

if __name__ == "__main__":
    main()
