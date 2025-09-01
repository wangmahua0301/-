import os
import csv
from pathlib import Path
from typing import List

import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration, GenerationConfig

# ======================================
MODEL_DIR = 'model/noise/詔安腔noise/large/checkpoint-35452'
PROCESSED_DIR = '原資料/驗證'
OUT_DIR = './data'
OUTPUT_FILE_HAN = '詔安腔.csv'
BATCH_SIZE = 16
LANGUAGE = 'zh'
TASK = 'transcribe'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PROCESSOR_MODEL = 'openai/whisper-large-v3'
VIDEO_EXTS = ['.mp4', '.avi', '.mkv', '.wav', '.mp3']
# ======================================


def get_all_video_files(directory: str) -> List[str]:
    video_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in VIDEO_EXTS):
                video_files.append(os.path.join(root, file))
    return sorted(video_files)


def load_wav_16k_mono(path: str):
    wav, sr = torchaudio.load(path)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = wav.mean(dim=0).numpy().astype('float32')
    return wav


def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(DEVICE)
    print(f"Using device: {device}\n")

    print('Loading ASR processor & model...')
    processor = WhisperProcessor.from_pretrained(PROCESSOR_MODEL, language=LANGUAGE, task=TASK)
    asr_model = WhisperForConditionalGeneration.from_pretrained(MODEL_DIR).to(device).eval()
    try:
        gen_cfg = GenerationConfig.from_pretrained(MODEL_DIR)
        asr_model.generation_config = gen_cfg
    except Exception:
        pass

    print(f"Scanning folder for media files: {PROCESSED_DIR} ...")
    files = get_all_video_files(PROCESSED_DIR)
    print(f"Found {len(files)} files.\n")

    rows_han = []
    audio_batch = []
    utt_id_batch = []

    n = 0
    with torch.no_grad():
        for file_path in files:
            try:
                utt_id = os.path.splitext(os.path.basename(file_path))[0]
                print(f"Loading {utt_id} ({file_path})...")
                wav = load_wav_16k_mono(file_path)
                audio_batch.append(wav)
                utt_id_batch.append(utt_id)

                if len(audio_batch) == BATCH_SIZE or file_path == files[-1]:
                    print(f"Processing batch of {len(audio_batch)} files...")

                    inputs = processor(audio_batch, sampling_rate=16000, return_tensors="pt").to(device)
                    generated_ids = asr_model.generate(inputs.input_features, language=LANGUAGE, task=TASK)
                    transcriptions = processor.batch_decode(generated_ids, skip_special_tokens=True)

                    for utt_id, trans in zip(utt_id_batch, transcriptions):
                        rows_han.append((utt_id, trans))

                    audio_batch = []
                    utt_id_batch = []

                    n += len(transcriptions)
                    if n % 20 == 0:
                        print(f"Processed {n} files...")

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    with open(out_dir / OUTPUT_FILE_HAN, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["錄音檔檔名", "辨認出之客語漢字"])
        for fn, pred in rows_han:
            writer.writerow([fn, pred])

    print(f"\nDone. Output saved to: {out_dir / OUTPUT_FILE_HAN}")


if __name__ == '__main__':
    main()
