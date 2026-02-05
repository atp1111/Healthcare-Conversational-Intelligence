"""
Audio emotion processing pipeline (wav2vec2 + Databricks).

Module purpose:
- Reads diarization segments, fetches audio from cloud storage.
- Slices segments, runs a wav2vec2 emotion classifier per segment (thread-safe).
- Writes per-segment emotion labels + confidences into a Spark table.
"""
import os
import torch
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm
import torch.nn.functional as F
import time
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import io
from pydub import AudioSegment
import asyncio
from functools import lru_cache
import json
import re
from pyspark.sql import Row
from pyspark.sql.types import StructType, StructField, StringType, BooleanType, ArrayType, FloatType
from datetime import datetime
import gc
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

class Config:
    """
    Configuration container for the emotion processing pipeline.
    """
    DELETE_EMOTION_TABLE_CONTENT = False 
    UPDATE_ALL_EMOTIONS = True 
    MAX_SEGMENT_TEXT_LENGTH = 350 # Roughly 65 words per segment

    # Threading settings
    MAX_WORKERS = 6 
    THREAD_BATCH_SIZE = 2 
    USE_THREADING = True 

    # Verbosity and Persistence
    VERBOSE_LOGGING = False 
    CHECKPOINT_FREQUENCY = 50 
    SAVE_FREQUENCY = 5 

    # Model settings
    EMOTION_MODEL_PATH = "/dbfs/FileStore/models/wav2vec2-emotion-recognition"

    # Audio processing constraints
    TARGET_SAMPLE_RATE = 16000 
    MIN_SEGMENT_LENGTH = 16000 # min 1 sec
    MAX_SEGMENT_LENGTH = 120000 # max 8 sec

    # --- PHI REMOVAL ---
    # Replaced internal organization catalog and schema names to protect 
    # proprietary infrastructure details.
    CATALOG_NAME = "healthcare_prod_catalog"
    SCHEMA_NAME = "analytics_schema"
    DIARIZATION_TABLE = "audio_diarization_results"
    EMOTION_TABLE = "audio_emotion_results"

    # --- PHI REMOVAL ---
    # Sanitized Azure Storage account and container details to prevent 
    # disclosure of internal data lake locations.
    STORAGE_ACCOUNT = "sanitized_storage_account"
    CONTAINER_NAME = "clinical-audio-storage"
    FOLDER = "ClinicalCalls"

    @classmethod
    def get_blob_path(cls):
        return f"wasbs://{cls.CONTAINER_NAME}@{cls.STORAGE_ACCOUNT}.blob.core.windows.net/{cls.FOLDER}/"

# Use config
BLOB_BASE_PATH = Config.get_blob_path()

# --- PHI REMOVAL ---
# Removed specific secret scopes and keys used at the internship site 
# to prevent unauthorized access to original data sources.
scope = 'confidential_vault_scope'
key = 'storage_access_key'
service_credential = dbutils.secrets.get(scope=scope, key=key)
spark.conf.set(f"fs.azure.account.key.{Config.STORAGE_ACCOUNT}.blob.core.windows.net", service_credential)

# Set global thread limits for stability
torch.set_num_threads(6)
os.environ["OMP_NUM_THREADS"] = "6"

device = "cuda" if torch.cuda.is_available() else "cpu"

class ThreadSafeEmotionModel:
    """
    Thread-local loader for the wav2vec2 emotion model and processor.
    Lazily loads a model/processor per thread to ensure thread safety during parallel inference.
    """
    def __init__(self, model_path):
        self.model_path = model_path
        self._local = threading.local()
        self._lock = threading.Lock()
        
    def get_model_and_processor(self):
        if not hasattr(self._local, 'model'):
            with self._lock:
                # Load models within a thread lock to prevent race conditions during initialization
                self._local.processor = Wav2Vec2Processor.from_pretrained(self.model_path)
                self._local.model = Wav2Vec2ForSequenceClassification.from_pretrained(self.model_path)
                self._local.model = self._local.model.cpu()
                self._local.model.eval()
        return self._local.model, self._local.processor

try:
    thread_safe_emotion_model = ThreadSafeEmotionModel(Config.EMOTION_MODEL_PATH)
    emotion_model_loaded = True
except Exception as e:
    print(f"Model Load Error: {e}")
    emotion_model_loaded = False

# Mapping output indices to clinical emotion labels
label_map = {0: 'angry', 1: 'calm', 2: 'upset', 3: 'fearful', 4: 'happy', 5: 'sad', 6: 'neutral', 7: 'surprised'}

def enterprise_memory_cleanup():
    """
    Performs periodic memory cleanup to prevent buffer overflows during large-scale audio processing.
    """
    for i in range(2):
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(enterprise_memory_cleanup, 'call_count'):
        enterprise_memory_cleanup.call_count += 1
    else:
        enterprise_memory_cleanup.call_count = 1
    if enterprise_memory_cleanup.call_count % 15 == 0:
        spark.catalog.clearCache()

emotion_schema = StructType([
    StructField("audio_filename", StringType(), True),
    StructField("speaker", StringType(), True),
    StructField("text", StringType(), True),
    StructField("start", FloatType(), True),
    StructField("end", FloatType(), True),
    StructField("emotion_label", StringType(), True),
    StructField("confidence", FloatType(), True),
    StructField("processing_timestamp", StringType(), True)
])

def validate_audio_segment(audio_chunk, min_length=16000, max_length=80000):
    """Validates signal quality and length to ensure meaningful inference results."""
    if len(audio_chunk) < min_length: return False, "Too short"
    if len(audio_chunk) > max_length: return False, "Too long"
    if np.all(audio_chunk == 0): return False, "Silent segment"
    rms = np.sqrt(np.mean(audio_chunk**2))
    if rms < 0.001: return False, "Signal too quiet"
    return True, "Valid"

def fetch_audio_data_from_blob_storage_with_preprocessing(file_name):
    """Fetches binary audio from cloud storage, converts to numpy, and resamples."""
    try:
        audio_path = f"{BLOB_BASE_PATH}{file_name}"
        audio_df = spark.read.format("binaryFile").load(audio_path)
        audio_row = audio_df.collect()[0]
        if audio_row and audio_row['content']:
            audio_bytes = audio_row['content']
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
            audio_array = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
            
            if audio_segment.channels == 2:
                audio_array = audio_array.reshape((-1, 2))
            
            original_sr = audio_segment.frame_rate
            if original_sr != Config.TARGET_SAMPLE_RATE:
                audio_array = librosa.resample(audio_array, orig_sr=original_sr, target_sr=Config.TARGET_SAMPLE_RATE)
            return audio_array, Config.TARGET_SAMPLE_RATE
        return None, None
    except Exception:
        return None, None

def prepare_batch_segments(segments_info, stereo_audio):
    """Slices audio based on diarization timestamps and speaker identity."""
    valid_segments, segment_metadata = [], []
    for segment_info in segments_info:
        start_time, end_time = float(segment_info['start']), float(segment_info['end'])
        speaker = segment_info.get('speaker', 'PROVIDER') # --- PHI REMOVAL: Generic Speaker Label ---
        mono_s, mono_e = int(start_time * Config.TARGET_SAMPLE_RATE), int(end_time * Config.TARGET_SAMPLE_RATE)
        
        if mono_e <= len(stereo_audio) and mono_s >= 0:
            if stereo_audio.ndim == 2:
                # Select channel based on diarized speaker
                audio_chunk = stereo_audio[mono_s:mono_e, 0] if speaker.upper() == 'LEFT' else stereo_audio[mono_s:mono_e, 1]
            else:
                audio_chunk = stereo_audio[mono_s:mono_e]
            
            is_valid, reason = validate_audio_segment(audio_chunk, Config.MIN_SEGMENT_LENGTH, Config.MAX_SEGMENT_LENGTH)
            if is_valid:
                valid_segments.append(audio_chunk)
                segment_metadata.append({'start': start_time, 'end': end_time, 'speaker': speaker, 'text': segment_info.get('text', '')})
            else:
                segment_metadata.append({'start': start_time, 'end': end_time, 'speaker': speaker, 'text': segment_info.get('text', ''), 'skip_reason': reason})
    return valid_segments, segment_metadata

def process_segment_batch_threaded(segment_batch_data):
    """Runs batch inference on a single thread using the shared model instance."""
    segment_batch, thread_id = segment_batch_data
    try:
        model, processor = thread_safe_emotion_model.get_model_and_processor()
        batch_results = []
        for audio_chunk in segment_batch:
            if audio_chunk.dtype != np.float32: audio_chunk = audio_chunk.astype(np.float32)
            max_val = np.abs(audio_chunk).max()
            if max_val > 1.0: audio_chunk = audio_chunk / max_val
            
            inputs = processor(audio_chunk, sampling_rate=Config.TARGET_SAMPLE_RATE, return_tensors="pt", padding=True, truncation=True)
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = F.softmax(logits / 1.5, dim=-1) # Temperature scaling for better confidence calibration
                conf = round(float(torch.max(probs).item()), 2)
                label = label_map.get(torch.argmax(probs, dim=-1).item(), "neutral")
                batch_results.append({'emotion_label': label, 'confidence': conf})
        return batch_results
    except Exception:
        return [{'emotion_label': 'neutral', 'confidence': 0.0} for _ in segment_batch]

def process_emotion_multithread(audio_segments, max_workers=None):
    """Coordinates the thread pool to execute parallel inference across multiple audio segments."""
    if not audio_segments: return []
    max_workers = max_workers or Config.MAX_WORKERS
    thread_batches = [(audio_segments[i:i + Config.THREAD_BATCH_SIZE], i) for i in range(0, len(audio_segments), Config.THREAD_BATCH_SIZE)]
    
    batch_results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {executor.submit(process_segment_batch_threaded, b): idx for idx, b in enumerate(thread_batches)}
        for future in tqdm(as_completed(future_to_batch), total=len(future_to_batch), desc="Inferring Emotions"):
            batch_results[future_to_batch[future]] = future.result()
            
    all_results = []
    for idx in sorted(batch_results.keys()): all_results.extend(batch_results[idx])
    return all_results

def update_emotions_to_new_table():
    """
    Main orchestration function for the pipeline.
    Identifies new records, processes audio in parallel, and appends to the target Delta table.
    """
    if not emotion_model_loaded: return []

    # --- PHI REMOVAL: Sanitized SQL queries to hide internal schema structure ---
    files_query = f"""
        SELECT DISTINCT d.audio_filename FROM {Config.CATALOG_NAME}.{Config.SCHEMA_NAME}.{Config.DIARIZATION_TABLE} d
        LEFT JOIN {Config.CATALOG_NAME}.{Config.SCHEMA_NAME}.{Config.EMOTION_TABLE} e ON d.audio_filename = e.audio_filename
        WHERE e.audio_filename IS NULL
    """
    file_list = [row.audio_filename for row in spark.sql(files_query).collect()]
    
    if not file_list: return []

    all_emotion_results = []
    total_files = len(file_list)
    
    for idx, filename in enumerate(file_list):
        # Log progress at defined checkpoints
        if idx % Config.CHECKPOINT_FREQUENCY == 0:
            print(f"Progress: {idx}/{total_files} files processed.")

        audio, sr = fetch_audio_data_from_blob_storage_with_preprocessing(filename)
        if audio is None: continue
        
        # Retrieve segments associated with this specific file
        file_segments_df = spark.sql(f"SELECT * FROM {Config.CATALOG_NAME}.{Config.SCHEMA_NAME}.{Config.DIARIZATION_TABLE} WHERE audio_filename = '{filename}'")
        file_segments = [row.asDict() for row in file_segments_df.collect()]

        valid_segments, segment_metadata = prepare_batch_segments(file_segments, audio)
        
        if valid_segments:
            emotion_results = process_emotion_multithread(valid_segments)
            
            res_idx = 0
            for metadata in segment_metadata:
                res = {'audio_filename': filename, 'speaker': metadata['speaker'], 'text': metadata['text'], 'start': metadata['start'], 'end': metadata['end'], 'processing_timestamp': datetime.now().isoformat()}
                
                if 'skip_reason' in metadata:
                    res.update({'emotion_label': 'neutral', 'confidence': 0.0})
                else:
                    res.update(emotion_results[res_idx])
                    res_idx += 1
                all_emotion_results.append(res)
        
        # Intermediate save to prevent data loss in long-running jobs
        if (idx + 1) % Config.SAVE_FREQUENCY == 0 and all_emotion_results:
            emotion_df = spark.createDataFrame(all_emotion_results, schema=emotion_schema)
            emotion_df.write.mode("append").saveAsTable(f"{Config.CATALOG_NAME}.{Config.SCHEMA_NAME}.{Config.EMOTION_TABLE}")
            all_emotion_results = []
            enterprise_memory_cleanup()

    # Final save for remaining results
    if all_emotion_results:
        emotion_df = spark.createDataFrame(all_emotion_results, schema=emotion_schema)
        emotion_df.write.mode("append").saveAsTable(f"{Config.CATALOG_NAME}.{Config.SCHEMA_NAME}.{Config.EMOTION_TABLE}")

    return file_list

if __name__ == "__main__":
    if Config.UPDATE_ALL_EMOTIONS:
        update_emotions_to_new_table()
        print("Pipeline Execution Successful.")
