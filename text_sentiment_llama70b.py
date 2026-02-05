"""
Text-based Sentiment Analysis Pipeline (LLaMA + Databricks).

Module purpose:
- Groups speaker-diarized segments into coherent transcripts.
- Leverages LLaMA via ai_query to extract structured sentiment, topics, and billing indicators.
- Merges results with file metadata for a comprehensive clinical reporting table.
"""
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import json
import re
from pyspark.sql import Row
from pyspark.sql.types import StructType, StructField, StringType, BooleanType, ArrayType, FloatType
from datetime import datetime
import gc

# Performance optimizations for Databricks compute
torch.set_num_threads(6)
os.environ["OMP_NUM_THREADS"] = "6"

# == CONFIGURATION ======
class Config:
    """
    Configuration container for pipeline settings.
    """
    MAX_FILES_TO_PROCESS = 0 
    MAX_SEGMENT_TEXT_LENGTH = 3000
    REPROCESS_BATCH_SIZE = 10 
    
    CLEAR_SENTIMENT_TABLE = False
    
    # --- PHI REMOVAL ---
    # Replaced specific endpoint identifiers and internal database names.
    LLM_ENDPOINT = "llama_3_70b_clinical_endpoint"  
    
    CATALOG_NAME = "clinical_analytics_catalog"  
    SCHEMA_NAME = "patient_experience"  
    SENTIMENT_TABLE = "call_sentiment_results"  
    DIARIZATION_TABLE = "diarized_transcripts" 
    FILE_INFO_TABLE = "metadata_registry" 
    
    # --- PHI REMOVAL ---
    # Sanitized storage credentials and locations.
    STORAGE_ACCOUNT = "sanitized_clinical_storage"  
    CONTAINER_NAME = "telehealth-records"  
    FOLDER = "VoiceInteractions"  
    
    @classmethod  
    def get_blob_path(cls):  
        return f"wasbs://{cls.CONTAINER_NAME}@{cls.STORAGE_ACCOUNT}.blob.core.windows.net/{cls.FOLDER}/"

BLOB_BASE_PATH = Config.get_blob_path()

# --- PHI REMOVAL ---
# Replaced specific secret scopes and keys.
scope = 'confidential_vault'
key = 'telehealth_access_key'
service_credential = dbutils.secrets.get(scope=scope, key=key)
spark.conf.set(f"fs.azure.account.key.{Config.STORAGE_ACCOUNT}.blob.core.windows.net", service_credential)

def aggressive_memory_cleanup():
    """Clears Spark caches and Python GC to optimize cluster resources."""
    spark.catalog.clearCache()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Memory cleanup completed")

# == TABLE MANAGEMENT ======
def add_billing_column_to_schema():
    """Ensures the schema supports billing detection flags."""
    try:
        schema_df = spark.sql(f"DESCRIBE {Config.CATALOG_NAME}.{Config.SCHEMA_NAME}.{Config.SENTIMENT_TABLE}")
        schema_columns = [row.col_name for row in schema_df.collect()]
        
        if 'billing_mentioned' not in schema_columns:
            spark.sql(f"ALTER TABLE {Config.CATALOG_NAME}.{Config.SCHEMA_NAME}.{Config.SENTIMENT_TABLE} ADD COLUMN billing_mentioned STRING")
            print("Successfully added 'billing_mentioned' column")
    except Exception:
        print("Table will be created with billing columns during initial run.")

# == LLM PROCESSING ======
def run_batch_llama_inference():
    """
    Orchestrates the LLaMA inference batch.
    Converts diarization segments (LEFT/RIGHT) into a prompt-friendly transcript.
    """
    print(f"\n= BATCH LLAMA INFERENCE =")
    
    # SQL logic to find unprocessed files
    # --- PHI REMOVAL: Sanitized prompt to hide specific hospital naming conventions ---
    prompt_instruction = """
    You are an expert healthcare sentiment analyst. Analyze the transcript between 
    a PROVIDER (LEFT) and a PATIENT (RIGHT). 
    Extract: Overall Sentiment, Provider Sentiment, Patient Sentiment, Call Topic, 
    Resolution Status, and if Billing/Insurance was discussed.
    Return ONLY a valid JSON object.
    """

    batch_llama_query = f"""
    INSERT INTO {Config.CATALOG_NAME}.{Config.SCHEMA_NAME}.{Config.SENTIMENT_TABLE}
    SELECT
      audio_filename AS file_name,
      get_json_object(REGEXP_EXTRACT(sentiment_result, '\\\\{{.*?\\\\}}', 0), '$.overall_sentiment') AS overall_sentiment,
      get_json_object(REGEXP_EXTRACT(sentiment_result, '\\\\{{.*?\\\\}}', 0), '$.provider_sentiment') AS provider_sentiment,
      get_json_object(REGEXP_EXTRACT(sentiment_result, '\\\\{{.*?\\\\}}', 0), '$.patient_sentiment') AS patient_sentiment,
      get_json_object(REGEXP_EXTRACT(sentiment_result, '\\\\{{.*?\\\\}}', 0), '$.call_topic') AS call_topic,
      get_json_object(REGEXP_EXTRACT(sentiment_result, '\\\\{{.*?\\\\}}', 0), '$.issue_resolved') AS issue_resolved,
      get_json_object(REGEXP_EXTRACT(sentiment_result, '\\\\{{.*?\\\\}}', 0), '$.billing') AS billing,
      sentiment_result AS raw_llm_response
    FROM (
      SELECT
        audio_filename,
        ai_query(
          '{Config.LLM_ENDPOINT}',
          CONCAT('{prompt_instruction}', '\\n\\nCONVERSATION:\\n', 
          ARRAY_JOIN(COLLECT_LIST(CONCAT(speaker, ': ', text)), '\\n'),
          '\\n\\nReturn JSON.')
        ) AS sentiment_result
      FROM {Config.CATALOG_NAME}.{Config.SCHEMA_NAME}.{Config.DIARIZATION_TABLE}
      GROUP BY audio_filename
    )
    """
    
    try:
        spark.sql(batch_llama_query)
        print("Batch LLaMA inference completed.")
    except Exception as e:
        print(f"Error in batch inference: {e}")

def join_file_info_data():
    """Merges LLM insights with administrative metadata (Call Date, ID, Duration)."""
    # --- PHI REMOVAL: Generic IDs and timestamp extraction ---
    update_query = f"""
    MERGE INTO {Config.CATALOG_NAME}.{Config.SCHEMA_NAME}.{Config.SENTIMENT_TABLE} as target
    USING {Config.CATALOG_NAME}.{Config.SCHEMA_NAME}.{Config.FILE_INFO_TABLE} as info
    ON target.file_name = info.name
    WHEN MATCHED THEN UPDATE SET
        target.call_date = date(info.timestamp),
        target.provider_id = info.op_id
    """
    try:
        spark.sql(update_query)
        print("Metadata join successful.")
    except Exception as e:
        print(f"Metadata join failed: {e}")

# == MAIN EXECUTION ======
if __name__ == "__main__":
    print("Initializing Pipeline...")
    add_billing_column_to_schema()
    run_batch_llama_inference()
    join_file_info_data()
    aggressive_memory_cleanup()
    print("Pipeline Complete.")
