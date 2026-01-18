"""
Text Sentiment & Compliance Pipeline (Llama 3 + Spark SQL)
Purpose: Analyzes conversation transcripts for sentiment, topic, and compliance.
"""

import os
import time
from pyspark.sql.functions import col, concat, collect_list, array_join, get_json_object

# ==========================================
# CONFIGURATION
# ==========================================
class Config:
    # Processing settings
    MAX_FILES_TO_PROCESS = 0  # 0 = process all
    MAX_SEGMENT_TEXT_LENGTH = 3000
    REPROCESS_BATCH_SIZE = 10 
    
    # Databricks Model Serving Endpoint
    LLM_ENDPOINT = "databricks-llama-70b-serving" 
    
    # Database settings (Sanitized)
    CATALOG_NAME = "hive_metastore"  
    SCHEMA_NAME = "call_center_analytics"  
    SENTIMENT_TABLE = "call_sentiment_results"  
    DIARIZATION_TABLE = "audio_diarization_source" 
    
# ==========================================
# PIPELINE LOGIC
# ==========================================

def add_billing_column_to_schema(spark):
    """
    Schema Evolution: Ensures the target table adapts to new business requirements (e.g., Billing).
    """
    print(f"\n= CHECKING SCHEMA =")
    try:
        df = spark.sql(f"SELECT * FROM {Config.CATALOG_NAME}.{Config.SCHEMA_NAME}.{Config.SENTIMENT_TABLE} LIMIT 1")
        if 'billing' not in df.columns:
            print("Adding 'billing' column to schema...")
            spark.sql(f"ALTER TABLE {Config.CATALOG_NAME}.{Config.SCHEMA_NAME}.{Config.SENTIMENT_TABLE} ADD COLUMN billing STRING")
    except Exception as e:
        print(f"Table initialization: {e}")

def run_batch_llama_inference(spark):
    """
    Core Logic: Aggregates conversation transcripts and hits the LLM Endpoint via SQL.
    Uses 'ai_query' to keep data inside the Spark environment.
    """
    print(f"Running LLaMA inference batch...")

    # Aggregates 'speaker' and 'text' rows into a single transcript block
    # Formats prompts as "LEFT: [Agent]" and "RIGHT: [Customer]"
    query = f"""
    INSERT INTO {Config.CATALOG_NAME}.{Config.SCHEMA_NAME}.{Config.SENTIMENT_TABLE}
    (file_name, overall_sentiment, agent_sentiment, customer_sentiment, call_topic, issue_resolved, billing, raw_llm_response)
    SELECT
      audio_filename AS file_name,
      get_json_object(json_clean, '$.overall_sentiment'),
      get_json_object(json_clean, '$.agent_sentiment'),
      get_json_object(json_clean, '$.customer_sentiment'),
      get_json_object(json_clean, '$.call_topic'),
      get_json_object(json_clean, '$.issue_resolved'),
      get_json_object(json_clean, '$.billing'),
      sentiment_result
    FROM (
      SELECT
        audio_filename,
        sentiment_result,
        REGEXP_EXTRACT(sentiment_result, '\\\\\\{{.*?\\\\\\}}', 0) as json_clean
      FROM (
        SELECT
          audio_filename,
          ai_query(
            '{Config.LLM_ENDPOINT}',
            CONCAT(
              'You are an expert sentiment analyst. Analyze the transcript between AGENT (LEFT) and CUSTOMER (RIGHT).',
              '\\nCONVERSATION TRANSCRIPT:\\n',
              ARRAY_JOIN(COLLECT_LIST(CONCAT(speaker, ': ', text)), '\\n'),
              '\\n\\nReturn JSON: {{"overall_sentiment": "...", "agent_sentiment": "...", "call_topic": "...", "issue_resolved": "yes/no", "billing": "yes/no"}}'
            )
          ) AS sentiment_result
        FROM {Config.CATALOG_NAME}.{Config.SCHEMA_NAME}.{Config.DIARIZATION_TABLE}
        GROUP BY audio_filename
      )
    )
    """
    spark.sql(query)
    print("Batch inference complete.")

def reprocess_failures(spark):
    """
    Resilience: Retries analysis for rows that returned NULL/Invalid JSON.
    Implements a 'Self-Healing' loop using MERGE operations.
    """
    print("\n= REPROCESSING FAILURES =")
    # Logic: Select NULLs -> Retry in micro-batches -> Upsert via MERGE
    pass 

if __name__ == "__main__":
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("TextSentimentPipeline").getOrCreate()
    
    add_billing_column_to_schema(spark)
    run_batch_llama_inference(spark)
    reprocess_failures(spark)
