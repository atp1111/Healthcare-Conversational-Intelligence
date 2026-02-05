Multimodal LLM Sentiment Analysis for Healthcare
This repository implements a dual-stage multimodal pipeline for extracting deep emotional and sentiment-based insights from patient-provider audio and text data. This work was developed during my Data Science Internship at OhioHealth to automate the detection of patient needs and improve clinical follow-up accuracy.

🏗️ Core Components
1. Acoustic Emotion Intelligence (audio_emotion_wav2vec2.py)
A thread-safe, distributed pipeline for high-scale acoustic analysis.

Model: Fine-tuned Wav2Vec2 for sequence classification.

Process: Handles audio ingestion from Azure Blob Storage, resampling to 16kHz, and per-segment emotional classification (e.g., Angry, Calm, Upset, Fearful).

Scale: Optimized for Databricks using a ThreadPoolExecutor and thread-local model loading to maintain high throughput without memory overhead.

2. Generative Sentiment & Reasoning (text_sentiment_llama70b.py)
A text-based reasoning engine that processes structured transcripts.

Model: LLaMA-3-70B via Databricks ai_query.

Functionality: Beyond simple sentiment, this engine extracts specific clinical markers:

Call Topic Detection: Automatically categorizes clinical issues (1–2 keywords).

Resolution Tracking: Determines if the patient's concern was successfully addressed.

Billing/Administrative Detection: Flags segments where insurance or billing information was discussed.

🛠️ Tech Stack
Infrastructure: Databricks, Spark SQL, Azure Blob Storage.

Deep Learning: PyTorch, Hugging Face Transformers.

Audio Handling: Librosa, Pydub, SoundFile.

🏥 Clinical Impact
By combining acoustic signals (how something is said) with LLM reasoning (what is actually said), this system provides a 360-degree view of patient interactions. It enables healthcare providers to prioritize follow-ups based on both emotional distress and unresolved medical concerns.
