# Healthcare-Conversational-Intelligence
# Multimodal Call Intelligence Platform

[![PySpark](https://img.shields.io/badge/PySpark-3.4-orange)](https://spark.apache.org/)
[![HuggingFace](https://img.shields.io/badge/Models-Wav2Vec2%20%2B%20Llama3-yellow)](https://huggingface.co/)
[![Azure](https://img.shields.io/badge/Cloud-Azure%20Databricks-blue)](https://azure.microsoft.com/)

A production-grade, multimodal AI pipeline designed to analyze enterprise call center interactions. This system fuses **Audio Analysis** (Wav2Vec2) and **Text Analysis** (Llama 3 70B) to provide a 360-degree view of customer sentiment, agent performance, and compliance.

## 💼 Business Impact

In high-volume call centers, manual QA can only audit ~2% of calls. This automated pipeline enables **100% coverage**, allowing the organization to:
* **Detect Churn Risk:** By correlating high-energy audio (Anger) with negative text sentiment.
* **Automate Compliance:** Detecting billing information and mandatory disclosures automatically.
* **Scale Efficiently:** Optimized to process thousands of audio hours using distributed Spark computing and GPU acceleration.

## 🏗 Architecture

The system operates on a Lakehouse architecture within Azure Databricks. It processes data in two parallel streams that merge for final analysis.

```mermaid
graph TD
    A[Raw Call Audio] -->|Speaker Diarization| B[Diarized Segments]
    
    subgraph Audio_Pipeline_Wav2Vec2
    B -->|Fetch Blob Data| C[Audio Slicer]
    C -->|Threaded Inference| D[Wav2Vec2 Emotion Model]
    D -->|Output| E[Acoustic Emotion Score]
    end
    
    subgraph Text_Pipeline_Llama3
    B -->|Aggregated Transcripts| F[Prompt Engineering]
    F -->|Batch AI Query| G[Llama 70B Inference]
    G -->|Output| H[Semantic Sentiment & Topics]
    end
    
    E --> I{Multimodal Fusion}
    H --> I
    I --> J[Final Dashboard / Delta Table]
