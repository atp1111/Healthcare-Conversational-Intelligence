## Multimodal Healthcare Intelligence

![Banner](https://img.shields.io/badge/Status-Production-success?style=for-the-badge)
![Tech](https://img.shields.io/badge/Stack-PySpark_|_Llama_3_|_Wav2Vec2_|_Azure-blue?style=for-the-badge)
![Impact](https://img.shields.io/badge/Impact-$31k_Annual_Savings-green?style=for-the-badge)

> **A multimodal AI pipeline that fuses acoustic emotion detection (Audio) with semantic sentiment analysis (Text) to automate quality assurance for patient interactions.**

---

## 🚀 Business Impact & Results

This system was deployed to process **50,000+ patient files**, replacing manual auditing with automated intelligence.

| Metric | Improvement | Context |
| :--- | :--- | :--- |
| **💰 Cost Savings** | **$31,000 / yr** | Reduced operational overhead for manual QA teams. |
| **⚡ Compute Efficiency** | **12% Reduction** | Optimized inference costs via PySpark batching & PyTorch threading. |
| **⏱️ Time Saved** | **10% Reduction** | Reduced manual call review time via automated Power BI dashboards. |
| **🔍 Scale** | **50k+ Files** | Successfully processed large-scale historical archives on Azure Databricks. |

---

## 🧠 System Architecture

The pipeline operates on a **Lakehouse Architecture** within Azure Databricks. It splits raw customer service calls into two parallel processing streams—Audio and Text—before fusing them for final analysis.

```mermaid
graph TD
    subgraph Ingestion_Layer
    A["Raw Audio Files<br/>(Azure Blob Storage)"] -->|Spark Read| B{Preprocessing}
    B -->|Filter Silence| C[Valid Segments]
    end

    subgraph Audio_Physics_Engine
    C -->|Threaded Slicing| D[Wav2Vec2 Model]
    D -->|Acoustic Feature Extraction| E[Emotion Logits]
    E -->|Softmax| F["Tone Score<br/>(Anger/Fear/Happy)"]
    end

    subgraph Semantic_Engine
    C -->|Speaker Diarization| G[Transcripts]
    G -->|Prompt Engineering| H["Llama 3 (70B)"]
    H -->|Batch Inference| I[Semantic Context]
    I -->|JSON Parsing| J["Compliance Flags<br/>(Billing/HIPAA)"]
    end

🛠️ Technical Walkthrough
Phase 1: The Audio Physics Engine (audio_emotion_spark.py)
Goal: Detect HOW the patient is speaking, not just what they are saying.

Processing thousands of hours of audio requires massive concurrency. We bypassed standard Python limitations by implementing Thread-Safe Model Loading within Spark workers.

Model: superb/wav2vec2-base-superb-er (Fine-tuned for Emotion Recognition).

Optimization: Custom ThreadPoolExecutor implementation to saturate GPU cores on Databricks clusters.

Safety: Implemented aggressive memory cleanup (GC + CUDA Cache) to prevent OOM errors during long-running batch jobs.

Python
# Snippet: Thread-Safe Lazy Loading to maximize GPU throughput
class ThreadSafeEmotionModel:
    def get_model_and_processor(self):
        if not hasattr(self._local, 'model'):
             # Lazy load only when thread requires it
             self._local.model = Wav2Vec2ForSequenceClassification.from_pretrained(...)
        return self._local.model
Phase 2: The Semantic Llama Engine (text_sentiment_spark.py)
Goal: Understand context, intent, and compliance risks.

We utilized Llama 3 (70B) via Databricks Model Serving. The challenge was handling non-deterministic LLM outputs at scale.

Prompt Engineering: Designed strict "JSON-only" system prompts to force structured outputs ({overall_sentiment, issue_resolved, billing_detected}).

Self-Healing Data: Built a Retry Loop that identifies malformed JSON responses and re-processes them in micro-batches using Delta Lake MERGE operations.

Phase 3: Visualization & Insight
Goal: Make the data actionable for clinicians and operational managers.

The resulting Delta Tables feed into a Power BI Dashboard that visualizes:

Sentiment Trends: Identifying spikes in "Anger" correlated with specific billing issues.

Agent Performance: Heatmaps showing call resolution rates vs. acoustic sentiment.

(Place your dashboard screenshot here) ![Dashboard Screenshot](img/dashboard_placeholder.png)

⚙️ Optimization Strategy
We didn't just build models; we optimized them for cost and scale.

Batch Processing vs. Streaming: Switched to PySpark batch processing for historical backfills, reducing cluster uptime by 12%.

Resource Management: Tuned OMP_NUM_THREADS and torch.set_num_threads(6) to align PyTorch internal parallelism with Spark Executor cores, eliminating context-switching overhead.

Cross-Functional Alignment: Collaborated with Clinicians to define "Negative Sentiment" in a healthcare context (e.g., distinguishing "Pain" from "Anger"), ensuring the model aligned with clinical workflows.

💻 Installation & Usage
Prerequisites
Azure Databricks (Runtime 13.0+ ML)

GPU Cluster (T4 or A10 recommended)

HuggingFace API Token

    subgraph Analytics_Layer
    F --> K["Delta Lake<br/>Fusion Table"]
    J --> K
    K --> L[Power BI Dashboard]
    end
