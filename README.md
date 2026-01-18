# VocalVitals: Multimodal Healthcare Intelligence

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
    A[Raw Audio Files<br/>(Azure Blob Storage)] -->|Spark Read| B{Preprocessing}
    B -->|Filter Silence| C[Valid Segments]
    end

    subgraph Audio_Physics_Engine
    C -->|Threaded Slicing| D[Wav2Vec2 Model]
    D -->|Acoustic Feature Extraction| E[Emotion Logits]
    E -->|Softmax| F[Tone Score<br/>(Anger/Fear/Happy)]
    end

    subgraph Semantic_Engine
    C -->|Speaker Diarization| G[Transcripts]
    G -->|Prompt Engineering| H[Llama 3 (70B)]
    H -->|Batch Inference| I[Semantic Context]
    I -->|JSON Parsing| J[Compliance Flags<br/>(Billing/HIPAA)]
    end

    subgraph Analytics_Layer
    F --> K[Delta Lake<br/>Fusion Table]
    J --> K
    K --> L[Power BI Dashboard]
    end
