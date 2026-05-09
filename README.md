---
title: VLARS Recommender
sdk: streamlit
app_file: app.py
pinned: false
python_version: "3.10"
---

# 🎬 Synesthesia: Visuo-Lingual Affective Recommender System (VLARS)
 
[![Live Demo](https://img.shields.io/badge/Demo-HuggingFace-yellow?style=for-the-badge&logo=huggingface)](https://tm-vettel-vlars-recommender.hf.space/)
![Python](https://img.shields.io/badge/Python-3.10-blue.svg?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

**🚀 Live Web App:** [https://tm-vettel-vlars-recommender.hf.space/](https://tm-vettel-vlars-recommender.hf.space/)

## 📌 Project Overview
Traditional recommendation systems often suffer from the "cold-start" problem and fail to capture the immediate emotional state of a user. **VLARS** is a multimodal AI engine designed to solve this by fusing **physiological state** (facial expressions) and **psychological state** (textual sentiment) into a single affective query for high-fidelity movie recommendations.

## 🧠 Technical Architecture

### 1. Multimodal Affective Inference
*   **Visual Modality:** Uses a **ResNet/VGG** backbone (via DeepFace) to perform real-time facial emotion detection across 7 core micro-expressions.
*   **Textual Modality:** Implements a knowledge-distilled Transformer (**DistilRoBERTa**) to extract nuanced sentiment from unstructured user text.

### 2. Late-Fusion & Semantic Retrieval
*   **Fusion Engine:** Implements an $L_1$ normalized weighted heuristic to resolve modality contradictions (e.g., sarcasm), with a tunable $\alpha$ weight for textual priority.
*   **Vector Search:** Utilizes **FAISS (Facebook AI Similarity Search)** to query a 384-dimensional dense vector space of 5,000+ movies in $O(\log N)$ time.
*   **Embeddings:** Powered by a **MiniLM-L6 Sentence Transformer** for deep semantic understanding of movie plots.

## 📊 Quantitative Evaluation (Ablation Study)
The system was validated through an ablation study on complex emotional edge cases (e.g., "Tears of Joy", Stoicism). Results proved that the **Late-Fusion architecture** reduces semantic retrieval error ($L_2$ distance) by up to **1.98%** compared to unimodal baselines.

## 🛠️ Tech Stack
*   **Frameworks:** Streamlit, PyTorch, TensorFlow
*   **Vector DB:** FAISS
*   **Models:** DeepFace, Hugging Face Transformers (Sentence-Transformers)
*   **Environment:** Python 3.10, Linux (Debian)

## 🚀 How to Run Locally
1. Clone the repo: `git clone https://github.com/YourUsername/VLARS-Movie-Recommender.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Launch app: `streamlit run app.py`
