# 🎬 Synesthesia: Visuo-Lingual Affective Recommender System (VLARS)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-F9AB00?logo=huggingface)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)

**Live Web App:** [Insert your Streamlit Cloud Link Here]

## 📌 Project Overview
Basic recommendation systems rely on collaborative filtering, which suffers from the "cold-start" problem and ignores real-time user context. **VLARS** is a multimodal AI system that solves this by acting as a contextual engine. It extracts physiological state via computer vision (facial emotion) and psychological state via NLP (text sentiment), fusing them to recommend content using a high-dimensional Semantic Vector Search.

*(Insert a screenshot of your web app UI here: `![App UI](assets/ui_screenshot.png)`)*

## 🧠 Core Architecture & Features

### 1. Multimodal Affective Inference
* **Visual Modality:** Utilizes a pre-trained **ResNet/VGG backend via DeepFace** to extract a probability distribution of 7 core micro-expressions from facial data.
* **Textual Modality:** Implements a knowledge-distilled Transformer (**DistilRoBERTa**) to map unstructured text into an aligned latent emotional space.

### 2. Late-Fusion Decision Engine
To resolve contradictions between modalities (e.g., sarcasm or stoicism), the system implements a dynamic Late-Fusion engine utilizing an $L_1$ normalized weighted heuristic. The text modality is given a higher tunable weight ($\alpha = 0.65$) to act as a contextual safeguard against visual misclassification.

### 3. Semantic Retrieval (RAG Architecture)
Instead of rigid genre-filtering, the system leverages a **MiniLM Sentence Transformer** to embed 5,000+ movie plot summaries into 384-dimensional dense vectors. Recommendations are retrieved in $O(\log N)$ time using **FAISS (Facebook AI Similarity Search)** via $L_2$ Euclidean distance.

## 📊 Quantitative Evaluation (Ablation Study)
To validate the Multimodal Late-Fusion architecture, an ablation study was conducted against single-modality baselines using complex emotional edge-cases (e.g., "Tears of Joy", Sarcasm).

*(Insert your terminal screenshot here: `![Ablation Study](assets/ablation_results.png)`)*

**Results:** The Late-Fusion architecture successfully acts as a fault-tolerant layer, strictly minimizing $L_2$ semantic error distance and improving contextual retrieval accuracy by up to **1.98%** compared to standard Vision-Only baselines.

## 🚀 How to Run Locally

1. Clone the repository:
   ```bash
   git clone [https://github.com/YourUsername/VLARS-Movie-Recommender.git](https://github.com/YourUsername/VLARS-Movie-Recommender.git)
   cd VLARS-Movie-Recommender
