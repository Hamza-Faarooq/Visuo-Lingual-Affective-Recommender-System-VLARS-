import streamlit as st
import pandas as pd
import numpy as np
import cv2
import tempfile
import os
from PIL import Image
import faiss
import torch
from deepface import DeepFace
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from typing import Dict, Optional, Union, Tuple

# ==========================================
# 1. CORE PIPELINE CLASSES
# ==========================================

class VisualEmotionAnalyzer:
    def __init__(self, detector_backend='opencv'):
        self.detector_backend = detector_backend
    def analyze_image(self, image_input) -> Optional[Dict[str, float]]:
        try:
            res = DeepFace.analyze(img_path=image_input, actions=['emotion'],
                                   enforce_detection=True, detector_backend=self.detector_backend, silent=True)
            return res[0]['emotion'] if isinstance(res, list) else res['emotion']
        except Exception:
            return None

class TextEmotionAnalyzer:
    def __init__(self, model_name="j-hartmann/emotion-english-distilroberta-base"):
        device = 0 if torch.cuda.is_available() else -1
        try:
            self.classifier = pipeline("text-classification", model=model_name, device=device, top_k=None)
        except Exception:
            self.classifier = None
    def analyze_text(self, text_input: str) -> Optional[Dict[str, float]]:
        if not self.classifier or not text_input.strip(): return None
        try:
            raw = self.classifier(text_input)[0]
            return {entry['label']: entry['score'] for entry in raw}
        except Exception:
            return None

class MultimodalFusionEngine:
    def __init__(self, text_weight=0.65, vision_weight=0.35):
        tot = text_weight + vision_weight
        self.text_weight, self.vision_weight = text_weight / tot, vision_weight / tot
        self.map = {'happy': 'joy', 'sad': 'sadness', 'angry': 'anger', 'fear': 'fear',
                    'disgust': 'disgust', 'surprise': 'surprise', 'neutral': 'neutral'}
        self.emotions = list(self.map.values())

    def _align_vision(self, v_probs):
        aligned = {e: 0.0 for e in self.emotions}
        for k, v in v_probs.items():
            if self.map.get(k.lower()): aligned[self.map[k.lower()]] = v / 100.0
        return aligned

    def fuse_modalities(self, v_probs, t_probs):
        if not v_probs and not t_probs: raise ValueError("Both modalities missing.")
        fused = {e: 0.0 for e in self.emotions}

        if not v_probs: fused = t_probs.copy()
        elif not t_probs: fused = self._align_vision(v_probs)
        else:
            aligned_v = self._align_vision(v_probs)
            for e in self.emotions:
                fused[e] = (self.vision_weight * aligned_v.get(e, 0.0)) + (self.text_weight * t_probs.get(e, 0.0))

        tot = sum(fused.values())
        if tot > 0: fused = {k: v / tot for k, v in fused.items()}
        return max(fused, key=fused.get), fused

class SemanticAffectiveRecommender:
    def __init__(self, dataset_path: str):
        self.movies_df = pd.read_csv(dataset_path).dropna(subset=['overview', 'genres'])
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.movies_df['combined_features'] = self.movies_df['genres'] + " " + self.movies_df['overview']
        self.movie_embeddings = self.encoder.encode(self.movies_df['combined_features'].tolist(), convert_to_numpy=True)
        self.index = faiss.IndexFlatL2(self.encoder.get_sentence_embedding_dimension())
        self.index.add(self.movie_embeddings)

    def get_semantic_recommendations(self, core_mood, user_text, top_n=5):
        query = f"A movie with themes of {core_mood}. The viewer is feeling this way because: {user_text}"
        query_vector = self.encoder.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_vector, top_n)
        recs = self.movies_df.iloc[indices[0]].copy()
        recs['semantic_distance'] = distances[0]
        return recs[['title', 'genres', 'overview', 'semantic_distance']]

# ==========================================
# 2. STREAMLIT UI & SERVER LOGIC
# ==========================================
st.set_page_config(page_title="VLARS | Multimodal Recommender", layout="wide")

@st.cache_resource
def load_system():
    # Looks for the CSV in the exact same folder as app.py on the cloud server
    file_path = "tmdb_5000_movies.csv"
    
    if not os.path.exists(file_path):
        st.error(f"Dataset missing! Please ensure '{file_path}' is uploaded to the GitHub repository.")
        st.stop() 

    return (VisualEmotionAnalyzer(),
            TextEmotionAnalyzer(),
            SemanticAffectiveRecommender(file_path))

with st.spinner("Initializing AI Models & Vector Database (O(1) Load)..."):
    vision_agent, nlp_agent, recommender = load_system()

st.sidebar.title("⚙️ Engine Parameters")
t_weight = st.sidebar.slider("Text Modality Weight (α)", 0.0, 1.0, 0.65, 0.05)
fusion_agent = MultimodalFusionEngine(text_weight=t_weight, vision_weight=1.0 - t_weight)

st.title("🎬 Visuo-Lingual Affective Recommender System (VLARS)")

col1, col2 = st.columns(2)
with col1:
    img_buffer = st.camera_input("1. Visual Input (Take a picture)")
with col2:
    user_text = st.text_area("2. Textual Input (How are you feeling?)", height=150)

if st.button("Generate Contextual Recommendations", type="primary"):
    if img_buffer is None and not user_text:
        st.warning("Provide at least one input modality.")
    else:
        v_probs, t_probs = None, None
        with st.spinner("Executing Inference and FAISS Vector Search..."):
            if img_buffer:
                img_array = np.array(Image.open(img_buffer))
                cv2_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                    cv2.imwrite(tmp.name, cv2_img)
                    v_probs = vision_agent.analyze_image(tmp.name)
                os.remove(tmp.name)

            if user_text:
                t_probs = nlp_agent.analyze_text(user_text)

            try:
                mood, dist = fusion_agent.fuse_modalities(v_probs, t_probs)
                st.success(f"### Mathematical Core Mood: **{mood.upper()}**")
                st.bar_chart(dist)

                # st.markdown(" preconceived thoughts.") # Commented out potentially problematic line
                st.subheader("🍿 Semantic Matches via FAISS")

                # Fetch Vector Search Results
                recs = recommender.get_semantic_recommendations(mood, user_text if user_text else "No text provided.", 5)

                for _, row in recs.iterrows():
                    st.markdown(f"**{row['title']}** (L2 Distance: `{row['semantic_distance']:.2f}`)")
                    st.caption(f"Genres: {row['genres']}")
                    st.write(row['overview'])
                    st.markdown("---")
            except Exception as e:
                st.error(f"Pipeline Error: {e}")
