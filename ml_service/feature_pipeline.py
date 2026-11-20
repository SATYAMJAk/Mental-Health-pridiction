import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

def generate_behavior_narrative(row):
    """
    Synthesizes a rich behavioral textual description from student tabular attributes
    for Sentence Transformer and NLP feature extraction.
    """
    return (
        f"A {row['Age']} year old {row['Gender']} student at the {row['Academic_Level']} level living in {row['Country']}. "
        f"Primary social media platform is {row['Most_Used_Platform']} used predominantly for {row['Purpose_Of_Use']}. "
        f"Daily screen engagement is {row['Avg_Daily_Usage_Hours']} hours with {row['Daily_Unlocks']} phone unlocks. "
        f"Spends {row['Study_Hours']} hours studying, {row['Physical_Activity_Hours']} hours on physical activities, "
        f"and averages {row['Sleep_Hours_Per_Night']} hours of sleep per night."
    )

class HybridFeaturePipeline:
    def __init__(self, use_sbert=True):
        self.use_sbert = use_sbert
        self.sbert_model = None
        self.tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=150, sublinear_tf=True)
        self.scaler = StandardScaler()
        self.numerical_cols = ['Age', 'Avg_Daily_Usage_Hours', 'Daily_Unlocks', 'Study_Hours', 'Physical_Activity_Hours', 'Sleep_Hours_Per_Night']
        self.categorical_cols = ['Gender', 'Country', 'Academic_Level', 'Most_Used_Platform', 'Purpose_Of_Use']
        self.cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.feature_selector = None

    def _get_sbert_model(self):
        if self.sbert_model is None and self.use_sbert:
            try:
                from sentence_transformers import SentenceTransformer
                self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            except Exception as e:
                print(f"Warning loading SentenceTransformer: {e}")
                self.sbert_model = None
        return self.sbert_model

    def extract_f1_sbert(self, texts):
        model = self._get_sbert_model()
        if model is not None:
            embeddings = model.encode(texts, show_progress_bar=False, batch_size=128)
            return np.array(embeddings, dtype=np.float32)
        else:
            from sklearn.feature_extraction.text import HashingVectorizer
            hv = HashingVectorizer(n_features=384, alternate_sign=False)
            return hv.transform(texts).toarray().astype(np.float32)

    def fit(self, df):
        texts = [generate_behavior_narrative(row) for _, row in df.iterrows()]
        self.tfidf_vectorizer.fit(texts)
        self.scaler.fit(df[self.numerical_cols])
        self.cat_encoder.fit(df[self.categorical_cols])
        return self

    def extract_all_features(self, df, precomputed_f1=None):
        texts = [generate_behavior_narrative(row) for _, row in df.iterrows()]
        
        # 1. f1
        if precomputed_f1 is not None:
            f1_feats = precomputed_f1
        else:
            f1_feats = self.extract_f1_sbert(texts)
            
        # 2. f2
        f2_feats = self.tfidf_vectorizer.transform(texts).toarray().astype(np.float32)
        
        # 3. combined
        comb_feats = np.hstack([f1_feats, f2_feats])
        
        # 4. numerical & categorical
        num_feats = self.scaler.transform(df[self.numerical_cols]).astype(np.float32)
        cat_feats = self.cat_encoder.transform(df[self.categorical_cols]).astype(np.float32)
        
        # 5. f3 raw
        f3_raw = np.hstack([f1_feats, f2_feats, num_feats, cat_feats])
        
        return {
            'f1': f1_feats,
            'f2': f2_feats,
            'combined': comb_feats,
            'f3_raw': f3_raw
        }

    def transform(self, df, mode='f3'):
        res = self.extract_all_features(df)
        if mode == 'f1':
            return res['f1']
        elif mode == 'f2':
            return res['f2']
        elif mode == 'combined':
            return res['combined']
        elif mode == 'f3':
            return res['f3_raw']
        raise ValueError(f"Unknown mode: {mode}")

    def save(self, filepath):
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath):
        return joblib.load(filepath)
