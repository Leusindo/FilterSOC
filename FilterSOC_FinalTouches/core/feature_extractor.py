# core/feature_extractor.py
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from typing import List
import joblib
import os
from .config import Config


class SlovakBERTFeatureExtractor:
    def __init__(self):
        self.config = Config()
        self.logger = logging.getLogger(__name__)

        self.logger.info(f"Načítavam Slovak BERT: {self.config.BERT_MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.BERT_MODEL_NAME)
        self.model = AutoModel.from_pretrained(self.config.BERT_MODEL_NAME)
        self.model.to(self.config.DEVICE)
        self.model.eval()

        self.logger.info(f"BERT model načítaný na: {self.config.DEVICE}")

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Extrahuje BERT embeddings"""
        self.logger.info(f"Extrahujem BERT embeddings pre {len(texts)} textov...")

        all_embeddings = []

        for i in range(0, len(texts), self.config.BATCH_SIZE):
            batch_texts = texts[i:i + self.config.BATCH_SIZE]

            # Tokenizácia
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.config.MAX_LENGTH,
                return_tensors="pt"
            )

            # Presun na správne zariadenie
            encoded = {k: v.to(self.config.DEVICE) for k, v in encoded.items()}

            # Forward pass bez gradientov
            with torch.no_grad():
                outputs = self.model(**encoded)
                # Použijeme mean pooling
                embeddings = torch.mean(outputs.last_hidden_state, dim=1)
                all_embeddings.append(embeddings.cpu().numpy())

        embeddings_array = np.vstack(all_embeddings)
        self.logger.info(f"BERT embeddings shape: {embeddings_array.shape}")
        return embeddings_array


class HybridFeatureExtractor:
    def __init__(self):
        self.config = Config()
        self.logger = logging.getLogger(__name__)

        self.use_bert = self.config.USE_BERT
        self.use_tfidf = self.config.USE_TFIDF

        if self.use_tfidf:
            self.tfidf = TfidfVectorizer(
                max_features=self.config.TFIDF_MAX_FEATURES,
                ngram_range=(1, 2),
                min_df=2
            )

        if self.use_bert:
            self.bert_extractor = SlovakBERTFeatureExtractor()
            if self.config.REDUCED_BERT_DIM > 0:
                self.bert_svd = TruncatedSVD(n_components=self.config.REDUCED_BERT_DIM)

        self.is_fitted = False

    def fit(self, texts: List[str]):
        """Trénovanie feature extractorov"""
        self.logger.info("Trénujem hybridný feature extractor...")

        if self.use_tfidf:
            self.logger.info("Trénujem TF-IDF...")
            self.tfidf.fit(texts)

        if self.use_bert:
            self.logger.info("Extrahujem BERT embeddings pre trénovacie dáta...")
            bert_features = self.bert_extractor.get_embeddings(texts)

            if self.config.REDUCED_BERT_DIM > 0:
                self.logger.info(f"Redukujem BERT dimenzie na {self.config.REDUCED_BERT_DIM}")
                self.bert_svd.fit(bert_features)

        self.is_fitted = True
        self.logger.info("Feature extractor úspešne natrénovaný")
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        """Transformácia textov na features"""
        if not self.is_fitted:
            raise ValueError("Feature extractor musí byť najprv natrénovaný!")

        features_list = []

        if self.use_tfidf:
            tfidf_features = self.tfidf.transform(texts).toarray()
            features_list.append(tfidf_features)
            self.logger.debug(f"TF-IDF features shape: {tfidf_features.shape}")

        if self.use_bert:
            bert_features = self.bert_extractor.get_embeddings(texts)

            if self.config.REDUCED_BERT_DIM > 0:
                bert_features = self.bert_svd.transform(bert_features)

            features_list.append(bert_features)
            self.logger.debug(f"BERT features shape: {bert_features.shape}")

        # Kombinácia features
        combined_features = np.hstack(features_list)
        self.logger.info(f"Kombinované features shape: {combined_features.shape}")

        return combined_features

    def save(self):
        """Uloží natrénovaný feature extractor"""
        os.makedirs(self.config.MODELS_DIR, exist_ok=True)

        if self.use_tfidf:
            joblib.dump(self.tfidf, os.path.join(self.config.MODELS_DIR, 'tfidf_vectorizer.joblib'))

        if self.use_bert and self.config.REDUCED_BERT_DIM > 0:
            joblib.dump(self.bert_svd, os.path.join(self.config.MODELS_DIR, 'bert_svd.joblib'))

        self.logger.info("Feature extractor uložený")

    def load(self):
        """Načíta natrénovaný feature extractor"""
        if self.use_tfidf:
            self.tfidf = joblib.load(os.path.join(self.config.MODELS_DIR, 'tfidf_vectorizer.joblib'))

        if self.use_bert and self.config.REDUCED_BERT_DIM > 0:
            self.bert_svd = joblib.load(os.path.join(self.config.MODELS_DIR, 'bert_svd.joblib'))

        self.is_fitted = True
        self.logger.info("Feature extractor načítaný")