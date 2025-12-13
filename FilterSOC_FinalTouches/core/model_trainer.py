# core/model_trainer.py
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import numpy as np
from .config import Config


class ModelTrainer:
    def __init__(self):
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        self.model = None

    def train_model(self, X_train, y_train, X_test, y_test):
        """Trénovanie modelu"""
        self.logger.info("Začínam trénovanie modelu...")

        # Random Forest s optimalizovanými hyperparametrami pre malý dataset
        # core/model_trainer.py - ešte lepšie nastavenie
        self.model = RandomForestClassifier(
            n_estimators=300,  # Viac stromov
            max_depth=20,  # Hlbšie stromy
            min_samples_split=5,  # Menej rozdelení
            min_samples_leaf=2,  # Menšie listy
            max_features=0.7,  # 70% features na strom
            class_weight='balanced',  # 👈 Toto funguje skvele!
            random_state=42,
            bootstrap=True
        )

        self.logger.info("Trénujem Random Forest...")
        self.model.fit(X_train, y_train)

        # Evaluácia
        train_accuracy = self.model.score(X_train, y_train)
        test_accuracy = self.model.score(X_test, y_test)

        self.logger.info(f"Trénovacia presnosť: {train_accuracy:.4f}")
        self.logger.info(f"Testovacia presnosť: {test_accuracy:.4f}")

        # Detailný report
        y_pred = self.model.predict(X_test)
        self.logger.info("\nClassification Report:")
        self.logger.info(f"\n{classification_report(y_test, y_pred)}")

        return self.model

    def save_model(self):
        """Uloží natrénovaný model"""
        if self.model is not None:
            os.makedirs(self.config.MODELS_DIR, exist_ok=True)
            model_path = os.path.join(self.config.MODELS_DIR, 'trained_model.joblib')
            joblib.dump(self.model, model_path)
            self.logger.info(f"Model uložený do: {model_path}")

    def load_model(self):
        """Načíta natrénovaný model"""
        model_path = os.path.join(self.config.MODELS_DIR, 'trained_model.joblib')
        self.model = joblib.load(model_path)
        self.logger.info("Model načítaný")
        return self.model