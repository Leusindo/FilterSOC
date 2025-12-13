# core/classifier.py
import logging
import numpy as np
import pandas as pd
import joblib
import os
import re
from typing import List, Dict, Tuple, Any
from tqdm import tqdm

from .feature_extractor import HybridFeatureExtractor
from .model_trainer import ModelTrainer
from .data_processor import DataProcessor
from .config import Config


class NewsClassifier:
    """
    Hlavná trieda pre klasifikáciu titulkov do kategórií dezinformácií
    Používa Slovak BERT + TF-IDF features a Random Forest
    """

    def __init__(self, use_bert: bool = True, use_tfidf: bool = True):
        """
        Inicializácia klasifikátora

        Args:
            use_bert: Použiť BERT features
            use_tfidf: Použiť TF-IDF features
        """
        self.config = Config()
        self.logger = logging.getLogger(__name__)

        # Nastavenie feature extractoru
        self.config.USE_BERT = use_bert
        self.config.USE_TFIDF = use_tfidf

        # Inicializácia komponentov
        self.feature_extractor = HybridFeatureExtractor()
        self.model_trainer = ModelTrainer()
        self.data_processor = DataProcessor()

        self.is_trained = False
        self.is_loaded = False

        self.logger.info(f"Classifier inicializovaný - BERT: {use_bert}, TF-IDF: {use_tfidf}")
        self.logger.info(f"Používané zariadenie: {self.config.DEVICE}")

    def train(self, enable_augmentation: bool = True) -> Dict[str, Any]:
        """
        Kompletný pipeline trénovania

        Args:
            enable_augmentation: Povoliť rozšírenie dát

        Returns:
            Dictionary s výsledkami trénovania
        """
        self.logger.info("=== ZAČÍNAM TRÉNOVANIE KLASIFIKÁTORA ===")

        try:
            # 1. Načítanie a preprocessing dát
            df = self.data_processor.load_data()
            self.logger.info(f"Načítaných {len(df)} trénovacích príkladov")

            # 2. Data augmentation (voliteľné)
            if enable_augmentation:
                df = self.data_processor.augment_data(df)
                self.logger.info(f"Po augmentácii: {len(df)} príkladov")

            # 3. Preprocessing
            X, y, original_labels = self.data_processor.preprocess_data(df)

            # 4. Rozdelenie dát
            X_train, X_test, y_train, y_test = self.data_processor.split_data(X, y)

            # 5. Feature extraction
            self.logger.info("Začínam feature extraction...")
            self.feature_extractor.fit(X_train)
            X_train_features = self.feature_extractor.transform(X_train)
            X_test_features = self.feature_extractor.transform(X_test)

            self.logger.info(f"Trénovacie features: {X_train_features.shape}")
            self.logger.info(f"Testovacie features: {X_test_features.shape}")

            # 6. Trénovanie modelu
            trained_model = self.model_trainer.train_model(
                X_train_features, y_train, X_test_features, y_test
            )

            # 7. Uloženie modelov
            self._save_models()

            # 8. Evaluácia
            results = self._evaluate_training(X_test_features, y_test, original_labels)

            self.is_trained = True
            self.is_loaded = True

            self.logger.info("=== TRÉNOVANIE ÚSPEŠNE DOKONČENÉ ===")
            return results

        except Exception as e:
            self.logger.error(f"Chyba pri trénovaní: {e}")
            raise

    def _save_models(self):
        """Uloží všetky natrénované modely a komponenty"""
        self.logger.info("Ukladám modely...")

        os.makedirs(self.config.MODELS_DIR, exist_ok=True)

        # Uloženie feature extractoru
        self.feature_extractor.save()

        # Uloženie klasifikátora
        self.model_trainer.save_model()

        # Uloženie label encoderu
        self.data_processor.save_label_encoder()

        # Uloženie konfigurácie
        config_path = os.path.join(self.config.MODELS_DIR, 'training_config.joblib')
        joblib.dump({
            'use_bert': self.config.USE_BERT,
            'use_tfidf': self.config.USE_TFIDF,
            'categories': self.config.CATEGORIES,
            'feature_dimensions': {
                'bert_original': self.config.BERT_EMBEDDING_DIM,
                'bert_reduced': self.config.REDUCED_BERT_DIM,
                'tfidf': self.config.TFIDF_MAX_FEATURES
            }
        }, config_path)

        self.logger.info(f"Všetky modely uložené do: {self.config.MODELS_DIR}")

    def _evaluate_training(self, X_test, y_test, original_labels) -> Dict[str, Any]:
        """Komplexná evaluácia trénovania"""
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        import seaborn as sns
        import matplotlib.pyplot as plt

        # Predikcie
        y_pred = self.model_trainer.model.predict(X_test)
        y_pred_proba = self.model_trainer.model.predict_proba(X_test)

        # Metriky
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Výpočet confidence scores
        confidence_scores = np.max(y_pred_proba, axis=1)
        avg_confidence = np.mean(confidence_scores)

        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'average_confidence': avg_confidence,
            'test_set_size': len(X_test),
            'feature_dimensions': X_test.shape[1],
            'categories': self.data_processor.label_encoder.classes_.tolist()
        }

        # Log výsledkov
        self.logger.info(f"Konečná presnosť: {accuracy:.4f}")
        self.logger.info(f"Priemerná istota: {avg_confidence:.4f}")
        self.logger.info(f"Počet features: {X_test.shape[1]}")

        return results

    def load_models(self) -> bool:
        """
        Načítanie natrénovaných modelov

        Returns:
            True ak sa načítanie podarilo, inak False
        """
        try:
            self.logger.info("Načítavam natrénované modely...")

            # Kontrola existencie modelov
            required_files = [
                'trained_model.joblib',
                'label_encoder.joblib',
                'training_config.joblib'
            ]

            for file in required_files:
                if not os.path.exists(os.path.join(self.config.MODELS_DIR, file)):
                    self.logger.error(f"Chýbajúci súbor: {file}")
                    return False

            # Načítanie komponentov
            self.feature_extractor.load()
            self.model_trainer.load_model()
            self.data_processor.load_label_encoder()

            # Načítanie konfigurácie
            config_path = os.path.join(self.config.MODELS_DIR, 'training_config.joblib')
            training_config = joblib.load(config_path)

            self.is_loaded = True
            self.logger.info("✅ Všetky modely úspešne načítané")
            self.logger.info(f"📊 Kategórie: {', '.join(training_config['categories'])}")

            return True

        except Exception as e:
            self.logger.error(f"❌ Chyba pri načítavaní modelov: {e}")
            return False

    def predict(self, text: str, return_confidence: bool = True) -> Tuple[str, Dict[str, float]]:
        """
        Predikcia kategórie pre jeden text

        Args:
            text: Text titulku na klasifikáciu
            return_confidence: Vrátiť pravdepodobnosti pre všetky kategórie

        Returns:
            Tuple: (predpovedaná_kategória, pravdepodobnosti)
        """
        if not self.is_loaded and not self.load_models():
            raise ValueError("Modely nie sú načítané a nepodarilo sa ich načítať!")

        # Čistenie textu
        cleaned_text = self._clean_text(text)

        if not cleaned_text:
            raise ValueError("Text je prázdny po vyčistení!")

        # Feature extraction
        try:
            features = self.feature_extractor.transform([cleaned_text])
        except Exception as e:
            self.logger.error(f"Chyba pri feature extraction: {e}")
            # Fallback na základnú klasifikáciu podľa kľúčových slov
            return self._fallback_prediction(text)

        # Predikcia
        prediction = self.model_trainer.model.predict(features)[0]
        probabilities = self.model_trainer.model.predict_proba(features)[0]

        # Decode label
        predicted_label = self.data_processor.label_encoder.inverse_transform([prediction])[0]

        # Pravdepodobnosti pre všetky kategórie
        prob_dict = {
            category: float(prob) for category, prob in zip(
                self.data_processor.label_encoder.classes_,
                probabilities
            )
        }

        # Logovanie vysokej istoty
        max_prob = max(probabilities)
        if max_prob > 0.8:
            self.logger.debug(f"Vysoká istota ({max_prob:.3f}) pre: '{text}' -> {predicted_label}")

        return predicted_label, prob_dict

    def predict_batch(self, texts: List[str], show_progress: bool = True) -> List[Dict[str, Any]]:
        """
        Hromadná predikcia pre viacero textov

        Args:
            texts: Zoznam textov na klasifikáciu
            show_progress: Zobraziť progress bar

        Returns:
            Zoznam výsledkov pre každý text
        """
        if not self.is_loaded and not self.load_models():
            raise ValueError("Modely nie sú načítané!")

        if not texts:
            return []

        # Čistenie textov
        cleaned_texts = [self._clean_text(text) for text in texts]
        valid_texts = [text for text in cleaned_texts if text]

        if not valid_texts:
            raise ValueError("Žiadny validný text po čistení!")

        # Feature extraction
        if show_progress:
            self.logger.info(f"Spracovávam {len(valid_texts)} textov...")

        features = self.feature_extractor.transform(valid_texts)

        # Predikcia
        predictions = self.model_trainer.model.predict(features)
        probabilities = self.model_trainer.model.predict_proba(features)

        # Decode labels
        predicted_labels = self.data_processor.label_encoder.inverse_transform(predictions)

        # Zostavenie výsledkov
        results = []
        iterator = zip(texts, predicted_labels, probabilities)

        if show_progress:
            iterator = tqdm(iterator, total=len(texts), desc="Klasifikácia")

        for original_text, label, probs in iterator:
            prob_dict = {
                category: float(prob) for category, prob in zip(
                    self.data_processor.label_encoder.classes_,
                    probs
                )
            }

            results.append({
                'original_text': original_text,
                'cleaned_text': self._clean_text(original_text),
                'predicted_category': label,
                'probabilities': prob_dict,
                'confidence': float(max(probs)),
                'is_high_confidence': max(probs) > 0.7
            })

        return results

    def _clean_text(self, text: str) -> str:
        """
        Základné čistenie textu

        Args:
            text: Vstupný text

        Returns:
            Vyčistený text
        """
        if not isinstance(text, str):
            return ""

        # Odstránenie prebytočných bielych znakov
        text = re.sub(r'\s+', ' ', text.strip())

        # Odstránenie špeciálnych znakov (ponechá základné diakritiku)
        text = re.sub(r'[^\w\sáäčďéíľĺňóôřŕšťúýžÁÄČĎÉÍĽĹŇÓÔŘŔŠŤÚÝŽ\-\'",.!?;]', '', text)

        return text.lower()

    def _fallback_prediction(self, text: str) -> Tuple[str, Dict[str, float]]:
        """
        Záložná klasifikácia podľa kľúčových slov (ak feature extraction zlyhá)
        """
        self.logger.warning(f"Používam fallback klasifikáciu pre: {text}")

        text_lower = text.lower()

        # Kľúčové slová pre každú kategóriu
        keywords = {
            'clickbait': ['šokujúce', 'neuveríte', 'kliknite', 'zistíte', 'tajomstvo', 'odhalenie'],
            'false_news': ['zákaz', 'ruší', 'zatvorené', 'zakazuje', 'zakázal', 'povinné'],
            'conspiracy': ['tajné', 'elity', 'ovládajú', 'alien', 'mimozemšťan', 'bunkor'],
            'propaganda': ['jediná', 'záchrana', 'naša strana', 'lídra', 'úspechy'],
            'satire': ['zrušený', 'zakázal dážď', 'povinné nosenie', 'bryndzové halušky'],
            'misleading': ['všetkých', 'zázračne', 'úplne', 'absolútne', '100%'],
            'biased': ['neschopný', 'zlyhal', 'podvodník', 'hlúpy', 'kritizuje'],
            'legitimate': ['schválila', 'oznámil', 'vydalo', 'objavili', 'otvorí']
        }

        # Počet zhôd pre každú kategóriu
        scores = {category: 0 for category in self.config.CATEGORIES}

        for category, words in keywords.items():
            for word in words:
                if word in text_lower:
                    scores[category] += 1

        # Nájdenie kategórie s najvyšším skóre
        max_score = max(scores.values())
        if max_score > 0:
            predicted_category = max(scores, key=scores.get)
        else:
            predicted_category = 'legitimate'  # Default

        # Simulácia pravdepodobností
        total_score = sum(scores.values()) or 1
        prob_dict = {
            category: score / total_score for category, score in scores.items()
        }

        return predicted_category, prob_dict

    def get_model_info(self) -> Dict[str, Any]:
        """
        Získanie informácií o natrénovanom modeli

        Returns:
            Dictionary s informáciami o modeli
        """
        if not self.is_loaded:
            return {"status": "Modely nie sú načítané"}

        info = {
            "status": "Natrénovaný a načítaný",
            "feature_extractor": {
                "uses_bert": self.config.USE_BERT,
                "uses_tfidf": self.config.USE_TFIDF,
                "bert_model": self.config.BERT_MODEL_NAME
            },
            "classifier": {
                "type": "RandomForest",
                "n_estimators": self.config.N_ESTIMATORS
            },
            "categories": self.config.CATEGORIES,
            "device": self.config.DEVICE
        }

        if hasattr(self.model_trainer, 'model') and self.model_trainer.model is not None:
            info["classifier"]["n_features"] = (
                self.model_trainer.model.n_features_in_
                if hasattr(self.model_trainer.model, 'n_features_in_')
                else "Unknown"
            )

        return info

    def evaluate_custom_data(self, texts: List[str], true_labels: List[str]) -> Dict[str, Any]:
        """
        Evaluácia modelu na vlastných dátach

        Args:
            texts: Zoznam textov
            true_labels: Skutočné kategórie

        Returns:
            Výsledky evaluácie
        """
        from sklearn.metrics import classification_report, accuracy_score

        if not self.is_loaded:
            raise ValueError("Modely musia byť načítané!")

        if len(texts) != len(true_labels):
            raise ValueError("Texty a labels musia mať rovnakú dĺžku!")

        # Predikcia
        results = self.predict_batch(texts, show_progress=True)
        predicted_labels = [result['predicted_category'] for result in results]

        # Metriky
        accuracy = accuracy_score(true_labels, predicted_labels)
        report = classification_report(true_labels, predicted_labels, output_dict=True)

        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(true_labels, predicted_labels,
                              labels=self.data_processor.label_encoder.classes_)

        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'sample_size': len(texts),
            'predictions': results
        }


# Jednoduchá factory funkcia pre rýchle vytvorenie klasifikátora
def create_classifier(use_bert: bool = True, use_tfidf: bool = True) -> NewsClassifier:
    """
    Rýchle vytvorenie klasifikátora

    Args:
        use_bert: Použiť BERT features
        use_tfidf: Použiť TF-IDF features

    Returns:
        Inicializovaný NewsClassifier
    """
    return NewsClassifier(use_bert=use_bert, use_tfidf=use_tfidf)


# Testovacia funkcia
def test_classifier():
    """Testovanie klasifikátora"""
    import logging
    logging.basicConfig(level=logging.INFO)

    print("🧪 Testovanie NewsClassifier...")

    # Vytvorenie klasifikátora
    classifier = create_classifier()

    # Informácie o modeli
    info = classifier.get_model_info()
    print(f"📊 Model info: {info}")

    # Testovacie príklady
    test_texts = [
        "Šokujúce odhalenie v Bratislave!",
        "Vláda schválila nový rozpočet",
        "Tajné spolky ovládajú parlament",
        "Nový zákon zakazuje bicykle"
    ]

    print("\n🔍 Testovacie predikcie:")
    for text in test_texts:
        try:
            category, probs = classifier.predict(text)
            print(f"  '{text}' -> {category} (istota: {max(probs.values()):.3f})")
        except Exception as e:
            print(f"  ❌ Chyba pre '{text}': {e}")


if __name__ == "__main__":
    test_classifier()