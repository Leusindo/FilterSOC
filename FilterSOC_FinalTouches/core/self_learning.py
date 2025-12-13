# core/self_learning.py
import pandas as pd
import numpy as np
import logging
import joblib
import os
from datetime import datetime
from typing import List, Dict, Tuple
from .config import Config


# core/self_learning.py - PRIDAJ TÚTO METÓDU:

class SelfLearningSystem:
    def __init__(self, classifier, aggressive_learning=True):
        self.classifier = classifier
        self.config = Config()
        self.logger = logging.getLogger(__name__)

        # 🚀 EXTREME AGGRESSIVE LEARNING
        self.confidence_threshold = 0.0  # Učí sa zo všetkého
        self.buffer_size = 10  # Veľmi rýchle pretrénovanie
        self.aggressive_learning = True  # Vždy zapnuté

        self.learning_buffer = []
        self.learning_file = "data/self_learning/learning_data.csv"
        self.backup_file = "data/self_learning/learning_data_backup.csv"

        os.makedirs("data/self_learning", exist_ok=True)
        self._initialize_learning_data()

        self.logger.info("🚀 EXTREME AGGRESSIVE LEARNING - Učí sa zo všetkých predikcií!")

    def _initialize_learning_data(self):
        """Inicializácia learning dát s potrebnými stĺpcami"""
        try:
            # Skús načítať existujúce dáta
            learning_data = self.load_learning_data()

            # Ak súbor existuje ale chýbajú stĺpce, pridaj ich
            if not learning_data.empty:
                required_columns = ['text', 'category', 'confidence', 'timestamp', 'verified', 'processed']
                missing_columns = [col for col in required_columns if col not in learning_data.columns]

                if missing_columns:
                    self.logger.info(f"🔧 Pridávam chýbajúce stĺpce: {missing_columns}")
                    for col in missing_columns:
                        if col == 'verified':
                            learning_data[col] = False
                        elif col == 'processed':
                            learning_data[col] = False
                        elif col == 'timestamp':
                            learning_data[col] = datetime.now().isoformat()
                        else:
                            learning_data[col] = ''

                    learning_data.to_csv(self.learning_file, index=False)
                    self.logger.info("✅ Learning data inicializované s potrebnými stĺpcami")

        except Exception as e:
            self.logger.info("📝 Vytváram nový learning data súbor")
            # Vytvor prázdny DataFrame s potrebnými stĺpcami
            empty_df = pd.DataFrame(columns=['text', 'category', 'confidence', 'timestamp', 'verified', 'processed'])
            empty_df.to_csv(self.learning_file, index=False)

    def predict_with_learning(self, text: str) -> Tuple[str, Dict[str, float], bool]:
        """
        Predikcia s možnosťou učenia z vysokospoľahlivých predikcií

        Returns:
            Tuple: (kategória, pravdepodobnosti, pridané_do_účenia)
        """
        try:
            # Štandardná predikcia
            category, probabilities = self.classifier.predict(text)
            confidence = max(probabilities.values())

            # Kontrola či pridáme do učenia
            added_to_learning = False
            if confidence > self.confidence_threshold:
                self._add_to_learning_buffer(text, category, confidence)
                added_to_learning = True
                self.logger.info(f"🧠 Pridané do učenia: '{text}' -> {category} ({confidence:.3f})")

            return category, probabilities, added_to_learning

        except Exception as e:
            self.logger.error(f"Chyba v predict_with_learning: {e}")
            return "unknown", {}, False

    def _add_to_learning_buffer(self, text: str, category: str, confidence: float):
        """Pridanie príkladu do učiaceho bufferu"""
        learning_example = {
            'text': text,
            'category': category,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'verified': False,  # Môže byť neskôr overené používateľom
            'processed': False  # Označenie či bol použitý pri pretrénovaní
        }

        self.learning_buffer.append(learning_example)

        # Auto-ukladanie každých 10 príkladov
        if len(self.learning_buffer) >= 10:
            self._save_learning_data()

        # Kontrola či netreba pretrénovať
        if len(self.learning_buffer) >= self.buffer_size:
            self.logger.info(f"🔄 Buffer plný ({len(self.learning_buffer)} príkladov), navrhujem pretrénovanie")

    def _save_learning_data(self):
        """Uloženie učiacich dát do CSV"""
        try:
            if not self.learning_buffer:
                return

            # Načítanie existujúcich dát
            try:
                existing_df = pd.read_csv(self.learning_file)
            except FileNotFoundError:
                # Vytvorenie nového DataFrame s potrebnými stĺpcami
                existing_df = pd.DataFrame(
                    columns=['text', 'category', 'confidence', 'timestamp', 'verified', 'processed'])

            # Pridanie nových dát
            new_df = pd.DataFrame(self.learning_buffer)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)

            # Odstránenie duplikátov
            combined_df = combined_df.drop_duplicates(subset=['text'])

            # Uloženie
            combined_df.to_csv(self.learning_file, index=False)
            combined_df.to_csv(self.backup_file, index=False)  # Záloha

            self.logger.info(f"💾 Uložených {len(new_df)} self-learning príkladov")
            self.learning_buffer.clear()

        except Exception as e:
            self.logger.error(f"Chyba pri ukladaní learning dát: {e}")

    def load_learning_data(self) -> pd.DataFrame:
        """Načítanie existujúcich učiacich dát"""
        try:
            df = pd.read_csv(self.learning_file)

            # Kontrola a doplnenie chýbajúcich stĺpcov
            required_columns = ['text', 'category', 'confidence', 'timestamp', 'verified', 'processed']
            for col in required_columns:
                if col not in df.columns:
                    if col == 'verified':
                        df[col] = False
                    elif col == 'processed':
                        df[col] = False
                    elif col == 'timestamp':
                        df[col] = datetime.now().isoformat()
                    else:
                        df[col] = ''

            self.logger.info(f"📖 Načítaných {len(df)} self-learning príkladov")
            return df

        except FileNotFoundError:
            self.logger.info("Self-learning súbor neexistuje, vrátený prázdny DataFrame")
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Chyba pri načítavaní learning dát: {e}")
            return pd.DataFrame()

    # core/self_learning.py - OPRAV metódu retrain_with_learning_data():

    def retrain_with_learning_data(self) -> bool:
        """
        Pretrénovanie modelu s novými učiacimi dátami
        """
        try:
            self.logger.info("🔄 Začínam pretrénovanie s self-learning dátami...")

            # Načítanie pôvodných a nových dát
            original_data = pd.read_csv(self.config.DATA_PATH)
            learning_data = self.load_learning_data()

            if learning_data.empty:
                self.logger.info("ℹ️ Žiadne learning dáta pre pretrénovanie")
                return False

            # 🔽 DÔLEŽITÁ OPRAVA: Zmeň filtrovanie!
            # PÔVODNÉ (zlé):
            # verified_mask = (
            #     (learning_data.get('verified', pd.Series([False] * len(learning_data))) == True) |
            #     (learning_data.get('confidence', pd.Series([0] * len(learning_data))) > 0.9)
            # )

            # OPRAVENÉ (jednoduchšie):
            # 1. Zober všetky príklady ktoré sú overené ALEBO majú vysokú istotu
            verified_mask = pd.Series([True] * len(learning_data))  # Všetky príklady!

            # Alebo ešte lepšie:
            # verified_mask = (
            #     learning_data.get('verified', pd.Series([False] * len(learning_data)).fillna(False)) |
            #     (learning_data.get('confidence', pd.Series([0.0] * len(learning_data)).fillna(0.0)) > 0.25)
            # )

            verified_data = learning_data[verified_mask]

            if len(verified_data) == 0:
                self.logger.info("ℹ️ Žiadne overené dáta pre pretrénovanie")
                return False

            self.logger.info(f"📊 Pretrénujem s {len(verified_data)} overenými príkladmi")

            # Vytvorenie zlúčeného datasetu
            new_data = pd.DataFrame({
                'title': verified_data['text'],
                'category': verified_data['category']
            })

            combined_data = pd.concat([original_data, new_data], ignore_index=True)

            # Uloženie zálohy pôvodného modelu
            self._backup_current_models()

            # 🔽 OPRAVA: Resetuj feature extractor pred pretrénovaním
            self.classifier.feature_extractor.is_fitted = False

            # Pretrénovanie klasifikátora
            results = self.classifier.train(enable_augmentation=False)

            self.logger.info(f"✅ Pretrénovanie úspešné! Nová presnosť: {results['accuracy']:.3f}")

            # Označenie použitých príkladov ako spracovaných
            self._mark_processed_examples(verified_data)

            return True

        except Exception as e:
            self.logger.error(f"❌ Chyba pri pretrénovaní: {e}")
            self._restore_backup_models()  # Obnova zálohy
            return False

    def _backup_current_models(self):
        """Zálohovanie aktuálnych modelov"""
        import shutil
        import glob

        try:
            model_files = glob.glob(os.path.join(self.config.MODELS_DIR, "*"))
            for file_path in model_files:
                if os.path.isfile(file_path):
                    filename = os.path.basename(file_path)
                    shutil.copy2(file_path, f"data/backup_models/{filename}")

            self.logger.info("💾 Aktuálne modely zazálohované")
        except Exception as e:
            self.logger.error(f"Chyba pri zálohovaní modelov: {e}")

    def _restore_backup_models(self):
        """Obnova modelov zo zálohy"""
        import shutil
        import glob

        try:
            backup_files = glob.glob("data/backup_models/*")
            for file_path in backup_files:
                if os.path.isfile(file_path):
                    filename = os.path.basename(file_path)
                    shutil.copy2(file_path, os.path.join(self.config.MODELS_DIR, filename))

            self.logger.info("🔄 Modely obnovené zo zálohy")
            self.classifier.load_models()  # Re-načítanie modelov
        except Exception as e:
            self.logger.error(f"Chyba pri obnove modelov: {e}")

    def _mark_processed_examples(self, processed_data: pd.DataFrame):
        """Označenie spracovaných príkladov"""
        try:
            learning_data = self.load_learning_data()

            if learning_data.empty:
                return

            # BEZPEČNÁ KONTROLA - vytvorenie stĺpca ak neexistuje
            if 'processed' not in learning_data.columns:
                learning_data['processed'] = False

            # Označenie použitých príkladov
            processed_texts = set(processed_data['text'])
            learning_data['processed'] = learning_data['text'].isin(processed_texts)

            learning_data.to_csv(self.learning_file, index=False)
            self.logger.info(f"🏷️ Označených {len(processed_texts)} príkladov ako spracovaných")

        except Exception as e:
            self.logger.error(f"Chyba pri označovaní príkladov: {e}")

    def get_learning_stats(self) -> Dict[str, any]:
        """Štatistiky self-learning systému"""
        try:
            learning_data = self.load_learning_data()
            buffer_size = len(self.learning_buffer)

            stats = {
                'buffer_size': buffer_size,
                'saved_examples': len(learning_data),
                'ready_for_retrain': buffer_size >= self.buffer_size,
                'confidence_threshold': self.confidence_threshold
            }

            # BEZPEČNÉ POČÍTANIE - kontrola existencie stĺpcov
            if not learning_data.empty:
                # Overené príklady
                if 'verified' in learning_data.columns:
                    stats['verified_examples'] = len(learning_data[learning_data['verified'] == True])
                else:
                    stats['verified_examples'] = 0

                # Vysoko istotné príklady
                if 'confidence' in learning_data.columns:
                    stats['high_confidence_examples'] = len(learning_data[learning_data['confidence'] > 0.9])
                else:
                    stats['high_confidence_examples'] = 0

            # Rozdelenie podľa kategórií
            if not learning_data.empty and 'category' in learning_data.columns:
                category_counts = learning_data['category'].value_counts().to_dict()
                stats['category_distribution'] = category_counts

            return stats

        except Exception as e:
            self.logger.error(f"Chyba pri získavaní štatistík: {e}")
            return {
                'buffer_size': len(self.learning_buffer),
                'saved_examples': 0,
                'verified_examples': 0,
                'high_confidence_examples': 0,
                'ready_for_retrain': False,
                'confidence_threshold': self.confidence_threshold
            }

    def manual_verification(self, text: str, correct_category: str):
        """
        Manuálne overenie a pridanie príkladu do učenia
        """
        try:
            learning_example = {
                'text': text,
                'category': correct_category,
                'confidence': 1.0,  # Maximálna istota pre manuálne overené
                'timestamp': datetime.now().isoformat(),
                'verified': True,
                'processed': False
            }

            self.learning_buffer.append(learning_example)
            self._save_learning_data()

            self.logger.info(f"✅ Manuálne overené: '{text}' -> {correct_category}")

        except Exception as e:
            self.logger.error(f"Chyba pri manuálnom overení: {e}")