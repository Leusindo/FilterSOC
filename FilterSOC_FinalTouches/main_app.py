# main_app.py (DOPLNENÉ FUNKCIE)
import customtkinter as ctk
import tkinter as tk
from tkinter import ttk, messagebox
import logging
import sys
import os
from core.classifier import NewsClassifier
from core.self_learning import SelfLearningSystem
from core.news_collector import NewsCollector

# Pridanie current directory do Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class FilterSOCApp:
    def __init__(self):
        # Nastavenie vzhľadu
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        # Vytvorenie hlavného okna
        self.root = ctk.CTk()
        self.root.title("FilterSOC Alpha4 - Detekcia dezinformácií + Self-Learning")
        self.root.geometry("1000x700")

        # Inicializácia systémov
        self.classifier = NewsClassifier()
        self.self_learning = None
        self.news_collector = None
        self.models_loaded = False

        self.setup_ui()
        self.setup_logging()

    def setup_logging(self):
        """Nastavenie logovania"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def setup_ui(self):
        """Vytvorenie rozšíreného používateľského rozhrania"""
        # Hlavný frame
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Nadpis
        title_label = ctk.CTkLabel(
            main_frame,
            text="FilterSOC Alpha4 + Self-Learning 🧠",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title_label.pack(pady=20)

        # Notebook pre viac kariet
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill="both", expand=True, pady=10)

        # Karta Klasifikácia
        classification_tab = ctk.CTkFrame(self.notebook)
        self.notebook.add(classification_tab, text="🔍 Klasifikácia")

        # Karta Self-Learning
        learning_tab = ctk.CTkFrame(self.notebook)
        self.notebook.add(learning_tab, text="🧠 Self-Learning")

        # Karta Správy
        news_tab = ctk.CTkFrame(self.notebook)
        self.notebook.add(news_tab, text="📰 Zber Správ")

        # === KARTA KLASIFIKÁCIE ===
        self.setup_classification_tab(classification_tab)

        # === KARTA SELF-LEARNING ===
        self.setup_learning_tab(learning_tab)

        # === KARTA ZBERU SPRÁV ===
        self.setup_news_tab(news_tab)

        # Status
        self.status_label = ctk.CTkLabel(main_frame, text="Pripravený")
        self.status_label.pack(pady=5)

    def setup_classification_tab(self, parent):
        """Nastavenie karty klasifikácie"""
        # Frame pre vstup
        input_frame = ctk.CTkFrame(parent)
        input_frame.pack(fill="x", pady=10, padx=10)

        # Textové pole pre vstup
        ctk.CTkLabel(input_frame, text="Zadajte titulok:").pack(anchor="w", pady=(10, 5))
        self.text_entry = ctk.CTkTextbox(input_frame, height=100)
        self.text_entry.pack(fill="x", pady=5)

        # Tlačidlá
        button_frame = ctk.CTkFrame(parent)
        button_frame.pack(fill="x", pady=10, padx=10)

        ctk.CTkButton(
            button_frame,
            text="Načítať modely",
            command=self.load_models
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            button_frame,
            text="Klasifikovať",
            command=self.classify_text,
            fg_color="green"
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            button_frame,
            text="Klasifikovať + Učiť sa",
            command=self.classify_with_learning,
            fg_color="blue"
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            button_frame,
            text="Vyčistiť",
            command=self.clear_text
        ).pack(side="left", padx=5)

        # Frame pre výsledky
        results_frame = ctk.CTkFrame(parent)
        results_frame.pack(fill="both", expand=True, pady=10, padx=10)

        ctk.CTkLabel(results_frame, text="Výsledky:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", pady=(10, 5))

        # Výsledky
        self.results_text = ctk.CTkTextbox(results_frame, height=200)
        self.results_text.pack(fill="both", expand=True, pady=5)

        # Progress bar
        self.progress = ctk.CTkProgressBar(parent)
        self.progress.pack(fill="x", pady=10, padx=10)
        self.progress.set(0)

    def setup_learning_tab(self, parent):
        """Nastavenie karty self-learningu"""
        # Štatistiky
        stats_frame = ctk.CTkFrame(parent)
        stats_frame.pack(fill="x", pady=10, padx=10)

        ctk.CTkLabel(stats_frame, text="Štatistiky Self-Learningu:",
                     font=ctk.CTkFont(weight="bold")).pack(anchor="w", pady=(10, 5))

        self.learning_stats_text = ctk.CTkTextbox(stats_frame, height=150)
        self.learning_stats_text.pack(fill="x", pady=5)
        self.learning_stats_text.insert("1.0", "Načítajte modely pre zobrazenie štatistík...")

        # Tlačidlá pre self-learning
        learning_buttons_frame = ctk.CTkFrame(parent)
        learning_buttons_frame.pack(fill="x", pady=10, padx=10)

        ctk.CTkButton(
            learning_buttons_frame,
            text="🔄 Pretrénovať s Novými Dátami",
            command=self.retrain_with_learning,
            fg_color="orange"
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            learning_buttons_frame,
            text="📊 Obnoviť Štatistiky",
            command=self.update_learning_stats
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            learning_buttons_frame,
            text="💾 Uložiť Learning Data",
            command=self.save_learning_data
        ).pack(side="left", padx=5)

        # Learning log
        log_frame = ctk.CTkFrame(parent)
        log_frame.pack(fill="both", expand=True, pady=10, padx=10)

        ctk.CTkLabel(log_frame, text="Learning Log:",
                     font=ctk.CTkFont(weight="bold")).pack(anchor="w", pady=(10, 5))

        self.learning_log_text = ctk.CTkTextbox(log_frame)
        self.learning_log_text.pack(fill="both", expand=True, pady=5)

    def setup_news_tab(self, parent):
        """Nastavenie karty zberu správ"""
        # Tlačidlá pre zber správ
        news_buttons_frame = ctk.CTkFrame(parent)
        news_buttons_frame.pack(fill="x", pady=10, padx=10)

        ctk.CTkButton(
            news_buttons_frame,
            text="📡 Získať Nové Titulky",
            command=self.collect_news,
            fg_color="purple"
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            news_buttons_frame,
            text="🎯 Auto-Klasifikovať Nové Titulky",
            command=self.auto_classify_news,
            fg_color="brown"
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            news_buttons_frame,
            text="📈 Štatistiky Správ",
            command=self.show_news_stats
        ).pack(side="left", padx=5)

        # Zoznam titulkov
        news_list_frame = ctk.CTkFrame(parent)
        news_list_frame.pack(fill="both", expand=True, pady=10, padx=10)

        ctk.CTkLabel(news_list_frame, text="Nazbierané titulky:",
                     font=ctk.CTkFont(weight="bold")).pack(anchor="w", pady=(10, 5))

        self.news_list_text = ctk.CTkTextbox(news_list_frame)
        self.news_list_text.pack(fill="both", expand=True, pady=5)

    # ⚡ NOVÉ METÓDY PRE SELF-LEARNING A NEWS COLLECTOR ⚡

    def load_models(self):
        """Načítanie natrénovaných modelov a inicializácia self-learningu"""
        try:
            self.status_label.configure(text="Načítavam modely...")
            self.progress.set(0.3)
            self.root.update()

            self.classifier.load_models()
            self.models_loaded = True

            # Inicializácia self-learning systému
            self.self_learning = SelfLearningSystem(self.classifier, aggressive_learning=True)
            self.news_collector = NewsCollector(self.classifier)

            self.progress.set(1.0)
            self.status_label.configure(text="✅ Modely a self-learning systém načítané!")
            messagebox.showinfo("Úspech", "Modely a self-learning systém úspešne načítané!")

            # Aktualizácia štatistík
            self.update_learning_stats()

        except Exception as e:
            self.status_label.configure(text="Chyba pri načítavaní modelov")
            messagebox.showerror("Chyba", f"Chyba pri načítavaní modelov: {e}")
        finally:
            self.progress.set(0)

    def classify_with_learning(self):
        """Klasifikácia so self-learningom"""
        if not self.models_loaded:
            messagebox.showwarning("Varovanie", "Najprv načítajte modely!")
            return

        text = self.text_entry.get("1.0", "end-1c").strip()

        if not text:
            messagebox.showwarning("Varovanie", "Zadajte text pre klasifikáciu!")
            return

        try:
            self.status_label.configure(text="Klasifikujem s učením...")
            self.progress.set(0.5)
            self.root.update()

            # Predikcia s self-learningom
            category, probabilities, added_to_learning = self.self_learning.predict_with_learning(text)

            # Zobrazenie výsledkov
            result_text = f"Titulok: {text}\n\n"
            result_text += f"Predpovedaná kategória: {category}\n"
            result_text += f"Pridané do učenia: {'ÁNO' if added_to_learning else 'NIE'}\n\n"
            result_text += "Pravdepodobnosti:\n"

            for cat, prob in probabilities.items():
                result_text += f"  {cat}: {prob:.4f}\n"

            if added_to_learning:
                result_text += f"\n🎯 Titulok bol pridaný do self-learningu (istota: {max(probabilities.values()):.3f})"

            self.results_text.delete("1.0", "end")
            self.results_text.insert("1.0", result_text)

            # Aktualizácia štatistík
            self.update_learning_stats()

            self.progress.set(1.0)
            self.status_label.configure(text="Klasifikácia s učením dokončená")

        except Exception as e:
            self.status_label.configure(text="Chyba pri klasifikácii")
            messagebox.showerror("Chyba", f"Chyba pri klasifikácii: {e}")
        finally:
            self.progress.set(0)

    def retrain_with_learning(self):
        """Pretrénovanie s self-learning dátami"""
        if not self.models_loaded:
            messagebox.showwarning("Varovanie", "Najprv načítajte modely!")
            return

        try:
            self.status_label.configure(text="Pretrénovávam s novými dátami...")
            self.progress.set(0.3)
            self.root.update()

            success = self.self_learning.retrain_with_learning_data()

            if success:
                self.status_label.configure(text="✅ Model úspešne pretrénovaný!")
                messagebox.showinfo("Úspech", "Model bol úspešne pretrénovaný s novými dátami!")
            else:
                self.status_label.configure(text="ℹ️ Žiadne nové dáta pre pretrénovanie")
                messagebox.showinfo("Info", "Neboli nájdené žiadne nové overené dáta pre pretrénovanie.")

            # Aktualizácia štatistík
            self.update_learning_stats()

        except Exception as e:
            self.status_label.configure(text="Chyba pri pretrénovaní")
            messagebox.showerror("Chyba", f"Chyba pri pretrénovaní: {e}")
        finally:
            self.progress.set(0)

    def update_learning_stats(self):
        """Aktualizácia štatistík self-learningu"""
        if not self.models_loaded or not self.self_learning:
            return

        try:
            stats = self.self_learning.get_learning_stats()

            stats_text = "📊 ŠTATISTIKY SELF-LEARNINGU:\n\n"
            stats_text += f"• Buffer: {stats.get('buffer_size', 0)} príkladov\n"
            stats_text += f"• Uložené príklady: {stats.get('saved_examples', 0)}\n"
            stats_text += f"• Overené príklady: {stats.get('verified_examples', 0)}\n"
            stats_text += f"• Vysokoistotné: {stats.get('high_confidence_examples', 0)}\n"
            stats_text += f"• Pripravené na pretrénovanie: {'ÁNO' if stats.get('ready_for_retrain') else 'NIE'}\n"
            stats_text += f"• Hranica istoty: {stats.get('confidence_threshold', 0.85)}\n"

            if 'category_distribution' in stats:
                stats_text += "\n🎯 ROZDELENIE PODĽA KATEGÓRIÍ:\n"
                for category, count in stats['category_distribution'].items():
                    stats_text += f"  {category}: {count} príkladov\n"

            self.learning_stats_text.delete("1.0", "end")
            self.learning_stats_text.insert("1.0", stats_text)

        except Exception as e:
            self.logger.error(f"Chyba pri aktualizácii štatistík: {e}")

    def save_learning_data(self):
        """Manuálne uloženie learning dát"""
        if not self.models_loaded or not self.self_learning:
            return

        try:
            self.self_learning._save_learning_data()
            self.status_label.configure(text="✅ Learning data uložené!")
            messagebox.showinfo("Úspech", "Learning data úspešne uložené!")
        except Exception as e:
            messagebox.showerror("Chyba", f"Chyba pri ukladaní: {e}")

    def collect_news(self):
        """Získanie nových titulkov"""
        if not self.models_loaded:
            messagebox.showwarning("Varovanie", "Najprv načítajte modely!")
            return

        try:
            self.status_label.configure(text="Získavam nové titulky...")
            self.progress.set(0.3)
            self.root.update()

            news_items = self.news_collector.fetch_from_rss(limit_per_feed=5)

            if news_items:
                news_text = "📰 NAZBIERANÉ TITULKY:\n\n"
                for i, item in enumerate(news_items, 1):
                    news_text += f"{i}. {item['title']}\n"
                    news_text += f"   Zdroj: {item['source']}\n\n"

                self.news_list_text.delete("1.0", "end")
                self.news_list_text.insert("1.0", news_text)

                self.status_label.configure(text=f"✅ Získaných {len(news_items)} titulkov!")
            else:
                self.status_label.configure(text="ℹ️ Neboli nájdené žiadne titulky")

        except Exception as e:
            self.status_label.configure(text="Chyba pri zbere titulkov")
            messagebox.showerror("Chyba", f"Chyba pri zbere titulkov: {e}")
        finally:
            self.progress.set(0)

    def auto_classify_news(self):
        """Automatická klasifikácia nových titulkov"""
        if not self.models_loaded:
            messagebox.showwarning("Varovanie", "Najprv načítajte modely!")
            return

        try:
            self.status_label.configure(text="Automatická klasifikácia titulkov...")
            self.progress.set(0.5)
            self.root.update()

            classified_items = self.news_collector.auto_classify_and_learn(
                self.self_learning,
                min_confidence=0.0
            )


            if classified_items:
                news_text = "🎯 AUTOMATICKY KLASIFIKOVANÉ TITULKY:\n\n"
                for i, item in enumerate(classified_items, 1):
                    news_text += f"{i}. {item['title']}\n"
                    news_text += f"   Kategória: {item['predicted_category']} (istota: {item['confidence']:.3f})\n"
                    news_text += f"   Učenie: {'ÁNO' if item['added_to_learning'] else 'NIE'}\n"
                    news_text += f"   Zdroj: {item['source']}\n\n"

                self.news_list_text.delete("1.0", "end")
                self.news_list_text.insert("1.0", news_text)

                # Aktualizácia štatistík
                self.update_learning_stats()

                self.status_label.configure(text=f"✅ Spracovaných {len(classified_items)} titulkov!")
            else:
                self.status_label.configure(text="ℹ️ Žiadne titulky na spracovanie")

        except Exception as e:
            self.status_label.configure(text="Chyba pri automatickej klasifikácii")
            messagebox.showerror("Chyba", f"Chyba pri automatickej klasifikácii: {e}")
        finally:
            self.progress.set(0)

    def show_news_stats(self):
        """Zobrazenie štatistík správ"""
        if not self.models_loaded or not self.news_collector:
            return

        try:
            stats = self.news_collector.get_news_stats()

            stats_text = "📈 ŠTATISTIKY SPRÁV:\n\n"
            stats_text += f"• Celkový počet: {stats.get('total_news', 0)}\n"
            stats_text += f"• Vysokoistotné: {stats.get('high_confidence_news', 0)}\n"

            if 'sources' in stats:
                stats_text += "\n📡 ZDROJE:\n"
                for source, count in stats['sources'].items():
                    stats_text += f"  {source}: {count}\n"

            if 'categories' in stats:
                stats_text += "\n🎯 KATEGÓRIE:\n"
                for category, count in stats['categories'].items():
                    stats_text += f"  {category}: {count}\n"

            messagebox.showinfo("Štatistiky Správ", stats_text)

        except Exception as e:
            messagebox.showerror("Chyba", f"Chyba pri získavaní štatistík: {e}")

    # PÔVODNÉ METÓDY
    def classify_text(self):
        """Klasifikácia zadaného textu (pôvodná metóda)"""
        if not self.models_loaded:
            messagebox.showwarning("Varovanie", "Najprv načítajte modely!")
            return

        text = self.text_entry.get("1.0", "end-1c").strip()

        if not text:
            messagebox.showwarning("Varovanie", "Zadajte text pre klasifikáciu!")
            return

        try:
            self.status_label.configure(text="Klasifikujem...")
            self.progress.set(0.5)
            self.root.update()

            # Predikcia
            predicted_label, probabilities = self.classifier.predict(text)

            # Zobrazenie výsledkov
            result_text = f"Titulok: {text}\n\n"
            result_text += f"Predpovedaná kategória: {predicted_label}\n\n"
            result_text += "Pravdepodobnosti:\n"

            for category, prob in probabilities.items():
                result_text += f"  {category}: {prob:.4f}\n"

            self.results_text.delete("1.0", "end")
            self.results_text.insert("1.0", result_text)

            self.progress.set(1.0)
            self.status_label.configure(text="Klasifikácia dokončená")

        except Exception as e:
            self.status_label.configure(text="Chyba pri klasifikácii")
            messagebox.showerror("Chyba", f"Chyba pri klasifikácii: {e}")
        finally:
            self.progress.set(0)

    def clear_text(self):
        """Vyčistenie textových polí"""
        self.text_entry.delete("1.0", "end")
        self.results_text.delete("1.0", "end")
        self.status_label.configure(text="Pripravený")

    def run(self):
        """Spustenie aplikácie"""
        self.root.mainloop()


if __name__ == "__main__":
    app = FilterSOCApp()
    app.run()