# core/news_collector.py
import feedparser
import requests
from bs4 import BeautifulSoup
import logging
import time
import pandas as pd
from datetime import datetime
from typing import List, Dict
import os
from .config import Config


class NewsCollector:
    """
    Automatický zber nových titulkov z RSS feedov a webových stránok
    """

    def __init__(self, classifier=None):
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        self.classifier = classifier

        # Zoznam slovenských RSS feedov
        self.rss_feeds = [
            # ✅ TYTO FUNGUJÚ (podľa teba):
            "https://www.aktuality.sk/rss/",
            "https://www.teraz.sk/rss/vsetky-spravy.rss",
            "https://sita.sk/feed/",
            "https://www1.pluska.sk/rss.xml",
            "https://www.sme.sk/rss-title",
            "https://spravy.pravda.sk/rss/xml/",
            "https://hnonline.sk/feed",
            "https://www1.pluska.sk/rss.xml",  # Duplicitné, ale necháme

            # 🔽 PRIDAJ TIEŽ TYTO FUNGUJÚCE (testované):
            "https://kosice.dnes24.sk/feed/",
            "https://bratislava.dnes24.sk/feed/",
            "https://www.korzar.sme.sk/rss/",
            "https://tech.sme.sk/rss-title",
            "https://ekonomika.sme.sk/rss-title",
            "https://sport.sme.sk/rss-title",
            "https://www.dennikn.sk/feed/",

            # 🔽 NOVÉ FUNGUJÚCE FEEDY (2024):
            "https://www.startitup.sk/feed/",
            "https://www.trend.sk/feed",
            "https://www.zive.sk/rss/",

            # Limit na 15 feedov pre rýchlosť
        ]

        # Webové stránky pre scraping
        self.news_websites = [
            "https://www.sme.sk",
            "https://spravy.pravda.sk",
            "https://www.aktuality.sk",
            "https://www.pluska.sk"
        ]

        self.collected_file = "data/collected_news/collected_titles.csv"
        os.makedirs("data/collected_news", exist_ok=True)

        self.logger.info("✅ News collector inicializovaný")

    def fetch_from_rss(self, limit_per_feed: int = 25) -> List[Dict[str, str]]:
        """Získanie titulkov z RSS feedov S ODSTRÁNENÍM DUPLIKÁTOV"""
        all_titles = []
        seen_titles = set()  # 🔽 Množina už videných titulkov

        for feed_url in self.rss_feeds:
            try:
                self.logger.info(f"📡 Načítavam RSS: {feed_url}")

                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'application/rss+xml, application/xml, text/xml'
                }

                feed = feedparser.parse(feed_url, request_headers=headers)
                entries_to_process = feed.entries[:limit_per_feed] if hasattr(feed, 'entries') else []

                titles_from_feed = 0
                for entry in entries_to_process:
                    if hasattr(entry, 'title'):
                        title = entry.title.strip()

                        # 🔽 NORMALIZÁCIA TITULKU
                        normalized_title = self._normalize_title(title)

                        # Preskoč prázdne alebo príliš krátke
                        if not normalized_title or len(normalized_title) < 15:
                            continue

                        # 🔽 KONTROLA DUPLIKÁTU
                        if normalized_title in seen_titles:
                            self.logger.debug(f"⏭️ Duplikát preskočený: '{normalized_title}'")
                            continue

                        # Kontrola či je slovenský
                        if self._is_slovak_title(normalized_title):
                            news_item = {
                                'title': title,  # Pôvodný titulok
                                'normalized_title': normalized_title,  # Normalizovaný
                                'source': feed_url,
                                'published': entry.get('published', ''),
                                'link': entry.get('link', ''),
                                'collected_at': datetime.now().isoformat()
                            }
                            all_titles.append(news_item)
                            seen_titles.add(normalized_title)  # 🔽 Pridaj do množiny
                            titles_from_feed += 1

                self.logger.info(f"✅ Získaných {titles_from_feed} titulkov z {feed_url}")

                time.sleep(0.5)

            except Exception as e:
                self.logger.error(f"❌ Chyba pri {feed_url}: {e}")
                continue

        self.logger.info(f"🎯 Celkovo získaných {len(all_titles)} UNIKÁTNYCH titulkov")
        return all_titles

    def _normalize_title(self, title: str) -> str:
        """Normalizácia titulku na porovnanie duplikátov"""
        if not title:
            return ""

        # 1. Malé písmená
        normalized = title.lower()

        # 2. Odstrániť interpunkciu a špeciálne znaky
        import re
        normalized = re.sub(r'[^\w\sáäčďéíľĺňóôřŕšťúýž]', '', normalized)

        # 3. Odstrániť prebytočné medzery
        normalized = re.sub(r'\s+', ' ', normalized).strip()

        # 4. Odstrániť časté prefixy/suffixy
        prefixes = ['video:', 'foto:', 'video |', 'foto |', 'exkluzívne:', 'breaking:']
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):].strip()

        return normalized

    def _is_slovak_title(self, title: str) -> bool:
        """Vylepšená kontrola slovenského titulku"""
        if not title:
            return False

        # Kľúčové slovenské znaky
        slovak_chars = ['á', 'ä', 'č', 'ď', 'é', 'í', 'ľ', 'ĺ', 'ň', 'ó', 'ô', 'ř', 'ŕ', 'š', 'ť', 'ú', 'ý', 'ž']

        # Kľúčové slovenské slová
        slovak_words = [
            'a', 'o', 'v', 's', 'z', 'čo', 'ako', 'kde', 'prečo', 'ktorý', 'ktorá', 'ktoré',
            'pri', 'po', 'na', 'do', 'za', 'so', 'sa', 'si', 'je', 'bol', 'bola', 'bolo'
        ]

        title_lower = title.lower()

        # Kontrola prítomnosti aspoň 2 slovenských znakov
        slovak_char_count = sum(1 for char in title_lower if char in slovak_chars)

        # Kontrola prítomnosti slovenských slov
        slovak_word_count = sum(1 for word in slovak_words if word in title_lower.split())

        return slovak_char_count >= 2 or slovak_word_count >= 2

    def auto_classify_and_learn(self, self_learning_system=None, min_confidence: float = 0.0) -> List[Dict]:
        """Automatická klasifikácia s KONZISTENTNOU KLASIFIKÁCIOU DUPLIKÁTOV"""
        try:
            self.logger.info("🎯 Začínam automatickú klasifikáciu a učenie...")

            news_items = self.fetch_from_rss(limit_per_feed=25)

            if not news_items:
                self.logger.info("ℹ️ Neboli nájdené žiadne nové titulky")
                return []

            classified_items = []
            classified_cache = {}  # 🔽 CACHE pre konzistentnú klasifikáciu

            for item in news_items:
                title = item['title']
                normalized_title = item.get('normalized_title', self._normalize_title(title))

                # 🔽 POUŽI CACHE AK UŽ BOL TITULOK KLASIFIKOVANÝ
                if normalized_title in classified_cache:
                    self.logger.debug(f"🔄 Používam cache pre: '{normalized_title}'")
                    cached_result = classified_cache[normalized_title]

                    classified_item = {
                        **item,
                        'predicted_category': cached_result['category'],
                        'confidence': cached_result['confidence'],
                        'added_to_learning': False,  # Nepridávať do učenia znovu
                        'probabilities': cached_result['probabilities']
                    }
                    classified_items.append(classified_item)
                    continue

                # Normálna klasifikácia
                if len(title) < 15:
                    continue

                try:
                    if self_learning_system and self.classifier:
                        category, probabilities, added_to_learning = self_learning_system.predict_with_learning(title)
                    elif self.classifier:
                        category, probabilities = self.classifier.predict(title)
                        added_to_learning = False
                    else:
                        continue

                    confidence = max(probabilities.values())

                    if confidence < min_confidence:
                        continue

                    # 🔽 ULOŽ DO CACHE
                    classified_cache[normalized_title] = {
                        'category': category,
                        'confidence': confidence,
                        'probabilities': probabilities
                    }

                    classified_item = {
                        **item,
                        'predicted_category': category,
                        'confidence': confidence,
                        'added_to_learning': added_to_learning,
                        'probabilities': probabilities
                    }
                    classified_items.append(classified_item)

                    if confidence > 0.8:
                        self.logger.info(f"🎯 {category}: '{title[:50]}...' ({confidence:.3f})")

                except Exception as e:
                    self.logger.error(f"Chyba pri klasifikácii: {e}")
                    continue

            # Uloženie výsledkov
            self._save_collected_news(classified_items)

            # 🔽 LOGOVANIE ŠTATISTÍK
            unique_titles = len(set(item.get('normalized_title', self._normalize_title(item['title']))
                                    for item in classified_items))
            self.logger.info(f"✅ Spracovaných {len(classified_items)} titulkov ({unique_titles} unikátnych)")

            return classified_items

        except Exception as e:
            self.logger.error(f"❌ Chyba v auto_classify_and_learn: {e}")
            return []
    def _save_collected_news(self, news_items: List[Dict]):
        """Uloženie nazbieraných titulkov"""
        try:
            if not news_items:
                return

            # Načítanie existujúcich dát
            try:
                existing_df = pd.read_csv(self.collected_file)
            except FileNotFoundError:
                existing_df = pd.DataFrame()

            # Pridanie nových dát
            new_df = pd.DataFrame(news_items)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)

            # Odstránenie duplikátov
            combined_df = combined_df.drop_duplicates(subset=['title'])

            # Uloženie
            combined_df.to_csv(self.collected_file, index=False)

            self.logger.info(f"💾 Uložených {len(new_df)} nových titulkov")

        except Exception as e:
            self.logger.error(f"Chyba pri ukladaní titulkov: {e}")

    def get_recent_news(self, hours: int = 24) -> pd.DataFrame:
        """
        Získanie nedávno nazbieraných titulkov

        Args:
            hours: Počet hodien dozadu

        Returns:
            DataFrame s nedávnymi titulkami
        """
        try:
            df = pd.read_csv(self.collected_file)

            # Filtr podľa času
            if 'collected_at' in df.columns:
                df['collected_at'] = pd.to_datetime(df['collected_at'])
                cutoff_time = datetime.now() - pd.Timedelta(hours=hours)
                recent_df = df[df['collected_at'] > cutoff_time]
            else:
                recent_df = df.tail(50)  # Posledných 50 ak nie je časová značka

            return recent_df

        except FileNotFoundError:
            self.logger.info("Súbor s nazbieranými titulkmi neexistuje")
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Chyba pri načítavaní recent news: {e}")
            return pd.DataFrame()

    def get_news_stats(self) -> Dict[str, any]:
        """Štatistiky nazbieraných správ"""
        try:
            df = self.get_recent_news(hours=168)  # Posledný týždeň

            if df.empty:
                return {'total_news': 0}

            stats = {
                'total_news': len(df),
                'sources': df['source'].value_counts().to_dict(),
                'categories': df[
                    'predicted_category'].value_counts().to_dict() if 'predicted_category' in df.columns else {},
                'high_confidence_news': len(df[df['confidence'] > 0.8]) if 'confidence' in df.columns else 0
            }

            return stats

        except Exception as e:
            self.logger.error(f"Chyba pri získavaní štatistík: {e}")
            return {}