import feedparser
import csv

# Slovenské spravodajské RSS feedy
RSS_FEEDS = {
    "SME": "https://rss.sme.sk/rss/rss.asp",
    "DennikN": "https://dennikn.sk/feed/",
    "Pravda": "https://www.pravda.sk/rss/",
    "Aktuality": "https://www.aktuality.sk/rss/"
}

# Výstupný súbor
OUTPUT_FILE = "slovak_headlines.csv"

def fetch_headlines():
    headlines = []
    for source, url in RSS_FEEDS.items():
        feed = feedparser.parse(url)
        for entry in feed.entries:
            title = entry.title.strip()
            headlines.append((title, source))
    return headlines

def save_to_csv(headlines, filename=OUTPUT_FILE):
    with open(filename, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["title", "source"])  # hlavička
        for title, source in headlines:
            writer.writerow([title, source])

if __name__ == "__main__":
    headlines = fetch_headlines()
    print(f"Načítaných {len(headlines)} titulkov.")
    save_to_csv(headlines)
    print(f"Titulky uložené do {OUTPUT_FILE}")
