import feedparser
import csv
from datetime import datetime
import time

# Zdroje zoradené podľa žánru
slovenske_media = {
    # Mainstreamové médiá
    'mainstream': [
        'https://www.sme.sk/rss-title',
        'https://spravy.pravda.sk/rss/',
        'https://dennikn.sk/rss/',
        'https://hnonline.sk/rss',
    ],

    # Bulvárne médiá
    'bulvar': [
        'https://www.topky.sk/rss',
        'https://www.pluska.sk/rss',
        'https://www.cas.sk/rss',
        'https://www.markiza.sk/rss',
    ],

    # Zábavné a lifestyle
    'zabava': [
        'https://www.zivot.sk/rss',
        'https://www.zena.sk/rss',
    ],

    # Regionálne médiá
    'regionalne': [
        'https://kosice.dnes24.sk/rss',
        'https://bratislava.dnes24.sk/rss',
    ]
}


def download_titulky():
    vsetky_titulky = []

    for zaner, zdroje in slovenske_media.items():
        for zdroj_url in zdroje:
            try:
                print(f"Načítavam {zdroj_url}...")
                feed = feedparser.parse(zdroj_url)

                for entry in feed.entries[:15]:  # Berieme prvých 15 titulkov
                    titulok = entry.title.strip()
                    if titulok and len(titulok) > 10:  # Filtrujeme príliš krátke titulky
                        vsetky_titulky.append({
                            'titulok': titulok,
                            'zdroj': zdroj_url.split('/')[2],
                            'zaner': zaner,
                            'datum': datetime.now().strftime('%Y-%m-%d'),
                            'kategoria': ''  # PRÁZDNA - pripravíme pre TEBU!
                        })

                time.sleep(1)  # Rešpektujeme politiku médií

            except Exception as e:
                print(f"Chyba pri načítavaní {zdroj_url}: {e}")

    return vsetky_titulky


def uloz_do_csv(titulky, subor):
    with open(subor, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['titulok', 'kategoria', 'zaner', 'zdroj', 'datum']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for titulok in titulky:
            writer.writerow(titulok)


# Hlavný program
if __name__ == "__main__":
    print("Sťahujem titulky zo slovenských médií...")

    # Stiahnutie titulkov
    titulky = download_titulky()
    print(f"Načítaných {len(titulky)} titulkov")

    # Uloženie kompletného datasetu S PRÁZNYMI KATEGÓRIAMI
    uloz_do_csv(titulky, 'titulky_pre_klasifikaciu.csv')
    print("Uložené do titulky_pre_klasifikaciu.csv")

    # Uloženie zjednodušenej verzie
    with open('titulky_jednoduche.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['titulok', 'kategoria'])
        for titulok in titulky:
            writer.writerow([titulok['titulok'], ''])  # PRÁZDNA KATEGÓRIA

    print("Uložené do titulky_jednoduche.csv")

    # Štatistika zdrojov
    zanre = {}
    for titulok in titulky:
        zan = titulok['zaner']
        zanre[zan] = zanre.get(zan, 0) + 1

    print("\nŠtatistika žánrov:")
    for zaner, pocet in zanre.items():
        print(f"  {zaner}: {pocet} titulkov")

    print("\nTeraz môžeš manuálne klasifikovať titulky v CSV súboroch!")
    print("Kategórie: clickbait, conspiracy, false_news, propaganda, satire, misleading, biased, legitimate")