import pandas as pd
from collections import defaultdict

# Načítanie dát
df = pd.read_csv('learningdata.csv')  # Zmeň názov súboru

# 1. Kontrola duplikátov
print("=" * 60)
print("1. DUPLIKÁTY PODĽA TITULKOV:")
print("=" * 60)

# Získanie duplikátov podľa titulku
title_counts = df['title'].value_counts()
duplicates = title_counts[title_counts > 1]

if len(duplicates) > 0:
    for title, count in duplicates.items():
        print(f"\nTitulok sa opakuje {count}-krát:")
        print(f"'{title[:100]}...'")
        indices = df[df['title'] == title].index.tolist()
        print(f"Riadky: {indices}")
        # Ukáž kategórie duplikátov
        categories = df.loc[indices, 'category'].tolist()
        print(f"Kategórie: {categories}")
        if len(set(categories)) > 1:
            print("⚠️  RÔZNE KATEGÓRIE v duplikátoch!")
else:
    print("Žiadne duplikáty podľa titulkov.")

# 2. Kontrola nekonzistentných kategórií pre rovnaké typy titulkov
print("\n" + "=" * 60)
print("2. NEKONZISTENTNÉ KATEGÓRIE:")
print("=" * 60)


# Funkcia na kontrolu vzorov titulkov
def check_title_patterns(df):
    issues = []

    patterns = {
        'clickbait_keywords': ['?', '!', 'TOTO', 'ako', 'prečo', 'tajomstvo', 'zaručený', 'FOTO', 'VIDEO',
                               'MIMORIADNE'],
        'biased_keywords': ['tvrdí', 'žiada', 'kritizuje', 'obvinili', 'vyzýva', 'odsudzuje'],
        'misleading_keywords': ['môže', 'mohlo by', 'údajne', 'podľa niektorých', 'asi', 'pravdepodobne']
    }

    for idx, row in df.iterrows():
        title = row['title']
        category = row['category']

        # Kontrola clickbait vzorov
        if any(keyword in title for keyword in patterns['clickbait_keywords']):
            if category != 'clickbait':
                issues.append((idx, title[:80], category, 'clickbait', 'Obsahuje clickbait vzory'))

        # Kontrola legitímnych titulkov s nízkou confidence
        if category == 'legitimate' and row['confidence'] < 0.5:
            issues.append((idx, title[:80], category, f"confidence={row['confidence']}",
                           "Nízka confidence pre legitímny titulok"))

        # Kontrola biased titulkov s vysokou confidence
        if category == 'biased' and row['confidence'] > 0.7:
            issues.append(
                (idx, title[:80], category, f"confidence={row['confidence']}", "Vysoká confidence pre biased titulok"))

    return issues


issues = check_title_patterns(df)

if issues:
    for idx, title, current_cat, expected, reason in issues:
        print(f"\nRiadok {idx}: '{title}...'")
        print(f"  Súčasná kategória: {current_cat}")
        print(f"  Problém: {reason}")
        print(f"  Očakávané: {expected}")
else:
    print("Žiadne zjavné nekonzistencie v kategóriách.")

# 3. Štatistiky kategórií
print("\n" + "=" * 60)
print("3. ŠTATISTIKY KATEGÓRIÍ:")
print("=" * 60)

category_stats = df['category'].value_counts()
print("Počet titulkov podľa kategórií:")
for cat, count in category_stats.items():
    percentage = (count / len(df)) * 100
    print(f"  {cat}: {count} ({percentage:.1f}%)")

# 4. Kontrola časových značiek
print("\n" + "=" * 60)
print("4. PROBLÉMY S TIMESTAMP:")
print("=" * 60)

# Konvertovanie timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Kontrola rovnakých timestampov pre rôzne titulky
time_counts = df['timestamp'].value_counts()
duplicate_times = time_counts[time_counts > 1]

if len(duplicate_times) > 0:
    print("Rovnaké timestampy pre rôzne titulky:")
    for timestamp, count in duplicate_times.head(5).items():  # Len prvých 5
        print(f"\n{timestamp}: {count} titulkov")
        titles = df[df['timestamp'] == timestamp]['title'].head(3).tolist()
        for title in titles:
            print(f"  - {title[:60]}...")
else:
    print("Žiadne duplicitné timestampy.")

# 5. Kontrola confidence rozsahu podľa kategórií
print("\n" + "=" * 60)
print("5. CONFIDENCE PODĽA KATEGÓRIÍ:")
print("=" * 60)

print("Priemerná confidence podľa kategórií:")
for category in df['category'].unique():
    cat_data = df[df['category'] == category]
    avg_conf = cat_data['confidence'].mean()
    min_conf = cat_data['confidence'].min()
    max_conf = cat_data['confidence'].max()
    print(f"\n{category}:")
    print(f"  Priemer: {avg_conf:.3f}")
    print(f"  Rozsah: {min_conf:.3f} - {max_conf:.3f}")

    # Kontrola extrémnych hodnôt
    if category == 'legitimate' and avg_conf < 0.6:
        print("  ⚠️  Nízka priemerná confidence pre legitímne titulky")
    if category == 'clickbait' and avg_conf > 0.7:
        print("  ⚠️  Vysoká priemerná confidence pre clickbait")

# 6. Hľadanie podozrivých kombinácií
print("\n" + "=" * 60)
print("6. PODOZRIVÉ KOMBINÁCIE:")
print("=" * 60)

suspicious = []

for idx, row in df.iterrows():
    title = row['title'].lower()
    category = row['category']
    confidence = row['confidence']

    # Legitímny titulok s veľmi nízkou confidence
    if category == 'legitimate' and confidence < 0.3:
        suspicious.append((idx, "Legitímny titulok s veľmi nízkou confidence", confidence))

    # Clickbait titulok s veľmi vysokou confidence
    if category == 'clickbait' and confidence > 0.8:
        suspicious.append((idx, "Clickbait s veľmi vysokou confidence", confidence))

    # Politické titulky označené ako clickbait
    political_keywords = ['vláda', 'prezident', 'ministerstvo', 'parlament', 'voľby']
    if any(keyword in title for keyword in political_keywords) and category == 'clickbait':
        suspicious.append((idx, "Politický titulok označený ako clickbait", None))

if suspicious:
    for idx, reason, confidence in suspicious[:10]:  # Len prvých 10
        title = df.loc[idx, 'title'][:70]
        print(f"\nRiadok {idx}: '{title}...'")
        print(f"  Problém: {reason}")
        if confidence:
            print(f"  Confidence: {confidence}")
else:
    print("Žiadne podozrivé kombinácie.")

# 7. Výpis prvých 5 problémových riadkov na opravu
print("\n" + "=" * 60)
print("7. TOP 5 RIADKOV NA OPRAVU:")
print("=" * 60)

# Zoradenie podľa priority problémov
problem_rows = []

for idx, row in df.iterrows():
    score = 0
    reasons = []

    # Duplikáty majú najvyššiu prioritu
    if title_counts[row['title']] > 1:
        score += 100

    # Nízka confidence pre legitímne
    if row['category'] == 'legitimate' and row['confidence'] < 0.4:
        score += 50
        reasons.append("nízka confidence pre legitímny")

    # Vysoká confidence pre clickbait
    if row['category'] == 'clickbait' and row['confidence'] > 0.7:
        score += 30
        reasons.append("vysoká confidence pre clickbait")

    # Nezrovnalosti v kategóriách
    title_lower = row['title'].lower()
    if ('?' in row['title'] or '!' in row['title']) and row['category'] != 'clickbait':
        score += 20
        reasons.append("otáznik/výkričník ale nie clickbait")

    if score > 0:
        problem_rows.append((idx, score, reasons, row['title'][:60]))

# Zoradenie podľa skóre
problem_rows.sort(key=lambda x: x[1], reverse=True)

for idx, score, reasons, title in problem_rows[:5]:
    print(f"\nRiadok {idx} (skóre problémov: {score}):")
    print(f"  Titulok: '{title}...'")
    print(f"  Kategória: {df.loc[idx, 'category']}")
    print(f"  Confidence: {df.loc[idx, 'confidence']:.3f}")
    print(f"  Dôvody: {', '.join(reasons)}")

print("\n" + "=" * 60)
print("ZHRNUTIE:")
print("=" * 60)
print(f"Celkový počet riadkov: {len(df)}")
print(f"Počet unikátnych titulkov: {len(df['title'].unique())}")
print(f"Počet duplikátov: {len(df) - len(df['title'].unique())}")
print(f"\nOdporúčaná akcia: Odstrániť riadky {list(duplicates.index) if len(duplicates) > 0 else 'žiadne'}")