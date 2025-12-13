import pandas as pd
import csv


def find_bad_confidence(filename):
    print(f"🔍 Kontrolujem {filename}...")

    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)  # Preskoč hlavičku

        for line_num, row in enumerate(reader, 2):  # Začína od riadku 2
            if len(row) >= 3:  # Aspoň 3 stĺpce (text, category, confidence)
                confidence = row[2]  # Tretí stĺpec je confidence

                # Kontrola problémov
                problems = []

                # 1. Prázdne
                if not confidence or confidence.strip() == '':
                    problems.append("PRÁZDNY")

                # 2. Úvodzovky
                if confidence.startswith('"') and confidence.endswith('"'):
                    problems.append("ÚVODZOVKY")

                # 3. Nie je číslo
                try:
                    float(confidence.replace(',', '.').replace('"', '').replace("'", ""))
                except ValueError:
                    problems.append("NIE ČÍSLO")

                # 4. Obsahuje písmená
                if any(c.isalpha() for c in confidence):
                    problems.append("PÍSMENÁ")

                # 5. Percentá
                if '%' in confidence:
                    problems.append("PERCENTÁ")

                if problems:
                    print(f"🚨 RIADOK {line_num}: {confidence} -> {', '.join(problems)}")
                    print(f"   Celý riadok: {row[:5]}...")
                    print()


find_bad_confidence("learning_data.csv")