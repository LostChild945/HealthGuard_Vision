# create_labels.py - Labélisation automatique anémie
import pandas as pd

# 1. Charger votre fichier métadonnée actuel
df = pd.read_csv('data/Italy/Italy.csv')  # Ajustez le nom si différent
print(f"📊 Dataset chargé: {len(df)} images")
print(f"Colonnes: {df.columns.tolist()}")
print("\nPremières lignes:")
print(df.head())

# 2. Fonction labélisation WHO
def label_anemia_who(hgb, gender, age):
    """
    Retourne 1 si anémie, 0 si normal
    Selon seuils WHO
    """
    gender_lower = str(gender).lower()
    
    # Seuils selon genre
    if gender_lower == 'f':
        if age < 15:
            threshold = 11.5
        else:
            threshold = 12.0
    elif gender_lower == 'm':
        if age < 15:
            threshold = 11.5
        else:
            threshold = 13.0
    else:
        if age < 15:
            threshold = 11.5
        else:
            threshold = 12.5  
    return 1 if hgb < threshold else 0

df['Hgb'] = df['Hgb'].str.replace(',', '.').astype(float)
df['anemia_label'] = df.apply(
    lambda row: label_anemia_who(row['Hgb'], row['Gender'], row['Age']), 
    axis=1
)

print("\n✅ Labélisation terminée!")
print("\nDistribution:")
print(df['anemia_label'].value_counts())

print("\nExemples:")
print(df[['Number', 'Hgb', 'Gender', 'Age', 'anemia_label']].head(10))

# 6. Stats par groupe
print("\n📊 Statistiques Hgb:")
print(df.groupby(['Gender', 'anemia_label'])['Hgb'].describe())
# 7. Sauvegarder nouveau fichier
output_path = 'data/Italy/labels.csv'
df.to_csv(output_path, index=False)
print(f"\n✅ Fichier créé: {output_path}")
print(f"Colonnes: {df.columns.tolist()}")
