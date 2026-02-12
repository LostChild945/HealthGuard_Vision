import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Chemins des datasets sources
    'india_train_csv': 'ML/data/India/train_multi.csv',
    'india_val_csv': 'ML/data/India/val_multi.csv',
    'huggingface_csv': 'ML/data/hugging-face/anemia_eyes_all.csv',
    
    # Répertoire de base pour vérification des images
    'base_data_dir': 'ML/data',
    
    # Chemins de sortie
    'output_train_csv': 'ML/data/train_combined.csv',
    'output_val_csv': 'ML/data/val_combined.csv',
    
    'india_prefix': 'India/',
    'huggingface_prefix': '',
    
    'india_source_label': 'india',
    'huggingface_source_label': 'huggingface',
    
    'hf_train_split_name': 'train',
    'hf_val_split_name': 'validation',
    'hf_test_size': 0.2,  
    'hf_random_state': 42,
    
    'temp_columns_to_drop': ['split', 'class'],
    
    'valid_labels': [0, 1],  
    
    'shuffle_combined': True,
    'shuffle_random_state': 42,
    
    'verify_paths': True,
    'max_examples_to_show': 3,
}


def load_india_datasets(config):
    """
    Charge les datasets India avec préfixe des chemins
    
    Returns:
        tuple: (train_df, val_df)
    """
    print(f"\n📊 Chargement dataset India...")
    
    india_train = pd.read_csv(config['india_train_csv'])
    india_val = pd.read_csv(config['india_val_csv'])
    
    # Ajoute le préfixe aux chemins
    if config['india_prefix']:
        print(f"   📝 Ajout du préfixe '{config['india_prefix']}' aux chemins...")
        india_train['image_path'] = config['india_prefix'] + india_train['image_path'].astype(str)
        india_val['image_path'] = config['india_prefix'] + india_val['image_path'].astype(str)
    
    # Ajoute la colonne source
    if 'source' not in india_train.columns:
        india_train['source'] = config['india_source_label']
        india_val['source'] = config['india_source_label']
    
    print(f"   ✅ Train: {len(india_train)} images")
    print(f"   ✅ Val: {len(india_val)} images")
    print(f"   Anémie train: {(india_train['anemia_label']==1).sum()}")
    print(f"   Non-anémie train: {(india_train['anemia_label']==0).sum()}")
    
    if len(india_train) > 0:
        print(f"   Exemple: {india_train['image_path'].iloc[0]}")
    
    return india_train, india_val


def load_huggingface_dataset(config):
    """
    Charge et split le dataset HuggingFace
    
    Returns:
        tuple: (train_df, val_df)
    """
    print(f"\n📊 Chargement dataset HuggingFace...")
    
    hf_csv = Path(config['huggingface_csv'])
    
    if not hf_csv.exists():
        print(f"   ❌ Fichier non trouvé: {hf_csv}")
        print(f"   Lance d'abord le script de création CSV HuggingFace")
        return None, None
    
    hf_all = pd.read_csv(hf_csv)
    
    # Filtre les labels invalides
    initial_count = len(hf_all)
    hf_all = hf_all[hf_all['anemia_label'].isin(config['valid_labels'])]
    
    if len(hf_all) < initial_count:
        print(f"   ⚠️  {initial_count - len(hf_all)} images avec labels invalides filtrées")
    
    print(f"   Total: {len(hf_all)} images")
    print(f"   Anémie: {(hf_all['anemia_label']==1).sum()}")
    print(f"   Non-anémie: {(hf_all['anemia_label']==0).sum()}")
    
    if len(hf_all) > 0:
        print(f"   Exemple: {hf_all['image_path'].iloc[0]}")
    
    # Split en train/val
    if 'split' in hf_all.columns:
        print(f"   📂 Utilisation des splits existants...")
        hf_train = hf_all[hf_all['split'] == config['hf_train_split_name']].copy()
        hf_val = hf_all[hf_all['split'] == config['hf_val_split_name']].copy()
        
        # Drop colonnes temporaires
        hf_train = hf_train.drop(columns=config['temp_columns_to_drop'], errors='ignore')
        hf_val = hf_val.drop(columns=config['temp_columns_to_drop'], errors='ignore')
    else:
        print(f"   📂 Création d'un split {int((1-config['hf_test_size'])*100)}/{int(config['hf_test_size']*100)}...")
        hf_train, hf_val = train_test_split(
            hf_all,
            test_size=config['hf_test_size'],
            stratify=hf_all['anemia_label'],
            random_state=config['hf_random_state']
        )
    
    # Ajoute la source
    hf_train['source'] = config['huggingface_source_label']
    hf_val['source'] = config['huggingface_source_label']
    
    print(f"   ✅ Train: {len(hf_train)} images")
    print(f"   ✅ Val: {len(hf_val)} images")
    
    return hf_train, hf_val


def harmonize_columns(df1, df2, name1="Dataset1", name2="Dataset2"):
    """
    Harmonise les colonnes entre deux DataFrames
    
    Returns:
        tuple: (df1_harmonized, df2_harmonized)
    """
    print(f"\n🔍 Harmonisation des colonnes...")
    
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    
    common_cols = cols1.intersection(cols2)
    print(f"   Colonnes communes: {len(common_cols)}")
    
    # Colonnes manquantes
    missing_in_df2 = cols1 - cols2
    missing_in_df1 = cols2 - cols1
    
    if missing_in_df2:
        print(f"   ⚠️  Manquantes dans {name2}: {sorted(missing_in_df2)}")
        for col in missing_in_df2:
            df2[col] = None
    
    if missing_in_df1:
        print(f"   ⚠️  Manquantes dans {name1}: {sorted(missing_in_df1)}")
        for col in missing_in_df1:
            df1[col] = None
    
    return df1, df2


def verify_image_paths(df, base_dir, max_examples=3):
    """
    Vérifie que les chemins d'images existent
    
    Returns:
        list: Chemins invalides
    """
    base_path = Path(base_dir)
    invalid_paths = []
    
    for idx, row in df.iterrows():
        img_path = base_path / row['image_path']
        if not img_path.exists():
            invalid_paths.append(row['image_path'])
    
    if invalid_paths:
        print(f"   ❌ {len(invalid_paths)} chemins invalides")
        if max_examples > 0:
            print(f"      Exemples: {invalid_paths[:max_examples]}")
    else:
        print(f"   ✅ Tous les chemins sont valides ({len(df)} images)")
    
    return invalid_paths


def combine_datasets():
    """
    Combine India + HuggingFace datasets avec configuration
    """
    print("="*70)
    print("🔗 COMBINAISON DES DATASETS")
    print("="*70)
    
    # 1. Charge India
    india_train, india_val = load_india_datasets(CONFIG)
    
    if india_train is None or len(india_train) == 0:
        print("\n❌ Impossible de charger le dataset India")
        return None, None
    
    # 2. Charge HuggingFace
    hf_train, hf_val = load_huggingface_dataset(CONFIG)
    
    if hf_train is None or len(hf_train) == 0:
        print("\n⚠️  Dataset HuggingFace non chargé, utilisation d'India uniquement")
        return india_train, india_val
    
    # 3. Harmonise les colonnes
    india_train, hf_train = harmonize_columns(
        india_train, hf_train, 
        name1="India", name2="HuggingFace"
    )
    india_val, hf_val = harmonize_columns(
        india_val, hf_val,
        name1="India", name2="HuggingFace"
    )
    
    # 4. Combine
    print(f"\n🔗 Combinaison des datasets...")
    combined_train = pd.concat([india_train, hf_train], ignore_index=True)
    combined_val = pd.concat([india_val, hf_val], ignore_index=True)
    
    # 5. Shuffle si demandé
    if CONFIG['shuffle_combined']:
        print(f"   🔀 Shuffle des données...")
        combined_train = combined_train.sample(
            frac=1, 
            random_state=CONFIG['shuffle_random_state']
        ).reset_index(drop=True)
        combined_val = combined_val.sample(
            frac=1, 
            random_state=CONFIG['shuffle_random_state']
        ).reset_index(drop=True)
    
    # 6. Affiche le résumé
    print(f"\n" + "="*70)
    print(f"✅ DATASET COMBINÉ FINAL")
    print(f"="*70)
    
    print(f"\n📊 TRAIN: {len(combined_train)} images")
    print(f"   India: {len(india_train)} ({len(india_train)/len(combined_train)*100:.1f}%)")
    print(f"   HuggingFace: {len(hf_train)} ({len(hf_train)/len(combined_train)*100:.1f}%)")
    print(f"   Anémie: {(combined_train['anemia_label']==1).sum()}")
    print(f"   Non-anémie: {(combined_train['anemia_label']==0).sum()}")
    
    print(f"\n📊 VALIDATION: {len(combined_val)} images")
    print(f"   India: {len(india_val)} ({len(india_val)/len(combined_val)*100:.1f}%)")
    print(f"   HuggingFace: {len(hf_val)} ({len(hf_val)/len(combined_val)*100:.1f}%)")
    print(f"   Anémie: {(combined_val['anemia_label']==1).sum()}")
    print(f"   Non-anémie: {(combined_val['anemia_label']==0).sum()}")
    
    print(f"\n📁 Exemples de chemins:")
    india_samples = combined_train[combined_train['source']==CONFIG['india_source_label']]
    hf_samples = combined_train[combined_train['source']==CONFIG['huggingface_source_label']]
    
    if len(india_samples) > 0:
        print(f"   India: {india_samples['image_path'].iloc[0]}")
    if len(hf_samples) > 0:
        print(f"   HF: {hf_samples['image_path'].iloc[0]}")
    
    # 7. Vérifie les chemins si demandé
    if CONFIG['verify_paths']:
        print(f"\n🔍 Vérification des chemins d'images...")
        
        invalid_train = verify_image_paths(
            combined_train, 
            CONFIG['base_data_dir'],
            CONFIG['max_examples_to_show']
        )
        
        invalid_val = verify_image_paths(
            combined_val, 
            CONFIG['base_data_dir'],
            CONFIG['max_examples_to_show']
        )
        
        # Bloque la sauvegarde si chemins invalides
        if invalid_train or invalid_val:
            print(f"\n❌ CSV non sauvegardés : chemins invalides détectés")
            print(f"   Vérifie la structure des dossiers dans {CONFIG['base_data_dir']}")
            return None, None
    
    # 8. Sauvegarde
    print(f"\n💾 Sauvegarde des fichiers...")
    combined_train.to_csv(CONFIG['output_train_csv'], index=False)
    combined_val.to_csv(CONFIG['output_val_csv'], index=False)
    
    print(f"   ✅ {CONFIG['output_train_csv']}")
    print(f"   ✅ {CONFIG['output_val_csv']}")
    
    # 9. Statistiques finales
    print(f"\n🎯 Gains:")
    print(f"   Train: {len(india_train)} → {len(combined_train)} (×{len(combined_train)/len(india_train):.2f})")
    print(f"   Val: {len(india_val)} → {len(combined_val)} (×{len(combined_val)/len(india_val):.2f})")
    
    return combined_train, combined_val


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    train_df, val_df = combine_datasets()
    
    if train_df is not None:
        print(f"\n{'='*70}")
        print(f"✅ COMBINAISON RÉUSSIE")
        print(f"{'='*70}")
    else:
        print(f"\n{'='*70}")
        print(f"❌ ÉCHEC DE LA COMBINAISON")
        print(f"{'='*70}")
