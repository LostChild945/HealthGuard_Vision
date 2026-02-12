import pandas as pd
from pathlib import Path
from collections import defaultdict

def create_multi_image_csv(base_csv, data_dir, output_csv):
    """
    Crée un CSV avec TOUTES les images par patient
    """
    df = pd.read_csv(base_csv)
    
    data_path = Path(data_dir)
    all_rows = []
    
    print(f"🔍 Scan des dossiers patients...")
    
    for _, row in df.iterrows():
        patient_num = row['patient_number']
        patient_dir = data_path / str(patient_num)
        
        if not patient_dir.exists():
            print(f"⚠️  Dossier manquant: {patient_num}")
            continue
        
        # Liste TOUTES les images du patient (jpg + png valides)
        images = []
        
        # JPG (image originale)
        jpg_files = list(patient_dir.glob('*.jpg'))
        images.extend(jpg_files)
        
        # PNG (variantes - sauf les corrompues)
        png_files = list(patient_dir.glob('*.png'))
        for png in png_files:
            # Vérifie que le PNG n'est pas corrompu (test rapide)
            try:
                from PIL import Image
                with Image.open(png) as img:
                    img.verify()
                images.append(png)
            except:
                print(f"❌ Skip corrompu: {png}")
        
        # Crée une ligne par image avec les mêmes métadonnées patient
        for img_path in images:
            relative_path = f"{patient_num}/{img_path.name}"
            all_rows.append({
                'patient_number': patient_num,
                'image_path': relative_path,
                'image_name': img_path.name,
                'anemia_label': row['anemia_label'],
                'hgb': row['hgb'],
                'gender': row['gender'],
                'age': row['age'],
                'n_images': len(images)  # Total images pour ce patient
            })
    
    # Crée le nouveau DataFrame
    df_multi = pd.DataFrame(all_rows)
    
    print(f"\n📊 Résultats:")
    print(f"   Patients: {df['patient_number'].nunique()}")
    print(f"   Images totales: {len(df_multi)}")
    print(f"   Images/patient moyen: {len(df_multi) / df['patient_number'].nunique():.1f}")
    print(f"\n   Distribution anemia_label:")
    print(df_multi['anemia_label'].value_counts())
    
    # Sauvegarde
    df_multi.to_csv(output_csv, index=False)
    print(f"\n💾 Sauvegardé: {output_csv}")
    
    return df_multi


if __name__ == '__main__':
    print("="*70)
    print("🚀 EXPANSION DATASET - TOUTES LES IMAGES")
    print("="*70)
    
    # Train
    print("\n📝 TRAIN SET:")
    print("-"*70)
    df_train = create_multi_image_csv(
        'ML/data/India/train_clean.csv',
        'ML/data/India',
        'ML/data/India/train_multi.csv'
    )
    
    # Validation
    print("\n" + "="*70)
    print("📝 VALIDATION SET:")
    print("-"*70)
    df_val = create_multi_image_csv(
        'ML/data/India/val_clean.csv',
        'ML/data/India',
        'ML/data/India/val_multi.csv'
    )
    
    print("\n" + "="*70)
    print("✅ EXPANSION TERMINÉE")
    print("="*70)
    print(f"\n🎯 Dataset final:")
    print(f"   Train: {len(df_train)} images (×{len(df_train)/48:.1f})")
    print(f"   Val: {len(df_val)} images (×{len(df_val)/15:.1f})")
    print(f"   Total: {len(df_train) + len(df_val)} images")
