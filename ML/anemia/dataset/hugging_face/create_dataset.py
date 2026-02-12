import pandas as pd
from pathlib import Path
from PIL import Image
import numpy as np

def validate_image_for_ml(img_path, min_size=50, max_size=10000):
    """
    Vérifie si une image est viable pour le ML
    
    Args:
        img_path: Chemin vers l'image
        min_size: Taille minimale (largeur/hauteur)
        max_size: Taille maximale (largeur/hauteur)
    
    Returns:
        (bool, str): (est_valide, message_erreur)
    """
    try:
        # 1. Ouvre et vérifie l'intégrité basique
        with Image.open(img_path) as img:
            # Vérifie que c'est bien une image
            img.verify()
        
        # 2. Recharge l'image (verify() ferme le fichier)
        with Image.open(img_path) as img:
            # Vérifie le format
            if img.format not in ['JPEG', 'JPG', 'PNG', 'BMP', 'TIFF']:
                return False, f"Format non supporté: {img.format}"
            
            # Vérifie les dimensions
            width, height = img.size
            if width < min_size or height < min_size:
                return False, f"Trop petite: {width}x{height}"
            
            if width > max_size or height > max_size:
                return False, f"Trop grande: {width}x{height}"
            
            # Vérifie que l'image peut être convertie en RGB
            try:
                img_rgb = img.convert('RGB')
            except Exception as e:
                return False, f"Conversion RGB impossible: {e}"
            
            # 3. Vérifie que l'image a du contenu (pas juste du blanc/noir)
            img_array = np.array(img_rgb)
            
            # Check que ce n'est pas une image vide
            if img_array.size == 0:
                return False, "Image vide (0 pixels)"
            
            # Vérifie la variance (image pas complètement uniforme)
            variance = np.var(img_array)
            if variance < 1.0:  # Variance très faible = image uniforme
                return False, f"Variance trop faible: {variance:.2f}"
            
            # 4. Test de redimensionnement (simulation transform ML)
            try:
                img_resized = img_rgb.resize((224, 224), Image.LANCZOS)
                # Vérifie que le resize n'a pas produit une image noire
                resized_array = np.array(img_resized)
                if np.mean(resized_array) < 1.0:
                    return False, "Image noire après resize"
            except Exception as e:
                return False, f"Resize impossible: {e}"
            
            # 5. Vérifie la taille du fichier
            file_size = img_path.stat().st_size
            if file_size < 1000:  # Moins de 1KB
                return False, f"Fichier trop petit: {file_size} bytes"
            
            return True, "OK"
            
    except Exception as e:
        return False, f"Erreur ouverture: {str(e)[:50]}"


def create_csv_with_validation():
    """
    Crée un CSV avec validation complète des images
    """
    print("📊 CRÉATION DU CSV AVEC VALIDATION DES IMAGES")
    print("="*70)
    
    hf_dir = Path("API/ML/data/hugging-face")
    all_rows = []
    
    stats = {
        'total_scanned': 0,
        'valid': 0,
        'corrupted': 0,
        'errors': {}
    }
    
    for split_name in ['train', 'validation', 'test']:
        split_dir = hf_dir / split_name
        
        if not split_dir.exists():
            print(f"⚠️  {split_dir} non trouvé")
            continue
        
        print(f"\n📂 Processing {split_name}...")
        print("-" * 70)
        
        for class_folder in ['Anemia', 'NoAnemia']:
            class_dir = split_dir / class_folder
            
            if not class_dir.exists():
                print(f"   ⚠️  {class_dir} non trouvé")
                continue
            
            anemia_label = 1 if class_folder == 'Anemia' else 0
            
            # Liste toutes les images
            image_files = sorted(list(class_dir.glob('*.jpg')) + 
                               list(class_dir.glob('*.png')) + 
                               list(class_dir.glob('*.jpeg')))
            
            print(f"\n   {class_folder}: {len(image_files)} images à vérifier")
            
            valid_count = 0
            
            for idx, img_path in enumerate(image_files):
                stats['total_scanned'] += 1
                
                # Validation complète
                is_valid, error_msg = validate_image_for_ml(img_path)
                
                if is_valid:
                    # Image OK, ajoute au CSV
                    relative_path = f"hugging-face/{split_name}/{class_folder}/{img_path.name}"
                    
                    all_rows.append({
                        'patient_number': f'HF_{split_name}_{class_folder}_{valid_count}',
                        'image_path': relative_path,
                        'image_name': img_path.name,
                        'anemia_label': anemia_label,
                        'hgb': -1.0,
                        'gender': 'U',
                        'age': -1,
                        'n_images': 1,
                        'source': 'huggingface',
                        'split': split_name,
                        'class': class_folder
                    })
                    
                    valid_count += 1
                    stats['valid'] += 1
                else:
                    # Image corrompue, log l'erreur
                    print(f"      ❌ {img_path.name}: {error_msg}")
                    stats['corrupted'] += 1
                    
                    # Compte les types d'erreurs
                    error_type = error_msg.split(':')[0]
                    stats['errors'][error_type] = stats['errors'].get(error_type, 0) + 1
            
            print(f"      ✅ {valid_count} images valides")
            if len(image_files) - valid_count > 0:
                print(f"      ❌ {len(image_files) - valid_count} images rejetées")
    
    # Crée le DataFrame
    df = pd.DataFrame(all_rows)
    
    print(f"\n" + "="*70)
    print(f"📊 STATISTIQUES DE VALIDATION:")
    print(f"="*70)
    print(f"\nTotal scanné: {stats['total_scanned']} images")
    print(f"✅ Valides: {stats['valid']} ({stats['valid']/stats['total_scanned']*100:.1f}%)")
    print(f"❌ Corrompues: {stats['corrupted']} ({stats['corrupted']/stats['total_scanned']*100:.1f}%)")
    
    if stats['errors']:
        print(f"\n📋 Types d'erreurs rencontrées:")
        for error_type, count in sorted(stats['errors'].items(), key=lambda x: x[1], reverse=True):
            print(f"   - {error_type}: {count}")
    
    print(f"\n" + "="*70)
    print(f"📊 RÉSUMÉ DATASET FINAL:")
    print(f"="*70)
    print(f"\nTotal images valides: {len(df)}")
    
    print(f"\n📈 Par split:")
    for split in ['train', 'validation', 'test']:
        count = (df['split'] == split).sum()
        if count > 0:
            print(f"   {split}: {count} images")
    
    print(f"\n📊 Distribution globale:")
    print(f"   Anémie (1): {(df['anemia_label']==1).sum()} images")
    print(f"   Non-anémie (0): {(df['anemia_label']==0).sum()} images")
    
    print(f"\n📊 Distribution par split:")
    for split in ['train', 'validation', 'test']:
        df_split = df[df['split'] == split]
        if len(df_split) > 0:
            anem = (df_split['anemia_label']==1).sum()
            no_anem = (df_split['anemia_label']==0).sum()
            print(f"   {split}: Anémie={anem}, Non-anémie={no_anem}")
    
    # Sauvegarde le CSV principal
    csv_path = hf_dir / 'anemia_eyes_all.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n💾 CSV sauvegardé: {csv_path}")
    
    # Sauvegarde aussi un log des images corrompues
    if stats['corrupted'] > 0:
        corrupted_log = hf_dir / 'corrupted_images.txt'
        with open(corrupted_log, 'w') as f:
            f.write(f"Images corrompues trouvées: {stats['corrupted']}\n")
            f.write(f"Date: {pd.Timestamp.now()}\n\n")
            for error_type, count in stats['errors'].items():
                f.write(f"{error_type}: {count}\n")
        print(f"📝 Log des erreurs: {corrupted_log}")
    
    return df, stats


def verify_csv_images(csv_path):
    """
    Vérifie que toutes les images du CSV sont accessibles et valides
    """
    print(f"\n🔍 VÉRIFICATION FINALE DU CSV: {csv_path}")
    print("="*70)
    
    df = pd.read_csv(csv_path)
    base_dir = Path("API/ML/data")
    
    invalid_count = 0
    
    for idx, row in df.iterrows():
        img_path = base_dir / row['image_path']
        
        if not img_path.exists():
            print(f"❌ Fichier manquant: {row['image_path']}")
            invalid_count += 1
        else:
            is_valid, error_msg = validate_image_for_ml(img_path)
            if not is_valid:
                print(f"❌ Image invalide: {row['image_path']} - {error_msg}")
                invalid_count += 1
    
    if invalid_count == 0:
        print(f"✅ Toutes les {len(df)} images du CSV sont valides et accessibles!")
    else:
        print(f"\n⚠️  {invalid_count} images invalides trouvées sur {len(df)}")
    
    return invalid_count == 0


if __name__ == '__main__':
    # Crée le CSV avec validation
    df, stats = create_csv_with_validation()
    
    # Vérification finale
    if len(df) > 0:
        print("\n" + "="*70)
        verify_csv_images('API/ML/data/hugging-face/anemia_eyes_all.csv')
