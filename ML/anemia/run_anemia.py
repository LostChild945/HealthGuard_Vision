import torch
import torch.nn as nn
from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from pathlib import Path
from typing import List, Union, Dict

def analyze_images_anemia(
    image_paths: Union[str, List[str]], 
    model_path: str = "models/model_anemie.pt",
    device: str = None
) -> List[Dict]:
    """
    Analyse des images pour détecter l'anémie avec le modèle EfficientNet-B0.
    
    Args:
        image_paths: Chemin(s) vers la/les image(s) à analyser
        model_path: Chemin vers le modèle PyTorch (.pt)
        device: Device à utiliser ('cuda', 'cpu', ou None pour auto-détection)
    
    Returns:
        Liste de dictionnaires avec les résultats pour chaque image:
        {
            'image': nom du fichier,
            'prediction': classe prédite (0=sain, 1=anémie),
            'confidence': score de confiance en %,
            'probabilities': {'sain': %, 'anemie': %},
            'result': interprétation textuelle
        }
    """
    
    # Gestion du device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    # Conversion en liste si une seule image
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    
    # Chargement du checkpoint
    print(f"Chargement du modèle depuis {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extraction du state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        epoch = checkpoint.get('epoch', 'N/A')
        val_acc = checkpoint.get('val_acc', 'N/A')
        val_f1 = checkpoint.get('val_f1', 'N/A')
        val_acc_str = f"{val_acc:.4f}" if isinstance(val_acc, float) else str(val_acc)
        val_f1_str = f"{val_f1:.4f}" if isinstance(val_f1, float) else str(val_f1)
        print(f"  Epoch: {epoch} | Val Acc: {val_acc_str} | Val F1: {val_f1_str}")
    else:
        state_dict = checkpoint
    
    # Recréer l'architecture exacte (identique au train)
    model = timm.create_model(
        'efficientnet_b0',
        pretrained=False,
        num_classes=2,
        drop_rate=0.3  # Même dropout que dans le train
    )
    
    # Charger les poids
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    # Transformations (mêmes que ton dataset avec augment=False)
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    
    print(f"✅ Modèle chargé sur {device}")
    print(f"📊 Prêt à analyser {len(image_paths)} image(s)\\n")
    
    results = []
    
    # Analyse de chaque image
    with torch.no_grad():
        for img_path in image_paths:
            try:
                # Chargement et prétraitement
                image = Image.open(img_path).convert('RGB')
                img_tensor = transform(image).unsqueeze(0).to(device)
                
                # Inférence
                output = model(img_tensor)
                
                # Calcul des probabilités
                probabilities = torch.softmax(output, dim=1)
                confidence, prediction = torch.max(probabilities, 1)
                
                # Interprétation
                pred_class = prediction.item()
                conf_score = confidence.item()
                probs = probabilities[0].cpu().numpy()
                
                result_text = "Anémie détectée" if pred_class == 1 else "Pas d'anémie détectée"
                
                result = {
                    'image': Path(img_path).name,
                    'prediction': pred_class,
                    'confidence': round(conf_score * 100, 2),
                    'probabilities': {
                        'sain': round(probs[0] * 100, 2),
                        'anemie': round(probs[1] * 100, 2)
                    },
                    'result': result_text
                }
                
                results.append(result)
                
                # Affichage avec code couleur
                emoji = "🔴" if pred_class == 1 else "🟢"
                print(f"{emoji} {Path(img_path).name}")
                print(f"   {result_text} (confiance: {conf_score*100:.2f}%)")
                print(f"   Probabilités: Sain={probs[0]*100:.2f}% | Anémie={probs[1]*100:.2f}%\\n")
                
            except Exception as e:
                results.append({
                    'image': Path(img_path).name,
                    'error': str(e),
                    'result': 'Erreur lors de l analyse'
                })
                print(f"❌ {Path(img_path).name}: Erreur - {e}\\n")
    
    return results


if __name__ == "__main__":
    result = analyze_images_anemia("ML/data/hugging-face/test/Anemia/Anemia_test.7.jpg")
    
    images = [
        "ML/data/hugging-face/test/Anemia/Anemia_test.0.jpg",
        "ML/data/hugging-face/test/Anemia/Anemia_test.1.jpg",
        "ML/data/hugging-face/test/Anemia/Anemia_test.5.jpg"
    ]
    results = analyze_images_anemia(images)
    
    for r in results:
        if 'error' not in r:
            print(f"{r['image']}: {r['result']} ({r['confidence']}%)")
        else:
            print(f"{r['image']}: Erreur")
