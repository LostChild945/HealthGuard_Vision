import torch
import torch.nn as nn
from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from pathlib import Path
from typing import Optional, Union, Dict
import io
import logging

logger = logging.getLogger(__name__)


class AnemiaAnalyzer:
    """
    Analyseur d'anémie utilisant EfficientNet-B0 entraîné sur des images d'yeux
    """
    
    def __init__(
        self, 
        model_path: str = "models/model_anemie.pt",
        device: Optional[str] = None
    ):
        """
        Initialise l'analyseur avec le modèle entraîné
        
        Args:
            model_path: Chemin vers le modèle PyTorch (.pt)
            device: 'cuda', 'cpu' ou None (auto-détection)
        """
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device_obj = torch.device(self.device)
        
        self.model = None
        self.transform = None
        self.model_info = {}
        self._is_loaded = False
        
        logger.info(f"Analyseur initialisé pour device: {self.device}")
    
    def load_model(self) -> None:
        """Charge le modèle en mémoire"""
        if self._is_loaded:
            logger.info("Modèle déjà chargé")
            return
        
        try:
            logger.info(f"Chargement du modèle depuis {self.model_path}...")
            
            # Chargement du checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device_obj)
            
            # Extraction du state_dict et métadonnées
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                self.model_info = {
                    'epoch': checkpoint.get('epoch', 'N/A'),
                    'val_acc': checkpoint.get('val_acc', 'N/A'),
                    'val_f1': checkpoint.get('val_f1', 'N/A')
                }
                
                # Affichage des métriques
                val_acc = self.model_info['val_acc']
                val_f1 = self.model_info['val_f1']
                val_acc_str = f"{val_acc:.4f}" if isinstance(val_acc, float) else str(val_acc)
                val_f1_str = f"{val_f1:.4f}" if isinstance(val_f1, float) else str(val_f1)
                logger.info(f"  Epoch: {self.model_info['epoch']} | Val Acc: {val_acc_str} | Val F1: {val_f1_str}")
            else:
                state_dict = checkpoint
            
            # Recréer l'architecture exacte
            self.model = timm.create_model(
                'efficientnet_b0',
                pretrained=False,
                num_classes=2,
                drop_rate=0.3
            )
            
            # Charger les poids
            self.model.load_state_dict(state_dict)
            self.model = self.model.to(self.device_obj)
            self.model.eval()
            
            # Préparer les transformations
            config = resolve_data_config({}, model=self.model)
            self.transform = create_transform(**config)
            
            self._is_loaded = True
            logger.info("✅ Modèle chargé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement: {e}")
            raise
    
    def analyze(
        self, 
        image: Union[Image.Image, bytes, str]
    ) -> Dict:
        """
        Analyse une image pour détecter l'anémie
        
        Args:
            image: Image PIL, bytes, ou chemin vers l'image
            
        Returns:
            Dictionnaire contenant la prédiction et les probabilités:
            {
                'image': nom du fichier,
                'prediction': classe prédite (0=sain, 1=anémie),
                'confidence': score de confiance en %,
                'probabilities': {'sain': %, 'anemie': %},
                'result': interprétation textuelle
            }
        """
        if not self._is_loaded:
            self.load_model()
        
        # Conversion de l'image en PIL si nécessaire
        pil_image, img_name = self._prepare_image(image)
        
        # Prétraitement
        img_tensor = self.transform(pil_image).unsqueeze(0).to(self.device_obj)
        
        # Inférence
        with torch.no_grad():
            output = self.model(img_tensor)
            
            # Calcul des probabilités
            probabilities = torch.softmax(output, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
            
            # Extraction des résultats
            pred_class = prediction.item()
            conf_score = confidence.item()
            probs = probabilities[0].cpu().numpy()
            
            result_text = "Anémie détectée" if pred_class == 1 else "Pas d'anémie détectée"
            
            return {
                'image': img_name,
                'prediction': pred_class,
                'confidence': round(conf_score * 100, 2),
                'probabilities': {
                    'sain': round(float(probs[0]) * 100, 2),
                    'anemie': round(float(probs[1]) * 100, 2)
                },
                'result': result_text
            }
    
    def _prepare_image(self, image: Union[Image.Image, bytes, str]) -> tuple[Image.Image, str]:
        """
        Convertit différents formats d'image en PIL Image
        
        Returns:
            (image_pil, nom_fichier)
        """
        if isinstance(image, Image.Image):
            return image.convert("RGB"), "uploaded_image.jpg"
        
        elif isinstance(image, bytes):
            pil_img = Image.open(io.BytesIO(image)).convert("RGB")
            return pil_img, "uploaded_image.jpg"
        
        elif isinstance(image, str):
            # Chemin local ou URL
            if image.startswith(('http://', 'https://')):
                import requests
                response = requests.get(image, stream=True)
                pil_img = Image.open(io.BytesIO(response.content)).convert("RGB")
                return pil_img, Path(image).name
            else:
                pil_img = Image.open(image).convert("RGB")
                return pil_img, Path(image).name
        
        else:
            raise ValueError(f"Format d'image non supporté: {type(image)}")
    
    def get_model_info(self) -> Dict:
        """Retourne les informations du modèle chargé"""
        return self.model_info
    
    def unload_model(self) -> None:
        """Libère la mémoire du modèle"""
        if self._is_loaded:
            del self.model
            del self.transform
            if self.device == "cuda":
                torch.cuda.empty_cache()
            self._is_loaded = False
            logger.info("Modèle déchargé de la mémoire")
    
    def __enter__(self):
        """Support du context manager"""
        self.load_model()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Nettoyage automatique avec context manager"""
        self.unload_model()


# Fonction wrapper pour compatibilité avec l'ancienne API
def analyze_images_anemia(
    image_paths: Union[str, list[str]], 
    model_path: str = "models/model_anemie.pt",
    device: str = None
) -> list[Dict]:
    """
    Analyse des images pour détecter l'anémie avec le modèle EfficientNet-B0.
    
    Version wrapper pour compatibilité avec l'ancienne API fonctionnelle.
    Utilise la classe AnemiaAnalyzer en interne.
    
    Args:
        image_paths: Chemin(s) vers la/les image(s) à analyser
        model_path: Chemin vers le modèle PyTorch (.pt)
        device: Device à utiliser ('cuda', 'cpu', ou None pour auto-détection)
    
    Returns:
        Liste de dictionnaires avec les résultats pour chaque image
    """
    # Conversion en liste si une seule image
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    
    # Initialisation de l'analyseur
    analyzer = AnemiaAnalyzer(model_path=model_path, device=device)
    analyzer.load_model()
    
    print(f"📊 Prêt à analyser {len(image_paths)} image(s)\n")
    
    results = []
    
    # Analyse de chaque image
    for img_path in image_paths:
        try:
            result = analyzer.analyze(img_path)
            results.append(result)
            
            # Affichage avec code couleur
            emoji = "🔴" if result['prediction'] == 1 else "🟢"
            print(f"{emoji} {result['image']}")
            print(f"   {result['result']} (confiance: {result['confidence']}%)")
            print(f"   Probabilités: Sain={result['probabilities']['sain']}% | Anémie={result['probabilities']['anemie']}%\n")
            
        except Exception as e:
            results.append({
                'image': Path(img_path).name,
                'error': str(e),
                'result': 'Erreur lors de l\'analyse'
            })
            print(f"❌ {Path(img_path).name}: Erreur - {e}\n")
    
    return results


if __name__ == "__main__":
    # Test 1: Utilisation avec la classe (recommandé)
    print("=== Test avec la classe AnemiaAnalyzer ===\n")
    
    analyzer = AnemiaAnalyzer(model_path="models/model_anemie.pt")
    analyzer.load_model()
    
    # Analyse d'une image
    result = analyzer.analyze("ML/data/hugging-face/test/Anemia/Anemia_test.7.jpg")
    print(f"Résultat: {result['result']} - Confiance: {result['confidence']}%\n")
    
    # Context manager (charge et décharge automatiquement)
    print("=== Test avec context manager ===\n")
    with AnemiaAnalyzer(model_path="models/model_anemie.pt") as analyzer:
        result = analyzer.analyze("ML/data/hugging-face/test/Anemia/Anemia_test.0.jpg")
        print(f"{result['image']}: {result['result']}\n")
    
    # Test 2: Utilisation avec la fonction wrapper (compatibilité)
    print("=== Test avec la fonction wrapper (ancienne API) ===\n")
    
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