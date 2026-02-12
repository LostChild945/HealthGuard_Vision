import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import timm
from pathlib import Path

class AnemiaPredictor:
    def __init__(self, checkpoint_path, device='cuda'):
        """
        Prédicateur d'anémie depuis checkpoint
        
        Args:
            checkpoint_path: Chemin vers best_model.pt
            device: 'cuda' ou 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Charge le checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = checkpoint['config']
        
        print(f"🔧 Chargement du modèle...")
        print(f"   Epoch: {checkpoint['epoch']}")
        print(f"   Val Acc: {checkpoint['val_acc']:.4f}")
        print(f"   Val F1: {checkpoint['val_f1']:.4f}")
        print(f"   Val Recall: {checkpoint['val_recall']:.4f}")
        
        # Crée le modèle
        self.model = timm.create_model(
            config['model_name'],
            pretrained=False,
            num_classes=2,
            drop_rate=config.get('dropout', 0.3)
        )
        
        # Charge les poids
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Transform pour preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        print(f"✅ Modèle chargé sur {self.device}")
    
    @torch.no_grad()
    def predict(self, image_path, threshold=0.5):
        """
        Prédiction sur une image
        
        Args:
            image_path: Chemin vers l'image
            threshold: Seuil de décision (défaut 0.5)
        
        Returns:
            dict avec prediction, proba, confidence
        """
        # Charge et préprocess l'image
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Prédiction
        output = self.model(img_tensor)
        probs = torch.softmax(output, dim=1)[0]
        
        prob_no_anemia = probs[0].item()
        prob_anemia = probs[1].item()
        
        # Décision
        prediction = 1 if prob_anemia >= threshold else 0
        confidence = max(prob_no_anemia, prob_anemia)
        
        result = {
            'prediction': prediction,
            'prediction_label': 'Anémie' if prediction == 1 else 'Non-anémie',
            'prob_anemia': prob_anemia,
            'prob_no_anemia': prob_no_anemia,
            'confidence': confidence,
            'threshold': threshold
        }
        
        return result
    
    def predict_batch(self, image_paths, threshold=0.5):
        """
        Prédiction sur plusieurs images
        """
        results = []
        for img_path in image_paths:
            try:
                result = self.predict(img_path, threshold)
                result['image_path'] = str(img_path)
                results.append(result)
            except Exception as e:
                print(f"❌ Erreur sur {img_path}: {e}")
        return results


def test_single_image():
    """
    Test sur une image unique
    """
    # Charge le modèle
    predictor = AnemiaPredictor('models/india_anemia/best_model.pt')
    
    # Test sur une image
    image_path = 'API/ML/data/hugging-face/validation/NoAnemia/NoAnemia_val.10.jpg'
    print(f"\n🔍 Test sur: {image_path}")
    print("-" * 70)
    
    result = predictor.predict(image_path)
    
    print(f"\n📊 Résultat:")
    print(f"   Prédiction: {result['prediction_label']}")
    print(f"   Confiance: {result['confidence']*100:.2f}%")
    print(f"   Probabilités:")
    print(f"      Non-anémie: {result['prob_no_anemia']*100:.2f}%")
    print(f"      Anémie: {result['prob_anemia']*100:.2f}%")
    
    # Interprétation médicale
    if result['confidence'] > 0.8:
        conf_level = "Haute confiance ✅"
    elif result['confidence'] > 0.6:
        conf_level = "Confiance modérée ⚠️"
    else:
        conf_level = "Faible confiance - Vérification recommandée ❌"
    
    print(f"\n💡 Interprétation: {conf_level}")


if __name__ == '__main__':
    test_single_image()
