# healthguard/skin_lesion_analyzer.py
from typing import Dict, Optional, Union
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
import io
import logging

logger = logging.getLogger(__name__)


class SkinLesionAnalyzer:
    """
    Analyseur de lésions cutanées utilisant LLaVA fine-tuné sur HAM10000
    """
    
    def __init__(
        self, 
        model_id: str = "YuchengShi/LLaVA-v1.5-7B-HAM10000",
        device: Optional[str] = None
    ):
        """
        Initialise l'analyseur avec le modèle pré-entraîné
        
        Args:
            model_id: ID du modèle Hugging Face
            device: 'cuda', 'cpu' ou None (auto-détection)
        """
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        self.model = None
        self.processor = None
        self._is_loaded = False
        
        logger.info(f"Analyseur initialisé pour device: {self.device}")
    
    def load_model(self) -> None:
        """Charge le modèle en mémoire"""
        if self._is_loaded:
            logger.info("Modèle déjà chargé")
            return
        
        try:
            logger.info(f"Chargement du modèle {self.model_id}...")
            
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
            ).to(self.device)
            
            self.model.eval()
            
            # Optimisations GPU
            if self.device == "cuda":
                torch.backends.cudnn.benchmark = True
            
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            
            self._is_loaded = True
            logger.info("✅ Modèle chargé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement: {e}")
            raise
    
    def analyze(
        self, 
        image: Union[Image.Image, bytes, str],
        prompt: Optional[str] = None,
        max_tokens: int = 300
    ) -> Dict:
        """
        Analyse une image de lésion cutanée
        
        Args:
            image: Image PIL, bytes, ou chemin vers l'image
            prompt: Question personnalisée (optionnel)
            max_tokens: Nombre maximum de tokens pour la réponse
            
        Returns:
            Dictionnaire contenant le diagnostic et les métadonnées
        """
        if not self._is_loaded:
            self.load_model()
        
        # Conversion de l'image en PIL si nécessaire
        pil_image = self._prepare_image(image)
        
        # Prompt par défaut
        if prompt is None:
            prompt = "What type of skin lesion is this? Provide the diagnosis and explain your reasoning."
        
        # Préparation de la conversation
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"},
                ],
            },
        ]
        
        formatted_prompt = self.processor.apply_chat_template(
            conversation, 
            add_generation_prompt=True
        )
        
        # Inférence
        with torch.inference_mode():
            inputs = self.processor(
                images=pil_image,
                text=formatted_prompt,
                return_tensors='pt'
            ).to(self.device, self.dtype)
            
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=0.1,
                top_p=0.9
            )
        
        # Décodage
        diagnosis = self.processor.decode(output[0], skip_special_tokens=True)
        
        # Extraction du type de lésion
        lesion_type = self._extract_lesion_type(diagnosis)
        
        return {
            "diagnosis": diagnosis,
            "lesion_type": lesion_type,
            "model_id": self.model_id,
            "device": self.device
        }
    
    def _prepare_image(self, image: Union[Image.Image, bytes, str]) -> Image.Image:
        """Convertit différents formats d'image en PIL Image"""
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        
        elif isinstance(image, bytes):
            return Image.open(io.BytesIO(image)).convert("RGB")
        
        elif isinstance(image, str):
            # Chemin local ou URL
            if image.startswith(('http://', 'https://')):
                import requests
                response = requests.get(image, stream=True)
                return Image.open(io.BytesIO(response.content)).convert("RGB")
            else:
                return Image.open(image).convert("RGB")
        
        else:
            raise ValueError(f"Format d'image non supporté: {type(image)}")
    
    def _extract_lesion_type(self, diagnosis: str) -> str:
        """Extrait le type de lésion du diagnostic textuel"""
        lesion_mapping = {
            "melanoma": "MEL",
            "nevus": "NV",
            "nevi": "NV",
            "basal cell carcinoma": "BCC",
            "actinic keratosis": "AKIEC",
            "benign keratosis": "BKL",
            "dermatofibroma": "DF",
            "vascular lesion": "VASC"
        }
        
        diagnosis_lower = diagnosis.lower()
        
        for lesion_name, lesion_code in lesion_mapping.items():
            if lesion_name in diagnosis_lower:
                return lesion_code
        
        return "UNKNOWN"
    
    def unload_model(self) -> None:
        """Libère la mémoire du modèle"""
        if self._is_loaded:
            del self.model
            del self.processor
            torch.cuda.empty_cache() if self.device == "cuda" else None
            self._is_loaded = False
            logger.info("Modèle déchargé de la mémoire")
    
    def __enter__(self):
        """Support du context manager"""
        self.load_model()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Nettoyage automatique avec context manager"""
        self.unload_model()
