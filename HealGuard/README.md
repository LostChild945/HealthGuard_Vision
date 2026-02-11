# Application Mobile d'Analyse de Photos

Application mobile professionnelle développée avec React Native et Expo pour la prise et l'analyse de photos via API.

## Fonctionnalités

- **Prise de photos** : Interface caméra intuitive avec prévisualisation
- **Gestion de galerie** : Visualisation et suppression des photos prises
- **Envoi sécurisé** : Upload des photos via middleware API
- **Résultats d'analyse** : Affichage clair des résultats de l'API

## Structure du Projet

```
.
├── app/
│   ├── (tabs)/
│   │   ├── _layout.tsx       # Configuration des onglets
│   │   ├── index.tsx          # Écran caméra
│   │   └── gallery.tsx        # Écran galerie
│   ├── _layout.tsx            # Layout racine
│   └── results.tsx            # Écran résultats
├── components/
│   ├── Button.tsx             # Composant bouton réutilisable
│   └── PhotoItem.tsx          # Composant photo réutilisable
├── contexts/
│   └── PhotoContext.tsx       # Gestion d'état des photos
├── services/
│   └── photoUploadService.ts  # Service API (middleware)
└── types/
    ├── api.ts                 # Types API
    └── env.d.ts               # Types variables d'environnement
```

## Configuration

### Variables d'environnement

Modifiez le fichier `.env` à la racine du projet :

```env
EXPO_PUBLIC_API_KEY=votre_clé_api
EXPO_PUBLIC_API_BASE_URL=https://votre-api.com
```

### Installation

```bash
npm install
```

### Lancement

```bash
npx expo start
```

## Architecture

### Principe de Responsabilité Unique (SRP)

Le code respecte le principe SRP avec :

- **Services** : Logique métier et appels API isolés
- **Composants** : UI réutilisable et indépendante
- **Contextes** : Gestion d'état centralisée
- **Types** : Définitions TypeScript séparées

### Middleware API

Le service `photoUploadService.ts` gère :
- Configuration sécurisée de la clé API
- Appels HTTP vers l'endpoint `/upload`
- Gestion des erreurs
- Formatage des données

### Gestion d'État

Le `PhotoContext` fournit :
- Ajout de photos
- Suppression de photos
- Nettoyage de la galerie
- État partagé entre les écrans

## Format de l'API

### Requête

```json
POST /upload
{
  "key": "API_KEY",
  "imgs": ["uri1", "uri2", ...]
}
```

### Réponse

```json
{
  "reponse": {
    "message": "Résultat de l'analyse"
  }
}
```

## Technologies

- React Native 0.81.4
- Expo SDK 54
- Expo Router 6
- TypeScript
- Expo Camera
- Lucide Icons
