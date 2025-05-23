# Emotion-Recognition-for-Medical-Emergency-Purposes-and-Improved-Patient-Care
# MedTrackAI - Système d'Analyse Multimodale des Émotions en Urgence Médicale

MedTrackAI est un système d'intelligence artificielle multimodale conçu pour analyser les émotions des patients en milieu médical d'urgence. Le système intègre l'analyse vidéo (expression faciale) et audio (voix) pour fournir une évaluation en temps réel de l'état émotionnel du patient, avec des alertes en cas de conditions critiques.

## Contexte et Motivation

Dans les environnements d'urgence médicale, il est crucial d'évaluer rapidement l'état émotionnel des patients pour une prise en charge efficace. Cependant, la reconnaissance émotionnelle traditionnelle par des moyens humains peut être lente et subjective. Ce projet vise à automatiser ce processus en utilisant des modèles d'IA capables d'analyser les émotions via la vidéo et l'audio.

---

## Objectifs du Projet

1. **Développer un modèle d'IA multimodal** pour l'analyse des émotions (vision + audio).
2. **Mettre en place un système d'alerte** pour les états critiques (ex. détresse émotionnelle, angoisse).
3. **Construire une interface web interactive** pour la gestion des flux vidéo et audio en temps réel.

---

## Fonctionnalités

- **Capture vidéo et analyse faciale** : Utilisation de modèles EfficientNet pour détecter les émotions sur les visages des patients.
- **Capture audio et analyse vocale** : Utilisation de CNN pour analyser les émotions à partir des spectrogrammes des enregistrements vocaux.
- **Fusion des résultats** : Combinaison des prédictions des deux flux pour fournir une évaluation globale de l'état émotionnel du patient.
- **Système d'alerte** : Déclenchement d'une alerte visuelle sur l'interface si un état critique est détecté.

---

## Architecture du Système

1. **Frontend** : Interface web interactive, permettant à l'utilisateur d'interagir avec le système et de visualiser les résultats en temps réel.
2. **Backend** : Serveur Flask qui coordonne les flux de données et gère l'analyse des émotions.
3. **Modules** :
    - `camera.py` : Capture vidéo et analyse faciale.
    - `voice.py` : Capture audio et analyse vocale.
4. **Fusion des résultats** : La fusion des données vidéo et audio permet une évaluation plus précise de l'état émotionnel.

---

## Prérequis

Pour faire fonctionner ce projet, vous aurez besoin des éléments suivants :

- **Python 3.x** (3.6 ou supérieur)
- **Bibliothèques Python** :
    - TensorFlow / Keras
    - Flask
    - OpenCV
    - NumPy
    - Librosa (pour l’analyse vocale)
    - Matplotlib
    - Scikit-learn
    - etc.

Vous pouvez installer les dépendances en utilisant `pip` :
```bash
pip install -r requirements.txt
