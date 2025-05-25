# Churn-Clients
Prédiction de résiliation client TelcoNova avec le Deep Learning


## Modèle de Prédiction de Churn - MLP Résiduel

##Description

Ce projet implémente un modèle de réseau de neurones multi-couches (MLP) avec connexions résiduelles pour prédire le churn (désabonnement) des clients. Le modèle utilise PyTorch et intègre plusieurs techniques avancées d'apprentissage automatique pour gérer le déséquilibre des classes et optimiser les performances.

## Architecture du Modèle

### ResidualChurnMLP

- **Type**: Perceptron Multi-Couches avec connexions résiduelles
- **Couches d'entrée**: 15 features → 128 neurones
- **Bloc résiduel**: 128 → 128 → 128 neurones avec connexion résiduelle
- **Couches de sortie**: 128 → 64 → 32 → 1 neurone
- **Fonction d'activation**: LeakyReLU (pente négative = 0.1)
- **Normalisation**: BatchNorm1d après chaque couche linéaire
- **Régularisation**: Dropout (0.3, 0.3, 0.2)
- **Paramètres totaux**: 46,401

### Caractéristiques Techniques

- **Framework**: PyTorch
- **Fonction de perte**: BCEWithLogitsLoss avec pondération des classes
- **Optimiseur**: Adam (lr=0.0005, weight_decay=1e-5)
- **Scheduler**: ReduceLROnPlateau
- **Initialisation des poids**: Kaiming (He) normal

## Préparation des Données

### Préprocessing

1. **Normalisation**: StandardScaler sur les variables numériques (`tenure`, `MonthlyCharges`)
2. **Gestion du déséquilibre**: SMOTE (Synthetic Minority Oversampling Technique)
   - Ratio d'échantillonnage: 0.5
   - Distribution finale: 33.33% de classe positive
3. **Division des données**:
   - Entraînement: 68% (5,275 échantillons après SMOTE)
   - Validation: 17% (1,198 échantillons)
   - Test: 15% (1,057 échantillons)

### Features (15 variables)

- `Partner`: Présence d'un partenaire (0/1)
- `Dependents`: Présence de personnes à charge (0/1)
- `tenure`: Durée d'abonnement (normalisée)
- `PaperlessBilling`: Facturation électronique (0/1)
- `MonthlyCharges`: Charges mensuelles (normalisées)
- `SeniorCitizen_Yes`: Client senior (0/1)
- `InternetService_Fiber optic`: Service fibre optique (0/1)
- `InternetService_No`: Pas de service internet (0/1)
- `OnlineSecurity_No internet service`: Pas de sécurité en ligne (0/1)
- `OnlineSecurity_Yes`: Sécurité en ligne activée (0/1)
- `TechSupport_Yes`: Support technique activé (0/1)
- `Contract_One year`: Contrat d'un an (0/1)
- `Contract_Two year`: Contrat de deux ans (0/1)
- `PaymentMethod_Credit card (automatic)`: Paiement par carte automatique (0/1)
- `PaymentMethod_Electronic check`: Paiement par chèque électronique (0/1)

## Performances du Modèle

### Métriques sur l'ensemble de test

- **Accuracy**: 76.25%
- **Precision**: 53.75%
- **Recall**: 74.29%
- **F1-score**: 62.37%
- **AUC-ROC**: 84.09%

## Installation et Utilisation

### Dépendances

```python
torch
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
pickle
```

## Configuration d'Entraînement

### Hyperparamètres principaux

- **Batch size**: 32
- **Learning rate initial**: 0.0005
- **Époques maximum**: 100
- **Early stopping patience**: 15
- **Weight decay**: 1e-5
- **Dropout rates**: [0.3, 0.3, 0.2]

### Gestion du déséquilibre

1. **SMOTE**: Augmentation synthétique de la classe minoritaire
2. **Pondération des classes**: pos_weight dans BCEWithLogitsLoss
3. **Métriques équilibrées**: Focus sur F1-score et AUC-ROC

## Points Forts du Modèle

1. **Architecture robuste**: Connexions résiduelles pour éviter la dégradation du gradient
2. **Gestion du déséquilibre**: Combinaison SMOTE + pondération des classes
3. **Régularisation multiple**: Dropout, BatchNorm, Weight decay
4. **Optimisation avancée**: Scheduler adaptatif + Early stopping
5. **Reproductibilité**: Seeds fixés pour tous les composants aléatoires
