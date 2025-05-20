"""
Analyse et Prétraitement des Données pour la Prédiction de Churn
=================================================================
Ce script réalise l'analyse exploratoire des données et le prétraitement
pour préparer les données à l'entraînement d'un modèle de prédiction de churn.
"""

import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
# ============================================================================
# 1. CONFIGURATION ET IMPORTS
# ============================================================================
import pandas as pd
import seaborn as sns
import tensorflow as tf
from IPython.display import HTML, display
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Répétabilité
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# Création du dossier pour les visualisations
VISUALIZATIONS_DIR = "visualisations"
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)


# ============================================================================
# 2. CHARGEMENT ET EXPLORATION DES DONNÉES
# ============================================================================
def load_and_explore_data(csv_path):
    """
    Charge les données depuis un fichier CSV et effectue une exploration initiale.

    Args:
        csv_path (str): Chemin vers le fichier CSV.

    Returns:
        pandas.DataFrame: DataFrame nettoyé.
    """
    # Chargement des données
    df = pd.read_csv(csv_path)
    print(f"Dimensions du dataset: {df.shape}")

    # Aperçu des données
    print("\nAperçu des premières lignes:")
    print(df.head())

    # Informations sur les colonnes
    print("\nInformations sur les colonnes:")
    print(df.info())

    # Distribution de la variable cible
    print("\nDistribution de la variable Churn:")
    print(df["Churn"].value_counts(normalize=True).round(3))

    # Conversion de TotalCharges en numérique
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    print(f"\nValeurs manquantes dans TotalCharges: {df['TotalCharges'].isna().sum()}")

    # Imputation des valeurs manquantes
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Statistiques descriptives des variables numériques
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    print("\nStatistiques descriptives des variables numériques:")
    print(df[numeric_cols].describe())

    # Visualisation de la distribution des variables numériques
    plt.figure(figsize=(15, 5))
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(1, 3, i)
        sns.histplot(data=df, x=col, hue="Churn", multiple="stack")
        plt.title(f"Distribution de {col}")
    plt.tight_layout()

    # Sauvegarde de la figure
    plt.savefig(
        os.path.join(VISUALIZATIONS_DIR, "distribution_variables_numeriques.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # Corrélation entre variables numériques
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
    plt.title("Matrice de corrélation des variables numériques")

    # Sauvegarde de la figure
    plt.savefig(
        os.path.join(VISUALIZATIONS_DIR, "matrice_correlation.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # Analyse des variables catégorielles
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    categorical_cols.remove("customerID")  # On ignore l'ID client

    print("\nDistribution des variables catégorielles:")
    for col in categorical_cols:
        print(f"\n{col}:")
        print(df[col].value_counts(normalize=True).round(3))

    # Visualisation des variables catégorielles les plus importantes
    plt.figure(figsize=(15, 5))

    # Contract vs Churn
    plt.subplot(1, 3, 1)
    sns.countplot(data=df, x="Contract", hue="Churn")
    plt.title("Churn par type de contrat")
    plt.xticks(rotation=45)

    # InternetService vs Churn
    plt.subplot(1, 3, 2)
    sns.countplot(data=df, x="InternetService", hue="Churn")
    plt.title("Churn par service Internet")
    plt.xticks(rotation=45)

    # PaymentMethod vs Churn
    plt.subplot(1, 3, 3)
    sns.countplot(data=df, x="PaymentMethod", hue="Churn")
    plt.title("Churn par méthode de paiement")
    plt.xticks(rotation=45)

    plt.tight_layout()

    # Sauvegarde de la figure
    plt.savefig(
        os.path.join(VISUALIZATIONS_DIR, "variables_categorielles_importantes.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # Visualisation supplémentaire: Churn par ancienneté (tenure)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="Churn", y="tenure")
    plt.title("Distribution de l'ancienneté par statut de Churn")

    # Sauvegarde de la figure
    plt.savefig(
        os.path.join(VISUALIZATIONS_DIR, "anciennete_par_churn.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # Visualisation supplémentaire: Churn par charges mensuelles
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
    plt.title("Distribution des charges mensuelles par statut de Churn")

    # Sauvegarde de la figure
    plt.savefig(
        os.path.join(VISUALIZATIONS_DIR, "charges_mensuelles_par_churn.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    return df


# ============================================================================
# 3. PRÉTRAITEMENT DES DONNÉES
# ============================================================================
def preprocess_data(df):
    """
    Prépare les données pour l'entraînement du modèle.

    Args:
        df (pandas.DataFrame): DataFrame nettoyé.

    Returns:
        tuple: X_train_prep, X_val_prep, X_test_prep, y_train, y_val, y_test, preprocessor
    """
    # Séparation features/target
    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(["Churn", "customerID"], axis=1)

    # Définition des colonnes
    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    # Création du preprocesseur
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(drop="first", sparse_output=False),
                categorical_features,
            ),
        ]
    )

    # Split des données (train/val/test stratifié)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=SEED, stratify=y_temp
    )

    # Fit/transform sur train, transform sur val/test
    X_train_prep = preprocessor.fit_transform(X_train)
    X_val_prep = preprocessor.transform(X_val)
    X_test_prep = preprocessor.transform(X_test)

    print("Shapes après prétraitement :")
    print(f"Train : {X_train_prep.shape}")
    print(f"Val   : {X_val_prep.shape}")
    print(f"Test  : {X_test_prep.shape}")

    # Visualisation de la distribution des classes dans les ensembles
    plt.figure(figsize=(10, 6))

    # Calculer les proportions de churn dans chaque ensemble
    train_churn_prop = y_train.mean() * 100
    val_churn_prop = y_val.mean() * 100
    test_churn_prop = y_test.mean() * 100

    # Créer le graphique
    sns.barplot(
        x=["Train", "Validation", "Test"],
        y=[train_churn_prop, val_churn_prop, test_churn_prop],
    )
    plt.title("Pourcentage de Churn dans chaque ensemble")
    plt.ylabel("Pourcentage (%)")
    plt.ylim(0, 100)

    # Ajouter les valeurs sur les barres
    for i, v in enumerate([train_churn_prop, val_churn_prop, test_churn_prop]):
        plt.text(i, v + 1, f"{v:.1f}%", ha="center")

    # Sauvegarde de la figure
    plt.savefig(
        os.path.join(VISUALIZATIONS_DIR, "distribution_churn_ensembles.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    return (
        X_train_prep,
        X_val_prep,
        X_test_prep,
        y_train,
        y_val,
        y_test,
        preprocessor,
        X_train,
        X_val,
        X_test,
    )


def visualize_encoding(df, preprocessor, X_train_prep, X_train):
    """
    Visualise le résultat de l'encodage pour un exemple.

    Args:
        df (pandas.DataFrame): DataFrame original.
        preprocessor (ColumnTransformer): Préprocesseur utilisé.
        X_train_prep (numpy.ndarray): Données d'entraînement prétraitées.
        X_train (pandas.DataFrame): Données d'entraînement avant prétraitement.
    """
    # Récupérer un exemple de client du DataFrame original
    example_client = df.iloc[0].copy()  # Premier client comme exemple
    client_id = example_client["customerID"]

    # Récupérer les noms des colonnes après encodage
    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

    ohe = preprocessor.named_transformers_["cat"]
    cat_cols_encoded = ohe.get_feature_names_out(categorical_features)
    all_feature_names = np.concatenate([numeric_features, cat_cols_encoded])

    # Récupérer les valeurs encodées pour cet exemple
    example_encoded = X_train_prep[0]

    # Créer un dictionnaire pour stocker les valeurs encodées
    encoded_values = {}
    for i, feature in enumerate(all_feature_names):
        encoded_values[feature] = example_encoded[i]

    # Créer le tableau HTML
    html_table = f"""
    <div style="background-color: #222; color: #fff; padding: 20px; border-radius: 10px;">
        <h3 style="color: #fff;">Tableau d'encodage détaillé pour le client {client_id}</h3>
        <table border="1" style="border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; font-size: 14px; color: #fff;">
          <thead style="background-color: #4a4a4a; color: white; font-weight: bold; text-align: center;">
            <tr>
              <th style="padding: 8px;">Variable originale</th>
              <th style="padding: 8px;">Type original</th>
              <th style="padding: 8px;">Valeur originale</th>
              <th style="padding: 8px;">Traitement</th>
              <th style="padding: 8px;">Colonne encodée</th>
              <th style="padding: 8px;">Type encodé</th>
              <th style="padding: 8px;">Valeur encodée</th>
            </tr>
          </thead>
          <tbody>
    """

    # Définir les couleurs pour les types
    colors = {
        "Identifiant": "#FFF9C4;color:#333333",  # Jaune très pâle avec texte foncé
        "Catégorielle": "#FFCDD2;color:#333333",  # Rouge pâle avec texte foncé
        "Binaire": "#C8E6C9;color:#333333",  # Vert pâle avec texte foncé
        "Numérique": "#BBDEFB;color:#333333",  # Bleu pâle avec texte foncé
        "Numérique binaire": "#C8E6C9;color:#333333",  # Vert pâle avec texte foncé
    }

    # Ajouter l'ID client
    row_style = "background-color: #333; color: #fff;"
    html_table += f"""
        <tr style='{row_style}'>
          <td style='padding: 8px; text-align: left;'>customerID</td>
          <td style='padding: 8px; text-align: left; background-color: {colors["Identifiant"].split(';')[0]}; {colors["Identifiant"].split(';')[1]};'>Identifiant</td>
          <td style='padding: 8px; text-align: left;'>{client_id}</td>
          <td style='padding: 8px; text-align: left;'>Non utilisé</td>
          <td style='padding: 8px; text-align: left;'>-</td>
          <td style='padding: 8px; text-align: left;'>-</td>
          <td style='padding: 8px; text-align: center; font-weight: bold;'>-</td>
        </tr>
    """

    # Ajouter les variables numériques
    for i, col in enumerate(numeric_features):
        row_style = (
            "background-color: #444; color: #fff;"
            if i % 2 == 0
            else "background-color: #555; color: #fff;"
        )
        orig_value = example_client[col]
        enc_value = encoded_values[col]
        html_table += f"""
        <tr style='{row_style}'>
          <td style='padding: 8px; text-align: left;'>{col}</td>
          <td style='padding: 8px; text-align: left; background-color: {colors["Numérique"].split(';')[0]}; {colors["Numérique"].split(';')[1]};'>Numérique</td>
          <td style='padding: 8px; text-align: left;'>{orig_value}</td>
          <td style='padding: 8px; text-align: left;'>Standardisation</td>
          <td style='padding: 8px; text-align: left;'>{col}</td>
          <td style='padding: 8px; text-align: left; background-color: {colors["Numérique"].split(';')[0]}; {colors["Numérique"].split(';')[1]};'>Numérique standardisé</td>
          <td style='padding: 8px; text-align: center; font-weight: bold;'>{enc_value:.4f}</td>
        </tr>
        """

    # Ajouter les variables catégorielles
    for i, col in enumerate(categorical_features):
        orig_value = example_client[col]

        # Trouver toutes les colonnes encodées pour cette variable
        related_cols = [c for c in cat_cols_encoded if c.startswith(col + "_")]

        for j, enc_col in enumerate(related_cols):
            row_style = (
                "background-color: #444; color: #fff;"
                if (i + len(numeric_features)) % 2 == 0
                else "background-color: #555; color: #fff;"
            )

            # Déterminer si cette colonne encodée est active pour cet exemple
            is_active = encoded_values.get(enc_col, 0) == 1
            enc_value = encoded_values.get(enc_col, 0)

            # Extraire la valeur de la catégorie à partir du nom de la colonne
            category_value = enc_col.split("_", 1)[1] if "_" in enc_col else "Yes"

            html_table += f"""
            <tr style='{row_style}'>
              <td style='padding: 8px; text-align: left;'>{col}</td>
              <td style='padding: 8px; text-align: left; background-color: {colors["Catégorielle"].split(';')[0]}; {colors["Catégorielle"].split(';')[1]};'>Catégorielle</td>
              <td style='padding: 8px; text-align: left;'>{orig_value}</td>
              <td style='padding: 8px; text-align: left;'>One-Hot Encoding</td>
              <td style='padding: 8px; text-align: left;'>{enc_col}</td>
              <td style='padding: 8px; text-align: left; background-color: {colors["Binaire"].split(';')[0]}; {colors["Binaire"].split(';')[1]};'>Binaire</td>
              <td style='padding: 8px; text-align: center; font-weight: bold;'>{int(enc_value)}</td>
            </tr>
            """

    # Ajouter la variable cible
    row_style = "background-color: #333; color: #fff;"
    html_table += f"""
        <tr style='{row_style}'>
          <td style='padding: 8px; text-align: left;'>Churn</td>
          <td style='padding: 8px; text-align: left; background-color: {colors["Binaire"].split(';')[0]}; {colors["Binaire"].split(';')[1]};'>Binaire</td>
          <td style='padding: 8px; text-align: left;'>{example_client['Churn']}</td>
          <td style='padding: 8px; text-align: left;'>Encodage binaire</td>
          <td style='padding: 8px; text-align: left;'>Churn</td>
          <td style='padding: 8px; text-align: left; background-color: {colors["Numérique binaire"].split(';')[0]}; {colors["Numérique binaire"].split(';')[1]};'>Numérique binaire</td>
          <td style='padding: 8px; text-align: center; font-weight: bold;'>{1 if example_client['Churn'] == 'Yes' else 0}</td>
        </tr>
    """

    # Fermer la table et ajouter la légende
    html_table += (
        """
          </tbody>
        </table>
        <div style="margin-top: 20px;">
          <p style="font-weight: bold; margin-bottom: 10px; color: #fff;">Légende des types:</p>
          <div style="display: flex; flex-wrap: wrap; gap: 10px;">
            <div style="background-color: #FFF9C4; color: #333333; padding: 5px 10px; border-radius: 4px;">Identifiant</div>
            <div style="background-color: #FFCDD2; color: #333333; padding: 5px 10px; border-radius: 4px;">Catégorielle</div>
            <div style="background-color: #C8E6C9; color: #333333; padding: 5px 10px; border-radius: 4px;">Binaire</div>
            <div style="background-color: #BBDEFB; color: #333333; padding: 5px 10px; border-radius: 4px;">Numérique</div>
          </div>
        </div>

        <div style="margin-top: 20px;">
          <p style="font-weight: bold; margin-bottom: 10px; color: #fff;">Résumé:</p>
          <ul style="list-style-type: disc; padding-left: 20px; color: #fff;">
            <li>Variables originales: 21</li>
            <li>Variables après encodage: """
        + str(len(all_feature_names))
        + """</li>
            <li>Variables numériques: """
        + str(len(numeric_features))
        + """</li>
            <li>Variables binaires: """
        + str(len(cat_cols_encoded))
        + """</li>
          </ul>
        </div>
    </div>
    """
    )

    # Afficher le tableau HTML
    display(HTML(html_table))

    # Sauvegarder le tableau HTML dans un fichier
    with open(os.path.join(VISUALIZATIONS_DIR, "tableau_encodage.html"), "w") as f:
        f.write(html_table)

    print(
        f"Tableau d'encodage sauvegardé dans {os.path.join(VISUALIZATIONS_DIR, 'tableau_encodage.html')}"
    )


# ============================================================================
# 4. FONCTION PRINCIPALE
# ============================================================================
def main():
    """
    Fonction principale qui exécute toutes les étapes du projet.
    """
    # 1. Chargement et exploration des données
    print("=" * 80)
    print("1. CHARGEMENT ET EXPLORATION DES DONNÉES")
    print("=" * 80)
    df = load_and_explore_data("data/Customers.csv")

    # Sauvegarde du DataFrame nettoyé
    df.to_csv("data/Customers_cleaned.csv", index=False)
    print("DataFrame nettoyé sauvegardé dans Customers_cleaned.csv")

    # 2. Prétraitement des données
    print("\n" + "=" * 80)
    print("2. PRÉTRAITEMENT DES DONNÉES")
    print("=" * 80)
    (
        X_train_prep,
        X_val_prep,
        X_test_prep,
        y_train,
        y_val,
        y_test,
        preprocessor,
        X_train,
        X_val,
        X_test,
    ) = preprocess_data(df)

    # Visualisation de l'encodage
    visualize_encoding(df, preprocessor, X_train_prep, X_train)

    print(
        f"\nToutes les visualisations ont été sauvegardées dans le dossier '{VISUALIZATIONS_DIR}'"
    )


if __name__ == "__main__":
    main()
