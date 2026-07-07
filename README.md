# Waste Classification — Pipeline PySpark + CNN + Streamlit

Classification d'images de déchets (**organique** vs **recyclable**) à partir du
dataset Kaggle [waste-classification-data](https://www.kaggle.com/datasets/techsash/waste-classification-data).

## 1. Pourquoi ce projet, et pourquoi PySpark

L'objectif pédagogique de ce projet est d'apprendre à **manier PySpark** sur un
cas concret : le traitement massif et distribué d'images, avant de les donner
à un modèle de deep learning.

**PySpark** est l'API Python d'Apache Spark, un moteur de traitement de
données **distribué** : les données sont découpées en *partitions*, réparties
sur plusieurs cœurs/machines (workers), et traitées en parallèle. Cela permet
de faire passer un pipeline de "quelques milliers d'images sur mon PC" à
"des millions d'images sur un cluster" **sans changer le code**, seulement la
configuration (`local[*]` -> cluster YARN/Kubernetes).

**Pourquoi Spark n'entraîne pas le CNN lui-même :** Spark/MLlib n'a pas de
couches de réseaux de neurones convolutifs. Le standard professionnel est
donc : **Spark pour tout le traitement/ETL de données à grande échelle**
(lecture, validation, nettoyage, redimensionnement, augmentation), puis un
**framework de deep learning** (ici TensorFlow/Keras, qui reste du Python)
pour la partie modèle.

## 2. Règle de conception : pas de `.collect()`

Dans tout ce projet, **`.collect()` (et équivalents comme `.toPandas()` sur un
DataFrame complet) sont interdits**. Pourquoi cette règle est centrale à
l'esprit même de Spark :

- `.collect()` rapatrie **toutes** les lignes d'un DataFrame distribué vers la
  mémoire du **driver** (un seul processus, une seule machine). Si le dataset
  est trop gros, ça fait planter le driver (`OutOfMemoryError`) — c'est
  l'erreur n°1 des débutants sur Spark.
- Ça casse aussi le principe même du calcul distribué : autant ne pas utiliser
  Spark si on rapatrie tout en mémoire à la fin.
- La bonne pratique est **d'écrire les résultats sur disque** (`.write.parquet(...)`,
  `.write.json(...)`) et de laisser Spark gérer l'écriture distribuée, partition
  par partition, sans jamais tout regrouper sur le driver.

Ce projet applique cette règle strictement : chaque étape du pipeline **lit**
du Parquet (ou des fichiers images) et **écrit** du Parquet (ou des fichiers),
même pour les rapports de qualité de données.

## 3. Architecture du projet

```
waste-classification-pyspark/
├── data/DATASET/                 dataset brut (téléchargé depuis Kaggle)
├── output/
│   ├── stage1_clean/              Parquet : images validées + redimensionnées
│   ├── stage2_augmented/          Parquet : images augmentées
│   ├── stage3_ready/              images finales en fichiers (pour Keras/Streamlit)
│   ├── reports/                   rapports qualité (écrits par Spark, pas collectés)
│   └── model/                     modèle CNN entraîné
├── src/
│   ├── config.py                  configuration partagée (chemins, SparkSession)
│   ├── spark_pipeline.py          SERVICE PYSPARK UNIQUE, mono-bloc, en 3 stages internes
│   ├── train_model.py             entraînement du CNN (Python/TensorFlow)
│   └── predict.py                 fonction de prédiction réutilisable
└── app/
    └── streamlit_app.py           interface utilisateur (upload image -> prédiction)
```

### Le choix "mono-bloc mais subdivisé"

`spark_pipeline.py` est **un seul script, une seule `SparkSession`**, mais
organisé en fonctions bien séparées, comme des *stages* d'un job Spark :

- `stage_1_preprocessing(spark)` : lecture binaire des images, extraction du
  label depuis le chemin, validation qualité (fichier vide/corrompu/trop
  petit -> rejeté), normalisation RGB, redimensionnement homogène. Écrit
  `output/stage1_clean/` (Parquet) + `output/reports/` (rapport qualité, en
  Parquet/JSON écrit directement par Spark).
- `stage_2_augmentation(spark)` : lit `stage1_clean/`, génère plusieurs
  variantes par image (flip, rotation, luminosité, zoom). Écrit
  `output/stage2_augmented/`.
- `stage_3_export(spark)` : lit `stage2_augmented/` (train) et `stage1_clean/`
  (test), matérialise les images en fichiers `.jpg` dans une arborescence
  `output/stage3_ready/{train,test}/{organic,recyclable}/`, exploitable par
  Keras (`image_dataset_from_directory`) et par Streamlit.

Chaque stage est **indépendant et relançable séparément** (il lit son entrée
depuis le disque, pas depuis la mémoire d'un stage précédent), mais tout vit
dans un seul fichier / une seule exécution Spark pour rester "un seul service".

### En aval de Spark (Python pur)

- `train_model.py` : charge `output/stage3_ready/`, entraîne un CNN (Keras),
  sauvegarde le modèle dans `output/model/`.
- `predict.py` : charge le modèle sauvegardé, prétraite une image donnée,
  retourne la classe prédite + le score de confiance. Conçu comme une
  fonction importable (pas juste un script CLI) pour être réutilisée par
  Streamlit.
- `app/streamlit_app.py` : interface web où l'utilisateur dépose une image et
  obtient la prédiction, en appelant `predict.py`.

## 4. Installation

```bash
conda activate waste-spark
pip install -r requirements.txt
```

## 5. Utilisation (à venir, étape par étape)

```bash
# 1. Pipeline PySpark complet (stage 1 + 2 + 3)
python src/spark_pipeline.py

# 2. Entraînement du CNN
python src/train_model.py

# 3. Interface de prédiction
streamlit run app/streamlit_app.py
```
