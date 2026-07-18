"""
spark_pipeline.py
------------------
STAGE 1 - Preprocessing + validation qualité + échantillonnage
STAGE 2 - Data augmentation (train uniquement)
STAGE 3 - Export sur disque en arborescence de fichiers (pour Keras/Streamlit)

Utilisation :
    python spark_pipeline.py
"""

import io
import random
import uuid

from PIL import Image, ImageEnhance
from pyspark.sql import functions as F, Window
from pyspark.sql.types import (
    StructType, StructField, StringType, BinaryType, IntegerType, BooleanType,
    ArrayType,
)

from config import (
    get_spark_session,
    RAW_TRAIN_DIR, RAW_TEST_DIR,
    STAGE1_CLEAN_TRAIN, STAGE1_CLEAN_TEST,
    STAGE2_AUGMENTED_TRAIN,
    STAGE3_READY_TRAIN, STAGE3_READY_TEST,
    REPORTS_DIR,
    IMG_SIZE, MIN_VALID_SIZE, CLASS_MAP,
    SAMPLE_PER_CLASS_TRAIN, SAMPLE_PER_CLASS_TEST, SAMPLE_SEED,
)


# STAGE 1 - PREPROCESSING + VALIDATION + ECHANTILLONNAGE

PROCESS_SCHEMA = StructType([
    StructField("valid", BooleanType(), True),
    StructField("reason", StringType(), True),
    StructField("orig_width", IntegerType(), True),
    StructField("orig_height", IntegerType(), True),
    StructField("resized_bytes", BinaryType(), True),
])


def process_image(content: bytes):
    """UDF : décode, valide, normalise RGB et redimensionne une image."""
    if content is None or len(content) == 0:
        return (False, "fichier_vide_ou_manquant", None, None, None)
    try:
        img = Image.open(io.BytesIO(content))
        img.load()
    except Exception as e:
        return (False, f"decodage_impossible:{type(e).__name__}", None, None, None)

    width, height = img.size
    if width < MIN_VALID_SIZE[0] or height < MIN_VALID_SIZE[1]:
        return (False, "image_trop_petite", width, height, None)

    try:
        img = img.convert("RGB")
        img_resized = img.resize(IMG_SIZE, Image.BILINEAR)
        buf = io.BytesIO()
        img_resized.save(buf, format="JPEG", quality=90)
        resized_bytes = buf.getvalue()
    except Exception as e:
        return (False, f"resize_impossible:{type(e).__name__}", width, height, None)

    return (True, "ok", width, height, resized_bytes)


def label_from_path(path: str) -> str:
    parts = path.replace("\\", "/").split("/")
    for p in parts:
        if p in CLASS_MAP:
            return CLASS_MAP[p]
    return "unknown"


def stage_1_preprocessing(spark, raw_dir: str, output_path: str, split_name: str, sample_per_class: int):
    """
    Lit les images brutes, valide leur qualité, les redimensionne, puis
    échantillonne exactement `sample_per_class` images par classe (sans
    .collect() : on utilise une Window + row_number pour un tirage exact
    et déterministe, distribué).
    """
    print(f"\n[STAGE 1 - {split_name}] Lecture depuis : {raw_dir}")

    df = (
        spark.read.format("binaryFile")
        .option("pathGlobFilter", "*.{jpg,jpeg,png,JPG,JPEG,PNG}")
        .option("recursiveFileLookup", "true")
        .load(raw_dir)
    )

    label_udf = F.udf(label_from_path, StringType())
    process_udf = F.udf(process_image, PROCESS_SCHEMA)

    df = df.withColumn("label", label_udf(F.col("path")))
    df = df.withColumn("proc", process_udf(F.col("content")))
    df = df.select(
        "path", "label",
        F.col("proc.valid").alias("valid"),
        F.col("proc.reason").alias("reason"),
        F.col("proc.orig_width").alias("orig_width"),
        F.col("proc.orig_height").alias("orig_height"),
        F.col("proc.resized_bytes").alias("image"),
    )

    valid_df = df.filter(F.col("valid") & (F.col("label") != "unknown"))

    quality_report = (
        df.withColumn(
            "status",
            F.when(F.col("valid") & (F.col("label") != "unknown"), F.lit("valid"))
             .when(F.col("label") == "unknown", F.lit("unknown_label"))
             .otherwise(F.col("reason")),
        )
        .groupBy("status")
        .count()
        .withColumn("split", F.lit(split_name))
    )
    (
        quality_report.coalesce(1)
        .write.mode("overwrite")
        .json(f"{REPORTS_DIR}/quality_{split_name}")
    )
    print(f"[STAGE 1 - {split_name}] Rapport qualité écrit dans {REPORTS_DIR}/quality_{split_name}")
    quality_report.show(20, truncate=False)  # affichage debug, action bornée, pas un collect global

    # Window : on partitionne par label, on trie aléatoirement (seed fixe =
    # reproductible), puis on numérote les lignes (row_number) et on ne
    # garde que les N premières par classe.
    window_spec = Window.partitionBy("label").orderBy(F.rand(seed=SAMPLE_SEED))
    sampled_df = (
        valid_df.withColumn("rn", F.row_number().over(window_spec))
        .filter(F.col("rn") <= sample_per_class)
        .drop("rn")
    )

    sampled_df.groupBy("label").count().show()  # vérif visuelle : doit afficher sample_per_class pour chaque classe

    (
        sampled_df.select("label", "image")
        .write.mode("overwrite")
        .partitionBy("label")
        .parquet(output_path)
    )
    print(f"[STAGE 1 - {split_name}] Parquet échantillonné écrit : {output_path}")

# STAGE 2 - DATA AUGMENTATION (train uniquement)

AUGMENTATION_SCHEMA = ArrayType(StructType([
    StructField("aug_type", StringType(), True),
    StructField("image", BinaryType(), True),
]))


def augment_image(content: bytes):
    """UDF : génère plusieurs variantes augmentées d'une image déjà nettoyée."""
    variants = []
    try:
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        return variants

    def to_bytes(im):
        buf = io.BytesIO()
        im.resize(IMG_SIZE, Image.BILINEAR).save(buf, format="JPEG", quality=90)
        return buf.getvalue()

    variants.append(("original", to_bytes(img)))
    variants.append(("flip_horizontal", to_bytes(img.transpose(Image.FLIP_LEFT_RIGHT))))

    angle = random.uniform(-20, 20)
    variants.append(("rotation", to_bytes(img.rotate(angle, expand=False, fillcolor=(255, 255, 255)))))

    factor = random.uniform(0.6, 1.4)
    variants.append(("brightness", to_bytes(ImageEnhance.Brightness(img).enhance(factor))))

    w, h = img.size
    cw, ch = int(w * 0.85), int(h * 0.85)
    left, top = (w - cw) // 2, (h - ch) // 2
    variants.append(("zoom_crop", to_bytes(img.crop((left, top, left + cw, top + ch)))))

    return variants


def stage_2_augmentation(spark):
    print(f"\n[STAGE 2] Lecture : {STAGE1_CLEAN_TRAIN}")
    df = spark.read.parquet(STAGE1_CLEAN_TRAIN)

    augment_udf = F.udf(augment_image, AUGMENTATION_SCHEMA)
    df_aug = df.withColumn("variants", augment_udf(F.col("image")))
    df_aug = df_aug.withColumn("variant", F.explode(F.col("variants")))
    df_final = df_aug.select(
        "label",
        F.col("variant.aug_type").alias("aug_type"),
        F.col("variant.image").alias("image"),
    )

    df_final.groupBy("label", "aug_type").count().orderBy("label", "aug_type").show(50, truncate=False)

    (
        df_final.select("label", "image")
        .write.mode("overwrite")
        .partitionBy("label")
        .parquet(STAGE2_AUGMENTED_TRAIN)
    )
    print(f"[STAGE 2] Parquet augmenté écrit : {STAGE2_AUGMENTED_TRAIN}")

# STAGE 3 - EXPORT SUR DISQUE (arborescence exploitable par Keras/Streamlit)

def write_partition_to_disk(rows, base_dir: str):
    """Exécuté sur chaque worker : écrit les images de SA partition sur disque."""
    import os
    for row in rows:
        label_dir = os.path.join(base_dir, row["label"])
        os.makedirs(label_dir, exist_ok=True)
        filepath = os.path.join(label_dir, f"{uuid.uuid4().hex}.jpg")
        with open(filepath, "wb") as f:
            f.write(row["image"])


def stage_3_export(spark, parquet_path: str, output_dir: str, split_name: str):
    print(f"\n[STAGE 3 - {split_name}] Export : {parquet_path} -> {output_dir}")
    df = spark.read.parquet(parquet_path).select("label", "image")
    df.foreachPartition(lambda rows: write_partition_to_disk(rows, output_dir))
    df.groupBy("label").count().show()
    print(f"[STAGE 3 - {split_name}] Terminé.")

# Notre Main function 

def main():
    spark = get_spark_session()

    stage_1_preprocessing(spark, RAW_TRAIN_DIR, STAGE1_CLEAN_TRAIN, "train", SAMPLE_PER_CLASS_TRAIN)
    stage_1_preprocessing(spark, RAW_TEST_DIR, STAGE1_CLEAN_TEST, "test", SAMPLE_PER_CLASS_TEST)

    stage_2_augmentation(spark)

    stage_3_export(spark, STAGE2_AUGMENTED_TRAIN, STAGE3_READY_TRAIN, "train")
    stage_3_export(spark, STAGE1_CLEAN_TEST, STAGE3_READY_TEST, "test")

    print("\n[PIPELINE] Terminé avec succès.")
    spark.stop()


if __name__ == "__main__":
    main()
