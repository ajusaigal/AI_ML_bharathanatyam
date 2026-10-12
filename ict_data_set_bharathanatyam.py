# -*- coding: utf-8 -*-
"""ict data set bharathanatyam.ipynb


"""

from google.colab import files
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from collections import Counter
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import zipfile
# 1. Upload and extract ZIP (folder name is NOT mudra.zip)
print('Upload mudra.zip')
uploaded = files.upload()
zip_filename = next((name for name in uploaded if name.lower().endswith('.zip')), None)
if zip_filename is None:
    raise ValueError('Please upload a .zip file')
extract_path = Path('/content/mudra_dataset')
extract_path.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(zip_filename, 'r') as archive:
    # Reject unsafe ZIP paths
    for member in archive.infolist():
        target = (extract_path / member.filename).resolve()
        if not target.is_relative_to(extract_path.resolve()):
            raise ValueError('Unsafe ZIP member path: ' + member.filename)
    archive.extractall(extract_path)
print('Extracted to:', extract_path)

# 2. Discover valid images and class folders (supports one wrapper folder)
exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
records = []
invalid = []
for path in sorted(extract_path.rglob('*')):
    if not path.is_file() or path.suffix.lower() not in exts or '__MACOSX' in path.parts:
        continue
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            w, h = im.size
            mode = im.mode
    except (UnidentifiedImageError, OSError, ValueError):
        invalid.append(str(path))
        continue
    # Parent directory is the mudra class; root-level images need manual labels
    records.append({'filepath':str(path), 'class_name':path.parent.name,
                    'width':w, 'height':h, 'mode':mode,
                    'format':path.suffix.lower().lstrip('.'), 'aspect_ratio':w/h})
df = pd.DataFrame(records)
if df.empty:
    raise ValueError('No readable images found. Check ZIP folder contents.')
print('Valid images:',len(df),'| Unreadable images:',len(invalid))
print('Class folders:',sorted(df.class_name.unique()))
if df.class_name.nunique() < 2:
    raise ValueError('Only one image folder detected. Images must be arranged in separate class folders.')
if df.class_name.nunique() != 5:
    print('WARNING: Expected 5 classes. Inspect the detected folder names and ZIP structure.')
display(df.head(10))

# 3. Class distribution
counts = df.class_name.value_counts().sort_index()
summary = pd.DataFrame({'Images':counts, 'Percentage':(counts/len(df)*100).round(2)})
display(summary)
fig, ax = plt.subplots(figsize=(9,4))
counts.plot.bar(ax=ax)
ax.set(title='Images per Mudra Class', xlabel='Mudra', ylabel='Number of Images')
plt.xticks(rotation=35, ha='right'); plt.tight_layout(); plt.show()
print('Balance ratio (smallest/largest):', round(counts.min()/counts.max(),3))

# 4. Image characteristics
print('\nImage characteristics:')
display(df[['width','height','aspect_ratio']].describe().round(2))
print('Image modes:',df['mode'].value_counts().to_dict())
print('Image formats:',df['format'].value_counts().to_dict())
fig, axes = plt.subplots(1,2,figsize=(11,4))
axes[0].hist(df.width,bins=min(20,df.width.nunique()),edgecolor='black')
axes[0].set(title='Image Width Distribution',xlabel='Width (pixels)',ylabel='Count')
axes[1].hist(df.height,bins=min(20,df.height.nunique()),edgecolor='black')
axes[1].set(title='Image Height Distribution',xlabel='Height (pixels)',ylabel='Count')
plt.tight_layout();plt.show()

# 5. One sample image per class
classes = sorted(df.class_name.unique())
fig, axes = plt.subplots(1,len(classes),figsize=(3*len(classes),3.5))
axes = np.atleast_1d(axes)
for ax,cls in zip(axes,classes):
    sample = df.loc[df.class_name.eq(cls),'filepath'].iloc[0]
    with Image.open(sample) as im:
        ax.imshow(im.convert('RGB'))
    ax.set_title(cls);ax.axis('off')
plt.tight_layout();plt.show()

# 6. Stratified 70/15/15 split; ensures representative classes
# A two-stage stratified split requires enough images in each class.
if counts.min() < 3:
    raise ValueError('Each class needs at least 3 images for train/validation/test splits.')
try:
    train_df, temp_df = train_test_split(df,test_size=0.30,random_state=42,stratify=df.class_name)
    val_df, test_df = train_test_split(temp_df,test_size=0.50,random_state=42,stratify=temp_df.class_name)
except ValueError as exc:
    raise ValueError('Not enough images per class for stratified 70/15/15 split. Add images to small classes.') from exc
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)
print('Train:',len(train_df),'Validation:',len(val_df),'Test:',len(test_df))
split_counts = pd.DataFrame({'Train':train_df.class_name.value_counts(),
                             'Validation':val_df.class_name.value_counts(),
                             'Test':test_df.class_name.value_counts()}).fillna(0).astype(int)
display(split_counts)
assert set(train_df.filepath).isdisjoint(val_df.filepath)
assert set(train_df.filepath).isdisjoint(test_df.filepath)
assert set(val_df.filepath).isdisjoint(test_df.filepath)

# 7. Save split manifests for later tasks
for name, part in [('train',train_df),('validation',val_df),('test',test_df)]:
    part.to_csv(f'/content/{name}_mudra.csv',index=False)
print('Saved train_mudra.csv, validation_mudra.csv, test_mudra.csv')
print('TASK 1 COMPLETED. Use the DataFrames train_df, val_df and test_df for Task 2.')

# ============================================================
# TASK 2: IMAGE PREPROCESSING AND DATA AUGMENTATION
# Bharatanatyam Mudra Recognition
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

# ------------------------------------------------------------
# STEP 1: CONFIGURATION
# ------------------------------------------------------------

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
SEED = 42

tf.random.set_seed(SEED)

# Use existing dataframes from Task 1
print("TASK 2: PREPROCESSING AND AUGMENTATION")
print("-" * 50)

print("Training images:", len(train_df))
print("Validation images:", len(val_df))
print("Testing images:", len(test_df))

# ------------------------------------------------------------
# STEP 2: IDENTIFY DATAFRAME COLUMNS
# ------------------------------------------------------------

print("\nTraining dataframe columns:")
print(train_df.columns.tolist())

# Automatically identify image path and label columns
path_candidates = [
    "filepath", "file_path", "path",
    "image_path", "filename", "file"
]

label_candidates = [
    "label", "class", "class_name",
    "category", "mudra"
]

path_col = next(
    (c for c in path_candidates if c in train_df.columns),
    None
)

label_col = next(
    (c for c in label_candidates if c in train_df.columns),
    None
)

if path_col is None or label_col is None:
    raise ValueError(
        "Could not identify image path or label columns. "
        "Check train_df.columns."
    )

print("Image path column:", path_col)
print("Label column:", label_col)

# ------------------------------------------------------------
# STEP 3: CLASS NAMES AND LABEL ENCODING
# ------------------------------------------------------------

class_names = sorted(
    train_df[label_col].astype(str).unique().tolist()
)

num_classes = len(class_names)

print("\nMudra classes:", class_names)
print("Number of classes:", num_classes)

class_to_index = {
    name: i for i, name in enumerate(class_names)
}

print("Class encoding:", class_to_index)

# ------------------------------------------------------------
# STEP 4: IMAGE PREPROCESSING FUNCTION
# ------------------------------------------------------------

def preprocess_image(filepath, label):

    # Read image
    image = tf.io.read_file(filepath)

    # Decode JPG/PNG image
    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    # Set image shape
    image.set_shape([None, None, 3])

    # Resize image
    image = tf.image.resize(
        image,
        [IMG_HEIGHT, IMG_WIDTH]
    )

    # Normalize pixel values to [0, 1]
    image = tf.cast(image, tf.float32) / 255.0

    return image, label


# ------------------------------------------------------------
# STEP 5: CREATE TENSORFLOW DATASETS
# ------------------------------------------------------------

def create_dataset(dataframe, training=False):

    paths = dataframe[path_col].astype(str).values

    labels = dataframe[label_col].astype(str).map(
        class_to_index
    ).values.astype(np.int32)

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    dataset = dataset.map(
        preprocess_image,
        num_parallel_calls=AUTOTUNE
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(dataframe),
            seed=SEED
        )

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTOTUNE)

    return dataset


train_ds = create_dataset(
    train_df,
    training=True
)

val_ds = create_dataset(
    val_df,
    training=False
)

test_ds = create_dataset(
    test_df,
    training=False
)

print("\nTensorFlow datasets created successfully!")

# ------------------------------------------------------------
# STEP 6: VERIFY PREPROCESSING
# ------------------------------------------------------------

for images, labels in train_ds.take(1):

    print("\nTraining batch shape:", images.shape)
    print("Label batch shape:", labels.shape)

    print("Minimum pixel value:",
          tf.reduce_min(images).numpy())

    print("Maximum pixel value:",
          tf.reduce_max(images).numpy())

# ------------------------------------------------------------
# STEP 7: VISUALIZE PREPROCESSED IMAGES
# ------------------------------------------------------------

plt.figure(figsize=(12, 8))

for images, labels in train_ds.take(1):

    for i in range(min(9, len(images))):

        plt.subplot(3, 3, i + 1)

        plt.imshow(images[i].numpy())

        plt.title(
            class_names[int(labels[i].numpy())]
        )

        plt.axis("off")

plt.suptitle(
    "Preprocessed Bharatanatyam Mudra Images",
    fontsize=15
)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# STEP 8: DATA AUGMENTATION PIPELINE
# ------------------------------------------------------------

augmentation = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    ),

    tf.keras.layers.RandomRotation(
        factor=0.08
    ),

    tf.keras.layers.RandomZoom(
        height_factor=0.10,
        width_factor=0.10
    ),

    tf.keras.layers.RandomTranslation(
        height_factor=0.08,
        width_factor=0.08
    ),

    tf.keras.layers.RandomContrast(
        factor=0.10
    )

], name="mudra_augmentation")

print("\nAugmentation pipeline created successfully!")

# ------------------------------------------------------------
# STEP 9: VISUALIZE AUGMENTED IMAGES
# ------------------------------------------------------------

# Get one image from training dataset
for images, labels in train_ds.take(1):

    sample_image = images[0]
    sample_label = labels[0]

# sample_image shape: (224, 224, 3)
print("\nSample image shape:", sample_image.shape)

# Add one batch dimension
sample_batch = tf.expand_dims(
    sample_image,
    axis=0
)

print("Sample batch shape:", sample_batch.shape)

plt.figure(figsize=(16, 8))

# Original image
plt.subplot(2, 4, 1)

plt.imshow(sample_image.numpy())

plt.title(
    "Original: " +
    class_names[int(sample_label.numpy())]
)

plt.axis("off")

# Generate 7 augmented images
for i in range(1, 8):

    augmented_batch = augmentation(
        sample_batch,
        training=True
    )

    augmented_image = augmented_batch[0]

    plt.subplot(2, 4, i + 1)

    plt.imshow(
        tf.clip_by_value(
            augmented_image,
            0.0,
            1.0
        ).numpy()
    )

    plt.title(f"Augmented {i}")

    plt.axis("off")

plt.suptitle(
    "Bharatanatyam Mudra Data Augmentation",
    fontsize=16
)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# STEP 10: APPLY AUGMENTATION TO TRAINING DATA ONLY
# ------------------------------------------------------------

# This pipeline applies random transformations
# each time training images are accessed.

train_augmented_ds = train_ds.map(
    lambda images, labels: (
        augmentation(images, training=True),
        labels
    ),
    num_parallel_calls=AUTOTUNE
)

train_augmented_ds = train_augmented_ds.prefetch(
    AUTOTUNE
)

# Validation and test data remain unchanged
# No augmentation is applied to them.

print("\nAugmentation applied to training dataset only.")

# ------------------------------------------------------------
# STEP 11: VERIFY AUGMENTED TRAINING BATCH
# ------------------------------------------------------------

for images, labels in train_augmented_ds.take(1):

    print("\nAugmented training batch:", images.shape)
    print("Labels batch:", labels.shape)

    print(
        "Pixel range:",
        float(tf.reduce_min(images).numpy()),
        "to",
        float(tf.reduce_max(images).numpy())
    )

# ------------------------------------------------------------
# STEP 12: SAVE SAMPLE VISUALIZATION
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

for images, labels in train_augmented_ds.take(1):

    for i in range(min(6, len(images))):

        plt.subplot(2, 3, i + 1)

        plt.imshow(
            tf.clip_by_value(
                images[i],
                0.0,
                1.0
            ).numpy()
        )

        plt.title(
            class_names[int(labels[i].numpy())]
        )

        plt.axis("off")

plt.suptitle(
    "Augmented Training Samples",
    fontsize=15
)

plt.tight_layout()

plt.savefig(
    "/content/task2_augmented_samples.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

# ------------------------------------------------------------
# STEP 13: FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 55)
print("TASK 2 COMPLETED SUCCESSFULLY")
print("=" * 55)

print("Image size:", IMG_HEIGHT, "x", IMG_WIDTH)
print("Image channels: RGB")
print("Normalization: 0 to 1")

print("\nAugmentation techniques:")
print("1. Random rotation")
print("2. Random zoom")
print("3. Random translation")
print("4. Random contrast")

print("\nDataset summary:")
print("Training:", len(train_df))
print("Validation:", len(val_df))
print("Testing:", len(test_df))

print("\nDatasets ready for Task 3:")
print("train_augmented_ds")
print("train_ds")
print("val_ds")
print("test_ds")

print("\nVisualization saved at:")
print("/content/task2_augmented_samples.png")

# ============================================================
# TASK 3: CUSTOM CNN DEVELOPMENT AND TRAINING
# Bharatanatyam Mudra Recognition
# ============================================================

import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

# ------------------------------------------------------------
# STEP 1: CONFIGURATION
# ------------------------------------------------------------

SEED = 42
tf.keras.utils.set_random_seed(SEED)

IMG_HEIGHT = 224
IMG_WIDTH = 224
NUM_CLASSES = 5
EPOCHS = 25

class_names = [
    "Musti",
    "Pataka",
    "Sikhara",
    "Simhamukha",
    "Trisula"
]

print("=" * 60)
print("TASK 3: CNN DEVELOPMENT")
print("=" * 60)

print("TensorFlow version:", tf.__version__)
print("Classes:", class_names)

print("\nAvailable GPU devices:")
print(tf.config.list_physical_devices("GPU"))

# ------------------------------------------------------------
# STEP 2: CHECK DATASETS FROM TASK 2
# ------------------------------------------------------------

required_variables = [
    "train_ds",
    "val_ds",
    "test_ds"
]

for variable in required_variables:
    if variable not in globals():
        raise ValueError(
            f"{variable} not found. "
            "Run Task 1 and Task 2 first."
        )

print("\nDatasets loaded successfully.")

for images, labels in train_ds.take(1):
    print("Training image shape:", images.shape)
    print("Training label shape:", labels.shape)

# ------------------------------------------------------------
# STEP 3: CREATE CUSTOM CNN MODEL
# ------------------------------------------------------------

def build_cnn(
    learning_rate=0.001,
    dropout_rate=0.3,
    filters=32
):

    model = models.Sequential([

        layers.Input(
            shape=(IMG_HEIGHT, IMG_WIDTH, 3)
        ),

        # Convolution Block 1
        layers.Conv2D(
            filters,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D((2, 2)),

        # Convolution Block 2
        layers.Conv2D(
            filters * 2,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D((2, 2)),

        # Convolution Block 3
        layers.Conv2D(
            filters * 4,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D((2, 2)),

        # Convolution Block 4
        layers.Conv2D(
            filters * 8,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D((2, 2)),

        # Classification layers
        layers.GlobalAveragePooling2D(),

        layers.Dense(
            128,
            activation="relu",
            kernel_regularizer=regularizers.l2(0.001)
        ),

        layers.Dropout(dropout_rate),

        # Output: 5 mudra classes
        layers.Dense(
            NUM_CLASSES,
            activation="softmax"
        )

    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"]
    )

    return model


# ------------------------------------------------------------
# STEP 4: HYPERPARAMETER TUNING
# ------------------------------------------------------------

# Two hyperparameters:
# 1. Learning rate
# 2. Dropout rate

hyperparameter_configs = [

    {
        "learning_rate": 0.001,
        "dropout_rate": 0.3
    },

    {
        "learning_rate": 0.0001,
        "dropout_rate": 0.5
    }

]

tuning_results = []

best_val_accuracy = -1
best_config = None

print("\nStarting hyperparameter tuning...")

for index, config in enumerate(
    hyperparameter_configs
):

    print("\n" + "=" * 60)

    print(
        f"Experiment {index + 1}"
    )

    print("Configuration:", config)

    tf.keras.backend.clear_session()

    tuning_model = build_cnn(
        learning_rate=config["learning_rate"],
        dropout_rate=config["dropout_rate"]
    )

    callbacks = [

        EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        )

    ]

    start_time = time.time()

    history = tuning_model.fit(

        train_augmented_ds
        if "train_augmented_ds" in globals()
        else train_ds,

        validation_data=val_ds,

        epochs=8,

        callbacks=callbacks,

        verbose=1
    )

    training_time = time.time() - start_time

    # Evaluate validation accuracy
    val_loss, val_accuracy = tuning_model.evaluate(
        val_ds,
        verbose=0
    )

    result = {

        "learning_rate":
            config["learning_rate"],

        "dropout_rate":
            config["dropout_rate"],

        "validation_accuracy":
            float(val_accuracy),

        "training_time_seconds":
            round(training_time, 2)

    }

    tuning_results.append(result)

    print("\nValidation Accuracy:",
          round(val_accuracy * 100, 2), "%")

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        best_config = config.copy()


# ------------------------------------------------------------
# STEP 5: DISPLAY TUNING RESULTS
# ------------------------------------------------------------

import pandas as pd

tuning_df = pd.DataFrame(tuning_results)

print("\nHYPERPARAMETER TUNING RESULTS")

print(tuning_df.to_string(index=False))

print("\nBest configuration:", best_config)

print(
    "Best validation accuracy:",
    round(best_val_accuracy * 100, 2),
    "%"
)

# ------------------------------------------------------------
# STEP 6: BUILD FINAL CNN MODEL
# ------------------------------------------------------------

tf.keras.backend.clear_session()

model = build_cnn(

    learning_rate=best_config["learning_rate"],

    dropout_rate=best_config["dropout_rate"]

)

print("\nFINAL CNN MODEL ARCHITECTURE")

model.summary()

# ------------------------------------------------------------
# STEP 7: TRAIN FINAL CNN
# ------------------------------------------------------------

callbacks = [

    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6
    ),

    ModelCheckpoint(
        "/content/best_mudra_cnn.keras",
        monitor="val_loss",
        save_best_only=True
    )

]

print("\nStarting final CNN training...")

start_time = time.time()

history = model.fit(

    train_augmented_ds
    if "train_augmented_ds" in globals()
    else train_ds,

    validation_data=val_ds,

    epochs=EPOCHS,

    callbacks=callbacks,

    verbose=1

)

total_training_time = time.time() - start_time

print("\nTraining completed.")

print(
    "Total training time:",
    round(total_training_time, 2),
    "seconds"
)

# ------------------------------------------------------------
# STEP 8: PLOT TRAINING AND VALIDATION ACCURACY
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy",
    marker="o"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy",
    marker="o"
)

plt.title(
    "CNN Training and Validation Accuracy"
)

plt.xlabel("Epochs")
plt.ylabel("Accuracy")

plt.legend()
plt.grid(True)

plt.savefig(
    "/content/task3_accuracy_curve.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

# ------------------------------------------------------------
# STEP 9: PLOT TRAINING AND VALIDATION LOSS
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss",
    marker="o"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss",
    marker="o"
)

plt.title(
    "CNN Training and Validation Loss"
)

plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.legend()
plt.grid(True)

plt.savefig(
    "/content/task3_loss_curve.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

# ------------------------------------------------------------
# STEP 10: FINAL VALIDATION EVALUATION
# ------------------------------------------------------------

# Load best saved model
model = tf.keras.models.load_model(
    "/content/best_mudra_cnn.keras"
)

val_loss, val_accuracy = model.evaluate(
    val_ds,
    verbose=1
)

print("\nFINAL VALIDATION RESULTS")

print(
    "Validation Accuracy:",
    round(val_accuracy * 100, 2),
    "%"
)

print(
    "Validation Loss:",
    round(val_loss, 4)
)

# Test dataset is reserved for Task 4.

# ------------------------------------------------------------
# STEP 11: SAVE TRAINED MODEL
# ------------------------------------------------------------

model.save(
    "/content/mudra_cnn_model.keras"
)

# Save class names
with open(
    "/content/mudra_class_names.json",
    "w"
) as f:

    json.dump(
        class_names,
        f,
        indent=4
    )

# Save training history
history_df = pd.DataFrame(history.history)

history_df.to_csv(
    "/content/task3_training_history.csv",
    index=False
)

# Save hyperparameter tuning results
tuning_df.to_csv(
    "/content/task3_hyperparameter_results.csv",
    index=False
)

# ------------------------------------------------------------
# STEP 12: ANALYZE OVERFITTING / UNDERFITTING
# ------------------------------------------------------------

train_accuracy = history.history["accuracy"][-1]

validation_accuracy = history.history[
    "val_accuracy"
][-1]

accuracy_gap = train_accuracy - validation_accuracy

print("\nMODEL LEARNING ANALYSIS")

print(
    "Final training accuracy:",
    round(train_accuracy * 100, 2),
    "%"
)

print(
    "Final validation accuracy:",
    round(validation_accuracy * 100, 2),
    "%"
)

print(
    "Accuracy gap:",
    round(accuracy_gap * 100, 2),
    "percentage points"
)

if accuracy_gap > 0.10:

    print(
        "Possible overfitting: "
        "Training accuracy is substantially "
        "higher than validation accuracy."
    )

elif (
    train_accuracy < 0.60
    and validation_accuracy < 0.60
):

    print(
        "Possible underfitting: "
        "Both accuracies are relatively low."
    )

else:

    print(
        "No obvious overfitting based on "
        "the final accuracy gap. "
        "Inspect the learning curves."
    )

# ------------------------------------------------------------
# STEP 13: FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 60)

print("TASK 3 COMPLETED SUCCESSFULLY")

print("=" * 60)

print("Model: Custom CNN")

print("Number of classes:", NUM_CLASSES)

print("Input size: 224 x 224 x 3")

print("Best hyperparameters:", best_config)

print(
    "Validation accuracy:",
    round(val_accuracy * 100, 2),
    "%"
)

print("\nSaved files:")

print("/content/mudra_cnn_model.keras")

print("/content/best_mudra_cnn.keras")

print("/content/mudra_class_names.json")

print("/content/task3_accuracy_curve.png")

print("/content/task3_loss_curve.png")

print("/content/task3_training_history.csv")

print("/content/task3_hyperparameter_results.csv")
