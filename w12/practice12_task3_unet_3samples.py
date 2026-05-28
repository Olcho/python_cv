from pathlib import Path
import os
import random

import cv2 as cv
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img

# Oxford-IIIT Pet dataset path.
# Put the extracted dataset under one of these common folder structures:
#   ./datasets/oxford_pets/images/images/*.jpg
#   ./datasets/oxford_pets/images/*.jpg
#   ./datasets/oxford_pets/annotations/annotations/trimaps/*.png
#   ./datasets/oxford_pets/annotations/trimaps/*.png
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / 'practice12_results'
OUT_DIR.mkdir(exist_ok=True)

IMG_SIZE = (160, 160)
N_CLASS = 3          # 0: object, 1: background, 2: boundary after subtracting 1
BATCH_SIZE = 32
EPOCHS = 30


def find_existing_dir(candidates):
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError('Dataset folder was not found. Check the Oxford-IIIT Pet dataset path.')


input_dir = find_existing_dir([
    BASE_DIR / 'datasets/oxford_pets/images/images',
    BASE_DIR / 'datasets/oxford_pets/images',
    BASE_DIR / 'oxford_pets/images/images',
    BASE_DIR / 'oxford_pets/images',
])
target_dir = find_existing_dir([
    BASE_DIR / 'datasets/oxford_pets/annotations/annotations/trimaps',
    BASE_DIR / 'datasets/oxford_pets/annotations/trimaps',
    BASE_DIR / 'oxford_pets/annotations/annotations/trimaps',
    BASE_DIR / 'oxford_pets/annotations/trimaps',
])

img_paths = sorted([str(p) for p in input_dir.glob('*.jpg')])
label_paths = sorted([str(p) for p in target_dir.glob('*.png') if not p.name.startswith('.')])

if len(img_paths) == 0 or len(label_paths) == 0:
    raise FileNotFoundError('No Oxford-IIIT Pet images or trimap labels were found.')


class OxfordPets(keras.utils.Sequence):
    def __init__(self, batch_size, img_size, img_paths, label_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_paths = img_paths
        self.label_paths = label_paths

    def __len__(self):
        return len(self.label_paths) // self.batch_size

    def __getitem__(self, idx):
        i = idx * self.batch_size
        batch_img_paths = self.img_paths[i:i + self.batch_size]
        batch_label_paths = self.label_paths[i:i + self.batch_size]

        x = np.zeros((len(batch_img_paths),) + self.img_size + (3,), dtype='float32')
        for j, path in enumerate(batch_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = np.asarray(img, dtype='float32') / 255.0

        y = np.zeros((len(batch_label_paths),) + self.img_size + (1,), dtype='uint8')
        for j, path in enumerate(batch_label_paths):
            img = load_img(path, target_size=self.img_size, color_mode='grayscale')
            y[j] = np.expand_dims(img, 2)
            y[j] -= 1       # Convert trimap labels 1,2,3 into class labels 0,1,2.
        return x, y


def make_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))

    # Downsampling path
    x = layers.Conv2D(32, 3, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    previous_block_activation = x

    for filters in [64, 128, 256]:
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(3, strides=2, padding='same')(x)

        residual = layers.Conv2D(filters, 1, strides=2, padding='same')(previous_block_activation)
        x = layers.add([x, residual])
        previous_block_activation = x

    # Upsampling path
    for filters in [256, 128, 64, 32]:
        x = layers.Activation('relu')(x)
        x = layers.Conv2DTranspose(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2DTranspose(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.UpSampling2D(2)(x)

        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding='same')(residual)
        x = layers.add([x, residual])
        previous_block_activation = x

    outputs = layers.Conv2D(num_classes, 3, activation='softmax', padding='same')(x)
    return keras.Model(inputs, outputs)


# Shuffle in the same way as the practice example.
random.Random(1).shuffle(img_paths)
random.Random(1).shuffle(label_paths)

test_samples = int(len(img_paths) * 0.1)
train_img_paths = img_paths[:-test_samples]
train_label_paths = label_paths[:-test_samples]
test_img_paths = img_paths[-test_samples:]
test_label_paths = label_paths[-test_samples:]

train_gen = OxfordPets(BATCH_SIZE, IMG_SIZE, train_img_paths, train_label_paths)
test_gen = OxfordPets(BATCH_SIZE, IMG_SIZE, test_img_paths, test_label_paths)

model = make_model(IMG_SIZE, N_CLASS)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

weight_path = BASE_DIR / 'oxford_seg.keras'
callbacks = [keras.callbacks.ModelCheckpoint(str(weight_path), save_best_only=True)]
model.fit(train_gen, epochs=EPOCHS, validation_data=test_gen, callbacks=callbacks)

if weight_path.exists():
    model = keras.models.load_model(str(weight_path))

# Select 3 images from the test set and save input / label / prediction results.
selected_indices = [0, 1, 2]
for out_idx, sample_idx in enumerate(selected_indices, start=1):
    img = load_img(test_img_paths[sample_idx], target_size=IMG_SIZE)
    x = np.asarray(img, dtype='float32') / 255.0
    pred = model.predict(np.expand_dims(x, axis=0), verbose=0)[0]
    pred_mask = np.argmax(pred, axis=-1).astype('uint8')

    gt = load_img(test_label_paths[sample_idx], target_size=IMG_SIZE, color_mode='grayscale')
    gt_mask = np.asarray(gt, dtype='uint8') - 1

    input_img = np.uint8(x * 255.0)
    gt_vis = cv.cvtColor(np.uint8(gt_mask * 127), cv.COLOR_GRAY2RGB)
    pred_vis = cv.cvtColor(np.uint8(pred_mask * 127), cv.COLOR_GRAY2RGB)

    combined = np.hstack([input_img, gt_vis, pred_vis])
    cv.putText(combined, 'Input', (10, 24), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv.putText(combined, 'GT label', (IMG_SIZE[1] + 10, 24), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv.putText(combined, 'Prediction', (IMG_SIZE[1] * 2 + 10, 24), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    out_path = OUT_DIR / f'task3_unet_sample_{out_idx}.png'
    cv.imwrite(str(out_path), cv.cvtColor(combined, cv.COLOR_RGB2BGR))
    print(f'Saved: {out_path}')
