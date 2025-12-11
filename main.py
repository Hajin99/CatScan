import tensorflow as tf
import tensorflow_addons as tfa
import os
import cv2
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import keras
from keras import preprocessing

os.makedirs('checkpoints', exist_ok=True)
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn import metrics
import math
from tensorflow.keras import layers

image_shape = (256, 256, 3)
num_classes = 4


def make_test_data(route):
    X = []
    y = []

    for _, folder_name in enumerate(('pituitary_tumor', 'no_tumor', 'meningioma_tumor', 'glioma_tumor')):
        folder_path = os.path.join(route, folder_name)

        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)

            img = cv2.imread(fpath, cv2.IMREAD_COLOR)
            img = cv2.resize(img, image_shape[:2])

            X.append(img)
            if folder_name == "pituitary_tumor":
                y.append(0)
            elif folder_name == "no_tumor":
                y.append(1)
            elif folder_name == "meningioma_tumor":
                y.append(2)
            elif folder_name == "glioma_tumor":
                y.append(3)

    X = np.array(X)
    y = np.array(y).reshape(-1, 1)

    X = (X / 255.)
    # X = X.transpose(0, 3, 1, 2)
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    return X, y


train_path = './data/Training'
test_path = './data/Testing'

x_train1, y_train1 = make_test_data(train_path)
x_test, y_test = make_test_data(test_path)

x_train, x_val, y_train, y_val = train_test_split(x_train1, y_train1, test_size=1 - 0.6, random_state=1337,
                                                  stratify=y_train1)

encoder = LabelEncoder()

encoder.fit(y_train)
encoder.fit(y_val)
encoder.fit(y_test)

y_train = encoder.transform(y_train)
y_val = encoder.transform(y_val)
y_test = encoder.transform(y_test)

y_train = tf.keras.utils.to_categorical(y_train)
y_val = tf.keras.utils.to_categorical(y_val)
y_test = tf.keras.utils.to_categorical(y_test)

epochs = 30


def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu", name='last_layer')(x)

    x = layers.GlobalAveragePooling2D()(x)
    activation = "softmax"
    outputs = layers.Dense(num_classes, activation=activation)(x)
    return keras.Model(inputs, outputs)


model = make_model(image_shape, num_classes=num_classes)

ckpoint = tf.keras.callbacks.ModelCheckpoint(filepath=f'./model_save/cnn.h5', monitor='val_loss', save_best_only=True,
                                             save_weights_only=True)
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='max')

precision = tf.keras.metrics.Precision()
recall = tf.keras.metrics.Recall()
auc = tf.keras.metrics.AUC()
f1_score = tfa.metrics.F1Score(num_classes=num_classes, threshold=0.5)
learningrate = 0.0007

model.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=learningrate)

model.compile(
    loss="categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy", precision, recall, auc, f1_score]
)

history = model.fit(x_train, y_train, batch_size=32, epochs=epochs, validation_data=(x_val, y_val),
                    callbacks=[ckpoint, earlystopping])

results = model.evaluate(x_test, y_test)
print(f"Test results - Loss: {results[0]}, Accuracy: {results[1]}")

model.save(f'./model_save/cnn.h5')

model = keras.models.load_model(f'./model_save/cnn.h5')

class_names = ['pituitary_tumor', 'no_tumor', 'meningioma_tumor', 'glioma_tumor']

y_pred = model.predict(x_test[:5])

# 예측된 클래스와 실제 클래스 비교
for i in range(5):
    predicted_class = np.argmax(y_pred[i])  # 예측된 클래스 인덱스
    true_class = np.argmax(y_test[i])  # 실제 클래스 인덱스

    print(f"Sample {i + 1}:")
    print(f"    Actual:    {class_names[true_class]}")
    print(f"    Predicted: {class_names[predicted_class]}")
    print()
