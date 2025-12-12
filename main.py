import tensorflow as tf
import tensorflow_addons as tfa
import os
import json
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
# 반복문(for 루프 등)의 진행 상황을 시각적으로 보여주는 라이브러리
from tqdm import tqdm
from pathlib import Path

def make_data_from_folder(route):
    X = []
    y = []

    # 1. CAT 폴더 경로
    folder_path = os.path.join(route, 'CAT')

    if not os.path.exists(folder_path):
        print(f'경로 에러: {folder_path} 폴더가 없습니다.')
        return np.array(X), np.array(y)

    print(f'데이터 로딩 시작: {folder_path} (하위 폴더 검색 중...)')

    # 2. 모든 하위 폴더를 뒤져서 .json 파일 찾기 (재귀 탐색)
    # 윈도우/맥 호환을 위해 Path 객체 사용
    # [label]로 시작하는 폴더 안의 모든 json을 찾기.
    label_files = list(Path(folder_path).glob('label*/**/*.json'))

    if len(label_files) == 0:
        print("JSON 파일을 찾지 못했습니다. 경로를 확인해주세요.")
        return np.array(X), np.array(y)

    print(f"총 {len(label_files)}개의 라벨(JSON) 파일을 찾았습니다. 이미지 매칭 시작...")

    for label_path in tqdm(label_files):
        try:
            label_path_str = str(label_path)

            # 4. JSON 읽어서 통증(Pain) 라벨 확인
            with open(label_path_str, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 데이터 구조에 따라 pain 위치 찾기
            is_pain = False

            # 'owner' 안에 있는 'pain' 확인
            owner_info = data.get("metadata", {}).get("owner")
            if owner_info:
                pain_value = owner_info.get("pain")
                # "Y"면 통증 있음, "N"이면 없음
                if pain_value == "Y":
                    is_pain = True

            label = 1 if is_pain else 0

            # 3. 짝꿍 이미지 경로 찾기
            # [label] -> [img] 로 변경, .json 확장자 없애고 폴더명으로 만들기
            img_path_str = label_path_str.replace("label", "img").replace(".json", "")

            # 폴더 안의 이미지 파일들 하나씩 읽기
            file_list = os.listdir(img_path_str)

            for file_name in file_list:
                if not (file_name.lower().endswith(".jpg")):
                    continue

                full_img_path = os.path.join(img_path_str, file_name)

                # 5. 이미지 로드 (한글 경로 호환)
                img_array = np.fromfile(full_img_path, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if img is None: continue

                # 리사이즈 & 컬러 변환
                img = cv2.resize(img, (256, 256))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                X.append(img)
                y.append(label)

        except Exception:
            continue

    # 6. 결과 변환
    X = np.array(X).astype(np.float32) / 255.0  # 정규화
    y = np.array(y).astype(np.float32)

    # 원-핫 인코딩 (필수)
    y = tf.keras.utils.to_categorical(y, num_classes=2)

    print(f"로딩 완료! 총 {len(X)}개의 데이터 준비됨.")
    return X, y

# 실행 부분
train_path = r'C:\CatScan\Training'
test_path = r'C:\CatScan\Validation'

# 함수 실행
x_train, y_train = make_data_from_folder(train_path)
x_test, y_test = make_data_from_folder(test_path)

#
# x_train1, y_train1 = make_test_data(train_path)
# x_test, y_test = make_test_data(test_path)
#
# x_train, x_val, y_train, y_val = train_test_split(x_train1, y_train1, test_size=1 - 0.6, random_state=1337,
#                                                   stratify=y_train1)
#
# encoder = LabelEncoder()
#
# encoder.fit(y_train)
# encoder.fit(y_val)
# encoder.fit(y_test)
#
# y_train = encoder.transform(y_train)
# y_val = encoder.transform(y_val)
# y_test = encoder.transform(y_test)
#
# y_train = tf.keras.utils.to_categorical(y_train)
# y_val = tf.keras.utils.to_categorical(y_val)
# y_test = tf.keras.utils.to_categorical(y_test)
#
# epochs = 30
#
#
# def make_model(input_shape, num_classes):
#     inputs = keras.Input(shape=input_shape)
#
#     # Entry block
#     x = layers.Rescaling(1.0 / 255)(inputs)
#     x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation("relu")(x)
#
#     previous_block_activation = x  # Set aside residual
#
#     for size in [256, 512, 728]:
#         x = layers.Activation("relu")(x)
#         x = layers.SeparableConv2D(size, 3, padding="same")(x)
#         x = layers.BatchNormalization()(x)
#
#         x = layers.Activation("relu")(x)
#         x = layers.SeparableConv2D(size, 3, padding="same")(x)
#         x = layers.BatchNormalization()(x)
#
#         x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
#
#         residual = layers.Conv2D(size, 1, strides=2, padding="same")(
#             previous_block_activation
#         )
#         x = layers.add([x, residual])  # Add back residual
#         previous_block_activation = x  # Set aside next residual
#
#     x = layers.SeparableConv2D(1024, 3, padding="same")(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation("relu", name='last_layer')(x)
#
#     x = layers.GlobalAveragePooling2D()(x)
#     activation = "softmax"
#     outputs = layers.Dense(num_classes, activation=activation)(x)
#     return keras.Model(inputs, outputs)
#
#
# model = make_model(image_shape, num_classes=num_classes)
#
# ckpoint = tf.keras.callbacks.ModelCheckpoint(filepath=f'./model_save/cnn.h5', monitor='val_loss', save_best_only=True,
#                                              save_weights_only=True)
# earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='max')
#
# precision = tf.keras.metrics.Precision()
# recall = tf.keras.metrics.Recall()
# auc = tf.keras.metrics.AUC()
# f1_score = tfa.metrics.F1Score(num_classes=num_classes, threshold=0.5)
# learningrate = 0.0007
#
# model.summary()
#
# optimizer = tf.keras.optimizers.Adam(learning_rate=learningrate)
#
# model.compile(
#     loss="categorical_crossentropy",
#     optimizer=optimizer,
#     metrics=["accuracy", precision, recall, auc, f1_score]
# )
#
# history = model.fit(x_train, y_train, batch_size=32, epochs=epochs, validation_data=(x_val, y_val),
#                     callbacks=[ckpoint, earlystopping])
#
# results = model.evaluate(x_test, y_test)
# print(f"Test results - Loss: {results[0]}, Accuracy: {results[1]}")
#
# model.save(f'./model_save/cnn.h5')
#
# model = keras.models.load_model(f'./model_save/cnn.h5')
#
# class_names = ['pituitary_tumor', 'no_tumor', 'meningioma_tumor', 'glioma_tumor']
#
# y_pred = model.predict(x_test[:5])
#
# # 예측된 클래스와 실제 클래스 비교
# for i in range(5):
#     predicted_class = np.argmax(y_pred[i])  # 예측된 클래스 인덱스
#     true_class = np.argmax(y_test[i])  # 실제 클래스 인덱스
#
#     print(f"Sample {i + 1}:")
#     print(f"    Actual:    {class_names[true_class]}")
#     print(f"    Predicted: {class_names[predicted_class]}")
#     print()
