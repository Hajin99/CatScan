# 1. Basic CNN (기초)
# 데이터 불균형 문제로 적은 데이터 수 증강 시도.. 성능이 더 안좋아짐...
# 결국 pain->disease 사용으로 바꿨다.

import tensorflow as tf
import tensorflow_addons as tfa
import os
import json
import cv2
import numpy as np
import keras

os.makedirs('checkpoints', exist_ok=True)
from sklearn.model_selection import train_test_split
import math
from tensorflow.keras import layers
from tqdm import tqdm
from pathlib import Path
from sklearn.utils import class_weight

image_shape = (256, 256, 3)
num_classes = 2


# ==============================================================================
# 1. 데이터 로드 함수 (엄격한 기준: Pain="Y"만 찾기)
# ==============================================================================
def make_data_from_folder(route):
    X = []
    y = []

    folder_path = os.path.join(route, 'CAT')
    if not os.path.exists(folder_path):
        return np.array(X), np.array(y)

    # 모든 하위 폴더 검색 (SITDOWN, ARCH, ROLL, LYING 전부)
    label_files = list(Path(folder_path).glob('label*/**/*.json'))

    print(f"[{route}] 검색 중... 파일 {len(label_files)}개 확인")

    for label_path in tqdm(label_files):
        try:
            label_path_str = str(label_path)
            with open(label_path_str, "r", encoding="utf-8") as f:
                data = json.load(f)

            is_pain = False

            # --- [수동 기준] 오직 주인이 "Y"라고 한 것만 Pain으로 인정 ---
            owner_info = data.get("metadata", {}).get("owner", {})
            if owner_info and owner_info.get("disease") == "Y":
                is_pain = True
            # --------------------------------------------------------

            label = 1 if is_pain else 0

            # 이미지 경로 매칭
            img_path_str = label_path_str.replace("label", "img").replace(".json", "")

            if not os.path.exists(img_path_str):
                continue

            file_list = os.listdir(img_path_str)
            file_list = [f for f in file_list if f.lower().endswith(".jpg")]

            if len(file_list) == 0:
                continue

            # 대표 이미지 1장만 가져오기 (중간 프레임)
            file_list.sort()
            selected_file = file_list[len(file_list) // 2]
            full_img_path = os.path.join(img_path_str, selected_file)

            X.append(full_img_path)
            y.append(label)

        except Exception:
            continue

    return np.array(X), np.array(y)  # One-hot 인코딩은 나중에 함


# ==============================================================================
# 2. 전체 데이터 로드 및 '수동' 비율 맞추기
# ==============================================================================

print("1. 전체 데이터 긁어모으기 (시간이 좀 걸릴 수 있습니다)...")
x_tr, y_tr = make_data_from_folder(r'C:\CatScan\Training')
x_te, y_te = make_data_from_folder(r'C:\CatScan\Test')

# 전체 병합
all_x = np.concatenate([x_tr, x_te])
all_y = np.concatenate([y_tr, y_te])  # 아직 0, 1 정수 형태

# 인덱스 분리
pain_idx = np.where(all_y == 1)[0]
nopain_idx = np.where(all_y == 0)[0]

print(f"\n=== [중요] 확보된 데이터 현황 ===")
print(f"통증(Pain=Y) 데이터: {len(pain_idx)}장")
print(f"정상(Pain=N) 데이터: {len(nopain_idx)}장")

# --------------------------------------------------------------------------
# [핵심] 수동 밸런싱: 통증 데이터는 다 쓰고, 정상 데이터는 통증의 N배수만 쓴다
# --------------------------------------------------------------------------
# 비율 설정 (예: 1:2 비율 추천 - 데이터가 너무 적으므로 정상 데이터라도 좀 더 확보)
ratio = 2
target_nopain_count = len(pain_idx) * ratio

# 정상 데이터 랜덤 셔플 후 자르기
np.random.shuffle(nopain_idx)
selected_nopain_idx = nopain_idx[:target_nopain_count]

# 최종 사용할 인덱스 합치기
final_idx = np.concatenate([pain_idx, selected_nopain_idx])
np.random.shuffle(final_idx)

# 데이터 필터링 적용
X_final = all_x[final_idx]
y_final = all_y[final_idx]
y_final_onehot = tf.keras.utils.to_categorical(y_final, num_classes=2)  # 이제 One-hot 변환

print(f"=== 최종 학습에 사용할 데이터 ===")
print(f"총 {len(X_final)}장 (통증 {len(pain_idx)} : 정상 {len(selected_nopain_idx)})")

# ==============================================================================
# 3. Train / Val / Test 분할 (이제 개수가 적으므로 신중하게)
# ==============================================================================
# Test셋에 최소한의 Pain 데이터(예: 20%)는 남겨둬야 검증이 됩니다.
x_train, x_test, y_train, y_test = train_test_split(
    X_final, y_final_onehot,
    test_size=0.2,
    random_state=42,
    stratify=y_final  # 비율 유지하며 자름
)

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)


# ==============================================================================
# 4. 제너레이터 및 모델 (기존 코드 연결)
# ==============================================================================

# 경로(X)와 라벨(y)을 받아서, 이미지를 배치 단위로 읽어주는 함수
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size, image_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.image_size = image_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x_paths = self.x[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size: (idx + 1) * self.batch_size]
        images = []
        for path in batch_x_paths:
            img_array = np.fromfile(path, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.resize(img, self.image_size[:2])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0
                images.append(img)
            else:
                images.append(np.zeros((self.image_size[0], self.image_size[1], 3)))
        return np.array(images), np.array(batch_y)


# 배치 사이즈를 데이터가 적으니 좀 줄입니다 (16 -> 8)
BATCH_SIZE = 8
train_gen = DataGenerator(x_train, y_train, BATCH_SIZE, (256, 256, 3))
val_gen = DataGenerator(x_val, y_val, BATCH_SIZE, (256, 256, 3))
test_gen = DataGenerator(x_test, y_test, BATCH_SIZE, (256, 256, 3))

# 클래스 가중치 (비율을 수동으로 맞췄으니 1:1에 가깝겠지만 계산해둡니다)
y_labels_idx = np.argmax(y_train, axis=1)
if len(np.unique(y_labels_idx)) > 1:
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_labels_idx), y=y_labels_idx)
    class_weights_dict = dict(enumerate(class_weights))
else:
    class_weights_dict = {0: 1.0, 1: 1.0}


# 모델 정의 (데이터가 매우 적으므로 과적합 방지 Dropout 필수)
def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # ★ 데이터 증강 (필수)
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.2)(x)  # 회전 좀 더 많이
    x = layers.RandomZoom(0.2)(x)

    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)  # 필터 수도 좀 줄임 (데이터가 적어서)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    x = layers.SeparableConv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)  # Dropout 강화

    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


model = make_model(image_shape, num_classes)

# 콜백 설정
ckpoint = tf.keras.callbacks.ModelCheckpoint(filepath='./model_save/cnn.h5', monitor='val_loss', save_best_only=True,
                                             save_weights_only=False)
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, mode='min',
                                                 restore_best_weights=True)  # patience 늘림

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)  # 학습률 낮춤

model.compile(
    loss="categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
)

print("\n학습 시작...")
history = model.fit(train_gen, epochs=10, validation_data=val_gen,
                    callbacks=[ckpoint, earlystopping],
                    class_weight=class_weights_dict)

print("\n테스트 평가:")
model.evaluate(test_gen)


# 분포 확인 출력
def count_dist(name, y_data):
    c = np.sum(y_data, axis=0)
    print(f"{name}: Normal={int(c[0])}, Pain={int(c[1])}")


print("\n--- 최종 데이터 분포 ---")
count_dist("Train", y_train)
count_dist("Val  ", y_val)
count_dist("Test ", y_test)