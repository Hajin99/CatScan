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
# 반복문(for 루프 등)의 진행 상황을 시각적으로 보여주는 라이브러리
from tqdm import tqdm
from pathlib import Path

image_shape = (256, 256, 3)
num_classes = 2

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

                X.append(full_img_path)
                y.append(label)

        except Exception:
            continue

    # 6. 결과 변환
    X = np.array(X)
    y = tf.keras.utils.to_categorical(y, num_classes=2)

    print(f"로딩 완료! 총 {len(X)}개의 데이터 준비됨.")
    return X, y

# 경로(X)와 라벨(y)을 받아서, 이미지를 배치 단위로 읽어주는 함수
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size, image_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.image_size = image_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        # 이번에 학습할 배치만큼 경로를 꺼냄
        batch_x_paths = self.x[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size: (idx + 1) * self.batch_size]

        images = []
        for path in batch_x_paths:
            # 여기서 실시간으로 이미지를 읽음 (메모리 효율적)
            img_array = np.fromfile(path, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.resize(img, self.image_size[:2])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0
                images.append(img)
            else:
                # 이미지가 깨졌을 경우 검은 화면으로 대체 (에러 방지)
                images.append(np.zeros((self.image_size[0], self.image_size[1], 3)))

        return np.array(images), np.array(batch_y)

# 실행 부분
train_path = r'C:\CatScan\Training'
test_path = r'C:\CatScan\Validation'


# 함수 실행
x_train_paths, y_train = make_data_from_folder(train_path)
x_test_paths, y_test = make_data_from_folder(test_path)

def sample_data(x, y, max_per_class=300):
    """
    각 클래스 별로 일정 개수(max_per_class)만 남기고 샘플링해주는 함수
    """
    y_labels = np.argmax(y, axis=1)
    sampled_x = []
    sampled_y = []

    for cls in np.unique(y_labels):
        idx = np.where(y_labels == cls)[0]
        np.random.shuffle(idx)
        selected = idx[:max_per_class]

        sampled_x.extend(x[selected])
        sampled_y.extend(y[selected])

    sampled_x = np.array(sampled_x)
    sampled_y = np.array(sampled_y)

    # 셔플
    shuffle_idx = np.random.permutation(len(sampled_x))
    return sampled_x[shuffle_idx], sampled_y[shuffle_idx]

# 데이터 줄이기 (예: 각 클래스당 300장만)
x_train_paths, y_train = sample_data(x_train_paths, y_train, max_per_class=300)
x_test_paths,  y_test  = sample_data(x_test_paths,  y_test,  max_per_class=150)

# train 데이터 split
x_train, x_val, y_train, y_val = train_test_split(
    x_train_paths, y_train,
    test_size=0.2,
    random_state=1337,
    stratify=y_train.argmax(axis=1)
)

# validation도 줄이기 (예: 100장만)
x_val, y_val = sample_data(x_val, y_val, max_per_class=100)

# 2. 제너레이터 연결 (경로를 이미지로 바꿔주는 역할)
train_gen = DataGenerator(x_train, y_train, 16, (256, 256, 3))
# validation generator
val_gen   = DataGenerator(x_val, y_val, 16, (256, 256, 3))
test_gen = DataGenerator(x_test_paths, y_test, 16, (256, 256, 3))

epochs = 1

def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = inputs
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
history = model.fit(train_gen, epochs=epochs, validation_data=val_gen,
                    callbacks=[ckpoint, earlystopping])

results = model.evaluate(test_gen)
print(f"Test results - Loss: {results[0]}, Accuracy: {results[1]}")

model.save(f'./model_save/cnn.h5')

model = keras.models.load_model(f'./model_save/cnn.h5')

class_names = ['no_pain', 'pain']

# 첫 배치에서 5개 가져오기
x_sample, y_sample = test_gen[0]
x_input = x_sample[:5]
y_true = y_sample[:5]

y_pred = model.predict(x_input)

# 예측된 클래스와 실제 클래스 비교
for i in range(5):
    predicted_class = np.argmax(y_pred[i])  # 예측된 클래스 인덱스
    true_class = np.argmax(y_test[i])  # 실제 클래스 인덱스

    print(f"Sample {i + 1}:")
    print(f"    Actual:    {class_names[true_class]}")
    print(f"    Predicted: {class_names[predicted_class]}")
    print()

def count_class_distribution(paths, labels):
    counts = {0: 0, 1: 0}
    for i in range(len(paths)):
        cls = int(np.argmax(labels[i]))
        counts[cls] += 1
    print("Class distribution:", counts)

print("Train:")
count_class_distribution(x_train, y_train)

print("\nValidation:")
count_class_distribution(x_val, y_val)

print("\nTest:")
count_class_distribution(x_test_paths, y_test)
