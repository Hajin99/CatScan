# CatScan
🔍 캣스캔: 고양이 자세를 통한 통증·질병 조기 탐지 인공지능 모델

## Project
- 고양이의 특정 자세 4가지(arch, lying, roll, sitdown) 이미지를 분석하여 '질병 있음(Y)' vs '없음(N)'을 분류
- input : Cat Image Data(jpg)
- model : Mini-Xception, MobileNetV2, ResNet50, EfficientNetB0

## Data 
- [이미지 출처]
https://www.aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=115&topMenu=100&srchOptnCnd=OPTNCND001&searchKeyword=%EA%B3%A0%EC%96%91%EC%9D%B4&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&aihubDataSe=data&dataSetSn=59

## Folder
<img width="219" height="789" alt="image" src="https://github.com/user-attachments/assets/4c2d2091-650c-44c5-b16a-8f29b3ef2198" />

## Environment
- Anaconda PowerShell Prompt

## Code
0. 가상환경이 없는 경우 다음 명령어로 가상환경 만들기

```
conda create --name <환경이름> python=3.9
```
⬇️ e.g.
```
conda create --name aimedic python=3.9
```

1. 가상환경 활성화 및 필요한 라이브러리 설치
```
conda activate <환경 이름>
pip install -r .\requirements.txt
```
(만약 다음과 같은 오류 코드가 출력되었다면 필요한 라이브러리를 직접 설치)
```
ERROR: Ignored the following versions that require a different python version: 0.28.0 Requires-Python >=3.7, <3.11; 1.21.2 Requires-Python >=3.7,<3.11; 1.21.3 Requires-Python >=3.7,<3.11; 1.21.4 Requires-Python >=3.7,<3.11; 1.21.5 Requires-Python >=3.7,<3.11; 1.21.6 Requires-Python >=3.7,<3.11
ERROR: Could not find a version that satisfies the requirement tensorflow-io-gcs-filesystem==0.37.1 (from versions: 0.29.0, 0.30.0, 0.31.0)
ERROR: No matching distribution found for tensorflow-io-gcs-filesystem==0.37.1
```
라이브러리 설치 명령어
```
pip install <라이브러리 이름>
```
⬇️ e.g.
```
ModuleNotFoundError: No module named 'tensorflow'
--> pip install tensorflow
ModuleNotFoundError: No module named 'cv2'
--> pip install opencv-python
```


2. 코드 실행
```
python main.py
```
