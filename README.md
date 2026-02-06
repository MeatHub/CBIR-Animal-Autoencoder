🦁 Animal Image Retrieval System (CBIR)
Convolutional Autoencoder(CAE)를 활용한 비지도 학습 기반의 동물 이미지 검색 시스템입니다. 이미지의 픽셀 값을 직접 비교하는 것이 아니라, 딥러닝 모델을 통해 추출한 잠재 특징(Latent Feature)을 기반으로 유사한 이미지를 검색합니다.

📌 프로젝트 개요
주제: 오토인코더(Autoencoder) 기반의 Content-Based Image Retrieval (CBIR) 구현

목표: 동물 이미지 데이터셋을 학습하여, 입력된 Query 이미지와 시각적(색상, 질감, 형태)으로 가장 유사한 이미지를 검색하고 시각화한다.

핵심 기술: Deep Learning (PyTorch), Convolutional Autoencoder, KNN (K-Nearest Neighbors), Latent Space Representation

🛠 사용 기술 및 라이브러리 (Tech Stack)
Language: Python 3.x

Deep Learning: PyTorch, Torchvision

Data Processing: NumPy, Pandas, PIL

Visualization: Matplotlib, Tqdm

Utility: Scikit-learn (KNN), Torchsummary

📂 데이터셋 (Dataset)
출처: Kaggle CBIR Dataset (Animal Images)

구성: 다양한 동물(개, 고양이, 야생동물 등)의 이미지 데이터

전처리: Resize-> 128x128 픽셀

Normalization-> 픽셀 값을 -1 ~ 1 범위로 정규화 (Tanh 활성화 함수 대응)

🏗 모델 구조 (Model Architecture)
이미지를 압축하고 복원하는 Convolutional Autoencoder를 설계하여 사용했습니다.

1. Encoder (Feature Extractor)
이미지의 차원을 줄이며 핵심 특징(Latent Vector)을 추출합니다.

구조: Conv2d -> ReLU (3단계)

입력: (3, 128, 128) → 출력: (64, 16, 16)

2. Decoder (Reconstructor)
압축된 특징을 다시 원본 이미지 크기로 복원합니다.

구조: ConvTranspose2d -> ReLU -> Tanh

입력: (64, 16, 16) → 출력: (3, 128, 128)

🚀 실행 과정 (Process)
Step 1. 모델 학습 (Training)
입력 이미지와 Decoder가 복원한 이미지 사이의 차이(MSE Loss)를 최소화하도록 학습합니다.

이 과정을 통해 Encoder는 이미지를 가장 잘 요약하는 방법을 학습합니다.

Step 2. 특징 추출 (Indexing)
학습된 모델의 Encoder 부분만 사용합니다.

데이터셋의 모든 이미지를 통과시켜 고차원 벡터(Latent Feature)로 변환하여 저장합니다.

Step 3. 유사 이미지 검색 (Retrieval)
Query 이미지가 입력되면 동일하게 Encoder를 통과시켜 특징 벡터를 얻습니다.

KNN(K-Nearest Neighbors) 알고리즘을 사용하여, 저장된 특징 벡터들 중 유클리드 거리(Euclidean Distance)가 가장 가까운 Top-K 이미지를 찾습니다.

📊 결과 시각화 (Visualization)
검색 결과를 한눈에 비교할 수 있도록 Grid 형태로 시각화했습니다.

Query Image: 사용자가 입력한 기준 이미지

Rank 1 ~ 11: 가장 유사도가 높은 순서대로 나열된 검색 결과 (Distance 값 포함)

<img src="https://github.com/user-attachments/assets/03efb5e9-47a6-4f8f-92ef-63fae8f32d6c" width="800" alt="결과 이미지">



💻 설치 및 실행 방법 (How to Run)
필수 라이브러리 설치

Bash
pip install torch torchvision matplotlib scikit-learn tqdm torchsummary
데이터셋 준비

데이터셋 압축 파일을 해제하고 경로를 지정합니다.

코드 실행

Jupyter Notebook 또는 Python 스크립트를 실행하여 학습 및 검색을 진행합니다.

Author: [MeatHub] Date: 2024.02.06
