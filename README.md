# Learning Transferable Visual Models From Natural Language Supervision

### Abstract

CLIP (Contrastive Language-Image Pre-Training)은 4억 개의 텍스트-이미지 쌍으로 학습되었으며, 사전 학습된 모델을 다양한 이미지 인식 데이터셋에서 제로샷(Zero-Shot)으로 적용할 수 있다. 

CLIP이 30개의 다양한 이미지 분류 데이터셋에서 높은 성능을 발휘하며, 특히 제로샷 학습에서 경쟁력 있는 성능을 보이는 것을 확인했다. 또한, CLIP은 다양한 인식 작업에 있어 기존의 최첨단 모델들과 비교해도 우수한 성능을 나타냈다.

### [Keywords](./subpages/Keywords.md)

# 1. Introduction and Motivating Work

이 장에서는 연구의 배경과 동기 부여를 설명합니다. CLIP 모델의 개발 동기와 기존 방법론의 한계를 논의하며, 자연어 감독을 활용한 새로운 접근법의 필요성을 제시합니다.

# 2. Approach

## 2.1 Natural Language Supervision

자연어 감독에서 학습하는 방법을 설명합니다. 이미지와 텍스트를 결합하여 지각 학습을 수행하는 기존 연구들과의 차이점을 강조합니다.

## 2.2 Creating a Sufficiently Large Dataset

대규모 데이터셋을 생성하는 과정과 그 중요성에 대해 설명합니다. 4억 개의 이미지-텍스트 쌍으로 이루어진 데이터셋을 만들고 이를 통해 모델의 성능을 높이는 방법을 제시합니다.

## 2.3 Selecting an Efficient Pre-Training Method

효율적인 사전 학습 방법을 선택하는 과정과 그 이유를 설명합니다. ConVIRT의 단순화 버전인 CLIP을 사용하여 자연어 감독을 효과적으로 학습할 수 있음을 보여줍니다.

## 2.4 Choosing and Scaling a Model

모델을 선택하고 확장하는 과정에 대해 설명합니다. 다양한 컴퓨팅 규모에서 모델을 훈련시켜 전이 성능을 분석합니다.

## 2.5 Training

모델 훈련 과정에 대해 설명합니다. 5개의 ResNets와 3개의 Vision Transformers를 사용하여 모델을 훈련시키고, 이를 통해 CLIP의 강력한 성능을 입증합니다.

# 3. Experiments

## 3.1 Zero-Shot Transfer

### 3.1.1 Motivation

제로샷 학습의 필요성과 동기에 대해 설명합니다. 기존 모델과 비교하여 CLIP의 장점을 논의합니다.

### 3.1.2 Using CLIP For Zero-Shot Transfer

CLIP을 제로샷 전이에 사용하는 방법을 설명합니다. 다양한 데이터셋에서의 성능 평가 결과를 제시합니다.

### 3.1.3 Initial Comparison to Visual N-Grams

Visual N-Grams와의 초기 비교 결과를 제시합니다. CLIP의 우수한 성능을 강조합니다.

### 3.1.4 Prompt Engineering and Ensembling

프롬프트 엔지니어링과 앙상블 방법을 사용하여 CLIP의 성능을 최적화하는 과정을 설명합니다.

### 3.1.5 Analysis of Zero-Shot CLIP Performance

제로샷 CLIP 성능 분석 결과를 제시합니다. 다양한 작업에서의 성능을 평가합니다.

## 3.2 Representation Learning

표현 학습에 대한 실험 결과를 설명합니다. CLIP이 다양한 작업에서 강력한 표현을 학습할 수 있음을 보여줍니다.

## 3.3 Robustness to Natural Distribution Shift

자연스러운 분포 변화에 대한 견고성을 평가한 결과를 설명합니다. CLIP의 강력한 성능을 입증합니다.

# 4. Comparison to Human Performance

인간 성능과의 비교 결과를 설명합니다. CLIP이 인간과 유사한 수준의 성능을 발휘할 수 있음을 보여줍니다.

# 5. Data Overlap Analysis

데이터 중복 분석 결과를 설명합니다. CLIP이 훈련된 데이터와 테스트 데이터 간의 중복을 최소화하는 방법을 제시합니다.

# 6. Limitations

CLIP의 한계점을 설명합니다. 모델의 성능을 개선하기 위해 추가 연구가 필요한 부분을 논의합니다.

# 7. Broader Impacts

## 7.1 Bias

CLIP의 편향 문제를 논의합니다. 편향을 줄이기 위한 방법을 제안합니다.

## 7.2 Surveillance

감시와 관련된 윤리적 문제를 논의합니다. CLIP이 악용될 가능성에 대해 경고합니다.

## 7.3 Future Work

향후 연구 방향을 제시합니다. CLIP의 성능을 더욱 향상시키기 위한 방법을 논의합니다.

# 8. Related Work

기존 연구와의 관련성을 설명합니다. CLIP이 기존 연구들과 어떻게 차별화되는지 강조합니다.

# 9. Conclusion

연구의 결론을 요약합니다. CLIP의 성과와 향후 연구 방향을 제시합니다.
