# Keywords

## **Representation Learning**

데이터의 특징을 유용한 형식으로 변환하는 것을 목표로 하는 머신 러닝 기법

원본 데이터(예: 이미지, 텍스트)로부터 중요한 특징을 추출하여 저차원 또는 고차원 벡터(임베딩)를 생성한다.

이러한 벡터는 원본 데이터의 중요한 정보를 함축하고 있으며, 이후 다양한 머신 러닝 작업에 사용된다.

주요 목표로는,

- 추상화
    - 데이터의 중요한 패턴이나 구조를 캡처하여, 원본 데이터보다 더 간단하고 유용한 표현을 만듭니다.
    - 예를 들어, 이미지의 픽셀값 대신 고수준의 특징(예: 모서리, 색상, 형태)을 추출합니다.
- 차원 축소
    - 고차원 데이터(예: 고해상도 이미지)를 저차원 벡터로 변환하여, 계산 효율성을 높이고 노이즈를 줄입니다.
    - PCA(Principal Component Analysis)나 t-SNE와 같은 기법이 사용됩니다.
- 일반화 능력
    - 다양한 작업에서 재사용 가능한 일반화된 표현을 학습합니다.
    - 사전 학습된 모델의 표현을 다른 다운스트림 작업에 적용하여, 전이 학습(transfer learning)을 용이하게 합니다.

가 있다.

## **Self-supervised Learning**

Label 없이 input 내에서 target으로 쓰일만 한 것을 정해서, 즉 self로 task를 정해서 supervision방식으로  모델을 학습

일부러 어떤 구실을 만들어서 푸는 문제, pretext task라고 한다.

이 모델을 downstream task에 transfer하여 사용할 수 있다.

Unlabelled dataset으로부터 좋은 representation을 얻고자 하는 학습 방식.

representation learning의 일종.

## **Contrastive Learning**

서로 다른 데이터 포인트 간의 유사성 학습

- 긍정 쌍(Positive Pair): 서로 관련 있는 데이터 포인트 (예: 같은 이미지의 두 다른 뷰).
- 부정 쌍(Negative Pair): 서로 관련 없는 데이터 포인트.

모델은 긍정 쌍의 유사도를 높이고 부정 쌍의 유사도를 낮추는 방향으로 학습된다. 

이를 통해 모델은 데이터의 의미 있는 표현을 학습하게 된다.

Self-supervised learning의 일종.

본 논문에서는 Contrastive learning을 사용하여, 관련 있는 텍스트와 이미지의 임베딩이 가깝도록 학습된다.

## **Pretext task**

모델이 유용한 표현(특징)을 학습하기 위해 수행하는 초기 학습 과제

주로 self-supervised learning에서 사용되며, 레이블이 없는 데이터로부터 의미 있는 표현을 학습하기 위해 설계된다. 

전처리 과제는 본래의 목적과는 다른, 하지만 표현 학습에 도움이 되는 간단한 작업을 의미한다.

본 논문에서는 텍스트-이미지 쌍의 contrastive learning을 pretext task로 사용한다.

## **Downstream task**

모델이 pretext task를 통해 학습한 표현을 실제로 활용하는 본래의 목적 작업

전처리 과제를 통해 학습된 모델의 임베딩을 사용하여, 특정 작업을 수행

후속 과제는 전처리 과제에서 학습된 표현의 유용성을 평가하는 데 사용된다.

본 논문에서는 pretext task를 통해 학습한 표현을 사용하여, 

- Zero-shot learning: 레이블이 없는 새로운 클래스에 대해 분류 작업을 수행.
- Image classification: 임베딩을 사용하여 이미지 분류 작업을 수행. pretext task를 통해 학습한 표현을 활용하여 높은 성능을 발휘한다.

를 downstream task로 사용한다.

## Natural Language Supervision

텍스트를 사용하여 모델을 학습시키는 방법

텍스트 데이터를 라벨로 사용하여 이미지 데이터를 설명하고, 이를 기반으로 모델을 학습시킨다.