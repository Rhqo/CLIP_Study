# Learning Transferable Visual Models From Natural Language Supervision

### Abstract

CLIP (Contrastive Language-Image Pre-Training)은 4억 개의 텍스트-이미지 쌍으로 학습되었으며, 사전 학습된 모델을 다양한 이미지 인식 데이터셋에서 제로샷(Zero-Shot)으로 적용할 수 있다. 

CLIP이 30개의 다양한 이미지 분류 데이터셋에서 높은 성능을 발휘하며, 특히 제로샷 학습에서 경쟁력 있는 성능을 보이는 것을 확인했다. 또한, CLIP은 다양한 인식 작업에 있어 기존의 최첨단 모델들과 비교해도 우수한 성능을 나타냈다.

### [Keywords](./subpages/Keywords.md)

# 1. Introduction and Motivating Work

> 이 장에서는 연구의 배경과 동기 부여를 설명합니다. CLIP 모델의 개발 동기와 기존 방법론의 한계를 논의하며, 자연어 감독을 활용한 새로운 접근법의 필요성을 제시합니다.

# 2. Approach

## 2.1 Natural Language Supervision

> 자연어 감독에서 학습하는 방법을 설명합니다. 이미지와 텍스트를 결합하여 지각 학습을 수행하는 기존 연구들과의 차이점을 강조합니다.

Training signal 로써의 자연어를 이해하는 것.

원래는 토픽 모델과 n-그램 표현을 사용할 때 자연어의 복잡성으로 어려움을 겪었지만, 심층 컨텍스트 표현 학습(Word2Vec 등)의 개선은 이제 이 풍부한 감독 소스를 효과적으로 활용할 수 있는 도구를 갖게 되었음.

자연어를 통한 학습은 다른 훈련 방법에 비해 몇 가지 잠재적 강점이 있다.

- 이미지 분류를 위한 라벨링에 비해 natural language supervision을 확장하는 것이 쉽다.
- 자연어를 통한 학습은 표현을 "단순히" 학습할 뿐만 아니라 해당 표현을 유연한 zero-shot transfer를 가능하게 하는 언어에 연결한다는 점에서 대부분의 다른  unsupervised 또는 self-supervised 접근 방식에 비해 좋다.

## 2.2 Creating a Sufficiently Large Dataset

> 대규모 데이터셋을 생성하는 과정과 그 중요성에 대해 설명합니다. 4억 개의 이미지-텍스트 쌍으로 이루어진 데이터셋을 만들고 이를 통해 모델의 성능을 높이는 방법을 제시합니다.

기존 작업은 주로 MS-COCO, Visual Genome, YFCC100M의 세 가지 데이터 세트를 사용했다.

MS-COCO와 Visual Genome은  100,000장의 훈련 사진으로 데이터셋의 크기가 충분하지 않다.

YFCC100M은 1억 장의 사진으로 데이터셋의 크기가 꽤 크지만, 각 이미지에 대한 메타데이터는 희소하고 품질이 다양한데, 이 데이터셋 중 자연어 제목 및/또는 설명이 있는 이미지만 영어로 유지하도록 필터링한 후 데이터 세트는 600~1,500만 장의 사진으로 축소되었다. 이는 이미지넷과 거의 동일한 크기이다.

자연어 감독의 주요 동기는 인터넷에서 공개적으로 사용할 수 있는 이러한 형태의 많은 양의 데이터이다. 기존 데이터 세트는 이러한 가능성을 적절하게 반영하지 못하기 때문에 이에 대한 결과만 고려하면 이 연구 라인의 잠재력을 과소평가할 수 있다.

이를 해결하기 위해 인터넷에서 공개적으로 사용 가능한 다양한 소스에서 수집된 4억 개의 (이미지, 텍스트) 쌍으로 구성된 새로운 데이터 세트를 구성했다. 

가능한 한 광범위한 시각적 개념을 다루기 위해 텍스트가 500,000개의 쿼리 세트 중 하나를 포함하는 (이미지, 텍스트) 쌍을 구성 프로세스의 일부로 검색한다.

1 쿼리당 최대 20,000개의 (이미지, 텍스트) 쌍을 포함하여 결과의 균형을 대략적으로 조정합니다. 

결과 데이터 세트는 GPT-2를 훈련하는 데 사용되는 웹텍스트 데이터 세트와 유사한 총 단어 수를 가지고 있다.

이 데이터 세트는 WebImageText의 WIT이다.

## 2.3 Selecting an Efficient Pre-Training Method

> 효율적인 사전 학습 방법을 선택하는 과정과 그 이유를 설명합니다. ConVIRT의 단순화 버전인 CLIP을 사용하여 자연어 감독을 효과적으로 학습할 수 있음을 보여줍니다.

훈련의 효율이 natural language supervision의 key가 될 것이라고 생각했다.

처음에는 image CNN과 text transformer를 사용했는데, 너무 효율이 좋지 못했다.

그래서 단어의 “정확한”의미를 예측하기보다는 contrastive learning을 사용해, 단어가 정확히 일치하지 않아도 되도록 했다.

그리고 N쌍의 positive pair 과 N^2-N의 negative pair의 모든 계산을 하는 것도 비효율적이므로, 

- symmetric cross entropy loss
- batch construction technique and objective (= InfoNCE loss)

를 사용해 이 유사도를 최적화했다.

큰 데이터를 사용하기 때문에, overfitting은 큰 문제가 되지는 않을 것이다.

ImageNet과 Text encoder의 pre-trained weight를 (initializing 없이) 그대로 사용했다.

Representation과 contrastive embedding space에 non-linear projection 대신 linear projection을 사용한다.

linear 이든 non-linear 이든 training efficiency에는 차이가 없었지만, non-linear는 self-supervised learning에서’만’ 현재 이미지의 세부 정보와 함께 사용할 수 있다고 추측했다.

단일 문장을 균일하게 샘플링하는 텍스트 변환 함수 $t_u$를 제거, 이미지 변환 함수 $t_v$를 단순화하였다.

크기가 조정된 이미지의 random 사각형 crop이 훈련중의 유일한 데이터 증강이다.

Softmax의 logits를 조정하는 temperature parameter $\tau$가 hyperparameter가 되는 것을 피하기 위해 log-parameterized된 곱셈 스칼라로 훈련 중에 최적화된다. (logits는 softmax의 입력, 각 클래스에 대한 스코어)

→ $\tau$가 hyperparameter가 되는 것을 피해야 하는 이유 : 학습 과정의 복잡성 감소, 자동 조정, 일관성 유지 (?)

→ log-parameterized multiplicative scalar는 $\tau$를 학습 가능한 파라미터로 만드는 한 가지 방법, $\tau$를 직접 학습하는 대신, $\tau$의 로그 값을 학습하며, $\tau$가 입력 logits에 곱해지는 형태가 되도록 사용된다.

## 2.4 Choosing and Scaling a Model

> 모델을 선택하고 확장하는 과정에 대해 설명합니다. 다양한 컴퓨팅 규모에서 모델을 훈련시켜 전이 성능을 분석합니다.

첫번째 image encoder로는 ResNet-50과 ResNet-D를 사용한 변형을 사용했는데,
global average pooling layer를 attention pooling mechanism으로 바꿨다. 
이 attention pooling mechanism은 transformer style의 multi-head attention으로서 작동한다.
이 attention에서 query는 global average pooling된 표현을 기반으로 한다.

두번째 image encoder로는 Vision Transformer(ViT)를 사용했는데, 
transformer전의 patch와 position embedding을 결합하여 추가적인 layer normalization을 사용했고,
약간 다른 initialization 체계를 사용했다.

Text encoder로는 Transformer를 사용했다.
63M개의 paramter, 12개의 layer, 512의 각 layer의 폭, 8개의 attention head를 가지고 있고,
모든 문자를 소문자로 처리, PBE(byte pair encoding)을 사용하여 tokenization되고,
49,512의 어휘 크기를 가지며, max sequnce length는 76 token이다.
text의 feature representation들은 layer normalize된 후 multi-modal embedding space에 linear projection된다. 

기존의 computer vision model들은 width, depth를 ‘독립적’으로 증가시켜 모델 파라미터를 증가시켰지만, 
본 논문의 image encoder는 width, depth, resolution을 계산하여 한번에 증가시키는 방법이 개발되어 이를 사용하였다.
Text encoder는 width만 비례적으로 증가시켰다. 텍스트 인코더의 용량(depth)가 성능에 덜 민감하기 때문이다.

## 2.5 Training

> 모델 훈련 과정에 대해 설명합니다. 5개의 ResNets와 3개의 Vision Transformers를 사용하여 모델을 훈련시키고, 이를 통해 CLIP의 강력한 성능을 입증합니다.

ResNet → RN 50, 101, x4, x16, x64

Vision Transformer → ViT /32, /16, /64

들을 사용하여, 32 epoch씩 학습시켰다.

Optimizer는 Adam을 사용하는데, decoupled weight decay regularization 방법과 learning rate를  cosine schedule에 따라 감소시키는 방법을 활용하였다.

Hyperparameter들은  gird searches, random search의 합성, ResNet-50으로 1 epoch실행해봤을때의 결과로 manual tuning된 것으로 설정하였다.

Temperature $\tau$의 초기값으로 0.07

Mini-batch의 크기는 32,768

Mixed-precision (계산 일부를 더 낮은 정밀도로 계산) 하여 훈련을 가속화하고 메모리를 아꼈다.

이외에도 gradient checkpointing, half-precision Adam statistics, half-precision stochastically rounded text encoder weights로 메모리를 아꼈다.

Embedding similarity의 계산은 local embedding batch에 필요한 pairwise similarity의 하위 집합만 계산하는 개별 GPU에서도 공유되었다.

ViT/14 모델에 추가적인 336 pixel resolution의 pre-train (1 epoch)의 과정을 거쳐 ViT/14@336px 모델을 추가하였고, 이 모델이 가장 높은 성능을 보였다.

# 3. Experiments

## 3.1 Zero-Shot Transfer

### 3.1.1 Motivation

> 제로샷 학습의 필요성과 동기에 대해 설명합니다. 기존 모델과 비교하여 CLIP의 장점을 논의합니다.

### 3.1.2 Using CLIP For Zero-Shot Transfer

> CLIP을 제로샷 전이에 사용하는 방법을 설명합니다. 다양한 데이터셋에서의 성능 평가 결과를 제시합니다.

### 3.1.3 Initial Comparison to Visual N-Grams

> Visual N-Grams와의 초기 비교 결과를 제시합니다. CLIP의 우수한 성능을 강조합니다.

### 3.1.4 Prompt Engineering and Ensembling

> 프롬프트 엔지니어링과 앙상블 방법을 사용하여 CLIP의 성능을 최적화하는 과정을 설명합니다.

### 3.1.5 Analysis of Zero-Shot CLIP Performance

> 제로샷 CLIP 성능 분석 결과를 제시합니다. 다양한 작업에서의 성능을 평가합니다.

## 3.2 Representation Learning

> 표현 학습에 대한 실험 결과를 설명합니다. CLIP이 다양한 작업에서 강력한 표현을 학습할 수 있음을 보여줍니다.

## 3.3 Robustness to Natural Distribution Shift

> 자연스러운 분포 변화에 대한 견고성을 평가한 결과를 설명합니다. CLIP의 강력한 성능을 입증합니다.

# 4. Comparison to Human Performance

> 인간 성능과의 비교 결과를 설명합니다. CLIP이 인간과 유사한 수준의 성능을 발휘할 수 있음을 보여줍니다.

# 5. Data Overlap Analysis

> 데이터 중복 분석 결과를 설명합니다. CLIP이 훈련된 데이터와 테스트 데이터 간의 중복을 최소화하는 방법을 제시합니다.

# 6. Limitations

> CLIP의 한계점을 설명합니다. 모델의 성능을 개선하기 위해 추가 연구가 필요한 부분을 논의합니다.

# 7. Broader Impacts

## 7.1 Bias

> CLIP의 편향 문제를 논의합니다. 편향을 줄이기 위한 방법을 제안합니다.

## 7.2 Surveillance

> 감시와 관련된 윤리적 문제를 논의합니다. CLIP이 악용될 가능성에 대해 경고합니다.

## 7.3 Future Work

> 향후 연구 방향을 제시합니다. CLIP의 성능을 더욱 향상시키기 위한 방법을 논의합니다.

# 8. Related Work

> 기존 연구와의 관련성을 설명합니다. CLIP이 기존 연구들과 어떻게 차별화되는지 강조합니다.

# 9. Conclusion

> 연구의 결론을 요약합니다. CLIP의 성과와 향후 연구 방향을 제시합니다.
