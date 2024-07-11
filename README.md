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

CLIP은 image와 text 조각이 데이터셋 안에서 쌍으로 존재하는지에 대해 예측하도록 pre-train되어 있다.

각각의 데이터셋에서, 모든 class를 사용하여 잠재적인 text 쌍을 찾고, CLIP을 통해 가장 적절한 쌍이 무엇인지 예측한다.

1. 각각의 encoder로부터 image와 possible text들의 집합을 embedding한다.
2. 이 embedding들의 cosine similarity를 측정하고, $\tau$로 scaling, softmax로부터의 확률분포로 normalize된다.
    (이 예측 layer는 L2-normalized input, L2-normalized weights, no bias, temperature scaling이 된 multinomial logistic regression classifier 이다.)
    이렇게 해석하면 image encoder는 이미지에 대한 특징 표현을 계산하는 컴퓨터 비전 backbone이고,    
    text encoder는 클래스가 나타내는 시각적 개념을 지정하는 텍스트를 기반으로 선형 분류기의 가중치를 생성하는 hyper network 라고 볼 수 있다.

CLIP pre-training의 모든 step은 class당 1개의 예제를 포함하고 자연어 설명을 통해 정의된 32,768개의 총 class가 있는 컴퓨터 비전 dataset에 무작위로 생성된 proxy의 성능을 최적화하는 것으로 볼 수 있다.

### 3.1.3 Initial Comparison to Visual N-Grams

> Visual N-Grams와의 초기 비교 결과를 제시합니다. CLIP의 우수한 성능을 강조합니다.

CLIP은 Visual N-Grams에 비해 상당히 좋은 성능을 가지고 있다.

하지만, Visual N-Grams와의 비교는 CLIP의 성능을 맥락화하기 위한 것으로, 두 시스템 간의 직접적인 방법 비교로 해석해서는 안 된다.

- 본 연구에서는 10배 더 큰 데이터셋을 훈련에 사용하고,
- 예측당 거의 100배 더 많은 컴퓨팅을 요구하는 비전 모델을 사용하며,
- Visual N-Grams가 출판될 당시 존재하지 않았던 Transformer 기반 모델을 사용하는

등 두 시스템 간의 많은 성능 관련 차이점들이 통제되지 않았기 때문이다.

### 3.1.4 Prompt Engineering and Ensembling

> 프롬프트 엔지니어링과 앙상블 방법을 사용하여 CLIP의 성능을 최적화하는 과정을 설명합니다.

Issue 1: 단어의 다의성

Class의 이름이 CLIP의 encoder에 전해지는 유일한 정보일 때, 문맥 부족으로 어떤 의미의 단어가 사용되는지 구분이 불가능하다. 여러 의미를 가진, 같은 단어가 다른 class로 분류되는 문제가 발생된다.
ex) 날고 있는 두루미와, 두루미 2개의 image가 서로 다른 class로 분류된다.
ex) 복서 image가 복서가 아닌, 운동선수라고 분류된다.

Issue 2: text는 full sentence 인데 반해, image는 single word이다.
text는 image를 묘사하는 어떤 문장이 될 것이고, image는 해당 class를 표현하는 어떤 한 단어일 가능성이 높다. 이를 해결하기 위해 “A photo of a {label}.”과 같은 prompt 템플릿을 사용했고, 이것이 실제로 성능을 향상시켰다.

이 관점에서, 우리는 task에 맞게 prompt를 조정하는 prompt engineering을 통해 zero-shot 성능 또한 발전시킬 수 있음을 확인했다.
ex) Specify the category: “A photo of a {label}, a type of pet.”
ex) Put quotes around the text or number: “a satellite photo of a {label}.”

또한, 여러 zero-shot classifier를 ensemble 하여 성능을 발전시킬 수 있음을 경험했다.
“A photo of a big {label}.” and “A photo of a small {label}.”
Ensemble을 embedding space가 아닌, probability space 위에서 구성하였다.

종합하면, prompt engineering과 ensembling을 사용하여 3.5% 이상의 성능 향상을 이뤘다.

### 3.1.5 Analysis of Zero-Shot CLIP Performance

> 제로샷 CLIP의 특성과 성능 분석 결과를 제시합니다. 다양한 작업에서의 성능을 평가합니다.

Zero-shot CLIP vs ResNet-50

    Fully supervised, regularize 된 logistic regression classifier를 사용한 ResNet-50과 zero-shot CLIP을 비교했다.
    CLIP의 성능이 높은 데이터셋도 있고, ResNet의 성능이 높은 데이터셋도 있었다. 이 차이를 WIT와 ImageNet 간의 task별 다양한 감독량 차이에 있다고 보았다.
    CLIP과 ResNet의 성능이 비슷한 datasets:  “일반적인” object classification datasets
    ex) ImageNet, CIFAR, STL, PascalVOC
    CLIP이 ResNet에 비해 성능이 뛰어난 datasets: video에서 액션 인식을 측정하는 datasets
    ImageNet의 명사 중심 객체 감독에 비해 자연어가 동사와 관련된 시각적 개념에 대한 더 넓은 감독을 제공하는 것이라고 추측했다.
    CLIP이 ResNet에 비해 성능이 떨어지는 datasets: specialized, complex, abstract datasets
    ex) 위성 이미지 (EuroSAT, RESISC45), 림프 노드 종양 탐지 (PatchCamelyon), …
    CLIP이 수행한 classification에 대해서, 
    비전문가인 사람은 갯수를 세거나, 위성 이미지 분류, 교통 표지판 인식과 같은 작업들을 수행할 수 있으므로, 개선의 여지가 있지만, 
    림프절 종양 분류와 같이 학습자가 이전에 경험한 적이 없는 어려운 작업에 대해 제로샷 전송을 측정, 평가하는 것은 의미있는 평가가 아닐 수 있다.

Zero-shot CLIP vs Few-shot CLIP

    Zero-shot의 한계 때문에, zero-shot CLIP과 ResNet을 비교하는 것 보다, few-shot을 비교하는 것이 더 직접적인 비교가 될 것이다.
    Zero-shot CLIP이 의외로 4-shot CLIP과 성능이 비슷했다. 이는, "정상적인" 지도 학습은 훈련 예제에서 간접적으로 개념을 추론해야 하는데, 문맥이 없는 예제 기반 학습은 (특히 원샷의 경우) 데이터와 일치할 수 있는 다양한 다른 가설이 있다는 단점이 있다. 또한, 하나의 image에 다른 다양한 개념이 담긴 경우, image의 주요 대상이 될 것이라고 가정할 순 있지만, 보장할 순 없다.
    잠재적 해결책은 zero-shot classifier를 few-shot classifier의 weight에 대한 선행으로 사용하는 것이다.
    생성된 weight에 L2 penalty만 추가하면 이를 구현할 수 있지만, 하이퍼파라미터 최적화 과정에서 regularizer의 큰 값만 선택될 수 있고, 이는 few-shot이 zero-shot이 되게 만든다. Zero-shot CLIP과 few-shot CLIP을 유연하게 결합하는 것은 향후 연구에서 다루어져야 할 것이다.

Zero-shot CLIP vs Few shot logistic regression (BiT-M, SimCLRv2)

    Zero-shot CLIP과 다른 few shot 모델을 비교했을 때에는, 16-shot과 비슷한 성능을 보였다.
    독립적인 데이터셋에서 각각 비교해 보면, zero-shot CLIP이 성능이 뛰어나다는 것을 확인할 수 있다. 표는 해당 모델이 각각의 데이터셋에 대해서 몇 번의 shot을 가져야 CLIP과 비슷해질지 나태나는 지표이다.
    다른 평가 dataset이 커서 훈련이 잘 이루어진다면, zero-shot CLIP 또한 linear classifier이기 때문에, 다른 fully-supervised linear classifier가 zero-shot CLIP의 상한이 될 것이다.
    다름 그래프에서 zero-shot CLIP이 fully-supervised classifier보다 10~20%정도 낮은 성능을 보임을 알 수 있다.
    r = 0.82는 CLIP이 underlying representation과 task 학습을 zero-shot 연결하는 데 비교적 일관성이 있음을 의미한다.
    또한, zero-shot CLIP이 STL10, CIFAR10, Food101, OxfordPets, Caltech101 5가지 dataset에서는 fully-supervised classifier과 비슷한 성능을 내는데, 이는 underlying representation의 퀄리티가 높아서 그랬을 것이라고 추측된다.
    또한, fully-supervised classifier가 1% 성능이 향상될 때 마다, zero-shot CLIP은 1.28% 성능이 향상된다. 

지난 연구들에 따르면, 딥러닝 모델의 성능은 training compute와 dataset size로 예측이 가능하다는 결과가 있었다. 이에 따라 GPT 모델들의 zero-shot  훈련의 계산량(비용)의 성능이 1000배 증가했다. 이를 바탕으로 CLIP을 평가해보았을 때, CLIP은 44배 증가했다.

## 3.2 Representation Learning

> 표현 학습에 대한 실험 결과를 설명합니다. CLIP이 다양한 작업에서 강력한 표현을 학습할 수 있음을 보여줍니다.

3.1에서 CLIP의 task-learning에 관한 능력을 다뤘다면, 이번 장에서는 CLIP의 representation 능력을 다룬다.

이상적인 표현 뿐 만 아니라, 표현의 품질 또한 평가대상이 된다.

1. 모델로부터 추출한 representation에 linear classifier를 맞추고 다른 dataset들에서 성능을 평가(일반적)
2. 모델의 end-to-end fine-tuning된 모델을 평가

2번째 평가방법이 flexibility를 높이고, fine-tuning이 대부분의 dataset에서 linear classifier의 성능을 능가한다.

실용적으로는 fine-tuning을 사용하는 것이 맞지만, 본 연구에서는 몇가지 이유 때문에 linear classifier를 사용할 예정이다.

우리는 high-performing task, dataset에 구애받지 않는 pre-training 접근방식을 연구하는 데 중점을 가지고 있는데, fine-tuning은 각 dataset에 대한 표현을 조정하기 때문에 pre-training 단계에서 일반적이고 강건한 representation을 학습하기 위해 실패를 보상하고 잠재적으로 마스킹할 수 있다. 이에 반해 linear classifier는, flexibility가 제한적이기 때문에, 이러한 실패를 강조하고 명확한 피드백을 제공할 수 있다는 장점이 있어, 이를 선택한다.

CLIP에서, supervised linear classifier를 훈련하는것은 3.1에서 분석했던 zero-shot classifier과 유사하다는 장점을 가지고 있다.

CLIP의 포괄적인 성능을 평가하기 위해 27개의 서로 다른 데이터 세트에서 66개의 서로 다른 모델을 연구하려면 1782개의 서로 다른 평가를 조정해야 한다.

Fine-tuning은 다양한 기법 세트를 공정하게 평가하는 것이 계산 비용이 많이 든다. 이에 비해 선형 분류기는 최소한의 하이퍼 파라미터 튜닝이 필요하고 표준화된 구현 및 평가 절차가 있다.

Figure 10으로 인해 알 수 있는 사실

- 작은 CLIP 모델 성능: ResNet-50 및 ResNet-101과 같은 작은 CLIP 모델은 ImageNet-1K에서 훈련된 다른 ResNet보다 성능이 뛰어나지만, ImageNet-21K에서 훈련된 ResNet (BiT-M)보다는 성능이 낮다.
- EfficientNet과의 비교: 작은 CLIP 모델은 유사한 컴퓨팅 요구 사항을 가진 EfficientNet 계열의 모델보다 성능이 낮다.
- CLIP 모델의 스케일링: CLIP로 훈련된 모델은 매우 잘 확장되며, 가장 큰 모델인 ResNet-50x64는 최고의 기존 모델인 Noisy Student EfficientNet-L2보다 전체 점수와 컴퓨팅 효율성에서 약간 더 우수한 성능을 가진다.
- ViT의 효율성: CLIP ViT는 CLIP ResNet보다 약 3배 더 컴퓨팅 효율적이며, 이는 주어진 컴퓨팅 예산 내에서 더 높은 성능을 달성할 수 있게 한다.
- ViT의 효율성 재현: 충분히 큰 데이터셋에서 훈련된 경우 ViT가 CNN보다 더 효율적이라는 것을 확인하였습니다.
- 최고 성능 모델: 336 픽셀 해상도로 1 에폭 추가 훈련된 ViT-L/14 모델이 12개 데이터셋 평가 세트에서 기존 최고의 모델을 평균 2.6%의 성능 향상으로 능가합니다.

CLIP 모델은 무작위 초기화로부터 끝까지 훈련된 단일 컴퓨터 비전 모델에서 이전에 시연된 것보다 더 넓은 작업 세트 (지리적 위치 확인(geo-localization), 광학 문자 인식(optical character recognition), 얼굴 감정 인식(facial emotion recognition), 행동 인식(action recognition) 등) 도 학습이 가능하다.

하지만 기존의 12개의 dataset 평가에는 이러한 부분이 없으므로, 새로운 27개 데이터셋 평가 세트에서 성능을 측정했고, Figure 11에 이에 대한 내용을 담고 있다.

넓은 평가 범위에서, CLIP은 다음과 같은 장점을 가지고 있다.

- 컴퓨팅 효율성: 모든 CLIP 모델은 규모에 상관없이 컴퓨팅 효율성 측면에서 모든 평가된 시스템을 능가한다.
- 평균 점수 향상: 최고의 모델이 이전 시스템보다 평균 점수에서 2.6%에서 5% 정도 향상됩니다.
- 자가 지도 학습 시스템 성능: self-supervised learning 시스템이 더 넓은 평가 세트에서 눈에 띄게 더 좋은 성능을 보입니다.
- SimCLRv2 성능: SimCLRv2는 12개 데이터셋에서 BiT-M보다 평균적으로 성능이 떨어지지만, 27개 데이터셋 평가 세트에서는 BiT-M을 능가합니다.

시스템의 "일반적인" 성능을 더 잘 이해하기 위해 작업의 다양성과 커버리지를 계속 확장하는 것이 중요할 것이다.

조금 더 좁은 평가 범위에서 (Noisy Student EfficientNet-L2와 비교)

- CLIP 모델은 27개의 데이터셋 중 21개에서 Noisy Student EfficientNet-L2를 능가한다.
- 주요 성능 향상 분야:
    - Optical Character Recognition(OCR) 작업 (SST2, HatefulMemes ..)
    - Geo-localization, scene recognition (Country211, SUN397)
    - Activity recognition (Kinetics700, UCF101 ..)
    - Car and traffic sign recognition (Stanford Cars, GTSRB ..)
- 세밀한 인식 작업 향상: GTSRB에서 14.7%의 성능 향상은 ImageNet-1K의 지나치게 좁은 감독의 문제를 반영된 결과일 것이라고 예측된다. 단일 레이블로 인해 세부 사항을 놓치는 문제가 있을 수 있다.
- CLIP은 여전히 여러 데이터셋에서 EfficientNet보다 성능이 낮다.
- EfficientNet은 CLIP에 비해 ImageNet에서 가장 잘 작동하며, 이는 EfficientNet이 ImageNet에서 훈련되었기 때문이다.
- EfficientNet은 CIFAR10 및 CIFAR100과 같은 저해상도 데이터셋에서도 약간 더 나은 성능을 보이는데, 이는 CLIP에서 스케일 기반 data argumentation이 부족하기 때문일 수 있다.
- PatchCamelyon 및 CLEVRCounts와 같은 데이터셋에서는 두 접근 방식 모두 성능이 낮지만, EfficientNet이 약간 더 나은 성능을 보인다.

## 3.3 Robustness to Natural Distribution Shift

> 자연스러운 분포 변화에 대한 견고성을 평가한 결과를 설명합니다. CLIP의 강력한 성능을 입증합니다.

2015년에는 ImageNet의 성능이 인간을 능가한다는 평가 통계가 있었다. 새로운 벤치마크가 생긴 뒤에는 ImageNet과 인간의 정확도가 많이 줄어들었다. 왜 이런 차이가 발생했을까?

딥러닝 모델은 dataset에서 상관관계와 패턴을 찾는 것을 목표로 한다. 하지만, 많은 상관관계와 패턴들이 사실은 거짓된 것이었고, 다른 분포에서는 적용되지 않는다는 것을 발견했다.

ImageNet에 대한 7가지 natural distribution shift를 설정하고, 이에 대한 평가를 진행한다.

Robustness를 분석하는 2가지 기준이 있다.

- Effective robustness: 원래의 훈련 데이터와 분포가 다른 새로운 데이터, 환경에서 얼마나 잘 작동하는지, in-distribution과 out-of-distribution의 정확도의 개선을 포착
- Relative robustness: 특정 기준 모델이나 다른 모델들과 비교하여 상대적으로 얼마나 강건한지, out-of-distribution 정확도의 모든 개선을 포착
    - In-distribution: 모델이 훈련된 데이터와 동일하거나 매우 유사한 분포를 가진 데이터
    - out-of-distribution: 모델이 훈련된 데이터와 다른 분포를 가진 데이터

ImageNet dataset distribution에 대한 training 또는 adaptation이 관찰된 robustness의 차이 때문인가?

직관적으로, zero-shot 모델은 특정 분포에 대해 학습하지 않았기 때문에, 해당 분포의 가짜 상관관계, 패턴을 파악하지 못해야 한다.

하지만, 그렇기 때문에 zero-shot모델에서 높은 effective robustness를 기대할 수 있다.

그래서 본 연구에서는 7가지의 natural distribution shift된 ImageNet model과 zero-shot CLIP 모델의 성능을 비교했다.

Zero-shot CLIP이 더 견고함을 확인할 수 있었지만, 이는 supervised learning이 robustness 차이를 발생시킨다는 의미는 아니다. CLIP의 크기가 다양한 pre-training dataset과 natural language supervision 또한 그 차이를 발생시켰을 것이다.

이를 알아보기 위해, CLIP을 ImageNet training dataset에 맞는 L2 regularized logistic regression classifier를 통해 ImageNet의 distribution에 적응한 후, zero-shot CLIP과 비교해 보았다.

이는 ImageNet 정확도를 전반적으로 9.2에서 85.4% 올렸고, 2018의 SOTA와 맞먹는 성능을 보였다.

이는 놀라운 발전이지만, distribution shift에 대한 평균 performance 향상으로 이어지지는 않는다. 

또한 Figure 14에서 dataset당 zero-shot 정확도와 linear classifier의 정확도의 차이를 분석한 결과, 하나의 데이터 세트인 ImageNetV2에서 성능이 여전히 크게 향상되는 것으로 나타났습니다.

ImageNetV2는 기존 ImageNet dataset의 distribution과 가장 유사하며, supervised adaptation으로 인한 정확도 향상이 ImageNet의 분포에 밀접하게 집중되어 있음을 의미한다. ImageNet-R, ObjectNet, ImageNet Sketch, ImageNet-A 도 이와 유사하게 ImageNet과의 분포차이가 4%내이다.

Youtube-BB와 ImageNet Vid의 정확도 변화는 매우 중요하다.

이에 따라 다음과 같은 질문을 던질 수 있다.

- Distribution shift 되어서 정확도가 거의 또는 전혀 증가하지 않으면서 ImageNet dataset에서 정확도를 9.2% 향상시킬 수 있는 방법은 무엇인가
- 주로 "가짜 상관 관계 이용"에서 얻을 수 있는 이점은 무엇인가
- 이 동작은 CLIP, ImageNet dataset 및 연구된 distribution shift의 일부 조합에 고유합니까, 아니면 보다 일반적인 현상인가
- linear classifier뿐만 아니라 end-to-end fine-tuning에도 적용되는가

현재로서는 이러한 질문에 대한 확실한 답변을 가지고 있지 않다. 

또한 flexible한 zero-shot natural language 기반 image classifier를 통해 가능한 또다른 robustness의 개입을 조사한다.

Youtube-BB와 ImageNet-Vid는 ImageNet의 super-class(상위 클래스)로 구성되어 있다.

ImageNet 모델의 고정된 1000개 클래스 분류기를 사용하여 예측하려고 할 때 문제가 발생하는데, 

과거 연구에서는 ImageNet 클래스 계층에 따라 모든 하위 클래스에 대한 예측을 최대 풀링(max-pooling)하여 이 문제를 해결했다. 그러나 이 맵핑은 완벽하지 않다. 예를 들어, Youtube-BB의 person 클래스는 baseball player, bridegroom, scuba diver 등의 ImageNet class들로부터 예측을 풀링해야 한다.

여기서, CLIP을 사용하면 각 데이터셋의 클래스 이름을 직접 기반으로 맞춤형 zero-shot classifier를 생성할 수 있는데, 이는 고정된 ImageNet 클래스 맵핑의 문제를 해결하는 데 도움이 됩니다.

이는 ImageNet에서 평균적으로 effective robustness가 5%정도 증가하지만, 몇몇 데이터에서만 큰 개선이 이루어졌다. 

ObjectNet에서도 정확도가 2.3% 향상되었다. ObjectNet의 클래스가 ImageNet 클래스와 밀접하게 겹치도록 설계되었지만, 여전히 ObjectNet 클래스 이름을 사용하는 것이 ImageNet 클래스 이름을 사용하고 필요할 때 예측을 풀링하는 것보다 조금 더 도움이 될 것이다.

이렇게 zero-shot 설정에서 CLIP 모델은 강인성이 향상됨을 보았고, 이는 CLIP이 기존 모델보다 더 유연하게 다양한 dataset에 대응할 수 있음을 의미한다.

하지만, fully-supervised learning 설정에서는 이러한 이점이 거의 사라진다. Figure 14에서는 이를 확인할 수 있다. 

Adapt to class shift가 위 문단에서 언급한, 고정된 1000개의 class classifier를 사용하는 것 대신 CLIP을 사용하여 맞춤형 zero-shot classifier를 생성하는 것이고,

Adapt to ImageNet이 fully-supervised learning에 해당하는 것으로, CLIP의 이점이 거의 사라지는 모습을 볼 수 있다.

위의 실험들을 요약하자면, 높은 효과적 강인성은 모델이 접근할 수 있는 특정 distribution (training data)의 양을 최소화하는 것에서 비롯됩니다. 그러나 이는 데이터셋 특정 성능을 감소시키는 비용이 따릅니다.

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
