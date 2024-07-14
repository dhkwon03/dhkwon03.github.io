---
title: 'What is U-Net?; U-Net: Convolutional Networks for biomedical Image Segmentation'
tags:
- paper_review
- concept
- 9S_July2024
categories:
- AI
---

원래 U-Net 은 CNN 계열의 image segmentation 에서 많이 사용되던 architecture 였다. 최근에 Diffusion 을 공부하다보니 U-Net 을 backbone network 로 하여 training 하는 것을 볼 수 있었는데 복습 겸 다시 정리하고자 U-Net 에 관하여 정리한다.

U-Net: Convolutional Networks for biomedical Image Segmentation 논문은 U-Net이라는 것을 처음으로 제안한 논문이다. 의료계에서 image segmentation 에서 활용하기 위해 만들어졌다. 세포들이 모여있는 현미경 사진이 있으면 세포 간에 분리해서 인식하는 AI 가 필요했나 보다. 지금은 굉장히 major 한 U-Net 이 처음 제안된 논문이 의료계와 관련된 논문이라니 신기할 따름이다.

## 1. Introduction

앞에서 소개한 것처럼 U-Net 은 의료 영상에서 semantic segmentation 을 하기 위한 fully convolutional network 으로써 고안되었다. ‘Fully Convolutional Networks for Semantic Segmentation’ 에 등장하는 architecture 를 변형하여 만들었다고 한다. 

“적은 training image” 로 “정확한 segmentation” 이 가능하도록 하는 것을 목적으로 만들었다. 

## 2. Network Architecture

### 2.1 Main Ideas and properties of U-Net

![Untitled](https://github.com/user-attachments/assets/d3a295b5-8edf-4855-9da6-aa89f67588fe)

위 그림은 U-Net 의 전체 architecture 를 보여준다. 가운데를 기준으로 왼쪽 (channel 수가 늘어나고 dimension 이 줄어든다) 이 ‘contracting path’, 오른쪽 (channel 수가 감소하고 dimension 증가) 이 expansive path 이다.

U-Net 의 첫번째 아이디어는 원활한 localization (이미지 상에서 object 의 정확한 위치를 찾아내는 것) 을 위해 contracting path 의 feature가 upsample 과정에서의 output에 결합이 된다는 것이다. convolution 을 거치면서 처음에 추출되었던 세부적인 feature 가 없어질 수 있는데 convolution 을 통해 feature 를 추출하는 contracting path 에서는 비교적 세부적인 (high-resolution) feature 가 살아있으므로 이 정보를 upsampling 과정에 넣어서 세부적인 feature 정보도 output 에 최대한 살리겠다는 의도인 것 같다.

두번째로 U-Net 에서는 upsampling 하는 부분에서도 초반에는 많은 feature channel 이 있는데 이는 dimension 이 높아지는 high-resolution layer 에도 contracting path 에서 추출한 context를 전달하고자 하는 의도이다. 그래서 contracting path 와 expansive path 가 거의 symmetric 한 U 자 형태를 보이는 것이다. 또한, U-Net 에는 fully connected layer 가 없으며 input image 가 network 를 통과하면 output 에서 input 보다 크기(dimension)가 조금 작은 image 가 바로 나오는 형태이다. 즉, “image에서 image 로 mapping 하는 network” 라는 것이 특징이다. 그래서 매우 큰 image 에 대해서도 tiling (큰 image 를 잘라서 작은 image 들로 나누는 것) 을 하고 각각 tile image 를 input 으로 넣어서 segmentation 을 하면 그 부분에 대한 segmentation 결과 image 가 바로 output 으로 나오기 때문에 output 들을 이어붙이기만 하면 된다. 매우 큰 image 도 segmentation 이 쉽게 가능하다는 의미이다.

세번째로 elastic deformance (유연하게 변형) 을 통해 학습 데이터를 변형하여 데이터 개수를 불려서 매우 적은 학습 데이터 개수의 한계를 극복했다고 한다.

마지막으로 같은 class 의 object 가 서로 경계를 맞대고 있을 때 이를 구분하는 것이 관건이었다고 한다. 이를 위해 object 사이에 간격이 좁은 경우 background 와 object 를 구분하는 것에 더 많은 가중치 (weight) 를 주는 weighted loss 를 사용했다.

### 2.2 U-Net Architecture

위에 있는 U-Net 구조 figure 를 참고.

contracting path는 기존 convolutional network와 원리가 같다. 2번의 3x3 convolution + ReLU + 2x2 max pooling (stride = 2) 가 반복되는 downsampling이다. 각 downsampling 마다 feature channel 개수가 2배로 증가한다.

expansive network는 2x2 up-convolution + contracting path에 있는 feature map을 upsampling 부분의 dimension (크기) 에 맞게 자른 후 가져와 (경계에 있는 pixel 을 잘라내고 정가운데를 잘라서 가져옴) expansive network의 map에 결합 (feature 부분이 늘어나도록 결합, 즉 feature channel 개수가 2배 됨, dimension 은 그대로) + 2번의 3x3 convolution + ReLU 로 구성된다. convolution 을 지나며 border pixel의 정보를 잃기 때문에 contracting path 에 있는 feature map 을 잘라서 가져올 때 경계 pixel 들을 잘라내는 것이라고 한다.

final layer에서 1x1 convolution은 64개 feature vector를 각 class로 mapping 해주는 역할이다. 즉 각각의 pixel 마다 class 를 나타내도록 해준다.

### 2.3 Up-Convolution

Up-convolution, Upsampling 이라는 용어가 나오는데 이들은 무슨 뜻인가?

Upsampling 은 쉽게 말해서 저해상도의 data 를 고해상도로 만드는 것을 말한다. dimension 을 크게 하는 것이라 볼 수 있겠다. Upsampling 에는 Nearest neighbor interpolation, Bi-linear interpolation, Bi-cubic interpolation 등의 방식이 있다. 다 interpolation 인 것을 보아 저해상도의 pixel 값들을 기반으로 linear 또는 non-linear 하게 계산을 해서 고해상도로 해상도를 올렸을 때 비어있는 pixel을 채우겠다는 것이다.

근데 이렇게 저해상도에서 고해상도로 만드는 filter 값을 학습해보자는 것이 Up-convolution 이다.  Transposed Convolution 이라고도 한다.

Up-convolution 은 쉽게 말해 convolution의 역과정이다. convolution을 거치면서 downsampling 된 output을 입력으로 해서 convolution 되기 전의 input을 유추하는 것이다.

일반적인 convolution 연산은 아래와 같이 이루어진다. kernel 을 하나하나 옮기면서 input 에 곱하고 output 을 계산하면 비효율적이므로 kernel 을 바탕으로 Toeplitz Matrix 를 만들어서 matrix multiplication 으로 한번에 convolution 연산을 한다. 또한, 일반적인 convolution에서 우리는 kernel에 있는 weight 값들을 학습을 통해 얻어낸다.

![Untitled](https://github.com/user-attachments/assets/f0424e3b-990d-42a6-a04a-7ed20e184d8e)

이 때 Sparse matrix C (Toeplitz Matrix) 를 transpose 해서 output과 multiplication 하면 input이 나온다. 그래서 up-convolution 을 transposed convolution 이라고도 부르는 것이다.

![Untitled](https://github.com/user-attachments/assets/e9d102b7-a882-4ea8-a9cc-70e7994fc2ad)

여기서 sparse matrix $C^T$ 의 weight 들을 학습하는 것이 Up-convolution 이다. convolution 이랑 원리는 거의 비슷하다. input 보다 output 이 더 dimension 이 크다는 것이 다를 뿐이다. 위 그림을 보면 input 주변에 padding 을 하고 kernel 을 옮겨가며 convolution 연산을 하면 dimension 이 더 큰 output (초록색)이 되는 것을 볼 수 있다.

## 3. Training

Caffe framework 를 사용해서 SGD (stochastic gradient descent) 로 training 했다. overhead 를 최소화하고 GPU 메모리를 최대한 활용하기 위해 batch size 를 줄이고 input size 를 크게 했다. 결과적으로 batch 당 하나의 image 가 되도록 했다.

SGD 에서 0.99 의 높은 모멘텀 값을 사용했는데 현재 optimization step 에서 과거의 training sample 을 더 많이 고려하도록 한 것이다. 

### 3.1 Loss function

training 의 loss function 으로 아래와 같은 cross entropy loss function 이 사용되었다. 

![Untitled](https://github.com/user-attachments/assets/ea0138b5-1294-4f8a-b4ab-993ea80a46f0)

위 식에서 $\Omega$ 는 image 상 2차원 좌표 전체의 집합 (각 pixel 전체의 집합이라 생각하면 됨)이다. w(x) 는 각 pixel 마다 loss function 의 가중치를 설정한 함수인데 이에 대해서는 뒤에 설명하겠다.

$l(\mathbf x)$ 는 해당 pixel 에서의 ground truth class (해당 pixel 이 실제 class, 정답을 의미) 이다. class k 에서 $p\_k(\mathbf x)$ 는 아래와 같다. 

![Untitled](https://github.com/user-attachments/assets/06803a78-0512-436a-9384-1e0c132cd5bd)

$a\_k(\mathbf x)$ 는 feature channel k 에서의 activation 이다. 즉 network 가 최종적으로 내놓은, pixel x 가 k 일 가능성을 나타내는 값이다. 이를 soft-max 로 처리하여 “x 가 class k 일 확률”을 나타낸 것이 $p\_k(\mathbf x)$ 이다. 따라서 $p\_k(\mathbf x)$ 는 0과 1 사이의 값이며 $p\_{l(k)}(\mathbf x)$ 이 1에 가까울 수록 예측과 ground truth (정답) 가 같다는 것이고 곧 network 가 예측을 잘한다는 것이다. loss function 을 보면  $p\_{l(k)}(\mathbf x)$ 가 1일 때 log 값이 0 이고 1보다 작아질수록 log 값이 음수로 작아진다는 것을 볼 수 있다.

### 3.2 Weight map

w(x) 를 논문에서 weight map 이라 하고 있다. object 가 경계를 맞대고 있는 경우 background 인 pixel 과 object 인 pixel 을 구분하는 것에 대해 더 가중치를 둔다고 했었다. (chapter 2.1 참고) 이를 구현한 것이 weight map 이다. w(x) 는 아래와 같다.

![Untitled](https://github.com/user-attachments/assets/3ecbd869-b088-4a17-addb-62db16acd2bb)

$w\_c(\mathbf x)$ 는 class frequency 균형을 맞추기 위한 weight map 이다. (뭔지는 잘 모르겠으나 필자 뇌피셜로는 주변에 같은 class 가 있으면 더 커지는 weight map 이 아닐까 추측된다) $w\_0$ 는 hyperparameter 이며 논문에서는 10으로 설정했다. $\sigma$ 또한 hyperparameter 이며 5 정도로 설정했다. $d\_1$ 은 가장 가까운 object 의 경계까지의 거리, $d\_2$ 는 두번째로 가까운 object 의 경계까지의 거리이다. 가까운 object 까지의 거리가 짧을수록 weight 가 더 커지는 것을 알 수 있다. 논문에서 이를 figure 로 나타냈는데 아래 그림의 (d) 를 보면 경계 사이가 좁을수록 weight 가 큰 것을 볼 수 있다.

![Untitled](https://github.com/user-attachments/assets/83324194-5a2b-4336-8222-fcc7dac1a73c)

### 3.3 Weight Initialization

원활한 학습을 위해 각 feature map의 initial weight 들의 variance 가 1이 되도록 해야 한다고 한다. 이를 위해 initial weight 를 표준 편차가 $\sqrt{2/N}$ 인 Gaussian distribution 에서 sampling 해서 설정했다. N 은 하나의 neuron (= layer 의 한 node) 에 연결되어 “들어오는 (incoming)” node 개수이다. 

각 node 가 layer 2개를 연결하므로 1/2 씩 개수를 나누어가진다고 생각할 때 한 feature mapping 에 있는 모든 neuron의 weight 를 모두 합하면 표준 편차가 1 이 되어 논문에서 말했던 variance 1 이 된다고 하는 것 같다. 

### 3.4 Data Augmentation

기존에 있던 dataset에 변형을 가하여 새로운 data 들을 만들어냄으로써 적은 dataset의 한계를 극복하겠다는 것이다. 본 논문에서는 3 by 3 에서의 displacement (변위) 벡터를 Gaussian distribution 으로부터 생성하여 bi-cubic interpolation (근사법 중 하나) 으로 각 pixel의 displacement 를 계산하여 적용했다고 한다. 

contracting path 의 끝에 dropout layer 가 있는데 이 또한 data augmentation 에 기여한다고 설명하고 있다. dropout layer 는 fully connected layer 에서 몇 개의 node 를 확률적으로 없애버리는 것으로, 같은 data에 대해서도 layer 의 weight 를 달라지게 하기 때문에 간접적인 data augmentation 효과가 있다고 알려져 있다. (참고; Dropout layer 넣는 건 특히 dataset 개수가 적을 때 overfitting 을 방지하는 대표적인 방법 중 하나이다. 대신 regularization 을 사용하는 경우도 많다.)

## 4. Experiments

ISBI 라는 곳에서 주최하는 EM segmentation challenge (의학 영상에서 세포 구분하는 computer vision challenge 인 것 같다) 에서 좋은 성능을 냈다고 하며 논문에서 보여주는 warping error rate, rand error rate, pixel error rate 등을 보면 상대적으로 U-Net 이 우수한 것을 볼수 있으며 IoU (intersection over union) 수치 상으로도 우수한 것을 볼 수 있다. 의학계 논문이다보니 어려운 의학 용어가 논문에 많이 나오는데 필자는 그런거에는 관심 없고 딥러닝에만 관심있기 때문에 그냥 그 당시에 성능 좋았구나~ 하고 넘어가자.

## 5. Conclusion

Diffusion 에서 쓰인 U-Net architecture 에 대해 자세히 알아보고 논문에서 image segmentation 을 위해 U-Net architecture 를 training 하기 위해 사용한 soft-max 와 cross entropy loss function, weight initialization, data augmentation 기법 등을 살펴보았다.

### Comment

본 post 는 Diffusion 에서 사용된 U-Net 에 관한 복습 차원에서 작성되었음

This post is the part of ‘Uploading 5 academic blog posts’ which is July’s resolution project in 9th squad.

본 post 는 9th Squad 의 7월 목표에서 필자의 목표인 ‘학술 블로그 포스트 5개 업로드’ 의 일환임을 알림

### Reference

- [2) Up-sampling - 한땀한땀 딥러닝 컴퓨터 비전 백과사전 (wikidocs.net)](https://wikidocs.net/149326)
