---
title: What is VAE (Variational Autoencoder)?
tags:
- concept
- 9S_July2024
categories:
- AI
---

이 글은 “Understanding Deep Learning” (UDL book) 이라는 책의 설명 방식과 순서를 따랐으며 이 책을 참고하고 정리하여 작성되었다. 본 post 에서 설명하는 일부 개념이 Diffusion 에서도 쓰이는 개념임을 알린다.

Variational Autoencoder, 줄여서 VAE 는 probabilistic generative model 의 일종이다. dataset 이 있을 때 “data의 확률 분포 $Pr(\mathbf x)$” 를 학습하는 것을 목적으로 한다. 혼동하면 안되는 것이 VAE 자체가 $Pr(\mathbf x)$ 를 모델링하는 것이 아니다. VAE 는 $Pr(\mathbf x)$ 의 model 을 학습하는데 “도움을 주기 위한” neural architecture 이다. VAE 는 data 의 분포의 modeling 을 원래의 data space 가 아니라 latent space 에서 하도록 해주는 것이다. 따라서 VAE를 사용할 때 $Pr(\mathbf x)$ 의 model 은 “nonlinear latent variable model” 이 된다.

## 1. What are “latent variable models”?

multi-dimensional variable $\mathbf x$ 에 대한 확률 분포 $Pr(\mathbf x)$ 를 “간접적”으로 표현하기 위해 고안된 것이 “Latent variable model” 이다. (아마 $Pr(\mathbf x)$를 그대로 표현하는 것이 불가능할 때 이렇게 간접적으로 표현하는 방식을 쓰는 것이겠죠?) 그러한 간접적 표현을 위해 latent variable $\mathbf z$ 를 도입한다. data $\mathbf x$ 와 latent variable (숨겨진, 잠재적 값) $\mathbf z$ 에 대해 joint distribution $Pr(\mathbf x, \mathbf z)$ 를 모델링한다. $Pr(\mathbf x, \mathbf z)$ 를 이용하면 $Pr(\mathbf x)$ 를 아래와 같이 marginalization 을 통해 나타낼 수 있다.

![Untitled](https://github.com/user-attachments/assets/89e03a6e-54ab-4219-84be-757e5e9b2ab0)

conditional probability 를 활용하면 아래와 같이 prior $Pr(\mathbf z)$ 와 latent variables term $Pr(\mathbf x \mid \mathbf z)$ 로 나타낼 수 있다.

![Untitled 1](https://github.com/user-attachments/assets/81bd46b5-efcc-4e41-bede-6bddbbd72256)

$Pr(\mathbf z)$, $Pr(\mathbf x \mid \mathbf z)$ 는 보통 $Pr(\mathbf x)$ 보다 더 간단하게 표현되기 때문에 이러한 방식은 $Pr(\mathbf x)$ 를 표현하기에 좋은 방법이라 할 수 있다.

UDL book 에 좋은 예시가 있다.

![Untitled 2](https://github.com/user-attachments/assets/3f1820a4-0c05-431b-a361-3f59f771c916)

위의 그림에서 a) 를 보면 여러 Gaussian distribution 이 섞여 있는 것을 볼 수 있다. 청록색으로 그려진 분포는 표현하기 어려운 복잡한 distribution 이지만 여러 Gaussian distribution의 합으로 표현한 것이다. 또한, 위의 예시에서 latent variable $z$ 는 discrete value 이며 $Pr(z)$ 는 각 $z$ 값마다 확률이 $\lambda\_n$ 으로 동일한 categorical distribution 이다. 또한 우리는 likelihood $Pr(x \mid z=n)$ 을 아래와 같이 정의할 수 있다.

![Untitled 3](https://github.com/user-attachments/assets/b1254449-438f-4999-917d-2ab54916d2ac)

위에서 설명한 것과 같이 marginalization 을 하면,

![Untitled 4](https://github.com/user-attachments/assets/ae18913d-0ed8-4c9e-b9a9-6054c15cd6e8)

위의 식을 보면 likelihood (예시에서 Gaussian distribution) 와 prior (예시에서는 일정한 확률 $\lambda\_n$) 만으로 복잡한 multi-modal probability distribution 인 $Pr(x)$  를 표현한 것을 볼 수 있다.

생각해둘 것은, 위의 예시는 linear latent variable model 의 경우이다.

## 2. Nonlinear latent variable model

nonlinear latent variable model 에서는 $\mathbf x, \mathbf z$ 모두 연속적이고 multivariate (변수가 여러 개이면서 종속변수가 2개 이상, 참고; 독립변수로만 이루어진 변수가 여러개이면 univariate) 이다. 

prior $Pr(\mathbf z)$ 는 아래와 같이 standard multivariate normal 로 정의한다.

![Untitled 5](https://github.com/user-attachments/assets/ad70553c-6511-40dd-8ca9-b794738f7d10)

likelihood $Pr(\mathbf x \mid \mathbf z, \boldsymbol \phi)$ 또한 normal distribution 이며 mean 은 latent variable $\mathbf z$ 에 대한 nonlinear function 이고 covariance 는 spherical (구) 하다. 아래와 같이 표현된다.

![Untitled 6](https://github.com/user-attachments/assets/82600971-adbb-4021-96f1-799ff02c928a)

parameter $\boldsymbol \phi$ 는 deep network 의 parameter 이다. 위의 distribution 의 변수가 아니라 nonlinear function 을 모델링하기 위한 neural network 의 parameter 를 표현한 것이라 생각하면 된다.

latent variable $\mathbf z$ 는 $\mathbf x$ 보다 low dimension (저차원) 이다. $\mathbf f[\mathbf z, \boldsymbol \phi]$ 는 data의 중요한 특징들을 표현하는 nonlinear 함수이고 $\sigma^2 \mathbf I$ 는 noise 이며 $\mathbf f[\mathbf z, \boldsymbol \phi]$ 로 모델링되지 못한 나머지를 표현하는 부분이다.

따라서, data 의 확률 분포는 아래와 같이 나타내어진다.

![Untitled 7](https://github.com/user-attachments/assets/893437d1-ac48-4e47-9f84-6e3cd5f1f734)

이는 각자 다른 mean 을 가진 무수히 많은 spherical (covariance 가 $\sigma^2 \mathbf I$ 형태)  Gaussian distribution 이 다른 weight (가중치) 로 합해진 것이 data distribution 임을 나타낸다. (chapter 1 에서 설명한 예시를 생각해보면 이해가 될 것이다) 이 때 weight (가중치) 는 $Pr(\mathbf z)$ 이고 각 Gaussian distribution 의 mean 은 network $\mathbf f[\mathbf z, \boldsymbol \phi]$ 의 결과값이 되는 것이다. 이를 나타낸 것이 아래 그림인데 참고하라.

![Untitled 8](https://github.com/user-attachments/assets/6701392e-fe69-4f39-8177-d1078283a30e)


### 2.1 Generation?

그럼 latent variable model 에서 어떻게 data 를 생성할까? 아래 그림을 보자.

![Untitled 9](https://github.com/user-attachments/assets/ac5943dd-ac5e-490c-86f4-1f00f8794920)

먼저 prior distribution $Pr(\mathbf z)$ 에서 $\mathbf z^\ast$ 를 sampling 한다. sampling 한 latent variable $\mathbf z^\ast$ 를 network $\mathbf f[\mathbf z^\ast, \boldsymbol \phi]$ 에 통과시킨 값이 $Pr(\mathbf x \mid \mathbf z^\ast, \boldsymbol \phi)$ 의 mean이 되고 noise $\sigma^2 \mathbf I$ 는 따로 sampling 하여 $Pr(\mathbf x \mid \mathbf z^\ast, \boldsymbol \phi)$ distribution 을 얻을 수 있다. 이로부터 data $\mathbf x^\ast$ 를 sampling 할 수 있다.

## 3. Training

앞에서 설명한 VAE model 을 학습시키는 것은 training dataset $\{\mathbf x\_i \}^I\_{i=1}$ 에 대해 log-likelihood 를 가장 크게 만드는 model parameter $\boldsymbol \phi$ 를 찾는 것이다. 

현재 설명에서는 좀 단순하게 설명하기 위해 likelihood $Pr(\mathbf x \mid \mathbf z^\ast, \boldsymbol \phi)$ 에서 variance 에 해당하는 $\sigma^2$ 가 고정된 값이라 가정한다.

그러면 data distribution 은 아래와 같고,

![Untitled 10](https://github.com/user-attachments/assets/318d989b-7cd6-49e6-a594-09addf39d15e)

우리가 찾아야 하는 model parameter $\boldsymbol \phi$ 를 수식적으로 표현하면 아래와 같다.

![Untitled 11](https://github.com/user-attachments/assets/81be85c1-da47-408e-b5ff-0453ea51ebb9)

안타깝게도 위의 $Pr(\mathbf x\_i \mid \boldsymbol \phi)$ 를 나타내는 식이 intractable (계산 불가능) 하다. 따라서, ELBO 를 도입하게 된다.

### 3.1 ELBO (Evidence Lower Bound)

ELBO 혹은 VLB (Variational Lower Bound) 라고 부르기도 한다. 위에서 설명한 것처럼 log-likelihood 를 직접 계산하는 것은 불가능하기 때문에 대신 log-likelihood 의 “lower bound (target 값보다 작거나 같은 값)” 를 표현하고 이 lower bound 를 최대화하는 것을 학습의 목적으로 하는 것이다. 

ELBO 를 도출하는 과정에서 Jensen’s inequality 가 사용되는데 이에 대한 설명은 [Jensen's_inequality_Dohoney](https://dhkwon03.github.io/ai/deep_dive_into_diffusion_DDPM_paper/#41-jensens-inequality) 을 참고하라. 

그럼 ELBO 를 유도해보자.

임의의 probability distribution $q(\mathbf z)$ 가 있다고 하자. $q(\mathbf z)$ 를 이용해서 log-likelihood 를 아래와 같이 나타낼 수 있다.

![Untitled 12](https://github.com/user-attachments/assets/069b9b1c-500b-4d26-aa54-3a7721ab2bdd)

Jensen’s inequliaty 를 적용하면 아래와 같은 lower bound 를 얻을 수 있다.

![Untitled 13](https://github.com/user-attachments/assets/62c1a0a7-54ea-43a6-8cb8-06a76b74be61)

오른쪽 항이 ELBO (Evidence Lower Bound) 이다. Bayes’ rule 에서 $Pr(\mathbf x \mid \boldsymbol \phi)$ 가 ‘evidence’ 라 불리기 때문에 그런 이름이 붙었다고 한다.

$q(\mathbf z)$ 가 지금까지는 임의의 확률 분포 였는데 우리는 여기에 parameter $\boldsymbol \theta$ 를 부여한다. 즉, ELBO는 아래와 같이 $\boldsymbol \theta$ 와 $\boldsymbol \phi$ 로 나타낼 수 있다. (ELBO 를 표현하는 첫번째 방법)

![Untitled 14](https://github.com/user-attachments/assets/75ae4bb9-ca27-4e83-bc2c-48ea48981217)

즉, nonlinear latent variable model 의 학습은 $\boldsymbol \theta$ 와 $\boldsymbol \phi$ 를 조정하여 위의 ELBO 를 최대화하는 것이며 이 최대값을 계산하는 neural architecture 가 바로 VAE 이다.

## 4. Properties of ELBO

ELBO 를 최대화하기 위해 ELBO의 특성들에 대해 더 알아보자.

![Untitled 15](https://github.com/user-attachments/assets/aa268eaa-07ed-4b6d-9f68-eae260a25246)

일단 원래의 log-likelihood 인 $\log[Pr(\mathbf x \mid \boldsymbol \phi)]$ 는 $\boldsymbol \phi$에 대한 함수이고 $\boldsymbol \theta$ 에는 영향을 받지 않는다. ELBO 함수는 항상 log-likelihood 아래에 있으며 $\boldsymbol \theta$ 와 $\boldsymbol \phi$ 에 모두 영향을 받는다. 위의 그림에서 볼 수 있듯이 $\boldsymbol \theta$ 가 변하면 ELBO 값은 변하지만 log-likelihood 함수는 그대로 이고, $\boldsymbol \phi$ 가 변하면 위의 그림 b) 처럼 ELBO 함수 상에서 이동하는 효과가 발생하고 이는 ELBO 값과 log-likelihood 값 모두 변하게 된다.

### 4.1 Tightness of bound

$\boldsymbol \phi$ 가 고정되어 있을 때 likelihood 함수와 ELBO가 만나게 되면 ELBO 가 “tight” 하다고 한다. 즉, $\boldsymbol \theta$ 를 잘 조정해서 log-likelihood 에 ELBO 함수가 닿을 정도로 차이를 줄였다는 것을 의미한다. 이런 ELBO 를 만드는 $q(\mathbf z \mid \boldsymbol \theta)$ 를 찾기 위해 conditional probability 를 활용하여 아래와 같이 ELBO 식을 변형할 수 있다. (ELBO 를 표현하는 두번째 방법)

![04090fa6-1606-4465-951e-741a63cb71ff](https://github.com/user-attachments/assets/717ccbac-c72e-4d3c-b730-77a0854439b6)

세번째 줄에서 네번째 줄로 넘어갈 때 $\log[Pr(\mathbf x \mid \boldsymbol \phi)]$ 는 $\mathbf z$ 와 independent 하고 $q(\mathbf z \mid \boldsymbol \theta)$ 는 $\mathbf z$ 에 대해 적분하면 1 이기 때문에 결과적으로 $\log[Pr(\mathbf x \mid \boldsymbol \phi)]$ 가 되는 것이다.

즉, 위의 식에 따르면 ELBO 는 원래의 log-likelihood 에서 KL-divergence 를 뺀 값이 된다. KL-divergence 는 distribution 간의 “거리” 를 나타내는 값이다. 따라서, KL-divergence 항이 0이 되어야 “tight” 하다는 것이고, 이는 $q(\mathbf z \mid \boldsymbol \theta)=Pr(\mathbf z \mid \mathbf x, \boldsymbol \phi)$ 임을 의미한다. $Pr(z \mid \mathbf x^\ast, \boldsymbol \phi)$ 는 data point $\mathbf x^\ast$ 를 만들기 위해 특정 latent variable $z$ 가 얼마만큼의 weight (가중치) 로 포함되어 있는지 나타내는 분포이다.

### 4.2 ELBO as reconstruction loss minus KL distance to prior

(ELBO 를 나타내는 세번째 방법)

![Untitled 16](https://github.com/user-attachments/assets/8c515110-2362-4e86-8d3d-27369e6bb9d6)

첫번째 항은 각 latent variable z 가 x 에 얼마나 “agree” 하는지 평균적인 agreement 를 측정하였다. 즉, latent variable 로부터 생성한 data가 기존의 data와 얼마나 비슷한지를 나타내는 error 인 reconstruction error 에 해당하는 항이다. 두번째 KL-divergence 항은 $q(\mathbf z \mid \boldsymbol \theta)$ 와 prior distribution 이 얼마나 가까운지 분포간 거리를 나타내는 항이다. 

ELBO 를 표현할 수 있는 3가지 방식을 살펴보았는데 “**VAE 에서 사용되는 ELBO 의 표현방식은 (reconstruction loss - KL distance) 로 표현한 세번째 방식이다.”**

## 5. Variational approximation

chapter 4.1 에서 $q(\mathbf z \mid \boldsymbol \theta)=Pr(\mathbf z \mid \mathbf x, \boldsymbol \phi)$ 일 때 ELBO 가 가장 tight 하다고 했다. posterior $Pr(\mathbf z \mid \mathbf x, \boldsymbol \phi)$ 를 아래와 같이 Bayes’ rule 로 표현할 수 있지만, 

![Untitled 17](https://github.com/user-attachments/assets/539ffd44-c19b-4d67-96c2-e15ede349e09)

이는 intractable 하다.

그래서 Variational approximation을 도입하게 된다. $q(\mathbf z \mid \boldsymbol \theta)$ 를 parameter $\boldsymbol \theta$ 에 대한 함수로 간단하게 정의 (normal distribution 으로 정의함) 하고 이를 posterior $Pr(\mathbf z \mid \mathbf x, \boldsymbol \phi)$ 에 최대한 가깝게 되도록 근사하는 방식이다. posterior 에 완전히 같은 distribution 을 찾을 수는 없겠지만 “가장 가까운” distribution 을 찾을 수 있다는 것이 바로 Variational approximation 의 논리이다. (위의 Figure 17.6 a) 에서 ELBO curve 를 log-likelihood 에 가장 가깝게 붙이는 $\boldsymbol \theta$ 를 찾는 것이다) 

$q(\mathbf z \mid \boldsymbol \theta)$ 를 posterior $Pr(\mathbf z \mid \mathbf x)$ 에 가장 가깝게 만들려면 data example $\mathbf x$ 도 condition 으로 들어간다. $q(\mathbf z \mid \mathbf x,  \boldsymbol \theta)$ 는 아래와 같이 정의된다.

![Untitled 18](https://github.com/user-attachments/assets/a0c6eb92-7620-4f29-925d-8790e35d9504)

이 때 $\mathbf g[\mathbf x, \boldsymbol \theta]$ 는 neural network 이며 normal variational approximation $q(\mathbf z \mid \mathbf x,  \boldsymbol \theta)$ 의 mean 과 variance 를 예측한다.

## 6. The Variational Autoencoder

최종적으로 $q(\mathbf z \mid \mathbf x,  \boldsymbol \theta)$ 가 위와 같을 때, VAE는 아래의 ELBO 를 계산하게 된다. (세번째 ELBO 표현 방식과 동일)

![Untitled 19](https://github.com/user-attachments/assets/181daa73-199f-4faa-8f74-257be62aaf75)

첫번째 항은 원래 intractable 하지만 $q(\mathbf z \mid \mathbf x,  \boldsymbol \theta)$ 에 대한 expectation 이기 때문에 이는 Monte Carlo estimate 로 근사할 수 있다. Monte Carlo estimate 란 임의의 함수 $a[\cdot]$ 가 있을 때 아래와 같이 expectation 을 sampling 을 통해 근사하는 것을 말한다. 

![Untitled 20](https://github.com/user-attachments/assets/94af79e0-3cfc-4213-b645-6226e4ac98ad)

이 때 $\mathbf z\_n^\ast$ 은 $q(\mathbf z \mid \mathbf x,  \boldsymbol \theta)$ 에서 n 번째로 sampling 한 것이다. 하나의 $\mathbf z\_n^\ast$ 으로 근사하면 아래와 같다. (이는 상당히 근사를 많이 한 것이 된다)

![Untitled 21](https://github.com/user-attachments/assets/a292ab81-87f5-4003-9e46-afca03cb9a12)

앞에서 살펴본 바에 따르면 $q(\mathbf z \mid \mathbf x,  \boldsymbol \theta)=Norm\_{\mathbf z}[\boldsymbol \mu, \boldsymbol \Sigma]$ 이고 $Pr(\mathbf z)=Norm\_{\mathbf z}[\boldsymbol 0, \mathbf I]$ 이므로 두 normal distribution 간의 KL-divergence 공식을 활용하면 (이미 알려진 공식임) 

![Untitled 22](https://github.com/user-attachments/assets/441908bf-69bc-4617-a2e7-2eb2d8900ac7)

이다. 따라서, 우리는 ELBO 를 계산할 수 있게 되었다. (tractable form 으로 만들기 위해 근사하긴 했지만)

### 6.1 VAE algorithm

정리하면 data point x 에 대한 ELBO 를 계산하기 위한 model 을 만들고 ($\mathbf g[\mathbf x, \boldsymbol \theta]$) 이 ELBO 를 dataset 에 대해 최대화 하기 위한 optimization 을 진행하여 log-likelihood 를 최대화하는 것이 VAE의 궁극적인 목표이다.

![Untitled 23](https://github.com/user-attachments/assets/4bc29c48-14f7-4f9e-a94f-a302ec9796b8)

위의 그림과 같이 encoder network $\mathbf g[\mathbf x, \boldsymbol \theta]$ 를 통해 예측된 mean과 variance 로 $q(\mathbf z \mid \mathbf x,  \boldsymbol \theta)$ 분포를 얻어내고 이 분포에서 sample $\mathbf z^\ast$ 를 sampling 한다. 이후 decoder network $\mathbf f[\mathbf z^\ast, \boldsymbol \phi]$를 통과한 $Pr(\mathbf x \mid \mathbf z^\ast, \boldsymbol \phi)$ 와 $q(\mathbf z \mid \mathbf x,  \boldsymbol \theta)$ 를 통해 ELBO 를 계산한다. 

VAE 는 위의 모델을 기반으로 parameter $\boldsymbol \theta$ 와 $\boldsymbol \phi$ 를 모두 변화시키며 SGD 또는 Adam 등의 optimization algorithm 으로 ELBO 값을 optimize 한다. 

## 7. Reparameterization trick

앞에서 설명한 VAE algorithm 에서는 sampling 을 통한 approximation 이 있고 이러한 stochastic 한 요소는 gradient 를 계산하기에 (미분하기에) 매우 까다로운 부분이다.

이를 reparametrization 을 통해 해결할 수 있다. standard normal distribution 에서 $\boldsymbol \epsilon^\ast$ 을 sampling 하고 이를 통해 아래와 같이 $\mathbf z^\ast$ 를 도출할 수 있다.

<img width="173" alt="Untitled 24" src="https://github.com/user-attachments/assets/99e5d184-875e-4cb1-9d2b-771f9ded8aa7">

또한, 아래 그림과 같이 random node 가 바뀌기 때문에 backpropagation 이 용이해진다.

![Untitled 25](https://github.com/user-attachments/assets/6ae97583-1d9c-4e3e-a3cb-235afc61becc)

## 8. Applications

이 VAE 는 denoising, anomaly detection, compression 등 많은 곳에 적용될 수 있다. 필자가 논문을 본 경험상 VAE 를 단독으로 사용하는 경우도 있지만 VAE 구조를 model architecture 에 포함시켜서 이용하는 경우가 굉장히 많은 것 같다. 

대표적으로 Latent diffusion 의 경우 이 VAE 구조를 활용하여 data space 를 latent space 로 변환한 후에 latent space 상에서 diffusion 을 적용한 사례이며 Diffusion 이라는 구조 자체가 hierarchical VAE 라 볼 수 있을만큼 Diffusion 과도 많은 연관이 있다.

이외에도 다른 연구에서 제안한 model architecture 에서도 VAE 를 많이 이용하고 있으니 논문을 읽다가 VAE 를 보면 반가움을 느껴보자.

### Comment

This post is the part of ‘Uploading 5 academic blog posts’ which is July’s resolution project in 9th squad.

본 post 는 9생의 7월 목표에서 필자의 목표인 ‘학술 블로그 포스트 5개 업로드’ 의 일환임을 알림

## Reference

- Understanding Deep Learning (Simon J.D. Prince, The MIT Press, 2023)
- [[용어정리] reparameterization trick (tistory.com)](https://heygeronimo.tistory.com/40)
