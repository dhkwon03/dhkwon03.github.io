---
title: Deep Dive into Diffusion; Denoising Diffusion Probabilistic Models (DDPM)
tags:
- paper_review
- concept
- 9S_July2024
categories:
- AI
---

이 글은 “Understanding Deep Learning” 이라는 책의 설명 방식과 순서를 따랐으며 이 책을 많이 참고하고 정리하여 작성되었다. 또한 이 내용은 DDPM 논문과 관련된 내용이며 DDPM 논문과 연계하여 설명하였다.

순서상으로는 VAE에 관해 먼저 공부하고 이 글을 보면 더 이해가 잘 된다. Diffusion 을 먼저 정리하고 싶어서 VAE에 관한 글은 추후에 쓸 예정이다.

# 1. Overview

Diffusion model 은 확률론적 (probabilistic) 모델이다. latent variable 에서 observed data (생성하는 data) 로의 nonlinear mapping 이라 할 수 있으며 이 때 observed data 와 latent variable의 차원 크기가 항상 같다는 것이 특징이다. 참고로, input data 를 VAE (Variational Autoencoder) 에 통과시켜 “차원을 낮춘” latent vector을 가지고 diffusion을 진행한 후 그 결과물을 VAE 의 decoder에 통과시켜 고해상도의 생성 이미지를 얻는 stable diffusion 이라는 것도 있는데 이는 추후에 다른 post 에서 다룰 것이다. 여튼, 이 post 에서 설명하는 건 latent variable과 input data, output data 가 모두 차원 (dimension) 이 같은 기본적인 Diffusion 이다. 

VAE 에서는 encoder와 decoder를 동시에 학습하며 data가 encoder를 통과하면 latent variable이 튀어나오며 이를 decoder에 그대로 넣어서 output data를 생성하는 방식이다. 학습을 위해 VAE에서는 data likelihood의 lower bound를 도출한다. (data likelihood 를 수식적으로 도출해보면 intractable, 즉 실제로 계산이 불가능하기 때문) VAE에서는 encoder 또한 학습의 대상이다.

하지만, Diffusion model 에서는 encoder 가 이미 정해져있으며 학습의 대상이 아니다. 따라서, decoder 만 학습의 대상이다. VAE와 마찬가지로 intractable 한 data likelihood 의 lower bound (which is tractable) 를 도출하는데 이 때 encoder의 definition 을 활용한다. 이 lower bound 를 기반으로 학습을 할 때 사용하는 Loss 를 정의한다.

### 1.1 기본적인 schema

Diffusion model은 “encoder”와 “decoder” 로 구성된다. encoder와 decoder 모두 deterministic 하지 않은, stochastic (확률론적인) mapping 을 한다.    
     
![Untitled](https://github.com/dhkwon03/dhkwon03.github.io/assets/83265598/86a036ad-74ef-458f-a8fd-c9aefa5198f0)       
위 그림에서 encoder는 input data x가 들어오면 $z\_1, z\_2, ..., z\_T$ 를 내놓는다. $z\_1, z\_2, ..., z\_T$ 는 “latent variable” 이라 부른다. encoder 는 입력으로 들어온 data image에 noise를 점차 추가하며 처음 noise 를 추가한 결과물이 $z\_1$, 거기에 더 noise 를 추가하면 $z\_2$, 최종적으로 T step 이후에는 Gaussian noise 인 $z_T$ 가 나온다. 즉, 충분한 step 이 지나면 conditional distribution $q(z\_T \mid x)$ 와 marginal distribution $q(z\_T)$ 가 모두 standard normal distribution 이 된다. Diffusion 에서 encoder 는 이미 정해져있다. 즉, encoder 에서 학습되는 parameter 는 없다. 
 
decoder 는 encoder와 반대의 process를 한다. $z\_T$ 에서  $z\_{T-1}, ..., z\_1$ 순으로 점차 de-noising (noise를 제거) 을 하여 최종적으로 data x 를 생성하는 것이 목적이다. Diffusion에서 loss function 은 encoder step 의 invert process가 구현되도록 decoder 를 학습시킨다. 즉, decoder 는 $q(z\_T)$ 에서 sampling 된 sample을 입력으로 받아 새로운 data x 를 생성하며 decoder의 parameter 는 학습의 대상이다. 

## 2. Encoder (forward process)

diffusion process 또는 forward process 라고 한다. data x를 아래의 수식에 따라 $z\_1, z\_2, ..., z\_T$ 의 latent variable 로 mapping 하며 모두 x와 같은 크기 (dimension) 이다. 
 
latent variable은 아래와 같이 표현된다.  
$$  
\mathbf{z}_1=\sqrt{1-\beta_1} \cdot \mathbf x + \sqrt{\beta_1} \cdot \mathbf{\epsilon}_1 \\ \mathbf{z}_t=\sqrt{1-\beta_t} \cdot \mathbf z_{t-1} + \sqrt{\beta_t} \cdot \mathbf{\epsilon}_t \,\,\,\,\, \forall t \in 2, ..., T 
$$       

위 식에서 $\epsilon\_t$ 는 standard normal distribution 으로부터 나온 noise 값이다. 수식을 보면 이전 latent variable 혹은 data (첫번째 step 경우) 의 크기를 줄이고 standard normal distribution 에서 나온 noise를 scaling 하여 추가하는 것을 볼 수 있다. latent variable의 점화식을 저렇게 정의한 이유는 그냥 추후에 나올 diffusion kernel 을 정의하기 수월하도록 하기 위해 noise를 추가하는 과정을 저렇게 정의한 것으로 추측된다는 것이 필자의 뇌피셜이다. $\beta\_t$ 는 [0, 1] 의 범위를 가지며 hyperparameter 이다. $\beta\_t$ 는 noise가 얼마나 빨리 추가되는지 결정하며 “noise schedule” 이라 불린다. noise schedule 은 모델 설계자가 정하기 나름이며 CosineAnnealingWarmUp 등을 사용한다고 하는데 정확하지는 않다. 
 
forward process 는 다음과 같이 수식적으로 표현된다. 

$$  
q(\mathbf{z}_1 \mid \mathbf{x})=Norm_{\mathbf{z}_{1}}[\sqrt{1-\beta_1}\mathbf{x}, \beta_1\mathbf I]\\q(\mathbf{z}_t \mid \mathbf{z}_{t-1})=Norm_{\mathbf{z}_t}[\sqrt{1-\beta_t}\mathbf{z}_{t-1}, \beta_t\mathbf I]\,\,\,\,\, \forall t \in 2, ..., T  
$$       

이 forward process 는 Markov Chain 이다. Markov Chain 이라 함은 과거의 값에는 확률분포가 영향을 받지 않으며 오직 바로 이전 step의 값에만 영향을 받는 것을 말한다. 즉, $q(\mathbf z\_{t} \mid \mathbf{z}\_{t-1})=q(\mathbf{z}\_t \mid \mathbf{z}\_{t-1}, \mathbf{z}\_{t-2})=q(\mathbf{z}\_t \mid \mathbf{z}\_{t-1}, \mathbf{z}\_{t-2}, \mathbf{z}\_{t-3})=...$ 이다. 위의 수식을 보면  $\mathbf{z}\_{t}$ 의 확률분포가 오직 $\mathbf{z}\_{t-1}$의 영향만 받는 것을 볼 수 있다. Markov Chain 임을 확인할 수 있다. 
“충분한” step 을 지난 후 (T step) 처음 data x의 값에 상관없이 $q(\mathbf{z}\_T \mid \mathbf x)=q(\mathbf z\_T)=Norm(0, \mathbf I)$ 가 된다.   
그리고 우리는 모든 x가 주어졌을 때 latent variable 의 joint distribution 을 아래와 같이 구할 수 있다. joint distribution 이라 함은 모든 latent variable 하나하나를 모두 확률변수로 하는 distribution 이다. ($\mathbf z\_{1...t}$ 라는 notation 은 $\mathbf z\_1, \mathbf z\_2, ... \mathbf z\_t$ 를 의미) 
 
$$
q(\mathbf{z}_{1...T} \mid \mathbf{x})=q(\mathbf{z}_1 \mid \mathbf{x}) \displaystyle \Pi_{t=2}^{T} q(\mathbf{z}_t \mid \mathbf{z}_{t-1})
$$   

 
### 2.1 Diffusion Kernel

우리는 앞에서 설명한 것처럼 $q(\mathbf{z}\_T \mid \mathbf x)=q(\mathbf z\_T)=Norm(0, \mathbf I)$  가 되려면 “충분한 step T” 가 필요하다. 엄청나게 큰 t 에 대해서 순차적으로 $\mathbf{z}\_t$ 를 모두 생성하려면 너무 오랜 시간이 걸린다. 다행히 $q(\mathbf{z}\_t \mid 
\mathbf x)$ 가 closed-form 으로 표현된다. 이를 “diffusion kernel” 이라 한다. 이걸 이용하면 $\mathbf{x}$ 만 있어도 모든 $\mathbf z\_t$ 를 계산할 수 있다. diffusion kernel 을 유도해보자.    
앞에서 $q(\mathbf{z}\_t \mid \mathbf{z}\_{t-1})=Norm_{\mathbf{z}\_t}[\sqrt{1-\beta\_t}\mathbf{z}\_{t-1}, \beta\_t\mathbf I]$ 라 했다. 따라서, 첫 2개 latent variable을 표현해보면   
  
$$
\mathbf z_1 = \sqrt{1-\beta_1} \cdot \mathbf x+\sqrt \beta_1 \cdot \mathbf \epsilon_1\\\mathbf z_2 = \sqrt{1-\beta_2}\cdot \mathbf z_1+\sqrt \beta_2 \cdot \mathbf \epsilon_2
$$   

대입하면     
  
$$
\begin{aligned}& \mathbf z_2 = \sqrt{1-\beta_2}\cdot( \sqrt{1-\beta_1} \cdot \mathbf x+\sqrt \beta_1 \cdot \mathbf \epsilon_1)+\sqrt \beta_2 \cdot \mathbf \epsilon_2\\&=\sqrt{1-\beta_2}\cdot( \sqrt{1-\beta_1} \cdot \mathbf x+\sqrt{1-(1-\beta_1)} \cdot \mathbf \epsilon_1)+\sqrt \beta_2 \cdot \mathbf \epsilon_2\\&=\sqrt{(1-\beta_2)(1-\beta_1)}\cdot \mathbf x+\sqrt{1-\beta_2-(1-\beta_2)(1-\beta_1)}\cdot \mathbf \epsilon_1 +\sqrt \beta_2 \cdot \mathbf \epsilon_2\\&=\sqrt{(1-\beta_2)(1-\beta_1)}\cdot \mathbf x+\sqrt{1-(1-\beta_2)(1-\beta_1)}\cdot \mathbf \epsilon\\\end{aligned}
$$    
  
이 때 $\mathbf \epsilon\_{1} , \mathbf \epsilon\_{2}$ 는 standard normal distribution 을 따르므로 세번째 줄의 마지막 2개 항이 각각 ${1-\beta\_2-(1-\beta\_2)(1-\beta\_1)},  \beta\_2$ 의 variance 를 가지고 mean 0 인 normal distribution 을 따른다. 둘은 독립적이므로 세번째 줄 마지막 2개 항의 합은 두 variance 를 합한 ${1-(1-\beta\_2)(1-\beta\_1)}$ 의 variance를 가지고 mean 0 (합하면 당연히 0 이니까) 인 normal distribution 을 따른다. 따라서 4번째 줄이 성립한다. 이것을 $\mathbf z\_3, \mathbf z\_4, ...$ 에 대해서 반복하면 아래와 같이 일반화가 가능하며, 그에 따라 $q(\mathbf{z}\_t \mid \mathbf x)$ 가 특정 normal distribution 으로 도출된다. 

$$
\mathbf z_t=\sqrt{\alpha_t}\cdot \mathbf x + \sqrt {1-\alpha_t} \cdot \mathbf \epsilon\\where\,\,\,\, \alpha_t = \Pi^{t}_{s=1}(1-\beta_s)\\q(\mathbf z_t  \mid \mathbf x)=Norm_{\mathbf z_t}[\sqrt \alpha_t \cdot \mathbf x, (1-\alpha_t) \cdot \mathbf I]
$$     

우리는 이제 x 만 있으면 x에 대한 $\mathbf z\_1, \mathbf z\_2, ...\mathbf z\_T$ 의 distribution을 알 수 있게 되었다!  

### 2.2 Marginal distribution $q(\mathbf z\_t)$

marginal distribution 인 $q(\mathbf z\_t)$ 는 모든 가능한 x 에 대해 $q(\mathbf z\_t \mid \mathbf x)$ 를 합친 distribution 이다. (conditional probability 의 정의를 생각해보면 아래 식이 당연하다)  

$$
q\left( \mathbf{z_t} \right)=\int q \left(\mathbf{z}_t \mid \mathbf{x} \right) \operatorname{Pr}( \mathbf{x} ) d \mathbf{x}
$$  

근데 우리는 기존 dataset 의 data distribution 인 $\operatorname{Pr}(\mathbf{x})$ 를 모르기 때문에 marginal distribution 을 이렇게는 표현할 수 없다. 그러면 $q(\mathbf{z}\_t \mid \mathbf{z}\_{t-1})$ 을 알고 있으니까 Bayes’ Rule 을 써볼까?  

$$
q\left(\mathbf{z}_{t-1} \mid \mathbf{z}_t\right)=\frac{q\left(\mathbf{z}_t \mid \mathbf{z}_{t-1}\right) q\left(\mathbf{z}_{t-1}\right)}{q\left(\mathbf{z}_t\right)}
$$  

이것도 $q(\mathbf z\_{t-1})$ 을 모르기 때문에 계산이 불가능하다 (intractable).  

여기서 중요한게 하나 있는데 $q\left(\mathbf{z}\_{t-1} \mid \mathbf{z}\_t\right)$ 은 생각해보면 reverse process 의 일환임을 알 수 있다. 간단한 1-dimensional 에서는 위의 수식으로 이를 계산할 수 있다고 한다. 하지만, 우리는 더 복잡한 차원의 모델을 다룰 것이기에 의미는 없다. 알아둘 것은 “$q\left(\mathbf{z}\_{t-1} \mid \mathbf{z}\_t\right)$ 이 normal distribution 으로 근사된다”는 것이다. 실제로 뒤에서 설명할 decoder 에서는 이 process 를 normal distribution 으로 근사한다. 직관적으로 보면 $q\left(\mathbf{z}\_{t-1} \mid \mathbf{z}\_t\right)$ 이 normal distribution 이 명백하고 위의 수식에서 marginal distribution 끼리 어느정도 상쇄된다고 보여지는데 이 때문에 $q\left(\mathbf{z}\_{t-1} \mid \mathbf{z}\_t\right)$ 을 normal distribution 으로 근사할 수 있다고 하는 것 같다. 결론은 marginal distribution $q(\mathbf z\_t)$ 를 계산할 수 없다는 거다. 계산할 수 없기에 (후에 설명할) 복잡한 방법을 쓰는 것이다.   

### 2.3 Conditional diffusion distribution $q\left(\mathbf{z}\_{t-1} \mid \mathbf{z}\_t, \mathbf{x} \right)$

$q\left(\mathbf{z}\_{t-1} \mid \mathbf{z}\_t\right)$ 는 계산할 수 없는데, x 를 안다고 가정하면, 즉 $q\left(\mathbf{z}\_{t-1} \mid \mathbf{z}\_t, \mathbf {x}\right)$ 는 계산가능하다. (tractable) $q\left(\mathbf{z}\_{t-1} \mid \mathbf{z}\_t, \mathbf {x}\right)$ distribution 은 decoder 를 학습시키는 데 사용된다. $q\left(\mathbf{z}\_{t-1} \mid \mathbf{z}\_t, \mathbf {x}\right)$ 은 다음과 같이 표현된다.  

$$
\begin{aligned}
q\left(\mathbf{z}_{t-1} \mid \mathbf{z}_t, \mathbf{x}\right) & =\frac{q\left(\mathbf{z}_t \mid \mathbf{z}_{t-1}, \mathbf{x}\right) q\left(\mathbf{z}_{t-1} \mid \mathbf{x}\right)}{q\left(\mathbf{z}_t \mid \mathbf{x}\right)} \\
& \propto q\left(\mathbf{z}_t \mid \mathbf{z}_{t-1}\right) q\left(\mathbf{z}_{t-1} \mid \mathbf{x}\right) \\
& =\operatorname{Norm}_{\mathbf{z}_t}\left[\sqrt{1-\beta_t} \cdot \mathbf{z}_{t-1}, \beta_t \mathbf{I}\right] \operatorname{Norm}_{\mathbf{z}_{t-1}}\left[\sqrt{\alpha_{t-1}} \cdot \mathbf{x},\left(1-\alpha_{t-1}\right) \mathbf{I}\right] \\
& \propto \operatorname{Norm}_{\mathbf{z}_{t-1}}\left[\frac{1}{\sqrt{1-\beta_t}} \mathbf{z}_t, \frac{\beta_t}{1-\beta_t} \mathbf{I}\right] \operatorname{Norm}_{\mathbf{z}_{t-1}}\left[\sqrt{\alpha_{t-1}} \cdot \mathbf{x},\left(1-\alpha_{t-1}\right) \mathbf{I}\right] \\&=\operatorname{Norm}_{\mathbf{z}_{t-1}}\left[\frac{\left(1-\alpha_{t-1}\right)}{1-\alpha_t} \sqrt{1-\beta_t} \mathbf{z}_t+\frac{\sqrt{\alpha_{t-1}} \beta_t}{1-\alpha_t} \mathbf{x}, \frac{\beta_t\left(1-\alpha_{t-1}\right)}{1-\alpha_t} \mathbf{I}\right]
\end{aligned}
$$  

첫번째 줄에서는 Bayes’ Rule 을 사용했다. x 가 모든 항에 condition 으로 공통적으로 들어가있음을 고려하면 Bayes’ Rule 에 따라 저렇게 쓸 수 있다. forward process 가 Markov Chain 이라 했으므로 $q\left(\mathbf{z}\_{t-1} \mid \mathbf{z}\_t, \mathbf {x}\right)=q\left(\mathbf{z}\_{t-1} \mid \mathbf{z}\_t\right)$ 이다. 두번째 줄까지 설명이 되었다.  

두번째 줄에서 곱해진 두 distribution 은 각각 closed-form 이 앞에서 유도되었다. 그대로 대입하면 세번째 줄이다.  

세번째 줄에서 네번째 줄로 넘어가는 부분은 아래의 Gaussian change of variables identity 를 사용한다. (위 식에서 이 identity 를 활용하여 $\mathbf {z}\_t$  에 대한 normal distribution 을 mean을 나타내는 항에 포함되어 있는 $\mathbf {z}\_{t-1}$ 에 대한 normal distribution 으로 바꿨다) (이 identity 증명이 궁금하면 검색해보고 아니면 그냥 저런 identity가 성립하는구나 하고 넘어가도록 하자)  

$$
\operatorname{Norm}_{\mathbf{v}}[\mathbf{A w}, \mathbf{B}] \propto \operatorname{Norm}_{\mathbf{w}}\left[\left(\mathbf{A}^T \mathbf{B}^{-1} \mathbf{A}\right)^{-1} \mathbf{A}^T \mathbf{B}^{-1} \mathbf{v},\left(\mathbf{A}^T \mathbf{B}^{-1} \mathbf{A}\right)^{-1}\right]
$$       

또한 아래와 같이 두 개의 normal distribution을 하나의 normal distribution으로 합치는 identity 가 있다. 이는 네번째 줄에서 다섯번째 줄로 넘어갈 때 사용했다. (마찬가지로 궁금하면 검색해보면 되고 아니면 받아들이고 넘어가자)  

$$
\operatorname{Norm}_{\mathbf{w}}[\mathbf{a}, \mathbf{A}] \cdot \operatorname{Norm}_{\mathbf{w}}[\mathbf{b}, \mathbf{B}] \\\propto
\operatorname{Norm}_{\mathbf{w}}\left[\left(\mathbf{A}^{-1}+\mathbf{B}^{-1}\right)^{-1}\left(\mathbf{A}^{-1} \mathbf{a}+\mathbf{B}^{-1} \mathbf{b}\right),\left(\mathbf{A}^{-1}+\mathbf{B}^{-1}\right)^{-1}\right]
$$       

유도과정 중간에 proportionality 가 있지만 결과적으로 유도된 수식이 normalized probability distribution 이므로 (즉, 모든 확률의 합이 1인 온전한 확률 분포이므로) proportionality 에 의한 상수 (constant) 는 모두 상쇄된 것이라 볼 수 있다. 따라서, 아래의 등식이 성립한다.  

$$
\begin{aligned}
q\left(\mathbf{z}_{t-1} \mid \mathbf{z}_t, \mathbf{x}\right) =\operatorname{Norm}_{\mathbf{z}_{t-1}}\left[\frac{\left(1-\alpha_{t-1}\right)}{1-\alpha_t} \sqrt{1-\beta_t} \mathbf{z}_t+\frac{\sqrt{\alpha_{t-1}} \beta_t}{1-\alpha_t} \mathbf{x}, \frac{\beta_t\left(1-\alpha_{t-1}\right)}{1-\alpha_t} \mathbf{I}\right]
\end{aligned}
$$      

이로써 $q\left(\mathbf{z}\_{t-1} \mid \mathbf{z}\_t, \mathbf {x}\right)$ 의 closed-form 을 유도했다. $q\left(\mathbf{z}\_{t-1} \mid \mathbf{z}\_t, \mathbf {x}\right)$ 의 closed-form 은 후에 설명할 loss 를 표현하는 과정에서 사용된다.  

## 3. Decoder (reverse process)

decoder는 diffusion model 에서 학습이 이루어져야 하는 부분이다. latent variable $\mathbf {z}\_T$ 에서 $\mathbf {z}\_{T-1}$, $\mathbf {z}\_{T-1}$ 에서 $\mathbf {z}\_{T-2}$, …., $\mathbf {z}\_1$ 에서 data $\mathbf x$ 로 mapping 하는 (이어주는) 확률 분포를 전부 학습하는 것이다. 아래 figure 를 보자.  

![Untitled](https://github.com/dhkwon03/dhkwon03.github.io/assets/83265598/fa5fd2c4-99ea-4035-a75f-512b61d2bcc0)  

![Untitled](https://github.com/dhkwon03/dhkwon03.github.io/assets/83265598/1c4d9c25-520d-489c-99ca-bf6d927c9c74)  

(참고; Figure 18.5 (a) 는 1 dimension data (실제 모델에서는 쓸 일이 없는, 오직 위와 같은 예시를 위해 만들어진 듯 하다) 에 대해서는 $q\left(\mathbf{z}\_{t}\right)$ 가 계산이 가능하기 때문에 실제 시뮬레이션 후 그림으로 나타낸 것 같다.)$q\left(\mathbf{z}\_{t-1} \mid \mathbf{z}\_t\right)$와 $q\left(\mathbf{z}\_{t-1} \mid \mathbf{z}\_t, \mathbf {x}\right)$ 의 확률분포를 나타내고 있다. 그림에 있는 설명과 함께 이전에 살펴본 수식들과 비교하며 확률분포를 확인하면 이해에 도움이 될 것이다. (그냥 참고용이다. 굳이 안 해도 될 듯)  

여튼 위 그림에서 중요한 부분은 Figure 18.5 (b) 에서 $q\left(\mathbf{z}\_{t-1} \mid \mathbf{z}\_t\right)$ 의 확률분포이다. 갈색 분포는 항상 정규분포임을 생각하면 t=3 일 때 $q\left(\mathbf{z}\_{2} \mid \mathbf{z}\_3\right)$ 가 정규분포로 근사가 잘 되지 않음을 볼 수 있다. 하지만 그 이후 t가 커지면 어느정도 정규분포에 가깝게 근사할 수 있는 것을 볼 수 있다.   

reverse process 의 실제 distribution 인  $q\left(\mathbf{z}\_{t-1} \mid \mathbf{z}\_t\right)$ 은 complex multi-modal distribution 이다. 다양한 변수의 영향을 받는 굉장히 복잡한 distribution 이라는 건데 이 복잡한 distribution을 closed-form 으로 계산하는 건 불가능하니까 “$q\left(\mathbf{z}\_{t-1} \mid \mathbf{z}\_t\right)$ 를 그냥 normal distribution 으로 근사“한다. 이 부분은 Diffusion 모델에서 굉장히 중요한 지점이다.   

어떻게 보면 참 말이 안되는 부분이라 생각이 들 수 있지만 normal distribution 으로 근사하면 결과적으로 어느정도의 performance 가 나오기도 하고 forward process에서 normal distribution 을 기반으로 하며 분포를 비교해보았을 때 normal distribution 이 가장 개연성이 있기에 필자는 그렇게 근사를 한 것이 아닐까 추측한다. (필자 뇌피셜이다)  

### 3.1 Decoder model by approximation

normal distribution 으로 근사한 decoder model (reverse process) 는 다음과 같이 표현한다.  

$$
Pr(\mathbf{z}_T)=\operatorname {Norm}_{\mathbf{z}_T}[0, \mathbf I]\\Pr(\mathbf{z}_{t-1} \mid \mathbf{z}_t, \, \mathbf \phi_t) = \operatorname {Norm}_{\mathbf{z}_{t-1}}[\mathbf f_t[\mathbf z_t, \mathbf \phi_t], \sigma_t^2\mathbf I]\\Pr(\mathbf x \mid\mathbf{z}_1, \, \mathbf \phi_1) = \operatorname {Norm}_{\mathbf{x}}[\mathbf f_1[\mathbf z_1, \mathbf \phi_1], \sigma_1^2\mathbf I]
$$  

$\mathbf f_t[\mathbf z\_t, \mathbf \phi\_t]$ 는 neural network 이며 $\mathbf \phi\_t$  가 neural network 의 parameter 이다. 즉, 각 step에 대한 normal distribution 의 mean 을 예측하는 neural network 를 학습하는 것이다. $ {\sigma\_t^2}$ 은 모두 미리 정해진 값 (hyperparameter) 이다. (후에 설명할 부분에서 diffusion 모델의 성능을 높이기 위해 variance 또한 neural network 로 학습하는 방법도 제시가 되었는데 일단 여기서는 hyperparameter 로 두었다.)  

hyperparameter $\beta\_t$  가 0에 가깝고 time step T 가 클수록 실제 decoder model 이 우리가 근사한 normal distribution 에 더 가까워진다. 따라서 여러가지 컴퓨팅 자원이 따라주는 선에서 $\beta\_t$  는 0에 가깝게, T는 가능하면 크게 적절한 값으로 설정해야 한다.  

이 decoder model 을 사용해서 $\Pr(\mathbf x)$ 로 부터 ancestral sampling 을 통해 새로운 샘플 데이터를 생성하게 된다. ancestral sampling 이라 함은 특별한 것이 아니라 $\Pr(\mathbf z\_T)$ 에서 $\mathbf z\_T$ 를 sampling 하고, $\Pr(\mathbf z\_{T-1} \mid \mathbf z\_T, \, \mathbf \phi\_T)$ 에서 $\mathbf z\_{T-1}$, $\Pr(\mathbf z\_{T-2} \mid \mathbf z\_{T-1}, \, \mathbf \phi\_{T-1})$ 에서 $\mathbf z\_{T-2}$, …, $\Pr(\mathbf x \mid \mathbf z\_1, \, \mathbf \phi\_1)$ 에서 $\mathbf x$ 를 순차적으로 sampling 하는 것을 의미한다.  

## 4. Training

$\mathbf {x}$ 와 latent variable  $\{\mathbf z\_t\}$ 의 joint distribution 은 아래와 같다.  

$$
\operatorname{Pr}\left(\mathbf{x}, \mathbf{z}_{1 \ldots T} \mid \mathbf{\phi}_{1 \ldots T}\right)=\operatorname{Pr}\left(\mathbf{x} \mid \mathbf{z}_1, \mathbf{\phi}_1\right) \prod_{t=2}^T  \operatorname{Pr}\left(\mathbf{z}_{t-1} \mid \mathbf{z}_t, \mathbf{\phi}_t\right) \cdot \operatorname{Pr}(\mathbf{z}_T)
$$  

latent variable 들에 대해 marginalize 하면 (integral 로 합하면) 아래와 같이 $Pr(\mathbf x \mid \mathbf \phi\_{1...T})$ 를 표현할 수 있다.  

$$
\operatorname{Pr}\left(\mathbf{x} \mid \phi_{1 \ldots T}\right)=\int \operatorname{Pr}\left(\mathbf{x}, \mathbf{z}_{1 \ldots T} \mid \phi_{1 \ldots T}\right) d \mathbf{z}_{1 \ldots T}
$$  

우리는 diffusion model 을 training 할 때 parameter $\mathbf \phi$ 에 대한 training data ${\mathbf x\_i}$ 의 log-likelihood 를 최대화해야 한다. 즉, neural network 의 parameter 를 학습시켜서 생성된 data가 최대한 training dataset 의 분포 (distribution) 에 근접하도록 만드는 것이다. 따라서 아래와 같이 log-likelihood 를 최대화하는 parameter $\mathbf \phi$ 를 찾는 것이 학습의 목표이다.   

$$
\\ \hat{\mathbf{\phi}}_{1 \ldots T}=\underset{\mathbf{\phi}_{1 \ldots T}}{\operatorname{argmax}}\left[\sum_{i=1}^I \log \left[\operatorname{Pr}\left(\mathbf{x}_i \mid \mathbf{\phi}_{1 \ldots T}\right)\right]\right]
$$  

문제는 위에서 $Pr(\mathbf x \mid \mathbf \phi\_{1...T})$ 를 표현한 식이 intractable 하기 때문에 대신 “likelihood 의 lower bound를 최대화” 한다. likelihood의 lower bound 는 Jensen’s inequality (젠센 부등식) 을 이용하여 유도한다. (VAE 와 같은 방식이다)  

### 4.1 Jensen’s inequality

![Untitled](https://github.com/dhkwon03/dhkwon03.github.io/assets/83265598/373007c9-b5d2-4ec7-937c-5e81f62b8c27)  

(Jensen’s inequality); 어떤 concave function (함수의 기울기가 x축을 따라서 점점 작아지는 함수) $g[\cdot]$ 가 있을 때 $g[\mathbb E[y]] \geq \mathbb E[g[y]]$  

log 함수는 concave function 이기 때문에 Jensen’s inequality 를 적용할 수 있다. 따라서,  

$$
\log [\mathbb{E}[y]] \geq \mathbb{E}[\log [y]]
$$  

expectation 정의에 따라 풀어서 쓰면   

$$
\log \left[\int \operatorname{Pr}(y) y d y\right] \geq \int \operatorname{Pr}(y) \log [y] d y
$$  

$h[y]$ (y에 대한 함수)를 y 대신 대입하여 (h[y] 는 새로운 distribution 이자 다른 확률 변수이기 때문에 성립한다) 좀 더 일반적으로 표현하면  

$$
\log \left[\int \operatorname{Pr}(y) h[y] d y\right] \geq \int \operatorname{Pr}(y) \log [h[y]] d y
$$  

위의 식이 $\log {\mathbb E\_y[h[y]]} \geq \mathbb E\_y[\log h[y]]$  와 같은 식임을 생각하면 이해가 더 쉽다.  

### 4.2 Evidence lower bound (ELBO)

log likelihood 를 우리가 알고 있는 encoder distribution 인 $q(\mathbf z\_{1...T} \mid \mathbf x)$ 를 이용해 표현하고 Jensen’s inequality 를 적용한다.  

$$
\begin{aligned}\log \left[\operatorname{Pr}\left(\mathbf{x} \mid \phi_{1 \ldots T}\right)\right] & =\log \left[\int \operatorname{Pr}\left(\mathbf{x}, \mathbf{z}_{1 \ldots T} \mid \phi_{1 \ldots T}\right) d \mathbf{z}_{1 \ldots T}\right] \\& =\log \left[\int q\left(\mathbf{z}_{1 \ldots T} \mid \mathbf{x}\right) \frac{\operatorname{Pr}\left(\mathbf{x}, \mathbf{z}_{1 \ldots T} \mid \mathbf{\phi}_{1 \ldots T}\right)}{q\left(\mathbf{z}_{1 \ldots T} \mid \mathbf{x}\right)} d \mathbf{z}_{1 \ldots T}\right] \\& \geq \int q\left(\mathbf{z}_{1 \ldots T} \mid \mathbf{x}\right) \log \left[\frac{\operatorname{Pr}\left(\mathbf{x}, \mathbf{z}_{1 \ldots T} \mid \phi_{1 \ldots T}\right)}{q\left(\mathbf{z}_{1 \ldots T} \mid \mathbf{x}\right)}\right] d \mathbf{z}_{1 \ldots T} \\&=ELBO[\mathbf \phi_{1...T}]\end{aligned}
$$  

VAE 에서는 encoder의 $q(\mathbf z \mid \mathbf x)$ 를 조정하여 ELBO 와 실제 log-likelihood 간의 차이를 줄이고 decoder 가 ELBO를 최대화 하는 방식이다. 하지만, Diffusion model 에서는 $q(\mathbf z \mid \mathbf x)$ 가 diffusion kernel (closed-form) 이므로, decoder 에서 log-likehood 와 ELBO 간의 차이를 줄이고 ELBO를 최대화 하는 것이 동시에 일어난다. 즉, decoder의 parameter 를 조정함으로서 posterior 인 $Pr(\mathbf z\_{1...T} \mid \mathbf x, \mathbf \phi\_{1...T})$ 를 encoder (encoder 의 수식은 고정) 와 최대한 가깝게 만들고 lower bound 에 대해서 parameter 를 optimize 한다.  

ELBO 식에서 log 항을 정리해야 한다. 일단 아래와 같은 식이 성립한다.  

$$
q\left(\mathbf{z}_{t} \mid \mathbf{z}_{t-1}\right)=q\left(\mathbf{z}_{t} \mid \mathbf{z}_{t-1}, \mathbf{x}\right)=\frac{q\left(\mathbf{z}_{t-1} \mid \mathbf{z}_{t}, \mathbf{x}\right) q\left(\mathbf{z}_{t} \mid \mathbf{x}\right)}{q\left(\mathbf{z}_{t-1} \mid \mathbf{x}\right)}
$$  

첫번째 등호는 Markov Chain 의 특성 때문에 성립한다. ($\mathbf z\_t$ 의 확률 분포는 오직 $\mathbf z\_{t-1}$ 에만 depend 하므로 condition 부분에 $\mathbf x$ 를 추가해도 상관없다) 두 번째 등호는 Bayes’ Rule 을 사용한 것이다. 이는 아래 식에서 두번째 줄에서 세번째 줄로 정리하는 부분에서 대입하였다.  

그리고 ELBO 에서 log 항을 정리하면 다음과 같다.  

$$
\begin{aligned}&\log \left[\frac{\operatorname{Pr}\left(\mathbf{x}, \mathbf{z}_{1 \ldots T} \mid \mathbf{\phi}_{1 \ldots T}\right)}{q\left(\mathbf{z}_{1 \ldots T} \mid \mathbf{x}\right)}\right]\\&=\log \left[\frac{\operatorname{Pr}\left(\mathbf{x} \mid \mathbf{z}_{1}, \mathbf{\phi}_{1}\right) \prod_{t=2}^{T} \operatorname{Pr}\left(\mathbf{z}_{t-1} \mid \mathbf{z}_{t}, \mathbf{\phi}_{t}\right) \cdot \operatorname{Pr}\left(\mathbf{z}_{T}\right)}{q\left(\mathbf{z}_{1} \mid \mathbf{x}\right) \prod_{t=2}^{T} q\left(\mathbf{z}_{t} \mid \mathbf{z}_{t-1}\right)}\right]\\
&=\log \left[\frac{\operatorname{Pr}\left(\mathbf{x} \mid \mathbf{z}_{1}, \mathbf{\phi}_{1}\right)}{q\left(\mathbf{z}_{1} \mid \mathbf{x}\right)}\right]+\log \left[\frac{\prod_{t=2}^{T} \operatorname{Pr}\left(\mathbf{z}_{t-1} \mid \mathbf{z}_{t}, \mathbf{\phi}_{t}\right)}{\prod_{t=2}^{T} q\left(\mathbf{z}_{t} \mid \mathbf{z}_{t-1}\right)}\right]+\log \left[\operatorname{Pr}\left(\mathbf{z}_{T}\right)\right]\\ &=\log \left[\frac{\operatorname{Pr}\left(\mathbf{x} \mid \mathbf{z}_{1}, \mathbf{\phi}_{1}\right)}{q\left(\mathbf{z}_{1} \mid \mathbf{x}\right)}\right]+\log \left[\frac{\prod_{t=2}^{T} \operatorname{Pr}\left(\mathbf{z}_{t-1} \mid \mathbf{z}_{t}, \mathbf{\phi}_{t}\right) \cdot q\left(\mathbf{z}_{t-1} \mid \mathbf{x}\right)}{\prod_{t=2}^{T} q\left(\mathbf{z}_{t-1} \mid \mathbf{z}_{t}, \mathbf{x}\right) \cdot q\left(\mathbf{z}_{t} \mid \mathbf{x}\right)}\right]+\log \left[\operatorname{Pr}\left(\mathbf{z}_{T}\right)\right] \\
&=\log \left[\operatorname{Pr}\left(\mathbf{x} \mid \mathbf{z}_{1}, \mathbf{\phi}_{1}\right)\right]+\log \left[\frac{\prod_{t=2}^{T} \operatorname{Pr}\left(\mathbf{z}_{t-1} \mid \mathbf{z}_{t}, \mathbf{\phi}_{t}\right)}{\prod_{t=2}^{T} q\left(\mathbf{z}_{t-1} \mid \mathbf{z}_{t}, \mathbf{x}\right)}\right]+\log \left[\frac{\operatorname{Pr}\left(\mathbf{z}_{T}\right)}{q\left(\mathbf{z}_{T} \mid \mathbf{x}\right)}\right] \\&
 \approx \log \left[\operatorname{Pr}\left(\mathbf{x} \mid \mathbf{z}_{1}, \mathbf{\phi}_{1}\right)\right]+\sum_{t=2}^{T} \log \left[\frac{\operatorname{Pr}\left(\mathbf{z}_{t-1} \mid \mathbf{z}_{t}, \mathbf{\phi}_{t}\right)}{q\left(\mathbf{z}_{t-1} \mid \mathbf{z}_{t}, \mathbf{x}\right)}\right]
\end{aligned}
$$  

첫번째 줄은 log 항의 분모, 분자를 풀어서 쓰는 것이고 항을 나눠서 정리한 것이 두번째 줄이다. 세번째 줄은 바로 이전에 설명한 $q(\mathbf z\_{t} \mid \mathbf z\_{t-1})$ 에 대한 식을 대입한 것이다. 그리고 telescoping 으로 항을 곱하다 보면 $q(\mathbf z\_{t-1} \mid \mathbf x)$ 와 $q(\mathbf z\_t \mid \mathbf x)$ 가 서로 상쇄되고 $q(\mathbf z\_1 \mid \mathbf x)$ 도 상쇄되어 사라지며 $\frac{1}{q(\mathbf z\_T \mid \mathbf x)}$ 만 남는 것을 알 수 있는데 네번째 줄에 나타내 있다. 네번째 줄 세번째 항에서 T가 충분히 크면 $q(\mathbf z\_T \mid \mathbf x)$ 는 standard normal distribution 으로 근사 가능하며 $\operatorname{Pr}(\mathbf z\_T)$ 는 standard normal distribution 이므로 log[1] = 0 으로 근사할 수 있다. (chapter 2.1, 3.1 참고)  

위의 결과를 ELBO 식에 대입하면 ,  

$$
\begin{aligned}&ELBO[\mathbf \phi_{1...T}]\\&= \int q(\mathbf z_{1...T} \mid \mathbf x) \log {\left[\frac{Pr(\mathbf x, \mathbf z_{1...T} \mid \mathbf \phi_{1...T})}{q(\mathbf z_{1...T} \mid \mathbf x)}\right]}d \mathbf z_{1...T}\\&\approx \int q(\mathbf z_{1...T} \mid \mathbf x) \left(\log \left[\operatorname{Pr}\left(\mathbf{x} \mid \mathbf{z}_{1}, \mathbf{\phi}_{1}\right)\right]+\sum_{t=2}^{T} \log \left[\frac{\operatorname{Pr}\left(\mathbf{z}_{t-1} \mid \mathbf{z}_{t}, \mathbf{\phi}_{t}\right)}{q\left(\mathbf{z}_{t-1} \mid \mathbf{z}_{t}, \mathbf{x}\right)}\right]\right)d \mathbf z_{1...T}\\&=\mathbb E_{q(\mathbf z_1 \mid \mathbf x)}\left[\log{\left[Pr(\mathbf x  \mid  \mathbf z_1, \mathbf \phi_1 )\right]}\right]-\sum^{T}_{t=2}\mathbb{E}_{q(\mathbf z_t \mid \mathbf x)}\left[D_{KL}[q(\mathbf z_{t-1} \mid \mathbf z_t, \mathbf x) \mid \mid Pr(\mathbf z_{t-1} \mid \mathbf z_t, \mathbf \phi_t)]\right]\end{aligned}
$$  

두번째 줄에서 세번째 줄로 넘어가는 부분을 좀 더 설명하고자 한다. joint distribution $q(\mathbf z\_{1...T} \mid \mathbf x)$ 는 $\mathbf x$ 가 정해져 있을 때 $\mathbf z\_1, \mathbf z\_2, ... \mathbf z\_T$ 각각 확률 변수 값의 모든 조합에 대한 확률을 나타내는 distribution 이라는 것을 생각해보면 결국 $q(\mathbf z\_{1...T} \mid \mathbf x)$ 와 곱해지는 항이 $\mathbf z\_t$ 에 depend 하면 $q(\mathbf z\_t \mid \mathbf x)$ 에 대해서만 integral 하면 된다. (이는 joint distribution 의 성질을 찾아보면 더 자세히 알 수 있을 것 같다)   

이를 이용하면,  

$$
\begin{aligned}\int q(\mathbf z_{1...T} \mid \mathbf x) \log \left[\operatorname{Pr}\left(\mathbf{x} \mid \mathbf{z}_{1}, \mathbf{\phi}_{1}\right)\right] d \mathbf z_{1...T}&=\int q(\mathbf z_1 \mid \mathbf x) \log \left[\operatorname{Pr}\left(\mathbf{x} \mid \mathbf{z}_{1}, \mathbf{\phi}_{1}\right)\right] d \mathbf z_1\\&=\mathbb E_{q(\mathbf z_1 \mid \mathbf x)}\left[\log{\left[Pr(\mathbf x  \mid \mathbf z_1, \mathbf \phi_1 )\right]}\right]\end{aligned}
$$  

$\log \left[\operatorname{Pr}(\mathbf{x} \mid \mathbf{z}\_{1}, \mathbf{\phi}\_{1})\right]$ 은 $\mathbf z\_1$ 에만 depend 하므로 위와 같이 정리된다.  

$$
\begin{aligned}&\int q(\mathbf z_{1...T} \mid \mathbf x) \sum_{t=2}^{T} \log \left[\frac{\operatorname{Pr}\left(\mathbf{z}_{t-1} \mid \mathbf{z}_{t}, \mathbf{\phi}_{t}\right)}{q\left(\mathbf{z}_{t-1} \mid \mathbf{z}_{t}, \mathbf{x}\right)}\right]d \mathbf z_{1...T}\\&=\sum_{t=2}^{T} \int q(\mathbf z_{1...T} \mid \mathbf x) \log \left[\frac{\operatorname{Pr}\left(\mathbf{z}_{t-1} \mid \mathbf{z}_{t}, \mathbf{\phi}_{t}\right)}{q\left(\mathbf{z}_{t-1} \mid \mathbf{z}_{t}, \mathbf{x}\right)}\right]d \mathbf z_{1...T}\\&=\sum_{t=2}^{T} \iint q(\mathbf z_{t-1}, \mathbf z_t \mid \mathbf x) \log \left[\frac{\operatorname{Pr}\left(\mathbf{z}_{t-1} \mid \mathbf{z}_{t}, \mathbf{\phi}_{t}\right)}{q\left(\mathbf{z}_{t-1} \mid \mathbf{z}_{t}, \mathbf{x}\right)}\right]d \mathbf z_{t-1}d\mathbf z_t\\&=\sum_{t=2}^{T} \iint q(\mathbf z_{t-1} \mid \mathbf z_t \mid \mathbf x) q(\mathbf z_{t} \mid \mathbf x) \log \left[\frac{\operatorname{Pr}\left(\mathbf{z}_{t-1} \mid \mathbf{z}_{t}, \mathbf{\phi}_{t}\right)}{q\left(\mathbf{z}_{t-1} \mid \mathbf{z}_{t}, \mathbf{x}\right)}\right]d \mathbf z_{t-1}d\mathbf z_t\\&=\sum_{t=2}^{T} \int q(\mathbf z_{t} \mid \mathbf x) \int q(\mathbf z_{t-1} \mid \mathbf z_t \mid \mathbf x)  \log \left[\frac{\operatorname{Pr}\left(\mathbf{z}_{t-1} \mid \mathbf{z}_{t}, \mathbf{\phi}_{t}\right)}{q\left(\mathbf{z}_{t-1} \mid \mathbf{z}_{t}, \mathbf{x}\right)}\right]d \mathbf z_{t-1}d\mathbf z_t\\&=\sum_{t=2}^{T} \int q(\mathbf z_{t} \mid \mathbf x) \int q(\mathbf z_{t-1} \mid \mathbf z_t,\mathbf x)  \log \left[\frac{\operatorname{Pr}\left(\mathbf{z}_{t-1} \mid \mathbf{z}_{t}, \mathbf{\phi}_{t}\right)}{q\left(\mathbf{z}_{t-1} \mid \mathbf{z}_{t}, \mathbf{x}\right)}\right]d \mathbf z_{t-1}d\mathbf z_t\\&=\sum_{t=2}^{T} \int q(\mathbf z_{t} \mid \mathbf x)\left[-D_{KL}[q(\mathbf z_{t-1} \mid \mathbf z_t, \mathbf x) \mid \mid Pr(\mathbf z_{t-1} \mid \mathbf z_t, \mathbf \phi_t)]\right] d\mathbf z_t\\&=-\sum^{T}_{t=2}\mathbb{E}_{q(\mathbf z_t \mid \mathbf x)}\left[D_{KL}[q(\mathbf z_{t-1} \mid \mathbf z_t, \mathbf x) \mid \mid Pr(\mathbf z_{t-1} \mid \mathbf z_t, \mathbf \phi_t)]\right]\end{aligned}
$$  

KL-divergence 의 정의는 아래와 같다. 위의 식 정리에서 KL-divergence 로 표현하는 부분에 이 정의의 연속형 부분이 적용되었다.  

![Untitled](https://github.com/dhkwon03/dhkwon03.github.io/assets/83265598/181a8738-7c29-4a4c-ac43-877aff32d66c)  

ELBO 를 이렇게 수식적으로 정리해보았다. 우리가 지금까지 유도해 온 수식들 중 ELBO를 정리한 식에 나타나는 것들을 보자.  

$$
\operatorname{Pr}\left(\mathbf{x} \mid \mathbf{z}_{1}, \mathbf{\phi}_{1}\right)=\operatorname{Norm}_{\mathbf{x}}\left[\mathbf{f}_{1}\left[\mathbf{z}_{1}, \mathbf{\phi}_{1}\right], \sigma_{1}^{2} \mathbf{I}\right]\\
\operatorname{Pr}\left(\mathbf{z}_{t-1} \mid \mathbf{z}_{t}, \mathbf{\phi}_{t}\right)  =\operatorname{Norm}_{\mathbf{z}_{t-1}}\left[\mathbf{f}_{t}\left[\mathbf{z}_{t}, \mathbf{\phi}_{t}\right], \sigma_{t}^{2} \mathbf{I}\right] \\
q\left(\mathbf{z}_{t-1} \mid \mathbf{z}_{t}, \mathbf{x}\right) =\operatorname{Norm}_{\mathbf{z}_{t-1}}\left[\frac{\left(1-\alpha_{t-1}\right)}{1-\alpha_{t}} \sqrt{1-\beta_{t}} \mathbf{z}_{t}+\frac{\sqrt{\alpha_{t-1}} \beta_{t}}{1-\alpha_{t}} \mathbf{x}, \frac{\beta_{t}\left(1-\alpha_{t-1}\right)}{1-\alpha_{t}} \mathbf{I}\right]
$$  

두 normal distribution 사이의 KL-divergence 는 아래와 같다.   

![Untitled](https://github.com/dhkwon03/dhkwon03.github.io/assets/83265598/4bad9425-145e-4fb3-b53a-ce9fbb7860a3)  

이것을 ELBO에서 KL divergence 항에 적용하면 parameter $\mathbf \phi\_t$ 에 depend 하지 않는 항을 C (constant) 로 놓을 수 있고 두 normal distribution 의 mean 의 차이의 제곱으로 정리된다. (closed form 으로 정리됨)  

$$
\begin{aligned}
&D_{K L}\left[q\left(\mathbf{z}_{t-1} \mid \mathbf{z}_{t}, \mathbf{x}\right) \ \mid \operatorname{Pr}\left(\mathbf{z}_{t-1} \mid \mathbf{z}_{t}, \mathbf{\phi}_{t}\right)\right]\\&= \frac{1}{2 \sigma_{t}^{2}}\|\frac{\left(1-\alpha_{t-1}\right)}{1-\alpha_{t}} \sqrt{1-\beta_{t}} \mathbf{z}_{t}+\frac{\sqrt{\alpha_{t-1}} \beta_{t}}{1-\alpha_{t}} \mathbf{x}-\mathbf{f}_{t}\left[\mathbf{z}_{t}, \mathbf{\phi}_{t}\right]\|^{2}+C
\end{aligned}
$$  

### 4.3 Diffusion loss function

우리는 diffusion model의 학습을 통해 parameter $\mathbf \phi\_{1...T}$ 를 조정하여 ELBO 를 최대화하는 것이 목표이다. ELBO에 (-1) 을 곱하고 expectation 은 dataset sample 에 대해 모두 합하는 방식으로 근사하여 최종적으로 loss function을 정의한다. 우리는 이 loss function 을 학습을 통해 “최소화” 하는 것이 목표이다. (ELBO에 -1 을 곱해서 최소화하는 것으로 바뀜)  

Loss function 은 아래와 같다.  

$$
\begin{aligned}
L\left[\phi_{1 \ldots T}\right]= & \sum_{i=1}^{I} \overbrace{\left(-\log \left[\operatorname{Norm}_{\mathbf{x}_{i}}\left[\mathbf{f}_{1}\left[\mathbf{z}_{i 1}, \mathbf{\phi}_{1}\right], \sigma_{1}^{2} \mathbf{I}\right]\right]\right.}^{\text {reconstruction term }} \\
& +\sum_{t=2}^{T} \frac{1}{2 \sigma_{t}^{2}}\left\| \underbrace{\frac{1-\alpha_{t-1}}{1-\alpha_{t}} \sqrt{1-\beta_{t}} \mathbf{z}_{i t}+\frac{\sqrt{\alpha_{t-1}} \beta_{t}}{1-\alpha_{t}} \mathbf{x}_{i}}_{\text {target, mean of } q\left(\mathbf{z}_{t-1} \mid \mathbf{z}_{t}, \mathbf{x}\right)}-\underbrace{\mathbf{f}_{t}\left[\mathbf{z}_{i t}, \mathbf{\phi}_{t}\right]}_{\text {predicted } \mathbf{z}_{t-1}}\right\|^2)
\end{aligned}
$$  

i 는 주어진 각 data 들의 index 를 의미한다. 즉 $\mathbf x\_i$ 는 i 번째 data point 이며 $\mathbf z\_{it}$ 는 i 번째 data point 의 step t 에서의 latent variable 이다.  

즉, “step t 에서 t-1의 latent variable 을 생성할 때”, ground truth (training을 할 때 정답으로 주어지는 data 라 보면 된다) 인 noise 가 없는 data $\mathbf x$ 가 주어졌을 때 $q(\mathbf z\_{t-1} \mid \mathbf z\_t , \mathbf x)$ 의 mean과 neural network ($\mathbf f\_t[\cdot ])$ 으로 예측된 $\mathbf z\_{t-1}$ 의 차이를 최소화하는 것이 곧 loss 를 최소화하는 것이 된다. 이 loss function 을 바탕으로 training 을 한다.  

## 5. Reparameterization of loss function

chapter 4 에서 유도한 loss function 으로 mean을 예측하며 training 할 수 있지만 reparameterization 을 통해 loss function 을 현재의 latent variable 을 만들기 위해 원래의 data example에 추가한 noise 의 관점으로 변형할 수 있다. 즉, training의 관점을 noise $\mathbf \epsilon$  을 예측하는 것으로 변형시킬 수 있으며 이 방식이 성능이 더 좋다고 말하고 있다. 필자도 정확히 왜 reparameterization 을 했을 때 성능이 더 좋은지 이유는 모르겠다.   

참고; (Diffusion 과는 전혀 관련이 없음) VAE 에서는 reparameterization 을 사용하는 이유가 random node 를 바꿈으로써  backpropagation 을 용이하게 한다는 것이다. 아래 그림을 참고하면 된다.  

![Untitled](https://github.com/dhkwon03/dhkwon03.github.io/assets/83265598/4c7c0550-818f-4cae-873a-8a68c2eae431)  

DDPM 논문을 보면 reparameterization 에 대해 조금 더 설명이 되어있다.   

DDPM 논문은 처음 Diffusion 이라는 개념을 제안한 “Deep Unsupervised Learning using Nonequilibrium Thermodynamics” 이라는 논문에서의 비평형 열역학에서 아이디어를 얻고 diffusion model 과 denoising score matching (denoising 하면서 loss 를 최소화하는 것을 의미하는 것 같다) 사이의 connection 을 제안한다고 말한다. 또한, 계속 Langevin dynamics (물리학에서 분자 시스템의 동역학을 수학적으로 모델링한 것이라 한다. 필자도 잘 모름) 와 비교를 하며 DDPM 에서 제안하는 모델의 정당성을 주장하고 있는 것 같다. reparametrization 을 사용하면 mean이 아니라 noise 를 예측하게 된다고 하였는데 예측한 noise 를 바탕으로 data 를 sampling 하는 과정 (추후에 설명한다) 이 Langevin dynamics 에서 예측한 $\mathbf \epsilon$  을 data density 의 learned gradient 로 보면 닮아있다고 한다. 또한, reparametrization 을 통해 나온 최종 loss function (추후에 유도한다) 이 Langevin-like 한 reverse process와 닮아있다고 한다. 정리하면, 필자는 이 논문에서 reparameterization 을 통해 $\mathbf \epsilon$  을 예측하도록 하는 방식이 Langevin dynamics 와 닮아있으며 diffusion model 의 variational bound (= ELBO) 가 denoising score matching 과 닮아있는 형태로 단순화된 것이라고 이해했다.  

DDPM 논문을 보면 reparameterization 을 해서 $\mathbf \epsilon$  을 예측하는 방식과 mean 을 예측하는 방식 각각에 대해 실험을 하여 performance 를 비교하였다. 실험적인 접근으로도 reparameterization 이 맞다는 것을 증명한 셈이며 직관적으로는 실험적으로 좋으니까 좋다고 이해할 수 있겠다.  

### 5.1 Reparameterization of target (mean of $q(\mathbf z_{t-1} \mid \mathbf z_t , \mathbf x)$)

loss function 에서 $q(\mathbf z\_{t-1} \mid \mathbf z\_t , \mathbf x)$ 의 mean 에 해당한다고 했던 부분 (target) 을 reparametrization 한다.   

원래의 diffusion kernel은 아래와 같다.  

$$
\mathbf{z}_{t}=\sqrt{\alpha_{t}} \cdot \mathbf{x}+\sqrt{1-\alpha_{t}} \cdot \mathbf{\epsilon}
$$  

이걸 $\mathbf x$ 에 대해 정리하면 다음과 같다.   

$$
\mathbf{x}=\frac{1}{\sqrt{\alpha_{t}}} \cdot \mathbf{z}_{t}-\frac{\sqrt{1-\alpha_{t}}}{\sqrt{\alpha_{t}}} \cdot \mathbf{\epsilon}
$$  

이걸 loss function의 target 에 대입하면 아래와 같다. ($\alpha\_t$ 의 정의에 의해서 $\sqrt{\alpha\_t} / \sqrt{\alpha\_{t-1}}=\sqrt{1-\beta\_t}$ 임을 두번째 줄에서 세번째 줄로 갈 때 사용함)  

$$
\begin{aligned}& \frac{\left(1-\alpha_{t-1}\right)}{1-\alpha_{t}} \sqrt{1-\beta_{t}} \mathbf{z}_{t}+\frac{\sqrt{\alpha_{t-1}} \beta_{t}}{1-\alpha_{t}} \mathbf{x} \\&= \frac{\left(1-\alpha_{t-1}\right)}{1-\alpha_{t}} \sqrt{1-\beta_{t}} \mathbf{z}_{t}+\frac{\sqrt{\alpha_{t-1}} \beta_{t}}{1-\alpha_{t}}\left(\frac{1}{\sqrt{\alpha_{t}}} \mathbf{z}_{t}-\frac{\sqrt{1-\alpha_{t}}}{\sqrt{\alpha_{t}}} \mathbf{\epsilon}\right) \\&= \frac{\left(1-\alpha_{t-1}\right)}{1-\alpha_{t}} \sqrt{1-\beta_{t}} \mathbf{z}_{t}+\frac{\beta_{t}}{1-\alpha_{t}}\left(\frac{1}{\sqrt{1-\beta_{t}}} \mathbf{z}_{t}-\frac{\sqrt{1-\alpha_{t}}}{\sqrt{1-\beta_{t}}} \mathbf{\epsilon}\right)\\& =\left(\frac{\left(1-\alpha_{t-1}\right) \sqrt{1-\beta_{t}}}{1-\alpha_{t}}+\frac{\beta_{t}}{\left(1-\alpha_{t}\right) \sqrt{1-\beta_{t}}}\right) \mathbf{z}_{t}-\frac{\beta_{t}}{\sqrt{1-\alpha_{t}} \sqrt{1-\beta_{t}}} \mathbf{\epsilon} \\
& =\left(\frac{\left(1-\alpha_{t-1}\right)\left(1-\beta_{t}\right)}{\left(1-\alpha_{t}\right) \sqrt{1-\beta_{t}}}+\frac{\beta_{t}}{\left(1-\alpha_{t}\right) \sqrt{1-\beta_{t}}}\right) \mathbf{z}_{t}-\frac{\beta_{t}}{\sqrt{1-\alpha_{t}} \sqrt{1-\beta_{t}}} \mathbf{\epsilon} \\
& =\frac{\left(1-\alpha_{t-1}\right)\left(1-\beta_{t}\right)+\beta_{t}}{\left(1-\alpha_{t}\right) \sqrt{1-\beta_{t}}} \mathbf{z}_{t}-\frac{\beta_{t}}{\sqrt{1-\alpha_{t}} \sqrt{1-\beta_{t}}} \mathbf{\epsilon}\\
& =\frac{1\cdot (1-\beta_t)+\beta_t-\alpha_{t-1}(1-\beta_t)}{\left(1-\alpha_{t}\right) \sqrt{1-\beta_{t}}} \mathbf{z}_{t}-\frac{\beta_{t}}{\sqrt{1-\alpha_{t}} \sqrt{1-\beta_{t}}} \mathbf{\epsilon} \\
& =\frac{1-\alpha_{t}}{\left(1-\alpha_{t}\right) \sqrt{1-\beta_{t}}} \mathbf{z}_{t}-\frac{\beta_{t}}{\sqrt{1-\alpha_{t}} \sqrt{1-\beta_{t}}} \epsilon \\
& =\frac{1}{\sqrt{1-\beta_{t}}} \mathbf{z}_{t}-\frac{\beta_{t}}{\sqrt{1-\alpha_{t}} \sqrt{1-\beta_{t}}} \mathbf{\epsilon}\end{aligned}
$$  

7번째 줄에서 8번째 줄로 갈 때 $\alpha\_{t-1}(1-\beta\_t)=\alpha\_t$ 임을 고려하였다. ($\alpha\_t$ 의 정의에 의해서 성립)   

이를 loss function 에 대입하면,  

$$
\begin{aligned}
L\left[\mathbf{\phi}_{1 \ldots T}\right]= & \sum_{i=1}^{I}-\log \left[\operatorname{Norm}_{\mathbf{x}_{i}}\left[\mathbf{f}_{1}\left[\mathbf{z}_{i 1}, \mathbf{\phi}_{1}\right], \sigma_{1}^{2} \mathbf{I}\right]\right] \\
& +\sum_{t=2}^{T} \frac{1}{2 \sigma_{t}^{2}}\left\| \left(\frac{1}{\sqrt{1-\beta_{t}}} \mathbf{z}_{i t}-\frac{\beta_{t}}{\sqrt{1-\alpha_{t}} \sqrt{1-\beta_{t}}} \mathbf{\epsilon}_{i t}\right)-\mathbf{f}_{t}\left[\mathbf{z}_{i t}, \mathbf{\phi}_{t}\right]\right\|^{2}
\end{aligned}
$$  

### 5.2 Reparameterization of network ($\mathbf f_t[\mathbf z\_t, \mathbf \phi\_t]$)

다음으로 loss function 에서 network 에 해당하는 $\mathbf f_t[\mathbf z\_t, \mathbf \phi\_t]$ 를 reparameterization 한다. 우리는 이 network 항을 최대한 target (5.1 참조) 을 따르도록 (비슷하도록) 만드는 것이 목표다. 그래서 $\mathbf z_t$ 를 생성하기 위해 $\mathbf x$ 에 첨가된 noise $\epsilon$ 을 예측하는 neural network 를 $\mathbf g_t[\mathbf z\_t, \mathbf \phi\_t]$ 라 하면 $\mathbf f\_t[\mathbf z\_t, \mathbf \phi\_t]$ 는 다음과 같이 표현가능하다.  

$$
\mathbf{f}_{t}\left[\mathbf{z}_{t}, \phi_{t}\right]=\frac{1}{\sqrt{1-\beta_{t}}} \mathbf{z}_{t}-\frac{\beta_{t}}{\sqrt{1-\alpha_{t}} \sqrt{1-\beta_{t}}} \mathbf{g}_{t}\left[\mathbf{z}_{t}, \phi_{t}\right]
$$  

대입하면,  

$$
\begin{aligned}
& L\left[\mathbf{\phi}_{1 \ldots T}\right]= \\
& \quad \sum_{i=1}^{I}(-\log \left[\operatorname{Norm}_{\mathbf{x}_{i}}\left[\mathbf{f}_{1}\left[\mathbf{z}_{i 1}, \mathbf{\phi}_{1}\right], \sigma_{1}^{2} \mathbf{I}\right]\right] \\& +\sum_{t=2}^{T} \frac{\beta_{t}^{2}}{\left(1-\alpha_{t}\right)\left(1-\beta_{t}\right) 2 \sigma_{t}^{2}}\left\| \mathbf{g}_{t}\left[\mathbf{z}_{i t}, \mathbf{\phi}_{t}\right]-\mathbf{\epsilon}_{i t}\right\|^{2})
\end{aligned}
$$  

normal distribution 의 probability density function (pdf) $f(x) = \frac{1}{\sigma\sqrt{2\pi}} 
  \exp\left( -\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^{2}\,\right)$ 에 log 를 씌우면 constant $C\_i$ 와 $(\mathbf x-\mu )^2/(2\sigma^2)$ 으로 정리가 되는데 이를 이용하여 위의 loss 식에서 첫번째 항을 정리한다.  

$$
\begin{aligned}
& L\left[\mathbf{\phi}_{1 \ldots T}\right]
\\&=\sum_{i=1}^{I} \frac{1}{2 \sigma_{1}^{2}}\left\| \mathbf{x}_{i}-\mathbf{f}_{1}\left[\mathbf{z}_{i 1},\mathbf{\phi}_{1}\right]\right\|^{2}\\&+\sum_{t=2}^{T} \frac{\beta_{t}^{2}}{\left(1-\alpha_{t}\right)\left(1-\beta_{t}\right) 2 \sigma_{t}^{2}}\left\| \mathbf{g}_{t}\left[\mathbf{z}_{i t}, \mathbf{\phi}_{t}\right]-\mathbf{\epsilon}_{i t}\right\|^{2}+C_{i}
\end{aligned}
$$  

$\mathbf{x}=\frac{1}{\sqrt{\alpha_{t}}} \cdot \mathbf{z}\_{t}-\frac{\sqrt{1-\alpha\_{t}}}{\sqrt{\alpha\_{t}}} \cdot \mathbf{\epsilon}$, $\mathbf{f}\_{1}\left[\mathbf{z}\_{1}, \phi\_{1}\right]=\frac{1}{\sqrt{1-\beta\_{1}}} \mathbf{z}\_{1}-\frac{\beta\_{1}}{\sqrt{1-\alpha\_{1}} \sqrt{1-\beta\_{1}}} \mathbf{g}\_{1}\left[\mathbf{z}\_{1}, \phi\_{1}\right]$ 을 대입하여 정리하면,  

$$
\begin{aligned}
&\frac{1}{2 \sigma_{1}^{2}}\left\| \mathbf{x}_{i}-\mathbf{f}_{1}\left[\mathbf{z}_{i 1}, \mathbf{\phi}_{1}\right]\right\|^{2}\\&=\frac{1}{2 \sigma_{1}^{2}}\left\| \frac{\beta_{1}}{\sqrt{1-\alpha_{1}} \sqrt{1-\beta_{1}}} \mathbf{g}_{1}\left[\mathbf{z}_{i 1}, \mathbf{\phi}_{1}\right]-\frac{\beta_{1}}{\sqrt{1-\alpha_{1}} \sqrt{1-\beta_{1}}} \mathbf{\epsilon}_{i 1}\right\|^{2}
\end{aligned}
$$  

정리한 loss function 은 constant 를 버리면,   

$$
L\left[\mathbf{\phi}_{1 \ldots T}\right]=\sum_{i=1}^{I} \sum_{t=1}^{T} \frac{\beta_{t}^{2}}{\left(1-\alpha_{t}\right)\left(1-\beta_{t}\right) 2 \sigma_{t}^{2}}\left\| \mathbf{g}_{t}\left[\mathbf{z}_{i t}, \mathbf{\phi}_{t}\right]-\mathbf{\epsilon}_{i t}\right\|^{2}
$$  

참고; DDPM 논문에서는 data scaling 이라는 것을 소개한다. $\mathbf z\_1$ 에서 $\mathbf x$ 로 mapping 할 때의 확률분포를 논문에서 임의로 정의해본 것이다. 그래서 0~255 의 pixel value 를 [-1, 1] 로 linear 하게 scaling 하고 이를 time step 1 에 depend 하는 Gaussian distribution 에 1/255 간격으로 discrete 하게 확률분포를 mapping 하는 방식이다. 이외에도 여러가지 추가적인 내용이 있는데 이는 논문을 참조하기 바란다. 근데 중요한건 결국에는 이거 사용안하고 simplified training objective (아래에서 설명하는데 scaling factor를 제거한 최종적인 loss function 형태이다) 로 그냥 t=1 부터 T 까지 training 하는게 더 sampling quality 이 좋다고 하여 그냥 사용 안한 것 같다. (자세히는 이해 못했다)  

제곱 항 앞에 곱해진 scaling factor 는 t가 작을수록 값이 커진다. t 가 작은 경우는 noise 가 얼마 없는 경우인데 이러한 case 에 가중치가 더 많아지면 정작 denoising 이 더 어려운 t가 큰 경우에 대한 학습이 잘 안 이루어질 수 있다. 따라서, scaling factor를 없앰으로써 t가 큰 경우에 가중치를 더 주는 것이라 생각할 수 있다. 실제로 실험을 해봤을 때 scaling factor 가 없는 경우가 sample의 quality 가 더 좋았다고 한다. scaling factor를 버리고 diffusion kernel 을 대입한 최종 loss function 은 아래와 같다.  

$$
\begin{aligned}
L\left[\mathbf{\phi}_{1 \ldots T}\right] & =\sum_{i=1}^{I} \sum_{t=1}^{T}\left\| \mathbf{g}_{t}\left[\mathbf{z}_{i t}, \mathbf{\phi}_{t}\right]-\mathbf{\epsilon}_{i t}\right\|^{2} \\
& =\sum_{i=1}^{I} \sum_{t=1}^{T}\left\|\mathbf{g}_{t}\left[\sqrt{\alpha_{t}} \cdot \mathbf{x}_{i}+\sqrt{1-\alpha_{t}} \cdot \mathbf{\epsilon}_{i t}, \mathbf{\phi}_{t}\right]-\mathbf{\epsilon}_{i t}\right\|^{2}
\end{aligned}
$$  

(주의! $\epsilon\_{it}$ 는 reparameterization 을 통해 변형된 loss function 에서 예측한다고 한 noise 가 아니다. 위 식에서 $\epsilon\_{it}$ 는 $\mathbf z\_{it}$ 를 표현하는 식에서 나온 것이며 normal distribution 에서 random 하게 sampling 되는 값이다. 우리가 training 을 통해 예측한다고 하는 noise 는 neural network $\mathbf g\_t[\cdot ]$ 를 통해 예측되는 값이다)   

우리는 diffusion kernel 을 정의하고 decoder의 neural network 를 정의한 후 이 loss function 을 얻기 위해 여기까지 달려왔다.  

## 6. Implementation

위에서 유도한 loss function 을 가지고 아래와 같은 training 과 sampling 을 하는 것이 diffusion model의 기본적인 컨셉이다.  

![Untitled](https://github.com/dhkwon03/dhkwon03.github.io/assets/83265598/3624542a-3817-4e34-bec9-9c4985f3a590)  

training 방식이나 sampling 방식이 굉장히 간단하며 random 하게 sampling 하는 $\mathbf \epsilon$ 덕분에 dataset 에 있는 $\mathbf x\_i$ 를 training 에 여러번 재사용 가능하다는 장점이 있다. 하지만, 위의 sampling 방식은 수많은 t에 대해 $\mathbf g\_t[\mathbf z\_t, \mathbf \phi\_t]$ 를 순차적으로 processing 해야 한다는 단점이 있다. 이는 시간을 굉장히 많이 잡아먹기 때문에 치명적인 단점이다.  

$\mathbf g_t[\mathbf z\_t, \mathbf \phi\_t]$ neural network 의 모델은 U-Net 을 기본적으로 사용한다. (U-Net 은 image-to-image mapping 에 자주 사용됨) 이미지 생성에서 어느정도의 성능이 나와주려면 T=1000 step 정도가 요구되는데 image 의 경우 dimension 이 비교적 크기 때문에 모든 time step 에 대한 U-Net 을 저장하고 training 하는 것은 굉장히 비효율적이다. 이 때문에 “모든 time step 을 하나의 U-Net 으로 training” 하고 “time step 을 input 으로 하는 vector 를 U-Net 의 channel 수에 맞춰서 각 spatial position에 offset 혹은 scale 로 적용” 하는 방식을 취한다.  

같은 형태의 diffusion kernel 을 사용하고 같은 형태의 forward process 를 사용하면 어떠한 방식의 diffusion 에도 앞에서 유도한 loss function을 사용할 수 있다. 단지 forward process 를 진행시키는 방식과 예측된 noise $\mathbf g\_t[\mathbf z\_t, \mathbf \phi\_t]$ 로 $\mathbf z\_{t-1}$ 로 부터 $\mathbf z\_t$ 를 예측하는 방식이 달라지면 다른 방식의 diffusion model 이 된다. 예를 들어 denoising diffusion implicit models, accelerated sampling model 등이 있다.  

Conditional generation 이라는 것이 있다. 지금까지 살펴본 diffusion model 은 dataset 에 label 이 없을 때의 이야기지만 data에 각각 label 이 붙어있고 이를 활용하는 diffusion model 이 있다. 크게 classifier guidance, classifier-free guidance 로 나뉜다.  

classifier guidance 는 $\mathbf z\_{t-1}$ 로 부터 $\mathbf z\_t$ 를 예측하는 식에서 classifier ($\mathbf z\_t$ 에 대한 label c 의 확률분포) 의 gradient 에 depend 하는 항이 추가되는 것이다. 즉, classifier 라는 객체가 추가되어 모든 time step 에서 share 되며 U-Net 처럼 time step 을 같이 input 으로 받는다.  

classifier-free guidance 는 기존 neural network 에 class label 정보를 추가하는 것이다. ( $\mathbf g\_t[\mathbf z\_t, \mathbf \phi\_t, c]$ ) U-Net 의 layer 에 label c 에 기반한 embedding 을 추가하는 것이다 (U-Net 에 time step vector 를 적용하는 것과 비슷하다) 자세한 것은 관련 논문을 보면 된다. (다른 post 에 정리할 수도 있음)  

이 밖에도 더 고품질의 이미지를 생성하기 위해 variance 도 예측하는 network 를 추가하거나, forward process 에서 noise schedule 을 변경하거나 diffusion model 을 cascade 하는 등의 방법이 있다.  

## 7. Conclusion

Diffusion model 의 기본 개념과 forward process, reverse denoising process 를 정의하고 ELBO를 유도하여 loss function 까지 유도하였다. 또한, U-Net 을 통해 training 하고 최종 data 를 sampling 하는 방식과 diffusion model의 응용까지 살펴보았다.  

## Comment

DDPM 논문에는 세부적이고 재밌는 추가 내용도 많다. Progressive coding, interpolation 등이 설명되어있는데 자세한 건 논문을 참고하라.  

수식을 전부 레이텍으로 치려니 정말 힘들고 글 쓰는 시간이 너무 오래 걸린다. 대부분 이미지 캡처로 대체하거나 손글씨로 대체할 예정이다.  

본 post 는 9생 7월 목표 중 필자의 목표인 ‘학술 블로그 포스트 5개 이상 업로드’ 의 일환임을 알린다.  

## Reference

- Understanding Deep Learning (Simon J.D. Prince, The MIT Press, 2023)  
- [[용어정리] reparameterization trick (tistory.com)](https://heygeronimo.tistory.com/40)
