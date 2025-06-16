---
title: '논문 리뷰 (paper review); FlashAttention: Fast and Memory-Efficient Exact Attention
  with IO-Awareness'
tags:
- concept
- paper_review
categories:
- AI
---

군에 있을 때 24년 7월 쯤에 읽었던 논문입니다. 열심히 pdf 로 필기해놓았는데 그냥 안 올리기는 아까워서 기록용으로 올려놓습니다.
2022년 6월에 나온 FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness 논문입니다. (군에서 읽었을 때는 센세이셔널한 느낌을 받았는데 꽤나 오래된 논문이네요...)
대략 핵심만 정리하고 나머지는 제가 필기해놓았던 걸 첨부해놓겠습니다.

## 1. Flashattention intro
Transformer 는 sequence 길이가 길어질수록 memory bounded 한 알고리즘이 된다. 특히, transformer 에서 핵심적인 operation 인 self-attention 의 경우 sequence length 의 quadratic scale (제곱 scale) memory complexity 를 가지기 때문에 memory IO 를 빠르고 효율적으로 하는 것이 굉장히 중요하다는 문제가 있다. (이 문제는 지금까지도 이어지고 있다.)
따라서, 본 논문은 attention algorithm 을 실행할 때 GPU memory hierarchy 상에서 memory read & write 를 최대한 줄임으로써 효율적이고 빠른 attention 연산을 가능하게 하는 "IO-aware" algorithm 을 제안한다. 기존에는 approximation 등을 사용했지만 본 논문에서 제안하는 새로운 알고리즘은 원래의 attention 과 완전히 동일한 결과를 도출하면서도 (exact) memory IO 를 획기적으로 감소시켰다는 점에서 의의가 있다.
"tiling" 개념을 도입하여 GPU 의 HBM과 on-chip SRAM 간의 read/write 횟수를 감소시켰다. 또한, block-sparse attention 도 FlashAttention 을 적용하여 획기적인 성능을 도출하였다.

## 2. Standard Attention Implementation
기존의 standard 한 attention 연산은 다음과 같이 정의된다. 

<img src="../../assets/images/250616_flashattention/original_formula.jpg" width="1000px" height="400px" title="Original Attention"/> 

$O(N^2)$ 의 memory 가 필요하며 HBM access 횟수도 sequence length N 에 quadratic 하다.
## 3. FlashAttention: tiling
이 부분이 논문에서 제안하는 핵심이다. tiling 을 해서 size 가 작은 block 으로 쪼개서 계산을 할 수 있기 때문에 한번에 block 을 on-chip SRAM 에 올려서 계산을 하고, 결과만 HBM 에 write 한다는 것이다. 결국 계산 과정에서 HBM 에 access 를 여러번 하던 것을 on-chip SRAM 에서 한 번에 계산할 수 있도록 하여 HBM access 를 줄인다는 것이다. 
 
<img src="../../assets/images/250616_flashattention/formula1.jpg" width="1000px" height="400px" title=""/> 
<img src="../../assets/images/250616_flashattention/formula2.jpg" width="1000px" height="400px" title=""/>
 
자세한 memory complexity 및 HBM access 횟수에 대한 유도는 아래 첨부된 필기에 있다.

## 4. 정리 노트
<img src="../../assets/images/250616_flashattention/1.jpg" width="1000px" height="400px" title=""/> 
<img src="../../assets/images/250616_flashattention/2.jpg" width="1000px" height="400px" title=""/> 
<img src="../../assets/images/250616_flashattention/3.jpg" width="1000px" height="400px" title=""/> 
<img src="../../assets/images/250616_flashattention/4.jpg" width="1000px" height="400px" title=""/> 
<img src="../../assets/images/250616_flashattention/5.jpg" width="1000px" height="400px" title=""/> 
<img src="../../assets/images/250616_flashattention/6.jpg" width="1000px" height="400px" title=""/> 
<img src="../../assets/images/250616_flashattention/7.jpg" width="1000px" height="400px" title=""/> 
<img src="../../assets/images/250616_flashattention/8.jpg" width="1000px" height="400px" title=""/> 
<img src="../../assets/images/250616_flashattention/9.jpg" width="1000px" height="400px" title=""/> 
<img src="../../assets/images/250616_flashattention/10.jpg" width="1000px" height="400px" title=""/> 
<img src="../../assets/images/250616_flashattention/11.jpg" width="1000px" height="400px" title=""/>
