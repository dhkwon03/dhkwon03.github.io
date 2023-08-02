---
title: 논문리뷰 (paper review); Attention Is All You Need
categories:
- AI
tags:
- paper_review
---

구글에서 2017 NIPS 에서 발표했고 당시 엄청난 센세이션을 일으켜 AI의 흐름을 바꿨다는 평을 받은 논문, Attention is all you need 이다. 기존에 seq2seq (sequence to sequence) model에서 사용되던 RNN 부분을 self-attention으로 대체하여 performance를 높였다는 논문이다. 아무래도 잘은 모르겠지만 RNN 부분이 꽤나 많은 overhead를 잡아먹었나 본데 이를 attention으로 대체하여 오직 attention 만을 사용하는 transformer model을 제시한 것이 주요 아이디어라 할 수 있다. (심지어 performance도 기존 seq2seq보다 transformer가 더 좋다는 것이 놀랍다) 
# Introduction & Background
![transformer_1](https://github.com/dhkwon03/dhkwon03.github.io/assets/83265598/f8cb8849-dbfe-4479-a765-e1af5bbfe183)
# Model Architecture
## encoder & decoder
![transformer_2](https://github.com/dhkwon03/dhkwon03.github.io/assets/83265598/bbcbd647-2112-4628-9fc5-162bb0e6cf63)
## Attention
![transformer_3](https://github.com/dhkwon03/dhkwon03.github.io/assets/83265598/a9aca6c8-16bb-4a09-ae2c-ab697b7391c7)
## Multi-head attention
![transformer_4](https://github.com/dhkwon03/dhkwon03.github.io/assets/83265598/eb5afa35-9c1c-489a-ac12-64899b30f187)
## Applications of attention in Transformer 
Transformer 안에서 사용되는 attention은 총 3가지의 종류가 있다. 
![transformer_5](https://github.com/dhkwon03/dhkwon03.github.io/assets/83265598/1e6d9b19-d853-4f23-8a2d-c830a960b075)
![transformer_6](https://github.com/dhkwon03/dhkwon03.github.io/assets/83265598/7f7ee369-a285-4e75-aa6a-7dfaca216956)
## Position-wise Feed-Forward Networks
![transformer_7](https://github.com/dhkwon03/dhkwon03.github.io/assets/83265598/2fa4c4ae-4761-4f72-9474-2f34d7700174)
## Padding Mask
![transformer_8](https://github.com/dhkwon03/dhkwon03.github.io/assets/83265598/e747bc0a-0e55-46a0-bce2-829e53ebfe34)
## 전체 Transformer 구조
![transformer_9](https://github.com/dhkwon03/dhkwon03.github.io/assets/83265598/6b76cad9-aa8e-4bb4-a04e-387a9ecb0cc9)
## Positional Encoding
![transformer_10](https://github.com/dhkwon03/dhkwon03.github.io/assets/83265598/f5a7bd94-8a32-4118-afa2-5ba64143dbdc)
# Pros of Self-attention
![transformer_11](https://github.com/dhkwon03/dhkwon03.github.io/assets/83265598/ee1e3902-f51a-4197-8a35-4bdb80508d29)
# Training
이 paper에서 training 한 방식이다. 
![transformer_12](https://github.com/dhkwon03/dhkwon03.github.io/assets/83265598/08ad9264-a9fc-49ce-a67f-ea020c81d1b5)
