---
title: '논문리뷰 (paper review); Simba : Scaling Deep Learning Inference with Multi-chip-module
  based Architecture'
categories:
- AI_accelerator
- Multi_chip
tags:
- paper_review
---

엔비디아에서 만든 Simba 라는 multi-chip 딥러닝 가속 아키텍쳐에 관한 논문이다. Multi-chip과 AI 가속기의 개념이 결합되어 있다는 점이 특징이다. 논문 읽으면서 정리한 노트를 사진으로 첨부했다. 
# Introduction (이론적 배경 및 problem suggestion)
![simba_1](https://github.com/dhkwon03/dhkwon03.github.io/assets/83265598/7d8c85a0-d84f-4b21-bdf3-9c225d510962)
# Structure of Simba
![simba_2](https://github.com/dhkwon03/dhkwon03.github.io/assets/83265598/f881a304-d01b-4767-b733-1a26cc1ded7b)
![simba_3](https://github.com/dhkwon03/dhkwon03.github.io/assets/83265598/0a160812-5b44-42c7-a41d-e6d5c7d8b0ee)
# Characterization: Sensitivity
![simba_4](https://github.com/dhkwon03/dhkwon03.github.io/assets/83265598/56dc72ef-9e6f-417b-9529-2453585e5f8b)
![simba_5](https://github.com/dhkwon03/dhkwon03.github.io/assets/83265598/cdac8da2-6fb4-4606-b069-e6f5f9c7c7d6)
# Non-uniform tiling methods of Simba
(아래 사진에서 Non-uniform tiling methods 라 되어 있는 곳 이전 부분은 위의 chapter에 속함)
![simba_6](https://github.com/dhkwon03/dhkwon03.github.io/assets/83265598/daa98163-12ac-4bc7-b0c8-8ebec6005868)
![simba_7](https://github.com/dhkwon03/dhkwon03.github.io/assets/83265598/039bf157-a83e-41b9-b166-71ba48ff4b33)
# Implementation
Simba의 실제 implementation과 그에 대한 data를 제시한 논문은 아래의 논문이다. 아래 논문은 특별히 정리할 건 없고 implementation을 함축적으로 정리해놓은 짧은 논문이다. 
![simba_8](https://github.com/dhkwon03/dhkwon03.github.io/assets/83265598/a8bf86ec-d186-4e9b-9fae-8c984e908353)
