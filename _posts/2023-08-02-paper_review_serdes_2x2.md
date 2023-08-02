---
title: 논문리뷰 (paper review); An Efficient Design Framework for 2x2 CNN Accelerator
  Chiplet Cluster with SerDes Interconnects
categories:
- AI_accelerator
- chiplet
tags:
- paper_review
---

chiplet과 SerDes interconnect를 활용한 CNN 가속기를 만들었다는 논문이다. chiplet 구조나 더 빠른 CNN 가속을 위한 특별한 아이디어는 없고 job scheduling을 좀 더 효율적으로 해서 2 by 2 chiplet에 잘 분배할 수 있는 알고리즘을 설계했다는 것이 핵심 아이디어라 할 수 있다. 
논문 읽으면서 정리한 내용을 사진으로 첨부했다. 
# Architectural Model Design
## A. Chiplet Configuration
![serdes_1](https://github.com/dhkwon03/dhkwon03.github.io/assets/83265598/f67184fd-28c6-4e8e-a8be-b95e5bda0915)
## B. Interconnection
![serdes_2](https://github.com/dhkwon03/dhkwon03.github.io/assets/83265598/768bd3d4-33df-41d3-a45e-e875b1acba6a)
# Application Model
## A. Task graph generator
![serdes_3](https://github.com/dhkwon03/dhkwon03.github.io/assets/83265598/7a3bf89d-1e24-49b5-9617-7c1f73c17b5c)
![serdes_4](https://github.com/dhkwon03/dhkwon03.github.io/assets/83265598/ef8c8f78-d3d2-4656-8a59-ac143c2d2673)
## B. Scheduling Algorithms
![serdes_5](https://github.com/dhkwon03/dhkwon03.github.io/assets/83265598/78609cb6-2b51-4dc3-9a9f-4b45333dbba1)
# Experiment results
![serdes_6](https://github.com/dhkwon03/dhkwon03.github.io/assets/83265598/dfc33419-66c4-4eb2-9aec-6c0042f1f30d)
