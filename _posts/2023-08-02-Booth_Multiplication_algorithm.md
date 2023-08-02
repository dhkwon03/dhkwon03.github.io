---
title: Booth Multiplication Algorithm
categories:
- AI_accelerator
tags:
- algorithm
---

AI accelerator에서 arithmetic operation을 수행하는 부분에서 사용되는 FMA (Fused Mulitplication & Adder)가 있다. AI accelerator에서는 AI 알고리즘 특성상 숫자를 곱한 후 모두 합하는 연산이 많이 일어나고 이러한 연산을 효율적으로 처리하기 위해 곱셈과 덧셈을 같이 효율적으로 처리하는 부분이 FMA 이다. 필자는 FMA에 관한 논문을 읽던 중 booth multiplication algorithm을 알게 되었고 이 개념을 몰랐기에 이렇게 정리하였다. 꼭 FMA가 아니더라도 다른 아키텍쳐에서도 숫자를 곱할 때 booth multiplication algorithm이 많이 사용된다. 숫자를 전부 하나하나 일일이 곱하는 것보다 booth multiplication algorithm을 사용하여 연산량을 줄이는 것이 효율적이기 때문이다. 
# 일반적인 booth multiplication algorithm 원리
![booth_multiplication_algorithm_1](https://github.com/dhkwon03/dhkwon03.github.io/assets/83265598/e64c5197-b39e-478f-918b-8a41e17275c5)
# radix-4 booth algorithm
앞에서 나온 것은 radix-2 booth algorithm이다. 아래 사진처럼 radix-4 booth algorithm도 있으며 같은 원리로 8진법, 16진법으로 바꾼 radix-8, radix-16 booth algorithm도 있다. radix-n booth algorithm에서 n이 증가하면 입력 숫자가 어떤지에 따라 연산량이 증가할수도 감소할수도 있다. 매번 상황에 따라 다르기 때문에 해당 아키텍쳐에 입력으로 들어오는 숫자들의 특성을 파악하고 몇 진법의 booth algorithm을 선택할 것인가는 design choice 이다. 
![booth_multiplication_algorithm_2](https://github.com/dhkwon03/dhkwon03.github.io/assets/83265598/e1bce30d-9a0c-49fa-abb4-e4a4467b6dd4)
