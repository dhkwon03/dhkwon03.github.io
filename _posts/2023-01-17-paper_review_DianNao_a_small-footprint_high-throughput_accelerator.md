---
title: '논문리뷰 (Paper review); DianNao: A Small-Footprint High-Throughput Accelerator
  for Uniquitous Machine-Learning'
tags:
- paper_review
categories:
- AI_accelerator
---

최초의 AI 가속기 논문으로 유명한 논문이다. large-scale CNN과 DNN을 위한 accelerator 를 design 했는데, memory footprint를 중점적으로 줄여서 효율적이고 performance 가 높은 accelerator가 되도록 design 했다.
# Contribution
* A synthesized (place and route) accelerator design for large-scale CNNs and DNNs, the state-of-the-art machine-learning algorithms 
* the accelerator achieves high throughput in a small area, power and energy footprint (작은 area, 작은 power를 사용하면서 높은 퍼포먼스를 얻어냄) 
* 이전의 연구들은 computational primitives를 효율적으로 implement 하는 것에 초점을 맞추어 진행되었다면 본 연구는 memory behavior 에 초점을 맞추어 진행되었으며 memory transfer가 performance 와 energy 사용량에 미치는 영향도 모두 측정하였다. 
 
# Processor-based implementation of large NN 
* tiling 방법을 통해 main memory에 access 하는 횟수 최소화 
* chip 내부에 별도의 cache 사용하여 일정 크기만큼 layer에서 data를 잘라내어 caching 하는 것 (모든 conv, synapses, Input/Output Neurons, pooling layers 에 대해 적용)
* Classifier layers
	* tiling 하여 최종적으로 on chip 에서 계산하는 부분은 $T_n * T_i$
	* Fig 6. : 이전에는 DRAM에서 모든 matrix의 parameter를 불러와야 했지만, working memory set을 줄임으로써 working memory를 L1 cache 에 넣어서 요구되는 memory bandwidth (메모리 전송 속도) 를 줄일 수 있고 만약 weight matrix를 L2 cache 에 넣는다면 더욱 많이 bandwidth를 줄일 수 있다.
* Convolutional layer
	* Tile size를 $T_x * T_y * N_i$ ($N_i$ 가 depth)로 하여 tiling
	* kernel 은 작은 크기라서 on-chip 에 모두 들어감
	* for문으로 나타낸 figure 를 보면 depth 방향으로 먼저 탐색
	* stride가 작으면 kernel은 stride 만큼 옆으로 옮겨가기 때문에 겹치는 부분의 계산값은 재사용할 수 있다. 또한, depth가 너무 커서 cache에 안 들어가면 depth 또한 나눠서 (tiling 해서) 계산할 수 있다.
	* Fig 6 : convolution layer에 대해서는 input, output 모두 cache 에 가지고 있을 수 있다. CONV3 는 shared kernel의 경우이고, CONV5 는 private kernel의 경우인데 각 layer 마다 kernel 이 다른 경우 (즉, kernel 개수가 더 많음) 가 private kernel임. 요즘은 shared kernel의 경우만 있다. CONV3 를 보면 cache를 이용할 때 bandwidth가 굉장히 줄어든다.
* Pooling layer
	* tiling size 는 $T_x * T_y * N_i$
	* 마찬가지로 for문을 보면 비슷하게 tiling 하여 계산하는 것을 구현
	* kernel에 대한 weight 가 없기 때문에 그래프 (fig. 6)를 보면 synapse가 없기 때문에 memory bandwidth 가 애초에 작고, tiling 의 효과는 작다. 
 
# Accelerator for Large NN 

* NBin (input buffer for input neurons), NBout (output buffer for output neurons), SB (buffer for synaptic weights)
* 위의 세 개 buffer 가 NFU (Neural Functional Unit) 에 연결, CP (Control logic)에 연결
* NFU (Neural Functional Unit)
	* 2개 or 3개의 stage 로 분리 (multiplication - pooling layer에서는 없음, addition, sigmoid - pooling layer에서는 없음)
	* 16-bit truncated fixed-point multiplier 와 32-bit floating-point multiplier 간에 small accuracy tradeoff 만 있으므로 area와 power 효율이 좋은 16-bit fixed-point 사용 (현재는 fixed-point 를 쓰는 것이 매우 당연하게 받아들여지고 있으며 16-bit 가 아닌 더 작은 bit 수로 하기도 함)
	* NBin, NBout, SB, NFU-2 Registers storage 는 cache 가 아닌 ‘scratchpad (on-chip SRAM)’ 사용
		* cache는 general-purpose computer 에서는 좋지만 not optimal for accelerator. cache는 control logic 이 굉장히 많다. 하지만, 이 연구에서 연산이나 control이 단순하기 때문에 cache 대신 scratchpad 사용
		* scratchpad 는 사용자가 직접 무슨 데이터를 어디로 빼고 넣을지 명령 내려야 함 (다른 말로 software control cache 라고도 한다)
		* scratchpad 는 efficient storage, and both efficient and easy exploitation of locality because only a few algorithms have to be manually adapted.
* Split buffer (NBin, NBout, SB)
	* width : 각 buffer 에 대해 시간, energy 효율이 가장 좋은 appropriate read/write width로 조정한다
		* NBin 은 $T_i$ 개, NBout 은 $T_n$ 개, SB는 $T_i * T_n$ 개 의 working memory 들어가고 이들이 NFU 에 연결되어 있음
	* avoid conflict : highly associative cache 의 대안이 있으나 n-way cache는 fast read가 n ways/bank를 parallel 하게 읽는 방식이라 energy cost 가 너무 크다
* DMA
	* scratchpad 에서 spatial locality exploitation (locality 이용)을 위해 각 buffer에 하나씩 3개의 DMA (two load DMA, one store DMA for output)
	* DMA requests are issued to NBin in the form of “instructions” (described in 5.3.2)
	* requests are buffered in a separate FIFO associated with each buffer
	* DMA requests FIFOs enable to decouple the requests —> DMA requests can be preloaded in advance if there is enough buffer capacity. —> latency를 줄여준다
	* 연산을 하는 동안 DMA를 사용해서 미리 데이터를 로드 (prefatch) 할 수 있어서 latency를 줄일 수 있다.
* local transpose in NBin for pooling layers
* dedicated registers
	* NBin 에서 input neuron load 해서 partial sum compute 하고 NFU pipeline으로 내보낸 후 다시 매번 re-load 하는 것은 energy 효율에 안 좋다. “dedicated register”를 NFU-2 에 달아서 partial sum을 저장 (“pipeline register”)
* circular buffer
	* input neurons are split into chunks which fit in NBin, implement NBin as “circular buffer” —> naturally implemented by changing a register index
	* $T_n$ 개의 partial sum이 있으면 이를 memory에 다시 보내지 않고 NBout 에 rotate out
	* NBout 의 본래 목적인 final output neuron 과 충돌하는 role 이나 모든 input neuron 이 partial sum으로 integrated 되기 전까지는 NBout 은 idle —> 이 때 temporary storage buffer로 사용 
 
# Experimental methodology 
* C++ accelerator simulator 사용, custom cycle-accurate, bit-accurate
* Verilog -> synthesize (Synopsys Design Compiler) -> Synopsys ICC (place, routing) -> design simulation (VCS) -> Prime-time PX (power estimation)
* SIMD (Single Instruction Multiple Data,  하나의 명령어로 여러 값 동시에 계산하는 방식)와 비교함
	* baseline using GEM5 + McPAT combination
	* 4-issue superscalar x86 core, 128 bit SIMD unit @ 2GHz / L1 32KB, L2 2MB  both cache 8-way associative, 64-byte line 
 
# Experimental Result 
* $T_n=16$ (16 HW neurons, 16 synapses each) : 452 GOP/s (Giga fixed-point Operations per sec) @ 0.98 GHz
* NBin/NBout 에서 data reading 이 critical path —> 1.02 ns latency —> 후속 연구로 이 path 를 감소하는 방법 고안할 예정
* RAM capacity 44KB
* SIMD 보다 개선된 time
	* 1. appropriate combination of preloading and reuse in NBin and SB buffers
		* not implemented prefetcher in SIMD core, prefetcher in SIMD core will cancel performance boost
	* 2. control and scheduling overhead
		* tried  to minimize lost cycles in accelerator
		* NBout rotation without any pipeline stall
 
# Conclusion 
* implementation of large-scale layers
* exploiting the locality properties of such layers and by introducing storage structures custom designed to take advantage of localities, designed a machine-learning accelerator capable of high performance in a very small area footprint.
* speedup of 117.87x and an energy reduction of 21.08x over a 128-bit 2GHz SIMD core with normal cache hierarchy. layout of the design @65nm
* future work
	* improving accelerator behavior for short layers
	* alternating NFU to include some latest algorithmic improvements (Local Response Normalization)
	* reducing the impact of main memory transfers, investigating scalability (increasing $T_n$), and implementing training in HW
