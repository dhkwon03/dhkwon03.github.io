---
title: '논문리뷰 (Paper review); Eyeriss: An Energy-Efficient Reconfigurable Accelerator
  for Deep Convolutional Neural Networks'
tags:
- paper_review
categories:
- AI_accelerator
---

논문 제목  
Eyeriss : An Energy-Efficient Reconfigurable Accelerator for Deep Convolutional Neural Networks  

AI accelerator 하면 DianNao와 함께 대표적이고 가장 base가 되는 논문 중 하나이다.  
DianNao와는 다르게 tape-out을 하고 65nm 공정으로 실제 chip을 제작하여 test 했다는 것이 특징이다. 또한, 보편적인 CNN에 적용이 잘 되는 Chip 이라는 것이 Eyeriss의 강점이다.  
  
# Abstract
* accelerator for deep CNN
* optimizes for the energy efficiency of entire system including accelerator chip and off-chip DRAM for various CNN shapes by “reconfiguring the architecture”
* Minimize data movement energy cost for any CNN shape by using proposed processing dataflow, row stationary (RS), on a spatial architecture with 168 processing elements. RS reconfigures the computation mapping of given shape, optimizing energy efficiency by maximally reusing data locally, reducing expensive data movement (e.g. DRAM accesses)

# Introduction
* ‘dataflow’; support highly parallel compute paradigm while optimizing energy cost of data movement, exploiting data reuse in multilevel memory hierarchy & HW have to be reconfigurable to support different shapes
* data statistics -> compression & data adaptive processing can be applied to save both memory bandwidth and processing power
* implemented and fabricated CNN accelerator : Eyeriss; support high throughput CNN inference and optimizes for energy efficiency of entire sys. (including accelerator chip and off-chip DRAM) / reconfigurable to handle different CNN shapes, including square and nonsquare filters
* main features of Eyeriss
	* spatial architecture using array of 168 processing elements (PEs) that creates 4-level memory hierarchy, data movement exploit low-cost levels (PE scratch pads and inter-PE communication) minimizing data accesses to high-cost levels
	* Row Stationary (RS) dataflow reconfigures the spatial architecture to map computation of given CNN shape and optimize for best energy efficiency
	* network-on-chip (NoC) architecture; uses multicast & point-to-point single-cycle data delivery to support RS
	* Run-length compression (RLC) and PE data gating that exploit the statistics of zero data in CNNs
* ifmaps : input fmaps (feature maps)
* ofmaps : output fmaps
* psums : partial sums  
  
# System Architecture
* See Fig. 2
* two clock domains : core clock domain (for processing) & link clock domain for communication with off-chip DRAM (two domains CDC through asynchronous FIFO interface)
## core clock domain
	* spatial array of 168 PEs ($12 \times 14$ rectangle)
	* 108-kB GLB
	* RLC CODEC
	* ReLU module
* transfer data for computation; each PE communicate with its neighbor PEs or GLB (global buffer) through NoC (Network on chip), access local PE memory space (spads, scratchpads)
* memory hierarchy (energy per access); DRAM > GLB > inter-PE communication > spads
## System control (2 levels of control hierarchy)
	* top-level control
	1. traffic between the off-chip DRAM and GLB (through async. interface)
	2. traffic between GLB and PE array (through NoC)
	3. operation of RLC (run-length compression) CODEC and ReLU module
	* lower-level control
		* control logic in each PE (runs independently of each other)
		* 168 PEs are identical & run under same core clock, but each PE can start its own processing as soon as any fmaps or psums arrives
	* accelerator process CNN layer-by-layer
		* for each layer, loads ‘configuration bits’ -> reconfigure entire accelerator for filters and fmaps in certain shape
		* batches of ifmaps for same layer can be processed sequentially without further reconfiguration
		
# Energy-Efficient features  
## Energy-Efficient Dataflow; Row Stationary (RS)  
* data accesses to high-cost DRAM and GLB are minimized by maximizing reuse of data from low-cost spads and inter-PE communication  

### 1-D convolution primitive in a PE  
* divide into 1-D convolution primitives that can run in parallel
* each primitive operates on one row of filter weights and one row of ifmap values, generating one row of psums.
* psums from each primitives are ‘accumulated’ together to generate ofmap values
* PE can use local spads for convolutional data reuse & psum accumulation 
* required spad capacity depends only on filter row size (S)
	* row of filter weights : S
	* sliding window of ifmap values : S
	* psum accumulation : 1	
		
### 2-D convolution PE set
* See Fig. 4
* each row of filter is reused horizontally in PE array, each row of ifmap is reused diagonally, and rows of psum are accumulated vertically
* dimensions of PE set are determined by the filter and ofmap size of given layer (height of PE set = # of filter rows (R), width of PE set = # of ofmap rows (E) )

### PE set mapping
* strategy required to map PE sets onto the PE array
* map PE set using nearby PEs taking advantage of local data sharing and psum accumulation
* can be mapped to any group of PEs in array, but 2 exceptions
	* if PE set has more than 168 PEs; solved by ‘strip mining’ the 2-D convolution, only processes part of row of ofmap at a time ($R \times e$, $e \leq E$)
	* if PE set width is larger than 14 or height is larger than 12 (PE set has less than 168 PEs); divide into separated segments that are mapped independently to the array, Eyeriss not supporting PE set taller than height of PE array (max. supported filter height is 12)
* this mapping is realized by custom NoC (also optimized for energy efficiency)

### Dimensions betond 2-D in PE array
* additional dimensions; batch size (N), # of channels (C), and # of filters (M) -> requires processing of many 2-D convolutions
* varying only 1-D at a time, fixing rest two the same, the cases of two 2-D convolutions (See Fig. 6)
	* different ifmaps reuse the same filter (filter reuse)
	* different filters reuse the same ifmap (ifmap reuse)
	* filters and ifmaps from different channels can accumulate their psums together (psum accumulation)
* Multiple 2-D Convolutions in a PE set
	* if spad size is large enough, each PE can run multiple 1-D convolution primitives simultaneously by interleaving (interleaving :  서로 다른 memory bank에 번갈아 가며 연속적인 주소를 부여함으로써 한 memory를 처리하는 동안에도 다음 메모리 주소에 접근할 수 있게 되는 기법) computation
		1. interleaving the computation of primitives that run on the same ifmap, different filters : spad can buffer same ifmap value and reuse it to compute with a weight from each filter sequentially, requires larger filter and psum spad size
		2. interleaving the computation of primitives that run on different channels : PE can accumulate through all channels sequentially on the same psum, requires larger ifmap and filter spad size
	* ex.) q different channels of p different filters (each PE runs $p \times q$ primitives simultaneously)
		* rows of filter weights from q channels of p filters : $p \times q \times S$
		* q sliding windows of ifmap values from different channels : $q \times S$
		* accumulation of psums in p ofmap channels : p
* Multiple PE sets in the PE array
	* PE array can fit more than one PE set if PE set are small enough
	* advantages of mapping multiple PE sets
		* increase processing throughput
		* same ifmap is read once from the GLB and reused in multiple sets simultaneously
		* psums from different sets are accumulated within the PE array (more efficient)
	* ex.) r different channels of t different filters (PE array fits $r \times t$ PE sets in parallel)
		* share same ifmap with t filters, r channels accumulate their psums within PE array
		* finally, PE array can run multiple 2-D convolutions from up to $q \times r$ channels of $p \times t$ filters simultaneously
	* in each layer, PEs that are not covered by any sets are clock gated (clock gating; 특정 회로의 동작이 필요하지 않는 경우 그 회로에 clock을 공급하지 않고 그 회로의 Flip-flop은 상태 변이가 일어나지 않게 됨) to save energy consumption
	* PE array processing passes
		* ‘Processing Pass’ : amount of computation done in this fashion
		* one pass; input data read only once from GLB, psums stored back to GLB only once when processing finished
		* GLB buffer ifmaps and psums
			* ifmaps stored in GLB, reused across multiple processing passes
			* psums use GLB as intermediate storage so they don’t go off-chip until final ofmap value is obtained
		* scheduling of processing passes (See Fig. 7)
			* determines the storage allocation required for ifmaps and psums in the GLB (specific size는 논문 예시 참조)
			* parameters change based on mapping of each layer, GLB allocation has to be reconfigurable (store them in different proportions)
			* parameters in Table II (See Table II) are determined by optimization considering 1. energy cost of memory hierarchy & 2. hardware resources (GLB size, spad size, # of PEs)

## Exploit Data Statistics
1. reduce DRAM access by ‘compression’
2. skip unnecessary computations

* ReLU function introduces many zeros in the fmaps (rectifying all negative to zero)
	* &#35; of zero increases in deep layers
	* in recent study, 16%~78% filter weights in CNN can be pruned to zero
* RLC (Run-Length Compression) -> exploit zeros in fmaps, save DRAM bandwidth
	* Consecutive zeros (maximum length 31) : represent with 5-b number (‘Run’)
	* 16-b number (‘Level’; which is next to ‘Run’)
	* 3 pairs of ‘Run’ & ‘Level’ packed into 64-b word
	* all fmaps are stored in RLC compressed form in DRAM (except input data for first layer of CNN)
	* accelerator read RLC encoded ifmaps from DRAM -> decompression (RLC decoder) -> write it to GLB
	* read computed ofmaps from GLB -> ReLU -> compressed (RLC encoder) -> transmit to DRAM
	* saves space & Read/Write bandwidth of DRAM  

# System modules
## Global Buffer
* 108 kB
* communicate with DRAM through async. interface
* communicate with PE array though NoC
* stores ifmaps, filters, psums/ofmaps
* While PE array is on processing pass, GLB preload the filters used by next processing pass
* storage for ifmaps and psums should be reconfigurable to fit in different proportions (based on scan chain bits)

## Network-on-chip
* manages data delivery between GLB and PE array / between different PEs
* have to support the data delivery patterns used in RS dataflow
* leverage the data reuse achieved by the RS dataflow to further improve energy efficiency
* provide enough bandwidth for data delivery in order to support the highly parallel processing in PE array

### Global Input Network (GIN)
* See Fig. 10 & Fig. 11
* optimized for single-cycle multicast from GLB to group of PEs that receive same filter weight, ifmap value, psum
* 2 hierarchy : Y-bus & X-bus
	* Y-bus; consists of 12 horizontal X-buses (each row of PE array)
	* each X-bus connects to 14 PEs in the row
	* each X-bus has ‘row ID’, each PE has ‘col ID’
	* IDs are reconfigurable
	* data read from GLB is augmented with (row, col) tag -> GIN deliver the data to X-buses and PEs with ID that matches the tag within single cycle
	* using Multicast Controller (MC)
	* Eyeriss has separate GINs for each of three data types to provide sufficient bandwidth from GLB to PE array

### Globla Output Network (GON)
* read psums generated by processing pass from PE array to GLB
* same with GIN, transfer is reversed

### Local Network
* between PEs on two consecutive rows of same col., dedicated 64-b data bus pass psums from bottom PE to top PE directly  

## Processing Element and Data Gating
* See Fig. 12
* FIFO used at I/O to balance workload between NoC and computation
* datapath pipelined into 3 stages; 1 for spad access, 2 for computation
* data gating logic
	* implemented to exploit zeros in the ifmap, saving processing power
	* 12-b Zero Buffer record position of zeros in the ifmap spad
	* if zero ifmap value detected, gating logic will disable the read of filter spad & prevent MAC datapath from switching
	* saves PE power consumption by 45%

# Results
* See Fig. 13 & Fig. 14
* Eyeriss chip is implemented in 65-nm CMOS and integrated into the Caffe framework
* peak throughput 33.6 GMAC/s with 200-MHz core clock @ 1 V
* most of state-of-the-art CNNs have shapes that fits to native support of Eyeriss
* benchmark chip performance using AlexNet & VGG-16
* refer to the paper for measurements on AlexNet & VGG-16
