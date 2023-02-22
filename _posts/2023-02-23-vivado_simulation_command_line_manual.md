---
title: Vivado RTL simulation in command line on M1 pro windows 11 ARM virtual machine
tags:
- verilog
---

필자는 현재 이 post를 쓰는 시점에 M1 맥북을 사용중이다. m1은 지금까지 써본 컴 중 가장 팬이 안 돌아가고 가장 빠릿빠릿한 프로세서라고 생각하여 매우 만족하며 잘 쓰고 있고 주변 사람들에게도 강력히 추천하고 있다. 


하지만 엄청난 단점이 하나 있으니... 그건 바로 mac를 지원하지 않는 프로그램이 꽤 많다는 것이다. 또한, mac은 rosetta 라는 프로그램이 내장되어 있어 x86 프로그램을 arm 환경에서도 돌아갈 수 있도록 만들어주지만 음... 속도가 아주 살짝 느려진다 (그렇지만 거의 무시할 수 있을 정도이다) 다른 얘기로 잠깐 샜는데 rosetta 얘기는 이번 post와는 전혀 관련 없으니까 그냥 넘어가도록 하자. 
이번 학기 (23년 봄)에 학교에서 듣는 강의 중에 verilog로 rtl 작성하고 시뮬레이션을 해야하는 과목이 있다. 학기 들어가서 시뮬레이션 할 방법을 강구하기엔 너무 늦겠다 싶어서 미리 생각을 해보았다. 


이전에 연구실에서 인턴할 때는 시뮬레이션할 때 vivado를 사용했었다. 그럼 똑같이 vivado 쓰면 되지 않을까 했는데 생각해보니 vivado 는 mac을 지원하지 않고 windows나 linux에서만 돌아간다. 그럼 나에게는 parallels (거금 10만원을 바친) 가 있고 virtual machine으로 ubuntu와 windows 11이 깔려있는데 ARM linux에는 x86을 arm으로 번역해주는 내장 기능이 없으니 mac 처럼 x86 to ARM 번역기가 내장되어 있는 windows 11을 쓰면 되겠구나. windows 11에 무료로 vivado 설치해주고 어차피 x86도 돌아가니까 프로그램은 잘 실행될 것이고 원래 하던대로 시뮬 돌리면 잘 돌아가겠지? 라는 계산이 나왔다. 


실제로 해보니 vivado는 잘 열리는데 열리자마자 버벅대더니 rtl 파일을 추가하자 급기야 프로그램이 죽어버렸다. 컴퓨터 사양이 이렇게 좋은데 왜 죽지? 라는 생각이 들어 알아보니 parallels standard edition은 virtual machine 돌릴 때 쓸 수 있는 cpu 코어 개수가 4개, RAM이 8GB 로 제한된단다. (RAM이 32GB 인데 왜 쓰지를 못하니... ㅠㅠ) parallels 특유의 돈 뜯어내는 정책이 참 마음에 안 들었지만 mac 사용자로서 어쩔 수 없이 울며겨자먹기로 거금들여 샀는데 edition에 따라서 이런 제한을 걸어놓다니. 좀 화가 났지만 필자는 방법을 찾아내서 이 post에 담았다. 개인 기록용이자 혹시 나와 같은 상황인 사람이 있다면 참고하라는 의미에서... 

0. parallels의 windows 11에서 mac 전체 파일들은 Z 드라이브에 있다. windows powershell 열고 ```cd Z:``` 해서 원하는 폴더로 진입. (무슨 드라이브에 파일이 있는지는 다를 수 있다.) 
1. 전체 프로젝트 디렉토리를 만들고 그 안에 rtl 전용 디렉토리와 simulation 전용 디렉토리 2개를 만든다. ex.) hw01 디렉토리 만들고 hw01 하위 디렉토리로 rtl 디렉토리와 hw01_sim 디렉토리 생성 
2. rtl 디렉토리에 모든 testbench와 design RTL 파일들을 넣는다. RTL 파일이면 전부다 넣도록 하자. 
3. simulation 전용 디렉토리 (ex. hw01_sim) 에 prj 파일 생성. ex.) hw01_sim.prj 
4. prj 파일 안에는 rtl 파일에 종류에 따라 다음과 같은 내용을 적는다. (자세한 내용은 vivado logic simulation manual의 121 page 참조)\
rtl 파일이 verilog 일 때 : ```verilog <work_library> <file_name>```  \
rtl 파일이 vhdl 일 때 : ```vhdl <work_library> <file_name>```  \
rtl 파일이 systemverilog 일 때 : ```sv <work_library> <file_name>```  \
각각의 rtl 파일마다 한 줄 씩 알맞게 다 적어주어야 한다. <u>만약 package 파일 같이 constant value 를 정의해놓거나 컴파일 순서 상 먼저 컴파일 되야 하는 파일들이 있다면 순서에 맞게 앞줄에 적는다. prj 파일에 적힌 순서대로 컴파일 된다 </u> 특별한 경우가 아니면 <work_library> 는 work 로 그냥 적는다. <file_name> 에는 ../rtl/(rtl이름) (예시 ../rtl/test.sv) 과 같이 prj 파일 기준으로 rtl 파일의 file path를 적는다. 
5. prj 파일이 있는 폴더에서 다음의 커맨드를 실행한다.   \
```xelab -prj (prj 파일 이름) -debug typical (top module 이름) (-sv) -s (simulation snapshot 이름)```  \
예시) ```xelab -prj hw01_sim.prj -debug typical TB_APB -sv -s top_sim```  \
<u>top module 의 이름은 testbench의 top module의 이름을 적어야 한다. DUT의 모듈 이름을 적으면 안되고 DUT를 인스턴스한 testbench의 top module 이름을 적도록 하자.</u> ```-sv``` 옵션은 systemverilog 문법으로 컴파일하고 싶을 때 붙인다. verilog 로 코드를 작성해도 ```-sv``` 옵션 붙이는 건 상관없을 듯 하다 (혹시 에러가 난다면 ```-sv``` 옵션은 빼고 다시 돌려보자) ```-s``` 옵션 뒤에 나오는 것은 simulation snapshot 이름이다. 적당한 이름으로 적어넣어준다.  \
컴파일 과정에서 에러가 나면 이 단계에서 알 수 있다. 뭔가 에러가 났다면 메세지를 보고 고쳐서 컴파일 돌리고 성공하면 다음 단계로 넘어가자.  
6. 만약 testbench나 rtl 코드에서 특정 파일 (.txt, bitmap, 등등) 을 불러오는 코드가 있다면 그 파일이 필요하다. 그 파일은 simulation 전용 폴더 밑으로 그냥 복사해 놓도록 하자. (예시 hw01/hw01_sim에 mem0_init.txt, mem1_init.txt 등 파일들을 복사)  
7. 다음의 command를 실행한다.  \
```xsim (simulation snapshot 이름) -gui```  \
앞에서 옵션으로 넣었던 simulation snapshot 이름을 그대로 여기 커맨드에도 적어준다.  \
이 커맨드를 실행하면 vivado 창이 뜰 것이다. 로딩이 끝나면 나오는 object 들 중 원하는 signal을 선택해서 오른쪽 버튼 클릭하고 add to waveform 한다. 옆에 waveform 이 뜰 것이고 wave를 보고 싶은 신호들을 모두 추가했다면 vivado 창에서 위쪽에 있는 restart 버튼을 클릭하고 이후에 run all 버튼을 클릭한다. 그러면 waveform이 나오면서 vivado 창 아래쪽에 있는 command 창에서도 출력할 것이 있다면 출력된다. 만약 시뮬레이션을 다시하고 싶다면 <u>꼭 restart 버튼을 다시 누른 후에 </u> run all 버튼이나 run for ooo ns (simulation time은 바로 옆 입력칸에 입력해서 설정) 를 누른다.  
8. 이제 waveform 자유롭게 보면서 잘못된 것이 있다면 debugging 하고 코드 고쳐서 5번, 7번, 8번을 반복하며 simulation 결과를 확인하고 또 debugging 하고.... design 버그를 모두 잡아낼 때까지 그렇게 하면 된다.
