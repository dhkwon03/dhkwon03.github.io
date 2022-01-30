---
title: 'xcrun: error: invalid active developer path 해결법'
---

항상 MacOS 업데이트 하고 iTerm 에서 git, g++ 같은 명령어 사용하려고 하면 발생하는 에러다.  
도대체 이런 말도 안되는 버그는 언제 고치려나...  
맨날 인터넷 찾아보는데 그냥 내 사이트에 기록할 겸 남겨놓는다.  
iTerm 열고 command line에 
```
$ xcode-select --install
```

이 명령으로 xcode cli를 따로 설치해서 에러를 해결할 수 있다고 한다.  
시간은 좀 걸린다. 기다리면 알아서 완료된다.  
끝.
