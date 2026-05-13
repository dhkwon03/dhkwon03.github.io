---
title: 논문 리뷰 (paper review); Memory Resource Management in VMware ESX Server
---

Those are the slides made for the paper review presentation in OS lecture. (CS530 @ KAIST)

Title: Memory Resource Management in VMware ESX Server  
Author: Carl A. Waldspurger (VMware, Inc.)  
Presented at OSDI 2002.   

__All the slides are made by me.__
__Also, I made the presentation myself during the class.__
__No unauthorized citation. Contact me for the permission.__

KAIST CS530 운영체제 강의에서 논문 리뷰 발표의 일환으로 제작한 슬라이드 입니다.   
모든 슬라이드는 제가 직접 제작하였으며, 혼자 수업에서 발표하였습니다.   
무단 인용을 금지합니다.  
해당 자료 사용을 위해서는 댓글이나 이메일을 통해 연락주시고 출처를 꼭 표기하시기 바랍니다.  
## TL;DR
* Introduces "mechanisms and policies" used to manage memory resource in ESX Server.
* Describes how each virtual machine (VM) can __allocated memory across VMs__ and __retrieves memory__ from the VM, while __running unmodified commodity OS__ on each VM.
* Ballooning: Reclaim memory from VM by implicitly causing the geust OS to invoke its own memory management algorithms
* Content-based transparent page sharing: Additional optimization (i.e. sharing) to reduce overall memory pressure in the system without guest OS involvement
* Idle memory tax: Enable both performance isolation (via share, 지분) and efficient memory utilization
* Dynamic reallocation policy: Coordinate above techniques to efficiently support VM workloads that overcommit memory
* I/O page remapping: Reduce I/O copying overheads, utilizing physical-to-machine address mapping

## Comments
* This paper is an important milestone in OS virtualization research.
* Ballooning technique is still used nowadays. It's a beautiful technique that enables the participation of guest OS in memory reclamation.
* Economical analysis of shares-per-page and idle memory tax: If we think 'share' as the total assets and 'page' as a unit of item, then shares-per-page becomes the 'price'. This aligns with the min-funding revocation algorithm which revokes the page from the client with __lowest price__. (lowest shares-per-page) Also, if there is more actively using page, shares-per-page (i.e. price) increases. We can think it as 'tax' related to active page. (In real economy, tax is something that increases the price depending on some factor.)

## Paper review
<img width="2666" height="1500" alt="Image" src="https://github.com/user-attachments/assets/bd441429-2924-4210-b5da-df57cfb61e6b" />
<img width="2666" height="1500" alt="Image" src="https://github.com/user-attachments/assets/f3c7da1a-cef6-4918-9906-08dcc2e70226" />
<img width="2666" height="1500" alt="Image" src="https://github.com/user-attachments/assets/4d50b3fe-9a0d-4a95-ac6b-34e1f3cef8d5" />
<img width="2666" height="1500" alt="Image" src="https://github.com/user-attachments/assets/0bd0e4fa-8d09-4eec-9a3f-f5f06bf34f1f" />
<img width="2666" height="1500" alt="Image" src="https://github.com/user-attachments/assets/60e81628-5cab-46af-aef4-2dd6db7ed790" />
<img width="2666" height="1500" alt="Image" src="https://github.com/user-attachments/assets/1ff4fb04-f7f5-4dec-8049-fcd425a43fb8" />
<img width="2666" height="1500" alt="Image" src="https://github.com/user-attachments/assets/bed6b915-646d-453b-98f1-3c782795a966" />
<img width="2666" height="1500" alt="Image" src="https://github.com/user-attachments/assets/5126a59e-5bf0-41b1-b9b9-5bab2bab980a" />
<img width="2666" height="1500" alt="Image" src="https://github.com/user-attachments/assets/94745ea8-1bcf-44a6-887c-615eddcc2764" />
<img width="2666" height="1500" alt="Image" src="https://github.com/user-attachments/assets/b33a7f50-70fe-45c7-b395-a9d86783bfe5" />
<img width="2666" height="1500" alt="Image" src="https://github.com/user-attachments/assets/ff6a7c7d-1c23-46f6-8c8a-ee97f87c34b7" />
<img width="2666" height="1500" alt="Image" src="https://github.com/user-attachments/assets/bc5637a6-56c2-401b-9213-3f994be37106" />
<img width="2666" height="1500" alt="Image" src="https://github.com/user-attachments/assets/72c1a41a-699a-419a-b193-e1324bd237f9" />
<img width="2666" height="1500" alt="Image" src="https://github.com/user-attachments/assets/9e2d8491-f696-46e3-b20f-cbf875249cd7" />
<img width="2666" height="1500" alt="Image" src="https://github.com/user-attachments/assets/de2126c0-597d-4c5d-b94d-22fa7b77a890" />
<img width="2666" height="1500" alt="Image" src="https://github.com/user-attachments/assets/2c8d752c-4bbe-4766-be84-418e97bb5798" />
<img width="2666" height="1500" alt="Image" src="https://github.com/user-attachments/assets/3b5625a4-2500-474a-82ab-9ff2bd6f91de" />
<img width="2666" height="1500" alt="Image" src="https://github.com/user-attachments/assets/ebc721bf-6de1-4fc4-b605-169a3418c9a2" />
<img width="2666" height="1500" alt="Image" src="https://github.com/user-attachments/assets/5eaffb77-7157-4b61-a5c7-57a73744672b" />
<img width="2666" height="1500" alt="Image" src="https://github.com/user-attachments/assets/de233466-7d97-4b08-a7db-40baee9cfff3" />
<img width="2666" height="1500" alt="Image" src="https://github.com/user-attachments/assets/86d4978c-1d4b-49f1-8613-2d253ac4d381" />
<img width="2666" height="1500" alt="Image" src="https://github.com/user-attachments/assets/f39210fc-77ea-4b04-9049-0ecf89ebf35b" />
<img width="2666" height="1500" alt="Image" src="https://github.com/user-attachments/assets/4f4387a1-09d8-42ae-8a2d-a489236dc865" />
<img width="2666" height="1500" alt="Image" src="https://github.com/user-attachments/assets/0318ba2e-ba15-47cf-815d-ee4b076dca9b" />
<img width="2666" height="1500" alt="Image" src="https://github.com/user-attachments/assets/c0376db2-8a3c-4a88-ad64-61f0175324bc" />
<img width="2666" height="1500" alt="Image" src="https://github.com/user-attachments/assets/ad46b0a6-c8b4-4431-bab5-852d5aafcb49" />
<img width="2666" height="1500" alt="Image" src="https://github.com/user-attachments/assets/4f6bbecc-2911-4a1a-9025-6952c562054b" />
<img width="2666" height="1500" alt="Image" src="https://github.com/user-attachments/assets/d57f00e8-8c57-44ca-9a71-1a29f1fb8e11" />
