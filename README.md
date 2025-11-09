# 한밭대학교 인공지능소프트웨어학과 구보실팀

**팀 구성**
- 20221067 장채은 
- 20221046 김서연
- 20221073 정은주


  
## <u>Teamate</u> Project Background
- ### 필요성
  - **사이버 위협의 증가**: 법 해킹 및 사이버 공격이 지속적으로 급증하면서 개인·기업·공공기관을 가리지 않고 피해가 확산되고 있으며, 사회 전반의 정보 인프라를 위협하고 있다.
  - **암호화 트래픽의 한계**: 현대 네트워크 트래픽이 대부분 암호화(HTTPS, VPN 등)되어 있어 기존의 패킷 기반 분석으로는 이상 징후를 탐지하기 어렵다.
  - **지능형 대응의 필요성**: 패킷 내용이 아닌 트래픽의 흐름(flow) 패턴을 기반으로 학습하는 AI/딥러닝 모델 및 실시간으로 탐지하는 지능형 시스템 구현 필요

- ### 기존 해결책의 문제점
  - **패킷 기반 분석 한계**:
기존의 패킷 내용 직접 분석 방식은 트래픽이 대부분 암호화(HTTPS, VPN 등)되어 있어 내부 데이터를 확인하기 어렵다.
그 결과, 공격 여부를 정확히 식별하기 힘들고 암호화 환경에서는 탐지 성능이 급격히 저하된다.
  - **복호화 기반 접근의 한계**:
암호화된 트래픽을 복호화하여 분석하는 방식은 개인정보 침해 위험이 있으며, 복호화 과정에서 연산 부하가 증가해 시스템 성능 저하를 초래한다. 특히 대규모 네트워크 환경에서는 실시간 탐지가 어려워 실질적인 적용이 제한적이다.
  - **배치 처리의 한계**: 일정 시간동안 데이터를 모아서 분석하는 경우, 이상 징후가 발생한 시점과 이를 인지하는 시점 사이에 큰 시간 지연이 발생하여 즉각적인 대응 불가능
 
&rArr; 본 시스템은 이러한 문제점을 해결하기 위해 **실시간**으로 **트래픽 흐름 패턴**을 **딥러닝 모델**로 분석
  
## System Design

<div align="center"> <img width="1157" height="508" alt="Image" src="https://github.com/user-attachments/assets/a20e6675-20e0-4c07-ac62-ad8ffe8fa371" />
<br>
시스템 구조도
</div>

   - <h4>데이터 송신부 (Server Raspberry Pi):</h4>
<ul>
  <li>모델 test에 사용된 데이터 중 랜덤한 트래픽 송신</li>
  <li>해당 데이터는 <strong>58개의 특징(feature)</strong> 벡터로 구성</li>
  <li> 58개 특징 리스트를 JSON 형식으로 직렬화하여 '이상 탐지 서버' (클라이언트 Pi)로 1초마다 전송</li>
</ul>

<h4>이상 탐지 서버 (Client Raspberry Pi):</h4>
<ul>
  <li>TCP/IP 소켓을 열어 데이터 수신 대기</li>
  <li>데이터 송신부로부터 58개 특징이 담긴 JSON 데이터 수신</li>
  <li>수신한 데이터를 즉시 PyTorch 텐서(<code>torch.tensor</code>)로 변환</li>
</ul>

<h4>AI 모델 (Model):</h4>
<ul>
  <li>사전 훈련된 경량 PyTorch 모델(<code>enc.pth</code>, <code>clf.pth</code>) 사용</li>
  <li>모델은 입력된 58개 특징 벡터를 분석하여 '정상(0)' 또는 <strong>'이상(1)'</strong>으로 이진 분류</li>
  <li><code>torch.no_grad()</code> 컨텍스트 내에서 실행되어 제한된 자원(라즈베리파이 CPU)에서도 효율적인 추론 수행</li>
</ul>

<h4>시각화 (Visualization):</h4>
<ul>
  <li>탐지 서버(클라이언트 Pi)에 연결된 모니터에 Tkinter 기반의 GUI 출력 </li>
  <li>모델의 예측 결과에 따라 GUI 화면이 <mark>[정상 (초록색)]</mark> 또는 <strong><mark>[이상 (빨간색)]</mark></strong>으로 즉시 변경, 사용자에게 직관적인 경고 제공</li>
</ul>
<hr>

### System Requirements
<strong>Hardware:</strong>
<ul>
  <li>라즈베리파이 2대 (데이터 송신용, 수신/분석용)</li>
  <li>모니터 1대</li>
</ul>

<strong>Software (탐지 서버 기준):</strong>
<ul>
  <li>Python 3.x</li>
  <li>필수 라이브러리: <code>torch</code> (PyTorch), <code>numpy</code></li>
</ul>

<strong>필수 파일:</strong>
<ul>
  <li><code>server_gui.py</code>: 메인 서버 및 GUI 애플리케이션</li>
  <li><code>enc.pth</code>: 훈련된 인코더 모델 가중치</li>
  <li><code>clf.pth</code>: 훈련된 분류기 모델 가중치</li>
</ul>
<hr>
    
## Case Study

  - ### Description
    - ### 실험 환경 구성
      - 두 대의 라즈베리파이를 사용하여, 하나는 트래픽을 생성하는 송신 장치로, 다른 하나는 AI 기반 이상 탐지 시스템으로 구성하였다.
        
          > 라즈베리파이는 실제 IoT 환경을 가정하여 저전력·소형 하드웨어에서의 실시간 동작 가능성을 검증하기 위해 사용되었다.
          > <div align="center"> <img width="528" height="250" alt="Images" src="https://github.com/user-attachments/assets/01cf1670-470c-429e-8453-a399beeaec65" />
          > <br>
          > 전체 환경 구조 (라즈베리파이와 모니터 사용)
          > </div>
      <br>
      <br>

      
      - **서버 라즈베리파이**: 랜덤 트래픽 데이터를 정하고, 모니터의 버튼을 눌러 클라이언트 라즈베리파이에 데이터를 전송한다.

        > <div align="center"> <img width="523" height="317" alt="Image" src="https://github.com/user-attachments/assets/dd79a3e8-fbf4-42bd-89c7-a1962b6c41c9" />
        > <br>
        > 랜덤 트래픽 데이터 전송 </div>
     
      
        > <div align="center"> <img width="561" height="221" alt="Image" src="https://github.com/user-attachments/assets/1260b27a-6be1-4e46-b553-292009382530" /> 
        > <br>
        > 서버 라즈베리파이 모니터 화면 </div>
        
        <br>
      - **클라이언트 라즈베리파이**: AI 모델을 통해 실시간으로 이상 트래픽과 정상 트래픽을 탐지하고 모니터에 알림을 띄운다.

        > <div align="center"> <img width="575" height="343" alt="Image" src="https://github.com/user-attachments/assets/48b7fe0f-d0d5-4c9e-a6dc-c45849b8a3b2" />
        > <br>
        > 포트 개방 및 대기
        > </div>
     
        
        > <div align="center">  <img width="559" height="227" alt="Image" src="https://github.com/user-attachments/assets/203ad61e-b3d7-4af8-95eb-9d1c83cad600" />
        > <br>
        > 클라이언트 라즈베리파이 모니터 화면 </div>

  
  
## Conclusion

  - ### 연구결과

    - AI 기반 이상 트래픽 탐지 모델을 설계하고, 라즈베리파이를 활용하여 저전력 하드웨어 환경에서도 실시간으로 안정적 동작이 가능한 이상 트래픽 탐지 시스템을 구현하였다.

    - 네트워크 상의 비정상적인 트래픽을 실시간으로 탐지하고 이에 대응할 수 있는 그래픽 사용자 인터페이스(GUI)를 개발하여, 관리자와 사용자가 직관적으로 이상 트래픽 상태를 인식하고 대응할 수 있는 환경을 구현하였다.
  <img width="1682" height="1017" alt="Image" src="https://github.com/user-attachments/assets/534e10e4-6343-41f6-9e64-c0bb87398cc4" />
  <div align="center"> 전체 결과 동작 흐름  </div>


  - ### 기대효과

    - 경량 AI 모델의 실효성 입증: 제한된 리소스 환경에서도 AI 모델이 안정적으로 작동함을 확인하여, 향후 스마트홈·IoT 내장형 보안 모듈로 확장 가능성을 증명하였다.

    - AI·보안 융합 기술의 응용 확대: 실제 시연을 통해 AI 기반 네트워크 보안의 현실적 적용 가능성을 입증하였으며, 산업용 IoT·스마트시티 등 다양한 분야로의 확장이 기대된다.
  
## Project Outcome
- ### 20XX 년 OO학술대회 
