# 한밭대학교 인공지능소프트웨어학과 구보실팀

**팀 구성**
- 20221067 장채은 
- 20221046 김서연
- 20221073 정은주
  
## <u>Teamate</u> Project Background
- ### 필요성
  - **사이버 위협의 증가**: 불법 해킹 및 사이버 공격의 지속적인 급증
  - **암호화 트래픽의 한계**: 현대 네트워크 트래픽의 암호화로 인한 기존의 시그니처 기반 또는 패킷 내부를 분석하는 방식(DPI)활용의 어려움
  - **지능형 대응의 필요성**: 패킷 내용이 아닌 트래픽의 흐름(flow) 패턴을 기반으로 학습하는 AI/딥러닝 모델 및 실시간으로 탐지하는 지능형 시스템 구현 필요

- ### 기존 해결책의 문제점
  - **시그니처/룰 기반 방식**: 알려진 공격 패턴(Signature)이나 사전에 정의된 규칙(Rule)에 의존하는 방식은 알려지지 않은 새로운 형태의 공격(Zero-day)이나 기존 패턴을 우회하는 변종 공격 탐지 불가
  - **암호화 트래픽 분석의 어려움**: 트래픽 암호화에 따라 패킷의 내용을 검사하는 기존 방식 무력화
  - **배치 처리의 한계**: 일정 시간동안 데이터를 모아서 분석하는 경우, 이상 징후가 발생한 시점과 이를 인지하는 시점 사이에 큰 시간 지연이 발생하여 즉각적인 대응 불가능
 
&rArr; 본 시스템은 이러한 문제점을 해결하기 위해 **실시간**으로 **트래픽 흐름 패턴**을 **딥러닝 모델**로 분석
  
## System Design
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
  
  
## Conclusion
  - ### OOO
  - ### OOO
  
## Project Outcome
- ### 20XX 년 OO학술대회 
