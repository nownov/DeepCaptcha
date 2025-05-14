# DeepCaptcha
PBL Team SWAMPS

-이상치 탐지 방법론을 이용한 딥보이스 탐지-
### 1. 개발 배경

딥보이스 기술의 발전으로 증가한 딥보이스 기반 보이스피싱 피해를 낮추는 것이 목표이며 대화 내용 기반 보이스피싱 탐지 프로그램에 딥보이스 탐지 기술을 추가한 연구 및 애플리케이션을 개발한다.  


### 2. 전체 구성

2.1 딥보이스 탐지 : 실제 음성을 STFT로 변환 후, Bicoherence로 변환하여 구성한 데이터 셋을 Ganomaly 모델에 학습시켜 Anomal Score로 딥보이스 적용 여부 판별

2.2 내용 기반 탐지 : 피싱대화/일반대화 음성을 STT로 변환 후 벡터화하여 구성한 데이터 셋을 LSTM 모델에 학습 시켜 이진 분류를 통해 보이스피싱 대화 여부 판별

### 성과

- 2023 졸업 프로젝트 평가회 **Grand Prize (대상 수상)**
- 개인정보 혁신인재 양성 분야 PBL 프로젝트 보고회 **Grand Prize (대상 수상)**




# DeepCaptcha

PBL Team SWAMPS

DeepVoice Detection Using Anomaly Detection Methods
### 1. Background
With the advancement of deep voice technology, the goal is to reduce the damage caused by deep voice-based voice phishing. This project involves the development of a research and application system that integrates deep voice detection technology into a conversation-based voice phishing detection program.

### 2. System Overview
2.1 DeepVoice Detection

The real human voice is converted into STFT (Short-Time Fourier Transform) and then into Bicoherence. The dataset is trained using the Ganomaly model to classify the voice as deep voice or not based on the Anomaly Score.
2.2 Content-based Detection

Voice data from phishing and regular conversations is converted into text using STT (Speech-to-Text). The dataset is then vectorized and trained using an LSTM (Long Short-Term Memory) model for binary classification to detect whether the conversation is a voice phishing attempt.

### Achievements
Grand Prize at the 2023 Graduation Project Evaluation
Grand Prize at the PBL Project Presentation for Training Innovation Talent in Personal Information Security


### Brochure

![b4가 낫다  (1)](https://github.com/daeun6/DeepCaptcha/assets/81478444/961239cf-5431-4c1d-9a86-c45bf5b7a0ae)

![학부생  (7)](https://github.com/daeun6/DeepCaptcha/assets/81478444/d1e731b4-0df2-4634-a32c-d1cfe3abe8d0)



### ✌️ 참여 인원

|[송다은](https://github.com/daeun6)|[이나영]()|[김지원]()|
| --- | --- | --- |
|<img width="100" src="https://github.com/GDSC-SWU/2023-AI-ML-study/assets/81478444/21400679-dcc3-4731-9638-d8f717e0bc84"/>|<img width="100" src="https://github.com/daeun6/DeepCaptcha/assets/81478444/f1c138c1-8e9c-4f0c-aea1-27743fa18983"/>|<img width="100" src="https://github.com/daeun6/DeepCaptcha/assets/81478444/80e07910-6790-4f58-bf53-0960134e1777"/>|
