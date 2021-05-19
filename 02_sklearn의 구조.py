#!/usr/bin/env python
# coding: utf-8

# # 사이킷런(scikit-learn) 시작
# 
# <img src='https://media.vlpt.us/images/dustin/post/3559b7ad-256a-458e-ac16-f05a8dc044b4/1200px-Scikit_learn_logo_small.svg.png' width='200' />
# 

# In[2]:


import pandas as pd
import numpy as np
from sklearn import datasets


# ### scikit-learn의 특징
# - 다양한 머신러닝 알고리즘을 구현한 파이썬 라이브러리
# - 심플하고 일관성 있는 API, 유용한 온라인 문서, 풍부한 예제를 제공한다.
# - 머신러닝을 위한 쉽고 효율적 개발 라이브러리를 제공한다.
# - 다양한 머신러닝 관련 알고리즘과 개발을 위한 프레임워크와 API를 제공한다.
# - 많은 사람들이 사용하며 다양한 환경에서 검증된 라이브러리
# 
# ### 머신러닝 주요 용어
# - 지도학습 : 입력데이터와 결과 데이터를 주는 학습 방법. 결과 예측
# - 비지도학습 : 입력 데이터만 주는 학습 방법. 가공, 군집, 차원축소 등등의 목적
# - 강화학습 : 환경과 움직임, 상벌을 결정하여 상을 받는 쪽의 결과가 나올 수 있는 데이터를 생산하는 학습 방법
# 
# ---
# 
# - 분류 : 결과 데이터가 레이블형(혈액형, 성별 등)인 경우의 지도학습
# - 회귀 : 결과 데이터가 범위형데이터(주식시세, 키, 몸무게 등)인 경우의 지도학습
# - 클러스터링 : 주어진 데이터의 패턴을 보고 그룹으로 묶어주는 비지도학습
# 

# ### scikit-learn 주요 모듈
# 
# | 모듈 | 설명 |
# |------|-------|
# | sklearn.datasets | 내장된 예제 데이터 세트 |
# | sklearn.preprocessing | 다양한 데이터 전처리 기능을 제공 |
# | sklearn.feature_selection | 특징(feature)를 선택할 수 있는 기능 제공 |
# | sklearn.feature_extraction | 특징(feature)을 추출에 사용 |
# | sklearn.decomposition | 차원 축소 관련 알고리즘 지원 |
# | sklearn.model_selection | 검증, 평가 등을 통해 최적의 알고리즘을 찾을 수 있도록 하는 기능을 제공한다 |
# | sklearn.metrics | 머신러닝 알고리즘에 대해 평가할 수 있는 라이브러리 |
# | 학습 모델에 관련된 것들 | 다양한 머신러닝을 알고리즘을 구현해 제공하는 라이브러리들
# 
# ### scikit-learn 주요 모듈
# 
# | 모듈 | 설명 |
# |------|-------|
# | sklearn.datasets | 내장된 예제 데이터 세트 |
# | sklearn.preprocessing | 다양한 데이터 전처리 기능을 제공 |
# | sklearn.feature_selection | 특징(feature)를 선택할 수 있는 기능 제공 |
# | sklearn.feature_extraction | 특징(feature)을 추출에 사용 |
# | sklearn.decomposition | 차원 축소 관련 알고리즘 지원 |
# | sklearn.model_selection | 검증, 평가 등을 통해 최적의 알고리즘을 찾을 수 있도록 하는 기능을 제공한다 |
# | sklearn.metrics | 머신러닝 알고리즘에 대해 평가할 수 있는 라이브러리 |
# | 학습 모델에 관련된 것들 | 다양한 머신러닝을 알고리즘을 구현해 제공하는 라이브러리들 |
# 

# ### 작업 과정
# - scikit-learn에 제공하는 학습 알고리즘을 생성한다.
# - 생성한 알고리즘을 최적화 하기 위한 하이퍼 파라미터를 튜닝한다.
# - 데이터를 알고리즘에 주어 학습을 시킨다.
# - 학습이 완료된 알고리즘(모델)을 통해 새롭게 발생된 데이터를 예측한다.
# - 학습시에는 fit() 함수를 사용한다.
# - 지도학습의 경우 predict() 함수를 통해 결과를 예측한다.
# - 비지도학습의 경우 transform()으로 데이터를 변환하거나 predict() 결과를 예측한다.
# 

# ### scikit-learn에서 제공하공하는 알고리즘 선택 참고 시트
# 
# <img src='https://scikit-learn.org/stable/_static/ml_map.png'/>
# 

# In[ ]:




