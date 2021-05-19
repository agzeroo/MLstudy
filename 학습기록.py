#!/usr/bin/env python
# coding: utf-8

# In[2]:


# 기본
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 경고 뜨지 않게 설정
import warnings
warnings.filterwarnings('ignore')

# 그래프 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
# plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['font.size'] = 16
plt.rcParams['figure.figsize'] = 20, 10
plt.rcParams['axes.unicode_minus'] = False

# 데이터 전처리 알고리즘
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# 학습용과 검증용으로 나누는 함수
from sklearn.model_selection import train_test_split

# 교차 검증
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold


# In[5]:


iris_df = pd.read_csv('data/iris.csv')
iris_df.head()


# In[6]:


# 입력과 결과로 나눈다.
X = iris_df.drop('target', axis=1)
y = iris_df['target']

display(X)
display(y)


# In[7]:


# 문자열 -> 숫자
enc1 = LabelEncoder()
enc1.fit(y)
y = enc1.transform(y)
y


# In[ ]:





# In[ ]:





# ### Label Encoder
# - 문자 데이터가 저장되어 있는 컬럼의 데이터를 숫자로 변환하는 작업을 한다.
# - 지정된 컬럼에 저장되어 있는 값들을 확인하고 각 값들에 숫자를 부여해 변환해준다.
# - 복원도 가능하다.
# 

# In[ ]:


# 저장된 값을 이용해 딕셔너리를 생성한다.
dict1 = {
    'virginica' : 0,
    'setosa' : 1,
    'versicolor' : 2
}

# map 함수를 이용해 변환한다.
df_map = iris_df['target'].map(dict1)
print(df_map)


# ### 문자열 -> 숫자
# - LableEncoder함수를 사용해서 변환한다.
# - 먼저 학습을 시킨다.(fit) => encoder1.fit(df[column])
# - 학습을 토대로 변환(transform) = > encoder1.transform(df[column])

# In[ ]:


# LableEncoder
encoder1 = LabelEncoder()
# 데이터를 학습한다.
encoder1.fit(iris_df['target'])
# 학습한 것을 토대로 변환한다.
df_enc1 = encoder1.transform(iris_df['target'])

df_enc1


# ###  표준화 작업
# - 학습 데이터의 각 컬럼의 데이터를 비슷한 수준의 범위로 맞추는 작업
# - 성능을 약간 상승시킬 수 있는 효과를 얻을 수도 있다.
# - 시각화할 때도 도움이 될 수 있다.
# - target값을 제외한 모든 컬럼
# - 먼저 학습을 시킨다.(fit) 
# - 학습을 토대로 변환(transform)

# In[ ]:


# 입력 데이터에 대한 표준화
scaler1 = StandardScaler()
scaler1.fit(X)
X = scaler1.transform(X)
X


# ### 평가 원리
# - 검증은 학습하지 않은 패턴의 데이터를 통해 예측 결과를 추출하고 진짜 결과와 비교하는 과정
# - 결과데이터를 가지고 있는 전체 데이터를 학습용과 검증용으로 나눠 학습과 평가를 진행한다.

# In[ ]:


# 데이터를 나눈다.
# 입력데이터와 결과 데이터를 넣어주면 8:2로 분할해서 반환을 해준다.
# 전체 데이터의 행을 랜덤하게 섞은 후 테스트와 검증으로 나눈다.
# test_size : 검증용 데이터의 비율 (0 ~ 1), 기본 0.2
# random_state : 랜덤시드 설정. 시드를 설정하면 계속 같은 패턴으로 
# 섞이게 된다.
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=1)

print(len(train_X))
print(len(test_X))

train_X


# ### 학습과 평가를 진행한다.
# 

# In[ ]:


from sklearn.linear_model import LogisticRegression

# 학습한다.
model1 = LogisticRegression()
model1.fit(train_X, train_y)


# - 학습한 데이터를 통한 검증
# - train_test_split 나눈 결과중 train_X 로 위에서 학습한 로지스틱 회귀결과를 가져온다.
# - 예측결과 train_pred와 train_y를 비교해서 accuracy_score를 확인.
# - train_y로 학습 했기때문에 무조건 성능이 잘 나와야한다.

# In[ ]:


from sklearn.metrics import accuracy_score

# 학습 데이터를 통해 예측 결과를 가져온다.
train_pred = model1.predict(train_X)

# 평가한다.
r1 = accuracy_score(train_y, train_pred)
r1


# - 검증용 데이터(학습 하지 않은 데이터)=test_X,test_y 를 통한 평가
# - 학습하지 않은 데이터를 통해 예측 결과를 가져온다.
# - 위에서 학습된 모델을 가지고 새로운 데이터 test_X,test_y 를 사용해 다시 예측을 시도

# In[ ]:


test_pred = model1.predict(test_X)

# 평가한다.
r2 = accuracy_score(test_y, test_pred)
r2


# ### 교차 검증
# - 평가의 원리는 학습하지 않은 패턴의 데이터를 통해 결과를 예측하고 진짜 결과와 비교하여 얼마나 유사한지를 알아보는 것이다.
# - 허나 학습데이터와 검증데이터의 패턴이 바뀌면 성능 평가 결과가 달라질 수 있다.
# - 대부분의 데이터와 모델은 성능 평가 수치가 크게 달라지지 않는다.
# - 허나 크게 달라지는 경우도 있을 수 있기 때문에 이러한 평가를 수 차례 해야한다.
# - 즉 학습과 평과 데이터의 패턴을 바꿔가며 다수의 테스트를 거쳐 그 결과들을 통해 80% 이상의 정확도를 보이면서 정확도 패턴이 일정한 모델을 찾아야 한다.

# #### cross_val_score(학습모델, X, y, scoring='accuracy', cv=10)
# - 교차 검증을 실시 한다.
# - 첫 번째 : 평가해볼 학습 모델
# - 두 번째 : 입력데이터
# - 세 번째 : 결과데이터
# - scoring : 평가 지표.
# - cv : 교차검증 횟수
# - 만약 cv를 3으로 줬다면... 데이터가 총 3개의 꾸러미로 나뉜다.
# - 1회차 : 1+2 - 학습, 3 - 검증
# - 2회차 : 1+3 - 학습, 2 - 검증
# - 3회차 : 2+3 - 학습, 1 - 검증

# In[ ]:


X = iris_df.drop('target', axis=1)
y = iris_df['target']

# 문자열 -> 숫자
enc1 = LabelEncoder()
enc1.fit(y)
y = enc1.transform(y)

# 사용할 학습 모델을 생성한다.
model2 = LogisticRegression()

#교차검증
r1 = cross_val_score(model2, X, y, scoring='accuracy', cv=10)
r1


# ### KFold 교차 검증
# - Fold : 데이터의 꾸러미
# - K Fold : Fold가 K개 인것
# - 전체 데이터를 K 개의 묶음으로 나눠서 K 번 교차검을 한다.
# - 전체 데이터를 랜덤으로 섞을 것인지 아닌지를 결정할 수 있다.

# #### KFold(n_splits= , shuffle=True, random_state=1)
# - Fold 생성
# - n_splits : 폴드의 수. 데이터 꾸러미 개수
# - shuffle : True를 넣어주면 전체를 랜덤하게 섞고 폴드를 생성한다.
# - shuffle이 False(기본)라면 처음부터 순서대로 폴드를 생성한다.
# - random_state : 랜덤 시드 설정

# In[ ]:


kfold = KFold(n_splits=3, shuffle=True, random_state=1)

list(kfold.split(X))


# #### KFold 교차검증으로 cross_val_score의 교차검증 횟수 정한다.

# In[ ]:


# 교차 검증을 수행한다.
model3 = LogisticRegression()

r2 = cross_val_score(model3, X, y, scoring='accuracy', cv=kfold)
r2


# ### Stractified K Fold 교차 검증
# - KFold 교차 검증은 원본 데이터의 상태에 따라 학습과 검증데이터가 편향될 가능이 있다.
# - Stractified K Fold 교차 검증은 결과 데이터를 보고 모든 Fold의 결과 데이터 비율이 균등하게 될 수 있도록 보장해준다.
# - KFold보다 폴드 구성에 시간이 걸릴 수 있어 데이터량이 많으면 KFold를 먼저 해보는 것도 괜찮은 방법

# #### KFold, shuffle : False, KFold, shuffle : True, Stractified K Fold 비교

# In[3]:


kfold1 = KFold(n_splits=3)
kfold2 = KFold(n_splits=3, shuffle=True)
kfold3 = StratifiedKFold(n_splits=3)


# kfold1 = KFold(n_splits=3)
# - KFold, shuffle = False : 모든 데이터를 일정하게 나눈다.

# In[8]:


# KFold, shuffle = False
for train_idx, test_idx in kfold1.split(X) :
    # 학습용 데이터의 결과 데이터를 추출한다.
    y1 = y[train_idx]
    # 검증용 데이터의 결과 데이터를 추출한다.
    y2 = y[test_idx]
    
    # Series로 생성한다.
    s1 = pd.Series(y1)
    s2 = pd.Series(y2)
    
    display(s1.value_counts())
    display(s2.value_counts())


# kfold2 = KFold(n_splits=3, shuffle=True)
# - KFold, shuffle= True : 폴드의 데이터를 일정하게 나눈다.

# In[9]:


# KFold, shuffle = True
for train_idx, test_idx in kfold2.split(X) :
    # 학습용 데이터의 결과 데이터를 추출한다.
    y1 = y[train_idx]
    # 검증용 데이터의 결과 데이터를 추출한다.
    y2 = y[test_idx]
    
    # Series로 생성한다.
    s1 = pd.Series(y1)
    s2 = pd.Series(y2)
    
    display(s1.value_counts())
    display(s2.value_counts())


# kfold3 = StratifiedKFold(n_splits=3)
# - Stractified K Fold = shuffle=True 와 비슷함

# In[10]:


# Stractified K Fold
for train_idx, test_idx in kfold3.split(X, y) :
    # 학습용 데이터의 결과 데이터를 추출한다.
    y1 = y[train_idx]
    # 검증용 데이터의 결과 데이터를 추출한다.
    y2 = y[test_idx]
    
    # Series로 생성한다.
    s1 = pd.Series(y1)
    s2 = pd.Series(y2)
    
    display(s1.value_counts())
    display(s2.value_counts())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




