#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

import pandas as pd


# In[2]:


iris_df = pd.read_csv('data/iris.csv')
iris_df


# ### Label Encoder
# - 문자 데이터가 저장되어 있는 컬럼의 데이터를 숫자로 변환하는 작업을 한다.
# - 지정된 컬럼에 저장되어 있는 값들을 확인하고 각 값들에 숫자를 부여해 변환해준다.
# - 복원도 가능하다.
# 

# In[3]:


# pandas의 map 사용
# 먼저 컬럼에 어떠한 값들이 저장되어 있는지 파악해야 한다.
iris_df['target'].value_counts().index


# In[4]:


# 저장된 값을 이용해 딕셔너리를 생성한다.
dict1 = {
    'virginica' : 0,
    'setosa' : 1,
    'versicolor' : 2
}

# map 함수를 이용해 변환한다.
df_map = iris_df['target'].map(dict1)
print(df_map)


# In[6]:


#변환한 데이터를 다시 원상복구해보기
dict2={
    0: 'virginica' ,
    1 :'setosa',
    2: 'versicolor'
}

df_map2 = df_map.map(dict2)
print(df_map2)


# - LableEncoder함수를 사용해서 변환한다.
# - 먼저 학습을 시킨다.(fit) => encoder1.fit(df[column])
# - 학습을 토대로 변환(transform) = > encoder1.transform(df[column])

# In[7]:


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

# In[8]:


iris_df


# In[9]:


# 표준화 작업을 위해 결과 데이터를 제외한다.
X = iris_df.drop('target', axis=1)
X


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




