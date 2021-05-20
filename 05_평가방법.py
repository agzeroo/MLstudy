{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 기본\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "#경고 뜨지 않게 설정\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "#그래프 설정\n",
    "plt.rcParams['font.family']='Malgun Gothic'\n",
    "\n",
    "plt.rcParams['font.size']=16\n",
    "plt.rcParams['figure.figsize']=20,10\n",
    "plt.rcParams['axes.unicode_minus']= False\n",
    "\n",
    "# 데이터 전처리 알고리즘\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "\n",
    "# 학습용과 검증용으로 나누는 함수\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "# 교차 검증\n",
    "from sklearn.model_selection import cross_val_score\n",
    "\n",
    "from sklearn.model_selection import KFold\n",
    "\n",
    "\n",
    "from sklearn.model_selection import StratifiedKFold\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 데이터 준비하기"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>sepal length (cm)</th>\n",
       "      <th>sepal width (cm)</th>\n",
       "      <th>petal length (cm)</th>\n",
       "      <th>petal width (cm)</th>\n",
       "      <th>target</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>5.1</td>\n",
       "      <td>3.5</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "      <td>setosa</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>4.9</td>\n",
       "      <td>3.0</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "      <td>setosa</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>4.7</td>\n",
       "      <td>3.2</td>\n",
       "      <td>1.3</td>\n",
       "      <td>0.2</td>\n",
       "      <td>setosa</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4.6</td>\n",
       "      <td>3.1</td>\n",
       "      <td>1.5</td>\n",
       "      <td>0.2</td>\n",
       "      <td>setosa</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5.0</td>\n",
       "      <td>3.6</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "      <td>setosa</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  \\\n",
       "0                5.1               3.5                1.4               0.2   \n",
       "1                4.9               3.0                1.4               0.2   \n",
       "2                4.7               3.2                1.3               0.2   \n",
       "3                4.6               3.1                1.5               0.2   \n",
       "4                5.0               3.6                1.4               0.2   \n",
       "\n",
       "   target  \n",
       "0  setosa  \n",
       "1  setosa  \n",
       "2  setosa  \n",
       "3  setosa  \n",
       "4  setosa  "
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "iris_df = pd.read_csv('data/iris.csv')\n",
    "iris_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>sepal length (cm)</th>\n",
       "      <th>sepal width (cm)</th>\n",
       "      <th>petal length (cm)</th>\n",
       "      <th>petal width (cm)</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>5.1</td>\n",
       "      <td>3.5</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>4.9</td>\n",
       "      <td>3.0</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>4.7</td>\n",
       "      <td>3.2</td>\n",
       "      <td>1.3</td>\n",
       "      <td>0.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4.6</td>\n",
       "      <td>3.1</td>\n",
       "      <td>1.5</td>\n",
       "      <td>0.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5.0</td>\n",
       "      <td>3.6</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>145</th>\n",
       "      <td>6.7</td>\n",
       "      <td>3.0</td>\n",
       "      <td>5.2</td>\n",
       "      <td>2.3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>146</th>\n",
       "      <td>6.3</td>\n",
       "      <td>2.5</td>\n",
       "      <td>5.0</td>\n",
       "      <td>1.9</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>147</th>\n",
       "      <td>6.5</td>\n",
       "      <td>3.0</td>\n",
       "      <td>5.2</td>\n",
       "      <td>2.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>148</th>\n",
       "      <td>6.2</td>\n",
       "      <td>3.4</td>\n",
       "      <td>5.4</td>\n",
       "      <td>2.3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>149</th>\n",
       "      <td>5.9</td>\n",
       "      <td>3.0</td>\n",
       "      <td>5.1</td>\n",
       "      <td>1.8</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>150 rows × 4 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)\n",
       "0                  5.1               3.5                1.4               0.2\n",
       "1                  4.9               3.0                1.4               0.2\n",
       "2                  4.7               3.2                1.3               0.2\n",
       "3                  4.6               3.1                1.5               0.2\n",
       "4                  5.0               3.6                1.4               0.2\n",
       "..                 ...               ...                ...               ...\n",
       "145                6.7               3.0                5.2               2.3\n",
       "146                6.3               2.5                5.0               1.9\n",
       "147                6.5               3.0                5.2               2.0\n",
       "148                6.2               3.4                5.4               2.3\n",
       "149                5.9               3.0                5.1               1.8\n",
       "\n",
       "[150 rows x 4 columns]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "0         setosa\n",
       "1         setosa\n",
       "2         setosa\n",
       "3         setosa\n",
       "4         setosa\n",
       "         ...    \n",
       "145    virginica\n",
       "146    virginica\n",
       "147    virginica\n",
       "148    virginica\n",
       "149    virginica\n",
       "Name: target, Length: 150, dtype: object"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#입력과 결과로 나는다.\n",
    "X=iris_df.drop('target',axis=1)\n",
    "y=iris_df['target']\n",
    "\n",
    "display(X)\n",
    "display(y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
       "       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
       "       0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n",
       "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n",
       "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,\n",
       "       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,\n",
       "       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#문자열->숫자\n",
    "enc1=LabelEncoder()\n",
    "enc1.fit(y)\n",
    "y=enc1.transform(y)\n",
    "y"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 평가 원리\n",
    "\n",
    "- 검증은 학습하지 않은 패턴의 데이터를 통해 예측 결과를 추출하고 진짜 결과와 비교하는 과정\n",
    "- 결과 데이터를 가지고 있는 전체 데이터를 학습용과 검증용으로 나눠 학습과 평과를 진행한다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "150"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(X)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "105\n",
      "45\n"
     ]
    }
   ],
   "source": [
    "# 데이터를 나눈다.\n",
    "# 입력데이터와 결과 데이터를 넣어주면 8:2로 분할해서 반환을 해준다.\n",
    "# 전체 데이터의 행을 랜덤하게 섞은 후 테스트와 검증으로 나눈다.\n",
    "# test_size : 검증용 데이터의 비율 (0 ~ 1), 기본 0.2\n",
    "# random_state : 랜덤시드 설정. 시드를 설정하면 계속 같은 패턴으로 \n",
    "# 섞이게 된다.\n",
    "train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=1)\n",
    "\n",
    "print(len(train_X))\n",
    "print(len(test_X))\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 학습과 평가를 진행한다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LogisticRegression()"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.linear_model import LogisticRegression\n",
    "\n",
    "#학습한다.\n",
    "model1=LogisticRegression()\n",
    "model1.fit(train_X, train_y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9809523809523809"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 학습한 데이터를 통한 검증\n",
    "# 무조건 성능이 잘나와야 된다.\n",
    "from sklearn.metrics import accuracy_score\n",
    "\n",
    "# 학습 데이터를 통해 예측 결과를 가져온다.\n",
    "train_pred = model1.predict(train_X)\n",
    "\n",
    "# 평가한다.\n",
    "r1 = accuracy_score(train_y, train_pred)\n",
    "r1\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9777777777777777"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 검증용 데이터(학습 하지 않은 데이터)를 통한 평가\n",
    "# 학습하지 않은 데이터를 통해 예측 결과를 가져온다.\n",
    "test_pred = model1.predict(test_X)\n",
    "\n",
    "# 평가한다.\n",
    "r2 = accuracy_score(test_y, test_pred)\n",
    "r2\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 교차 검증\n",
    "- 평가의 원리는 학습하지 앟은 패턴의 데이터를 통해 결과를 예측하고 진짜 결과와 비교하여 얼마나 유사한지를 알아 봄\n",
    "- 허나 학습데이터와 검증데이터의 패턴이 바뀌면 성능평가 결과가 달라질 수 있다.\n",
    "- 대부분의 데이터와 모델은 성능 평가 수치가 크게 달라지지 않는다.\n",
    "- 허나 크게 달라지는 경우도 있을 수 있기 때문에 이러한 평가를 수 차례 해야한다.\n",
    "- 즉 학습과 평가 데이터의 패턴을 바꿔가며 다수의 테스트를 거쳐 그 결과들을 통해 80%이상의 정확도를 보이면서 정확도 패턴이 일정한 모델을 찾아야 한다.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1.        , 0.93333333, 1.        , 1.        , 0.93333333,\n",
       "       0.93333333, 0.93333333, 1.        , 1.        , 1.        ])"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 사용할 학습 모델을 생성한다.\n",
    "model2 = LogisticRegression()\n",
    "\n",
    "# 교차 검증을 실시 한다.\n",
    "# 첫 번째 : 평가해볼 학습 모델\n",
    "# 두 번째 : 입력데이터\n",
    "# 세 번째 : 결과데이터\n",
    "# scoring : 평가 지표.\n",
    "# cv : 교차검증 횟수\n",
    "# 만약 cv를 3으로 줬다면... 데이터가 총 3개의 꾸러미로 나뉜다.\n",
    "# 1회차 : 1+2 - 학습, 3 - 검증\n",
    "# 2회차 : 1+3 - 학습, 2 - 검증\n",
    "# 3회차 : 2+3 - 학습, 1 - 검증\n",
    "r1 = cross_val_score(model2, X, y, scoring='accuracy', cv=10)\n",
    "r1\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### KFold 교차 검증\n",
    "- Fold : 데이터의 꾸러미\n",
    "- K Fold : Fold가 K개 인것\n",
    "- 전체 데이터를 K 개의 묶음으로 나눠서 K 번 교차검을 한다.\n",
    "- 전체 데이터를 랜덤으로 섞을 것인지 아닌지를 결정할 수 있다.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Fold 생성\n",
    "#n_splits : 폴드의 수. 데이터 꾸러미 개수\n",
    "# shuffle=True : 전체를 랜덤하게 섞고 폴드를 생성.\n",
    "# shuffle=False(기본) : 처음부터 순서대로 폴드를 생성.\n",
    "# random_state : 랜덤 시드 설정\n",
    "\n",
    "kfold = KFold(n_splits=3, shuffle=True, random_state=1)\n",
    "# list(kfold.split(X))\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0.98, 0.94, 0.96])"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model3=LogisticRegression()\n",
    "\n",
    "r2=cross_val_score(model3,X,y,scoring='accuracy',cv=kfold)\n",
    "r2"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### StratifiedKFold\n",
    "- KFold 교차 검증은 원본 데이터의 상태에 따라 학습과 검증데이터가 편향될 가능이 있다.\n",
    "- Stractified K Fold 교차 검증은 결과 데이터를 보고 모든 Fold의 결과 데이터 비율이 균등하게 될 수 있도록 보장해준다.\n",
    "- KFold보다 폴드 구성에 시간이 걸릴 수 있어 데이터량이 많으면 KFold를 먼저 해보는 것도 괜찮은 방법\n",
    "- 데이터가 너무 불균형일때 사용"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 폴드 생성\n",
    "kfold1 = KFold(n_splits=3)\n",
    "kfold2 = KFold(n_splits=3, shuffle=True)\n",
    "kfold3 = StratifiedKFold(n_splits=3)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2    50\n",
       "1    50\n",
       "dtype: int64"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "0    50\n",
       "dtype: int64"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "2    50\n",
       "0    50\n",
       "dtype: int64"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "1    50\n",
       "dtype: int64"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "1    50\n",
       "0    50\n",
       "dtype: int64"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "2    50\n",
       "dtype: int64"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# KFold, shuffle : False\n",
    "for train_idx, test_idx in kfold1.split(X) :\n",
    "    # 학습용 데이터의 결과 데이터를 추출한다.\n",
    "    y1 = y[train_idx]\n",
    "    # 검증용 데이터의 결과 데이터를 추출한다.\n",
    "    y2 = y[test_idx]\n",
    "    \n",
    "    # Series로 생성한다.\n",
    "    s1 = pd.Series(y1)\n",
    "    s2 = pd.Series(y2)\n",
    "    \n",
    "    display(s1.value_counts())\n",
    "    display(s2.value_counts())\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    36\n",
       "1    35\n",
       "2    29\n",
       "dtype: int64"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "2    21\n",
       "1    15\n",
       "0    14\n",
       "dtype: int64"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "2    36\n",
       "0    34\n",
       "1    30\n",
       "dtype: int64"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "1    20\n",
       "0    16\n",
       "2    14\n",
       "dtype: int64"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "2    35\n",
       "1    35\n",
       "0    30\n",
       "dtype: int64"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "0    20\n",
       "2    15\n",
       "1    15\n",
       "dtype: int64"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# KFold, shuffle : True\n",
    "for train_idx, test_idx in kfold2.split(X) :\n",
    "    # 학습용 데이터의 결과 데이터를 추출한다.\n",
    "    y1 = y[train_idx]\n",
    "    # 검증용 데이터의 결과 데이터를 추출한다.\n",
    "    y2 = y[test_idx]\n",
    "    \n",
    "    # Series로 생성한다.\n",
    "    s1 = pd.Series(y1)\n",
    "    s2 = pd.Series(y2)\n",
    "    \n",
    "    display(s1.value_counts())\n",
    "    display(s2.value_counts())\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2    34\n",
       "1    33\n",
       "0    33\n",
       "dtype: int64"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "1    17\n",
       "0    17\n",
       "2    16\n",
       "dtype: int64"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "1    34\n",
       "2    33\n",
       "0    33\n",
       "dtype: int64"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "2    17\n",
       "0    17\n",
       "1    16\n",
       "dtype: int64"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "0    34\n",
       "2    33\n",
       "1    33\n",
       "dtype: int64"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "2    17\n",
       "1    17\n",
       "0    16\n",
       "dtype: int64"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Stractified K Fold\n",
    "for train_idx, test_idx in kfold3.split(X,y) :\n",
    "    # 학습용 데이터의 결과 데이터를 추출한다.\n",
    "    y1 = y[train_idx]\n",
    "    # 검증용 데이터의 결과 데이터를 추출한다.\n",
    "    y2 = y[test_idx]\n",
    "    \n",
    "    # Series로 생성한다.\n",
    "    s1 = pd.Series(y1)\n",
    "    s2 = pd.Series(y2)\n",
    "    \n",
    "    display(s1.value_counts())\n",
    "    display(s2.value_counts())\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0.98, 0.96, 0.98])"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model3 = LogisticRegression()\n",
    "\n",
    "r3 = cross_val_score(model3, X, y, scoring='accuracy', cv=kfold3)\n",
    "r3\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
