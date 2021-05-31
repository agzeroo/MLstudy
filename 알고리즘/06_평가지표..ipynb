{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 기본\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "# 경고 뜨지 않게 설정\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "# 그래프 설정\n",
    "plt.rcParams['font.family'] = 'Malgun Gothic'\n",
    "# plt.rcParams['font.family'] = 'AppleGothic'\n",
    "plt.rcParams['font.size'] = 16\n",
    "plt.rcParams['figure.figsize'] = 20, 10\n",
    "plt.rcParams['axes.unicode_minus'] = False\n",
    "\n",
    "# 데이터 전처리 알고리즘\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "\n",
    "# 학습용과 검증용으로 나누는 함수\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "# 교차 검증\n",
    "# 지표를 하나만 설정할 경우\n",
    "from sklearn.model_selection import cross_val_score\n",
    "# 지표를 하나 이상 설정할 경우\n",
    "from sklearn.model_selection import cross_validate\n",
    "from sklearn.model_selection import KFold\n",
    "from sklearn.model_selection import StratifiedKFold\n",
    "\n",
    "# 평가함수\n",
    "# 분류용\n",
    "from sklearn.metrics import accuracy_score\n",
    "from sklearn.metrics import precision_score\n",
    "from sklearn.metrics import recall_score\n",
    "from sklearn.metrics import f1_score\n",
    "from sklearn.metrics import roc_auc_score\n",
    "\n",
    "# 회귀용\n",
    "from sklearn.metrics import r2_score\n",
    "from sklearn.metrics import mean_squared_error"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 분류\n",
    "- 모든 평가지표는 1에 가까울 수록 성능이 좋다고 판단한다.\n",
    "- accuracy : 진짜 결과와 예측 결과를 비교해 얼마나 일치하는지.. 결과의 종류에 상관없이 전체를 보고 판단한다. 결과를 내기위한 결론 비율의 차이가 비슷한 경우도 있기 때문에 판단이 애매한 경우가 있을 수 있다.\n",
    "- precision : 정밀도. 모델이 True라고 예측한것 중에 실제 True인 것의 비율. 각 결과 데이터별로 확인하기 때문에 정확도 보다는 신뢰성이 높다.\n",
    "- recall : 재현율. 실제 True인 데이터를 모델이 True라고 예측한 비율. 각 결과 데이터 별로 확인하기 때문에 정확도 보다는 신뢰성이 높다.\n",
    "- f1 score : 정밀도 재현율을 모두 이용하여 오차 값을 수정한것.\n",
    "- roc_auc : 실제 false인 데이터를 true로 잘못 분류한 비율과 실제 True인 데이터 중에 True로 잘 분류한 비율을 각각 x, y 좌표로 설정하여 비율을 비교하는 그래프가 roc 곡선이다. 이 곡선의 면적을 계산한것이 auc이다. 그래프가 좌상쪽으로 있으면 최대 면적이 된다.\n",
    "- 이진 분류(결과가 두 개인 경우)에는 roc_auc가 좀더 정밀하고 결과의 종류가 3가지 이상인 경우에는 f1 score가 좀더 정밀하다고 하지만 차이는 크지 않다."
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
       "      <th>attr1</th>\n",
       "      <th>attr2</th>\n",
       "      <th>target</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>9.963466</td>\n",
       "      <td>4.596765</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>11.032954</td>\n",
       "      <td>-0.168167</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>11.541558</td>\n",
       "      <td>5.211161</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>8.692890</td>\n",
       "      <td>1.543220</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>8.106227</td>\n",
       "      <td>4.286960</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>8.309889</td>\n",
       "      <td>4.806240</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>11.930271</td>\n",
       "      <td>4.648663</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>9.672847</td>\n",
       "      <td>-0.202832</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>8.348103</td>\n",
       "      <td>5.134156</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>8.674947</td>\n",
       "      <td>4.475731</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>9.177484</td>\n",
       "      <td>5.092832</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>10.240289</td>\n",
       "      <td>2.455444</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12</th>\n",
       "      <td>8.689371</td>\n",
       "      <td>1.487096</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13</th>\n",
       "      <td>8.922295</td>\n",
       "      <td>-0.639932</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>14</th>\n",
       "      <td>9.491235</td>\n",
       "      <td>4.332248</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15</th>\n",
       "      <td>9.256942</td>\n",
       "      <td>5.132849</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16</th>\n",
       "      <td>7.998153</td>\n",
       "      <td>4.852505</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>17</th>\n",
       "      <td>8.183781</td>\n",
       "      <td>1.295642</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>18</th>\n",
       "      <td>8.733709</td>\n",
       "      <td>2.491624</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>19</th>\n",
       "      <td>9.322983</td>\n",
       "      <td>5.098406</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>20</th>\n",
       "      <td>10.063938</td>\n",
       "      <td>0.990781</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>21</th>\n",
       "      <td>9.500490</td>\n",
       "      <td>-0.264303</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>22</th>\n",
       "      <td>8.344688</td>\n",
       "      <td>1.638243</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23</th>\n",
       "      <td>9.501693</td>\n",
       "      <td>1.938246</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>24</th>\n",
       "      <td>9.150723</td>\n",
       "      <td>5.498322</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25</th>\n",
       "      <td>11.563957</td>\n",
       "      <td>1.338940</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        attr1     attr2  target\n",
       "0    9.963466  4.596765       1\n",
       "1   11.032954 -0.168167       0\n",
       "2   11.541558  5.211161       1\n",
       "3    8.692890  1.543220       0\n",
       "4    8.106227  4.286960       0\n",
       "5    8.309889  4.806240       1\n",
       "6   11.930271  4.648663       1\n",
       "7    9.672847 -0.202832       0\n",
       "8    8.348103  5.134156       1\n",
       "9    8.674947  4.475731       1\n",
       "10   9.177484  5.092832       1\n",
       "11  10.240289  2.455444       1\n",
       "12   8.689371  1.487096       0\n",
       "13   8.922295 -0.639932       0\n",
       "14   9.491235  4.332248       1\n",
       "15   9.256942  5.132849       1\n",
       "16   7.998153  4.852505       1\n",
       "17   8.183781  1.295642       0\n",
       "18   8.733709  2.491624       0\n",
       "19   9.322983  5.098406       1\n",
       "20  10.063938  0.990781       0\n",
       "21   9.500490 -0.264303       0\n",
       "22   8.344688  1.638243       0\n",
       "23   9.501693  1.938246       0\n",
       "24   9.150723  5.498322       1\n",
       "25  11.563957  1.338940       0"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df1 = pd.read_csv('data/forge.csv')\n",
    "df1"
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
       "      <th>attr1</th>\n",
       "      <th>attr2</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>9.963466</td>\n",
       "      <td>4.596765</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>11.032954</td>\n",
       "      <td>-0.168167</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>11.541558</td>\n",
       "      <td>5.211161</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>8.692890</td>\n",
       "      <td>1.543220</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>8.106227</td>\n",
       "      <td>4.286960</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>8.309889</td>\n",
       "      <td>4.806240</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>11.930271</td>\n",
       "      <td>4.648663</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>9.672847</td>\n",
       "      <td>-0.202832</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>8.348103</td>\n",
       "      <td>5.134156</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>8.674947</td>\n",
       "      <td>4.475731</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>9.177484</td>\n",
       "      <td>5.092832</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>10.240289</td>\n",
       "      <td>2.455444</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12</th>\n",
       "      <td>8.689371</td>\n",
       "      <td>1.487096</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13</th>\n",
       "      <td>8.922295</td>\n",
       "      <td>-0.639932</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>14</th>\n",
       "      <td>9.491235</td>\n",
       "      <td>4.332248</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15</th>\n",
       "      <td>9.256942</td>\n",
       "      <td>5.132849</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16</th>\n",
       "      <td>7.998153</td>\n",
       "      <td>4.852505</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>17</th>\n",
       "      <td>8.183781</td>\n",
       "      <td>1.295642</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>18</th>\n",
       "      <td>8.733709</td>\n",
       "      <td>2.491624</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>19</th>\n",
       "      <td>9.322983</td>\n",
       "      <td>5.098406</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>20</th>\n",
       "      <td>10.063938</td>\n",
       "      <td>0.990781</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>21</th>\n",
       "      <td>9.500490</td>\n",
       "      <td>-0.264303</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>22</th>\n",
       "      <td>8.344688</td>\n",
       "      <td>1.638243</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23</th>\n",
       "      <td>9.501693</td>\n",
       "      <td>1.938246</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>24</th>\n",
       "      <td>9.150723</td>\n",
       "      <td>5.498322</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25</th>\n",
       "      <td>11.563957</td>\n",
       "      <td>1.338940</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        attr1     attr2\n",
       "0    9.963466  4.596765\n",
       "1   11.032954 -0.168167\n",
       "2   11.541558  5.211161\n",
       "3    8.692890  1.543220\n",
       "4    8.106227  4.286960\n",
       "5    8.309889  4.806240\n",
       "6   11.930271  4.648663\n",
       "7    9.672847 -0.202832\n",
       "8    8.348103  5.134156\n",
       "9    8.674947  4.475731\n",
       "10   9.177484  5.092832\n",
       "11  10.240289  2.455444\n",
       "12   8.689371  1.487096\n",
       "13   8.922295 -0.639932\n",
       "14   9.491235  4.332248\n",
       "15   9.256942  5.132849\n",
       "16   7.998153  4.852505\n",
       "17   8.183781  1.295642\n",
       "18   8.733709  2.491624\n",
       "19   9.322983  5.098406\n",
       "20  10.063938  0.990781\n",
       "21   9.500490 -0.264303\n",
       "22   8.344688  1.638243\n",
       "23   9.501693  1.938246\n",
       "24   9.150723  5.498322\n",
       "25  11.563957  1.338940"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "0     1\n",
       "1     0\n",
       "2     1\n",
       "3     0\n",
       "4     0\n",
       "5     1\n",
       "6     1\n",
       "7     0\n",
       "8     1\n",
       "9     1\n",
       "10    1\n",
       "11    1\n",
       "12    0\n",
       "13    0\n",
       "14    1\n",
       "15    1\n",
       "16    1\n",
       "17    0\n",
       "18    0\n",
       "19    1\n",
       "20    0\n",
       "21    0\n",
       "22    0\n",
       "23    0\n",
       "24    1\n",
       "25    0\n",
       "Name: target, dtype: int64"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# 입력과 결과로 나눈다.\n",
    "X = df1.drop('target', axis=1)\n",
    "y = df1['target']\n",
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
       "LogisticRegression()"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.linear_model import LogisticRegression\n",
    "\n",
    "model1 = LogisticRegression()\n",
    "model1.fit(X, y)"
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
       "array([1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0,\n",
       "       0, 0, 1, 0], dtype=int64)"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 예측  결과를 가져온다.\n",
    "y_pred = model1.predict(X)\n",
    "y_pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.9230769230769231\n"
     ]
    }
   ],
   "source": [
    "# 정확도\n",
    "r1 = accuracy_score(y, y_pred)\n",
    "print(r1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9230769230769231"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 정밀도\n",
    "r2 = precision_score(y, y_pred)\n",
    "r2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9230769230769231"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 재현율\n",
    "r3 = recall_score(y, y_pred)\n",
    "r3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9230769230769231"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# f1 score\n",
    "r4 = f1_score(y, y_pred)\n",
    "r4"
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
       "0.9230769230769231"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# roc_auc\n",
    "r5 = roc_auc_score(y, y_pred)\n",
    "r5"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.9166666666666666\n",
      "0.85\n",
      "0.85\n",
      "0.8333333333333333\n"
     ]
    }
   ],
   "source": [
    "# 교차 검증\n",
    "model2 = LogisticRegression()\n",
    "\n",
    "kfold = KFold(n_splits=10, shuffle=True, random_state=1)\n",
    "\n",
    "r1 = cross_val_score(model2, X, y, scoring='accuracy', cv=kfold)\n",
    "r2 = cross_val_score(model2, X, y, scoring='precision', cv=kfold)\n",
    "r3 = cross_val_score(model2, X, y, scoring='recall', cv=kfold)\n",
    "r4 = cross_val_score(model2, X, y, scoring='f1', cv=kfold)\n",
    "# r5 = cross_val_score(model2, X, y, scoring='roc_auc', cv=kfold)\n",
    "\n",
    "print(r1.mean())\n",
    "print(r2.mean())\n",
    "print(r3.mean())\n",
    "print(r4.mean())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.9166666666666666\n",
      "0.8333333333333333\n"
     ]
    }
   ],
   "source": [
    "# 동시 여러 지표를 이용해 평가를 할 경우\n",
    "s1 = ['accuracy', 'f1']\n",
    "r10 = cross_validate(model2, X, y, scoring=s1, cv=kfold)\n",
    "print(r10['test_accuracy'].mean())\n",
    "print(r10['test_f1'].mean())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 회귀\n",
    "- mean_squared_error : 진짜 결과 값과 예측 결과값 간의 거리를 측정하여 거리의 제곰을 계산하여 오차 면적을 구한다. 모든 데이터에 대한 오차 면적의 합을 구해 오차의 정도를 계산한다.\n",
    "- neg_mean_squared_error : sklearn은 대부분의 평가 지표가 1에 가까울 수록 좋은 성능을 보인다고 평가한다. 이에 mean_squared_error는 오차가 심하면 값이 더 커지므로 음수로 변환하여 전달한다. 이를 통해 값으 크면 성능 좋다는 일관성을 유지해준다.\n",
    "- r2 : 분산을 기반으로 평가\n",
    "- neg_mean_squared_error는 예측 값과 결과 값이 얼마나 차이가 나는지를 보는 것이고 r2는 패턴이 얼마나 유산하지를 평가한다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
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
       "      <th>data</th>\n",
       "      <th>target</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>-0.752759</td>\n",
       "      <td>-0.448221</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2.704286</td>\n",
       "      <td>0.331226</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1.391964</td>\n",
       "      <td>0.779321</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.591951</td>\n",
       "      <td>0.034979</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>-2.063888</td>\n",
       "      <td>-1.387736</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       data    target\n",
       "0 -0.752759 -0.448221\n",
       "1  2.704286  0.331226\n",
       "2  1.391964  0.779321\n",
       "3  0.591951  0.034979\n",
       "4 -2.063888 -1.387736"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df1 = pd.read_csv('data/wave.csv')\n",
    "df1.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
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
       "      <th>data</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>-0.752759</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2.704286</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1.391964</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.591951</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>-2.063888</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>-2.064033</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>-2.651498</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>2.197057</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>0.606690</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>1.248435</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>-2.876493</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>2.819459</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12</th>\n",
       "      <td>1.994656</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13</th>\n",
       "      <td>-1.725965</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>14</th>\n",
       "      <td>-1.909050</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15</th>\n",
       "      <td>-1.899573</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16</th>\n",
       "      <td>-1.174547</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>17</th>\n",
       "      <td>0.148539</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>18</th>\n",
       "      <td>-0.408330</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>19</th>\n",
       "      <td>-1.252625</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>20</th>\n",
       "      <td>0.671117</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>21</th>\n",
       "      <td>-2.163037</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>22</th>\n",
       "      <td>-1.247132</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23</th>\n",
       "      <td>-0.801829</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>24</th>\n",
       "      <td>-0.263580</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25</th>\n",
       "      <td>1.711056</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>26</th>\n",
       "      <td>-1.801957</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>27</th>\n",
       "      <td>0.085407</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>28</th>\n",
       "      <td>0.554487</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>29</th>\n",
       "      <td>-2.721298</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>30</th>\n",
       "      <td>0.645269</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>31</th>\n",
       "      <td>-1.976855</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>32</th>\n",
       "      <td>-2.609690</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>33</th>\n",
       "      <td>2.693313</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>34</th>\n",
       "      <td>2.793792</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>35</th>\n",
       "      <td>1.850384</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>36</th>\n",
       "      <td>-1.172317</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>37</th>\n",
       "      <td>-2.413967</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>38</th>\n",
       "      <td>1.105398</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>39</th>\n",
       "      <td>-0.359085</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        data\n",
       "0  -0.752759\n",
       "1   2.704286\n",
       "2   1.391964\n",
       "3   0.591951\n",
       "4  -2.063888\n",
       "5  -2.064033\n",
       "6  -2.651498\n",
       "7   2.197057\n",
       "8   0.606690\n",
       "9   1.248435\n",
       "10 -2.876493\n",
       "11  2.819459\n",
       "12  1.994656\n",
       "13 -1.725965\n",
       "14 -1.909050\n",
       "15 -1.899573\n",
       "16 -1.174547\n",
       "17  0.148539\n",
       "18 -0.408330\n",
       "19 -1.252625\n",
       "20  0.671117\n",
       "21 -2.163037\n",
       "22 -1.247132\n",
       "23 -0.801829\n",
       "24 -0.263580\n",
       "25  1.711056\n",
       "26 -1.801957\n",
       "27  0.085407\n",
       "28  0.554487\n",
       "29 -2.721298\n",
       "30  0.645269\n",
       "31 -1.976855\n",
       "32 -2.609690\n",
       "33  2.693313\n",
       "34  2.793792\n",
       "35  1.850384\n",
       "36 -1.172317\n",
       "37 -2.413967\n",
       "38  1.105398\n",
       "39 -0.359085"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "0    -0.448221\n",
       "1     0.331226\n",
       "2     0.779321\n",
       "3     0.034979\n",
       "4    -1.387736\n",
       "5    -2.471962\n",
       "6    -1.527308\n",
       "7     1.494172\n",
       "8     1.000324\n",
       "9     0.229562\n",
       "10   -1.059796\n",
       "11    0.778964\n",
       "12    0.754188\n",
       "13   -1.513697\n",
       "14   -1.673034\n",
       "15   -0.904970\n",
       "16    0.084485\n",
       "17   -0.527347\n",
       "18   -0.541146\n",
       "19   -0.340907\n",
       "20    0.217782\n",
       "21   -1.124691\n",
       "22    0.372991\n",
       "23    0.097563\n",
       "24   -0.986181\n",
       "25    0.966954\n",
       "26   -1.134550\n",
       "27    0.697986\n",
       "28    0.436558\n",
       "29   -0.956521\n",
       "30    0.035279\n",
       "31   -2.085817\n",
       "32   -0.474110\n",
       "33    1.537083\n",
       "34    0.868933\n",
       "35    1.876649\n",
       "36    0.094526\n",
       "37   -1.415024\n",
       "38    0.254389\n",
       "39    0.093989\n",
       "Name: target, dtype: float64"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# 입력과 결과로 나눈다.\n",
    "X = df1.drop('target', axis=1)\n",
    "y = df1['target']\n",
    "\n",
    "display(X)\n",
    "display(y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LinearRegression()"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.linear_model import LinearRegression\n",
    "\n",
    "model1 = LinearRegression()\n",
    "\n",
    "model1.fit(X, y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([-0.4247581 ,  1.241749  ,  0.60912948,  0.22347442, -1.05680237,\n",
       "       -1.05687213, -1.34006639,  0.99723363,  0.2305796 ,  0.53994012,\n",
       "       -1.4485276 ,  1.29726956,  0.89966396, -0.89390291, -0.98216101,\n",
       "       -0.97759239, -0.62808534,  0.00972258, -0.25872207, -0.66572401,\n",
       "        0.2616375 , -1.10459809, -0.66307602, -0.44841267, -0.18894382,\n",
       "        0.76295139, -0.93053569, -0.02071088,  0.20541473, -1.37371387,\n",
       "        0.24917706, -1.01484719, -1.31991244,  1.23645953,  1.28489654,\n",
       "        0.83011615, -0.62701075, -1.22556192,  0.47098739, -0.23498304])"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 예측\n",
    "y_pred = model1.predict(X)\n",
    "y_pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.327028126409278"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# mean_squared_error\n",
    "r1 = mean_squared_error(y, y_pred)\n",
    "r1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.6855982772339115"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# r2\n",
    "r2 = r2_score(y, y_pred)\n",
    "r2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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n7/5zW+yZcuWpPaZin1kigSAhBBCCCGEEEKIFNrd7mJ9YznWImN8m6ZpNDtsUgKWQ5qamtiwYcO8X3/gwAHuu+++pPaRTaZsL0AIIYQQQgghhCgUwXCEVzrdvOX81hnPNTusUgKWQ26//XZuv/32eb/+i1/8IqFQiJtvvnnB+8gmyQASQgghhBBCCCFS5EDPCP5gZEr/n5gWh02mgCVB13V0Xc/2MvKWBICEEEIIIYQQQogU2d0+swF0TEullRF/iGFfMNPLykl33XUXb3/723n00Uc5++yzsVgsrF+/ngceeCD+Gk3T+MMf/sDNN99MUVERO3fuBOCZZ57hwgsvxGKxsHz5cr7//e9P2bfH4+Fv/uZvqKmpobS0lLe+9a2MjIxMec1nP/tZLr300inbXnvtNW699VbsdjslJSVceeWV8XX8+Mc/5mc/+xmapsW3z7aP5557jmuuuYaysjJKS0u5/vrrefXVV6e8RtM0HnjgAT74wQ/icDhobGzks5/9bFoDXBIAEkIIIYQQQgghUmRXu4vGCgsNFdYZz7XERsFLI+i4/fv386lPfYq7776b5557jssvv5zXv/71HDp0KP6au+++m2uvvZY9e/awevVqXnjhBa699lquvPJK/vznP/PJT36Sv/u7v2P79u3x99x555088sgj/OAHP2Dnzp00Njbyuc997oxrueiii7BYLNx33308/fTTXHjhhQCcOHGC2267jde//vWcOHGCX/ziF7Pu47nnnuOqq66ira2N7du389BDD1FcXMxll11GR0fHlNd++tOfpqamhqeeeoqPfvSjfP7zn+f3v//9Qv8qz0h6AAkhhBBCCCGEECmyp93FlqWVsz7XUqkCQJ0uLxuaKpI/2EOfgN59ye8nWfVnwQ3/saC3Hj58mMOHD9PY2AjA//zP//Dqq6/yta99je9+97sArFq1ig9+8IPx99x5553cdttt/Md/qGNu2rSJw4cP89WvfpWtW7fy1FNP8fDDD7N//35Wr14df83x48d54YUX5lzLxz72MS655BJ++ctfxrdt3rwZgKVLl1JaWkooFGLp0qVz7uMTn/gEb3jDG/jv//7v+LZLLrmEDRs28JWvfIX//M//jG8/55xz+MxnPgPAxo0beeihh7j33nt54xvfOK+/u0RJBpAQQgghhBBCCJEC3W4fPcN+NrfaZ31+IgNI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\n",
      "text/plain": [
       "<Figure size 1440x720 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(y, label='original')\n",
    "plt.plot(y_pred, label='prediction')\n",
    "\n",
    "plt.legend()\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "-0.3645458796480712"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "0.11328621026484162"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# 교차 검증\n",
    "model2 = LinearRegression()\n",
    "\n",
    "kfold = KFold(n_splits=10, shuffle=True, random_state=1)\n",
    "\n",
    "r1 = cross_val_score(model2, X, y, scoring='neg_mean_squared_error', cv=kfold)\n",
    "r2 = cross_val_score(model2, X, y, scoring='r2', cv=kfold)\n",
    "\n",
    "display(r1.mean())\n",
    "display(r2.mean())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "-0.3645458796480712\n",
      "0.11328621026484162\n"
     ]
    }
   ],
   "source": [
    "# 동시에 여러 지표 확인\n",
    "s1 = ['neg_mean_squared_error', 'r2']\n",
    "r2 = cross_validate(model2, X, y, scoring=s1, cv=kfold)\n",
    "\n",
    "print(r2['test_neg_mean_squared_error'].mean())\n",
    "print(r2['test_r2'].mean())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n"
   ]
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
