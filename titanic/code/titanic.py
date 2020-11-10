#%%
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

#%%
gender_submission = pd.read_csv("..//data//gender_submission.csv")
test = pd.read_csv("..//data//test.csv")
train = pd.read_csv("..//data//train.csv")

#%%
"""
NOTE
데이터 칼럼 정리:
    Name:   삭제
    Sex: 바이너리 변환
    Fare: 가격에 따른 방 위치, 등급, 가용인원 확인
    Cabin: 위치 확인 (층, 출입구, 사고 발생지점)
    Embarked: 원핫 인코딩 필요
"""

#%% Cleaning
train.Sex = train.Sex.apply(lambda x: 0 if x == "male" else 1)

# TODO train.Embarked = LabelEncoder().fit_transform(train.Embarked)
#%%


# %%
train.corr()

#%%
len(train) - train.count()


# %%
lr = LogisticRegression(random_state=0)
lr.fit(train, test)
lr.score(train, test)

