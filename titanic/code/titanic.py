#%%
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

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
train = train.drop(columns=["PassengerId", "Name", "Cabin", "Ticket"])
train.Sex = train.Sex.apply(lambda x: 0 if x == "male" else 1)
train = train.dropna(axis=0)
train.Embarked = LabelEncoder().fit_transform(train.Embarked)

#%%
test = test.drop(columns=["PassengerId", "Name", "Cabin", "Ticket"])
test.Sex = test.Sex.apply(lambda x: 0 if x == "male" else 1)
test.Embarked = LabelEncoder().fit_transform(test.Embarked)

#%%
X = train.drop(columns="Survived")
y = train.Survived

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)
# %%
lr = LogisticRegression(random_state=0)
lr.fit(X_train, y_train)
lr.score(X_train, y_train)

# %%
pd.DataFrame(X.columns, lr.coef_[0])
