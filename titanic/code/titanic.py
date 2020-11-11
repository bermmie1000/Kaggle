#%%
import pandas as pd
from sklearn import *

#%%
gender_submission = pd.read_csv("..//data//gender_submission.csv")
test = pd.read_csv("..//data//test.csv")
train = pd.read_csv("..//data//train.csv")

# %%
lr = linear_model.LogisticRegression(random_state=0)
lr.fit(train, test)
lr.score(train, test)
# %%
train.dtypes


# %%
train
# %%
