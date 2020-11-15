#%%
from sklearn.datasets import load_iris
import pandas as pd

# %%
iris = load_iris()
# %%
X = pd.DataFrame(iris.data, columns=[iris.feature_names])
y = pd.DataFrame(iris.target)
y = y.apply(lambda x: "setosa" if x == 0 else "versicolor" if x == 1 else "virginica")
Xy = pd.concat([X, y], axis=1)
#%%
Xy
#%%
