#%%
from sklearn.datasets import load_iris
import pandas as pd
import plotly.express as px

# %%
iris = load_iris()
# %%
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.DataFrame(iris.target)
y = y.replace(0, "setosa").replace(1, "versicolor").replace(2, "virginica")
Xy = pd.concat([X, y], axis=1)
Xy.rename(
    columns={
        "sepal length (cm)": "sl",
        "sepal width (cm)": "sw",
        "petal length (cm)": "pl",
        "petal width (cm)": "pw",
        0: "target",
    },
    inplace=True,
)
#%%
px.scatter(Xy, x="pw", y="pl", color="target")

#%%
Xy.iloc[:, :-1].corr()

