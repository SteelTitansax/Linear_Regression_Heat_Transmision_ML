import pandas as pd
import numpy as np
import copy
import tensorflow as tf
from sklearn.linear_model import  LinearRegression
import matplotlib.pyplot as plt

dataset_cols = ["Ln(r1/r)","-(T-T1)"]
df = pd.read_csv("temperaturavsradiolaton.csv")

print(df)



plt.scatter(df["Ln(r1/r)"], df["-(T-T1)"])
plt.title("ln(r1/r) vs (T-T1)")
plt.ylabel("Ln(r1/r)")
plt.xlabel("-(T-T1)")
plt.show()



train, val, test = np.split(df.sample(frac=1), [int(0.6 * len(df)), int(0.8 * len(df))])

def get_xy(dataframe, y_label, x_labels=None):
  dataframe = copy.deepcopy(dataframe)
  if x_labels is None:
    X = dataframe[[c for c in dataframe.columns if c!=y_label]].values
  else:
    if len(x_labels) == 1:
      X = dataframe[x_labels[0]].values.reshape(-1, 1)
    else:
      X = dataframe[x_labels].values

  y = dataframe[y_label].values.reshape(-1, 1)
  data = np.hstack((X, y))

  return data, X, y

_, X_train_temp, y_train_temp = get_xy(train, "Ln(r1/r)", x_labels=["-(T-T1)"])
_, X_val_temp, y_val_temp = get_xy(val, "Ln(r1/r)", x_labels=["-(T-T1)"])
_, X_test_temp, y_test_temp = get_xy(test, "Ln(r1/r)", x_labels=["-(T-T1)"])

temp_reg = LinearRegression()
temp_reg.fit(X_train_temp, y_train_temp)
print("Coeficient " + str(temp_reg.coef_))
print("Intercept" + str(temp_reg.intercept_))

LinearRegression()
print("r linear regression  trained : ")
print(temp_reg.score(X_test_temp, y_test_temp))

