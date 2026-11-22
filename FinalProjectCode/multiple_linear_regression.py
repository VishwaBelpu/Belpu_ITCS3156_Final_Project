import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression

df = pd.read_csv('diamonds_processed.csv')

df["cut"] =df["cut"].map({"Fair": 0, "Good": 1, "Very Good": 2, "Premium": 3, "Ideal": 4})
df["color"] = df["color"].map({c: i for i, c in enumerate("JIHGFED")})

X = df[["log_carat", "cut", "color", "clarity", "depth","table"]]
y = df[["log_price"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = make_pipeline(PolynomialFeatures(degree=3, include_bias=False), LinearRegression())
model.fit(X_train, y_train)

pred = np.exp(model.predict(X_test))
actual = np.exp(y_test)
print("R2:", r2_score(actual, pred))
print("RMSE:", mean_squared_error(actual, pred) ** 0.5)