import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.inspection import DecisionBoundaryDisplay

df = pd.read_csv('diamonds_processed.csv')

df["cut"] =df["cut"].map({"Fair": 0, "Good": 1, "Very Good": 2, "Premium": 3, "Ideal": 4})
df["color"] = df["color"].map({c: i for i, c in enumerate("JIHGFED")})
df["clarity"] = df["clarity"].map({c: i for i, c in enumerate(["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"])})
tier_names = [r"Low", r"Medium", r"High"]
X = df[["log_carat", "cut", "color", "clarity"]]
y = df["log_price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

VI = np.linalg.inv(np.cov(X_train.vlues, rowvar=False))
model = make_pipeline(KNeighborsClassifier(degree=3, metric="mahalanobis", metric_params={"VI":VI}))

