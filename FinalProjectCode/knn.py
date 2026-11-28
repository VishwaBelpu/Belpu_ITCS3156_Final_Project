import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay

df = pd.read_csv('diamonds_processed.csv')

df["cut"] =df["cut"].map({"Fair": 0, "Good": 1, "Very Good": 2, "Premium": 3, "Ideal": 4})
df["color"] = df["color"].map({c: i for i, c in enumerate("JIHGFED")})
df["clarity"] = df["clarity"].map({c: i for i, c in enumerate(["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"])})
tier_names = [r"Low", r"Medium", r"High"]
X = df[["log_carat", "cut", "color", "clarity"]]
y = pd.qcut(df["log_price"], 3, labels=False)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

VI = np.linalg.inv(np.cov(X_train.values, rowvar=False))
model = make_pipeline(KNeighborsClassifier(n_neighbors=13, metric="mahalanobis", metric_params={"VI":VI}))
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=tier_names, cmap="Purples")
plt.title("KNN Confusion Matrix")
plt.tight_layout()
plt.show()

xx, yy = np.meshgrid(np.linspace(X["log_carat"].min(), X["log_carat"].max(), 200), np.linspace(-.5, 7.5, 200))
grid = pd.DataFrame({"log_carat": xx.ravel(), "cut": X["cut"].median(), "color": X["color"].median(), "clarity": yy.ravel()})
Z = model.predict(grid).reshape(xx.shape)

sample = X_test.sample(300, random_state=42)
jitter = np.random.default_rng(42).uniform(-.25,.25,len(sample))
colors = ['red', 'yellow', 'blue']

plt.figure(figsize=(8, 5))
plt.pcolormesh(xx, yy, Z, cmap=ListedColormap(['lawngreen', 'lightseagreen', 'darkgreen']), shading="auto")
plt.scatter(sample["log_carat"], sample["clarity"] + jitter, c=y_test[sample.index], cmap=ListedColormap(colors), edgecolor="k", s=25)
plt.yticks(range(8), ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"])
plt.xlabel("log(carat)")
plt.ylabel("Clarity")
plt.title("KNN Decision Boundary (Mahanalobis, k=13)")
handles = [plt.Line2D([], [], marker="o", ls="", markerfacecolor=c, markeredgecolor="k") for c in colors]
plt.legend(handles, tier_names, title="Price tier", loc="lower right")
plt.tight_layout()
plt.show()