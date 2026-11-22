import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(42)

df = pd.read_csv('diamonds.csv')
# Unfortunately, there is a skew the in the diamonds dataset when it comes to carat. However, to fix this
# we will create a new feature called log of carat in order to fix this skew. Price will also have 
# to get the same treatment in order to ensure that the multiple linear regression model has better time
# predicting the price. Skew is something that is present in every dataset, so the log function in this 
# dataset helps the data be more evenly distributed.

df["log_carat"] = np.log(df["carat"])
df["log_price"] = np.log(df["price"])
df.to_csv("diamonds_processed.csv", index=False)

# A new csv will be saved with the logged carat and price.

df = pd.read_csv('diamonds_processed.csv')
print(df.describe())
print("\n", df.head())

corr = df.select_dtypes("number").corr()  
plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True)
plt.title("Correlation Between Diamond Features")
plt.tight_layout()
plt.show()


sample_df = df.sample(frac=.1, random_state=42)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(sample_df['table'], sample_df['depth'], sample_df['log_price'], c=sample_df['log_price'], cmap='PuBuGn', s=8, alpha=0.7)
ax.set_xlabel('table')
ax.set_ylabel('depth')
ax.set_zlabel('log price')
plt.show()