import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

data = pd.read_csv("/content/sample_data/CarPrice.csv")
data.head()

data.isnull().sum()

data.info()

print(data.describe())

data.CarName.unique()

sns.set_style("whitegrid")
plt.figure(figsize=(15, 10))
sns.distplot(data.price)
plt.show()

col=data[['car_ID','symboling','wheelbase','carlength','carwidth','carheight']]
correlation_matrix=col.corr()
print(correlation_matrix)


plt.figure(figsize=(20, 15))
sns.heatmap(correlation_matrix, cmap="coolwarm", annot=True)
plt.title("correlation Matrix Heatmap")
plt.show()

sns.countplot(x=data['symboling'])

sns.boxplot(x=data['price'])

data['peakrpm'].value_counts()

sns.scatterplot(x=data['peakrpm'], y=data['price'])

predict = "price"
data = data[["symboling", "wheelbase", "carlength",
             "carwidth", "carheight", "curbweight",
             "enginesize", "boreratio", "stroke",
             "compressionratio", "horsepower", "peakrpm",
             "citympg", "highwaympg", "price"]]
x = np.array(data.drop([predict],axis= 1))
y = np.array(data[predict])

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(xtrain, ytrain)
predictions = model.predict(xtest)

from sklearn.metrics import mean_absolute_error
model.score(xtest, predictions)


predict = "price"
data = data[["symboling", "wheelbase", "carlength",
             "carwidth", "carheight", "curbweight",
             "enginesize", "boreratio", "stroke",
             "compressionratio", "horsepower", "peakrpm",
             "citympg", "highwaympg", "price"]]

# Fix: use axis=1
x = np.array(data.drop([predict], axis=1))
y = np.array(data[predict])

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(xtrain, ytrain)

predictions = model.predict(xtest)

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

# R² Score (best metric for regression)
r2 = r2_score(ytest, predictions)

# Mean Absolute Error
mae = mean_absolute_error(ytest, predictions)

# Root Mean Squared Error
rmse = np.sqrt(mean_squared_error(ytest, predictions))
accuracy = r2_score(ytest, predictions) * 100


print("R² Score (Accuracy):", r2)
print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)
print(f"Model Accuracy: {accuracy:.2f}%")


