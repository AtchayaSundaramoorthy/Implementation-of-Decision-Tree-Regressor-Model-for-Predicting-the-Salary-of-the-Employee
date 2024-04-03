# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packagesprint the present data 
2. print the present data
3. print the null values
4. using decisiontreeRegressor, find the predicted values,mse,r2
5.print the result

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: 
RegisterNumber:  212222040021
*/
```
```
import pandas as pd
data=pd.read_csv("/content/Salary_EX7.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
x_train
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt
clf=DecisionTreeRegressor(criterion='gini')
plt.figure(figsize=(20,8))
plot_tree(dt,feature_names=x.columns,filled=True)
plt.show()
```

## Output:
![image](https://github.com/AtchayaSundaramoorthy/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119393516/a76702c6-0601-4f09-aa10-6fdd6a024b89)

![image](https://github.com/AtchayaSundaramoorthy/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119393516/b683cf48-e8de-460e-b977-7ba0cb5df71f)

![image](https://github.com/AtchayaSundaramoorthy/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119393516/21148867-9eea-4d9a-8031-c2aef75e06b1)

![image](https://github.com/AtchayaSundaramoorthy/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119393516/588e0a5f-5784-4eec-b1c6-76202513b5bc)

![image](https://github.com/AtchayaSundaramoorthy/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119393516/7f29d73c-e6e5-4bd9-a488-27dbf1f725de)

![image](https://github.com/AtchayaSundaramoorthy/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119393516/318a99eb-1eae-4609-a1ec-ddcc02765a6b)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
