# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the required libraries and read the dataframe.

2.Assign hours to X and scores to Y.

3.Implement training set and test set of the dataframe

4.Plot the required graph both for test data and training data.

5.Find the values of MSE , MAE and RMSE.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: HARISH RAGAVENDRA V
RegisterNumber: 212223080017 
*/
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(x_test,y_test,color='blue')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
Dataset

![image](https://github.com/23011811/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/160568623/b5185d6a-faa8-472b-a723-a0c546d37645)

df.head()

![image](https://github.com/23011811/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/160568623/32877a85-4a54-433c-8717-7ca06ab7dc73)

f.tail()

![image](https://github.com/23011811/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/160568623/531ba922-4c83-42a1-9499-4992e445ca19)

X and Y values
![image](https://github.com/23011811/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/160568623/3c395295-2ec1-43c7-ab7c-0d18c4dfa88b)

Predication values of X and Y

![image](https://github.com/23011811/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/160568623/89ba5598-9ca8-438b-8f4e-483ee5090df9)

MSE,MAE and RMSE


![image](https://github.com/23011811/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/160568623/8fb90636-3b3e-45fb-94e1-656986207882)

Training Set


![image](https://github.com/23011811/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/160568623/1b4dd939-a6e6-441a-bba5-864c97590057)

Testing Set


![image](https://github.com/23011811/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/160568623/3a27ac4c-87a6-4a03-a459-2665dd21d167)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
