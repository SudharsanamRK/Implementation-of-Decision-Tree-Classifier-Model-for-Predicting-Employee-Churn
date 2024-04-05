# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packagesprint the present data

2.print the present data

3.print the null values

4.using decisiontreeclassifier, find the predicted values

5.print the result
## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Sudharsanam R K
RegisterNumber:  212222040163
*/
```
```python
import pandas as pd
data=pd.read_csv("/content/Employee_EX6.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
![image](https://github.com/selva258963/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121961701/7c88a908-d75f-4c62-bd28-a002dfe0004a)
![image](https://github.com/selva258963/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121961701/4ff20919-59e8-41e3-b388-5a6b5bd5bf93)
![image](https://github.com/selva258963/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121961701/9c33630e-3de5-47f7-8d43-92bc4cdbc98d)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
