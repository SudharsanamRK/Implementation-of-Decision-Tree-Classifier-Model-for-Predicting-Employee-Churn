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
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Sudharsanam R K
RegisterNumber:  212222040163
```
```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

# Load the dataset
data = pd.read_csv("/content/Employee_EX6.csv")

# Display the first few rows of the dataset
data.head()

# Get information about the dataset
data.info()

# Check for missing values
data.isnull().sum()

# Count the number of employees who left and stayed
data["left"].value_counts()

# Encode the 'salary' column
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])

# Display the first few rows of the modified dataset
data.head()

# Select features (independent variables)
x = data[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", "time_spend_company", "Work_accident", "promotion_last_5years", "salary"]]
x.head()  # No departments and no left

# Select the target variable (dependent variable)
y = data["left"]

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

# Train the Decision Tree classifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train, y_train)

# Predict the target variable for the test set
y_pred = dt.predict(x_test)

# Calculate the accuracy of the model
accuracy = metrics.accuracy_score(y_test, y_pred)
accuracy

# Predict whether an employee will leave based on given features
dt.predict([[0.5, 0.8, 9, 260, 6, 0, 1, 2]])
```

## Output:

## data.head():
![image](https://github.com/SudharsanamRK/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/115523484/bfe1c1fe-1df3-475d-999f-e6cdc0490fff)

## data.info():
![image](https://github.com/SudharsanamRK/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/115523484/7e1d1369-3831-45b1-91f8-da691cda994f)

## isnull() and sum():
![image](https://github.com/SudharsanamRK/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/115523484/a0b491ef-2b44-401e-9384-a136061e8cba)

## data value counts():
![image](https://github.com/SudharsanamRK/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/115523484/81cfe5c9-9e15-4894-ab29-6ad9fcc63365)

## data.head() for salary:
![image](https://github.com/SudharsanamRK/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/115523484/bdd43498-e8e9-4170-b55e-1a8ec1abfdd4)

## x.head():
![image](https://github.com/SudharsanamRK/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/115523484/173ed5b6-b055-468a-a4e9-c4fa7b9a0649)

## accuracy value:
![image](https://github.com/SudharsanamRK/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/115523484/76296615-d85a-4a1c-9c68-d6ee30a9e693)

## data prediction:
![image](https://github.com/SudharsanamRK/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/115523484/9d41da4f-ef7a-41ed-aced-547db844985b)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
