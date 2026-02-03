# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and preprocess the dataset byremoving unneceswsary columns andconverting categorical attributes into n umerical values.
2. Split the dataset into training and testing sets to prepare data for model evaluation. 
3. Train the Logistic Regression model using the training dataset. 
4. Train the Logistic Regression model using the training dataset. 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Dharshini K
RegisterNumber:25002639  
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data =pd.read_csv("C:/Users/acer/Downloads/50_Startups.csv")
x = data["R&D Spend"].values
y = data["Profit"].values

x_mean = np.mean(x)
x_std = np.std(x)
x = (x - x_mean)/ x_std
w = 0.0
b = 0.0
alpha = 0.01
epochs = 100
n = len(x)

losses=[]
for _ in range(epochs):
    y_hat = w* x + b
    loss = np.mean((y_hat - y) ** 2)
    losses.append(loss)
    
    dw = (2/n) * np.sum((y_hat-y) * x)
    db = (2/n) * np.sum(y_hat - y)
    
    w -=alpha * dw
    b -=alpha * db

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(losses)
plt.xlabel("Iterations")
plt.ylabel("Loss (MSE)")
plt.title("Loss vs Iterations")

plt.subplot(1,2,2)
plt.scatter(x,y)
x_sorted = np.argsort(x)
plt.plot(x[x_sorted],(w*x+b)[x_sorted],color='red')
plt.xlabel("R&D Spend(scaled)")
plt.ylabel("Profit")
plt.title("Linear Regression Fit")

plt.tight_layout()
plt.show()

print("Final weight (w):",w)
print("Final bias (b):",b)

```

## Output:
![WhatsApp Image 2026-02-03 at 9 44 22 AM](https://github.com/user-attachments/assets/714838df-99d7-498b-8f13-da2617345560)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
