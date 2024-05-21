## ml_project

### Aim:
  To develop an accurate and reliable weather prediction model that takes into account multiple meteorological variables, historical weather data, and advanced machine learning algorithms to provide precise forecasts for various geographic locations and time horizons.
  
### Algorithm:
1.Import all required libraries for data manipulation, modeling, and evaluation.
2.Read the weather data CSV file and print the first few rows to understand its structure.
3.Split the data into training and test sets using train_test_split.
4.Initialize and train a logistic regression model on the training data.
5.Make predictions on the test set and evaluate the model using classification metrics.
6.Create a new test data point, convert it to a DataFrame, and use the trained model to make a prediction.

### Program:

``` py
import numpy as np
import pandas as pd
weather_df = pd.read_csv("/content/seattle-weather.csv")
print(weather_df.head())
X = weather_df.drop(columns=['date', 'weather'])
y = weather_df['weather']
print(X.isnull().sum())
print(y.isnull().sum())
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101, test_size=0.3)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000) 
model.fit(X_train, y_train)
predictions = model.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))
test_data = {
    'precipitation': 10.9,
    'temp_max': 10.6,
    'temp_min': 2.8,
    'wind': 4.5
}
test_df = pd.DataFrame([test_data])
prediction = model.predict(test_df)
print("Prediction for the test data:", prediction)

```
### Output:

![image](https://github.com/ShakthiSundar-K/ml_project/assets/128116143/a7fa072e-83ac-406f-900b-6e70025af0dd)
![image](https://github.com/ShakthiSundar-K/ml_project/assets/128116143/bc5410cb-6ddd-4cc2-8b63-07da46d652b8)
![image](https://github.com/ShakthiSundar-K/ml_project/assets/128116143/93ab0302-4605-42f9-9af4-2e45190114e0)

### Result:
  Thus the program is successfully executed.


