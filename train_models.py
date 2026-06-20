import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
df = pd.read_csv("data/students.csv")

X = df[['StudyHours', 'Attendance', 'PreviousMarks']]
y = df['FinalMarks']
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
lr_model = LinearRegression()

lr_model.fit(X_train, y_train)
lr_score = lr_model.score(X_test, y_test)

print("Linear Regression R² Score:", lr_score)
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)
rf_score = rf_model.score(X_test, y_test)

print("Random Forest R² Score:", rf_score)
print("\nModel Comparison")
print("-----------------")
print("Linear Regression :", lr_score)
print("Random Forest     :", rf_score)
if rf_score > lr_score:
    best_model = rf_model
    print("Best Model: Random Forest")
else:
    best_model = lr_model
    print("Best Model: Linear Regression")

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
lr_predictions = lr_model.predict(X_test)
rf_predictions = rf_model.predict(X_test)
lr_mae = mean_absolute_error(y_test, lr_predictions)

lr_mse = mean_squared_error(y_test, lr_predictions)

lr_rmse = lr_mse ** 0.5

lr_r2 = r2_score(y_test, lr_predictions)

rf_mae = mean_absolute_error(y_test, rf_predictions)

rf_mse = mean_squared_error(y_test, rf_predictions)

rf_rmse = rf_mse ** 0.5

rf_r2 = r2_score(y_test, rf_predictions)
print("\nLINEAR REGRESSION")
print("------------------")
print("MAE :", lr_mae)
print("MSE :", lr_mse)
print("RMSE:", lr_rmse)
print("R²  :", lr_r2)

print("\nRANDOM FOREST")
print("------------------")
print("MAE :", rf_mae)
print("MSE :", rf_mse)
print("RMSE:", rf_rmse)
print("R²  :", rf_r2)
plt.figure(figsize=(8,5))

plt.scatter(y_test, rf_predictions)

plt.xlabel("Actual Marks")
plt.ylabel("Predicted Marks")

plt.title("Actual vs Predicted Marks")

plt.show()
best_model = rf_model