import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

data = pd.read_csv("data/students.csv")

X = data[['StudyHours', 'Attendance', 'PreviousMarks']]
y = data['FinalMarks']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("Model Accuracy:", round(accuracy * 100, 2), "%")

with open("model/student_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model Saved Successfully")