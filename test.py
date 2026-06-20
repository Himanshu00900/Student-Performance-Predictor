import pickle

with open("model/student_model.pkl", "rb") as file:
    model = pickle.load(file)

prediction = model.predict([[6, 88, 78]])

print("Predicted Marks:", prediction[0])