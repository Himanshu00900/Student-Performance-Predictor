import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/students.csv")
plt.figure(figsize=(8,5))
plt.scatter(df["StudyHours"], df["FinalMarks"])

plt.title("Study Hours vs Final Marks")
plt.xlabel("Study Hours")
plt.ylabel("Final Marks")

plt.show()

plt.figure(figsize=(8,5))
plt.scatter(df["Attendance"], df["FinalMarks"])

plt.title("Attendance vs Final Marks")
plt.xlabel("Attendance (%)")
plt.ylabel("Final Marks")

plt.show()
plt.figure(figsize=(8,5))
plt.scatter(df["PreviousMarks"], df["FinalMarks"])

plt.title("Previous Marks vs Final Marks")
plt.xlabel("Previous Marks")
plt.ylabel("Final Marks")

plt.show()
print(df.corr())