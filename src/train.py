from sklearn.linear_model import LogisticRegression
import joblib
import numpy as np

X = np.array([[10], [15], [20], [25], [30], [5]])
y = np.array([0, 0, 1, 1, 1, 0])  # 0 = Minor, 1 = Adult

model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, "age_model.pkl")
print("Model saved as age_model.pkl")
