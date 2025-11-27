import joblib

model = joblib.load("age_model.pkl")

def predict_age(age):
    prediction = model.predict([[age]])[0]
    return "Adult" if prediction == 1 else "Minor"

if __name__ == "__main__":
    print("Prediction for 20:", predict_age(20))
    print("Prediction for 10:", predict_age(10))
