from flask import Flask, render_template, request
from src.predict import predict_heart_attack

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ⚠️ Maintain correct feature order (VERY IMPORTANT)
        features = [
            float(request.form["age"]),
            float(request.form["sex"]),
            float(request.form["cp"]),
            float(request.form["trestbps"]),
            float(request.form["chol"]),
            float(request.form["fbs"]),
            float(request.form["restecg"]),
            float(request.form["thalach"]),
            float(request.form["exang"]),
            float(request.form["oldpeak"]),
            float(request.form["slope"]),
            float(request.form["ca"]),
            float(request.form["thal"])
        ]

        result, prob = predict_heart_attack(features)

        return render_template("index.html", prediction_text=result, probability=prob)

    except ValueError:
        return render_template("index.html", prediction_text="Invalid input! Please enter valid numbers.")

    except Exception as e:
        return render_template("index.html", prediction_text="Something went wrong!")

if __name__ == "__main__":
    app.run(debug=True)