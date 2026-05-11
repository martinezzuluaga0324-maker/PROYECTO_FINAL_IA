from flask import Flask, render_template, request
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Cargar modelos
log_model = joblib.load('logistic_model.pkl')
nn_model = joblib.load('nn_model.pkl')
scaler = joblib.load('scaler.pkl')

# Dataset
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"

df = pd.read_csv(url)

# Variables
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Escalado
X_scaled = scaler.transform(X)

# Accuracy
log_acc = accuracy_score(y, log_model.predict(X_scaled))
nn_acc = accuracy_score(y, nn_model.predict(X_scaled))

@app.route('/', methods=['GET', 'POST'])
def index():

    prediction = None

    if request.method == 'POST':

        data = [
            float(request.form['pregnancies']),
            float(request.form['glucose']),
            float(request.form['bloodpressure']),
            float(request.form['skinthickness']),
            float(request.form['insulin']),
            float(request.form['bmi']),
            float(request.form['dpf']),
            float(request.form['age'])
        ]

        scaled = scaler.transform([data])

        result = nn_model.predict(scaled)[0]

        if result == 1:
            prediction = "Alto riesgo de diabetes"
        else:
            prediction = "Bajo riesgo de diabetes"

    return render_template(
        'index.html',
        prediction=prediction,
        log_acc=round(log_acc * 100, 2),
        nn_acc=round(nn_acc * 100, 2)
    )
import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)