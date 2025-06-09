from flask import render_template, request
from app import app
import joblib
import numpy as np

model = joblib.load('app/model/model.pkl')
scaler = joblib.load('app/model/scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_data = [
            float(request.form['study_hours']),
            float(request.form['attendance']),
            float(request.form['gpa'])
        ]
        input_scaled = scaler.transform([input_data])
        prediction = model.predict(input_scaled)[0]
        return render_template('index.html', prediction=prediction)

    return render_template('index.html', prediction=None)
