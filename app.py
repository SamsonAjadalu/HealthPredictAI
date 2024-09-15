from flask import Flask, render_template, request, jsonify
import model  # Assuming model.py has your heart and diabetes prediction logic

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    prediction_type = request.form.get('prediction_type')

    if prediction_type == 'heart':
        try:
            heart_data = [
                float(request.form['age']),
                float(request.form['sex']),
                float(request.form['cp']),
                float(request.form['chol']),
                float(request.form['fbs']),
                float(request.form['exang']),
                float(request.form['oldpeak']),
                float(request.form['slope']),
                float(request.form['ca']),
                float(request.form['thal']),
            ]
        except (KeyError, ValueError) as e:
            return f"Error: {str(e)}", 400  # Handle missing or invalid form data

        heart_result = model.predict_heart_disease(heart_data)
        return jsonify(result='Heart Disease', prediction='Positive' if heart_result[0] == 1 else 'Negative')

    elif prediction_type == 'diabetes':
        try:
            diabetes_data = [
                float(request.form['Pregnancies']),
                float(request.form['Glucose']),
                float(request.form['Blo< TrainMuscleFit> odPressure']),
                float(request.form['BMI']),
                float(request.form['DiabetesPedigreeFunction']),
                float(request.form['Age']),
            ]
        except (KeyError, ValueError) as e:
            return f"Error: {str(e)}", 400  # Handle missing or invalid form data

        diabetes_result = model.predict_diabetes(diabetes_data)
        return jsonify(result='Diabetes', prediction='Positive' if diabetes_result[0] == 1 else 'Negative')


if __name__ == '__main__':
    app.run(debug=True)
