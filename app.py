from flask import Flask, render_template, request
import joblib

model = joblib.load('linear_regression_model.pkl')


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':

        tv = float(request.form['tv'])
        radio = float(request.form['radio'])
        newspaper = float(request.form['newspaper'])

        features = [[tv, radio, newspaper]]
        prediction = model.predict(features)[0]

        return render_template('result.html', prediction=round(prediction, 2))

if __name__ == '__main__':
    app.run(debug=True)
