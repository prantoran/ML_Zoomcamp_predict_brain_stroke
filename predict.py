import pickle

from flask import Flask, render_template, request, jsonify


model_file = 'brain_stroke_model.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

print("__name__:", __name__)

app = Flask('stroke')


def create_patient(request):
    return {
        'gender': request.form['gender'],
        'age': request.form['age'],
        'hypertension': request.form['hypertension'],
        'heart_disease': request.form['heart_disease'],
        'ever_married': request.form['ever_married'],
        'work_type': request.form['work_type'],
        'residence_type': request.form['residence_type'],
        'avg_glucose_level': request.form['avg_glucose_level'],
        'bmi': request.form['bmi'],
        'smoking_status': request.form['smoking_status'],
    }


@app.route("/")
def my_form_home():
    return render_template('home.html')

 
@app.route('/', methods=['GET', 'POST'])
def predict():
    patient = create_patient(request)

    X = dv.transform([patient])
    y_pred = model.predict_proba(X)[0, 1]
    stroke = y_pred >= 0.5

    return render_template('pred.html', data=float(y_pred), data2=bool(stroke))


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)