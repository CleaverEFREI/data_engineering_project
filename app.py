import pickle
from flask import Flask, render_template, request
import sklearn.linear_model
import sklearn.feature_extraction.text
import sys

app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

@app.route('/')
def index():
    """
    Render the index base template
    """
    return render_template('index.html')


@app.route('/health_check')
def health_check():
    """
    Used to verify that the app is up and running
    """
    return "ok"


@app.route('/predict', methods = ["POST"])
def predict():
    """
    Get data from index form, use the pretrainned model pickle and return the setiment in prediction_text
    """
    r = request.form.get('inputdata', "This is a default value")
    if r[0] == '!':
        r = r[1:]
        debug = 1
    else:
        debug = 0
    print("Info request :",r, file=sys.stderr)
    prediction = model.predict_proba(vectorizer.transform([r]))
    if prediction[0][1] < 0.40:
        pred = "negative"
    elif prediction[0][1] < 0.60:
        pred = "neutral"
    else:
        pred = "positive"
    if debug :
        return render_template("index.html",prediction_text = 'The sentence "{}" is {}. neg {} pos {}'.format(r,pred,prediction[0][0],prediction[0][1]))
    else :
        return render_template("index.html",prediction_text = 'The sentence "{}" is {}.'.format(r,pred))

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0')