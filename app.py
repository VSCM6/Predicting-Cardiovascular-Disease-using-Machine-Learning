import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template


# Create flask app
flask_app = Flask(__name__,template_folder="template")
model=pickle.load(open("Cardiovascular_disease.p","rb"))
@flask_app.route("/")
def Home():
    return render_template("index.html")

gender={'M':2,'F':1}
general={'N':0 ,'Y':1}
levels={"Normal":1,"Above_Normal":2,"Well_Above_Normal":3}
Cardiovascular_Disease={0:"Low risk of Cardiovascular Disease",1:"High risk of Cardiovascular Disease"}
@flask_app.route("/predict", methods = ["POST"])
def predict():
    float_features =[]
    float_features.append(int(request.form['Age']))
    float_features.append(gender[request.form['gender']])
    float_features.append(float(request.form['HEIGHT']))
    float_features.append(float(request.form['WEIGHT']))
    float_features.append(float(request.form['AP_HIGH']))
    float_features.append(float(request.form['AP_LOW']))
    float_features.append(levels[request.form['CHOLESTEROL']])
    float_features.append(levels[request.form['GLUCOSE']])
    float_features.append(general[request.form['SMOKE']])
    float_features.append(general[request.form['ALCOHOL']])
    float_features.append(general[request.form['PHYSICAL_ACTIVITY']])
    float_features=np.array(float_features)
    float_features=float_features.reshape(1,-1)
    predict=model.predict(float_features)
    result=Cardiovascular_Disease[predict[0]]
    return render_template("index.html", prediction_text = "You have {}".format(result))

if __name__ == "__main__":
    flask_app.run(debug=True)
