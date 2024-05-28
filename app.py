import numpy as np
import pickle
import pandas
import os
from flask import Flask, request, render_template

app = Flask(__name__, template_folder='templates')

model = pickle.load(open('./training/Soil-moisture-model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/inner-page')
def output():
    return render_template("inner-page.html")

@app.route('/submit',methods=["POST","GET"])
def submit():
    input_feature = [float(x) for x in request.form.values() if x]

    input_feature=[np.array(input_feature)]
    print(input_feature)
    names = ['Month','Day','avg_temp','avg_humd']
    data = pandas.DataFrame(input_feature,columns=names)
    print(data)

    prediction=model.predict(data)
    print(prediction)
    out=prediction[0]

    return render_template("output.html",result = out)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=1111,debug=True)