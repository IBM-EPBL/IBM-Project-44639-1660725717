import os
import numpy as np 
from flask import Flask,request,render_template 
import pickle


app= Flask(__name__)
model = pickle.load(open('/Users/balasaravananvp/Documents/tensorflow-test/IBM-Project-26380-1660025744/Project Development Phase/Sprint 4/model.pkl', 'rb')) # loading the trained model

@app.route("/") 
def about():
    return render_template("about.html")

@app.route("/about")
def home():
    return render_template("about.html")

@app.route("/info") 
def information():
    return render_template("info.html")


@app.route("/predict",methods=["GET","POST"]) 
def upload():
    print(request)
    if request.method=='POST':
        init_features = [float(x) for x in request.form.values()]
        final_features = [np.array(init_features)]
        print(final_features)
        pred = model.predict(final_features)
        print(pred)
        return render_template("result.html",prediction=pred)

    else:
        return render_template("predict.html")


if __name__=="__main__":
    app.run(debug=False,port=5500)
            
            