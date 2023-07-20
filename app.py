import numpy as np
from flask import Flask,render_template,make_response,request
import pandas as pd
import pickle,json
import warnings
warnings.filterwarnings("ignore")


app=Flask(__name__)

import sys
sys.path.insert(0,"RAINFALL\project")
with open ("project/ann.pkl", "rb") as file:
    model=pickle.load(file)

with open ("project/standardscalar.pkl", "rb") as file:
    std=pickle.load(file)
    
with open ("project/data.json","r") as file:
    columns=json.load(file)["columns"]


@app.route("/",methods=["GET", "POST"])
def home():
    return render_template("start.html")

@app.route("/predictions",methods=["GET", "POST"])
def predictions():
    try:
        data=request.form
        test_array=np.zeros(len(columns))
        test_array[0]=data.get("Location")
        test_array[1]=data.get("MinTemp")
        test_array[2]=data.get("MaxTemp")
        test_array[3]=data.get("Rainfall")
        test_array[4]=data.get("Evaporation")
        test_array[5]=data.get("Sunshine")
        test_array[6]=data.get("WindGustDir")
        test_array[7]=data.get("WindGustSpeed")
        test_array[8]=data.get("WindDir9am")
        test_array[9]=data.get("WindDir3pm")
        test_array[10]=data.get("WindSpeed9am")
        test_array[11]=data.get("WindSpeed3pm")
        test_array[12]=data.get("Humidity9am")
        test_array[13]=data.get("Humidity3pm")
        test_array[14]=data.get("Pressure9am")
        test_array[15]=data.get("Pressure3pm")
        test_array[16]=data.get("Cloud9am")
        test_array[17]=data.get("Cloud3pm")
        test_array[18]=data.get("Temp9am")
        test_array[19]=data.get("Temp3pm")
        test_array[20]=data.get("RainToday")
        test_array[21]=data.get("enc_Location")
        std_array=std.fit_transform([test_array])
        result=model.predict(std_array)
        
        if result[0][0]>0.5:
            prediction="You are likely to enjoy rain tomorrow"
        else:
            prediction="You are likely enjoy vitamin D tomorrow"
            
        return render_template("pred.html",prediction=prediction)
        
    except Exception as e:
        return make_response({"error":"exception",
                              "message":str(e)},409)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5555,debug=True)
    
    
    
    """
    {
    "Location":"",
    "MinTemp":"",
    "MaxTemp":"",
    "Rainfall":"",
    "Evaporation":"",
    "Sunshine":"",
    "WindGustDir":"",
    "WindGustSpeed":"",
    "WindDir9am":"",
    "WindDir3pm":"",
    "WindSpeed9am":"",
    "WindSpeed3pm":"",
    "Humidity9am":"",
    "Humidity3pm":"",
    "Pressure9am":"",
    "Pressure3pm":"",
    "Cloud9am":"",
    "Cloud3pm":"",
    "Temp9am":"",
    "Temp3pm":"",
    "RainToday":"",
    "enc_Location":""
}
    
    """