#import Flask 
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

#create an instance of Flask
app = Flask(__name__)

#home page 
@app.route('/')
def home():
    return render_template('home.html')
    
#make predictions
@app.route('/predict', methods = ['Get', 'POST'])
def predict():
    if request.method == "POST":
        # Summary = request.form.get('Summary')
        Humidity = request.form.get('Humidity')
        WindSpeed = request.form.get('WindSpeed')
        WindBearing = request.form.get('WindBearing')
        Visibility = request.form.get('Visibility')
        Pressure = request.form.get('Pressure')
        prediction = predict(Humidity, WindSpeed, WindBearing, Visibility, Pressure)
        return render_template('predict.html', prediction = prediction)
    
def predict(Humidity, WindSpeed, WindBearing, Visibility, Pressure):
    test_data = [[Humidity, WindSpeed, WindBearing, Visibility, Pressure]]
    test_data = np.array(test_data)
    test_data = pd.DataFrame(test_data)
    print(test_data)
    file = open("model_v3.pkl","rb")
    trained_model = joblib.load(file)
    prediction = trained_model.predict(test_data)
    print(prediction)
    return prediction
    
if __name__ == '__main__':
    app.run(debug=True)




