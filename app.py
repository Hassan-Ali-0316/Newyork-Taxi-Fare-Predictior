from flask import Flask , render_template , request
import pandas as pd
import pickle
app = Flask(__name__)
import numpy as np


model = pickle.load(open('LinearRegressionModel.pkl','rb'))
data = pd.read_csv('C:/Users/ha159/OneDrive/Documents/TaxiFare prediction/taxi_fare/train.csv')


@app.route('/')
def index():
    duration = data['trip_duration']
    distance = data['distance_traveled']
    num_passengers = sorted(data['num_of_passengers'].unique())
    fare = data['fare']
    tip = data['tip']
    misfee = data['miscellaneous_fees']
    return render_template('index.html',duration=duration,distance=distance,num_passengers=num_passengers,fare=fare,tip=tip,misfee=misfee)

@app.route('/predict',methods=['POST'])
def predict():
    td = int(request.form.get('trip_duration'))
    dt = int(request.form.get('distance_traveled'))
    pas = int(request.form.get('num_of_passengers'))
    fare = int(request.form.get('fare'))
    tip = int(request.form.get('tip'))  
    mis = int(request.form.get('miscellaneous_fees'))
    print('egegegegegegegegegegeegegege')
    print(td,dt,pas,fare,tip,mis)
    # prediction = model.predict(pd.DataFrame)
    # return str(prediction[0])
    features = pd.DataFrame([[td, dt, pas, fare, tip, mis]],columns=['trip_duration', 'distance_traveled', 'num_of_passengers', 'fare', 'tip', 'miscellaneous_fees'])

    prediction = model.predict(features)[0]
    prediction = np.round(prediction,2)
    return str(prediction)

if __name__ == "__main__":
    app.run(debug=True)