from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from urllib.request import urlopen
import json

app = Flask(__name__)

@app.route('/')
def show_predict_stock_form():
    return render_template('index.html')

@app.route('/results', methods=['POST'])
def results():
    company = request.form['company']

    if(company == 'IBM'):
        value = predictIBM(company)
        return render_template('result.html', value=value[0][0])


if __name__ == "__main__":
    app.run(debug = True)

def getDataset(company):
    response = urlopen("https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol="+company+"&outputsize=full&apikey=E6JX3MJVM8YEK9NF")
    json_data = response.read().decode('utf-8', 'replace')
    data = json.loads(json_data)
    dataset = []
    for i in data['Time Series (Daily)']:
        dataset.append(data['Time Series (Daily)'][i]['1. open'])
    return [float(numeric_string) for numeric_string in dataset]

def predictIBM(company):
    model = load_model('models/IBMmodel.h5')
    sc = MinMaxScaler(feature_range = (0, 1))

    dataset = pd.DataFrame(getDataset(company))
    inputs = dataset.values
    inputs = inputs.reshape(-1,1)
    inputs = sc.fit_transform(inputs) 
    
    X_test = []
    X_test.append(inputs[1:60, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)

    return predicted_stock_price
    
