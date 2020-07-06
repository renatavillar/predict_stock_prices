import streamlit as st
import pandas as pd
import json
from pymongo import MongoClient
import numpy as np
import os
import time
import warnings
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import keras.backend.tensorflow_backend as tb

tb._SYMBOLIC_SCOPE.value = True

def main():
    st.title('Predicting QQQ Stock Price')
    st.subheader('LSTM Neural Network')
    #button = st.button('Run')
    nodes = st.slider('Choose the number of nodes in the hidden layer', min_value=50, max_value=256)
    epochs = st.slider('Choose the number of epochs', min_value=10, max_value=500)
    batch = st.radio('Select the batch size', (16, 32, 64))

    def network():
    	
        #data = pd.read_csv('historical_stock_prices.csv')
        data_QQQ = pd.read_csv('QQQ.csv')

        #data_QQQ = data[data['ticker'] == 'QQQ']
        #data_QQQ.to_csv('QQQ.csv', index=False)

        #Extract
        class Extract:
    
            def __init__(self):
                #loading json file
                self.data_sources = json.load(open('configs.json'))
                self.csv_path = self.data_sources['data_sources']['csv']
        
            def getCSVData(self, csv_name):
                #we can use multiple csv files; so, we will pass csv name as an argument to fetch the
                #desired csv data
                df = pd.read_csv(self.csv_path[csv_name])
                return df

    #Transformation
    #class Transformation:
    
        #def __init__(self, dataSource, dataSet):
            # creating Extract class to fetch data using its generic methods for csv data sources
            #extractObj = Extract()
            #self.data = extractObj.getCSVData(dataSet)
            #funcName = dataSource + dataSet
            #getattr(self, funcName)()
        
        #Stocks Market Data Transformation
        #def csvStocksPrices(self):
            #ETFsCode = [list(data.unique())]
        
            # coverting open, close, high and low price into GBP values since current price is in Dollars
            # if currency belong to the list ETFsCode
            #self.csv_df['open'] = self.csv_df[['open', 'ticker']].apply(lambda x: (float(x[0]) * 0.75) if x[1] in ETFsCode else np.nan, axis=1)
            #self.csv_df['close'] = self.csv_df[['close', 'ticker']].apply(lambda x: (float(x[0]) * 0.75) if x[1] in ETFsCode else np.nan, axis=1)
            #self.csv_df['high'] = self.csv_df[['high', 'ticker']].apply(lambda x: (float(x[0]) * 0.75) if x[1] in ETFsCode else np.nan, axis=1)
            #self.csv_df['low'] = self.csv_df[['low', 'ticker']].apply(lambda x: (float(x[0]) * 0.75) if x[1] in ETFsCode else np.nan, axis=1)
        
            # dropping rows with null values by asset column
            #self.csv_df.dropna(inplace=True)
        
            # saving new csv file
            #self.csv_df.to_csv('stocks-prices-GBP.csv')

        #Transformation
        class Transformation:
    
            def __init__(self, dataSource, dataSet):
                # creating Extract class to fetch data using its generic methods for csv data sources
                extractObj = Extract()
                self.data = extractObj.getCSVData(dataSet)
                funcName = dataSource + dataSet
                getattr(self, funcName)()
        
            #QQQ Market Data Transformation
            def csvQQQPrices(self):
                ETFsCode = ['QQQ']
        
                # coverting open, close, high and low price into GBP values since current price is in Dollars
                # if currency belong to the list ETFsCode
                self.data['open'] = self.data[['open', 'ticker']].apply(lambda x: (float(x[0]) * 0.75) if x[1] in ETFsCode else np.nan, axis=1)
                self.data['close'] = self.data[['close', 'ticker']].apply(lambda x: (float(x[0]) * 0.75) if x[1] in ETFsCode else np.nan, axis=1)
                self.data['high'] = self.data[['high', 'ticker']].apply(lambda x: (float(x[0]) * 0.75) if x[1] in ETFsCode else np.nan, axis=1)
                self.data['low'] = self.data[['low', 'ticker']].apply(lambda x: (float(x[0]) * 0.75) if x[1] in ETFsCode else np.nan, axis=1)
        
                # dropping rows with null values by asset column
                self.data.dropna(inplace=True)
        
                # saving new csv file
                self.data.to_csv('QQQ-prices-GBP.csv', index=False)

        #Loading
        class MongoDB:
    
            #Initilize the common usable variables in below function:
            def __init__(self, user, password, host, db_name ,port='27017', authSource='admin'):
                self.user = user
                self.password = password
                self.host = host
                self.port = port
                self.db_name = db_name
                self.authSource = authSource
                self.uri = 'mongodb://' + self.user + ':' + self.password + '@'+ self.host + ':' + self.port + '/' + self.db_name + '?authSource=' + self.authSource
                try:
                    self.client = MongoClient(self.uri)
                    self.db = self.client[self.db_name]
                    print('MongoDB Connection Successful. CHEERS!!!')
                except Exception as e:
                    print('Connection Unsuccessful!! ERROR!!')
                    print(e)
            
            #Function to insert data in DB, could handle Python dictionary and Pandas dataframes
            def insert_into_db(self, data, collection):
                if isinstance(data, pd.DataFrame):
                    try:
                        self.db[collection].insert_many(data.to_dict('records'))
                        print('Data Inserted Successfully')
                    except Exception as e:
                        print('OOPS!! Some ERROR Occurred')
                        print(e)
                else:
                    try:
                        self.db[collection].insert_many(data)
                        print('Data Inserted Successfully')
                    except Exception as e:
                        print('OOPS!! Some ERROR Occurred')
                        print(e)


        QQQ = pd.read_csv('QQQ-prices-GBP.csv')


        #LSTM
        configs = json.loads(open(os.path.join(os.path.dirname('your_path'), 'configs.json')).read())
        warnings.filterwarnings("ignore")

        model = Sequential()

        model.add(LSTM(
        units = len('QQQ.csv'), return_sequences=True))
        model.add(Dropout(0.2))

        #model.add(LSTM(units = 50, return_sequences=False))
        model.add(LSTM(units = nodes, return_sequences=False))
        model.add(Dropout(0.2))

        model.add(Dense(units=1))
        model.add(Activation("tanh"))

        start = time.time()
        model.compile(
        loss=configs['model']['loss_function'],
        optimizer=configs['model']['optimiser_function'])
    
        #print("> Compilation Time : ", time.time() - start)


        training_processed = QQQ.iloc[:, 1:2].values

        scaler = MinMaxScaler(feature_range = (0, 1))
        training_scaled = scaler.fit_transform(training_processed)

        features_set = []
        labels = []
        for i in range(60, len(QQQ)):
            features_set.append(training_scaled[i-60:i, 0])
            labels.append(training_scaled[i, 0])
    
        features_set, labels = np.array(features_set), np.array(labels)

        features_set, labels = np.array(features_set), np.array(labels)

        features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))

        X_train = features_set[0:int(0.8*len(features_set))]
        y_train = labels[0:int(0.8*len(labels))]


        model.fit(X_train, y_train, epochs = epochs, batch_size = batch)


        X_test = features_set[int(0.8*len(features_set))+1:len(features_set)]
        y_test = labels[int(0.8*len(labels))+1:len(labels)]


        predictions = model.predict(X_test)


        plt.style.use('ggplot')
        plt.figure(figsize=(10,6))
        plt.plot(y_test, color='turquoise', label='Actual QQQ Stock Price')
        plt.plot(predictions , color='red', label='Predicted QQQ Stock Price')
        plt.grid(None)
        plt.title('QQQ Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('QQQ Stock Price')
        plt.legend()
        plt.show()

        st.pyplot()

    if st.button('Run'):
        network()

if __name__ == '__main__':
    main()
