# predict_stock_prices
A web app built with streamlit to predict QQQ stock price. A trained LSTM Neural Network to build a model to do the predictions.

Write your path in 'your_path' in the following line of app.py:

configs = json.loads(open(os.path.join(os.path.dirname('your_path'), 'configs.json')).read())

Then, from your directory, run the app through the command streamlit run app.py

Bwlow is an image of the app:

![alt text](https://github.com/renatavillar/predict_stock_prices/blob/master/app.png)
