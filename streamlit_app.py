#from flask import Flask, render_template
#from flask import Flask, flash, redirect, render_template, request, session, abort
#from flask import json

# Model Imports
#import pandas as pd
#import datetime as dt
#import pickle

#File delete imports
#import os
#from os import listdir
#from os.path import isfile, join

# Data soruce imports
#from pandas_datareader import data as pdr
#from datetime import date
#import yfinance as yf
#yf.pdr_override() 

#import requests
#import json
#from os import listdir
#from os.path import isfile, join

#Model Imports
#from sklearn.model_selection import train_test_split
#from sklearn.tree import DecisionTreeRegressor
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.linear_model import LinearRegression



def Get_Stock_Tickers():
    tickers_list = ['AAPL', 'AAP', 'ABBV', 'ABC', 'ABT', 'ACN']
    return tickers_list

app = Flask(__name__)

@app.route('/stocks' , methods=['GET','POST'])
def stocks():
    # Remove csv and date from filenames
    file_names_list_2 = []
    for filename in file_names_list:
	    filename = filename.split("_")[0]
	    file_names_list_2.append(filename)

    # Add heading to list
    file_names_list_2.insert(0,"Stock")
    y_pred_list.insert(0,"Prediction")
    Prev_Open_Price.insert(0,"Yesterday_Price")

    Modeldata = zip(file_names_list_2, y_pred_list ,Prev_Open_Price)

    #file_names_list.pop(0)
    #y_pred_list.pop(0)
    #Prev_Open_Price.pop(0)
    
    return render_template('stocks.html', data = Modeldata)


@app.route('/home' , methods=['GET','POST'])
def home():

	data = ""
	if request.method == 'GET':
		return render_template('index.html', msg =data)


	if request.method =='POST':
		print ("POST request")
		task = request.form['req']
		print (task)

		data= ""

		if task == "remove":
			data = DeleteFile()

		if task == "load":
			data = SourceData()
		if task == "run":
			data = RunModel()

		return render_template('index.html', msg = data)
	
			
	return render_template('index.html',msg = data)

@app.route('/search' , methods=['GET','POST'])
def search():
    if request.method =='POST':
        print ("POST request")
        task = request.form['req']
        stock_name = request.form['stockval']
        print (task)
        print (stock_name)

        if task == "dsearch":

            # Daily Prediction Data
            today = date.today()
            start_date= "2017-01-01"
            df = pdr.get_data_yahoo(stock_name, start=start_date, end=today)
            df = df.fillna(0)

            df.to_csv("TMP_df.csv")
            df = pd.read_csv("TMP_df.csv")
            df['Date'] = pd.to_datetime(df['Date'])
            df['year'] = df['Date'].dt.year
            df['month'] = df['Date'].dt.month
            df['day'] = df['Date'].dt.day
            df = df.drop(['Date'], axis= 1)
            X = df.drop(['Open'], axis=1)
            y = df['Open']

            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=30)
            df1 = df.tail(100)
            X_test = df1.drop(['Open'], axis=1)

            y_test = df1['Open']
            reg = RandomForestRegressor(n_estimators=100)
            reg.fit(X_train, y_train)
            y_pred = reg.predict(X_test)
            print (len (y_pred))

            date_list = pd.to_datetime((df1.year*10000+df1.month*100+df1.day).apply(str),format='%Y%m%d')
            
            date_list2 = []
            for dt in date_list:
                date_list2.append(dt)
            print (len(date_list2))

            date_list2.insert(0,"Date")
            y_pred = list(y_pred)
            y_pred.insert(0,"Daily_Prediction")

            daily_pred_data = zip(date_list2, y_pred)
            return render_template('daily_predictions.html', data = daily_pred_data)


        if task == "wsearch":
            
            # Daily Prediction Data
            today = date.today()
            start_date= "2017-01-01"
            df = pdr.get_data_yahoo(stock_name, start=start_date, end=today)
            df = df.fillna(0)

            df.to_csv("TMP_df.csv")

            df = pd.read_csv("TMP_df.csv")
            df['Date'] = pd.to_datetime(df['Date'])
            #df = df.resample('W', on='Date').mean() 
            #df['Date'] = pd.to_datetime(df['Date'])
            df['year'] = df['Date'].dt.year
            df['month'] = df['Date'].dt.month
            df['day'] = df['Date'].dt.day
            df = df.drop(['Date'], axis= 1)
            X = df.drop(['Open'], axis=1)
            y = df['Open']

            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=30)
            df1 = df.tail(100)
            X_test = df1.drop(['Open'], axis=1)

            y_test = df1['Open']
            reg = RandomForestRegressor(n_estimators=100)
            reg.fit(X_train, y_train)
            y_pred = reg.predict(X_test)
            y_pred = list(y_pred)
            print (len (y_pred))

            date_list = pd.to_datetime((df1.year*10000+df1.month*100+df1.day).apply(str),format='%Y%m%d')
            
            date_list2 = []
            for dt in date_list:
                date_list2.append(dt)
            print (len(date_list2))

            df3 = pd.DataFrame(date_list2 , columns=['Date'])
            df3['Predictions'] = y_pred
            df3 = df3.resample('w',on='Date').mean()

            date_l = df3.index
            y_pred2 = df3['Predictions']

            y_pred3 = []
            for p in y_pred2:
                p = round(p,3)
                y_pred3.append(round(p,3))

            date_l = list(date_l)
            date_l.insert(0,"Date")
    
            y_pred3.insert(0,"Weekly_Prediction")

            weekly_pred_data = zip(date_l, y_pred3)

            return render_template('weekly_predictions.html', data = weekly_pred_data)


def DeleteFile():
	mypath= './data/'

	datafiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

	print (datafiles)
	for file in datafiles:
		os.remove('./data/'+ str (file))

	return "Files_are_deleted."


def SourceData():
	# Tickers List
	tickers_list = Get_Stock_Tickers()
	today = date.today()

	# We can get data by our choice by giving days bracket
	start_date= "2017-01-01"
	end_date="2019-11-30"

	# Function to get Data, Using Yahoo finance API
	files=[]
	def getData(ticker):
	    print (ticker)
	    data = pdr.get_data_yahoo(ticker, start=start_date, end=today)
	    dataname= ticker+'_'+str(today)
	    files.append(dataname)
	    SaveData(data, dataname)
	    
	def SaveData(df, filename):
	    df.to_csv('./data/'+filename+'.csv')

	# Call function using list
	for tik in tickers_list:
		getData(tik)

	return "Data_Collected_For_Stocks"


def Show_all_Model_Data():
    print ("Random Forest Regressor.")
    print (len(file_names_list))
    print (len(y_pred_list))
    print (len(Prev_Open_Price))


def RunModel():
    mypath= './data/'

    # read data file from data folder

    mypath= './data/'

    datafiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    print (datafiles)

    for file in datafiles:
        #print (file)
        Loadcsv(file)
        #print ("\n")
    #print (y_pred_list)
    
    Show_all_Model_Data()
    return "Model_Training_Completed."


y_pred_list = []

def Save_Pred(pred):
    y_pred_list.append(pred)
    print (y_pred_list)

file_names_list = []
def Save_File_name(file_name):
    file_names_list.append(file_name)

# to save Previou open price
Prev_Open_Price = []
def SavePrevPrice(dataframe):
    val = dataframe.values[-1].tolist()
    pric = val[1]
    #print ("Previous Open Price")
    #print (pric)
    Prev_Open_Price.append(pric)

def Loadcsv(file):
    dataframe = pd.read_csv('./data/'+file)
    
    # Calling model training functions
    
    #DT_ModelPrediction(dataframe, file)
    RF_ModelPrediction(dataframe, file)

    Save_File_name(file)

    SavePrevPrice(dataframe)




def DT_ModelPrediction(df, file):
    
    # Null values
    df = df.fillna(0)
    
    # Extract days, month, year from Date column
    df['Date'] = pd.to_datetime(df['Date'])

    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day

    df = df.drop(['Date'], axis= 1)
    #print (df.columns)
    
    # Split into our target variabe and independent features
    X = df.drop(['Open'], axis=1)
    y = df['Open']
    
    # Split into training and testing 
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=30)
    
    df1 = df.tail(7)
    X_test = df1.drop(['Open'], axis=1)
    y_test = df1['Open']
    
    # Decision Tree Regressor Model instance
    tree = DecisionTreeRegressor(criterion='mse', max_depth=30)
    
    # Model training part
    tree.fit(X_train, y_train)
    
    # Model testing to make predictions
    y_pred = tree.predict(X_test)
    #y_pred = str(y_pred)[1:-1]
    y_pred = sum(y_pred) / len (y_pred)
    
    # Extract Ticker name by the file name
    file = file.split("_")
    file = str(file[0])
    
    print ("Decision Tree Model")
    print (file)
    print (y_pred)
    Save_Pred(y_pred)



def RF_ModelPrediction(df, file):
    
    # Null values
    df = df.fillna(0)
    
    # Extract days, month, year from Date column
    df['Date'] = pd.to_datetime(df['Date'])

    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day

    df = df.drop(['Date'], axis= 1)
    #print (df.columns)
    
    # Split into our target variabe and independent features
    X = df.drop(['Open'], axis=1)
    y = df['Open']
    
    # Split into training and testing 
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=30)
    
    df1 = df.tail(7)
    X_test = df1.drop(['Open'], axis=1)
    y_test = df1['Open']
    
    # Decision Tree Regressor Model instance
    reg = RandomForestRegressor(n_estimators=100)
    
    # Model training part
    reg.fit(X_train, y_train)
    
    # Model testing to make predictions
    y_pred = reg.predict(X_test)
    #y_pred = str(y_pred)[1:-1]
    y_pred = sum(y_pred) / len (y_pred)
    
    # Extract Ticker name by the file name
    file = file.split("_")
    file = str(file[0])
    
    #print ("Random Forest Model")
    #print (file)
    #print (y_pred)
    Save_Pred(y_pred)















if __name__ == "__main__":
    
    app.run(debug=True)

