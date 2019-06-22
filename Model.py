#!/usr/bin/env python 3.6.8
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle
import sys
flag=False
def Create_model():
	Used_car=pd.read_csv('true_car_listings_dataset_1.csv')
	Used_car.drop('Vin',inplace=True,axis=1)
	X=Used_car.drop('Price',axis=1)
	y=Used_car['Price']
	numerical_data=X.drop(['City','State','Make','Model'],axis=1)
	non_numerical_data=X.drop(['Year','Mileage'],axis=1)
	num_pipeline=Pipeline(
    	[
        	('imputer',SimpleImputer(strategy='median')),
        	('scaler',StandardScaler())
    	]
	)
	full_pipeline=ColumnTransformer(
    	[
        	('num_pipeline',num_pipeline,numerical_data.columns),
        	('ordinal',OrdinalEncoder(),non_numerical_data.columns)
    	]
	) 
	X_train=full_pipeline.fit_transform(X)
	lin_reg=LinearRegression()
	lin_reg.fit(X_train,y)
	pickle.dump(lin_reg,open('Linear_Model.sav','wb'))
	flag=True

def Use_Model(Year,milage,city,state,make,model):
	test_data={
    	'Year':Year,
    	'Mileage':milage,
    	'City':city,
    	'State':state,
    	'Make':make,
    	'Model':model
	}
	test_data_df=pd.DataFrame(test_data,index=[0])
	numerical_data=test_data_df.drop(['City','State','Make','Model'],axis=1)
	non_numerical_data=test_data_df.drop(['Year','Mileage'],axis=1)
	num_pipeline=Pipeline(
    	[
        	('imputer',SimpleImputer(strategy='median')),
        	('scaler',StandardScaler())
    	]
	)	
	full_pipeline=ColumnTransformer(
    	[
        	('num_pipeline',num_pipeline,numerical_data.columns),
        	('ordinal',OrdinalEncoder(),non_numerical_data.columns)
    	]
	)
	X_test=full_pipeline.fit_transform(test_data_df)
	if(flag==False):
		Create_model()
	loaded_model = pickle.load(open('Linear_Model.sav', 'rb'))
	result = loaded_model.predict(X_test)
	return (result[0])

print("Welcome To used Car Prediction System")
print("Enter the input in the form")
Prediction=Use_Model(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6])
print ('The value of your car is:  ',round(Prediction-8000.00,2),'to ',round(Prediction+4000.00,2))
