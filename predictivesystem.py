# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pickle 
from sklearn.preprocessing import StandardScaler

loaded_model = pickle.load(open('C:\\Users\\HP\\Downloads\\Diabetesprediction\\trained_model.sav', 'rb'))

input_data = (1,97,66,15,140,23.2,0.487,22)
# changing the data input_data to numpy array
input_data_as_array_form = np.asarray(input_data)  
# Reshaping the data using numpy
input_data_reshape = input_data_as_array_form.reshape(1,-1)

#predicting the output
prediction = loaded_model.predict(input_data_reshape)
print(prediction) 

if(prediction[0]==0):
  print('The person is Not Diabetic')
else :  # Added a colon after else
  print('This is  Diabetic Person') 