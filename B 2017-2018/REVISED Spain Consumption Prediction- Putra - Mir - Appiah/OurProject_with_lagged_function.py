# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_file(file_name):

    data_set = pd.read_csv(file_name,sep=',',index_col=4)
    
    desired_index = data_set.index

    datetime_index = pd.to_datetime(desired_index)
    
    data_set.index = datetime_index
    
    return data_set
    
DF_DataSet = read_file("C:/Users/Danish/Desktop/Project Building System/DataDriven Analysis/dataset1.csv")
DF_DataSet.head()
DF_DataSet.describe() #obtain a summary of the data...e.g mean max min std etc

#we use seaborn to check the correlation of the variables
import seaborn as sns
fig = plt.figure()
plot = fig.add_axes()
plot = sns.heatmap(DF_DataSet.corr(), annot=False)
plot.xaxis.tick_top() 
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.show()

DF_SelectedVariables = DF_DataSet[['4:Temperature_Habitacion_Sensor', '5:Weather_Temperature','14:Meteo_Exterior_Viento',
                                    '22:Temperature_Exterior_Sensor','18:Meteo_Exterior_Piranometro']]  
DF_SelectedVariables.rename(columns={'4:Temperature_Habitacion_Sensor':'Inside_Temp_sensor','5:Weather_Temperature':'Weather_Temperature',
                                    '14:Meteo_Exterior_Viento':'External_Wind[m/s]','22:Temperature_Exterior_Sensor': 'Outside_Temp_sensor',
                                    '18:Meteo_Exterior_Piranometro':'Solar_irradiance'},inplace=True)

#DF_SelectedVariables.columns
DF_SelectedVariables.head()

import seaborn as sns
fig = plt.figure()
plot = fig.add_axes()
plot = sns.heatmap(DF_SelectedVariables.corr(), annot=False)
plot.xaxis.tick_top() 
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.show()

# for one week..Without Normalization
Selected_Data = DF_SelectedVariables['2012-03-29 00:00:00':'2012-04-04 23:45:00'] 
Selected_Data.plot()
plt.xlabel('Timestamp ')
plt.ylabel('Variables')
plt.show()

#Nomalizing all variable data by defining a function
def normalize(df):
    return (df - df.min())/(df.max()-df.min())   
dataSet_Normalized = normalize(DF_SelectedVariables)
dataSet_Normalized.plot()
plt.title('Normalized Variables vs Timestamp')
plt.xlabel('Timestamp ')
plt.ylabel('Variables')
plt.show()

# In order to better check for lag with a better value we make a day slice and plot
Selected_Data1 = dataSet_Normalized['2012-03-20 00:00:00':'2012-03-21 23:45:00'] # for a day
Selected_Data1.plot()
plt.xlabel('Timestamp ')
plt.ylabel('Variables')
plt.show()

#--------------------------------------------------------------------------------------------------------
# For giving lagged features, we first create the copy of our selected data frame. 
#Then defining and using the function,we will create the lagged features for Sun Irradiation and Outdoor Temperature.
DF_lagged=DF_SelectedVariables.copy() 

'''A function to create lag columns of selected columns passed as arguments'''
def lag_column(df,column_names,lag_period):    
    for column_name in column_names:
        if(column_name=="Solar_irradiance"):
            for i in range(3,lag_period+1):
                new_column_name = column_name+"_"+str(i)+"hr"
                df[new_column_name]=(df[column_name]).shift(i*4)
        elif(column_name=="Outside_Temp_sensor"):
            for i in range(1,lag_period+1):
                new_column_name = column_name+"_"+str(i)+"hr"
                df[new_column_name]=(df[column_name]).shift(i*4)
        else:
            for i in range(1,lag_period*4):
                new_column_name = column_name+"_"+str(i)+"hr before"
                df[new_column_name]=(df[column_name]).shift(-i*4)
                  
    return df      

DF_lagged=lag_column(DF_lagged,["Solar_irradiance","Outside_Temp_sensor","Inside_Temp_sensor"],6)  #Passing values in function
DF_lagged.dropna(inplace=True)

DF_lagged.head(0)

#Creating a plot using heatmap functionality to provide correlations between columns of dataset
fig = plt.figure("Figure for providing insight about Correlations")
plot = fig.add_axes()
plot = sns.heatmap(DF_lagged.corr(), annot=False)
plot.xaxis.tick_top() 
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.show()
#--------------------------------------------------------------------------------------------------------------------

# Since the plot for wind shows a lag of approx 10hrs we need to shift the plot by 40 time steps
# for Outside temp and Solar_irradiance, we shift by 4 time steps for better correlation
dataSet_Normalized['Wind_10hours'] = dataSet_Normalized['External_Wind[m/s]'].shift(40)
dataSet_Normalized['Inside_Temp_sensor_1hr'] = dataSet_Normalized['Inside_Temp_sensor'].shift(-4)
dataSet_Normalized['Solar_irradiance_3hr'] = dataSet_Normalized['Solar_irradiance'].shift(12)
dataSet_Normalized.dropna(inplace=True)
Selected = dataSet_Normalized[['Inside_Temp_sensor','Inside_Temp_sensor_1hr','Wind_10hours','Outside_Temp_sensor','Solar_irradiance_3hr']]

DF_lagged['2012-03-20 00:00:00':'2012-03-21 23:45:00'].plot()
plt.xlabel('Timestamp ')
plt.ylabel('Variables')
plt.show()

#for 24hrs plot
DF_lagged['2012-03-21 00:00:00':'2012-03-21 23:59:00'].plot()
plt.xlabel('Timestamp ')
plt.ylabel('Variables')
plt.show()

#We choose to correlate the effect of Wind variation and Outdoor Temperature on Inside Temperature
dataSet_Normalized_sliced = dataSet_Normalized['2012-03-21 00:00:00':'2012-03-21 23:59:00']
Temp_Effect=dataSet_Normalized_sliced[['Solar_irradiance_3hr','Inside_Temp_sensor_1hr','Outside_Temp_sensor']]
Wind_Effect=dataSet_Normalized_sliced[['Inside_Temp_sensor_1hr','Wind_10hours','External_Wind[m/s]']]
#Solar_Irrad=Selected_Data1['2012-03-21 00:00:00':'2012-03-22 23:45:00'][['Inside_Temp_sensor','Solar_irradiance_6hrs']]
#Solar_Irrad_wind=Selected_Data1['2012-03-21 00:00:00':'2012-03-22 23:45:00'][['Wind_10hours','Solar_irradiance_6hrs']]

plt.figure()
plt.subplot(2,1,1)
plt.plot(Temp_Effect)
plt.title('Variation of T_in with T_out and Solar Irradiance')
plt.xlabel('Time')
plt.ylabel('variables')
plt.legend(Temp_Effect)

plt.subplot(2,1,2)
plt.plot(Wind_Effect)
plt.title('T_in & Wind_speed')
plt.xlabel('Time')
plt.ylabel('variables')
plt.tight_layout()
plt.legend(Wind_Effect)
plt.show()

#checking correlation after lagging 
fig = plt.figure()
plot = fig.add_axes()
plot = sns.heatmap(dataSet_Normalized_sliced.corr(), annot=False)
plot.xaxis.tick_top() 
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.show()

#dataSet_Normalized_sliced.columns
#Variation of T out with T in
fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)

dataSet_Normalized_sliced.iloc[:,3].plot(ax=ax1,legend=True,color="r")
dataSet_Normalized_sliced.iloc[:,6].plot(ax=ax2,legend=True,color="b")
ax1.set_ylabel("Outside Temperature sensor, [C]", color="r")
ax2.set_ylabel("Inside Temperature sensor, [C]", color="b")
ax1.tick_params(axis='y',colors='r')
ax2.tick_params(axis='y',colors='b')
plt.tight_layout()
plt.show()

#Variation of T in  with Solar irradiation
fig1 = plt.figure()
ax3 = fig1.add_subplot(2,1,1)
ax4 = fig1.add_subplot(2,1,2)
dataSet_Normalized_sliced.iloc[:,7].plot(ax=ax3,legend=True,color="coral")
dataSet_Normalized_sliced.iloc[:,6].plot(ax=ax4,legend=True,color="b")
ax1.set_ylabel("Solar_irradiance[W/m2]", color="coral")
ax2.set_ylabel("Inside Temperature sensor, [C]", color="b")
ax1.tick_params(axis='y',colors='coral')
ax2.tick_params(axis='y',colors='b')
plt.tight_layout()
plt.show()


"""
plt.close('all')
"""
#===============================================================================================================
DF_SelectedVariables=DF_lagged[['Inside_Temp_sensor_1hr before','Outside_Temp_sensor','Solar_irradiance_3hr','External_Wind[m/s]']]

#Testing Our Model
target = DF_SelectedVariables['Inside_Temp_sensor_1hr before']
features = DF_SelectedVariables[['Outside_Temp_sensor','Solar_irradiance_3hr','External_Wind[m/s]']]

#.......................................................................................................
#Method 1:Using Linear regression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=123524)

from sklearn import linear_model
linear_reg = linear_model.LinearRegression()
linear_reg.fit(X_train,y_train)
prediction = linear_reg.predict(X_test)
predict_series = pd.Series(prediction.ravel(),index=y_test.index).rename('Predicted_T_in')
joined = pd.DataFrame(predict_series).join(y_test)
#joined.isnull()

#for 24hrs
joined['2012-03-21 00:00:00':'2012-03-21 23:59:00'].plot()
plt.show()
plt.xlabel('Timestamp')
plt.ylabel('variables')

#for one Month
#joined['2012-03-15':'2012-04-15'].plot()
#plt.xlabel('Timestamp')
#plt.ylabel('variables')
#plt.show()

# Calculating the accuracy metrics of the implemented machine learning model
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
R2_score = r2_score(y_test,prediction)
mean_absolute_error = mean_absolute_error(y_test,prediction)
mean_squared_error = mean_squared_error(y_test,prediction)
coeff_variation = np.sqrt(mean_squared_error)/y_test.mean()
print ("The R2_score is: "+str(R2_score))
print ("The Mean absoulute error is: "+str(mean_absolute_error))
print ("The Mean squared error is: "+str(mean_squared_error))
print ("The Coefficient of variation is: "+str(coeff_variation))


#========================================================================================================
#Method 2:Using support vector machines,SVM
#Input for SVR should be normalized tables only
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVR
SVR_reg = SVR(kernel='rbf',C=10,gamma=1)
predict_SVR_CV = cross_val_predict(SVR_reg,features,target,cv=10)
predict_DF_SVR_CV=pd.DataFrame(predict_SVR_CV, index = target.index,columns=["Predicted T_in_SVR_CV"])
predict_DF_SVR_CV = predict_DF_SVR_CV.join(target).dropna()

# Plotting the learned dataset and verifying the predicted values with actual ones
predict_DF_SVR_CV['2012-03-21 00:00:00':'2012-03-21 23:59:00'].plot()
plt.show()

#for  one month prediction
#predict_DF_SVR_CV['2012-03-15':'2012-04-15'].plot()
#plt.show()

fig = plt.figure("Actual Vs Prediction by Random Forest")
ax1 = fig.add_subplot(111)
plot = sns.regplot(x='Inside_Temp_sensor_1hr before', y='Predicted T_in_SVR_CV',
                   data = predict_DF_SVR_CV,ax=ax1,line_kws={"lw":3,"alpha":0.5})
                   
plt.title('Predicted Inside Room Temp VS. Actual inside room Temperature')
regline = plot.get_lines()[0];
regline.set_color('coral')
plot.set_xlabel('Indoor Temperature Sensor')
plot.set_ylabel('Predicted Inside_Temp')
plt.show()

# Calculating the accuracy metrics of the implemented machine learning model
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
R2_score_DF_SVR_CV = r2_score(predict_DF_SVR_CV['Inside_Temp_sensor_1hr before'],predict_DF_SVR_CV["Predicted T_in_SVR_CV"])
mean_absolute_error_SVR_CV = mean_absolute_error(predict_DF_SVR_CV["Inside_Temp_sensor_1hr before"],predict_DF_SVR_CV["Predicted T_in_SVR_CV"])
mean_squared_error_SVR_CV = mean_squared_error(predict_DF_SVR_CV["Inside_Temp_sensor_1hr before"],predict_DF_SVR_CV["Predicted T_in_SVR_CV"])
coeff_variation_SVR_CV = np.sqrt(mean_squared_error_SVR_CV)/predict_DF_SVR_CV["Inside_Temp_sensor_1hr before"].mean()
print ("The R2_score for SVR is: "+str(R2_score_DF_SVR_CV))
print ("The Mean absoulute error for SVR is: "+str(mean_absolute_error_SVR_CV))
print ("The Mean squared error for SVR is: "+str(mean_squared_error_SVR_CV))
print ("The Coefficient of variation for SVR is: "+str(coeff_variation_SVR_CV))

   
#========================================================================================================================================         
# Method 3: Implementing Random Forest regression approach to test the performance of the machine learning model
from sklearn.model_selection import cross_val_predict 
from sklearn.ensemble import RandomForestRegressor
reg_RF = RandomForestRegressor()
predict_RF_CV = cross_val_predict(reg_RF,features,target,cv=10) 
predict_DF_RF_CV=pd.DataFrame(predict_RF_CV, index = target.index,columns=['Predicted_T_in_RF_CV'])
predict_DF_RF_CV = predict_DF_RF_CV.join(target).dropna()

predict_DF_RF_CV['2012-03-21 00:00:00':'2012-03-21 23:59:00'].plot()

fig = plt.figure("Actual Vs Prediction by Random Forest")
ax1 = fig.add_subplot(111)
plot = sns.regplot(x='Inside_Temp_sensor_1hr before', y='Predicted_T_in_RF_CV',
                   data = predict_DF_RF_CV,ax=ax1,line_kws={"lw":3,"alpha":0.5})
                   
plt.title('Predicted Inside Room Temp VS. Actual inside room Temperature')
regline = plot.get_lines()[0];
regline.set_color('red')
plot.set_xlabel('Inside_Temp_sensor_1hr before')
plot.set_ylabel('Predicted Inside_Temp')
plt.show()

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
R2_score_DF_RF_CV = r2_score(predict_DF_RF_CV['Inside_Temp_sensor_1hr before'],predict_DF_RF_CV['Predicted_T_in_RF_CV'])
mean_absolute_error_DF_CV =  mean_absolute_error(predict_DF_RF_CV['Inside_Temp_sensor_1hr before'],predict_DF_RF_CV['Predicted_T_in_RF_CV'])
mean_squared_error_DF_CV = mean_squared_error(predict_DF_RF_CV['Inside_Temp_sensor_1hr before'],predict_DF_RF_CV['Predicted_T_in_RF_CV'])
coeff_variation_DF_CV = np.sqrt(mean_squared_error_DF_CV)/predict_DF_RF_CV['Inside_Temp_sensor_1hr before'].mean()
print ("The R2_score for RF is: "+str(R2_score_DF_RF_CV))
print ("The Mean absoulute for RF error is: "+str(mean_absolute_error_DF_CV))
print ("The Mean squared error for RF is: "+str(mean_squared_error_DF_CV))
print ("The Coefficient of variation for RF is: "+str(coeff_variation_DF_CV))

#Making a table of our anaylsis from the machine Learning process..using Dictionary
df = pd.DataFrame({'Metrics':['R2_score','mean absolute error','mean squared error','coefficient variation'],
                'Linear Regression':[R2_score,mean_absolute_error,mean_squared_error,coeff_variation],
                'SVM':[R2_score_DF_SVR_CV,mean_absolute_error_SVR_CV,mean_squared_error_SVR_CV,coeff_variation_SVR_CV],
                'Random Forest':[R2_score_DF_RF_CV,mean_absolute_error_DF_CV,mean_squared_error_DF_CV,coeff_variation_DF_CV]})
df.set_index('Metrics',inplace=True)
df.index.name=None
df.round(4)

