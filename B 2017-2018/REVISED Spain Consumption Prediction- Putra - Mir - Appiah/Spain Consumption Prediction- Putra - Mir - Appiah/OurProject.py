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
    
DF_DataSet = read_file("C:/Users/Akwesi/Desktop/Our_Project/dataset1.csv")
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

#Now we select two days data to analyse
#Selected_Data = dataSet_Normalized['2012-03-29 00:00:00':'2012-04-04 23:45:00'] # for one week
#Selected_Data.plot()
# In order to better check for lag with a better value we make a day slice and plot
Selected_Data1 = dataSet_Normalized['2012-03-20 00:00:00':'2012-03-21 23:45:00'] # for a day
Selected_Data1.plot()
plt.xlabel('Timestamp ')
plt.ylabel('Variables')
plt.show()
#DF = Selected_Data1[['Inside_Temp_sensor','Solar_irradiance']]
#DF.plot()

# Since the plot for wind shows a lag of approx 10hrs we need to shift the plot by 40 time steps
# for Outside temp and Solar_irradiance, we shift by 4 time steps for better correlation
dataSet_Normalized['Wind_10hours'] = dataSet_Normalized['External_Wind[m/s]'].shift(40)
dataSet_Normalized['Outside_Temp_sensor_4hrs'] = dataSet_Normalized['Outside_Temp_sensor'].shift(4)
dataSet_Normalized['Solar_irradiance_1hr'] = dataSet_Normalized['Solar_irradiance'].shift(4)
dataSet_Normalized.dropna(inplace=True)
Selected = dataSet_Normalized[['Inside_Temp_sensor','Wind_10hours','Outside_Temp_sensor_4hrs','Solar_irradiance_1hr']]

Selected['2012-03-20 00:00:00':'2012-03-21 23:45:00'].plot()
plt.xlabel('Timestamp ')
plt.ylabel('Variables')
plt.show()

#We choose to correlate the effect of Wind variation and Outdoor Temperature on Inside Temperature
dataSet_Normalized_sliced = dataSet_Normalized['2012-03-21 00:00:00':'2012-03-25 23:45:00']
Temp_Effect=dataSet_Normalized_sliced[['Inside_Temp_sensor','Outside_Temp_sensor_4hrs','Outside_Temp_sensor']]
Wind_Effect=dataSet_Normalized_sliced[['Inside_Temp_sensor','Wind_10hours','External_Wind[m/s]']]
#Solar_Irrad=Selected_Data1['2012-03-21 00:00:00':'2012-03-22 23:45:00'][['Inside_Temp_sensor','Solar_irradiance_6hrs']]
#Solar_Irrad_wind=Selected_Data1['2012-03-21 00:00:00':'2012-03-22 23:45:00'][['Wind_10hours','Solar_irradiance_6hrs']]

plt.figure()
plt.subplot(2,1,1)
plt.plot(Temp_Effect)
plt.title('T_in & T_out')
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
fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)
dataSet_Normalized_sliced.iloc[:,6].plot(ax=ax1,legend=True,color="r")
dataSet_Normalized_sliced.iloc[:,0].plot(ax=ax2,legend=True,color="b")
ax1.set_ylabel("Outside Temperature sensor, [C]", color="r")
ax2.set_ylabel("Inside Temperature sensor, [C]", color="b")
ax1.tick_params(axis='y',colors='r')
ax2.tick_params(axis='y',colors='b')
plt.show()

#===============================================================================================================
DF_SelectedVariables=dataSet_Normalized[['Inside_Temp_sensor','Outside_Temp_sensor_4hrs','Wind_10hours']]

#Testing Our Model
target = DF_SelectedVariables['Inside_Temp_sensor']
features = DF_SelectedVariables[['Outside_Temp_sensor_4hrs','Wind_10hours']]

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

joined['2012-03-20':'2012-03-27'].plot()
plt.show()
plt.xlabel('Timestamp')
plt.ylabel('variables')

# Calculating the accuracy metrics of the implemented machine learning model
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
R2_score = r2_score(y_test,prediction)
mean_absolute_error = mean_absolute_error(y_test,prediction)
mean_squared_error = mean_squared_error(y_test,prediction)
coeff_variation = np.sqrt(mean_squared_error)/y_test.mean()
print "The R2_score is: "+str(R2_score)
print "The Mean absoulute error is: "+str(mean_absolute_error)
print "The Mean squared error is: "+str(mean_squared_error)
print "The Coefficient of variation is: "+str(coeff_variation)


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
predict_DF_SVR_CV['2012-03-20':'2012-03-27'].plot()
fig = plt.figure()
ax1 = fig.add_subplot(111)
plot = sns.regplot(x='Inside_Temp_sensor', y="Predicted T_in_SVR_CV",
                   data=predict_DF_SVR_CV,ax=ax1,
                   line_kws={"lw":3,"alpha":0.5})
plt.title('Actual indoor T VS. Predicted Indoor T from SVM')
plot.set_xlim([0,1])
plot.set_ylim([0,1])
plot.set_xlabel('Actual indoor Temp')
plot.set_ylabel('Predicted Indoor T from SVM')
regline = plot.get_lines()[0];
regline.set_color('coral')
plt.show()

# Calculating the accuracy metrics of the implemented machine learning model
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
R2_score_DF_SVR_CV = r2_score(predict_DF_SVR_CV['Inside_Temp_sensor'],predict_DF_SVR_CV["Predicted T_in_SVR_CV"])
mean_absolute_error_SVR_CV = mean_absolute_error(predict_DF_SVR_CV["Inside_Temp_sensor"],predict_DF_SVR_CV["Predicted T_in_SVR_CV"])
mean_squared_error_SVR_CV = mean_squared_error(predict_DF_SVR_CV["Inside_Temp_sensor"],predict_DF_SVR_CV["Predicted T_in_SVR_CV"])
coeff_variation_SVR_CV = np.sqrt(mean_squared_error_SVR_CV)/predict_DF_SVR_CV["Inside_Temp_sensor"].mean()
print "The R2_score for SVR is: "+str(R2_score_DF_SVR_CV)
print "The Mean absoulute error for SVR is: "+str(mean_absolute_error_SVR_CV)
print "The Mean squared error for SVR is: "+str(mean_squared_error_SVR_CV)
print "The Coefficient of variation for SVR is: "+str(coeff_variation_SVR_CV)

   
#========================================================================================================================================         
# Method 3: Implementing Random Forest regression approach to test the performance of the machine learning model
from sklearn.model_selection import cross_val_predict 
from sklearn.ensemble import RandomForestRegressor
reg_RF = RandomForestRegressor()
predict_RF_CV = cross_val_predict(reg_RF,features,target,cv=10) 
predict_DF_RF_CV=pd.DataFrame(predict_RF_CV, index = target.index,columns=['Predicted_T_in_RF_CV'])
predict_DF_RF_CV = predict_DF_RF_CV.join(target).dropna()

predict_DF_RF_CV['2012-03-20':'2012-03-27'].plot()
fig = plt.figure()
ax1 = fig.add_subplot(111)
plot = sns.regplot(x='Inside_Temp_sensor', y='Predicted_T_in_RF_CV',
                   data=predict_DF_RF_CV,ax=ax1,
                   line_kws={"lw":3,"alpha":0.5})
plt.title('Actual indoor T VS. Predicted Indoor T from RF')
plot.set_xlim([0,1])
plot.set_ylim([0,1])
plot.set_xlabel('Actual indoor Temp')
plot.set_ylabel('Predicted Indoor T from RF')
regline = plot.get_lines()[0];
regline.set_color('r')
plt.show()

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
R2_score_DF_RF_CV = r2_score(predict_DF_RF_CV['Inside_Temp_sensor'],predict_DF_RF_CV['Predicted_T_in_RF_CV'])
mean_absolute_error_DF_CV =  mean_absolute_error(predict_DF_RF_CV['Inside_Temp_sensor'],predict_DF_RF_CV['Predicted_T_in_RF_CV'])
mean_squared_error_DF_CV = mean_squared_error(predict_DF_RF_CV['Inside_Temp_sensor'],predict_DF_RF_CV['Predicted_T_in_RF_CV'])
coeff_variation_DF_CV = np.sqrt(mean_squared_error_DF_CV)/predict_DF_RF_CV['Inside_Temp_sensor'].mean()
print "The R2_score for RF is: "+str(R2_score_DF_RF_CV)
print "The Mean absoulute for RF error is: "+str(mean_absolute_error_DF_CV)
print "The Mean squared error for RF is: "+str(mean_squared_error_DF_CV)
print "The Coefficient of variation for RF is: "+str(coeff_variation_DF_CV)

#Making a table of our anaylsis from the machine Learning process..using Dictionary
df = pd.DataFrame({'Metrics':['R2_score','mean absolute error','mean squared error','coefficient variation'],
                'Linear Regression':[R2_score,mean_absolute_error,mean_squared_error,coeff_variation],
                'SVM':[R2_score_DF_SVR_CV,mean_absolute_error_SVR_CV,mean_squared_error_SVR_CV,coeff_variation_SVR_CV],
                'Random Forest':[R2_score_DF_RF_CV,mean_absolute_error_DF_CV,mean_squared_error_DF_CV,coeff_variation_DF_CV]})
df.set_index('Metrics',inplace=True)
df.index.name=None
df.round(4)

