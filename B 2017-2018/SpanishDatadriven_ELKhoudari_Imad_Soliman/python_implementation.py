# -*- coding: utf-8 -*-

"""
Data Driven Simulation of a domestic house in Spain

Created on Thu Feb 17 17:15:00 2018

@authored by:
    Ibrahim El Khoudari
    Ahmad Imad
    Mahmoud Abdel Azim Soliman
    
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def normalize(df):
    return (df - df.min())/(df.max()-df.min())  
        
#### Spanish Datadriven Building with Dataset from a monitor system mounted in a domotic house

dataframe = "C:/Users/BOB/Desktop/EETBS Project/SpanishDatadriven_ELKhoudari_Imad_Soliman"
DataFile = dataframe+'/'+"DataSet.csv"
Desired_Data = pd.read_csv(DataFile,sep=',',index_col=2)

#change our index into date and time index
previousIndex= Desired_Data.index
NewparsedIndex = pd.to_datetime(previousIndex)
Desired_Data.index= NewparsedIndex
Desired_Data.head()


# We use this code to find the correlation between all datas and inside temperature
"""
import seaborn as sns
fig = plt.figure()
plot = fig.add_axes()
plot = sns.heatmap(Desired_Data.corr(), annot=False)
plot.xaxis.tick_top() 
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.show()

"""

# chosing the datas that has the highest Correlation with the inside Temperature
Desired_Column = Desired_Data[['3:Temperature_Comedor_Sensor','6:CO2_Comedor_Sensor',
                    '15:Meteo_Exterior_Sol_Oest','18:Meteo_Exterior_Piranometro',
                    '22:Temperature_Exterior_Sensor']]
Desired_Column.describe()
#Renaming columns
Desired_Column.rename(columns={"3:Temperature_Comedor_Sensor":"Indoor Temperature Sensor","18:Meteo_Exterior_Piranometro":"Sun Irradiance", 
                           "22:Temperature_Exterior_Sensor":"Outdoor Temperature Sensor","6:CO2_Comedor_Sensor":"Carbon dioxide level [ppm]",
                             "15:Meteo_Exterior_Sol_Oest":"Sun light in west [lux]"},inplace=True)  #Renaming columns
Desired_Column.head()
Desired_Column.describe()

#Correlation of inside temperature with desired columns                                     
"""
import seaborn as sns
fig = plt.figure()
plot = fig.add_axes()
plot = sns.heatmap(Desired_Column.corr(), annot=False)
plot.xaxis.tick_top() 
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.show()
"""

#Normalize Our Datas to Bound them all under the same scale 
data_Normalized = normalize(Desired_Column)

####   data_Normalized.plot()

data_Normalized.fillna(0,inplace=True) #Dropping nan values

data_lagged = data_Normalized.copy()

# Shifting the graph of each data variables to make them aligned with the inside Temperature
data_lagged['SolarIrradiance_3hours'] = data_lagged['Sun Irradiance'].shift(12)
data_lagged['CarbonDioxide_10hours'] = data_lagged['Carbon dioxide level [ppm]'].shift(40)
data_lagged['OutSide_Temperature_1.5hour'] = data_lagged['Outdoor Temperature Sensor'].shift(6)
data_lagged['SunLight_West_1hour'] = data_lagged['Sun light in west [lux]'].shift(4)

inside_Temperature = data_lagged[["Indoor Temperature Sensor"]]

SolarIrradiance_3hours =data_lagged[["SolarIrradiance_3hours"]]
CarbonDioxide_10hours = data_lagged[["CarbonDioxide_10hours"]]
OutSide_Temperature_1andhalfHour = data_lagged[["OutSide_Temperature_1.5hour"]]
SunLight_West_1hour = data_lagged[["SunLight_West_1hour"]]

#joining all lagged datas and plotting the new correlation with inside temperature
df_joined = inside_Temperature.join([SolarIrradiance_3hours,CarbonDioxide_10hours,OutSide_Temperature_1andhalfHour,SunLight_West_1hour])

"""
import seaborn as sns
fig = plt.figure()
plot = fig.add_axes()
plot = sns.heatmap(df_joined.corr(), annot=False)
plot.xaxis.tick_top() 
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.show()
"""

# Replace NaN by 0
data_lagged.fillna(0,inplace=True) #Dropping nan values

#Plot Our Data Together
"""
data_lagged.plot()
plt.xlabel('Time')
plt.ylabel('variables')
plt.show()
"""

# Choosing Specific Time Interval
DataSet_Sliced = data_lagged["2012-04-28 00:00:00":"2012-04-29 23:45:00"]
"""
DataSet_Sliced.plot()
plt.xlabel('Time')
plt.ylabel('variables')
plt.show()
"""

# Defining Specific Set for each Data
Temperatures = DataSet_Sliced[['Indoor Temperature Sensor','Outdoor Temperature Sensor','OutSide_Temperature_1.5hour']]
Sun_Irradiance = DataSet_Sliced[['Indoor Temperature Sensor','Sun Irradiance','SolarIrradiance_3hours']]
West_Sun_Exposure = DataSet_Sliced[['Indoor Temperature Sensor','Sun light in west [lux]','SunLight_West_1hour']]
Carbon_Dioxide = DataSet_Sliced[['Indoor Temperature Sensor','Carbon dioxide level [ppm]','CarbonDioxide_10hours']]

# Creating Different Subplots with respect to inside Temperature
"""
plt.figure()
plt.subplot(1,1,1)
plt.plot(Temperatures)
plt.title('T_inside vs T_outside')
plt.xlabel('Time')
plt.ylabel('DataSet')
plt.legend(Temperatures)
plt.show()

plt.subplot(1,1,1)
plt.plot(Sun_Irradiance)
plt.title('T_inside vs Sun_Irradiance')
plt.xlabel('Time')
plt.ylabel('DataSet')
plt.legend(Sun_Irradiance)
plt.show()

plt.figure()
plt.subplot(1,1,1)
plt.plot(West_Sun_Exposure)
plt.title('T_inside vs West_Sun_Exposure')
plt.xlabel('Time')
plt.ylabel('DataSet')
plt.legend(West_Sun_Exposure)
plt.show()

plt.subplot(1,1,1)
plt.plot(Carbon_Dioxide)
plt.title('T_inside vs Carbon_Dioxide')
plt.xlabel('Time')
plt.ylabel('DataSet')
plt.legend(Carbon_Dioxide)
plt.show()
"""

#Testing Our Model and Prediction of Inside Temperature

#Defining our Target Value and the Effecting Features

target_data = Desired_Column['Indoor Temperature Sensor']
features_data = Desired_Column[['Carbon dioxide level [ppm]','Sun light in west [lux]',
                                'Sun Irradiance','Outdoor Temperature Sensor']]
                                

#First Method : Linear regression

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features_data, target_data, test_size=0.25, random_state=123524)

from sklearn import linear_model
linear_reg = linear_model.LinearRegression()

linear_reg.fit(X_train,y_train)

prediction = linear_reg.predict(X_test)

predict_series = pd.Series(prediction.ravel(),index=y_test.index).rename('Predicted T_inside')
joined = pd.DataFrame(predict_series).join(y_test)
joined.head()

"""
joined["2012-04-28 00:00:00":"2012-04-29 23:45:00"].plot()
plt.show()
plt.xlabel('Time')
plt.ylabel('variables')
"""

# Displaying the predicted datas
"""
print X_train.shape, y_train.shape
print X_test.shape, y_test.shape
print prediction
"""


"""
plt.scatter(y_test, prediction)
plt.xlabel('True Value')
plt.ylabel('Prediction')
plt.show()
"""

# Plotting The Actual and Predicted Data

"""
fig = plt.figure("Actual Vs Prediction by Linear Model")
ax1 = fig.add_subplot(111)
plot = sns.regplot(x='Indoor Temperature Sensor', y='Predicted T_inside',
                   data=joined,line_kws={"lw":3,"alpha":0.5})                   
plt.title('Actual Vs Prediction by Linear Model')
plot.set_xlabel('Indoor Temperature Sensor')
plot.set_ylabel('Predicted Inside_Temp')
regline = plot.get_lines()[0];
regline.set_color('orange')
plt.show()
"""


# Second Method : Cross-Validation and Random Forest Regression

from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestRegressor
reg_RF = RandomForestRegressor()
predicted_RanFor_CV = cross_val_predict(reg_RF,features_data,target_data,cv=10) 
predicted_DF_RanFor_CV = pd.DataFrame(predicted_RanFor_CV, index = target_data.index,columns=['Predicted T_inside_CV'])
predicted_DF_RanFor_CV = predicted_DF_RanFor_CV.join(target_data)
predicted_DF_RanFor_CV.head()

"""
predicted_DF_RanFor_CV["2012-04-28 00:00:00":"2012-04-29 23:45:00"].plot()
plt.xlabel('Time')
plt.ylabel('variables')
plt.show()
"""

# Plot of Learned Dataset with verification of predicted values vs actual ones
"""
fig = plt.figure("Actual Vs Prediction by Random Forest")
ax1 = fig.add_subplot(111)
plot = sns.regplot(x='Indoor Temperature Sensor', y='Predicted T_inside_CV',
                   data = predicted_DF_RanFor_CV,ax=ax1,line_kws={"lw":3,"alpha":0.5})
                   
plt.title('Predicted Inside Room Temp VS. Actual inside room Temperature')
regline = plot.get_lines()[0];
regline.set_color('red')
plot.set_xlabel('Indoor Temperature Sensor')
plot.set_ylabel('Predicted Inside_Temp')
plt.show()
"""


# Accuracy Analysis of the machine learning model

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
R2_score_DF_RF_CV = r2_score(predicted_DF_RanFor_CV['Indoor Temperature Sensor'],predicted_DF_RanFor_CV['Predicted T_inside_CV'])
mean_absolute_error_DF_CV = mean_absolute_error(predicted_DF_RanFor_CV['Indoor Temperature Sensor'],predicted_DF_RanFor_CV['Predicted T_inside_CV'])
mean_squared_error_DF_CV = mean_squared_error(predicted_DF_RanFor_CV['Indoor Temperature Sensor'],predicted_DF_RanFor_CV['Predicted T_inside_CV'])
coeff_variation_DF_CV = np.sqrt(mean_squared_error_DF_CV)/predicted_DF_RanFor_CV['Indoor Temperature Sensor'].mean()
print "The Accuracy is: "+str(R2_score_DF_RF_CV)
print "The Mean absoulute error is: "+str(mean_absolute_error_DF_CV)
print "The Mean squared error is: "+str(mean_squared_error_DF_CV)
print "The Coefficient of variation is: "+str(coeff_variation_DF_CV)