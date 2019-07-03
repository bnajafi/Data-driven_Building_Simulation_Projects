# -*- coding: utf-8 -*-
# In this project, I'm trying predict the indoor room temperature by means of correlation and machine learning methods from the data of sun irradiation, outdoor temperature, room lighting and outdoor humidity.
# Units of data used:
# 1 --> Date: in UTC. 
# 2 --> Time: in UTC. 
# 3 --> Indoor room temperature, in ºC.  
# 4 --> Room lighting, in Lux.   
# 5 --> Sun irradiance, in W/m2. 
# 6 --> Outdoor temperature, in ºC. 
# 7 --> Outdoor relative humidity, in %. 

# First, I'm gonna import all the required modules.
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

InputFolder = r"C:\Users\Mirko\Desktop\DataDrivenBuilding_Ferrari"
os.chdir(InputFolder)
# Now I'm going to read the initial data from the .csv file and convert it into a dataframe. I'm also changing the format of index to Date Time format. 
# From the data, I select days from 15/03/2012 to 22/03/2012, one week.
ExcelFileName = "BuildingDataSet.csv"
Path_ExcelFile = os.path.join(InputFolder,ExcelFileName)
DF_StartingDataSet = pd.read_csv(Path_ExcelFile,sep=",",index_col=0)
PreviousIndex = DF_StartingDataSet.index
NewParsedIndex = pd.to_datetime(PreviousIndex)  
DF_StartingDataSet.index = NewParsedIndex 
DF_TargetDataSet = DF_StartingDataSet["15-03-2012":"22-03-2012"]
# To describe my DataFrame.
DF_TargetDataSet.describe() 

# Here I'm selecting specific columns of the DataFrame so as to corelate target and features. In this case, I'm choosing indoor temperature, sun irradiance, outdoor temperature, room lighting, outdoor humidity.
DF_Choosen = DF_TargetDataSet[['Temperature_Habitacion_Sensor','Meteo_Exterior_Piranometro','Temperature_Exterior_Sensor','Lighting_Habitacion_Sensor','Humedad_Exterior_Sensor']]  
DF_Choosen.describe()
# For simplicity, I'm going to translate the name of the column from spanish to english.
DF_Choosen.rename(columns={"Temperature_Habitacion_Sensor":"Indoor Room Temperature","Meteo_Exterior_Piranometro":"Sun Irradiance","Temperature_Exterior_Sensor":"Outdoor Temperature","Lighting_Habitacion_Sensor":"Room Lighting","Humedad_Exterior_Sensor":"Outdoor Humidity"},inplace=True) 
DF_Choosen["Sun Irradiance"][DF_Choosen["Sun Irradiance"] < 0] = 0  

# To create lagged features, I first make a copy of my selected DataFrame. Then, defining and using a proper function, I will create the lagged features for sun irradiation, outdoor temperature and indoor room temperature.
DF_Lagged = DF_Choosen.copy() 

def lag_feature(DF,ColumnName,LagInterval):    
        if(ColumnName=="Sun Irradiance"):
            for i in range(3,LagInterval+1):
                NewColumnName = ColumnName+" -"+str(i)+"hr"
                DF[NewColumnName]=(DF[ColumnName]).shift(i*4)
        elif(ColumnName=="Outdoor Temperature"):
            for i in range(1,LagInterval-1):
                NewColumnName = ColumnName+" -"+str(i)+"hr"
                DF[NewColumnName]=(DF[ColumnName]).shift(i*4)
        elif(ColumnName=="Indoor Room Temperature"):
            for i in range(2,LagInterval-1,2):
                NewColumnName = ColumnName+" "+str(i*15)+"minutes before"
                DF[NewColumnName]=(DF[ColumnName]).shift(i)
        return DF                    

DF_Lagged = lag_feature(DF_Lagged,"Sun Irradiance",6)
DF_Lagged = lag_feature(DF_Lagged,"Outdoor Temperature",6)
DF_Lagged = lag_feature(DF_Lagged,"Indoor Room Temperature",6)
DF_Lagged.dropna(inplace=True)
# To describe my lagged DataFrame.
DF_Lagged.describe()

# With the next sequence of commands, I'm trying to see if there is any strong correlation between the different element of the DataFrame.
DF_Lagged.corr()
# In the next plot, I can see visually the strength of the correlation.
fig = plt.figure("Figure for providing insight about Correlations")
plot = fig.add_axes()
plot = sns.heatmap(DF_Lagged.corr(), annot=False)
plot.xaxis.tick_top() 
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.show()

# I'm now introducing a function to normalize the data and eliminate the problem of different unit. 

def normalize(DF):
    return (DF-DF.min())/(DF.max()-DF.min())

#I'm now plotting some specific columns of the normalized DataFrame.
# About this next plot, something can be noticed:
# 1) the indoor room temperature increases w.r.t. outdoor temperature almost in real time while it takes time for sun irradiation to heat up the room, therefore sun irradiation has its peak earlier than room indoor temperature; 
# 2) in some period, sun irradiance is irregular. This is probably due to rainy or cloudy days;  
# 3) during night time, sun irradiation is nill.

NormDF=normalize(DF_Choosen)
PlotDF=NormDF[["Indoor Room Temperature","Sun Irradiance","Outdoor Temperature"]]
PlotDF.plot()
plt.title("Normalized plot")
plt.show()

# I can now add some time-related features like hour, day of the week, day or night, weekend.
DF_Lagged['Hour'] = DF_Lagged.index.hour
DF_Lagged['Day Of The Week'] = DF_Lagged.index.dayofweek
# 1 if day, 0 if night.
DF_Lagged['Day Or Night'] = np.where((DF_Lagged['Hour']>=6)&(DF_Lagged['Hour']<=17), 1, 0)

def WeekendFinder(day):
    if (day==0 or day==1):
        weekend = 1
    else:
        weekend = 0
    return weekend 

DF_Lagged['Weekend'] = [WeekendFinder(s) for s in DF_Lagged['Day Of The Week']]
DF_Lagged.dropna(inplace=True)
DF_Lagged.head()

# So, now I've to split the DataFrame in target and features. As target I want to choose the indoor room temperature.

DF_Target = DF_Lagged["Indoor Room Temperature"]
DF_Features = DF_Lagged.drop("Indoor Room Temperature",axis=1)

# I'm now creating a training dataset for machine learning and providing an independent testset which follows same probabilistic distribution of the training one.

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(DF_Features,DF_Target,test_size=0.2,random_state=41234)

# I'm now implementing linear regression technique.

from sklearn import linear_model 
linear_reg = linear_model.LinearRegression()

# Now I want to fit my model ( which means train it ).
linear_reg.fit(X_train,Y_train)
Predicted_linearReg_split = linear_reg.predict(X_test)
Predicted_DF_linearReg_split = pd.DataFrame(Predicted_linearReg_split, index=Y_test.index, columns=["Indoor Room Temperature Prediction_LR"])
# To match indexes.
Predicted_DF_linearReg_split = Predicted_DF_linearReg_split.join(Y_test)
LearnedDataset = pd.DataFrame(Predicted_DF_linearReg_split).dropna()

# I want now to plot the learned Dataset in order to verify the quality of the prediction.
# From the plot it can be noticed how well the prediction has been made.
LearnedDataset.plot()
plt.title("Plot for Train-Test approach_LR")
plt.show()

# Now I want to calculate how accurate the prediction is. 
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

# Perfectly accurate model gives metric_R2_score as 1.
R2_linearReg_split = r2_score(Y_test,Predicted_linearReg_split) 
MAE_linearReg_split = mean_absolute_error(Y_test,Predicted_linearReg_split)
MSE_linearReg_split = mean_squared_error(Y_test,Predicted_linearReg_split)

# Let's print the results.
print "The R2_score is "+str(R2_linearReg_split)
print "The mean absoulute error is "+str(MAE_linearReg_split)
print "The mean squared error is "+str(MSE_linearReg_split)

# I'm now implementing K-fold cross-validation approach, in order to test the performances of the machine learning model.
from sklearn.model_selection import cross_val_predict

# cv is the number of subplots.
Predicted_linearReg_CV = cross_val_predict(linear_reg,DF_Features,DF_Target,cv=10)
Predicted_DF_linearReg_CV=pd.DataFrame(Predicted_linearReg_CV, index = DF_Target.index,columns=["Indoor Room Temperature Prediction_CV"])
Predicted_DF_linearReg_CV = Predicted_DF_linearReg_CV.join(DF_Target)

# I want now to plot the learned Dataset in order to verify the quality of the prediction.
# From the plot it can be noticed how well the prediction has been made.
Predicted_DF_linearReg_CV.plot()
plt.title("Plot for Cross-Validation approach_CV")
plt.show()

# Now I want to calculate how accurate the prediction is in this case.
R2_linearReg_CV = r2_score(Predicted_linearReg_CV,DF_Target)
MAE_linearReg_CV = mean_absolute_error(Predicted_linearReg_CV, DF_Target)
MSE_linearReg_CV = mean_squared_error(Predicted_linearReg_CV, DF_Target)

# Let's print the results.
print "The R2_score is "+str(R2_linearReg_CV)
print "The mean absoulute error is "+str(MAE_linearReg_CV)
print "The mean squared error is "+str(MSE_linearReg_CV)

# The first way of training the model gives a better result. Also this second one is very good though.
# I'm now implementing Random Forest regression technique.
from sklearn.ensemble import RandomForestRegressor
reg_RF = RandomForestRegressor()
Predicted_RF_CV = cross_val_predict(reg_RF,DF_Features,DF_Target,cv=10) 
Predicted_DF_RF_CV=pd.DataFrame(Predicted_RF_CV, index = DF_Target.index,columns=["Indoor Room Temperature Prediction_CV"])
Predicted_DF_RF_CV = Predicted_DF_RF_CV.join(DF_Target).dropna()

# I want now to plot the learned Dataset in order to verify the quality of the prediction.
Fig = plt.figure("Actual Vs Prediction with Random Forest")
ax1 = Fig.add_subplot(111)
plot = sns.regplot(x="Indoor Room Temperature", y="Indoor Room Temperature Prediction_CV",
                   data=Predicted_DF_RF_CV,ax=ax1,
                   line_kws={"lw":3,"alpha":0.5})
plt.title('Predicted Indoor Room Temperature VS. Actual Indoor Room Temperature from 15/03/2012 to 22/03/2012')
plot.set_xlim([10,26])                                                                  
plot.set_ylim([10,26])                   
plot.set_xlabel('Actual Indoor Room Temperature [deg C]')
plot.set_ylabel('Predicted Indoor Room Temperature [deg C]')
regline = plot.get_lines()[0];
regline.set_color('yellow')

# Now I want to calculate how accurate the prediction are in this case.
R2_RF_CV = r2_score(Predicted_RF_CV,DF_Target)
MAE_RF_CV = mean_absolute_error(Predicted_RF_CV,DF_Target)
MSE_RF_CV = mean_squared_error(Predicted_RF_CV,DF_Target)

# Let's print the results.
print "\nThe R2_score is: "+str(R2_RF_CV)
print "The Mean absoulute error is: "+str(MAE_RF_CV)
print "The Mean squared error is: "+str(MSE_RF_CV)

# I want now to use continuous learning.

DF_IndoorTemperaturePrediction = pd.DataFrame(index = DF_Lagged.index)
Period_of_training = pd.Timedelta(4, unit ="d")
FirstTimeStamp_measured = DF_Lagged.index[0]
LastTimeStamp_measured = DF_Lagged.index[-1]                                                     
# This is not included, it arrives until  03/22/12 at 23.30.

FirstTimeStamp_toPredict = FirstTimeStamp_measured + Period_of_training

Training_startTimeStamp = FirstTimeStamp_measured
Trainig_endTimeStamp = FirstTimeStamp_toPredict                                             
TimeStamp_toPredict = FirstTimeStamp_toPredict
# It starts at 03/23/12 at 00.00

DF_IndoorTemperaturePrediction = DF_IndoorTemperaturePrediction.truncate(before = Trainig_endTimeStamp) 

while (TimeStamp_toPredict < LastTimeStamp_measured): 
    DF_feature_train = DF_Features.truncate(before = Training_startTimeStamp, after = Trainig_endTimeStamp)        
    DF_target_train = DF_Target.truncate(before = Training_startTimeStamp, after = Trainig_endTimeStamp)
    DF_feature_test = DF_Features.loc[TimeStamp_toPredict].values.reshape(1,-1)
    DF_target_test = DF_Target.loc[TimeStamp_toPredict]
    # Now I'm training my model, using the LR technique.
    linear_reg.fit(DF_feature_train,DF_target_train)
    Predicted_Temperature = linear_reg.predict(DF_feature_test)
    DF_IndoorTemperaturePrediction.loc[TimeStamp_toPredict,"Predicted"] = Predicted_Temperature
    DF_IndoorTemperaturePrediction.loc[TimeStamp_toPredict,"Real"] = DF_target_test
    TimeStamp_toPredict = TimeStamp_toPredict + pd.Timedelta(15, unit = "m")
    Trainig_endTimeStamp = Trainig_endTimeStamp + pd.Timedelta(15, unit = "m")
    Training_startTimeStamp = Training_startTimeStamp + pd.Timedelta(15, unit = "m")
    
DF_IndoorTemperaturePrediction.dropna(inplace = True)
R2_score_continuous_linearReg = r2_score(DF_IndoorTemperaturePrediction[["Real"]],DF_IndoorTemperaturePrediction[["Predicted"]])
print("R2 value: "+str(R2_score_continuous_linearReg))
# I'm now plotting the behaviour of this model.
Fig = plt.figure("Actual Vs Prediction with LR")
ax1 = Fig.add_subplot(111)
plot = sns.regplot(x="Real", y="Predicted",
                   data=DF_IndoorTemperaturePrediction,ax=ax1,
                   line_kws={"lw":3,"alpha":0.5})
plt.title('Predicted Indoor Room Temperature VS. Actual Indoor Room Temperature from 19/03/2012 to 22/03/2012')
plot.set_xlim([10,23])                                                                  
plot.set_ylim([10,23])                   
plot.set_xlabel('Actual Indoor Room Temperature [deg C]')
plot.set_ylabel('Predicted Indoor Room Temperature [deg C]')
regline = plot.get_lines()[0];
regline.set_color('red')

# Let's do the same thing, this time using the Random Forest technique.
DF_IndoorTemperaturePrediction = pd.DataFrame(index = DF_Lagged.index)
Period_of_training = pd.Timedelta(4, unit ="d")
FirstTimeStamp_measured = DF_Lagged.index[0]
LastTimeStamp_measured = DF_Lagged.index[-1]                                                     
FirstTimeStamp_toPredict = FirstTimeStamp_measured + Period_of_training
Training_startTimeStamp = FirstTimeStamp_measured
Trainig_endTimeStamp = FirstTimeStamp_toPredict                                             
TimeStamp_toPredict = FirstTimeStamp_toPredict
# It starts at 03/23/12 at 00.00

DF_IndoorTemperaturePrediction = DF_IndoorTemperaturePrediction.truncate(before = Trainig_endTimeStamp) 

while (TimeStamp_toPredict < LastTimeStamp_measured):
    DF_feature_train = DF_Features.truncate(before = Training_startTimeStamp, after = Trainig_endTimeStamp)         # tronca tutto quello prima di before e tutto quello dopo after
    DF_target_train = DF_Target.truncate(before = Training_startTimeStamp, after = Trainig_endTimeStamp)
    DF_feature_test = DF_Features.loc[TimeStamp_toPredict].values.reshape(1,-1)
    DF_target_test = DF_Target.loc[TimeStamp_toPredict]
    # Now I'm training my model, using the RF technique.
    reg_RF.fit(DF_feature_train,DF_target_train)
    Predicted_Temperature = reg_RF.predict(DF_feature_test)
    DF_IndoorTemperaturePrediction.loc[TimeStamp_toPredict,"Predicted"] = Predicted_Temperature
    DF_IndoorTemperaturePrediction.loc[TimeStamp_toPredict,"Real"] = DF_target_test
    TimeStamp_toPredict = TimeStamp_toPredict + pd.Timedelta(15, unit = "m")
    Trainig_endTimeStamp = Trainig_endTimeStamp + pd.Timedelta(15, unit = "m")
    Training_startTimeStamp = Training_startTimeStamp + pd.Timedelta(15, unit = "m")

DF_IndoorTemperaturePrediction.dropna(inplace = True)

R2_score_continuous_RF = r2_score(DF_IndoorTemperaturePrediction[["Real"]],DF_IndoorTemperaturePrediction[["Predicted"]])

# I'm now plotting the behaviour of this model.
Fig = plt.figure("Actual Vs Prediction with RF")
ax1 = Fig.add_subplot(111)
plot = sns.regplot(x="Real", y="Predicted",
                   data=DF_IndoorTemperaturePrediction,ax=ax1,
                   line_kws={"lw":3,"alpha":0.5})
plt.title('Predicted Indoor Room Temperature VS. Actual Indoor Room Temperature from 19/03/2012 to 22/03/2012')
plot.set_xlim([10,23])                                                                  
plot.set_ylim([10,23])                   
plot.set_xlabel('Actual Indoor Room Temperature [deg C]')
plot.set_ylabel('Predicted Indoor Room Temperature [deg C]')
regline = plot.get_lines()[0];
regline.set_color('green')

























