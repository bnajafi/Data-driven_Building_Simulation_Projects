# -*- coding: utf-8 -*-
#DATA DRIVEN BUILDING SIMULATION
#funaro,romanelli,rossetti

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 

# importing and reading the external files
FilesFolder = "/Users/federicarossetti/Desktop/DATA DRIVEN PROJECT"
os.chdir(FilesFolder) 
DatasFile="spanish-dataset-correct.csv"
Path_DatasFile=os.path.join(FilesFolder,DatasFile)
Database=pd.read_csv(Path_DatasFile,sep=";", index_col=0)
Database.head()

#changing of the indexes of the dataframe to the format YYYY-MM-DD hh:mm:ss
PreviousIndex = Database.index
NewParsedIndex= pd.to_datetime(PreviousIndex)
Database.index =NewParsedIndex              

Database.head()
Database.index.hour
Database.index.month
Database.index.dayofweek

#creating a new database with chosen datas
Temperature_Room=Database[["4:Temperature_Habitacion_Sensor"]]  #dependent variable
Temperature_External=Database[["22:Temperature_Exterior_Sensor"]] #independentvariable
Light_WestFacade=Database[["15:Meteo_Exterior_Sol_Oest"]] #independent variable

ChosenDatas=Temperature_Room.join([Temperature_External,Light_WestFacade])

ChosenDatas.rename(columns={'4:Temperature_Habitacion_Sensor':'Room Temperature[°C]','22:Temperature_Exterior_Sensor':'External Temperature[°C]',
                                    '15:Meteo_Exterior_Sol_Oest':'Light on west facade[Lux]',})

ChosenDatas.head()
ChosenDatas.describe()

#plotting of a part of each esaminated data of the new dataframe to see the correlation between the chosen values
#we have to see if there's a lag between the datas
ChosenDatas_SelectedPeriod=ChosenDatas["2012-03-15 00:00:00":"2012-03-18 23:00:00"]

plt.figure("Chosen datas in selected period")
plt.subplot(3,1,1)       #it includes the number of rows, the number of columns and the number of items
plt.plot(ChosenDatas_SelectedPeriod.iloc[:,0],color="purple")
plt.ylabel("Room Temp")
plt.subplot(3,1,2)
plt.plot(ChosenDatas_SelectedPeriod.iloc[:,1],color="orange")
plt.ylabel("External Temp")
plt.subplot(3,1,3)
plt.plot(ChosenDatas_SelectedPeriod.iloc[:,2],color="green")
plt.ylabel("Light west facade")


#creating two new dataframes to evaluate the lag and correct it
Temperature_lag=ChosenDatas[["4:Temperature_Habitacion_Sensor"]].join(ChosenDatas[["22:Temperature_Exterior_Sensor"]])
Lighting_lag=ChosenDatas[["4:Temperature_Habitacion_Sensor"]].join(ChosenDatas[["15:Meteo_Exterior_Sol_Oest"]])

def lag_column(df,column_name,lag_period=1):
    for i in range(1,lag_period+1,1):
        new_column_name = column_name+" -"+str(i)+"hr"
        df[new_column_name]=(df[column_name]).shift(i)
    return df
    
Temperature_lag=lag_column(Temperature_lag,"22:Temperature_Exterior_Sensor",24)
Temperature_lag_correlation = Temperature_lag.corr()
Temperature_lag_correlation.head()                   #shift of 6 hours

Lighting_lag=lag_column(Lighting_lag,"15:Meteo_Exterior_Sol_Oest",24)
Lighting_lag_correlation = Lighting_lag.corr()
Lighting_lag_correlation.head()                 #shift for 8 hours

#shifting
ChosenDatas_cleaned = ChosenDatas.dropna()
ChosenDatas_cleaned["22:Temperature_Exterior_Sensor"]=ChosenDatas_cleaned["22:Temperature_Exterior_Sensor"].shift(-6)      
ChosenDatas_cleaned["15:Meteo_Exterior_Sol_Oest"]=ChosenDatas_cleaned["15:Meteo_Exterior_Sol_Oest"].shift(-8)
ChosenDatas_cleaned.head()

#normalization
TempRoom_min=ChosenDatas_cleaned["4:Temperature_Habitacion_Sensor"].min()
TempRoom_max=ChosenDatas_cleaned["4:Temperature_Habitacion_Sensor"].max()
TempExt_min=ChosenDatas_cleaned["22:Temperature_Exterior_Sensor"].min()
TempExt_max=ChosenDatas_cleaned["22:Temperature_Exterior_Sensor"].max()
LightWest_min=ChosenDatas_cleaned["15:Meteo_Exterior_Sol_Oest"].min()
LightWest_max=ChosenDatas_cleaned["15:Meteo_Exterior_Sol_Oest"].max()
ChosenDatas_cleaned["TempRoom_normalized"]=(ChosenDatas_cleaned["4:Temperature_Habitacion_Sensor"]-TempRoom_min)/(TempRoom_max-TempRoom_min)
ChosenDatas_cleaned["TempExt_normalized"]=(ChosenDatas_cleaned["22:Temperature_Exterior_Sensor"]-TempExt_min)/(TempExt_max-TempExt_min)
ChosenDatas_cleaned["LightWest_normalized"]=(ChosenDatas_cleaned["15:Meteo_Exterior_Sol_Oest"]-LightWest_min)/(LightWest_max-LightWest_min)
ChosenDatas_cleaned.head()
ChosenDatas_cleaned.describe()

ChosenDatas_cleaned_SelectedPeriod=ChosenDatas_cleaned["2012-03-15 00:00:00":"2012-03-18 23:00:00"]

plt.figure("Normalized datas in selected period")       
plt.plot(ChosenDatas_cleaned_SelectedPeriod.iloc[:,3],color="pink")
plt.plot(ChosenDatas_cleaned_SelectedPeriod.iloc[:,4],color="green")
plt.plot(ChosenDatas_cleaned_SelectedPeriod.iloc[:,5],color="blue")

#creating new database only with normalized datas
TempRoom_Norm=ChosenDatas_cleaned[["TempRoom_normalized"]]
TempExt_Norm=ChosenDatas_cleaned[["TempExt_normalized"]]
LighWest_Norm=ChosenDatas_cleaned[["LightWest_normalized"]]
Database_Norm=TempRoom_Norm.join([TempExt_Norm,LighWest_Norm])

df=Database_Norm
lag_start=1
lag_end=24
lag_interval=1
column_name="TempRoom_normalized"

for i in range(lag_start, lag_end+1, lag_interval):
    new_column_name=column_name+" -"+str(i)+"hr"
    print new_column_name
    df[new_column_name]=df[column_name].shift(i)
    df.dropna(inplace=True)
df.head()

def lag_feature (df,column_name,lag_start, lag_end, lag_interval):
    for i in range(lag_start, lag_end+1, lag_interval):
        new_column_name=column_name+" -"+str(i)+"hr"
        print new_column_name
        df[new_column_name]=df[column_name].shift(i)
        df.dropna(inplace=True)
    return df 
    
Database_Lag=lag_column(Database_Norm,"TempRoom_normalized",24)

FinalDatabase=Database_Lag.dropna()
FinalDatabase.head()


#MACHINE LEARNING

Target=FinalDatabase["TempRoom_normalized"] #this is the dataframe with the dependent variable (room temperature)
Features=FinalDatabase.drop("TempRoom_normalized",axis=1)# this is the dataframe with the variables the indoor temperature depends on

#The function train_test_split splits arrays or matrices into random train and test subsets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(Features,Target,test_size = 0.2, random_state=41234)

#guardare cos'è la differenza tra x_test e x_train ecc

#linear regression
from sklearn import linear_model
linear_reg = linear_model.LinearRegression()

linear_reg.fit(X_train, Y_train)
predicted_linearReg_split = linear_reg.predict(X_test)

predicted_DF_linearReg_split=pd.DataFrame(predicted_linearReg_split,index=Y_test.index, columns=["RoomTemp_linearReg"])
predicted_DF_linearReg_split=predicted_DF_linearReg_split.join(Y_test)

#plot between actual values and predicted ones
predicted_DF_linearReg_split_SelectedPeriod=predicted_DF_linearReg_split["2012-03-15 00:00:00":"2012-03-18 23:00:00"]
predicted_DF_linearReg_split_SelectedPeriod.plot()

#error
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
MAE_linearReg_split= mean_absolute_error(predicted_linearReg_split,Y_test)
MSE_linearReg_split= mean_squared_error(predicted_linearReg_split,Y_test)
R2_linearReg_split= r2_score(predicted_linearReg_split,Y_test)

fig=plt.figure('Actual room temperature - Predicted room temperature')
plot=sns.regplot(x="TempRoom_normalized", y="RoomTemp_linearReg",data=predicted_DF_linearReg_split,line_kws={"lw":3,"alpha":0.5})
plot.set_xlabel("Actual room temperature")
plot.set_ylabel("Predicted room temperature")
regline = plot.get_lines()[0]
regline.set_color('r')

#cross validation
from sklearn.model_selection import cross_val_predict
predict_CV = cross_val_predict(linear_reg,Features,Target,cv=10)
predicted_DF_CV=pd.DataFrame(predict_CV,index=Target.index,columns=["RoomTemp_CV"])
predicted_DF_CV=predicted_DF_CV.join(Target)
predicted_DF_CV_SelectedPeriod=predicted_DF_CV["2012-03-15 00:00:00":"2012-03-18 23:00:00"]
predicted_DF_CV_SelectedPeriod.plot()

MAE_CV= mean_absolute_error(predict_CV,Target)
MSE_CV= mean_squared_error(predict_CV,Target)
R2_CV= r2_score(predict_CV,Target)

fig=plt.figure('Actual room temperature - Predicted room temperature')
plot=sns.regplot(x="TempRoom_normalized", y="RoomTemp_CV",data=predicted_DF_CV,line_kws={"lw":3,"alpha":0.5})
plot.set_xlabel("Actual room temperature")
plot.set_ylabel("Predicted room temperature")
regline = plot.get_lines()[0]
regline.set_color('r')

#random forest
from sklearn.ensemble import RandomForestRegressor
reg_RF = RandomForestRegressor()
predict_RF= cross_val_predict(reg_RF,Features,Target,cv=10)

predicted_DF_RF=pd.DataFrame(predict_RF,index=Target.index, columns=["RoomTemp_RF"])
predicted_DF_RF=predicted_DF_RF.join(Target)
predicted_DF_RF_SelectedPeriod=predicted_DF_RF["2012-03-15 00:00:00":"2012-03-18 23:00:00"]
predicted_DF_RF_SelectedPeriod.plot()

MAE_RF_CV= mean_absolute_error(predict_RF,Target)
MSE_RF_CV= mean_squared_error(predict_RF,Target)
R2_RF_CV = r2_score(predict_RF,Target)

fig=plt.figure('Actual room temperature - Predicted room temperature')
plot=sns.regplot(x="TempRoom_normalized", y="RoomTemp_RF",data=predicted_DF_RF,line_kws={"lw":3,"alpha":0.5})
plot.set_xlabel("Actual room temperature")
plot.set_ylabel("Predicted room temperature")
regline = plot.get_lines()[0]
regline.set_color('r')

