# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


#Importing the external files

WhereToSaveFiles= r"C:\Users\Giulia\Desktop\Comedor Temperature Prediction--Blanco-Cordoba-Sora"
os.chdir(WhereToSaveFiles)
ExternalFilesFolder = r"C:\Users\Giulia\Desktop\Comedor Temperature Prediction--Blanco-Cordoba-Sora"
DataFileName="SpanishDataSet.xlsx"
path_DataFileName=os.path.join(ExternalFilesFolder,DataFileName)

DF_DataFile=pd.read_excel(path_DataFileName,index_col=0,header=1)
DF_DataFile.columns


#We transform the data in the file from 'undefined objects' to 'time and date data'

PreviousIndex=DF_DataFile.index
NewParsedIndex=pd.to_datetime(PreviousIndex)
DF_DataFile.index=NewParsedIndex

DF_DataFile.head(4)
DF_DataFile.tail(4)

DF_DataFile_cleaned=DF_DataFile.dropna()
DF_DataFile_cleaned.corr()


#We extract/select the data we are interested in:

DF_TemperatureInsideComedor=DF_DataFile["3:Temperature_Comedor_Sensor"]["2012-03-13 11:45:00":"2012-04-11 06:30:00"] #all the possible data


#We plot the TemperatureInsideComedor measured versus time during the period we are considering.

plt.figure()
DF_TemperatureInsideComedor.plot()
plt.xlabel("Time")
plt.ylabel("Temperature measured in the Comedor")
plt.show()


#Now we import the data in order to create our new dataframe

DF_TemperatureInsideComedor=DF_DataFile[["3:Temperature_Comedor_Sensor"]]
DF_TemperatureInsideComedor.head(10)

DF_TemperatureInsideHabitacion=DF_DataFile[["4:Temperature_Habitacion_Sensor"]]
DF_TemperatureInsideHabitacion.head(10)

DF_WeatherTemperature=DF_DataFile[["5:Weather_Temperature"]]
DF_WeatherTemperature.head(10)

DF_TemperatureExterior=DF_DataFile[["22:Temperature_Exterior_Sensor"]]
DF_TemperatureExterior.head(10)


DF_DataComedor_joined=DF_TemperatureInsideComedor.join([DF_TemperatureInsideHabitacion,DF_WeatherTemperature,DF_TemperatureExterior])
DF_DataComedor_joined.columns=["TempInsCom","TempInsHab", "WeaTemp", "TempExt"]
DF_DataComedor_joined.to_excel("DataComedor_joinedOk.xlsx")


#Doing the lag

DF_modLag=DF_DataComedor_joined.copy()
DF_modLag.dropna(inplace=True)
df= DF_modLag


def lag_feature(df,column_name, lag_start,lag_end,lag_interval):
    """This function takes add inputs the name of the file, the column name on which we want to do the shift,
        the first step, the last step, and the magnitude of the single step. It returns the file with each columns shifted as requested"""
    for i in range(lag_start,lag_end+1, lag_interval): 
        new_column_name= column_name+ "-15X" + str(i) + "min"  
        print new_column_name
        df[new_column_name]=df[column_name].shift(i)
        df.dropna(inplace=True)
    return df


DF_modLag= lag_feature(df, "TempInsHab",1,5,1)
DF_modLag.head()
DF_modLag= lag_feature(df, "WeaTemp",1,5,1)
DF_modLag.head()
DF_modLag= lag_feature(df, "TempExt",1,5,1)
DF_modLag.head()

DF_modLag.corr()


DF_modLag.to_excel("DF_FinalDataSetokok.xlsx")
FinalDataSetToPredict="DF_FinalDataSetokok.xlsx"
path_DataToPredict=os.path.join(ExternalFilesFolder,FinalDataSetToPredict)
DF_FinalDataSetTopredict=pd.read_excel(path_DataToPredict,index_col=0,header=0)


#First Normalizing, before doing my machine learning

DF_modLagChosenDatas=DF_FinalDataSetTopredict["2012-03-13":"2012-04-11"]
DF_modLagChosenDatas.head()
DF_modLagChosenDatas.dropna()


def normalizing(DF_modLagChosenDatas):
    """This function normalize the data.It calculates the minimum and the maximum values for each columns, then it does the normalization"""
    maximum=DF_modLagChosenDatas.max()
    minimum=DF_modLagChosenDatas.min()
    return (DF_modLagChosenDatas-minimum)/(maximum-minimum)
    
#Now after making my Final Data Set, we are going to start our Machine Learning

DF_target=DF_modLagChosenDatas["TempInsCom"]
DF_features=DF_modLagChosenDatas.drop("TempInsCom",axis=1)

#Normalizing

DF_FinalDataSet_ChosenDatas_norm = normalizing(DF_modLagChosenDatas)
DF_FinalDataSet_ChosenDatas_norm.corr()
DF_target_norm = DF_FinalDataSet_ChosenDatas_norm["TempInsCom"]
DF_features_norm = DF_FinalDataSet_ChosenDatas_norm.drop("TempInsCom",axis=1)

#Now we divide the data in two groups: the one on which we do the regression and the one on which we test the regression

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(DF_features,DF_target,test_size=0.2,random_state=41234)


from sklearn import linear_model
linear_reg= linear_model.LinearRegression()
linear_reg.fit(X_train,Y_train)
predicted_linearreg_split=linear_reg.predict(X_test)
predicted_DF_linearreg_split=pd.DataFrame(predicted_linearreg_split, index=Y_test.index, columns=["TempInsCom_predicted_linear_reg"])
predicted_DF_linearreg_split=predicted_DF_linearreg_split.join(Y_test)

#Graph for comparison of the real data and the one predicted from the Linear Regression model

predicted_DF_linearreg_split_alldata=predicted_DF_linearreg_split["2012-03-13":"2012-04-11"]
predicted_DF_linearreg_split_alldata.plot()

#Now we want to estimate how accurate are our predictions. So, we use the accuracy metrics: MAE, MSE and R2

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
MAE_linear_reg_split=mean_absolute_error(predicted_linearreg_split,Y_test)
MSE_linear_reg_split=mean_squared_error(predicted_linearreg_split,Y_test)
R2_linear_reg_split=r2_score(predicted_linearreg_split,Y_test)

#Let's make other prediction with Cross-Validation model with 40 subsets

from sklearn.model_selection import cross_val_predict
predicted_linearReg_CV=cross_val_predict(linear_reg,DF_features,DF_target,cv=40)
predicted_DF_linearReg_CV=pd.DataFrame(predicted_linearReg_CV,index=DF_target.index,columns=["TempInsCom_predicted_linear_reg_CV"])
predicted_DF_linearReg_CV=predicted_DF_linearReg_CV.join(DF_target)

#Graph for comparison of the real data and the one predicted from the Cross-Validation Model

predicted_DF_linearReg_CV_alldata=predicted_DF_linearReg_CV["2012-03-13":"2012-04-11"]
predicted_DF_linearReg_CV_alldata.plot()

#We use again the accuracy metrics: MAE, MSE and R2

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
MAE_linearReg_CV=mean_absolute_error(predicted_linearReg_CV,DF_target)
MSE_linearReg_CV=mean_squared_error(predicted_linearReg_CV,DF_target)
R2_linearReg_CV=r2_score(predicted_linearReg_CV,DF_target)



