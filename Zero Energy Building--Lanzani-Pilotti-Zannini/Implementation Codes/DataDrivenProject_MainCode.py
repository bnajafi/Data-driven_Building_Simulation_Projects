# -*- coding: utf-8 -*-

# Data Driven Project

import pandas as pd
import matplotlib.pyplot as plt


DataFolderPath="C:/Users/Lorenzo/Dropbox_Polimi_PC/Dropbox/Behzad Project" 
DataFileName="DataSet_1.txt"
DataSetFilePath=DataFolderPath+"/"+DataFileName

DF_DataSetSource=pd.read_csv(DataSetFilePath, sep=" ")
DF_DataSetSource.head()


time=DF_DataSetSource["2:Time"]
date=DF_DataSetSource["1:Date"]
New_column=date+" "+time
DF_joined=DF_DataSetSource.join([New_column])
DF_joined.head()

DF_DataSetSourceModified=DF_joined
DF_DataSetSourceModified.index=DF_joined[0]
DF_DataSetSourceModified.head()
DF_DataSetSourceModified.index.names=["DateTime"]
DF_DataSetSourceModified.head()


#salvo in un file csv il data frame corretto nel caso si volesse evitare questa prima parte di codice
#DF_DataSetSourceModified.to_csv("DATASETModified_1.csv", sep=";", decimal=",")

ParseIndex=pd.to_datetime(DF_DataSetSourceModified.index, dayfirst=True)
DF_DataSetSourceModified.index=ParseIndex

ChosenDate=DF_DataSetSourceModified["2012-03-17 00:45:00":"2012-03-19 00:45:00"]
ChosenDate.head()
ChosenDate.describe()

ChosenDate["4:Temperature_Habitacion_Sensor"].plot()
plt.show()

DF_DataSetSourceModified.corr()["4:Temperature_Habitacion_Sensor"]

#INSERIRE DATA CLEANING ---> pyranometer
booleanVector=(DF_DataSetSourceModified["18:Meteo_Exterior_Piranometro"]<0)
DF_DataSetSourceModified["18:Meteo_Exterior_Piranometro"][booleanVector]=0
#Converto i valori di lux su ogni facciata in W/m2 in modo da poterli confrontare con l'irradianza globale.
DF_DataSetSourceModified["15:Meteo_Exterior_Sol_Oest"]=DF_DataSetSourceModified["15:Meteo_Exterior_Sol_Oest"]/685
DF_DataSetSourceModified["16:Meteo_Exterior_Sol_Est"]=DF_DataSetSourceModified["16:Meteo_Exterior_Sol_Est"]/685
DF_DataSetSourceModified["17:Meteo_Exterior_Sol_Sud"]=DF_DataSetSourceModified["17:Meteo_Exterior_Sol_Sud"]/685
DF_DataSetSourceModified.head()
ChosenDate=DF_DataSetSourceModified["2012-03-17 00:45:00":"2012-03-19 00:45:00"]


#plotting togheter temperature, exterior irradiance and exterior temperature
fig=plt.figure()
ax1=fig.add_subplot(211)
ax2=fig.add_subplot(211)
ax3=fig.add_subplot(212)
ax4=fig.add_subplot(212)
ax5=fig.add_subplot(212)
ax6=fig.add_subplot(212)

ChosenDate["4:Temperature_Habitacion_Sensor"].plot(ax=ax1, color="b", legend=True)
ChosenDate["22:Temperature_Exterior_Sensor"].plot(ax=ax2, color="g", legend=True)
ChosenDate["18:Meteo_Exterior_Piranometro"].plot(ax=ax3, color="r", legend=True)
ChosenDate["15:Meteo_Exterior_Sol_Oest"].plot(ax=ax4, color="y", legend=True)
ChosenDate["16:Meteo_Exterior_Sol_Est"].plot(ax=ax5, color="g", legend=True)
ChosenDate["17:Meteo_Exterior_Sol_Sud"].plot(ax=ax6, color="b", legend=True)


#FINAL DATASET
DF_FinalDataSet=pd.DataFrame([], index=DF_DataSetSourceModified.index)
DF_FinalDataSet["indoor temperature"]=DF_DataSetSourceModified["4:Temperature_Habitacion_Sensor"]
DF_FinalDataSet["outdoor temperature"]=DF_DataSetSourceModified["22:Temperature_Exterior_Sensor"]
DF_FinalDataSet["outdoor pyranometer"]=DF_DataSetSourceModified["18:Meteo_Exterior_Piranometro"]
DF_FinalDataSet.head()

# Trovo la correlazione "manualmente"

def lag_column(df, column_name, lag_period=1, initial_time=1):
    """this function is useful to add one or more columns of lagged values to an existing dataframe.
    It recieves in input the dataframe, the dataframe column that has to be lagged, the lag period
    and the starting lag period. It returns the origianl dataframe plus lagged columns """
    
    for i in range(initial_time, lag_period+1, 1):
        new_column=column_name+" -"+str(i*15)+"min"
        df[new_column]=df[column_name].shift(i)
    return df

DF_FinalDataSet=lag_column(DF_FinalDataSet, "outdoor temperature", lag_period=9, initial_time=4)
DF_FinalDataSet=lag_column(DF_FinalDataSet, "outdoor pyranometer", lag_period=22, initial_time=16)
DF_FinalDataSet=lag_column(DF_FinalDataSet, "indoor temperature", lag_period=24*4, initial_time=4) # correlazione della temperatura stessa coi valori assunti nelle ore precedenti.
DF_FinalDataSet.dropna(inplace=True)
DF_FinalDataSet.head()

DF_FinalDataSet.corr()["indoor temperature"]

# correlazione temperature esterna -1hr circa (0.9) e buona anche con irradianza (0.6) con circa 5 ore di ritardo
#Temperatura interna laggata di un'ora così che sia possibile fare previsioni a partire dall'ora precedente.


# Regressione lineare
DF_target = DF_FinalDataSet["indoor temperature"]
DF_features = DF_FinalDataSet.drop(["indoor temperature"],axis=1)
DF_features = DF_features.drop(["outdoor temperature"],axis=1)
DF_features = DF_features.drop(["outdoor pyranometer"],axis=1)
DF_features.head()

# test_size=0.5 perchè servono almeno un paio di settimane all'algoritmo per "imparare" a predire 
# l'evoluzione della temperatura (dati campionati su un totale di circa 28 giorni)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(DF_features, DF_target, test_size=0.2)

from sklearn import linear_model
linear_reg=linear_model.LinearRegression()

linear_reg.fit(X_train, Y_train)
predict_linearReg_split=linear_reg.predict(X_test)
predict_DF_linearReg_split=pd.DataFrame(predict_linearReg_split, index=Y_test.index, columns=["temeperature_Pred_linearReg_split"])
predict_DF_linearReg_split=predict_DF_linearReg_split.join(Y_test)
predict_DF_linearReg_split=predict_DF_linearReg_split.join(DF_DataSetSourceModified["22:Temperature_Exterior_Sensor"])
predict_DF_linearReg_split.head()

predict_DF_linearReg_split_DF_ChosenDates=predict_DF_linearReg_split["2012-03-17":"2012-03-19"]
predict_DF_linearReg_split_DF_ChosenDates.plot()


from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
mean_absolute_error_linearReg_split=mean_absolute_error(Y_test, predict_linearReg_split)
mean_squared_error_linearReg_split=mean_squared_error(Y_test, predict_linearReg_split)
R2_score_linearReg_split=r2_score(Y_test, predict_linearReg_split)

# altri modelli per trovare la correlazione
from sklearn.model_selection import cross_val_predict
predict_linearReg_CV=cross_val_predict(linear_reg, DF_features, DF_target, cv=10)

predict_DF_linearReg_CV=pd.DataFrame(predict_linearReg_CV, index=DF_target.index, columns=["TempPred_linearReg_CV"])
predict_DF_linearReg_CV=predict_DF_linearReg_CV.join(DF_target)
predict_DF_linearReg_CV=predict_DF_linearReg_CV.join(DF_DataSetSourceModified["22:Temperature_Exterior_Sensor"])
predict_DF_linearReg_CV_DF_ChosenDates=predict_DF_linearReg_CV["2012-03-17":"2012-03-19"]
predict_DF_linearReg_CV_DF_ChosenDates.plot()

mean_absolute_error_linearReg_CV=mean_absolute_error(DF_target, predict_linearReg_CV)
mean_squared_error_linearReg_CV=mean_squared_error(DF_target, predict_linearReg_CV)
R2_score_linearReg_CV=r2_score(DF_target, predict_linearReg_CV)

# Randon Forest
from sklearn.ensemble import RandomForestRegressor
 
reg_RF=RandomForestRegressor()
predict_RF_CV=cross_val_predict(reg_RF, DF_features, DF_target, cv=10)

predict_DF_RF_CV=pd.DataFrame(predict_RF_CV, index=DF_target.index, columns=["TempPred_RF_CV"])
predict_DF_RF_CV=predict_DF_RF_CV.join(DF_target)
predict_DF_RF_CV=predict_DF_RF_CV.join(DF_DataSetSourceModified["22:Temperature_Exterior_Sensor"])
predict_DF_RF_CV_ChosenDates=predict_DF_RF_CV["2012-03-17":"2012-03-19"]
predict_DF_RF_CV_ChosenDates.plot()

mean_absolute_error_RF_CV=mean_absolute_error(DF_target, predict_RF_CV)
mean_squared_error_RF_CV=mean_squared_error(DF_target, predict_RF_CV)
R2_score_RF_CV=r2_score(DF_target, predict_RF_CV)




#   FINAL DATA SET SENZA CONSIDERARE L'IRRADIANZA GLOBALE MA SOLO QUELLA SULLE FACCIATE ESTERNE
DF_FinalDataSet=pd.DataFrame([], index=DF_DataSetSourceModified.index)
DF_FinalDataSet["indoor temperature"]=DF_DataSetSourceModified["4:Temperature_Habitacion_Sensor"]
DF_FinalDataSet["outdoor temperature"]=DF_DataSetSourceModified["22:Temperature_Exterior_Sensor"]
DF_FinalDataSet["irradiance west facade"]=DF_DataSetSourceModified["15:Meteo_Exterior_Sol_Oest"]
DF_FinalDataSet["irradiance est facade"]=DF_DataSetSourceModified["16:Meteo_Exterior_Sol_Est"]
DF_FinalDataSet["irradiance sud facade"]=DF_DataSetSourceModified["17:Meteo_Exterior_Sol_Sud"]
DF_FinalDataSet.head()

DF_FinalDataSet=lag_column(DF_FinalDataSet, "outdoor temperature", lag_period=9, initial_time=4)

DF_FinalDataSet=lag_column(DF_FinalDataSet, "irradiance west facade", lag_period=22, initial_time=4)
DF_FinalDataSet=lag_column(DF_FinalDataSet, "irradiance est facade", lag_period=22, initial_time=4)
DF_FinalDataSet=lag_column(DF_FinalDataSet, "irradiance sud facade", lag_period=22, initial_time=4)
DF_FinalDataSet=lag_column(DF_FinalDataSet, "indoor temperature", lag_period=24*4, initial_time=4) # correlazione della temperatura stessa coi valori assunti nelle ore precedenti.
DF_FinalDataSet.dropna(inplace=True)
DF_FinalDataSet.head()

DF_FinalDataSet.corr()["indoor temperature"]

DF_target = DF_FinalDataSet["indoor temperature"]
DF_features = DF_FinalDataSet.drop(["indoor temperature"],axis=1)
DF_features = DF_features.drop(["outdoor temperature"],axis=1)
DF_features = DF_features.drop(["irradiance west facade"],axis=1)
DF_features = DF_features.drop(["irradiance est facade"],axis=1)
DF_features = DF_features.drop(["irradiance sud facade"],axis=1)
DF_features.head()


X_train, X_test, Y_train, Y_test = train_test_split(DF_features, DF_target, test_size=0.2)

linear_reg.fit(X_train, Y_train)
predict_linearReg_split=linear_reg.predict(X_test)
predict_DF_linearReg_split=pd.DataFrame(predict_linearReg_split, index=Y_test.index, columns=["temeperature_Pred_linearReg_split"])
predict_DF_linearReg_split=predict_DF_linearReg_split.join(Y_test)
predict_DF_linearReg_split.head()

predict_DF_linearReg_split_DF_ChosenDates=predict_DF_linearReg_split["2012-03-17":"2012-03-19"]
predict_DF_linearReg_split_DF_ChosenDates.plot()

mean_absolute_error_linearReg_split=mean_absolute_error(Y_test, predict_linearReg_split)
mean_squared_error_linearReg_split=mean_squared_error(Y_test, predict_linearReg_split)
R2_score_linearReg_split=r2_score(Y_test, predict_linearReg_split)
