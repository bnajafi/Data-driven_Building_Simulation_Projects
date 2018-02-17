import pandas as pd
import matplotlib.pyplot as plt


DataFolderPath="C:/Users/Lorenzo/Dropbox_Polimi_PC/Dropbox/Behzad Project" 
DataFileName="DataSet_2.txt"
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

ParseIndex=pd.to_datetime(DF_DataSetSourceModified.index, dayfirst=True)
DF_DataSetSourceModified.index=ParseIndex

ChosenDate=DF_DataSetSourceModified["2012-04-18 00:00:00":"2012-04-20 23:45:00"]
ChosenDate.head()
ChosenDate.describe()

ChosenDate["3:Temperature_Comedor_Sensor"].plot()

#INSERIRE DATA CLEANING ---> pyranometer
booleanVector=(DF_DataSetSourceModified["16:Meteo_Exterior_Sol_Est"]<0)
DF_DataSetSourceModified["16:Meteo_Exterior_Sol_Est"][booleanVector]=0

#plotting togheter temperature, exterior irradiance and exterior temperature
fig=plt.figure()
ax1=fig.add_subplot(211)
ax2=fig.add_subplot(211)
ax3=fig.add_subplot(212)


ChosenDate["3:Temperature_Comedor_Sensor"].plot(ax=ax1, color="b", legend=True)
ChosenDate["22:Temperature_Exterior_Sensor"].plot(ax=ax2, color="g", legend=True)
ChosenDate["16:Meteo_Exterior_Sol_Est"].plot(ax=ax3, color="r", legend=True)

#FINAL DATASET
DF_FinalDataSet=pd.DataFrame([], index=DF_DataSetSourceModified.index)
DF_FinalDataSet["dinning room temperature"]=DF_DataSetSourceModified["3:Temperature_Comedor_Sensor"]
DF_FinalDataSet["outdoor temperature"]=DF_DataSetSourceModified["22:Temperature_Exterior_Sensor"]
DF_FinalDataSet["pyranometer est facade"]=DF_DataSetSourceModified["16:Meteo_Exterior_Sol_Est"]
DF_FinalDataSet["pyranometer sud facade"]=DF_DataSetSourceModified["17:Meteo_Exterior_Sol_Sud"]
DF_FinalDataSet.head()

def lag_column(df, column_name, lag_period=1, initial_time=1):
    for i in range(initial_time, lag_period+1, 1):
        new_column=column_name+" -"+str(i*15)+"min"
        df[new_column]=df[column_name].shift(i)
    return df

DF_FinalDataSet=lag_column(DF_FinalDataSet, "outdoor temperature", lag_period=9, initial_time=4)
DF_FinalDataSet=lag_column(DF_FinalDataSet, "pyranometer est facade", lag_period=38, initial_time=22)
DF_FinalDataSet=lag_column(DF_FinalDataSet, "pyranometer sud facade", lag_period=28, initial_time=14)
DF_FinalDataSet=lag_column(DF_FinalDataSet, "dinning room temperature", lag_period=8, initial_time=4)
DF_FinalDataSet.dropna(inplace=True)
DF_FinalDataSet.head()

DF_FinalDataSet.corr()["dinning room temperature"]

DF_target = DF_FinalDataSet["dinning room temperature"]
DF_features = DF_FinalDataSet.drop(["dinning room temperature"],axis=1)
DF_features = DF_features.drop(["outdoor temperature"],axis=1)
DF_features = DF_features.drop(["pyranometer sud facade"],axis=1)
DF_features = DF_features.drop(["pyranometer est facade"],axis=1)
DF_features.head()

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(DF_features, DF_target, test_size=0.5)

from sklearn import linear_model
linear_reg=linear_model.LinearRegression()

linear_reg.fit(X_train, Y_train)
predict_linearReg_split=linear_reg.predict(X_test)
predict_DF_linearReg_split=pd.DataFrame(predict_linearReg_split, index=Y_test.index, columns=["temeperature_Pred_linearReg_split"])
predict_DF_linearReg_split=predict_DF_linearReg_split.join(Y_test)
predict_DF_linearReg_split.head()

predict_DF_linearReg_split_DF_ChosenDates=predict_DF_linearReg_split["2012-04-18":"2012-04-19"]
predict_DF_linearReg_split_DF_ChosenDates.plot()


from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
mean_absolute_error_linearReg_split=mean_absolute_error(Y_test, predict_linearReg_split)
mean_squared_error_linearReg_split=mean_squared_error(Y_test, predict_linearReg_split)
R2_score_linearReg_split=r2_score(Y_test, predict_linearReg_split)
