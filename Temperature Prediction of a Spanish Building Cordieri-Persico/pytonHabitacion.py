# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


DataFolderPath = "C:\Users\SilviaAnna\Desktop\progetto\spanish datadriven building\DataSet"
FileName= "NEW-DATA-1.T15.txt"
FilePath = DataFolderPath+"/"+FileName


DataFrame = pd.read_csv(FilePath, sep= " ", index_col=11)
DataFrame["period"] = DataFrame["1:Date"].map(str) + " " + DataFrame["2:Time"].map(str)
DataFrame.index = DataFrame["period"]


previousIndex= DataFrame.index 
ParsedIndex= pd.to_datetime(previousIndex)
DataFrame.index= ParsedIndex
DF_myChosenDates = DataFrame["2012-03-11 12:00:00 ":"2012-03-23 12:00:00 "]
DF_cleaned = DF_myChosenDates.dropna()
DF_cleaned.corr()

fig = plt.figure()
ax1= fig.add_subplot(3,1,1) 
ax2= fig.add_subplot(3,1,2)
ax3= fig.add_subplot(3,1,3)

DF_cleaned["4:Temperature_Habitacion_Sensor"].plot(ax=ax1,color= "b", legend= True)
DF_cleaned["9:Humedad_Habitacion_Sensor"].plot(ax=ax2,color= "r", legend= True)
DF_cleaned["11:Lighting_Habitacion_Sensor"].plot(ax=ax3,color= "g", legend= True)

DF_FinalDataSet= DF_cleaned.copy()
def lag_column(df,column_name,lag_period=1):
    for i in range(1,lag_period+1,1):
        new_column_name= column_name+"-"+str(i)+"hr"
        df[new_column_name]= df[column_name].shift(i)
  
    return df

DF_FinalDataSet= lag_column(DF_FinalDataSet,"4:Temperature_Habitacion_Sensor",24)

DF_FinalDataSet.dropna(inplace=True)

DF_FinalDataSet.index      #lo stiamo facendo sul dataframe completo! e non solo sulla parte selezionata
DF_FinalDataSet.index.hour
#DF_FinalDataSet.index.dayofweek   #dayofweek is already included in python --> 0=monday, 6=sunday
DF_FinalDataSet.index.month 
DF_FinalDataSet.index.week   

# So first let's add this ones! #INSERISCE QUESTE COLONNE NEL DATAFRAME!
DF_FinalDataSet["hour"]= DF_FinalDataSet.index.hour              
#DF_FinalDataSet["day_of_week"]= DF_FinalDataSet.index.dayofweek 
DF_FinalDataSet["month"]= DF_FinalDataSet.index.month
DF_FinalDataSet["week_of_year"]= DF_FinalDataSet.index.week   

def weekendDetector(day):
    weekendLabel=0
    if (day== 5or day==6):
        weekendLabel=1
    else:
        weekendLabel=0
    return weekendLabel   #cioè se ottengo 1 vuol dire che ho trovato sabato o domenica (weekend)
    
def dayDetector(hour):
    dayLabel=1
    if(hour<20 and hour>9):
        dayLabel=1 
    else:
        dayLabel=0
    return dayLabel      #se 1 vuol dire che è giorno (dalle 9 alle 20)

simpleVectorOfDays = [0,1,2,3,4,5,6]
weekendorNotVector = [ weekendDetector(thisDay) for thisDay in simpleVectorOfDays] #applico la funzione per i giorni della settimana! stampa 0 0 0 0 0 1 1

hoursOfDayVector= range(0,24,1)
dayOrNotVector= [dayDetector(thisHour) for thisHour in hoursOfDayVector] #dove c'è 1 significa che è giorno :)
    
DF_FinalDataSet["weekend"]= [weekendDetector(thisDay) for thisDay in DF_FinalDataSet.index.dayofweek]   
DF_FinalDataSet["day_night"]= [dayDetector(thisHour) for thisHour in DF_FinalDataSet.index.hour]  #adds more columns by appling the functions previously defined
DF_FinalDataSet.drop(DF_FinalDataSet.columns[[0,1,6,8,9,10,11,12,13,14,15,17,18,19,23]],axis=1,inplace=True);
DF_FinalDataSet.dropna(inplace=True)    

DF_FinalDataSet.corr()

DF_target = DF_FinalDataSet["4:Temperature_Habitacion_Sensor"]
DF_features = DF_FinalDataSet.drop("4:Temperature_Habitacion_Sensor", axis = 1) 
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(DF_features,DF_target,test_size=0.2,random_state=41234)

#now we test the model

from sklearn import linear_model
linear_reg = linear_model.LinearRegression()

#let's train linear module with training data (like fitting a line)

linear_reg.fit(X_train,Y_train)
predict_linearReg_split = linear_reg.predict(X_test) #makes the plot with train data (0.2 is percentage) and then tests it

#in a dataframe

Y_test.index
predict_DF_linearReg_split = pd.DataFrame(predict_linearReg_split,index = Y_test.index,columns = ["Temperature prediction"])
predict_DF_linearReg_split = predict_DF_linearReg_split.join(Y_test)
predict_DF_linearReg_split_ChosenDates = predict_DF_linearReg_split["2012-03-11":"2012-03-23"]
predict_DF_linearReg_split_ChosenDates.plot()
plt.xlabel("time")
plt.ylabel("temperature")
plt.title("Linear Regression Prediction")

plt.figure()
plt.scatter(predict_DF_linearReg_split_ChosenDates["4:Temperature_Habitacion_Sensor"],predict_DF_linearReg_split_ChosenDates["Temperature prediction"])
plt.xlabel("Actual Indoor Room Temperature in degC")
plt.ylabel("Predicted Indoor Room Temperature in degC")
plt.title("Scatter plot: Linear Regression")
plt.show() 

#let's find the metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mean_absolute_error_linearReg_split = mean_absolute_error(Y_test,predict_linearReg_split)
mean_squared_error_linearReg_split = mean_squared_error(Y_test,predict_linearReg_split)
R2_score_linearReg_split = r2_score(Y_test,predict_linearReg_split)

print "Linear regression: mean absolute error "+str(mean_absolute_error_linearReg_split)
print "mean squared error "+str(mean_squared_error_linearReg_split)
print "R2 "+str(R2_score_linearReg_split)

#now we use cross validation (implemented in sklearn)

from sklearn.model_selection import cross_val_predict

"""
we have already imported the linear_reg algorithm, no deed to do it again
from sklearn import linear_model, linear_reg = linear_model.LinearRegression
"""

predict_linearReg_CV = cross_val_predict(linear_reg,DF_features,DF_target,cv=10)
predict_DF_linearReg_CV = pd.DataFrame(predict_linearReg_CV,index = DF_target.index,columns = ["TempPredict_linearReg_split"])
predict_DF_linearReg_CV = predict_DF_linearReg_CV.join(DF_target)
predict_DF_linearReg_CV_ChosenDates = predict_DF_linearReg_CV["2012-03-11 12:00:00 ":"2012-03-23 12:00:00"]
predict_DF_linearReg_CV_ChosenDates.plot()
plt.xlabel("time")
plt.ylabel("temperature")
plt.title("Cross Validation Prediction") 

plt.figure()
plt.scatter(predict_DF_linearReg_CV_ChosenDates["4:Temperature_Habitacion_Sensor"],predict_DF_linearReg_CV_ChosenDates["TempPredict_linearReg_split"])
plt.xlabel("Actual Indoor Room Temperature in degC")
plt.ylabel("Predicted Indoor Room Temperature in degC")
plt.title("Scatter plot: Cross Validation")
plt.show()    

#let's find the metrics

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mean_absolute_error_linearReg_CV = mean_absolute_error(DF_target,predict_linearReg_CV)
mean_squared_error_linearReg_CV = mean_squared_error(DF_target,predict_linearReg_CV)
R2_score_linearReg_CV = r2_score(DF_target,predict_linearReg_CV)

print "Cross Validation: mean absolute error "+str(mean_absolute_error_linearReg_CV)
print "mean squared error "+str(mean_squared_error_linearReg_CV)
print "R2"+str(R2_score_linearReg_CV)

#let's try a more complex algorithm, we will use only cross validation, called randomForest

from sklearn.ensemble import RandomForestRegressor
reg_RF = RandomForestRegressor()
predict_RF_CV = cross_val_predict(reg_RF,DF_features,DF_target,cv=10)
predict_DF_RF_CV = pd.DataFrame(predict_RF_CV,index = DF_target.index,columns = ["TempPredict_linearReg_split"])
predict_DF_RF_CV = predict_DF_RF_CV.join(DF_target)
predict_DF_RF_CV_ChosenDates = predict_DF_RF_CV["2012-03-11":"2012-03-23"]
predict_DF_RF_CV_ChosenDates.plot()
plt.xlabel("time")
plt.ylabel("temperature")
plt.title("Random Forest Regressor Prediction")


plt.figure()
plt.scatter(predict_DF_linearReg_CV_ChosenDates["4:Temperature_Habitacion_Sensor"],predict_DF_linearReg_CV_ChosenDates["TempPredict_linearReg_split"])
plt.xlabel("Actual Indoor Room Temperature in degC")
plt.ylabel("Predicted Indoor Room Temperature in degC")
plt.title("Scatter plot: Random Forest Regression")
plt.show()


#let's find the metrics

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mean_absolute_error_linearReg_RF = mean_absolute_error(DF_target,predict_RF_CV)
mean_squared_error_linearReg_RF = mean_squared_error(DF_target,predict_RF_CV)
R2_score_linearReg_RF = r2_score(DF_target,predict_RF_CV)

print "Random Forest: mean absolute error"+str(mean_absolute_error_linearReg_RF)
print "mean squared error "+str(mean_squared_error_linearReg_RF)
print "R2 "+str(R2_score_linearReg_RF)