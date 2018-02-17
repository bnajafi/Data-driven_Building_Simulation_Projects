import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
os.chdir("C:/Users/Giorgio/Desktop/Buildings python project/Data-driven simulation_Asnaashari - Casalicchio - Moretti")
os.getcwd()

import useful_functions as uf

DataFolderPath = "C:/Users/Giorgio/Desktop/Buildings python project/Data-driven simulation_Asnaashari - Casalicchio - Moretti/data sets"
FileName = "building5retail.csv"
FilePath = DataFolderPath + "/" + FileName

DF_start = pd.read_csv(FilePath, sep = ",", index_col = 0)

PreviousIndex = DF_start.index
ParsedIndex = pd.to_datetime(PreviousIndex)   
DF_start.index = ParsedIndex

DF_start["OAT (degC)"] = (DF_start["OAT (F)"] - 32)/1.8

DF_consumption = DF_start.drop(["OAT (F)"], axis = 1)

DF_chosenDates = DF_consumption["2010-07-19":"2010-07-25"] 
plt.subplot(211)
DF_chosenDates["Power (kW)"].plot(color = "b", grid = True)
plt.xlabel('')
plt.ylabel("Power conusmption [kW]")
plt.subplot(212)
DF_chosenDates["OAT (degC)"].plot(color = "g", grid = True)
plt.ylabel("Outdoor temperature [degC]")
plt.suptitle("POWER CONSUMPTION AND TEMPERATURE FOR SELECTED DAYS")
plt.show()

DF_consumption = uf.lag_column15(DF_consumption, "Power (kW)", 4*24)
DF_consumption = uf.lag_column15(DF_consumption, "OAT (degC)", 4*6)

DF_final = DF_consumption.dropna()
# DF_consumption.head(10)

DF_final["hour"] = DF_final.index.hour
DF_final["day"] = DF_final.index.dayofweek   
DF_final["month"] = DF_final.index.month
DF_final['weekend'] = [uf.weekendDetector(thisDay) for thisDay in DF_final.index.dayofweek]
DF_final['day_night'] = [uf.dayDetector(thisHour) for thisHour in DF_final.index.hour]
# DF_final.head(10)

DF_target = DF_final[["Power (kW)"]]
DF_features = DF_final.drop("Power (kW)", axis = 1)   

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(DF_features, DF_target, test_size = 0.2, random_state = 41234)

from sklearn import linear_model
linear_reg = linear_model.LinearRegression()
linear_reg.fit(X_train, y_train)
LinearPredictions_complete = linear_reg.predict(DF_features)

LinearPredictions_complete_DF = pd.DataFrame(LinearPredictions_complete, index = DF_final.index, columns = ["Power predicted (kW)"])
LinearPredictions_final = DF_target.join(LinearPredictions_complete_DF)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
linear_R2_score = r2_score(DF_target, LinearPredictions_final["Power predicted (kW)"])
linear_mean_absolute_error = mean_absolute_error(DF_target,LinearPredictions_final["Power predicted (kW)"])
linear_mean_squared_error = mean_squared_error(DF_target,LinearPredictions_final["Power predicted (kW)"])
linear_coeff_variation = np.sqrt(linear_mean_squared_error)/DF_target["Power (kW)"].mean()

print "R2 = " + str(linear_R2_score)
print "MAE = " + str(linear_mean_absolute_error)
print "MSE = " + str(linear_mean_squared_error)
print "CV = " + str(linear_coeff_variation)

from sklearn.model_selection import cross_val_predict
CV_LinearPredictions = cross_val_predict(linear_reg, DF_features, DF_target, cv = 10)
CV_LinearPredictions_DF = pd.DataFrame(CV_LinearPredictions, index = DF_target.index, columns=["Power predicted CV (kW)"])
final_CV_LinearPredictions_DF = DF_target.join(CV_LinearPredictions_DF)

CV_R2_score = r2_score(final_CV_LinearPredictions_DF["Power (kW)"],final_CV_LinearPredictions_DF["Power predicted CV (kW)"])
CV_mean_absolute_error = mean_absolute_error(final_CV_LinearPredictions_DF["Power (kW)"], final_CV_LinearPredictions_DF["Power predicted CV (kW)"])
CV_mean_squared_error = mean_squared_error(final_CV_LinearPredictions_DF["Power (kW)"], final_CV_LinearPredictions_DF["Power predicted CV (kW)"])
CV_coeff_variation = np.sqrt(CV_mean_squared_error)/final_CV_LinearPredictions_DF["Power (kW)"].mean()

print "R2 = " + str(CV_R2_score)
print "MAE = " + str(CV_mean_absolute_error)
print "MSE = " + str(CV_mean_squared_error)
print "CV = " + str(CV_coeff_variation)

compare = LinearPredictions_final.join(CV_LinearPredictions_DF)

compare['2010-05-017':'2010-05-18'].plot(colors= ("g","r","b"))
plt.title("LINEAR REGRESSION MODEL WITH AND WITHOUT CROSS VALIDATION")
plt.ylabel("Power consumption [kW]")
plt.grid()
plt.legend(('actual power consumption', 'linear regression model predictions',  'linear regression with cross validation predictions'))
plt.show()