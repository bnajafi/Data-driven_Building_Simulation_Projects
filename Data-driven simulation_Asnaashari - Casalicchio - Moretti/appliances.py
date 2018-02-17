import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
os.chdir("C:/Users/Giorgio/Desktop/Buildings python project/Data-driven simulation_Asnaashari - Casalicchio - Moretti")
os.getcwd()

import useful_functions as uf

##### BUILDING THE DATA SET #####

""" In the first part of the code we read the data set, we plot some data we need, in order to see them better, 
    and we create another data set with all the features we need (shifted data and other informations like day, month, etc)
    for the nex machine learning part. """
    
##### Select the file using its path #####

DataFolderPath = "C:/Users/Giorgio/Desktop/Buildings python project/Data-driven simulation_Asnaashari - Casalicchio - Moretti/data sets"
FileName = "energydata_complete.csv"
FilePath = DataFolderPath + "/" + FileName

##### Read the Data Set in a Pandas data frame #####

DF_complete = pd.read_csv(FilePath, sep = ",", index_col = 0)
# DF_complete.head(10)

##### Conversion of the indices into date-time objects #####

PreviousIndex = DF_complete.index
ParsedIndex = pd.to_datetime(PreviousIndex)   
DF_complete.index = ParsedIndex

##### Delete all rows with Nan #####

DF_cleaned = DF_complete.dropna()

##### Plots #####

# Energy consumption for the entire period in the data set and for one week #

DF_chosenDates = DF_cleaned["2016-05-02":"2016-05-08"] # This is for a week, monday to sunday

plt.figure()
plt.subplot(211)
DF_cleaned["Appliances"].plot(color = "b", grid = True)
plt.ylabel("Energy conusmption [Wh]")
plt.subplot(212)
DF_chosenDates["Appliances"].plot(color = "b", grid = True)
plt.ylabel("Energy conusmption [Wh]")
plt.suptitle("Appliances consumption")
plt.show()

# Temperature #

plt.figure()
DF_cleaned["T_out"].plot(color = "g", grid = True)
plt.ylabel("Temperature [deg C]")
plt.title("Outdoor temperature")
plt.show()

##### Make a data set with only the data considered #####

DF_energyCons = DF_cleaned[["Appliances"]]
DF_temperature = DF_cleaned[["T_out"]]
DF_joined = DF_energyCons.join([DF_temperature])
# DF_joined.head(10)

##### Building the final data set with the columns shifted #####

DF_finalDataSet = uf.lag_column10(DF_joined, "Appliances", 6*24)
DF_finalDataSet = uf.lag_column10(DF_joined, "T_out", 6*6)

DF_finalDataSet_cleaned = DF_finalDataSet.dropna()

##### Add columns with the hour, day and month to the data set #####

DF_finalDataSet_cleaned["hour"] = DF_finalDataSet_cleaned.index.hour
DF_finalDataSet_cleaned["day"] = DF_finalDataSet_cleaned.index.dayofweek   
DF_finalDataSet_cleaned["month"] = DF_finalDataSet_cleaned.index.month

##### Add columns with working/weekend and day/night #####

DF_finalDataSet_cleaned['weekend'] = [uf.weekendDetector(thisDay) for thisDay in DF_finalDataSet_cleaned.index.dayofweek]
DF_finalDataSet_cleaned['day_night'] = [uf.dayDetector(thisHour) for thisHour in DF_finalDataSet_cleaned.index.hour]

# DF_finalDataSet_cleaned.head(5) 

##### The final data set cleaned contains columns with appliances shifted until one day before, temperature shifted until 6 hours before, columns with hour, day, month, weekend/workday, day/night #####


##### MACHINE LEARNING #####

""" This is the machine learning part of the code. We take our data set built in the first part and
    we use it to predict the target we set  (in this case the energy consumption by the appliances)
    using as imputs the other features we have (in this case the consumption in the previous 24 hours, 
    the temperature in the previous 6  hours, the hour itself, the day, the month, and two indices that
    indicate if the day is a weekend day or not, and if the hour is in the daylight or in the night) """

##### Set target and features #####

DF_target = DF_finalDataSet_cleaned[["Appliances"]]
DF_features = DF_finalDataSet_cleaned.drop("Appliances", axis = 1)   

##### Import module sklearn for the training and testing #####

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(DF_features, DF_target, test_size = 0.2, random_state = 41234)

##### Predictions with linear model #####

from sklearn import linear_model
linear_reg = linear_model.LinearRegression()
linear_reg.fit(X_train, y_train)
# LinearPredictions_test = linear_reg.predict(X_test)
LinearPredictions_complete = linear_reg.predict(DF_features)


##### Create a data frame with the predicted values (for the test data) #####

# LinearPredictions_test_DF = pd.DataFrame(LinearPredictions_test, index = y_test.index, columns = ["Appliances_predicted (only for test data)"])
LinearPredictions_complete_DF = pd.DataFrame(LinearPredictions_complete, index = DF_finalDataSet_cleaned.index, columns = ["Appliances_predicted"])

# LinearPredictions_test_final = LinearPredictions_test_DF.join(y_test)
LinearPredictions_final = DF_target.join(LinearPredictions_complete_DF)


##### Chose only some data and plot them to see the comparison with the actual values #####
"""
predictions_ChosenDates = LinearPredictions_final['2016-05-05']

predictions_ChosenDates.plot()
plt.ylabel('Energy consumption [Wh]')
plt.grid()
plt.title("LINEAR REGRESSION PREDICTIONS")
plt.show()
"""
##### statistics #####

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
linear_R2_score = r2_score(DF_target, LinearPredictions_final["Appliances_predicted"])
linear_mean_absolute_error = mean_absolute_error(DF_target,LinearPredictions_final["Appliances_predicted"])
linear_mean_squared_error = mean_squared_error(DF_target,LinearPredictions_final["Appliances_predicted"])
linear_coeff_variation = np.sqrt(linear_mean_squared_error)/DF_target["Appliances"].mean()

print "R2 = " + str(linear_R2_score)
print "MAE = " + str(linear_mean_absolute_error)
print "MSE = " + str(linear_mean_squared_error)
print "CV = " + str(linear_coeff_variation)

##### linear model with cross-validation #####

from sklearn.model_selection import cross_val_predict
CV_LinearPredictions = cross_val_predict(linear_reg, DF_features, DF_target, cv = 10)
CV_LinearPredictions_DF = pd.DataFrame(CV_LinearPredictions, index = DF_target.index, columns = ["Appliances_predicted_CV"])
final_CV_LinearPredictions_DF = DF_target.join(CV_LinearPredictions_DF)

"""
residuals_CV = pd.DataFrame(final_CV_LinearPredictions_DF["Appliances_predicted_CV"] - final_CV_LinearPredictions_DF["Appliances"], columns = ["Residuals"])
residuals_dates_CV = residuals_CV['2016-05-05']

final_CV_predictions_ChosenDates = final_CV_LinearPredictions_DF['2016-05-05']

plt.subplot(211)
final_CV_predictions_ChosenDates["Appliances"].plot(color = 'g', legend = True)
final_CV_predictions_ChosenDates["Appliances_predicted_CV"].plot(color = 'b', legend = True)
plt.grid()
plt.ylabel('Energy consumption [Wh]')
plt.xlabel('')
plt.title("LINEAR REGRESSION WITH CROSS VALIDATION")
plt.subplot(212)
residuals_dates_CV["Residuals"].plot()
plt.axhline(y = 0, color = 'black')
plt.grid()
plt.title("RESIDUALS")
plt.xlabel("date")
plt.show()
"""

CV_R2_score = r2_score(final_CV_LinearPredictions_DF["Appliances"],final_CV_LinearPredictions_DF["Appliances_predicted_CV"])
CV_mean_absolute_error = mean_absolute_error(final_CV_LinearPredictions_DF["Appliances"], final_CV_LinearPredictions_DF["Appliances_predicted_CV"])
CV_mean_squared_error = mean_squared_error(final_CV_LinearPredictions_DF["Appliances"], final_CV_LinearPredictions_DF["Appliances_predicted_CV"])
CV_coeff_variation = np.sqrt(CV_mean_squared_error)/final_CV_LinearPredictions_DF["Appliances"].mean()

print "R2 = " + str(CV_R2_score)
print "MAE = " + str(CV_mean_absolute_error)
print "MSE = " + str(CV_mean_squared_error)
print "CV = " + str(CV_coeff_variation)

##### Comparison between linear with and without CV #####

compare = LinearPredictions_final.join(CV_LinearPredictions_DF)

compare['2016-05-05'].plot(colors= ("g","r","b"))
plt.title("LINEAR REGRESSION MODEL WITH AND WITHOUT CROSS VALIDATION")
plt.ylabel("Energy consumption [Wh]")
plt.grid()
plt.legend(('actual appliances consumption', 'linear regression model predictions',  'linear regression with cross validation predictions'))
plt.show()

##### Let's try a more complex algorithm - Random Forest Regressor #####

from sklearn.ensemble import RandomForestRegressor
reg_RF = RandomForestRegressor()

CV_RF_predictions = cross_val_predict(reg_RF, DF_features, DF_target, cv = 10)
CV_RF_predictions_DF = pd.DataFrame(CV_RF_predictions, index = DF_target.index, columns = ["Appliances_predicted_RF"])
final_CV_RF_predictions_DF = DF_target.join(CV_RF_predictions_DF)

RF_R2_score = r2_score(final_CV_RF_predictions_DF["Appliances"],final_CV_RF_predictions_DF["Appliances_predicted_RF"])
RF_mean_absolute_error = mean_absolute_error(final_CV_RF_predictions_DF["Appliances"], final_CV_RF_predictions_DF["Appliances_predicted_RF"])
RF_mean_squared_error = mean_squared_error(final_CV_RF_predictions_DF["Appliances"], final_CV_RF_predictions_DF["Appliances_predicted_RF"])
RF_coeff_variation = np.sqrt(CV_mean_squared_error)/final_CV_RF_predictions_DF["Appliances"].mean()

##### Comparison between linear and Random Forest regression #####

compare_RF = CV_LinearPredictions_DF.join(final_CV_RF_predictions_DF)

compare_RF['2016-05-05'].plot(color = ("b","g","r"))
plt.title("LINEAR VS RANDOM FOREST REGRESSION")
plt.ylabel("Energy consumption [Wh]")
plt.grid()
plt.legend(('linear regression model (with CV) predictions', 'actual appliances consumption',  'Random Forest regression model (with CV) predictions'))
plt.show()


##### MACHINE LEARNING PART USING THE ORIGINAL DATA FRAME #####

""" Here we use as features the data included in the original data frame, so without shifted columns.
    We only add the information about day, month, etc """
  
DF_final_original = DF_complete.dropna()

#DF_final_original = uf.lag_column10(DF_final_original, "Appliances", 6*24)
#DF_final_original = uf.lag_column10(DF_final_original, "T_out", 6*6) 
        
DF_final_original["hour"] = DF_final_original.index.hour
DF_final_original["day"] = DF_final_original.index.dayofweek   
DF_final_original["month"] = DF_final_original.index.month

DF_final_original['weekend'] = [uf.weekendDetector(thisDay) for thisDay in DF_final_original.index.dayofweek]
DF_final_original['day_night'] = [uf.dayDetector(thisHour) for thisHour in DF_final_original.index.hour] 

DF_final_original.dropna(inplace = True)

DF_target2 = DF_final_original[["Appliances"]]
DF_features2 = DF_final_original.drop("Appliances", axis = 1)   

X_train2, X_test2, y_train2, y_test2 = train_test_split(DF_features2, DF_target2, test_size = 0.2, random_state = 41234)

linear_reg2 = linear_model.LinearRegression()

CV_LinearPredictions2 = cross_val_predict(linear_reg2, DF_features2, DF_target2, cv = 10)
CV_LinearPredictions_DF2 = pd.DataFrame(CV_LinearPredictions2, index = DF_target2.index, columns=["Appliances_predicted_CV2"])
final_CV_LinearPredictions_DF2 = CV_LinearPredictions_DF2.join(DF_target2)

CV_R2_score_original = r2_score(final_CV_LinearPredictions_DF2["Appliances"],final_CV_LinearPredictions_DF2["Appliances_predicted_CV2"])
CV_mean_absolute_error_original = mean_absolute_error(final_CV_LinearPredictions_DF2["Appliances"], final_CV_LinearPredictions_DF2["Appliances_predicted_CV2"])
CV_mean_squared_error_original = mean_squared_error(final_CV_LinearPredictions_DF2["Appliances"], final_CV_LinearPredictions_DF2["Appliances_predicted_CV2"])
CV_coeff_variation_original = np.sqrt(CV_mean_squared_error)/final_CV_LinearPredictions_DF2["Appliances"].mean()

print "R2 = " + str(CV_R2_score_original)
print "MAE = " + str(CV_mean_absolute_error_original)
print "MSE = " + str(CV_mean_squared_error_original)
print "CV = " + str(CV_coeff_variation_original)

compare_features = final_CV_LinearPredictions_DF.join(CV_LinearPredictions_DF2)

compare_features['2016-03-021'].plot(colors = ("g","b","m"))
plt.title("COMPARISON BETWEEN PREDICTIONS WITH DIFFERENT FEATURES")
plt.ylabel("Energy consumption [Wh]")
plt.legend(('actual appliances consumption', 'linear regression with cross validation predictions', 'linear CV predictions - original dataset features'))
plt.grid()
plt.show()