# -*- coding: utf-8 -*-
"""## Importing modules"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# this will just check if we are logged out it logs in
authorizeIfLoggedOut()

file_list_GDrive = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()

for file1 in file_list_GDrive:  
    if "Data-Driven-Building-Simulation" in file1['title']:
        Folder_ID_Data_Driven_Building_Simulation= file1['id']

print(Folder_ID_Data_Driven_Building_Simulation)

file_list_Data_Driven_Building_Simulation = drive.ListFile({'q': "'%s' in parents and trashed=false" % Folder_ID_Data_Driven_Building_Simulation}).GetList()

for file2 in file_list_Data_Driven_Building_Simulation:  
    if "Data" in file2['title']:
        Folder_ID_Data= file2['id']
        
for file4 in file_list_Data_Driven_Building_Simulation:
  
    if "Figures" in file4['title']:
        Folder_ID_Figures= file4['id']        
        
        
file_list_Data = drive.ListFile({'q': "'%s' in parents and trashed=false" % Folder_ID_Data}).GetList()

for file3 in file_list_Data:  
    if "energydata_complete.csv" in file3['title']:
        print "energydata_complete.csv exists"
        ID_energydata_complete_gen= file3['id']
        file_energydata_complete_gen = drive.CreateFile({'id': ID_energydata_complete_gen})
        file_energydata_complete_gen.GetContentFile('energydata_complete.csv')

"""# Importing datasets

### Creating the path for our "energy_complete.csv" to be imported
"""

joinedDF_FileName = "energydata_complete.csv"

"""## Converting the index to real timestamps, so that the time indexes will be ment properly as time and not only as numbers"""

DF_main = pd.read_csv(joinedDF_FileName, sep =",", index_col = 0)
oldIndex_DF_main = DF_main.index
newIndex_DF_main = pd.to_datetime(oldIndex_DF_main) # hey computer, the indexes are TIME, not random numbers
DF_main.index = newIndex_DF_main
DF_main

DF_main_clean = DF_main.dropna() #deleting all NaN values, clean now
DF_main_clean.tail()

"""## Columns for time-related features

We create a column which tells us the corresponding hour of the day for each row
"""

DF_main_clean.loc[:,"hour"]=DF_main_clean.index.hour  #adding a column for corresponding hours 
DF_main_clean.head(30)

"""We create 2 columns that represent the time as a continuous feature, thanks to SIN and COS"""

DF_main_clean.loc[:,"sin(hour)"] = np.sin(DF_main_clean.index.hour*2*np.pi/24) # i add a column where time is continuous (from 1 to -1 and again 1)
DF_main_clean.loc[:,"cos(hour)"] = np.cos(DF_main_clean.index.hour*2*np.pi/24) # another column, with cosin

DF_main_clean.head(30) #look: 6.123234e-17 is zero!

"""Adding other columns that, for each row, will tell us the corresponding day of the week (from 1 to 7), month of the year (from 1 to 12), and week of the year (from 1 to 52)"""

DF_main_clean.loc[:,"month"] = DF_main_clean.index.month  #adding a column for the months
DF_main_clean.loc[:,"dayOfWeek"]=DF_main_clean.index.dayofweek
DF_main_clean.loc[:,"weekOfYear"] = DF_main_clean.index.week
DF_main_clean.tail(52)

"""We create a **function** that, for each row, will give to a flag variable "weekendLabel" the value of 1 in case that row corresponds to a weekend day.
Then we create a similar function that, for each row, will give to a flag variable "workingHourLabel" the value of 1 in case that row corresponds to a working hour.
"""

def WeekendDetector(day):
    if (day == 5 or day==6):
        weekendLabel = 1
    else:
        weekendLabel = 0
    return weekendLabel        # function: if it's weekend, flag = 1

def workingHourDetector(hour):
    if (hour>=9 and hour <= 19):
        workingHourLabel = 1
    else:
        workingHourLabel = 0
    return workingHourLabel    # if time is from 9 am to 7 pm, people are working! flag = 1

"""We can now apply these functions on the corresponding colums, which are created right now:"""

DF_main_clean.loc[:,"weekendLabel"] = DF_main_clean.loc[:,"dayOfWeek"].apply(WeekendDetector)  # apply functions above
DF_main_clean.loc[:,"WorkingHourLabel"] = DF_main_clean.loc[:,"hour"].apply(workingHourDetector)
DF_main_clean.head(30)

"""Thanks to the function **range(start, end, step)** we can build a script that creates columns of lagged parameters.
First we create this kind script valid only for one single column "T1":
"""

# first let's do this using a for to lag temperature for 6 hours

lag_start = 1
lag_end = 3
lag_interval = 1 

columnName = "T1"
inputDF = DF_main_clean
for i in range(lag_start,lag_end+1, lag_interval):  # "lag_end+1" because otherwise the vector "range" would be only [1,2]!
    new_column_name = columnName+" -"+str(i)+"0 min"
    print new_column_name
    inputDF.loc[:,new_column_name] = inputDF.loc[:,columnName].shift(i)
    inputDF.dropna(inplace=True)
    
inputDF.head()   # i create 3 new columns in a single step, each one shifted by 10 minutes and then cleaned from void spaces!

"""Now we create a **function** "lag_gen_feature" which will be able to **apply** the same type of script that we have above for **all the columns we want**"""

def lag_gen_feature(inputDF,columnName,lag_start,lag_end,lag_interval):
    for i in range(lag_start,lag_end+1, lag_interval):
        new_column_name = columnName+" -"+str(i)+"0 min"
        if new_column_name in inputDF.columns:
            pass
        else:
            print new_column_name
            inputDF.loc[:,new_column_name] = inputDF.loc[:,columnName].shift(i)
            inputDF.dropna(inplace=True)
    return inputDF
# creating the same type of function, valid for everything

DF_main_clean = lag_gen_feature(DF_main_clean,"RH_1",6,6,1) # apply the function above to relative humidity of room 1: new colum lagged 1 hour
DF_main_clean = lag_gen_feature(DF_main_clean,"Windspeed",6,6,1) # apply the function above to Windspeed: new colum lagged 1 hour
DF_main_clean.head()

DF_joined_WithGeneratedFeatures = DF_main_clean # changing name
FileName_DF_joined_WithGeneratedFeatures = "Belgium_DF_joined_WithGeneratedFeatures.csv" 
DF_joined_WithGeneratedFeatures.to_csv(FileName_DF_joined_WithGeneratedFeatures)

CompleteDF_metaData = {'title' : FileName_DF_joined_WithGeneratedFeatures,"parents": [{"kind": "drive#fileLink", "id": Folder_ID_Data}] }
CompleteDF_File = drive.CreateFile(CompleteDF_metaData)
CompleteDF_File.SetContentFile(FileName_DF_joined_WithGeneratedFeatures)
CompleteDF_File.Upload()

"""## Correlation Matrix!"""

DF_main_clean.corr() #CORRELATION MATRIX!

"""## Creating target and features dataframes"""

DF_april = DF_main_clean.loc["2016-04-01":"2016-04-30",:]

DF_target = DF_april.loc[:,["RH_1"]] # panda with only a column of time index and a column of relative humidity of room 1 for april
DF_featurs = DF_april.drop("RH_1", axis=1) # panda with all the other features EXCLUDING relative humidity of room 1

"""## **TRAIN / TEST splitting!** using Scikit."""

from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(DF_featurs,DF_target,test_size=0.2,random_state= 41234)

(X_train) 
# i'm creating 2 matrixes for training with 80% of the data, shuffled randomly: X_train is without ac_consumption, Y_train is ONLY ac_consumption
# the remaining 20 % of the data are used for the test: X_test and Y_test

"""## **LINEAR REGRESSION !**

First we import the linear model and then we use it for training!
"""

from sklearn import linear_model
linear_reg = linear_model.LinearRegression()
print(linear_reg)

"""Let's train the model with the train data (X_train and Y_train)"""

linear_reg.fit(X_train,Y_train) # LINEAR REGRESSION OF THE TRAINING HERE!

"""Predicting the values for the test set"""

predicted_linearReg_split = linear_reg.predict(X_test)  

predicted_linearReg_split #PREDICTING HERE! output: predictions WITHOUT time indexes!

Predicted_DF_linearReg_split = pd.DataFrame(predicted_linearReg_split, index= Y_test.index, columns=["RH_1_predicted_LinearReg_FirstWeekApril"])  
Predicted_DF_linearReg_split.head(24) # putting values into a data frame wit time indexes!
Predicted_DF_linearReg_split = Predicted_DF_linearReg_split.join(Y_test) # so merge it with the Y_test, which is the list of exact values
Predicted_DF_linearReg_split.head(50)

Predicted_DF_linearReg_split_FirstWeekApril = Predicted_DF_linearReg_split.loc["2016-04-01":"2016-04-08",:]
Predicted_DF_linearReg_split_FirstWeekApril.plot()

"""We can measure the accuracy of our linear regression model for predictions, by comparing it to the Y_test values: mean absolute error (MAE), mean squared error (MSE), R squared (R2)"""

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score  # we find an average value of error because error varies! look chart!
MAE_linearReg_split= mean_absolute_error(predicted_linearReg_split,Y_test) # function "mean_absolute_error"! MAE
MSE_linearReg_split= mean_squared_error(predicted_linearReg_split,Y_test)  # funtion "mean_squared_error"! MSE
R2_linearReg_split = r2_score(predicted_linearReg_split,Y_test)  # R squared! R2
print("MAE :"+str(MAE_linearReg_split))  
print("MSE :"+ str(MSE_linearReg_split))
print("R2 :"+str(R2_linearReg_split))

"""## **CROSS VALIDATION !**

First we import it and then we apply it toour linear regression model:
"""

from sklearn.model_selection import cross_val_predict
predict_linearReg_CV = cross_val_predict(linear_reg, DF_featurs, DF_target, cv=10)  #predictions without time indexes!

# we put our predictions inside a dataframe with time indexes
predicted_DF_linearReg_CV = pd.DataFrame(predict_linearReg_CV,index=DF_target.index, columns=["T1_predicted_linearReg_CV"])
predicted_DF_linearReg_CV.head(24)

predicted_DF_linearReg_CV = predicted_DF_linearReg_CV.join(DF_target) # again: predicted values together with actual values (DF_target)
predicted_DF_linearReg_CV.head(40)

predicted_DF_linearReg_CV_august = predicted_DF_linearReg_CV.loc["2016-04-01":"2016-04-08",:]
predicted_DF_linearReg_CV_august.plot()

"""Measuring the accuracy of Cross Validation in the same way we did before:"""

MAE_linearReg_CV= mean_absolute_error(predict_linearReg_CV, DF_target)
MSE_linearReg_CV= mean_squared_error(predict_linearReg_CV,DF_target)
R2_linearReg_CV = r2_score(predict_linearReg_CV,DF_target)
print("MAE :"+str(MAE_linearReg_CV))
print("MSE :"+ str(MSE_linearReg_CV))
print("R2 :"+str(R2_linearReg_CV))