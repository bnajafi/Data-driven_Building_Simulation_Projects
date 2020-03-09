
# coding: utf-8

# # EETBS Project                      
# ### (PART 2/2)
# 
# #### In this project we want to predict the room internal temperature of the next 15 min, 30 min and 1 hour and the room occupancy
# #### This part is dedicated to occupancy prediction

# #### We first import the useful modules

# In[308]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
get_ipython().magic(u'matplotlib notebook')


# #### After we have chosen the suitable datatset for our scope, we import it
# This dataset is collected from a monitor system mounted in a domotic house. It corresponds to approximately 40 days of monitoring data. The data was sampled every minute, computing and uploading it smoothed with 15 minute means.
# https://archive.ics.uci.edu/ml/datasets/SML2010
# 
# Since this dataset didn't include occupancy data we tried to complete it with another database from another residential building: https://openei.org/datasets/dataset/long-term-occupancy-data-for-residential-and-commercial-building.
# 
# Obviously that would lead to an inaccurate analysis because human behaviour for sure will not be perfectly matched.
# Anyways we expect occupancy to be correlated to humidity, CO2, light and working hour.

# In[309]:

ExternalFiles_folder = r"C:\Users\Angela\Google Drive\Progetto Building"
FileName = "DatasetNOSTRO_conOCCUPANCY.csv"
path_File = os.path.join(ExternalFiles_folder,FileName)


# In[310]:

DF_main= pd.read_csv(path_File, sep=";" , index_col=0)


# In[311]:

DF_main.head()


# #### We decide to construct a data frame made up of only the columns to which we are interested 

# In[312]:

DATASET = DF_main[["4:Temperature_Habitacion_Sensor","22:Temperature_Exterior_Sensor","18:Meteo_Exterior_Piranometro","7:CO2_Habitacion_Sensor","9:Humedad_Habitacion_Sensor","11:Lighting_Habitacion_Sensor","Occupancy"]]
DATASET.loc[DATASET.loc[:,"18:Meteo_Exterior_Piranometro"]<0,"18:Meteo_Exterior_Piranometro"]=0


# #### We have changed the negative values of the irradiance into zero

# In[313]:

DATASET.head(100)


# #### Time related features

# In[314]:

oldIndex_main = DATASET.index  
NewIndex_main = pd.to_datetime(oldIndex_main, dayfirst=True)
DATASET.index = NewIndex_main
DATASET.head()


# #### Renaming columns

# In[315]:

DATASET.columns = ["Tin", "Tout", "Sun", "CO2", "Humidity", "Light","Occupancy"]
DATASET.head()


# #### Adding time-related features: extracting the hour and assign it to a column called hour, same with minute, week, day of week, month. 

# In[316]:

DATASET.loc[:,"hour"] = DATASET.index.hour
DATASET.head()


# In[317]:

DATASET.loc[:,"min"]=DATASET.index.minute
DATASET.head()


# #### But the parameter hour does not represent the continuity of the time : one alternative solution is to use sin(hour) and cos(hour)

# In[318]:

DATASET.loc[:,"sin(hour)"] = np.sin(DATASET.index.hour*2*np.pi/24)
DATASET.loc[:,"cos(hour)"] = np.cos(DATASET.index.hour*2*np.pi/24)
DATASET.loc[:,"sin(min)"] = np.sin(DATASET.index.minute*2*np.pi/24)
DATASET.loc[:,"cos(min)"] = np.cos(DATASET.index.minute*2*np.pi/24)
DATASET.head()


# In[319]:

DATASET.loc[:,"month"] = DATASET.index.month
DATASET.loc[:,"dayOfWeek"] = DATASET.index.dayofweek
DATASET.loc[:,"weekOfYear"] = DATASET.index.week
DATASET.head(100)


# #### Finally we define two functions to find whether the day is a weekend day and whether the hour is a working hour. Considering that day 1 (13.03.2012) was Tuesday.

# In[320]:

def WeekendDetector(day):
    if day == 5 or day == 6:
        weekendLabel = 1
    else:
        weekendLabel = 0
    return weekendLabel


# In[321]:

def WorkingHourDetector(hour):
    if hour >= 9 and hour < 12:
        workingHourLabel = 1
    elif hour >= 15 and hour < 19:
        workingHourLabel = 1
    else:
        workingHourLabel = 0
    return workingHourLabel


# In[322]:

DATASET.loc[:,"WeekendLabel"]=DATASET.loc[:,"dayOfWeek"].apply(WeekendDetector)
DATASET.loc[:,"WorkingHourLabel"]=DATASET.loc[:,"hour"].apply(WorkingHourDetector)
DATASET.head(100)


# #### Create function that create lagged parameters for us

# In[323]:

def lag_gen_feature(inputDF, columnName, lag_start, lag_end, lag_interval):
 for i in range(lag_start,lag_end+1,lag_interval):
    new_column_name = columnName+ " -"+str(i)+"h/4"
    if new_column_name in inputDF.columns:
        pass
    else:
        print new_column_name
        inputDF.loc[:,new_column_name]=inputDF.loc[:,columnName].shift(i)
        inputDF.dropna(inplace=True)
 return inputDF


# In[324]:

DATASET = lag_gen_feature(DATASET,"Tout",1,16,1)
DATASET = lag_gen_feature(DATASET,"Sun",15,30,1)
DATASET = lag_gen_feature(DATASET,"CO2",10,25,1) #facciamo partire da 10 perchè notato che aumentava
DATASET = lag_gen_feature(DATASET,"Humidity",1,16,1)
DATASET = lag_gen_feature(DATASET,"Tin",1,16,1)
DATASET = lag_gen_feature(DATASET,"Occupancy",1,16,1)
DATASET.head(10)


# #### Correlation matrix

# In[325]:

CorrelationMatrix=DATASET.corr()
CorrelationMatrix


# #### Saving correlation matrix

# In[326]:

#Project_folder = r"C:\Users\Angela\Google Drive\Progetto Building"
#path_modified_csv = os.path.join(Project_folder,"CorrelationMatrixconOccupancy.csv")
#CorrelationMatrix.to_csv(path_modified_csv, sep=";")


# # OCCUPANCY PREDICTION in the next 15 min

# #### We want to see which column is more correlated to the one to which we are interested

# In[395]:

RowOccupancy=CorrelationMatrix.loc["Occupancy",:]
Ordered_Occ_corr=pd.Series(RowOccupancy).order(ascending=False)
Ordered_Occ_corr.head(50)


# We expected occupancy to be correlated to humidity, CO2, light and working hour but results are not very good already.
# In fact we took the occupancy column from another dataset of a domestic house, this means that human behaviour for sure will not be perfectly matched.

# #### Creating target and features dataframes

# In[328]:

DF_target_occ = DATASET.loc[:,["Occupancy"]]
DF_featurs_occ = DATASET.drop("Occupancy",axis=1)


# ## A) Train-Test splitting occupancy

# #### We decide that the test size will be the 20% of the train size but the row are chosen randomly

# In[329]:

from sklearn.model_selection import train_test_split 
X_train_occ, X_test_occ, Y_train_occ, Y_test_occ = train_test_split(DF_featurs_occ, DF_target_occ, test_size=0.2, random_state=41234)


# #### Let's import the linear model

# In[330]:

from sklearn import linear_model
linear_reg_occ = linear_model.LinearRegression()
print(linear_reg_occ)


# #### Let's first train my model with the train data

# In[331]:

linear_reg_occ.fit(X_train_occ,Y_train_occ)


# #### Predicting the values for the test set

# In[332]:

predicted_linearReg_split_occ = linear_reg_occ.predict(X_test_occ)


# In[333]:

Predicted_DF_linearReg_split_occ = pd.DataFrame(predicted_linearReg_split_occ, index= Y_test_occ.index, columns=["Occupancy predicted test train"])
Predicted_DF_linearReg_split_occ = Predicted_DF_linearReg_split_occ.join(Y_test_occ)
Predicted_DF_linearReg_split_occ.head()


# #### Plotting a sample

# In[399]:

Predicted_DF_linearReg_split_sample_occ = Predicted_DF_linearReg_split_occ.loc["2012-03-19":"2012-03-30",:]
Predicted_DF_linearReg_split_sample_occ.plot()
plt.xlabel("Time [days]")
plt.title("Real occupancy vs Predicted occupancy with Test-Train")


# #### Measuring the accuracy

# In[335]:

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
MAE_linearReg_split_occ= mean_absolute_error(Y_test_occ, predicted_linearReg_split_occ)
MSE_linearReg_split_occ= mean_squared_error(Y_test_occ,predicted_linearReg_split_occ)
R2_linearReg_split_occ = r2_score(Y_test_occ, predicted_linearReg_split_occ)
print("MAE test_train:"+str(MAE_linearReg_split_occ))
print("MSE test_train:"+ str(MSE_linearReg_split_occ))
print("R2 test_train:"+str(R2_linearReg_split_occ))


# #### R2 value is not high Q.E.D.

# ## B) Cross Validation occupancy

# #### Although we know that this a time-series let's carry out a cross valdiation
# #### We divide our data frame into 10 blocks: this time we train on 9 blocks and test on 1 block, changing them each time. In this way we will test all the rows of the dataframe 

# In[336]:

from sklearn.model_selection import cross_val_predict
predicted_linearReg_CV_occ = cross_val_predict(linear_reg_occ, DF_featurs_occ, DF_target_occ, cv=10)


# In[337]:

Predicted_DF_linearReg_CV_occ = pd.DataFrame(predicted_linearReg_CV_occ, index= DF_target_occ.index, columns=["Occupancy predicted Cross Validation"])
Predicted_DF_linearReg_CV_occ = Predicted_DF_linearReg_CV_occ.join(DF_target_occ)
Predicted_DF_linearReg_CV_occ.head(24)


# In[398]:

Predicted_DF_linearReg_CV_sample_occ = Predicted_DF_linearReg_CV_occ.loc["2012-03-19":"2012-03-30",:]
Predicted_DF_linearReg_CV_sample_occ.plot()
plt.xlabel("Time [days]")
plt.title("Real occupancy vs Predicted occupancy with Cross Validation")


# In[339]:

MAE_linearReg_CV_occ = mean_absolute_error(DF_target_occ, predicted_linearReg_CV_occ)
MSE_linearReg_CV_occ = mean_squared_error(DF_target_occ,predicted_linearReg_CV_occ)
R2_linearReg_CV_occ = r2_score(DF_target_occ,predicted_linearReg_CV_occ)
print("MAE cross_validation:"+str(MAE_linearReg_CV_occ))
print("MSE cross_validation:"+ str(MSE_linearReg_CV_occ))
print("R2 cross_validation:"+str(R2_linearReg_CV_occ))


# Cross Validation results are worse in fact, being the dataset a time series, it can happen for example that during weekends (or holidays) results are messed up (one block will contain holiday days). In these cases it is better to use methods taking random data (ex: Train-Test) or walk foward.

# ## C) Random Forest occupancy

# #### It operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.

# In[340]:

from sklearn.ensemble import RandomForestRegressor
RF_reg_occ = RandomForestRegressor()


# In[341]:

predicted_RF_reg_CV_occ= cross_val_predict(RF_reg_occ, DF_featurs_occ, DF_target_occ, cv=10)


# In[342]:

Predicted_DF_RF_Reg_CV_occ = pd.DataFrame(predicted_RF_reg_CV_occ, index= DF_target_occ.index, columns=["Occupancy predicted Random Forest"])
Predicted_DF_RF_Reg_CV_occ = Predicted_DF_RF_Reg_CV_occ.join(DF_target_occ)
Predicted_DF_RF_Reg_CV_occ.head(24)


# In[400]:

Predicted_DF_RF_Reg_CV_sample_occ = Predicted_DF_RF_Reg_CV_occ.loc["2012-03-19":"2012-03-30",:]
Predicted_DF_RF_Reg_CV_sample_occ.plot()
plt.xlabel("Time [days]")
plt.title("Real occupancy vs Predicted occupancy with Random Forest")


# In[344]:

MAE_RF_Reg_CV_occ = mean_absolute_error(DF_target_occ, predicted_RF_reg_CV_occ)
MSE_RF_Reg_CV_occ = mean_squared_error(DF_target_occ,predicted_RF_reg_CV_occ)
R2_RF_Reg_CV_occ = r2_score(DF_target_occ, predicted_RF_reg_CV_occ)
print("MAE random_forest:"+ str(MAE_RF_Reg_CV_occ))
print("MSE random_forest:"+ str(MSE_RF_Reg_CV_occ))
print("R2 random_forest:"+ str(R2_RF_Reg_CV_occ))


# It is better to predict low values (zeros)  because it doesn't use linear regression but it uses trees and branches. In our particular case this approach is not good.

# ## D) Walk Forward occupancy

# #### Instead of using a single large testing period, we split the whole data into smaller chunks to perform consecutive analyses. For example, instead of optimizing ten years of data and using the eleventh year for testing, the optimization is done first across the first three years, and the system is tested in the fourth. Once this test is completed, the four years window is moved forward one year, and the procedure is repeated until the last year is reached. 
# #### We'll do the same training on 10 days and moving forward 1 day until the last day (11.04.2012) of our dataset is reached.

# In[345]:

DF_online_prediction_occ = pd.DataFrame(index=DATASET.index)
DF_online_prediction_occ.head()


# In[460]:

FirstTimeStampMeasured_occ = DATASET.index[0]
PeriodOfTraining_occ = pd.Timedelta(12, unit="d")
FirstTimeStampToPredict_occ = FirstTimeStampMeasured_occ + PeriodOfTraining_occ
LastTimeStampMeasured_occ = DATASET.index[-1]
training_start_time_stamp_occ = FirstTimeStampMeasured_occ
training_end_time_stamp_occ = FirstTimeStampToPredict_occ - pd.Timedelta(15, unit="m")


# In[462]:

FirstTimeStampMeasured_occ


# In[463]:

PeriodOfTraining_occ


# In[464]:

FirstTimeStampToPredict_occ


# In[465]:

LastTimeStampMeasured_occ


# In[466]:

DF_online_prediction_occ = DF_online_prediction_occ.truncate(before = FirstTimeStampToPredict_occ)
DF_online_prediction_occ.head()


# In[467]:

time_stamp_to_predict_occ = FirstTimeStampToPredict_occ
while (time_stamp_to_predict_occ <= LastTimeStampMeasured_occ):
    DF_features_train_occ = DF_featurs_occ.truncate(before=training_start_time_stamp_occ,after=training_end_time_stamp_occ)
    DF_target_train_occ = DF_target_occ.truncate(before=training_start_time_stamp_occ,after=training_end_time_stamp_occ)
    DF_features_test_occ = DF_featurs_occ.loc[time_stamp_to_predict_occ,:].values.reshape(1,-1)
    DF_target_test_occ = DF_target_occ.loc[time_stamp_to_predict_occ,"Occupancy"]
  #let's train
    RF_reg_occ.fit(DF_features_train_occ, DF_target_train_occ)
    predicted_occ = RF_reg_occ.predict(DF_features_test_occ)
    DF_online_prediction_occ.loc[time_stamp_to_predict_occ,"Predicted"] = predicted_occ
    DF_online_prediction_occ.loc[time_stamp_to_predict_occ,"Real"] = DF_target_test_occ
    time_stamp_to_predict_occ = time_stamp_to_predict_occ + pd.Timedelta(15, unit="m")
    training_start_time_stamp_occ = training_start_time_stamp_occ + pd.Timedelta(15, unit="m")
    training_end_time_stamp_occ = training_end_time_stamp_occ + pd.Timedelta(15, unit="m")


# In[468]:

DF_online_prediction_occ.head()


# In[469]:

MAE_WF_occ= mean_absolute_error(DF_online_prediction_occ[["Real"]],DF_online_prediction_occ[["Predicted"]])
MSE_WF_occ= mean_squared_error(DF_online_prediction_occ[["Real"]],DF_online_prediction_occ[["Predicted"]])
R2_WF_occ = r2_score(DF_online_prediction_occ[["Real"]],DF_online_prediction_occ[["Predicted"]])
print("MAE walk_forward:"+ str(MAE_WF_occ))
print("MSE walk_forward:"+ str(MSE_WF_occ))
print("R2 walk_forward:"+ str(R2_WF_occ))


# #### Create a table containing the different methods and their relative errors

# In[470]:

column_name = ["MAE","MSE","R2"]
index_name = ["Test Train","Cross Validation","Random Forest", "Walk Forward"]
test_train_errors_occ = [MAE_linearReg_split_occ,MSE_linearReg_split_occ,R2_linearReg_split_occ]
cross_validation_errors_occ = [MAE_linearReg_CV_occ,MSE_linearReg_CV_occ,R2_linearReg_CV_occ]
random_forest_errors_occ = [MAE_RF_Reg_CV_occ,MSE_RF_Reg_CV_occ,R2_RF_Reg_CV_occ]
walk_forward_errors_occ = [MAE_WF_occ,MSE_WF_occ,R2_WF_occ]
errors_list_occ = [test_train_errors_occ,cross_validation_errors_occ,random_forest_errors_occ,walk_forward_errors_occ]
errors_data_frame_occ = pd.DataFrame(errors_list_occ, index=index_name, columns=column_name)

errors_data_frame_occ


# # OCCUPANCY PREDICTION in the next hour

# In[420]:

dataset_1 = DATASET.drop(["Occupancy -1h/4","Occupancy -2h/4","Occupancy -3h/4","Occupancy -4h/4"],axis=1)


# In[421]:

DF_target_1 = dataset_1.loc[:,["Occupancy"]]
DF_featurs_1 = dataset_1.drop("Occupancy",axis=1)


# #### Creating target and features dataframes

# In[422]:

DF_target_1 = dataset_1.loc[:,["Occupancy"]]
DF_featurs_1 = dataset_1.drop("Occupancy",axis=1)


# ## A) Train-Test splitting occupancy 1h

# #### We decide that the test size will be the 20% of the train size but the row are chosen randomly

# In[423]:

from sklearn.model_selection import train_test_split 
X_train_1, X_test_1, Y_train_1, Y_test_1 = train_test_split(DF_featurs_1, DF_target_1, test_size=0.2, random_state=41234)


# #### Let's import the linear model

# In[424]:

from sklearn import linear_model
linear_reg_1 = linear_model.LinearRegression()
print(linear_reg_1)


# #### Let's first train my model with the train data

# In[425]:

linear_reg_1.fit(X_train_1,Y_train_1)


# #### Predicting the values for the test set

# In[426]:

predicted_linearReg_split_1 = linear_reg_1.predict(X_test_1)


# In[427]:

Predicted_DF_linearReg_split_1 = pd.DataFrame(predicted_linearReg_split_1, index= Y_test_1.index, columns=["Occupancy predicted Test-Train"])
Predicted_DF_linearReg_split_1 = Predicted_DF_linearReg_split_1.join(Y_test_1)
Predicted_DF_linearReg_split_1.head()


# #### Plotting a sample

# In[428]:

Predicted_DF_linearReg_split_sample_1 = Predicted_DF_linearReg_split_1.loc["2012-03-19":"2012-03-30",:]
Predicted_DF_linearReg_split_sample_1.plot()
plt.xlabel("Time [days]")
plt.title("Real occupancy vs Predicted occupancy with Test-Train")


# In[429]:

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
MAE_linearReg_split_1= mean_absolute_error(Y_test_1,predicted_linearReg_split_1)
MSE_linearReg_split_1= mean_squared_error(Y_test_1, predicted_linearReg_split_1)
R2_linearReg_split_1 = r2_score(Y_test_1, predicted_linearReg_split_1)
print("MAE test_train:"+str(MAE_linearReg_split_1))
print("MSE test_train:"+ str(MSE_linearReg_split_1))
print("R2 test_train:"+str(R2_linearReg_split_1))


# #### R2 value is not high Q.E.D.

# ## B) Cross Validation occupancy 1h

# #### We divide our data frame into 10 blocks: this time we train on 9 blocks and test on 1 block, changing them each time. In this way we will test all the rows of the dataframe 

# In[430]:

from sklearn.model_selection import cross_val_predict
predicted_linearReg_CV_1 = cross_val_predict(linear_reg_1, DF_featurs_1, DF_target_1, cv=10)


# In[431]:

Predicted_DF_linearReg_CV_1 = pd.DataFrame(predicted_linearReg_CV_1, index= DF_target_1.index, columns=["Occupancy predicted Cross Validation"])
Predicted_DF_linearReg_CV_1 = Predicted_DF_linearReg_CV_1.join(DF_target_1)
Predicted_DF_linearReg_CV_1.head(24)


# In[432]:

Predicted_DF_linearReg_CV_sample_1 = Predicted_DF_linearReg_CV_1.loc["2012-03-19":"2012-03-30",:]
Predicted_DF_linearReg_CV_sample_1.plot()
plt.xlabel("Time [days]")
plt.title("Real occupancy vs Predicted occupancy with Cross Validation")


# In[433]:

MAE_linearReg_CV_1 = mean_absolute_error(DF_target_1, predicted_linearReg_CV_1)
MSE_linearReg_CV_1 = mean_squared_error(DF_target_1,predicted_linearReg_CV_1)
R2_linearReg_CV_1 = r2_score(DF_target_1,predicted_linearReg_CV_1)
print("MAE cross_validation:"+str(MAE_linearReg_CV_1))
print("MSE cross_validation:"+ str(MSE_linearReg_CV_1))
print("R2 cross_validation:"+str(R2_linearReg_CV_1))


# ## C) Random Forest occupancy 1h

# In[434]:

from sklearn.ensemble import RandomForestRegressor
RF_reg_1 = RandomForestRegressor()


# In[435]:

predicted_RF_reg_CV_1= cross_val_predict(RF_reg_1, DF_featurs_1, DF_target_1, cv=10)


# In[436]:

Predicted_DF_RF_Reg_CV_1 = pd.DataFrame(predicted_RF_reg_CV_1, index= DF_target_1.index, columns=["Occupancy predicted Random Forest"])
Predicted_DF_RF_Reg_CV_1 = Predicted_DF_RF_Reg_CV_1.join(DF_target_1)
Predicted_DF_RF_Reg_CV_1.head(24)


# In[437]:

Predicted_DF_RF_Reg_CV_sample_1 = Predicted_DF_RF_Reg_CV_1.loc["2012-03-19":"2012-03-30",:]
Predicted_DF_RF_Reg_CV_sample_1.plot()
plt.xlabel("Time [days]")
plt.title("Real occupancy vs Predicted occupancy with Random Forest")


# In[438]:

MAE_RF_Reg_CV_1 = mean_absolute_error(DF_target_1, predicted_RF_reg_CV_1)
MSE_RF_Reg_CV_1 = mean_squared_error(DF_target_1,predicted_RF_reg_CV_1)
R2_RF_Reg_CV_1 = r2_score(DF_target_1, predicted_RF_reg_CV_1)
print("MAE random_forest:"+ str(MAE_RF_Reg_CV_1))
print("MSE random_forest:"+ str(MSE_RF_Reg_CV_1))
print("R2 random_forest:"+ str(R2_RF_Reg_CV_1))


# ## D) Walk Forward occupancy 1h

# In[439]:

DF_online_prediction_1 = pd.DataFrame(index=dataset_1.index)
DF_online_prediction_1.head()


# In[481]:

FirstTimeStampMeasured_1 = dataset_1.index[0]
PeriodOfTraining_1 = pd.Timedelta(10, unit="d")
FirstTimeStampToPredict_1 = FirstTimeStampMeasured_1 + PeriodOfTraining_1
LastTimeStampMeasured_1 = dataset_1.index[-1]
training_start_time_stamp_1 = FirstTimeStampMeasured_1
training_end_time_stamp_1 = FirstTimeStampToPredict_1 - pd.Timedelta(15, unit="m")


# In[482]:

FirstTimeStampMeasured_1


# In[483]:

PeriodOfTraining_1


# In[484]:

FirstTimeStampToPredict_1


# In[485]:

LastTimeStampMeasured_1


# In[486]:

DF_online_prediction_1 = DF_online_prediction_1.truncate(before = FirstTimeStampToPredict_1)
DF_online_prediction_1.head()


# In[487]:

time_stamp_to_predict_1 = FirstTimeStampToPredict_1
while (time_stamp_to_predict_1 <= LastTimeStampMeasured_1):
    DF_features_train_1 = DF_featurs_1.truncate(before=training_start_time_stamp_1,after=training_end_time_stamp_1)
    DF_target_train_1 = DF_target_1.truncate(before=training_start_time_stamp_1,after=training_end_time_stamp_1)
    DF_features_test_1 = DF_featurs_1.loc[time_stamp_to_predict_1,:].values.reshape(1,-1)
    DF_target_test_1 = DF_target_1.loc[time_stamp_to_predict_1,"Occupancy"]
  #let's train
    RF_reg_1.fit(DF_features_train_1, DF_target_train_1)
    predicted_1 = RF_reg_1.predict(DF_features_test_1)
    DF_online_prediction_1.loc[time_stamp_to_predict_1,"Predicted"] = predicted_1
    DF_online_prediction_1.loc[time_stamp_to_predict_1,"Real"] = DF_target_test_1
    time_stamp_to_predict_1 = time_stamp_to_predict_1 + pd.Timedelta(15, unit="m")
    training_start_time_stamp_1 = training_start_time_stamp_1 + pd.Timedelta(15, unit="m")
    training_end_time_stamp_1 = training_end_time_stamp_1 + pd.Timedelta(15, unit="m")


# In[488]:

DF_online_prediction_1.head()


# In[491]:

MAE_WF_1= mean_absolute_error(DF_online_prediction_1[["Real"]],DF_online_prediction_1[["Predicted"]])
MSE_WF_1= mean_squared_error(DF_online_prediction_1[["Real"]],DF_online_prediction_1[["Predicted"]])
R2_WF_1 = r2_score(DF_online_prediction_1[["Real"]],DF_online_prediction_1[["Predicted"]])
print("MAE walk_forward:"+ str(MAE_WF_1))
print("MSE walk_forward:"+ str(MSE_WF_1))
print("R2 walk_forward:"+ str(R2_WF_1))


# #### Create a table containing the different methods and their relative errors

# In[490]:

column_name = ["MAE","MSE","R2"]
index_name = ["Test Train","Cross Validation","Random Forest", "Walk Forward"]
test_train_errors_1 = [MAE_linearReg_split_1,MSE_linearReg_split_1,R2_linearReg_split_1]
cross_validation_errors_1 = [MAE_linearReg_CV_1,MSE_linearReg_CV_1,R2_linearReg_CV_1]
random_forest_errors_1 = [MAE_RF_Reg_CV_1,MSE_RF_Reg_CV_1,R2_RF_Reg_CV_1]
walk_forward_errors_1 = [MAE_WF_1,MSE_WF_1,R2_WF_1]
errors_list_1 = [test_train_errors_1,cross_validation_errors_1,random_forest_errors_1,walk_forward_errors_1]
errors_data_frame_1 = pd.DataFrame(errors_list_1, index=index_name, columns=column_name)

errors_data_frame_1

