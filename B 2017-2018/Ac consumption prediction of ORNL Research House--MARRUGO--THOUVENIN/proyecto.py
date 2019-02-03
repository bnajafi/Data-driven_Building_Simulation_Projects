import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Data import 
DataFolderPath = "/Users/Marrugo/Dropbox/git_for_clone/Ac consumption prediction of ORNL Research House--MARRUGO--THOUVENIN/data_Oak_Ride_National_Laboratory.csv"
DF_Data_RAW = pd.read_csv(DataFolderPath,sep = ",",index_col=0) 
previousIndex= DF_Data_RAW.index
NewparsedIndex = pd.to_datetime(previousIndex)
DF_Data_RAW.index= NewparsedIndex

#Functions of lagged data 
def lag_column(df,column_name,lag_period=1):
    for i in range(1,lag_period+1,1):
        new_column_name = column_name+"-"+str(i*15)+"min"
        df[new_column_name]=df[column_name].shift(i)
    return df
    
def lag_columnall(df,lag_period=1):
    for column_name in df.columns.tolist():
        for i in range(1,36+1,1):
            new_column_name = column_name+"-"+str(i*15)+"min"
            df[new_column_name]=df[column_name].shift(i)
    return df

#Function of time features for the data 
def features_creation(df):
    # creatures time based features from pandas dataframe
    # such hour of day, weekday/weekend, day/night and so on
    # sin hour and cos hour as just indirect representation of time of day
    df['sin_hour'] = np.sin((df.index.hour)*2*np.pi/24)
    df['cos_hour'] = np.cos((df.index.hour)*2*np.pi/24)#later try 24 vector binary format
    df['hour'] = df.index.hour # 0 to 23
    df['day_of_week'] = df.index.dayofweek #Monday = 0, sunday = 6
    df['weekend'] = [ 1 if day in (5, 6) else 0 for day in df.index.dayofweek ] # 1 for weekend and 0 for weekdays
    df['month'] = df.index.month
    df['week_of_year'] = df.index.week
    # day = 1 if(10Hrs -19Hrs) and Night = 0 (otherwise)
    df['day_night'] = [1 if day<20 and day>9 else 0 for day in df.index.hour ]
    return df

#Data Lagged
DF_data_raw=DF_Data_RAW.drop("RECORD",axis=1)
DF_data_raw=DF_data_raw.astype("float64")
DF_data_raw_lagged=lag_columnall(DF_data_raw,36)

#correlation of data
correlation=DF_data_raw_lagged.corr()
DataFolderPath1 = "/Users/Marrugo/Dropbox/git_for_clone/Ac consumption prediction of ORNL Research House--MARRUGO--THOUVENIN/data_correlation.csv"
correlation.to_csv(DataFolderPath1,sep = ",") 

#Data selection
DF_TotalEnergy=DF_Data_RAW[["main_Tot"]]
DF_Input1=DF_Data_RAW.loc[:,"Outside_Tmp_Avg":"wind_speed_mean"]
DF_solar_radiation=DF_Data_RAW[["SlrW1_Avg"]]
DF_Precipitation=DF_Data_RAW[["Rain_in_Tot"]]
DF_dryer=DF_Data_RAW[["dryer_Tot"]]
DF_RH_lvl1=DF_Data_RAW[["LVL1_RH_Avg"]]
DF_RH_lvl2=DF_Data_RAW[["LVL2_RH_Avg"]]
DF_WallNava_RH=DF_Data_RAW[["WallNcav_RH_Avg"]]
DF_RoofN_RH=DF_Data_RAW[["RoofN_RH_Avg"]]
DF_Roofn_tmp=DF_Data_RAW[["RoofN_tmp_Avg"]]

#Data selected compilation
DF_Data_selected=DF_TotalEnergy.join([DF_Input1,DF_solar_radiation,DF_Roofn_tmp,DF_RoofN_RH,DF_Precipitation,DF_WallNava_RH,DF_dryer,DF_RH_lvl2,DF_RH_lvl1])
DF_Data_selected_clean = DF_Data_selected.dropna()

#Data conversion 
DF_Data_selected_float64=DF_Data_selected_clean.astype("float64")

#Changing the name of the columns 
Data_final=features_creation(DF_Data_selected_float64)
Data_final.rename(columns = {'main_Tot':"AC_consump"},inplace=True)
Data_final.rename(columns = {'Outside_Tmp_Avg':"Temperature_Avg"},inplace=True)
Data_final.rename(columns = {'Outside_RH_Avg':"Relative_Humidity"},inplace=True)
Data_final.rename(columns = {'SlrW1_Avg':"Solar_radiation"},inplace=True)
Data_final.rename(columns = {'Rain_in_Tot':"Precipitation"},inplace=True)
Data_final.rename(columns = {'dryer_Tot':"Dryer"},inplace=True)
Data_final.rename(columns = {'LVL1_RH_Avg':"RH_lvl1"},inplace=True)
Data_final.rename(columns = {'LVL2_RH_Avg':"RH_lvl2"},inplace=True)
Data_final.rename(columns = {'WallNcav_RH_Avg':"RH_roof_cavity"},inplace=True)
Data_final.rename(columns = {'RoofN_RH_Avg':"RH_roof_north"},inplace=True)
Data_final.rename(columns = {'RoofN_tmp_Avg':"RH_temp_notth"},inplace=True)
Data_final.head()

#solar radiation lagged data
Data_final["Solar_radiation-6.15 hrs"]=Data_final["Solar_radiation"].shift(25)
Data_final["Solar_radiation-6.3 hrs"]=Data_final["Solar_radiation"].shift(26)
Data_final["Solar_radiation-6.45 hrs"]=Data_final["Solar_radiation"].shift(27)

#temperature lagged data
Data_final["Temperature_Avg-5.15 hrs"]=Data_final["Temperature_Avg"].shift(21)
Data_final["Temperature_Avg-5.3 hrs"]=Data_final["Temperature_Avg"].shift(22)
Data_final["Temperature_Avg-5.45 hrs"]=Data_final["Temperature_Avg"].shift(23)

#relative humidity lagged data
Data_final["Relative_Humidity-5.0 hrs"]=Data_final["Relative_Humidity"].shift(20)
Data_final["Relative_Humidity-5.15 hrs"]=Data_final["Relative_Humidity"].shift(21)
Data_final["Relative_Humidity-5.3 hrs"]=Data_final["Relative_Humidity"].shift(22)

#precipitation lagged data 
Data_final["Precipitation-0.45 hrs"]=Data_final["Precipitation"].shift(3)
Data_final["Precipitation-3 hrs"]=Data_final["Precipitation"].shift(12)

#Roof north relative humidity lagged data 
Data_final["RH_roof_north-9 hrs"]=Data_final["RH_roof_north"].shift(36)
Data_final["RH_roof_north-9.15 hrs"]=Data_final["RH_roof_north"].shift(37)

#Roof north temperature lagged data 
Data_final["RH_temp_notth-9 hrs"]=Data_final["RH_temp_notth"].shift(36)
Data_final["RH_temp_notth-9.15 hrs"]=Data_final["RH_temp_notth"].shift(37)


#Final data for linear regression 
Data_final.dropna(inplace=True)
Data_final.head()
DF_target = Data_final["AC_consump"]
DF_features = Data_final.drop("AC_consump",axis=1)

#importing train test split 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(DF_features, DF_target, test_size=0.2, random_state=41234)

#importing random forest regression 
from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor()

#training random forest regression with cross validation 
from sklearn.model_selection import cross_val_predict
predict = cross_val_predict(reg,DF_features,DF_target,cv=25)
predictions = pd.Series(predict,index=DF_target.index).rename("AC_consump"+'_predicted')
predictions_frame = predictions.to_frame()
predictions_frame["AC_consump"]=DF_target

#Ploting linear regression results for a specifc data range
predictions_frame['2014-09-01 13:45:00':'2014-09-05 23:15:00'].plot()
plt.xlabel('Time')
plt.ylabel('AC Power [w]')
plt.ylim([0,1700])
plt.show()

#importing metrics for the data error 
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
R2_score = r2_score(predictions_frame["AC_consump"],predictions_frame["AC_consump_predicted"])
Mean_absolute=mean_absolute_error(predictions_frame["AC_consump"],predictions_frame["AC_consump_predicted"])
Mean_absolute_square=mean_squared_error(predictions_frame["AC_consump"],predictions_frame["AC_consump_predicted"])

#showing results of data error
print " The square error of the predicted data is "+str(R2_score)+"\n"
print " The mean absolute error of the predicted data is "+str(Mean_absolute)+"\n"
print " The mean absolute square error of the predicted data is "+str(Mean_absolute_square)+"\n"


