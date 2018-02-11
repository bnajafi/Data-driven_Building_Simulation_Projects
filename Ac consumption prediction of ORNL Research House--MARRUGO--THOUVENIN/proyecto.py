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
        new_column_name = column_name+"-"+str(i)+"hour"
        df[new_column_name]=df[column_name].shift(i)
    return df
    
def lag_columnall(df,delay,lag_period=1):
    for column_name in df.columns.tolist():
        for i in range(1*delay,lag_period+delay-1,1):
            new_column_name = column_name+"-"+str(i)+"hour"
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
    df['minute'] = df.index.minute # 00 to 59
    df['day_of_week'] = df.index.dayofweek #Monday = 0, sunday = 6
    df['weekend'] = [ 1 if day in (5, 6) else 0 for day in df.index.dayofweek ] # 1 for weekend and 0 for weekdays
    df['month'] = df.index.month
    df['week_of_year'] = df.index.week
    # day = 1 if(10Hrs -19Hrs) and Night = 0 (otherwise)
    df['day_night'] = [1 if day<20 and day>9 else 0 for day in df.index.hour ]
    return df

#RAW data with time features
DF_Data_features=features_creation(DF_Data_RAW)
#Selecting the data of each hour
DF_Data_hour=DF_Data_features.loc[DF_Data_features['minute'] == 0]
#Deleting the time features
DF_data_raw_hour=DF_Data_hour.drop(["RECORD","sin_hour","hour","minute","cos_hour","day_of_week","weekend","month","week_of_year","day_night"],axis=1)
#creating copy for the data
DF_data_raw_24=DF_data_raw_hour.copy()
#Data lagged for 9 hours
DF_data_raw_hour_lagged=lag_columnall(DF_data_raw_hour,1,9)
#Data type conversion
DF_for_correlation=DF_data_raw_hour_lagged.astype("float64")


#correlation of data
correlation=DF_for_correlation.corr()
DataFolderPath1 = "/Users/Marrugo/Dropbox/git_for_clone/Ac consumption prediction of ORNL Research House--MARRUGO--THOUVENIN/data_correlation.csv"
correlation.to_csv(DataFolderPath1,sep = ",") 

#Data selection
DF_HP_energy=DF_data_raw_hour[["HP_out_Tot"]]
DF_TotalEnergy=DF_data_raw_hour[["main_Tot"]]
DF_Input1=DF_data_raw_hour.loc[:,"Outside_Tmp_Avg":"wind_speed_mean"]
DF_solar_radiation=DF_data_raw_hour[["SlrW1_Avg"]]
DF_Precipitation=DF_data_raw_hour[["Rain_in_Tot"]]
DF_RH_lvl1=DF_data_raw_hour[["LVL1_RH_Avg"]]
DF_RH_lvl2=DF_data_raw_hour[["LVL2_RH_Avg"]]
DF_WallNava_RH=DF_data_raw_hour[["WallNcav_RH_Avg"]]
DF_RoofN_RH=DF_data_raw_hour[["RoofN_RH_Avg"]]
DF_Roofn_tmp=DF_data_raw_hour[["RoofN_tmp_Avg"]]

#Data selected compilation
DF_Data_selected=pd.concat([DF_HP_energy,DF_TotalEnergy,DF_Input1,DF_solar_radiation,DF_Roofn_tmp,DF_RoofN_RH,DF_Precipitation,DF_WallNava_RH,DF_RH_lvl2,DF_RH_lvl1], axis=1)
DF_Data_selected_clean = DF_Data_selected.dropna()

#Data conversion 
DF_Data_selected_float64=DF_Data_selected_clean.astype("float64")

#Changing the name of the columns 
Data_final=features_creation(DF_Data_selected_float64)
Data_final.rename(columns = {'main_Tot':"AC_consump"},inplace=True)
Data_final.rename(columns = {'HP_out_Tot':"HP_consump"},inplace=True)
Data_final.rename(columns = {'Outside_Tmp_Avg':"Temperature_Avg"},inplace=True)
Data_final.rename(columns = {'Outside_RH_Avg':"Relative_Humidity"},inplace=True)
Data_final.rename(columns = {'SlrW1_Avg':"Solar_radiation"},inplace=True)
Data_final.rename(columns = {'Rain_in_Tot':"Precipitation"},inplace=True)
Data_final.rename(columns = {'LVL1_RH_Avg':"RH_lvl1"},inplace=True)
Data_final.rename(columns = {'LVL2_RH_Avg':"RH_lvl2"},inplace=True)
Data_final.rename(columns = {'WallNcav_RH_Avg':"RH_roof_cavity"},inplace=True)
Data_final.rename(columns = {'RoofN_RH_Avg':"RH_roof_north"},inplace=True)
Data_final.rename(columns = {'RoofN_tmp_Avg':"RH_temp_notth"},inplace=True)
Data_final.head()

#copying dataframe 
Data_final24=Data_final.copy()

#solar radiation lagged data
Data_final["Solar_radiation-9 hrs"]=Data_final["Solar_radiation"].shift(9)
Data_final["Solar_radiation-10 hrs"]=Data_final["Solar_radiation"].shift(10)
Data_final["Solar_radiation-8 hrs"]=Data_final["Solar_radiation"].shift(8)

#Energy consumption lagged data
Data_final["AC_consump-1 hrs"]=Data_final["AC_consump"].shift(1)
#Energy of heat pump lagged data
Data_final["HP_consump-1 hrs"]=Data_final["HP_consump"].shift(1)

#temperature lagged data
Data_final["Temperature_Avg-3 hrs"]=Data_final["Temperature_Avg"].shift(3)
Data_final["Temperature_Avg-2 hrs"]=Data_final["Temperature_Avg"].shift(2)
Data_final["Temperature_Avg-1 hrs"]=Data_final["Temperature_Avg"].shift(1)

#relative humidity lagged data
Data_final["Relative_Humidity-9 hrs"]=Data_final["Relative_Humidity"].shift(9)
Data_final["Relative_Humidity-8 hrs"]=Data_final["Relative_Humidity"].shift(8)
Data_final["Relative_Humidity-10 hrs"]=Data_final["Relative_Humidity"].shift(10)

#precipitation lagged data 
Data_final["Precipitation-5 hrs"]=Data_final["Precipitation"].shift(5)
Data_final["Precipitation-2 hrs"]=Data_final["Precipitation"].shift(2)

#Roof north relative humidity lagged data 
Data_final["RH_roof_north-1 hrs"]=Data_final["RH_roof_north"].shift(1)
Data_final["RH_roof_north-2 hrs"]=Data_final["RH_roof_north"].shift(2)

#Roof north temperature lagged data 
Data_final["RH_temp_notth-9 hrs"]=Data_final["RH_temp_notth"].shift(9)
Data_final["RH_temp_notth-8 hrs"]=Data_final["RH_temp_notth"].shift(8)

#Relative humidity of story 1 lagged data 
Data_final["RH_lvl1-1 hrs"]=Data_final["RH_lvl1"].shift(1)

#Relative humidity of story 2  lagged data 
Data_final["RH_lvl2-1 hrs"]=Data_final["RH_lvl2"].shift(1)

#Relative humidity of the roof cavity lagged data 
Data_final["RH_roof_cavity-3 hrs"]=Data_final["RH_roof_cavity"].shift(3)
Data_final["RH_roof_cavity-1 hrs"]=Data_final["RH_roof_cavity"].shift(1)

#Final data for linear regression 
Data_final.dropna(inplace=True)
Data_final.head()
DF_target = Data_final["HP_consump"]
DF_features = Data_final.drop("HP_consump",axis=1)

#importing train test split 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(DF_features, DF_target, test_size=0.2, random_state=41234)

#importing random forest regression 
from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor()

#training random forest regression with cross validation 
from sklearn.model_selection import cross_val_predict
predict = cross_val_predict(reg,DF_features,DF_target,cv=25)
predictions = pd.Series(predict,index=DF_target.index).rename("HP_consump"+'_predicted')
predictions_frame = predictions.to_frame()
predictions_frame["HP_consump"]=DF_target

#Ploting linear regression results for a specifc data range
predictions_frame['2014-09-01 13:45:00':'2014-09-05 23:15:00'].plot()
plt.xlabel('Time')
plt.title("1 hr Prediction")
plt.ylabel('AC Power [w]')
plt.ylim([0,700])
plt.show()

#importing metrics for the data error 
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
R2_score = r2_score(predictions_frame["HP_consump"],predictions_frame["HP_consump_predicted"])
Mean_absolute=mean_absolute_error(predictions_frame["HP_consump"],predictions_frame["HP_consump_predicted"])
Mean_absolute_square=mean_squared_error(predictions_frame["HP_consump"],predictions_frame["HP_consump_predicted"])

#showing results of data error
print " The square error of the predicted data for 1 hr  is "+str(R2_score)+"\n"
print " The mean absolute error of the predicted data for 1 hr is "+str(Mean_absolute)+"\n"
print " The mean absolute square error of the predicted data for 1 hr is "+str(Mean_absolute_square)+"\n"


#Same calculation for 24 hrs

#delay
delay=24

#Creating 24 hrs lagged data
DF_data_raw_24h=lag_columnall(DF_data_raw_24,delay,24)
DF_data_raw_24h.head()

#Data type conversion
DF_for_correlation24=DF_data_raw_24h.astype("float64")

#correlation of data
correlation24=DF_for_correlation24.corr()
DataFolderPath2 = "/Users/Marrugo/Dropbox/git_for_clone/Ac consumption prediction of ORNL Research House--MARRUGO--THOUVENIN/data_correlation24.csv"
correlation24.to_csv(DataFolderPath2,sep = ",") 

#solar radiation lagged data
Data_final24["Solar_radiation-36 hrs"]=Data_final24["Solar_radiation"].shift(36)
Data_final24["Solar_radiation-35 hrs"]=Data_final24["Solar_radiation"].shift(35)
Data_final24["Solar_radiation-34 hrs"]=Data_final24["Solar_radiation"].shift(34)

#Energy consumption lagged data
Data_final24["AC_consump-24 hrs"]=Data_final24["AC_consump"].shift(delay)
#Energy of heat pump lagged data
Data_final24["HP_consump-24 hrs"]=Data_final24["HP_consump"].shift(delay)

#temperature lagged data
Data_final24["Temperature_Avg-26 hrs"]=Data_final24["Temperature_Avg"].shift(26)
Data_final24["Temperature_Avg-25 hrs"]=Data_final24["Temperature_Avg"].shift(25)
Data_final24["Temperature_Avg-24 hrs"]=Data_final24["Temperature_Avg"].shift(24)

#relative humidity lagged data
Data_final24["Relative_Humidity-39 hrs"]=Data_final24["Relative_Humidity"].shift(39)
Data_final24["Relative_Humidity-38 hrs"]=Data_final24["Relative_Humidity"].shift(38)
Data_final24["Relative_Humidity-40 hrs"]=Data_final24["Relative_Humidity"].shift(40)

#precipitation lagged data 
Data_final24["Precipitation-45 hrs"]=Data_final24["Precipitation"].shift(45)
Data_final24["Precipitation-46 hrs"]=Data_final24["Precipitation"].shift(46)

#Roof north relative humidity lagged data 
Data_final24["RH_roof_north-24 hrs"]=Data_final24["RH_roof_north"].shift(24)
Data_final24["RH_roof_north-25 hrs"]=Data_final24["RH_roof_north"].shift(25)

#Roof north temperature lagged data 
Data_final24["RH_temp_notth-35 hrs"]=Data_final24["RH_temp_notth"].shift(35)
Data_final24["RH_temp_notth-34 hrs"]=Data_final24["RH_temp_notth"].shift(34)

#Relative humidity of story 1 lagged data 
Data_final24["RH_lvl1-24 hrs"]=Data_final24["RH_lvl1"].shift(24)

#Relative humidity of story 2  lagged data 
Data_final24["RH_lvl2-24 hrs"]=Data_final24["RH_lvl2"].shift(24)

#Relative humidity of the roof cavity lagged data 
Data_final24["RH_roof_cavity-29 hrs"]=Data_final24["RH_roof_cavity"].shift(29)
Data_final24["RH_roof_cavity-28 hrs"]=Data_final24["RH_roof_cavity"].shift(28)

#Final data for linear regression 
Data_final24.dropna(inplace=True)
Data_final24.head()
DF_target24 = Data_final24["HP_consump"]
DF_features24 = Data_final24.drop("HP_consump",axis=1)

#importing train test split 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(DF_features24, DF_target24, test_size=0.2, random_state=41234)

#importing random forest regression 
from sklearn.ensemble import RandomForestRegressor
reg2 = RandomForestRegressor()

#training random forest regression with cross validation 
from sklearn.model_selection import cross_val_predict
predict24 = cross_val_predict(reg2,DF_features24,DF_target24,cv=25)
predictions24 = pd.Series(predict24,index=DF_target24.index).rename("HP_consump"+'_predicted')
predictions_frame24 = predictions24.to_frame()
predictions_frame24["HP_consump"]=DF_target24

#Ploting linear regression results for a specifc data range
predictions_frame24['2014-09-01 13:45:00':'2014-09-05 23:15:00'].plot()
plt.xlabel('Time')
plt.title("24 hrs Prediction")
plt.ylabel('AC Power [w]')
plt.ylim([0,500])
plt.show()

#importing metrics for the data error 
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
R2_score = r2_score(predictions_frame24["HP_consump"],predictions_frame24["HP_consump_predicted"])
Mean_absolute=mean_absolute_error(predictions_frame24["HP_consump"],predictions_frame24["HP_consump_predicted"])
Mean_absolute_square=mean_squared_error(predictions_frame24["HP_consump"],predictions_frame24["HP_consump_predicted"])

#showing results of data error
print " The square error of the predicted data for 24 hrs is "+str(R2_score)+"\n"
print " The mean absolute error of the predicted data for 24 hrs is "+str(Mean_absolute)+"\n"
print " The mean absolute square error of the predicted data for 24 hrs is "+str(Mean_absolute_square)+"\n"
