# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
#from matplotlib import pylab as plott
import pandas as pd
DataFolder="C:\Users\Manuel\Documents\Polimi\Building systems\Project\Phyton"
DataSetName="energydata_complete.csv"
#read DF adn change index

completeDataPath=DataFolder+"/"+DataSetName
completeDF=pd.read_csv(completeDataPath,sep = ",",index_col=0)
previousIndex= completeDF.index
NewparsedIndex = pd.to_datetime(previousIndex)
completeDF.index= NewparsedIndex



#find the target and the correlation with target
DFtarget=completeDF[["Appliances"]]
completeDF.corr()
#From the correlation we can see the variables that are most correlated with Appliance
corrAppliance=completeDF.corr().head(1)
#Plot an heat map with correlation
plt.matshow(completeDF.corr())
plt.colorbar()
#As we can see, there is low correlation for Appliances, our model would be low quality
#Those variable are lights,RH1, T2,T3,T6,Tout,Windspeed. Let's find a correlation with shifted one

DF_temperature_out = completeDF[['T_out']]

def lag_column(df,column_name,lag_period=1):
    for i in range(1,lag_period+1,1):
        new_column_name = column_name+"-"+str(i)+" times 10 min"
        df[new_column_name]=df[column_name].shift(i)
    return df

DF_tout_lagged=lag_column(DF_temperature_out,'T_out',6)
DF_corr_Tout=DFtarget.join([DF_tout_lagged])

#lets try with lights
DF_lights = completeDF[['lights']]
DF_lights_lagged=lag_column(DF_lights,"lights",6)
DF_corr_lights=DFtarget.join([DF_lights_lagged])
DF_corr_lights.corr()
#till -6


#lets try with RH1
DF_RH1 = completeDF[['RH_1']]
DF_RH1_lagged=lag_column(DF_RH1,"RH_1",6)
DF_corr_RH1=DFtarget.join([DF_RH1_lagged])
DF_corr_RH1.corr()
#Correlation decreasing

#lets try with Appliance
DF_Appliances = completeDF[['Appliances']]
DF_Appliances_lagged=lag_column(DF_Appliances,"Appliances",12)
DF_Appliances_lagged.corr()
#unitl -12


#lets try with T2
DF_T2 = completeDF[['T2']]
DF_T2_lagged=lag_column(DF_T2,"T2",4)
DF_corr_T2=DFtarget.join([DF_T2_lagged])
DF_corr_T2.corr()
#until -4

#lets try with T3
DF_T3 = completeDF[['T3']]
DF_T3_lagged=lag_column(DF_T3,"T3",6)
DF_corr_T3=DFtarget.join([DF_T3_lagged])
DF_corr_T3.corr()
#Correlation decreasing

#lets try with T6
DF_T6 = completeDF[['T6']]
DF_T6_lagged=lag_column(DF_T6,"T6",6)
DF_corr_T6=DFtarget.join([DF_T6_lagged])
DF_corr_T6.corr()
#till 6

#lets try with Windspeed
DF_Windspeed = completeDF[['Windspeed']]
DF_Windspeed_lagged=lag_column(DF_Windspeed,"Windspeed",8)
DF_corr_Windspeed=DFtarget.join([DF_Windspeed_lagged])
DF_corr_Windspeed.corr()
#work with wind speed  to -8


#Join all DF and set the day of week and day n night
DF_FinalSet=DFtarget.join([DF_lights_lagged,completeDF[['RH_1']],DF_Appliances_lagged.drop(["Appliances"],axis=1),DF_T2_lagged,completeDF[['T3']],DF_T6_lagged,DF_Windspeed_lagged])
DF_FinalSet.head()


DF_FinalSet['hour']=DF_FinalSet.index.hour
DF_FinalSet['day_of_week']=DF_FinalSet.index.dayofweek
DF_FinalSet['month']=DF_FinalSet.index.month
DF_FinalSet['week_of_the_year']=DF_FinalSet.index.week

def weekendDetector(day):
    weekendLabel=0
    if(day == 5 or day == 6):
        weekendLabel=1
    else:
        weekendLabel=0
    return weekendLabel

def dayDetector(hour):
    dayLabel=1
    if(hour<20 and hour>9):
        dayLabel=1
    else:
        dayLabel=0
    return dayLabel


simpleVectorOfDays = [0,1,2,3,4,5,6]
weekendOrNotVector = [weekendDetector(thisDay) for thisDay in simpleVectorOfDays]


hoursOfDayVector= range(0,24,1)
dayOrNotVEctor =[dayDetector(ThisHour) for ThisHour in hoursOfDayVector]

DF_FinalSet["weekend"] = [weekendDetector(thisDay) for thisDay in DF_FinalSet.index.dayofweek]
DF_FinalSet["day_nigth"] = [dayDetector(thisHour) for thisHour in DF_FinalSet.index.hour]
DF_FinalSet.head()
DF_FinalSet.dropna(inplace=True)






#Lets try to build models
DF_features = DF_FinalSet.drop(["Appliances","day_of_week","month","week_of_the_year","weekend"],axis=1)
DF_target=DF_FinalSet[["Appliances"]]

#import sklearn to define test and train, test size is the fraction that will be test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split (DF_features,DF_target,test_size=0.2)

#First model is simple linear model
from sklearn import linear_model
linear_reg=linear_model.LinearRegression() #empty alghoritm that we fill with fit

#Fill the algorithm with best fit of X train and Y train, and with this predict the output of Y test
linear_reg.fit(X_train,Y_train)
predict_linearAppliances=linear_reg.predict(X_test)

#Lets put in a data frame
#How to extract index of Y_test---> Y_test.index
predict_DF_linearReg_split=pd.DataFrame(predict_linearAppliances,index =Y_test.index,columns =["AppliancesEnergy_predic_linearReg_split"])
predict_DF_linearReg_split=predict_DF_linearReg_split.join(Y_test)
#Now we have a DF in which we have predicted value and value of completeDF, let's see if the prediction is good: plot a period and see if hte curves match
predict_DF_linearReg_split_period=predict_DF_linearReg_split["2016-03-01":"2016-03-07"]

predict_DF_linearReg_split_period.plot()
plt.xlabel("time")
plt.ylabel("Appliances")
plt.ylim([0,800])
plt.title("Linear regression")


#lets find the metrics !!
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mean_squared_error_linearREG=mean_squared_error(Y_test,predict_linearAppliances)
mean_absolute_error_linearREG=mean_absolute_error(Y_test,predict_linearAppliances)
R2_score_linearReg= r2_score(Y_test,predict_linearAppliances)

#R2 value is 0.587432, not a very high number.


#Lets use cross validation --its already implemented in sklearn
from sklearn.model_selection import cross_val_predict

predict_linearReg_CV=cross_val_predict(linear_reg,DF_features,DF_target,cv =10)
#Lets put in a DF
predict_DF_linearReg_CV=pd.DataFrame(predict_linearReg_CV,index =DF_target.index,columns =["Appliances_predic_linearReg_CV"])
predict_DF_linearReg_CV=predict_DF_linearReg_CV.join(DF_target)  #use DF target instead of Y_test
predict_DF_linearReg_CV_period= predict_DF_linearReg_CV["2016-03-01":"2016-03-03"] 


predict_DF_linearReg_CV_period.plot()
plt.xlabel("time")
plt.ylabel("Appliances")
plt.ylim([0,800])
plt.title("Crossing validation")

#lets do the metrics
mean_squared_error_CV=mean_squared_error(DF_target,predict_linearReg_CV)
mean_absolute_error_CV=mean_absolute_error(DF_target,predict_linearReg_CV)
R2_score_CV= r2_score(DF_target,predict_linearReg_CV)

#Lets try with a more complex algorithm, we will use only cross validation

from sklearn.ensemble import RandomForestRegressor
reg_RF=RandomForestRegressor()

predict_RF_CV=cross_val_predict(reg_RF,DF_features,DF_target,cv =10) #heavy procedure

predict_DF_RF_CV=pd.DataFrame(predict_RF_CV,index =DF_target.index,columns =["Appliances_predic_RF_CV"])
predict_DF_RF_CV=predict_DF_RF_CV.join(DF_target)
predict_DF_RF_CV_period= predict_DF_RF_CV["2016-03-01":"2016-03-03"]
predict_DF_RF_CV_period.plot()
plt.xlabel("time")
plt.ylabel("Appliances")
plt.ylim([0,800])
plt.title("Regression line with RandomForestRegressor")

mean_squared_error_RF=mean_squared_error(DF_target,predict_RF_CV)
mean_absolute_error_RF=mean_absolute_error(DF_target,predict_RF_CV)
R2_score_RF= r2_score(DF_target,predict_RF_CV)

#Try now with Support Vector Regression
from sklearn.svm import SVR
reg_SVR = SVR(kernel='rbf',C=10,gamma=1)

def normalize(df):
    return (df-df.min())/(df.max()-df.min())

DF_features_norm=normalize(DF_features)
DF_target_norm=normalize(DF_target)
#Input for SVR should be normalized tables only

predict_SVR_CV = cross_val_predict(reg_SVR,DF_features_norm,DF_target_norm,cv=10) #very heavy procedure
predict_DF_SVR_CV=pd.DataFrame(predict_SVR_CV, index = DF_target_norm.index,columns=["AC_ConsPred_SVR_CV"])
predict_DF_SVR_CV = predict_DF_SVR_CV.join(DF_target_norm).dropna()

predict_DF_SVR_CV["2016-03-01":"2016-03-03"].plot()
plt.xlabel("time")
plt.ylabel("Appliances ratio")
plt.ylim([0,0.65])
plt.title("Regression normalized line with SVR")

mean_squared_error_SVR=mean_squared_error(predict_DF_SVR_CV[["Appliances"]],predict_DF_SVR_CV[['AC_ConsPred_SVR_CV']])
mean_absolute_error_SVR=mean_absolute_error(predict_DF_SVR_CV[["Appliances"]],predict_DF_SVR_CV[['AC_ConsPred_SVR_CV']])
R2_score_SVR= r2_score(predict_DF_SVR_CV[["Appliances"]],predict_DF_SVR_CV[['AC_ConsPred_SVR_CV']])


#This model fits badly the set, if you try to compute the regression with non-normalized features you would find 
#a model with an horizontal line that predicts better your target since R2 coefficient is very low

reg_SVR.fit(X_train,Y_train)
predict_SVR_Appliances=reg_SVR.predict(X_test)
predict_DF_SVR_split=pd.DataFrame(predict_SVR_Appliances,index =Y_test.index,columns =["AppliancesEnergy_predic_SVR_split"])
predict_DF_SVR_split=predict_DF_SVR_split.join(Y_test)
#Now we have a DF in which we have predicted value and value of completeDF, let's see if the prediction is good: plot a period and see if hte curves match
predict_DF_SVR_split_period=predict_DF_SVR_split["2016-03-01":"2016-03-07"]

predict_DF_SVR_split_period.plot()
plt.xlabel("time")
plt.ylabel("Appliances")
plt.ylim([0,800])
plt.title("Regression line with SVR")

#In the plot you can see the horizontal line

#In conclusion with linear and RF regression you can find a model that more or less fits good the data (R2 sligthly less
#than 0.6. With SVR, model is bad fitted. To increase accurancy of model you should find other variables more correlated
#with Appliances or other models that fit better dataset 
