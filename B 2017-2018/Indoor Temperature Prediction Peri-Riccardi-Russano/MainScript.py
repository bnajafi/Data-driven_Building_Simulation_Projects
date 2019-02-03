# -*- coding: utf-8 -*-
#prediction of the indoor temperature of a house made with a linear regression model in function 
#of the outdoor temperature and the solar irradiance
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import Functions as func
import sys
reload(sys)
sys.setdefaultencoding("utf8")

os.chdir("C:\Users\Luca\Dropbox\Progetto Building system\Data-driven")
#importing the complete database
FullDatabase=pd.read_csv("Database_complete2.csv",sep=",",index_col=0)
#extracting the columns with the data needed to do the regression
IndoorTemperature=FullDatabase[["Temperature_Habitacion_Sensor"]] #target (depenent) variable
OutdoorTemperature=FullDatabase[["Temperature_Exterior_Sensor"]] #first independent variable
SolarIrradiance=FullDatabase[["Meteo_Exterior_Piranometro"]] #second independent variable

#negative values for solar irradiance don't have physical sense, so I set them to zero
SolarIrradiance[SolarIrradiance['Meteo_Exterior_Piranometro'] <0.0] = 0 

#The database containing the data needed for the prediction is built by joining the 3 databases that I created before
ChosenData=IndoorTemperature.join([OutdoorTemperature,SolarIrradiance])

#here I'm going to change the indeces of the dataframe to the format YYYY-MM-DD hh:mm:ss
previousIndex= ChosenData.index
NewparsedIndex = pd.to_datetime(previousIndex,dayfirst=True)
ChosenData.index= NewparsedIndex 

ChosenData.rename(columns = {'Temperature_Habitacion_Sensor':"IndoorTemperature [°C]","Temperature_Exterior_Sensor":"OutdoorTemperature [°C]","Meteo_Exterior_Piranometro":"SolarIrradiance [W/m2]"},inplace=True)
ChosenData=ChosenData.dropna()

ChosenData=func.features_creation(ChosenData,5,19) #see the script "Functions.py"

#now I'm going to plot 2 parts of the data in order to see if there's a lag between the temperatures and the irradiation

SmallDatabase=ChosenData["2012-03-15":"2012-03-17"]
fig = plt.figure("From 2012-03-15 to 2012-03-17")
ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2)
ax3= fig.add_subplot(3,1,3)
SmallDatabase.iloc[:,0].plot(ax=ax1,legend=False,color="b")
SmallDatabase.iloc[:,1].plot(ax=ax2,legend=False,color="r")
SmallDatabase.iloc[:,2].plot(ax=ax3,legend=False,color="g")
ax1.set_ylabel("Indoor Temperature [°C]", color="b")
ax2.set_ylabel("Outdoor Temperature [°C]", color="r")
ax3.set_ylabel("Solar Irradiation [W/m2]", color="g")
ax1.set_xlabel("")
ax2.set_xlabel("")
ax3.set_xlabel("")
ax1.tick_params(axis='y',colors='b')
ax2.tick_params(axis='y',colors='r')
ax3.tick_params(axis='y',colors='g')
fig.subplots_adjust(hspace=0.5)

SmallDatabase1=ChosenData["2012-04-4":"2012-04-6"]
fig = plt.figure("From 2012-04-4 to 2012-04-6")
ax1_1 = fig.add_subplot(3,1,1)
ax2_1 = fig.add_subplot(3,1,2)
ax3_1 = fig.add_subplot(3,1,3)
SmallDatabase1.iloc[:,0].plot(ax=ax1_1,legend=False,color="b")
SmallDatabase1.iloc[:,1].plot(ax=ax2_1,legend=False,color="r")
SmallDatabase1.iloc[:,2].plot(ax=ax3_1,legend=False,color="g")
ax1_1.set_ylabel("Indoor Temerature [°C]", color="b")
ax2_1.set_ylabel("Outdoor Temperature [°C]", color="r")
ax3_1.set_ylabel("Solar Irradiation [W/m2]", color="g")
ax1_1.set_xlabel("")
ax2_1.set_xlabel("")
ax3_1.set_xlabel("")
ax1_1.tick_params(axis='y',colors='b')
ax2_1.tick_params(axis='y',colors='r')
ax3_1.tick_params(axis='y',colors='g')
fig.subplots_adjust(hspace=0.5)

#from these two plots I can see that there's a periodical lag between the indoor temperature and the two other parameters, so I need to find a better correlation

Temperature_lag=ChosenData[["IndoorTemperature [°C]"]].join(ChosenData[["OutdoorTemperature [°C]"]])
Irradiance_lag=ChosenData[["IndoorTemperature [°C]"]].join(ChosenData[["SolarIrradiance [W/m2]"]])

Temperature_lag=func.lag_column(Temperature_lag,"OutdoorTemperature [°C]",20)
Temperature_lag_corr = Temperature_lag.corr()
#using the command .corr() applied to the database Temperature_lag, I find that the best correlation is achieved with a shift of 6 for the outdoor temperature.
#I can state this by looking to the Pearson's coefficients: the maximum value for this coefficient is reached with a shift of 6.

Irradiance_lag=func.lag_column(Irradiance_lag,"SolarIrradiance [W/m2]",24)
Irradiance_lag_corr = Irradiance_lag.corr()

#using the command .corr() applied to the database Irradiance_lag, I find that the best correlation is achieved with a shift of 19 for the solar irradiance
#So:
ChosenData["Outdoor Temp Shifted"]=ChosenData["OutdoorTemperature [°C]"].shift(6)
ChosenData["Irradiance Shifted"]=ChosenData["SolarIrradiance [W/m2]"].shift(19)

ShiftedDatabase=ChosenData[["IndoorTemperature [°C]"]].join([ChosenData[["Outdoor Temp Shifted"]],ChosenData[["Irradiance Shifted"]]])

#the final database will be the following (cleaned from all the lines with NaNs)
FinalDatabase=ShiftedDatabase.dropna()

#let's see the differences on a plot
SmallDatabase2=FinalDatabase["2012-03-15":"2012-03-17"]
fig = plt.figure("From 2012-03-15 to 2012-03-17 SHIFTED")
ax1_2 = fig.add_subplot(3,1,1)
ax2_2 = fig.add_subplot(3,1,2)
ax3_2 = fig.add_subplot(3,1,3)
SmallDatabase2.iloc[:,0].plot(ax=ax1_2,legend=False,color="b")
SmallDatabase2.iloc[:,1].plot(ax=ax2_2,legend=False,color="r")
SmallDatabase2.iloc[:,2].plot(ax=ax3_2,legend=False,color="g")
ax1_2.set_ylabel("Indoor Temperature [°C]", color="b")
ax2_2.set_ylabel("Outdoor Temperature [°C]", color="r")
ax3_2.set_ylabel("Solar Irradiation [W/m2]", color="g")
ax1_2.set_xlabel("")
ax2_2.set_xlabel("")
ax3_2.set_xlabel("")
ax1_2.tick_params(axis='y',colors='b')
ax2_2.tick_params(axis='y',colors='r')
ax3_2.tick_params(axis='y',colors='g')
fig.subplots_adjust(hspace=0.5)
#as observable from the plot, now we have a good correlation between the data: indeed the indoor temperature reaches its peak at the same time of outdoor temperature and solar irradiance

FinalDatabase_withLag = func.lag_column(FinalDatabase,"IndoorTemperature [°C]",24)
FinalDatabase_withLag.head(24)

FinalDatabase_withLag.dropna(inplace=True)

#Now I define two dataframes, one with the target data (indoor temperature) and the other one with the features
Target=FinalDatabase_withLag["IndoorTemperature [°C]"] #this is the dataframe with the dependent variable (indoor temperature)
Features=FinalDatabase_withLag.drop("IndoorTemperature [°C]",axis=1)# this is the dataframe with the variables the indoor temperature depends on

#normalized
TargetNorm=func.normalize(Target)
FeaturesNorm=func.normalize(Features)

#let's begin with the linear regression model
from sklearn.model_selection import train_test_split #the function train_test_split splits arrays or matrices into random train and test subsets
Xtrain, Xtest, Ytrain, Ytest = train_test_split(Features, Target, test_size=0.2, random_state=41234) #here I created two dataframes for the features (one to train the model and one to test it) and two for the target of the regession
XtrainNorm, XtestNorm, YtrainNorm, YtestNorm = train_test_split(FeaturesNorm, TargetNorm, test_size=0.2, random_state=41234)#they are the same dataframes but normalized

from sklearn import linear_model
linear_reg = linear_model.LinearRegression() 
linear_reg.fit(Xtrain, Ytrain)

LinearRegSplitPred= linear_reg.predict(Xtest)
LinearRegSplitPredDF=pd.DataFrame(LinearRegSplitPred, index = Ytest.index,columns=["IndoorTempPrediction"])

LinearRegSplitPredDF = LinearRegSplitPredDF.join(Ytest)

#this is a plot with the comparison between the actual values and the predicted ones
LinearRegSplitPredDF["2012-04-4":"2012-04-6"].plot()
plt.xlabel("Time")
plt.ylabel("Indoor Temperature")

#now I calculate the Rsquare,the mean absolute error and the mean squared error in order to verify the goodness of the model with the function Goodness
AccuracyLinReg=func.Accuracy(LinearRegSplitPredDF,"IndoorTemperature [°C]","IndoorTempPrediction","linear regression") #see the script Functions

#the partition of the available data into three sets drastically reduces the number of samples that can be used for training the model
#to see if the model is good under this point of view, we use a procedure called cross validation

from sklearn.model_selection import cross_val_predict
CVpredict = cross_val_predict(linear_reg,Features,Target,cv=10)
CVpredictDF=pd.DataFrame(CVpredict, index = Target.index,columns=["IndoorTempPredictCV"])

CVpredictDF = CVpredictDF.join(Target)

fig = plt.figure('Actual indoor temperature vs predicted indoor temperature')
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)
plot=sns.regplot(x="IndoorTemperature [°C]", y="IndoorTempPredictCV",data=CVpredictDF,ax=ax1,line_kws={"lw":3,"alpha":0.5})
plot.set_xlabel("Actual indoor temperature")
plot.set_ylabel("Predicted indoor temperature")
regline = plot.get_lines()[0]
regline.set_color('r')
CVpredictDF.plot(ax=ax2)
plt.xlabel("")
plt.ylabel("Indoor Temperature")
fig.subplots_adjust(hspace=0.5,bottom=0.15)

AccuracyCV=func.Accuracy(CVpredictDF,"IndoorTemperature [°C]","IndoorTempPredictCV","cross validation")

#support vector machines:
from sklearn.svm import SVR
SVR_reg = SVR(kernel='rbf',C=10,gamma=1)
PredictSVR_CV = cross_val_predict(SVR_reg,FeaturesNorm,TargetNorm,cv=10)
PredictDF_SVR_CV=pd.DataFrame(PredictSVR_CV, index = TargetNorm.index,columns=["IndoorTempNormSVR"])
PredictDF_SVR_CV = PredictDF_SVR_CV.join(TargetNorm)
PredictDF_SVR_CV=PredictDF_SVR_CV.dropna()

fig = plt.figure('Actual indoor temperature norm vs predicted indoor temperature norm')
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)
plot=sns.regplot(x="IndoorTemperature [°C]", y="IndoorTempNormSVR",data=PredictDF_SVR_CV,ax=ax1,line_kws={"lw":3,"alpha":0.5})
plot.set_xlabel("Actual indoor temperature norm")
plot.set_ylabel("Predicted indoor temperature norm")
regline = plot.get_lines()[0]
regline.set_color('r')
PredictDF_SVR_CV.plot(ax=ax2)
plt.xlabel("")
plt.ylabel("Indoor Temperature Norm")
fig.subplots_adjust(hspace=0.5,bottom=0.15)

AccuracySVR=func.Accuracy(PredictDF_SVR_CV,"IndoorTemperature [°C]","IndoorTempNormSVR","support vector machines")

#A random forest is a meta estimator that fits a number of classifying decision trees on various sub-samples of the dataset and use averaging 
#to improve the predictive accuracy and control over-fitting. The sub-sample size is always the same as the original input sample size
from sklearn.ensemble import RandomForestRegressor
reg_RF = RandomForestRegressor()
PredictRF_CV = cross_val_predict(reg_RF,Features,Target,cv=10)
PredictDF_RF_CV=pd.DataFrame(PredictRF_CV,index=Target.index,columns=["IndoorTempRF"])
PredictDF_RF_CV = PredictDF_RF_CV.join(Target)
PredictDF_RF_CV=PredictDF_RF_CV.dropna()

fig = plt.figure('Actual indoor temperature vs predicted indoor temperature with RandomForestRegressor')
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)
plot=sns.regplot(x="IndoorTemperature [°C]", y="IndoorTempRF",data=PredictDF_RF_CV,ax=ax1,line_kws={"lw":3,"alpha":0.5})
plot.set_xlabel("Actual indoor temperature")
plot.set_ylabel("Predicted indoor temperature")
regline = plot.get_lines()[0]
regline.set_color('r')
PredictDF_RF_CV.plot(ax=ax2)
plt.xlabel("")
plt.ylabel("Indoor Temperature")
fig.subplots_adjust(hspace=0.5,bottom=0.15)

AccuracyRF=func.Accuracy(PredictDF_RF_CV,"IndoorTemperature [°C]","IndoorTempRF","random forest regressor")






















