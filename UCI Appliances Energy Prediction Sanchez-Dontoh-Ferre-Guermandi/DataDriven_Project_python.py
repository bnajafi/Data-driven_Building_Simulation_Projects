# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#First of all, we create the path for the file containing the data we need. Once done, we convert it into a Data Frame
DataFolderPath="/Users/federico/Downloads"
DataFileName = "energydata_complete.csv"
DataFilePath= DataFolderPath+"/"+DataFileName
DF_data = pd.read_csv(DataFilePath,sep=",",index_col=0)
previousIndex= DF_data.index
NewparsedIndex = pd.to_datetime(previousIndex)
DF_data.index= NewparsedIndex
DF_data.head(24)
#the data frame is big, and it considers a period of one year. We could just take into account a shorter period
DF_period = DF_data["2016-02-10 00:00:00":"2016-02-15 00:00:00"]
#Now: the first column is called "Appliances". Then, we find several factors, mainly regarding the climate into the house, but also about the atmospheric 
#conditions in a close location. 
#We are going to consider the energy consmption as dependent variable, that is, our aim is to find a correlation between the energy consumption
#("Appliances") and the data we have at our disposal.
DF_appliance=DF_period['Appliances']
DF_appliance.head(5)
DF_appliance.describe() 
plt.figure()
plt.plot(DF_appliance)
DF_appliance.plot()
plt.xlabel('Days')
plt.ylabel('Energy consumption [Wh]')
plt.title("energy consumption during the days")
plt.show()
#Obviously, it is unlikely that all the data are suitable to find a correletion. First of all, 
#then, we have to find suitable features, that will be later employed to predict, through a machine learning algorithm, the energy consumption.
#We can see that there exists features that follow the consumption profile (for example, T_out. We can imagine that the colder it is, the bigger is the amount
#of energy used by the house, in fact during the hours dark the temperature will be lower and the illumination devices will be turned on). Other features' profile,
#on the other hand, will be less similar: those features are probably not closely related to the consumption. The column RH_3 (that is the humidity in laundry)
#can be taken as example
DF_appliance=DF_period['Appliances'].to_frame()
DF_T_out=DF_period['T_out'].to_frame()
DF_rh=DF_period['RH_3'].to_frame()
DFgraph=DF_appliance.join(DF_T_out)
DFgraph=DFgraph.join(DF_rh)


df_chosen_dates_normalized = (DFgraph- DFgraph.min())/(DFgraph.max()-DFgraph.min())

plt.figure()
df_chosen_dates_normalized.plot()
plt.title("comparison between consumption, external temperature and RH in bathroom (normalized)")
#This graph is a bit chaotic, we will plot only energy consumption and external temperature, in the same graph but with
#different scales
DFgraph=DFgraph.drop('RH_3',1)
plt.figure("profile of energy consumption and external temperature")

fig = plt.figure()

ax1 = fig.add_subplot(111) # axis for consumption
ax2 = ax1.twinx()          # axis for T_out
consum_col='Appliances'

DFgraph.plot(ax=ax1, y=consum_col, legend=False,color='b')
ax1.set_ylabel('Consumption',color='b')
ax1.tick_params(axis='y', colors='b')

DFgraph.plot(ax=ax2, y='T_out', legend=False, color='g')
ax2.set_ylabel('external Temperature deg C',color='g')
ax2.tick_params(axis='y',colors='g')

ax1.set_xlabel('Time')
plt.title("profile of energy consumption and external temperature")
plt.show()
#What we see is that the overall profile of the temperature can be asociated with the one of consumption, in fact the former's peaks coincide with the
#consumnption's peak. Obviously, they do no match perfectly

#Now, by means of the appropriate command, we will search correlation between column of our data frame. The command .corr() will give us a data frame containing values
#from -1 to 1: te closer to 1 is the value, the stronger is the correlation. We can plot the result to see the spread of the value. 

df_correlation = DF_data.corr()
fig = plt.figure()
mask = np.zeros_like(df_correlation)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    ax = sns.heatmap(df_correlation, mask=mask, vmax=.3, square=True,annot=False, fmt="d")
plt.title("scheme representing the degree of correlation")
plt.show()
ax.xaxis.tick_bottom() 
plt.yticks(rotation=0)
plt.xticks(rotation=90)
#We see that there are high values of correlation between temperatures and relative humidity, but unfortunatelly, in the first column, that is the one
#concerning energy consumption, that is what we are interested in, the values are low.
#We need to cream off those values and consider the highest ones (that are quite low anyway). Therefore firstly we will consider only the column realted
#with Appliances, then we will take into account only the higher than 0.05

df_schifo=df_correlation['Appliances']>0.05
df_schifo=df_schifo[df_schifo]
print df_schifo
#Those are the features whose tax of correlation with aplliances energy consumption is higher. They amount of features is low, therefore we will
#try to find other suitable features by means of the following functions. They do the same, that is shifting of a number that we can decide
#the values of a specific column. What we do here, then, is to shift of 10 the column of the suitable variables we found previously, cotained in 
#"nicefeatures". Then we will look for other correlations.
nicefeatures= df_schifo.index.tolist()
nicefeatures.pop(0)
DFF=DF_period
DF_period=DFF.copy()
def lag_column(df,column_names,lag_period=1):
#df              > pandas dataframe
#column_names    > names of column/columns as a list
#lag_period      > number of steps to lag ( +ve or -ve) usually postive 
#to include past values for current row 
    for column_name in column_names:
        column_name = [str(column_name)]
        for i in np.arange(1,lag_period+1,1):
            new_column_name = [col +'_'+str(i) for col in column_name]
            df[new_column_name]=(df[column_name]).shift(i)
    return df
    
def lagcolumn(df,column_name,lag_period=1):
    for i in range(1,lag_period+1,1):
        new_column_name = column_name+"-"+str(i)+"0 min"
        df[new_column_name]=df[column_name].shift(i)
    return df
 
lag_column(DFF, nicefeatures,10)
DFF=DFF.dropna()
df_correlation_bis = DFF.corr()
#Now we have a new dataframe, with columns of shifted values of thefeatures contained in nicefeatures containing values of the correlation.
#Again, we will take into account the highest values
df_schifo_bis=df_correlation_bis['Appliances']>0.03
df_schifo_bis=df_schifo_bis[df_schifo_bis]
verynicefeatures= df_schifo_bis.index.tolist()
print verynicefeatures
#Those are the features whose correlation with energy consumption is higher

#let's now create a data frame with the following columns:
#appliances and shifted appliances
#lights and lagged lights
#RH_1 and lagged RH_1
#T2 and lagged T2
#T6 and lagged T6
#T_out and lagged T_out
#windspeed and lagged windspeed


pd.options.mode.chained_assignment = None
DFAppl=lagcolumn(DF_data[['Appliances']],'Appliances',10)
DFlights=lagcolumn(DF_data[['lights']],'lights',10)
DFRH1=lagcolumn(DF_data[['RH_1']],'RH_1',10)
DFT6=lagcolumn(DF_data[['T6']],'T6',10)
DFT_out=lagcolumn(DF_data[['T_out']],'T_out',10)
DFwind=lagcolumn(DF_data[['Windspeed']],'Windspeed',10)
DFtotal=lagcolumn(DF_data[['Appliances']],'Appliances',10).join([lagcolumn(DF_data[['lights']],'lights',10)])
DFtotal=DFtotal.join(DFRH1)
DFtotal=DFtotal.join(DFT6)
DFtotal=DFtotal.join(DF_data[['T8']])
DFtotal=DFtotal.join(DFT_out)
DFtotal=DFtotal.join(DFwind)
DF_total=DFtotal.join(DF_data[['Visibility']])
DF_total.dropna(inplace=True)
DF_target = DF_total[["Appliances"]]
DF_features = DF_total.drop("Appliances",axis=1)
print ("DF index: ")+str(list(DF_features ))

#NaN are undesired, so we drop them
DF_total.dropna(inplace=True)
DF_target = DF_total[["Appliances"]]
DF_features = DF_total.drop("Appliances",axis=1)
print ("DF index: ")+str(list(DF_features ))
#it is now time to create a training dataset for machine learning and provide an independent testset which follows same probabilistic distribution of training
#The sklearn.model_selection module contains a machine learning algorithm which enable us to implement the linear regression method.
#We will plot the real value of consumption and the predicted one in order to make a comparison
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(DF_features, DF_target, test_size=0.2, random_state=41234)

from sklearn import linear_model
linear_reg = linear_model.LinearRegression()

linear_reg.fit(X_train, y_train)
predict = linear_reg.predict(X_test)
predictions = pd.Series(predict.ravel(),index=y_test.index).rename("energy_consumption"+"_predicted")
predictions_frame = pd.DataFrame(predictions).join(y_test).dropna()
predictions_frame["2016-02-10 00:00:00":"2016-02-15 00:00:00"].plot()
plt.xlabel('Time')
plt.ylabel('Energy consumption [Wh]')
plt.title("effective energy consumption VS predicted")
plt.ylim([0,500])
#We see that our prediction, although it does not match perfectly, follows the consumption profile: the peak of consuption are quite well predicted.
#There exist parameters that can objectively evaluate the quality of our predictions. Those are:
#The R2 score, that is ....  and acquires a value between 0 and 1. 1 means that our predictions perfectly match consuption, good estimates are obtained
#   with values around ...
#mean absolute error is ...
#mean squared error is ...
class accuracy_metrics:
    def coeff_var(self,df,actual_col,predicted_col):
        y_actual_mean = df[actual_col].mean()
        mse = mean_squared_error(df[actual_col],df[predicted_col])
        return np.sqrt(mse)/y_actual_mean
    def mean_bias_err(self,df,actual_col,predicted_col):
        y_actual_mean = df[actual_col].mean()
        return mean_absolute_error(df[actual_col],df[predicted_col])/y_actual_mean
    def r2_score(self,df,actual_col,predicted_col):
        return r2_score(df[actual_col],df[predicted_col])
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mean_squared_error_linearREG=mean_squared_error(y_test,predict)
mean_absolute_error_linearREG=mean_absolute_error(y_test,predict)
R2_score_linearReg= r2_score(y_test,predict)
print "Mean squared error is " + str(mean_squared_error_linearREG)
print "Mean absoulute error is " + str(mean_absolute_error_linearREG)
print "R2 score is " + str(R2_score_linearReg)
# In order to check if, even though our accuracy metrics values are not that high, our prediction is plausible, we create a scatter plot. What we see is
#that the line that represents  our prediction is contained into the region identified by the points
fig, ax = plt.subplots()
ax.scatter(predictions_frame['Appliances'], predictions_frame['energy_consumption_predicted'], edgecolors=(0, 0, 0))

ax.plot([predictions_frame['Appliances'].min(), predictions_frame['Appliances'].max()], [predictions_frame['Appliances'].min(), predictions_frame['Appliances'].max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.title("measured and predicted: general trend")
plt.show()

#Our R2 score is not that high, and also the errors' values are quite high.
#Now, in order to evaluate the quality of the model we developed, we employ the cross-validation approach
#The function we call (cross_val_predict) generate cross-validated estimates for each input data point
from sklearn.model_selection import cross_val_predict
predict_linearReg_CV = cross_val_predict(linear_reg,DF_features,DF_target,cv=10)
predict_linearReg_CV=pd.DataFrame(predict_linearReg_CV, index = DF_target.index,columns=["prediction with CV"])
predict_linearReg_CV = predict_linearReg_CV.join(DF_target)
predict_linearReg_CV_period=predict_linearReg_CV["2016-02-10 00:00:00":"2016-02-15 00:00:00"]

# cross_val_predict returns an array of the same size as `DF_features` where each entry
# is a prediction obtained by cross validation. We can plot a scatter plot  that enables us to compare the measured values
#and the estimated trend.
#(source: http://scikit-learn.org/stable/auto_examples/plot_cv_predict.html#sphx-glr-auto-examples-plot-cv-predict-py)
target=predict_linearReg_CV['Appliances']
estimated=predict_linearReg_CV['prediction with CV']
fig, ax = plt.subplots()
ax.scatter(target, estimated, edgecolors=(0, 0, 0))
ax.plot([target.min(), target.max()], [target.min(), target.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.title("comparison measured-predicted")
plt.show()
#We calculate again the values
"""
mean_squared_errorCV=mean_squared_error(DF_target,estimated)
mean_absolute_errorCV=mean_absolute_error(DF_target,estimated)
R2_scoreCV= r2_score(DF_target,estimated)
print "Mean squared error is " + str(mean_squared_errorCV)
print "Mean absoulute error is " + str(mean_absolute_errorCV)
print "R2 score is " + str(R2_scoreCV)
print " "

#We print the difference with the accuracy metrics values og linear regression. Are those values better?
print "Mean squared error difference " + str(mean_squared_errorCV-mean_squared_error_linearREG)
print "Mean absoulute error difference " + str(mean_absolute_errorCV-mean_absolute_error_linearREG)
print "R2 score difference " + str(R2_scoreCV-R2_score_linearReg)
#We lost some accuracy
"""

#Now we implement the Random Forest Regressor, that is "a meta estimator that fits a number of classifying decision trees on various sub-samples 
#of the dataset and use averaging to improve the predictive accuracy and control over-fitting. 
#The sub-sample size is always the same as the original input sample size but the samples are drawn with replacement if bootstrap=True (default).
#(source: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
from sklearn.ensemble import RandomForestRegressor
reg_RF=RandomForestRegressor()
predict = cross_val_predict(reg_RF,DF_features,DF_target,cv=10)
predictions = pd.Series(predict,index=DF_target.index).rename('Appliances_predic_RF_CV')
predictions_frame = pd.DataFrame(predictions).join(DF_target).dropna()
predicted_period=predictions_frame["2016-02-10 00:00:00":"2016-02-15 00:00:00"]
predicted_period.plot()
plt.xlabel("time")
plt.ylabel("energy consumption")
plt.ylim([0,800])
plt.title("Regression line with RandomForestRegressor")
plt.show()
meansquare_RF=mean_squared_error(DF_target,predict)
meanabs_error_RF=mean_absolute_error(DF_target,predict)
R2score_RF= r2_score(DF_target,predict)
print "Mean squared error is " + str(meansquare_RF)
print "Mean absoulute error is " + str(meanabs_error_RF)
print "R2 score is " + str(R2score_RF)

#We print the difference with the accuracy metrics values of linear regression. Are those values better?
print "Mean squared error difference " + str(meansquare_RF-mean_squared_error_linearREG)
print "Mean absoulute error difference " + str(meanabs_error_RF-mean_absolute_error_linearREG)
print "R2 score difference " + str(R2score_RF-R2_score_linearReg)
#NO



#Let's try to sum up what we have done until now. Basically, we organized our dataframe that contaide all the data and found some features that allowed
#us to establish a correlation with the energy consumption. The main problem is that we had just a few feasible features, so we tried to shift the values of some
#of our features in order to find some greature correlation rate. We chose only the values greater than 0.1. What if we increase this minimum value and
#include in DF_total only those specific features? We expect less values to calculate the correlation but easier to correlate.
#Once created again DF_total, the commands are identical to the previous part
df_schifo_bis=df_correlation_bis['Appliances']>0.15
df_schifo_bis=df_schifo_bis[df_schifo_bis]
verynicefeatures= df_schifo_bis.index.tolist()
print verynicefeatures
#We insert the column of the lights shifted of 2, the columns T_6 and T_out

DFAppl=lagcolumn(DF_data[['Appliances']],'Appliances',10)
DFtotal=DFAppl.join([lagcolumn(DF_data[['lights']],'lights',3)])
DFtotal=DFtotal.join(DFT6)
DFtotal=DFtotal.join(DFT_out)
DFtotal.dropna(inplace=True)
DFtarget = DFtotal[["Appliances"]]
DFfeatures = DFtotal.drop("Appliances",axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(DFfeatures, DFtarget, test_size=0.2, random_state=41234)

from sklearn import linear_model
linear_reg = linear_model.LinearRegression()

linear_reg.fit(X_train, y_train)
predict = linear_reg.predict(X_test)
predictions = pd.Series(predict.ravel(),index=y_test.index).rename("energy_consumption"+"_predicted")
predictions_frame = pd.DataFrame(predictions).join(y_test).dropna()
predictions_frame["2016-02-10 00:00:00":"2016-02-15 00:00:00"].plot()
plt.xlabel('Days')
plt.ylabel('Energy consumption [Wh]')
plt.title("energy consumption during the days")
plt.ylim([0,500])

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mean_squared_error_linearREG_2=mean_squared_error(y_test,predict)
mean_absolute_error_linearREG_2=mean_absolute_error(y_test,predict)
R2_score_linearReg_2= r2_score(y_test,predict)
print "Mean squared error is " + str(mean_squared_error_linearREG_2)
print "Mean absoulute error is " + str(mean_absolute_error_linearREG_2)
print "R2 score is " + str(R2_score_linearReg_2)
print " "
print "the Mean squared error difference is " + str(mean_squared_error_linearREG_2-mean_squared_error_linearREG)
print "Mean absoulute error is " + str(mean_absolute_error_linearREG_2-mean_absolute_error_linearREG)
print "R2 score is " + str(R2_score_linearReg_2-R2_score_linearReg)
#We see that the difference is null
"""
fig, ax = plt.subplots()
ax.scatter(predictions_frame['Appliances'], predictions_frame['energy_consumption_predicted'], edgecolors=(0, 0, 0))

ax.plot([predictions_frame['Appliances'].min(), predictions_frame['Appliances'].max()], [predictions_frame['Appliances'].min(), predictions_frame['Appliances'].max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
"""