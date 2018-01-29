import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DataFolderPath="/Users/chenrujing/Dropbox/"
ConsumptionFileName= "building5retail.csv"
ConsumptionFilePath = DataFolderPath+"/"+ConsumptionFileName 

ConsumptionFileName = "building5retail.csv"
ConsumptionFilePath= DataFolderPath+"/"+ConsumptionFileName
DF_consumption = pd.read_csv(ConsumptionFilePath,sep=",",index_col=0)
previousIndex= DF_consumption.index
NewparsedIndex = pd.to_datetime(previousIndex)
DF_consumption.index= NewparsedIndex
DF_consumption.head(24)
DF_JulyfirstTillthird = DF_consumption["2010-07-01 00:00:00":"2010-07-03 23:00:00"]
DF_JulyfirstTillthird.head(5)
DF_JulyfirstTillthird.describe()

DF_JulyfirstTillthird.plot()
plt.xlabel('Timestamp ')
plt.ylabel('Power (kW)')
plt.show()

DF_consumption = DF_consumption.dropna()

# comparingtogether

df_chosen_dates = DF_consumption['2010-06-16':'2010-06-19']
df_chosen_dates_normalized = (df_chosen_dates- df_chosen_dates.min())/(df_chosen_dates.max()-df_chosen_dates.min())
plt.figure()
df_chosen_dates_normalized.plot()

#Graph2, compare in two graphs seperately
df_chosen_dates.plot()
fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)

df_chosen_dates.iloc[:,0].plot(ax=ax1,legend=True,color="b")
df_chosen_dates.iloc[:,1].plot(ax=ax2,legend=True,color="r")
ax1.set_ylabel(" OAT (F) ", color="b")
ax2.set_ylabel(" Power (kW)", color="r")
ax1.tick_params(axis='y',colors='b')
ax2.tick_params(axis='y',colors='r')


#lagged the data

def lag_column(df,column_name,lag_period=1):
    for i in range(1,lag_period+1,1):
        new_column_name = column_name+" -"+str(i)+"hr"
        df[new_column_name]=(df[column_name]).shift(i)
    return df

df_FinalDataSet_withLaggedFeatures = DF_consumption.copy()
df_FinalDataSet_withLaggedFeatures = lag_column(df_FinalDataSet_withLaggedFeatures,"OAT (F)",24)
df_FinalDataSet_withLaggedFeatures.head(24)
df_FinalDataSet_withLaggedFeatures.dropna(inplace=True)

#plot lagged correlation data
fig = plt.figure("Figure for Correlations between all the lagged temperature and power consumption")
plot = fig.add_axes()
plot = sns.heatmap(df_FinalDataSet_withLaggedFeatures.corr(), annot=False)
plot.xaxis.tick_top() 
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.show()

##From the plot, it doesn't show lag features, so the time matches.

#creatures creating

def features_creation(df):

    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek #
    df['weekend'] = [ 1 if day in (5, 6) else 0 for day in df.index.dayofweek ] 
    df['month'] = df.index.month
    df['week_of_year'] = df.index.week
    df['day_night'] = [1 if day<20 and day>9 else 0 for day in df.index.hour ]
    return df
    
df_FinalDataSet  = DF_consumption.copy()

df_FinalDataSet['hour'] = df_FinalDataSet.index.hour
df_FinalDataSet['day_of_week'] = df_FinalDataSet.index.dayofweek 
df_FinalDataSet['month'] = df_FinalDataSet.index.month
df_FinalDataSet['week_of_year'] = df_FinalDataSet.index.week
 
#List Comprehension   
def DayDetector(hour):
    dayLabel=1
    if (hour<20 and hour > 9):
        dayLabel = 1
    else:
        dayLabel = 0
    return dayLabel

df_FinalDataSet['day_night'] = [DayDetector(thisHour) for thisHour in df_FinalDataSet.index.hour] # day = 1 if(10Hrs -19Hrs) and Night = 0 (otherwise

def WeekendDetector(day):
    weekendLabel = 0
    if (day==5 or day==6):
        weekendLabel = 1
    else:
        weekendLabel=0
    return weekendLabel
df_FinalDataSet['weekend'] = [ WeekendDetector(thisDay) for thisDay in df_FinalDataSet.index.dayofweek ] # 1 for weekend and 0 for weekdays
df_FinalDataSet.head()
df_FinalDataSet.describe()

#Normalize the data
def normalize(df):
    return (df-df.min())/(df.max()-df.min())

DF_target = df_FinalDataSet["Power (kW)"]
DF_features = df_FinalDataSet.drop("Power (kW)",axis=1)

df_FinalDataSet_norm = normalize(df_FinalDataSet)
DF_target_norm = df_FinalDataSet_norm["Power (kW)"]
DF_features_norm = df_FinalDataSet_norm.drop("Power (kW)",axis=1)


#methoads of prediction

#Method1
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(DF_features, DF_target, test_size=0.2, random_state=41234)
X_train_norm, X_test_norm, y_train_norm, y_test_norm = train_test_split(DF_features_norm, DF_target_norm, test_size=0.2, random_state=41234)

from sklearn import linear_model
linear_reg = linear_model.LinearRegression()

linear_reg.fit(X_train, y_train)
predict_linearReg_split= linear_reg.predict(X_test)
predict_DF_linearReg_split=pd.DataFrame(predict_linearReg_split, index = y_test.index,columns=["Power_ConsPred_linearReg_split"])

predict_DF_linearReg_split = predict_DF_linearReg_split.join(y_test)

predict_DF_linearReg_split['2010-08-01':'2010-08-20'].plot()
plt.xlabel('Time')
plt.ylabel('Power [KW]')
plt.ylim([0,560])

import seaborn as sns
fig = plt.figure()
ax1 = fig.add_subplot(111)
plot = sns.regplot(x="Power (kW)", y="Power_ConsPred_linearReg_split",
                   data=predict_DF_linearReg_split,ax=ax1,
                   line_kws={"lw":3,"alpha":0.5})
plt.title('Actual power VS. Predicted power_linearReg_split')
plot.set_xlim([0,600])
plot.set_ylim([0,600])
plot.set_xlabel('Actual power')
plot.set_ylabel('Predicted power')
regline = plot.get_lines()[0];
regline.set_color('red')

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
metric_R2_score = r2_score(y_test,predict_linearReg_split)
metric_mean_absolute_error = mean_absolute_error(y_test,predict_linearReg_split)
metric_mean_squared_error = mean_squared_error(y_test,predict_linearReg_split)
coeff_variation = np.sqrt(metric_mean_squared_error)/y_test.mean()
print "coefficient_variation of linear regression split is " + str (coeff_variation) 

#Method2 Cross Validation
from sklearn.model_selection import cross_val_predict
predict_linearReg_CV = cross_val_predict(linear_reg,DF_features,DF_target,cv=10)
predict_DF_linearReg_CV=pd.DataFrame(predict_linearReg_CV, index = DF_target.index,columns=["Power_ConsPred_linearReg_CV"])
predict_DF_linearReg_CV = predict_DF_linearReg_CV.join(DF_target)
predict_DF_linearReg_CV['2010-08-01':'2010-08-20'].plot()

import seaborn as sns
fig = plt.figure()
ax1 = fig.add_subplot(111)
plot = sns.regplot(x="Power (kW)", y="Power_ConsPred_linearReg_CV",
                   data=predict_DF_linearReg_CV,ax=ax1,
                   line_kws={"lw":3,"alpha":0.5})
plt.title('Actual power VS. Predicted power_Cross Validation')
plot.set_xlim([0,600])
plot.set_ylim([0,600])
plot.set_xlabel('Actual power')
plot.set_ylabel('Predicted power')
regline = plot.get_lines()[0];
regline.set_color('red')

R2_score_linearReg_CV = r2_score(predict_DF_linearReg_CV["Power (kW)"],predict_DF_linearReg_CV["Power_ConsPred_linearReg_CV"])
mean_absolute_error_linearReg_CV = mean_absolute_error(predict_DF_linearReg_CV["Power (kW)"],predict_DF_linearReg_CV["Power_ConsPred_linearReg_CV"])
mean_squared_error_linearReg_CV = mean_squared_error(predict_DF_linearReg_CV["Power (kW)"],predict_DF_linearReg_CV["Power_ConsPred_linearReg_CV"])
coeff_variation_linearReg_CV = np.sqrt(metric_mean_squared_error)/predict_DF_linearReg_CV["Power (kW)"].mean()

# Method3 support vector machines
from sklearn.svm import SVR
SVR_reg = SVR(kernel='rbf',C=10,gamma=1)
predict_SVR_CV = cross_val_predict(SVR_reg,DF_features_norm,DF_target_norm,cv=10)
predict_DF_SVR_CV=pd.DataFrame(predict_SVR_CV, index = DF_target_norm.index,columns=["Power_ConsPred_SVR_CV"])
predict_DF_SVR_CV = predict_DF_SVR_CV.join(DF_target_norm).dropna()
predict_DF_SVR_CV['2010-08-01':'2010-08-20'].plot()

fig = plt.figure()
ax1 = fig.add_subplot(111)
plot = sns.regplot(x="Power (kW)", y="Power_ConsPred_SVR_CV",
                   data=predict_DF_SVR_CV,ax=ax1,
                   line_kws={"lw":3,"alpha":0.5})
plt.title('Actual power VS. Predicted power_SVR')
plot.set_xlim([0,1])
plot.set_ylim([0,1])
plot.set_xlabel('Actual power')
plot.set_ylabel('Predicted power')
regline = plot.get_lines()[0];
regline.set_color('red')

R2_score_DF_SVR_CV = r2_score(predict_DF_SVR_CV["Power (kW)"],predict_DF_SVR_CV["Power_ConsPred_SVR_CV"])
mean_absolute_error_SVR_CV = mean_absolute_error(predict_DF_SVR_CV["Power (kW)"],predict_DF_SVR_CV["Power_ConsPred_SVR_CV"])
mean_squared_error_SVR_CV = mean_squared_error(predict_DF_SVR_CV["Power (kW)"],predict_DF_SVR_CV["Power_ConsPred_SVR_CV"])
coeff_variation_SVR_CV = np.sqrt(mean_squared_error_SVR_CV)/predict_DF_SVR_CV["Power (kW)"].mean()

#Method 4
from sklearn.ensemble import RandomForestRegressor
reg_RF = RandomForestRegressor()
predict_RF_CV = cross_val_predict(reg_RF,DF_features,DF_target,cv=10)
predict_DF_RF_CV=pd.DataFrame(predict_RF_CV, index = DF_target.index,columns=["Power_ConsPred_RF_CV"])
predict_DF_RF_CV = predict_DF_RF_CV.join(DF_target).dropna()
predict_DF_RF_CV['2010-08-01':'2010-08-20'].plot()

fig = plt.figure()
ax1 = fig.add_subplot(111)
plot = sns.regplot(x="Power (kW)", y="Power_ConsPred_RF_CV",
                   data=predict_DF_RF_CV,ax=ax1,
                   line_kws={"lw":3,"alpha":0.5})
plt.title('Actual Power (kW) VS. Predicted Power (kW)_RF')
plot.set_xlim([0,600])
plot.set_ylim([0,600])
plot.set_xlabel('Actual Power (kW)')
plot.set_ylabel('Predicted Power (kW)')
regline = plot.get_lines()[0];
regline.set_color('red')

R2_score_DF_RF_CV = r2_score(predict_DF_RF_CV["Power (kW)"],predict_DF_RF_CV["Power_ConsPred_RF_CV"])
mean_absolute_error_DF_CV = mean_absolute_error(predict_DF_RF_CV["Power (kW)"],predict_DF_RF_CV["Power_ConsPred_RF_CV"])
mean_squared_error_DF_CV = mean_squared_error(predict_DF_RF_CV["Power (kW)"],predict_DF_RF_CV["Power_ConsPred_RF_CV"])
coeff_variation_DF_CV = np.sqrt(mean_squared_error_DF_CV)/predict_DF_RF_CV["Power (kW)"].mean()

print " The coefficient variation of fouth methods (linear regression split, cross validation, support vector machines and Random Forest Regressor) "+ " are" + str(coeff_variation) + "," + str(coeff_variation_linearReg_CV ) + "," + str(coeff_variation_SVR_CV ) + "," + str (coeff_variation_DF_CV )






