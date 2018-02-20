import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_file(file_name):
    data_set = pd.read_csv(file_name,sep=',',index_col=0)
    desired_index = data_set.index
    datetime_index = pd.to_datetime(desired_index)
    data_set.index = datetime_index
    return data_set

def normalize(df):
    return (df - df.min())/(df.max()-df.min())

def weekend_detector(day):
    if (day==5 or day==6):
        weekend = 1
    else:
        weekend = 0
    return weekend

dataframe = read_file("/Users/camillatagliabue/Desktop/Project/energydatacomplete.csv")
dataframe.head()
dataframe['hour'] = dataframe.index.hour # creating hour column in the framework
dataframe['day_of_week'] = dataframe.index.dayofweek
dataframe['month'] = dataframe.index.month
dataframe['day_night'] = [1 if day<20 and day>9 else 0 for day in dataframe.index.hour ]

dataframe['weekend'] = [weekend_detector(s) for s in dataframe['day_of_week']]

def lag_column(df,column_names,lag_times=1):
#lag_period: number of steps to lag,usually postive 
#to include past values for current row 
    for column_name in column_names:
        column_name = str(column_name)
        for i in np.arange(1,lag_times+1,1):
            new_column_name = column_name+'_'+str(i)+'hour'
            df[new_column_name]=(df[column_name]).shift(i)
    return df

df_lagged = lag_column(dataframe,['Appliances'],lag_times=16)
df_lagged = lag_column(df_lagged,['T_out'],lag_times=4)
df_lagged = lag_column(dataframe,['RH_out'],lag_times=4)
df_lagged.dropna(inplace=True)

import seaborn as sns
fig = plt.figure()
plot = fig.add_axes()
plot = sns.heatmap(df_lagged.dropna().corr(), annot=False)
plot.xaxis.tick_top() 
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.show()

#let's define target and feature to check if there's a correlation btw variables
target = dataframe[['Appliances']] #output dependent variable
features = dataframe[[col for col in dataframe.columns if col not in ['Appliances']]]
# remove outliers
dataframe = dataframe.loc[(dataframe['Appliances']>900)]
dataframe=dataframe.drop(['lights','Visibility','Windspeed'],axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=123524)

from sklearn import linear_model
linear_reg = linear_model.LinearRegression()

linear_reg.fit(X_train,y_train)

prediction = linear_reg.predict(X_test)

predict_series = pd.Series(prediction.ravel(),index=y_test.index).rename('Prediction appliance')
joined = pd.DataFrame(predict_series).join(y_test).dropna()

plt.scatter(joined['Appliances'],joined['Prediction appliance'])
plt.figure()
lineStart = joined['Appliances'].min() 
lineEnd = joined['Prediction appliance'].max()  
plt.scatter(joined['Appliances'],joined['Prediction appliance'], color = 'k', alpha=0.5)
plt.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-', color = 'r')
plt.xlim(lineStart, lineEnd)
plt.ylim(lineStart, lineEnd)
plt.show()

from sklearn.metrics import r2_score
r2_score(joined['Appliances'],joined['Prediction appliance'])