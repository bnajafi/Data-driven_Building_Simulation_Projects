import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DataFolderPath = "C:\Users\Karla\Documents\Notebooks\Project_Funes_Wong_Pedroza\datos_edificio.csv"
DF_temperature = pd.read_csv(DataFolderPath,sep=";",index_col=0)

previousIndex= DF_temperature.index
NewparsedIndex = pd.to_datetime(previousIndex)
DF_temperature.index= NewparsedIndex

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
columnName=["Temperature_Exterior_Sensor","Temperature_Habitacion_Sensor","Weather_Temperature","Meteo_Exterior_Piranometro",
"Temperature_Exterior_Sensor","Temperature_Habitacion_Sensor","Lighting_Habitacion_Sensor","Meteo_Exterior_Sol_Sud",
"Meteo_Exterior_Crepusculo"]
lag_column(DF_temperature,columnName,26)

#corr=DF_temperature.corr()
#corr.to_csv("C:\Users\TOSHIBA\Documents\Italia\Building\Proyecto\datos_corr.csv",sep=";")

df_FinalDataSet=DF_temperature
df_FinalDataSet.to_csv("C:\Users\Karla\Documents\Notebooks\Project_Funes_Wong_Pedroza\datos_final.csv",sep=";")

#df_FinalDataSet_withLaggedFeatures = df_FinalDataSet
#df_FinalDataSet_withLaggedFeatures['Temperature_Exterior_Sensor_7'] = df_FinalDataSet_withLaggedFeatures['Temperature_Exterior_Sensor'].shift(7)
#df_FinalDataSet_withLaggedFeatures['Temperature_Habitacion_Sensor_1'] = df_FinalDataSet_withLaggedFeatures['Temperature_Habitacion_Sensor'].shift(1)
#df_FinalDataSet_withLaggedFeatures['Weather_Temperature_9'] = df_FinalDataSet_withLaggedFeatures['Weather_Temperature'].shift(9)
#df_FinalDataSet_withLaggedFeatures['Temperature_Habitacion_Sensor_3'] = df_FinalDataSet_withLaggedFeatures['Temperature_Habitacion_Sensor'].shift(3)
#df_FinalDataSet_withLaggedFeatures['Temperature_Habitacion_Sensor_4'] = df_FinalDataSet_withLaggedFeatures['Temperature_Habitacion_Sensor'].shift(4)
#df_FinalDataSet_withLaggedFeatures['Temperature_Habitacion_Sensor_5'] = df_FinalDataSet_withLaggedFeatures['Temperature_Habitacion_Sensor'].shift(5)
#df_FinalDataSet_withLaggedFeatures['Meteo_Exterior_Piranometro_20'] = df_FinalDataSet_withLaggedFeatures['Meteo_Exterior_Piranometro'].shift(20)
#df_FinalDataSet_withLaggedFeatures['Temperature_Exterior_Sensor_18'] = df_FinalDataSet_withLaggedFeatures['Temperature_Exterior_Sensor'].shift(18)
#df_FinalDataSet_withLaggedFeatures['Temperature_Habitacion_Sensor_10'] = df_FinalDataSet_withLaggedFeatures['Temperature_Habitacion_Sensor'].shift(10)
#df_FinalDataSet_withLaggedFeatures['Lighting_Habitacion_Sensor_22'] = df_FinalDataSet_withLaggedFeatures['Lighting_Habitacion_Sensor'].shift(22)
#df_FinalDataSet_withLaggedFeatures['Lighting_Habitacion_Sensor_21'] = df_FinalDataSet_withLaggedFeatures['Lighting_Habitacion_Sensor'].shift(21)
#df_FinalDataSet_withLaggedFeatures['Meteo_Exterior_Sol_Sud_21'] = df_FinalDataSet_withLaggedFeatures['Meteo_Exterior_Sol_Sud'].shift(21)
#df_FinalDataSet_withLaggedFeatures['Meteo_Exterior_Crepusculo_21'] = df_FinalDataSet_withLaggedFeatures['Meteo_Exterior_Crepusculo'].shift(21)

#df_FinalDataSet_withLaggedFeatures.to_csv("C:\Users\TOSHIBA\Documents\Italia\Building\Proyecto\datos_final.csv",sep=";")
#corr_lagged.dropna(inplace=True)
#df_correlation = corr_lagged.corr() 

#df_FinalDataSet_withLaggedFeatures = lag_column(df_FinalDataSet_withLaggedFeatures,"Temp Dining",24)
#df_FinalDataSet_withLaggedFeatures.head(24)
df_FinalDataSet_withLaggedFeatures=df_FinalDataSet
# Now I should remove all the lines with a  NAN
df_FinalDataSet_withLaggedFeatures.dropna(inplace=True)

# Now let's choose the features columns and the target one 

def normalize(df):
    return (df-df.min())/(df.max()-df.min())

DF_target = df_FinalDataSet_withLaggedFeatures["Temperature_Comedor_Sensor"]
DF_features = df_FinalDataSet_withLaggedFeatures.drop("Temperature_Comedor_Sensor",axis=1)

df_FinalDataSet_withLaggedFeatures_norm = normalize(df_FinalDataSet_withLaggedFeatures)
DF_target_norm = df_FinalDataSet_withLaggedFeatures_norm["Temperature_Comedor_Sensor"]
DF_features_norm = df_FinalDataSet_withLaggedFeatures_norm.drop("Temperature_Comedor_Sensor",axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(DF_features, DF_target, test_size=0.2, random_state=41234)

from sklearn import linear_model
linear_reg = linear_model.LinearRegression()

linear_reg.fit(X_train, y_train)
predict_linearReg_split= linear_reg.predict(X_test)
predict_DF_linearReg_split=pd.DataFrame(predict_linearReg_split, index = y_test.index,columns=["Temperature_Comedor_Sensor_Predicted"])

predict_DF_linearReg_split = predict_DF_linearReg_split.join(y_test)

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
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

def print_metrics(cv,mbe,r2):
    print "coefficient of variance = {:.2f}".format(cv)
    print "Mean bias error = {:.2f}".format(mbe)
    print "R Squared = {:.3f}".format(r2)
    
def get_metrics(df):
    metrics = accuracy_metrics()
    cv = metrics.coeff_var(df,df.columns[1],df.columns[0])*100
    mbe = (metrics.mean_bias_err(df,df.columns[1],df.columns[0])*100)
    r2 = metrics.r2_score(df,df.columns[1],df.columns[0])
    print_metrics(cv,mbe,r2)
    return cv, mbe, r2

get_metrics(predict_DF_linearReg_split)


predict_DF_linearReg_split['13/06/2012 11:45':'11/09/2012 6:30'].plot()
plt.xlabel('Time')
plt.ylabel('Dining Room Temperature [C]')
plt.ylim([15,25])
plot.show()