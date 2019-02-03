import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 

#First i have to import the data that we wish to analize from the file.csv

ExternalFilesFolder =  r"C:\Users\LoruPortatile\Desktop\clima 2\project Pyton"
DataFileName= "EnergyProjectData.csv"
path_EnergyProjectData= os.path.join(ExternalFilesFolder,DataFileName)
DF_EnergyProject = pd.read_csv(path_EnergyProjectData,sep=",", index_col=0)

DF_EnergyProject.head()

PreviousIndex = DF_EnergyProject.index
NewParsedIndex= pd.to_datetime(PreviousIndex)
DF_EnergyProject.index =NewParsedIndex 

DF_EnergyProject.head()

#The file contains a year worth of data, do we choose:

DF_Aril_EnergyProject = DF_EnergyProject["2016-04-01 00:00:00":"2016-04-17 23:00:00"]
DF_Aril_EnergyProject.describe()

#The first column is called "Appliances" and is actually the information on Energy Consumption.
#Then, we find several factors, mainly regarding the climate into the house, but also about the atmospheric condition. 

#I don't like the name Appliences for our Consumption datas so i rename that

DF_Aril_EnergyProject=DF_Aril_EnergyProject.rename(columns={"Appliances":"Consumption"})
DF_Aril_EnergyProject.describe()

#We are going to consider our Energy Consumption as the dependent variable

DF_April_Consumption=DF_Aril_EnergyProject['Consumption'].to_frame()
DF_April_Consumption.head()
DF_April_Consumption.describe()
#Our Consumption is in Watt per hour


plt.figure()
DF_April_Consumption.plot()
plt.xlabel("Days")
plt.ylabel("Energy Consumption (Wh)")
plt.title("energy consumption during the days")
plt.show()

#in the datas are presente differt kind of informations, so we need to choose the information that most likely the consumption data depens on
#like Tout datas (Outdoor temperature, the smaller it is, bigger is the energy consumption
#like lights datas direct electric consumption.
#other informations like the relative umidity can be ignored for the moment

DF_April_Tout=DF_Aril_EnergyProject['T_out'].to_frame()
DF_April_Tout.describe()

DF_April_Lights=DF_Aril_EnergyProject['lights'].to_frame()
DF_April_Lights.describe()

DF_joined = DF_April_Consumption.join([DF_April_Tout,DF_April_Lights])

DF_joined_cleaned = DF_joined.dropna() #just in case there is any element not definite
DF_joined_cleaned.head()

DF_joined_cleaned_copy = DF_joined.dropna().copy()

#we will plot this informations to see if there is an actual conection, but first i have to nomalize the datas

df_chosen_dates_normalized = (DF_joined_cleaned- DF_joined_cleaned.min())/(DF_joined_cleaned.max()-DF_joined_cleaned.min())
df_chosen_dates_normalized.head()

df_chosen_dates_normalized = df_chosen_dates_normalized["2016-04-12 00:00:00":"2016-04-17 23:00:00"]
#i have to choose a shorter time span because the plot is a bit chaotic

plt.figure()
df_chosen_dates_normalized.plot()
plt.title("scheme representing the degree of correlation")
plt.show()

# As we can see the connection is strong enough
# i need to find more data that are strongly enought related to the data on Consumptions
DF_EnergyProject=DF_EnergyProject.rename(columns={"Appliances":"Consumption"})
DF_degree=DF_EnergyProject.corr()

#i'm iterest only on the first raw so:

DF_degree_Consumption= DF_degree.loc[:,'Consumption']
DF_degree_Consumption.index
DF_degree_Consumption[:]

print("analizing the differnt value in relation with consumption we found out:"+ str(DF_degree_Consumption.describe()))
print("28 varible and the 75% have only a relation value of 0.085 on a max of 1")

#in order to choose a sufficent number of variables i had to choose even those variables which have a low level of reletion with consumpition's value

df_choosen= DF_degree_Consumption>0.06 # 6% as minimum value
df_choosen=df_choosen[df_choosen] # true value
print("choosen variable are:" +str(df_choosen.index.tolist()))

#even if we have choosen a higher number of varabile it is still not enough so we will now use lagged data 

lag_start=1
lag_end = 6
lag_interval=1

DF_test=DF_Aril_EnergyProject.copy()

def lag_feature(df,column_names,lag_start,lag_end,lag_interval):
    for column in column_names:
        for i in range(lag_start,lag_end+1,lag_interval):
            new_column_name = column+" -"+str(i*10)+"min"
            print new_column_name
            df[new_column_name]=df[column].shift(i)   
            df.dropna(inplace=True) #this removes all the row with a Nan
    return df
    
column_names=df_choosen.index.tolist()
DF_mod=lag_feature(DF_test,column_names,lag_start,lag_end,lag_interval)
DF_mod.head(24)
DF_mod.describe()

#now the number of variablies is higher, so i repet what i have done previusly

DF_degree2=DF_mod.corr()
DF_degree_Consumption2= DF_degree2.loc[:,'Consumption']
DF_degree_Consumption2.index
DF_degree_Consumption2[:]
DF_degree_Consumption2.describe() 

print("analizing the new different variables in relation with consumption we found out:"+ str(DF_degree_Consumption2.describe()))
print("70 varible and the 75% have a relation value of 0.24 on a max of 1, a much better situation than before")

df_choosen2= DF_degree_Consumption2>0.06 # 6% as minimum value
df_choosen2=df_choosen2[df_choosen2]# true value
print("choosen variable are:" +str(df_choosen2.index.tolist()))

#Those are the features whose correlation with energy consumption is higher
DF_test= DF_mod.loc[:,df_choosen2.index.tolist()]
DF_test.head(24)
DF_test.describe()
DF_test.dropna(inplace=True)
DF_target = DF_test[["Consumption"]]
DF_features = DF_test.drop("Consumption",axis=1)

print ("DF index: ")+str(list(DF_features ))

#now it is time to use a machine lerning method im order to predict th cosumptio of energy

from sklearn.model_selection import train_test_split 

X_train, X_test, Y_train, Y_test = train_test_split(DF_features,DF_target,test_size = 0.2, random_state=41234)

from sklearn import linear_model

linear_reg = linear_model.LinearRegression()

# The second step will be fitting a model

linear_reg.fit(X_train, Y_train)

predicted_linearReg_split = linear_reg.predict(X_test)

predicted_DF_linearReg_split=pd.DataFrame(predicted_linearReg_split,index=Y_test.index, columns=["Consumption_predicted"])
predicted_DF_linearReg_split=predicted_DF_linearReg_split.join(Y_test)
predicted_DF_linearReg_split_April=predicted_DF_linearReg_split["2016-04-12 00:00:00":"2016-04-17 23:00:00"]
predicted_DF_linearReg_split_April.plot()
plt.xlabel('Time')
plt.ylabel('Energy consumption [Wh]')
plt.title("energy consumption VS predicted")
plt.ylim([0,500])
plt.show()
#We see that our prediction, although it does not match perfectly, follows the consumption profile.
# Now we want calculate how accurate our predictions are !!
# again we import everything

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
MAE_linearReg_split= mean_absolute_error(Y_test,predicted_linearReg_split)
MSE_linearReg_split= mean_squared_error(Y_test,predicted_linearReg_split,)
R2_linearReg_split = r2_score(Y_test,predicted_linearReg_split)
print "Mean squared error is " + str(MAE_linearReg_split)
print "Mean absoulute error is " + str(MSE_linearReg_split)
print "R2 score is " + str(R2_linearReg_split)

#These parameters can objectively evaluate the quality of our predictions

#Our R2 score is not that high, and also the errors' values are quite high.
#Now, let's try another algorithm! Random forests are a very good candidate!

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict

reg_RF = RandomForestRegressor()

predict_RF_CV = cross_val_predict(reg_RF,DF_features,DF_target,cv=10)

predicted_DF_RF_CV=pd.DataFrame(predict_RF_CV,
                                       index=DF_target.index, 
                                       columns=["Consumption_predicted_RF_CV"])
predicted_DF_RF_CV=predicted_DF_RF_CV.join(DF_target)
predicted_DF_RF_CV_April=predicted_DF_RF_CV["2016-04-12 00:00:00":"2016-04-17 23:00:00"]
predicted_DF_RF_CV_April.plot()
plt.xlabel('Time')
plt.ylabel('Energy consumption [Wh]')
plt.title("energy consumption VS predicted")
plt.ylim([0,800])

MAE_RF_CV= mean_absolute_error(DF_target,predict_RF_CV)
MSE_RF_CV= mean_squared_error(DF_target,predict_RF_CV)
R2_RF_CV = r2_score(DF_target,predict_RF_CV)

print "Mean squared error is " + str(MAE_RF_CV)
print "Mean absoulute error is " + str(MSE_RF_CV)
print "R2 score is " + str(R2_RF_CV)
print " "
print "the Mean squared error difference is " + str(MSE_RF_CV-MSE_linearReg_split)
print "Mean absoulute error is " + str(MAE_RF_CV-MAE_linearReg_split)
print "delta R2 score is " + str(R2_linearReg_split-R2_RF_CV)

#given the value we see that the first method is better even if the error values are still high