#script containing the functions used in the main script
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

os.chdir("C:\Users\utente\Dropbox\Progetto Building system\Data-driven")

#this is a function that adds to the dataframe some informations, like the day of the week, the month of the year and if we are in light hours or in dark hours
def features_creation(df,Dawn,Sunset):
    #df ==> DataFrame
    #Dawn ==> time of the dawn
    #Sunset ==> time of the sunset
    #df['sin_hour'] = np.sin((df.index.hour)*2*np.pi/24)
    #df['cos_hour'] = np.cos((df.index.hour)*2*np.pi/24)#later try 24 vector binary format
    #df['hour'] = df.index.hour # 0 to 23
    df['Day of Week'] = df.index.dayofweek #Monday = 0, sunday = 6
    #df['weekend'] = [ 1 if day in (5, 6) else 0 for day in df.index.dayofweek ] # 1 for weekend and 0 for weekdays
    df['Month'] = df.index.month
    #df['week_of_year'] = df.index.week
    #light = 1 if (6Hrs -18Hrs) and dark = 0 (otherwise)
    df['Light or Dark'] = [1 if day<Sunset and day>Dawn else 0 for day in df.index.hour ] #so I'm expecting that if I have 1 the value of solar irradiation will be high, while if I have 0 this value will be low or null
    return df
    

#this is a function that shifts the columns of a dataframe of a value taken as an input and adds the shifted column to the dataframe
def lag_column(df,column_name,lag_period=1):
    for i in range(1,lag_period+1,1):
        new_column_name = column_name+" -"+str(i)+"hr"
        df[new_column_name]=(df[column_name]).shift(i)
    return df

        
def normalize(df):
    return (df-df.min())/(df.max()-df.min())
    

#function to determine Rsquare,MSE, MAE and the coefficient of variation

def Accuracy(df,ActualValues,PredictedValues,Model):
    #df ==> DataFrame
    #ActualValues ==> name of the column of the dataframe containing the measured values
    #PredictedValues ==> name of the column of the dataframe containing the predicted values
    #Model ==> type of model (linear regression, cross validation, support vector machines, random forest regressor)
    from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
    Rsquare=r2_score(df[ActualValues],df[PredictedValues])
    MAE=mean_absolute_error(df[ActualValues],df[PredictedValues])
    MSE=mean_squared_error(df[ActualValues],df[PredictedValues])
    CoeffVar=np.sqrt(MSE)/df[ActualValues].mean()
    Values={"Rquare":Rsquare,"MAE":MAE,"MSE":MSE,"Coefficient Variation":CoeffVar}
    print ("Rsquare for "+Model+": "+str(Rsquare))
    print ("Mean absolute error for "+Model+": "+str(MAE))
    print ("Mean squared error for "+Model+": "+str(MSE))
    print ("Coeffcient of variation for "+Model+": "+str(CoeffVar))
    return Values
    