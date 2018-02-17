def lag_column10(df,column_name, lag_period = 1):
    
    """ This functions adds to the data frame a column with a certain shift w.r.t the column chosen """
            
    for i in range(6, lag_period+1, 1):
        new_column_name = column_name + "-" + str(i*10) + "min"
        df[new_column_name] = df[column_name].shift(i)
    return df
    
def lag_column15(df,column_name, lag_period = 1):
    
    """ This functions adds to the data frame a column with a certain shift w.r.t the column chosen """
            
    for i in range(4, lag_period+1, 1):
        new_column_name = column_name + "-" + str(i*15) + "min"
        df[new_column_name] = df[column_name].shift(i)
    return df
    
def weekendDetector(day):
    
    """ function that defines if the day is a weekend day or not """
    
    weekendLabel = 0
    if(day == 5 or day == 6):
        weekendLabel = 1
    else:
        weekendLabel = 0
    return weekendLabel
    
def dayDetector(hour):
    
    """ function that defines if the hour is a day or night one """
    
    dayLabel = 1
    if(hour < 20 and hour > 9):
        dayLabel = 1
    else:
        dayLabel = 0
    return dayLabel