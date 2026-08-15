### Data Import
import pandas as pd

train_data = pd.read_csv('/Users/varunnair/Documents/GitHub/ATMOSLab-logisticreg-model-1/data/train.csv')
test_data = pd.read_csv('/Users/varunnair/Documents/GitHub/ATMOSLab-logisticreg-model-1/data/test.csv')

#Magnus Formula calculation
def magnus_calculation(dataframe):
    dataframe['magnus_calculation'] = dataframe['temp_c'] + dataframe['dewpoint_c']
    return dataframe['magnus_calculation']

def add_wind_interaction(dataframe):
    dataframe['wind_speed_kts'] = dataframe['dewpoint_depression_c'] * dataframe['wind_speed_ms']
    return dataframe['wind_speed_kts']





