import pickle
import pathlib
import pandas as pd

if __name__ == '__main__':
    path = pathlib.WindowsPath(r'C:\Users\earoge\Documents\Git\2019-order-intake-forecasting\data\temp_output\CPDD_North America-USA.pkl')
    data = pd.read_pickle(path)
    data.to_csv('C:/Users/earoge/Documents/Git/2019-order-intake-forecasting/data/temp_output/CPDD_North_America_USA_UNPICKLED.csv', index=False)
