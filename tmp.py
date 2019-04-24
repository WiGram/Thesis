import pandas as pd

path = 'C:/Users/willi/Dropbox/Matlab/MS_Regress-Matlab-master/data_Files/retdatafinal.txt'
data = pd.read_csv(path, sep = "\t", header = None)

data.head()
