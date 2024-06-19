import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')
import pickle

df = pd.read_csv('clean_data.csv')
df

X=df[['LP','DP','Port_Outgoing_Rake']]
y=df[['Plant_Incoming_Rake']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

reg=LinearRegression()
reg.fit(X_train,y_train)


pickle.dump(reg,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))