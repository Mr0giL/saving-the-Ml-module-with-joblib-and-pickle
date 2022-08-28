import numpy as nm
import pandas as pd
from sklearn.impute import SimpleImputer as sp
from sklearn.linear_model import LinearRegression
import pickle
import joblib
lireg = LinearRegression()
dfp = pd.read_csv('3rd data predict.csv')
df_reviewed = pd.read_csv('3rd data-reviewed.csv')
## 1- fill missing data using simple repeat regression method from scikit-learn Linear regression ##

# to fix the missing results using regression, two cection of simple repeat method used multiple times seprrately each time on only one column new dataset stored in first data-reviewed.csv file. each repeat stored in multi reg data missing fix.xlsx file #

##  2- predict salary ##
dfp_reviewed = dfp
ypt_reviewed = df_reviewed['salary($)']
df_reviewed.drop('salary($)',axis=1,inplace=True)
lireg.fit(df_reviewed,ypt_reviewed)
dfp_reviewed.drop('salary($)',axis=1,inplace=True)
predpt_reviewed = lireg.predict(dfp_reviewed)
dfp_reviewed['salary($)'] = predpt_reviewed

## 3- round and print the starting and prediction results ##
df_reviewed_rounded = round(df_reviewed)
dfp_reviewed_rounded = round (dfp_reviewed)
print('mean fileld table :')
print(df_reviewed_rounded)

print('mean sallary prediction results :')
print(dfp_reviewed_rounded)
## 4- save the model using pickle ##
filename = 'p_final.sav'
pickle.dump(lireg, open('p_final.sav', 'wb'))
## 5- save the model using joblib
filename = 'j_final.sav'
joblib.dump(lireg, 'j_final.sav')
## 6- # load the pickle model from disk ##
loaded_model = pickle.load(open('p_final.sav', 'rb'))
p_result = loaded_model.score(df_reviewed, ypt_reviewed)
print(p_result)
## 7- # load the joblib model from disk ##
loaded_model = joblib.load('j_final.sav')
j_result = loaded_model.score(df_reviewed, ypt_reviewed)
print(j_result)


















## End of Code ;/ ##

