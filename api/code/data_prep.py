import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
from json import loads, dumps


# read testing data
test = pd.read_csv('data/exercise_26_test.csv') 

# data prep for testing data

test['x12'] = test['x12'].str.replace('$','')
test['x12'] = test['x12'].str.replace(',','')
test['x12'] = test['x12'].str.replace(')','')
test['x12'] = test['x12'].str.replace('(','-')
test['x12'] = test['x12'].astype(float)
test['x63'] = test['x63'].str.replace('%','')
test['x63'] = test['x63'].astype(float)

# mean imputation
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
test_imputed = pd.DataFrame(imputer.fit_transform(test.drop(columns=['x5', 'x31', 'x82', 'x81'])), 
                             columns=test.drop(columns=['x5', 'x31', 'x82', 'x81']).columns)
# standardizing the data
std_scaler = StandardScaler()
test_imputed_std = pd.DataFrame(std_scaler.fit_transform(test_imputed), columns=test_imputed.columns)

# create dummy variables
dumb5 = pd.get_dummies(test['x5'], drop_first=True, prefix='x5', prefix_sep='_', dummy_na=True)
test_imputed_std = pd.concat([test_imputed_std, dumb5], axis=1, sort=False)

dumb31 = pd.get_dummies(test['x31'], drop_first=True, prefix='x31', prefix_sep='_', dummy_na=True)
test_imputed_std = pd.concat([test_imputed_std, dumb31], axis=1, sort=False)

dumb81 = pd.get_dummies(test['x81'], drop_first=True, prefix='x81', prefix_sep='_', dummy_na=True)
test_imputed_std = pd.concat([test_imputed_std, dumb81], axis=1, sort=False)

dumb82 = pd.get_dummies(test['x82'], drop_first=True, prefix='x82', prefix_sep='_', dummy_na=True)
test_imputed_std = pd.concat([test_imputed_std, dumb82], axis=1, sort=False)

del dumb5, dumb31, dumb81, dumb82

# selected variables from data scientist
variables = ['x5_saturday',
 'x81_July',
 'x81_December',
 'x31_japan',
 'x81_October',
 'x5_sunday',
 'x31_asia',
 'x81_February',
 'x91',
 'x81_May',
 'x5_monday',
 'x81_September',
 'x81_March',
 'x53',
 'x81_November',
 'x44',
 'x81_June',
 'x12',
 'x5_tuesday',
 'x81_August',
 'x81_January',
 'x62',
 'x31_germany',
 'x58',
 'x56']

formatted_data = test_imputed_std[variables].astype(float).to_json(orient='records')

file = open('data/batch_data.json', 'w')
file.write(dumps(loads(formatted_data)))