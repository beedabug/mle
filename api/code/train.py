import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
import joblib
import gzip


# read training data
train = pd.read_csv('data/exercise_26_train.csv') 

# data prep for training data

train['x12'] = train['x12'].str.replace('$','')
train['x12'] = train['x12'].str.replace(',','')
train['x12'] = train['x12'].str.replace(')','')
train['x12'] = train['x12'].str.replace('(','-')
train['x12'] = train['x12'].astype(float)
train['x63'] = train['x63'].str.replace('%','')
train['x63'] = train['x63'].astype(float)

# mean imputation
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
train_imputed = pd.DataFrame(imputer.fit_transform(train.drop(columns=['y', 'x5', 'x31', 'x82', 'x81'])), 
                             columns=train.drop(columns=['y', 'x5', 'x31', 'x82', 'x81']).columns)
# standardizing the data
std_scaler = StandardScaler()
train_imputed_std = pd.DataFrame(std_scaler.fit_transform(train_imputed), columns=train_imputed.columns)

# create dummy variables
dumb5 = pd.get_dummies(train['x5'], drop_first=True, prefix='x5', prefix_sep='_', dummy_na=True)
train_imputed_std = pd.concat([train_imputed_std, dumb5], axis=1, sort=False)

dumb31 = pd.get_dummies(train['x31'], drop_first=True, prefix='x31', prefix_sep='_', dummy_na=True)
train_imputed_std = pd.concat([train_imputed_std, dumb31], axis=1, sort=False)

dumb81 = pd.get_dummies(train['x81'], drop_first=True, prefix='x81', prefix_sep='_', dummy_na=True)
train_imputed_std = pd.concat([train_imputed_std, dumb81], axis=1, sort=False)

dumb82 = pd.get_dummies(train['x82'], drop_first=True, prefix='x82', prefix_sep='_', dummy_na=True)
train_imputed_std = pd.concat([train_imputed_std, dumb82], axis=1, sort=False)

train_imputed_std = pd.concat([train_imputed_std, train['y']], axis=1, sort=False)

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

# train and fit the model with selected variables

logit = sm.Logit(train_imputed_std['y'].astype(float), train_imputed_std[variables].astype(float)).fit() 

# checking for accuracy
outcomes_train = pd.DataFrame(logit.predict(train_imputed_std[variables].astype(float))).rename(columns={0:'phat'})
outcomes_train['y'] = train_imputed_std['y'].astype(float)
print('The C-Statistic for train is ', roc_auc_score(outcomes_train['y'], outcomes_train['phat']))

# calculate bins
outcomes_train['bin'] = pd.qcut(outcomes_train['phat'], q=4)
print(outcomes_train.head())

# based on the bins listed, (0.71, 0.995] is to be used to classify an event when running test data

# export model
joblib.dump(logit, gzip.open('model/model_binary.dat.gz', 'wb'))