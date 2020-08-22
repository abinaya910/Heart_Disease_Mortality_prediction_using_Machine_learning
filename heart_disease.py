import pandas as pd
import numpy as np
import math
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
def evaluate(x_train, x_eval, y_train, y_eval, model):
    model.fit(x_train, y_train)
    y_predict = model.predict(x_eval)
    return math.sqrt(mean_squared_error(y_predict, y_eval))

# Read data
df_X_train = pd.read_csv('Trainvaluesmean.csv')
df_Y_train = pd.read_csv('Training_labels.csv')
df_X_test  = pd.read_csv('Test_values.csv')

df_X_train.describe()

# Clean missing data , Convert categorical features
ID = 'row_id'
target = 'heart_disease_mortality_per_100k'

X_train = pd.get_dummies(df_X_train.drop(ID, axis=1).fillna(0)).values
Y_train = df_Y_train[target].values
X_test  = pd.get_dummies(df_X_test.drop(ID,axis=1).fillna(0)).values

# Split training data , Evaluate model , Tune hyper-parameters
x_train, x_eval, y_train, y_eval = train_test_split(X_train, Y_train, test_size=0.3, random_state=1)



#preprocessing
from sklearn.preprocessing import StandardScaler as ss
sc = ss()
x_train = sc.fit_transform(x_train)
x_eval = sc.transform(x_eval)



model = LGBMRegressor(n_estimators=1000, learning_rate=0.05, num_leaves=30)
rmse = evaluate(x_train, x_eval, y_train, y_eval, model)
print ('LGBMRegressor RMSE :', rmse)



# Train model , Predict
model.fit(X_train, Y_train)
Y_predict = model.predict(X_test)

# Submit
submit = np.column_stack((df_X_test[ID], np.round(Y_predict)))
np.savetxt("submit.csv", submit, fmt='%i', delimiter=",", header=ID+','+target, comments='')
#cm_test = confusion_matrix(y_train, y_eval)
print (Y_predict)
