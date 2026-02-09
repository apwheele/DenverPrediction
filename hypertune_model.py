'''
Estimating model
'''

from src import models
from src.prep import train_data, holdout_data, y_vars, x_vars
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.linear_model import LinearRegression

# Get k-folds split by person ID
k = 5
k_folds = models.kfold_split(train_data,k,split='pin')

# init a CatBoost model
y = 'property_vicoffy'
#cm = models.Mod(ide_vars=x_vars,y=y,mod=CatBoostRegressor(n_estimators=500,max_depth=3,silent=True),bin_y=False)
#cm = models.Mod(ide_vars=x_vars,y=y,mod=CatBoostClassifier(n_estimators=500,max_depth=3,silent=True))
#cm = models.Mod(ide_vars=x_vars,y=y,mod=LinearRegression(),bin_y=False)
#cm = models.Mod(ide_vars=x_vars,y=y,mod=Lasso(positive=True),bin_y=False)
cm = models.Mod(ide_vars=x_vars,y=y,mod=LogisticRegression(penalty=None,
fit_intercept=True,
solver='lbfgs',
max_iter=100000),bin_y=False)

# See metrics for k-folds
cm.met_eval(train_data,k_folds)
print(cm.metrics)
# average?

# Retrain on the big set
cm.fit(train_data)

# See metrics for hold out
holdout_data['pred'] = cm.predict(holdout_data)
hold_metrics = models.metrics(y,'pred',holdout_data)
print(hold_metrics)

# calibration plot for same y

# holdout metrics for other y variables



# Save the final model
models.save_model(cm,'./models/CatBoost.pkl')