from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.linear_model import LinearRegression

def compute_baseline(y_train):
    return np.array([y_train.mean()]*len(y_train))

def linear_model(X_train, y_train, df):
    lm=LinearRegression()
    lm.fit(X_train,y_train)
    lm_predictions=lm.predict(X_train)
    df['lm']=lm_predictions
    return df

def evaluate(actual, model):
    MSE = mean_squared_error(actual, model)
    SSE = MSE*len(actual)
    RMSE = sqrt(MSE)
    r2 = r2_score(actual, model)
    return MSE, SSE, RMSE, r2 
    
# MSE_1 = mean_squared_error(predictions.actual, predictions.lm1)
# SSE_1 = MSE_1*len(predictions.actual)
# RMSE_1 = sqrt(MSE_1)
# r2_1 = r2_score(predictions.actual, predictions.lm1)
# print(MSE_1,SSE_1,RMSE_1,r2_1)

# MSE_2 = mean_squared_error(predictions.actual, predictions.lm2)
# SSE_2 = MSE_2*len(predictions.actual)
# RMSE_2 = sqrt(MSE_2)
# r2_2 = r2_score(predictions.actual, predictions.lm2)
# print(MSE_2,SSE_2,RMSE_2,r2_2)