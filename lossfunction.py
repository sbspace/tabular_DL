
# 저장되는지 확인
import numpy as np
import pandas as pd
import torch

# Root Mena Squared Error (w/ log1p)
def rmse(y_true , y_pred):
    return np.sqrt(np.mean(np.power(y_true - y_pred, 2)))

# Mean Squared Error
def mse(y_true, y_pred):
    #y_true = np.expm1(y_true)
    #y_pred = np.expm1(y_pred)
    return np.mean(np.square(y_true - y_pred))

# Mean Absolute Error
def mae(y_true, y_pred):
    #y_true = np.expm1(y_true)
    #y_pred = np.expm1(y_pred)
    return np.mean(np.abs(y_true - y_pred))

# Mean Absolute Percentage Error
def mape(y_true, y_pred):
    #y_true = np.expm1(y_true)
    #y_pred = np.expm1(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# R-squared
def r_squared(y_true, y_pred):
    #y_true = np.expm1(y_true)
    #y_pred = np.expm1(y_pred)
    ss_res = np.sum(np.square(y_true - y_pred))
    ss_tot = np.sum(np.square(y_true - np.mean(y_true)))
    return 1 - (ss_res / ss_tot)

# Error 높은 Top n개 확인

def get_top_error_data(y_test, pred, n_tops):
    # DataFrame에 컬럼들로 실제 대여횟수(count)와 예측 값을 서로 비교 할 수 있도록 생성.
    result_df = pd.DataFrame(y_test.values, columns=['real_count'])
    result_df['predicted_count']= np.round(pred)
    result_df['diff'] = np.abs(result_df['real_count'] - result_df['predicted_count'])
    # 예측값과 실제값이 가장 큰 데이터 순으로 출력.
    top_errors = result_df.sort_values('diff', ascending=False)[:n_tops]
    print(top_errors)
    print('인덱스:',top_errors.index.tolist())

# Pytorch R-squared
def r_squared_torch(y_true, y_pred):
    """Compute R^2 score."""
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2

# Pytorch adjusted R-squared
def adjusted_r_squared(y_true, y_pred, n, p):
    """Compute adjusted R^2 score.

    Parameters:
    - y_true (tensor): Ground truth values.
    - y_pred (tensor): Predicted values.
    - n (int): Number of observations.
    - p (int): Number of predictors.
    """
    r2 = r_squared(y_true, y_pred)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return adj_r2