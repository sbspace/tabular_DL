# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from patsy import dmatrices
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler

# %matplotlib inline

"""**### · Data 불러오기, 탐색**:
---

✅ ct_df : 기본데이터 (전처리 X)
---
"""

# ▶ 행/렬을 최대 몇개씩 출력할지
import pandas as pd
pd.set_option('display.max_columns',100)
pd.set_option('display.max_rows',100)

# ▶ Data read
ct_df = pd.read_csv("cycletime_train.csv")
ct_df.head()

# LotID, StepID 컬럼삭제
ct_df.drop('LOT_ID', axis=1, inplace=True)
ct_df.drop('STEP_ID', axis=1, inplace=True)
ct_df.drop('PRE_STEPGBN2', axis=1, inplace=True)


# ▶ Null 값 확인
print(ct_df.isnull().sum())

# ▶ 숫자형, 범주형 변수 분할
numerical_list=[]
categorical_list=[]

for i in ct_df.columns :
  if ct_df[i].dtypes == 'O' :
    categorical_list.append(i)
  else :
    numerical_list.append(i)

numerical_list.remove('WAITTIME')

print("categorical_list {}:".format(len(categorical_list)), categorical_list)
print("numerical_list {}:".format(len(numerical_list)), numerical_list)

# ▶ 범주형 변수 클래스 개수 확인
list_of_df = []

for var in categorical_list :
  loop_df = pd.DataFrame({'var':[var], 'ncnt':[ct_df[var].nunique()]})
  list_of_df.append(loop_df)

df_concat = pd.concat(list_of_df).reset_index(drop=True) #reset_index 안하면 인덱스가 모두 0으로 되어있음
df_concat

"""✅ ct_df_ver0 : 기본적인 공통 전처리 진행
---
** NULL 처리, Outlier 제거(타겟+피쳐), LotID/StepID 컬럼삭제
"""

ct_df_ver0 = ct_df.copy()

# NULL값 처리
ct_df_ver0['HOLD_FLAG'].fillna('N',inplace=True)
ct_df_ver0['GRADE'].fillna(5,inplace=True)
ct_df_ver0['DUE_DATE'].fillna(200,inplace=True)
ct_df_ver0['WIPTURN'].fillna(0,inplace=True)
ct_df_ver0['BATCH'].fillna('N',inplace=True)

for i in ['Q_E','Q_P','Q_H','Q_R','Q_W','Q_2','Q_3','Q_4','Q_5','Q_7','Q_8','Q_10']:
    ct_df_ver0[i].fillna(0, inplace=True)

# Null 값 확인
# print(ct_df_ver0.isnull().sum())

# 이상치제거-① : 타겟기준 상,하한 5% 제거
print("이상치 제거 전", ct_df_ver0.shape)
lower_bound = ct_df_ver0['WAITTIME'].quantile(0.05)
upper_bound = ct_df_ver0['WAITTIME'].quantile(0.95)
ct_df_ver0 = ct_df_ver0[(ct_df['WAITTIME'] > lower_bound) & (ct_df['WAITTIME'] < upper_bound)]
print("타겟 데이터 이상치 제거 후", ct_df_ver0.shape)

# 이상치제거-② : 상관관계 가장 높은 10개 피쳐 5시그마 기준 제거

corr = ct_df_ver0.corr()
corr_top5_list = list(corr['WAITTIME'].sort_values(ascending=False)[:5].index)

def filter_outliers(column):
    mean, std = column.mean(), column.std()
    lower_bound, upper_bound = mean - 5 * std, mean + 5 * std

    return column[(column >= lower_bound) & (column <= upper_bound)]

for col in corr_top5_list:
    ct_df_ver0[col] = filter_outliers(ct_df_ver0[col])

# 위 코드까지는 outlier를 NULL처리함
ct_df_ver0.dropna(inplace=True)

print("피쳐 데이터 이상치 제거 후", ct_df_ver0.shape)

ct_df_ver0 = ct_df_ver0.reset_index(drop=True)
print(ct_df.shape)
print(ct_df_ver0.shape)

"""✅ ct_df_ver05 : log정규화 + No 인코딩
---
** ver0에서 추가로 숫자형 변수 log정규화만 진행
"""

ct_df_ver05 = ct_df_ver0.copy()

# 숫자형 변수 log1p 정규화

ct_df_ver05['WAITTIME'] = np.log1p(ct_df_ver05['WAITTIME'])

for col in numerical_list:
  ct_df_ver05[col] = np.log1p(ct_df_ver05[col])

"""✅ ct_df_ver1 : log정규화 + 원핫인코딩
---
** ver0에서 추가로
- 숫자형 변수 log정규화
- 범주형변수 **원-핫인코딩** 진행
"""

ct_df_ver1 = ct_df_ver0.copy()

print(categorical_list, numerical_list, sep='\n')

# 범주형 변수 원핫 인코딩

ct_df_ver1 = pd.get_dummies(ct_df_ver1, columns = categorical_list, drop_first=True) # 다중공선성 고려 drop_first
print(ct_df_ver1.shape)

ct_df_ver1[numerical_list].describe()

# 숫자형 변수 log1p 정규화

# 타겟
ct_df_ver1['WAITTIME'] = np.log1p(ct_df_ver1['WAITTIME'])

# 피쳐
for col in numerical_list:
  ct_df_ver1[col] = np.log1p(ct_df_ver1[col])

"""✅ ct_df_ver2 : log정규화 + Binary인코딩
---
** ver0에서 추가로
- 숫자형 변수 log정규화
- 범주형변수 **바이너리인코딩** 진행
"""

ct_df_ver2 = ct_df_ver05.copy()

ct_df_ver2.head()

import category_encoders as ce

categorical_list

# 범주형 피쳐에 대해 바이너리 인코딩 진행
binary_encoder = ce.BinaryEncoder(cols=categorical_list)
ct_df_ver2 = binary_encoder.fit_transform(ct_df_ver2)

print("원핫인코딩 결과: ",ct_df_ver1.shape,"바이너리인코딩 결과: ", ct_df_ver2.shape)

"""✅ train_v3, test_v3 : log정규화 + Target인코딩
---
** ver0에서 추가로
- 숫자형 변수 log정규화
- 범주형변수 **Mean Target인코딩** 진행
"""

ct_df_ver3 = ct_df_ver0.copy()

# Train-Test 데이터셋 분리
y_ct = ct_df_ver3['WAITTIME']
x_ct = ct_df_ver3.drop(columns='WAITTIME')
print(x_ct.shape, y_ct.shape)

x_train,x_test,y_train,y_test = train_test_split(x_ct,y_ct, test_size=0.2, random_state=11)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape, type(y_train))

x_train['WAITTIME'] = y_train
x_train.iloc[45:50].sort_values('WAITTIME', ascending=False)

# 스무딩 규제적용 후 Mean Target 인코딩 적용함수

def mean_target_encode(train, test, categorical_cols, target_col, smoothing_weight=5000):
    # Compute the global mean
    global_mean = train[target_col].mean()

    for col in categorical_cols:
        # Compute the number of values and mean of each group
        agg = train.groupby(col)[target_col].agg(['count', 'mean'])
        counts = agg['count']
        means = agg['mean']

        # Compute the "smoothed" means
        smooth = (counts * means + smoothing_weight * global_mean) / (counts + smoothing_weight)

        # Apply the transformation to training data
        train[col+'_'] = train[col].map(smooth)

        # Apply the transformation to test data
        test[col+'_'] = test[col].map(smooth).fillna(global_mean)

    return train, test

# 인코딩 적용
encoded_train, encoded_test = mean_target_encode(x_train, x_test, categorical_list, 'WAITTIME')

# 인코딩 이후 기존 컬럼 삭제
encoded_train = encoded_train.drop(categorical_list, axis=1)
encoded_train = encoded_train.drop('WAITTIME', axis=1)
encoded_test = encoded_test.drop(categorical_list, axis=1)

# 인코딩 이후 log정규화 진행
x_train_v3 = np.log1p(encoded_train)
x_test_v3 = np.log1p(encoded_test)
y_train_v3 = np.log1p(y_train)
y_test_v3 = np.log1p(y_test)
print(x_train_v3.shape, x_test_v3.shape, y_train_v3.shape, y_test_v3.shape)