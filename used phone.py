import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import norm, boxcox
from scipy import stats
from collections import Counter
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import statsmodels.api as sm
import pickle as pk

data = pd.read_csv("D:\\project\\Used Phone Price\\Used-Phone-Price-Prediction-Project-main\\Used-Phone-Price-Prediction-Project-main\\used_device_data.csv")

data.head()
data.columns

data.info()

data.describe()

data.isnull()
data.isnull().sum()

sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')

data = data.dropna()

sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')

data.duplicated()
data.duplicated().sum()

dict_G = {'yes':1,'no':0}

data['4g'] = data['4g'].map(dict_G)

data['5g'] = data['5g'].map(dict_G)

brand_dict = data.groupby(['device_brand'])['normalized_used_price'].median().to_dict()
data['Device_Brand'] = data['device_brand'].map(brand_dict)

data.head()

len(data.device_brand.value_counts())

data.device_brand.unique()

max_brand=data['device_brand'].value_counts().sort_values(ascending=False)[0:20]
max_brand.name='count'
max_brand.index.name='Brands'

fig=plt.figure(figsize=(14,8))
sns.barplot(x=max_brand.index,y=max_brand)
plt.tight_layout()

fig=plt.figure(figsize=(15,8))
sns.barplot(y=data['Device_Brand'],x=data["device_brand"])
plt.xticks(rotation=90)

fig = plt.figure(figsize=(15,8))
sns.barplot(y=data['Device_Brand'],x=data['device_brand'])
plt.tight_layout()

fig,ax=plt.subplots(2,2,figsize=(15,12))

sns.boxplot(x='os',y='normalized_used_price', data=data,ax=ax[0,0])
ax[0,0].set_title('os vs normalized_used_price')

sns.boxplot(x='4g',y='normalized_used_price',data=data,ax=ax[0,1])
ax[0,1].set_title('4g vs normalized_used_price')

sns.boxplot(x='5g',y='normalized_used_price',data=data,ax=ax[1,0])
ax[1,0].set_title('5g vs normalized_used_price')

sns.boxplot(x='release_year',y='normalized_used_price',data=data,ax=ax[1,1])
ax[1,1].set_title('ralease_year vs normalised_used_price')

plt.tight_layout()
plt.show

data.columns

numerical_features = ['screen_size','rear_camera_mp','front_camera_mp','battery','weight','days_used','normalized_new_price', 'normalized_used_price']

data.hist(figsize=(15,10),bins=30)
plt.tight_layout()

   
def bivariate_analysis(x):
    plt.figure(figsize=(10,6))
    ax = sns.regplot(x=x, y='normalized_used_price',data=data)
    ax.set_title("Used Price vs "+x, fontsize=25)
    ax.set_xlabel(x, fontsize=20)
    ax.set_ylabel('normalized_used_price', fontsize=20)
    plt.locator_params(axis='x', nbins=10)

cols = ['screen_size','rear_camera_mp','front_camera_mp','battery','weight','days_used','ram','internal_memory','normalized_new_price']
for x in cols:
    bivariate_analysis(x)
plt.tight_layout()


sns.pairplot(data[numerical_features],diag_kind='kde')

fig = plt.figure(figsize=(12,8))
sns.heatmap(data.corr(),annot=True)

cols2 = ['screen_size','rear_camera_mp','front_camera_mp','battery','weight','days_used','normalized_new_price','normalized_used_price']

fig,ax=plt.subplots(2,4,figsize=(12,8))
index=0
ax=ax.flatten()
for col in cols2:
    sns.boxplot(y=col, data=data, color='r', ax=ax[index])
    index+=1
plt.tight_layout(pad=0.5, w_pad=1, h_pad=5.0)

def detect_outliers(columns):
    outlier_indices = []

    for column in columns:
        # 1st quartile
        Q1 = np.percentile(data[column], 25)
        # 3st quartile
        Q3 = np.percentile(data[column], 75)
        # IQR
        IQR = Q3 - Q1
        # Outlier Step
        outlier_step = IQR * 1.5
        # detect outlier and their indeces
        outlier_list_col = data[(data[column] < Q1 - outlier_step)
                              | (data[column] > Q3 + outlier_step)].index
        # store indeces
        outlier_indices.extend(outlier_list_col)
        return outlier_indices

len(detect_outliers(cols2))

def check_skweness(columnName):
    print('''Before Correcting''')
    try:
        (mu, sigma) = norm.fit(data[columnName])
    except RuntimeError:
        (mu,sigma) = norm.fit(data[columnName].dropna())
    print("Mu before correcting {} : {}, Sigma before correcting {} : {}".format(
        columnName.upper(), mu, columnName.upper(), sigma))
    plt.figure(figsize=(20,10))
    plt.subplot(1,2,1)
    sns.distplot(data[columnName], fit=norm, color="orange")
    plt.title(columnName.upper() +
              " Distplot before Skewness Correction", color="black")
    plt.subplot(1,2,2)
    stats.probplot(data[columnName], plot=plt)
    plt.show();

skew_check_cols = ['screen_size','rear_camera_mp','front_camera_mp','battery','weight','days_used']
for columns in skew_check_cols:
    check_skweness(columns)


def trying_different_transformations(column,transformation):
    if transformation=='boxcox':
        try:
            print("BoxCox - "+column)
            temp,temp_params = boxcox(data[column]+1)
            (mu,sigma)=norm.fit(temp)
            print("mu ",mu," sigma ",sigma)
            plt.figure(figsize=(20,10))
            plt.subplot(1,2,1)
            sns.distplot(temp, fit=norm, color="orange")
            plt.subplot(1,2,2)
            stats.probplot(temp, plot = plt)
        except ValueError:
            pass
        except ValueError:
            pass
    elif transformation=='log':
        try:
            print("Log - "+column)
            (mu,sigma)=norm.fit(np.log1p(data[column]))
            print("mu ",mu," sigma ",sigma)
            plt.figure(figsize=(20,10))
            plt.subplot(1,2,1)
            sns.distplot(np.log1p(data[column]), fit=norm, color="orange")
            plt.subplot(1,2,2)
            stats.probplot(np.log1p(data[column]), plot = plt)
        except RuntimeError:
            pass
        except ValueError:
            pass
    elif transformation=='reciprocal':
        try:
            print("Reciprocal - "+column)
            temp_r = 1/data[column]
            temp_r = temp_r.replace([np.inf, -np.inf], 0)
            (mu,sigma)=norm.fit(temp_r)
            print("mu ",mu," sigma ",sigma)
            plt.figure(figsize=(20,10))
            plt.subplot(1,2,1)
            sns.distplot(temp_r, fit=norm, color="orange")
            plt.subplot(1,2,2)
            stats.probplot(temp_r, plot = plt)
        except RuntimeError:
            pass
        except ValueError:
            pass
    elif transformation=='sqroot':
        try:
            print("Square_Root - "+column)
            (mu,sigma)=norm.fit(data[column]**(1/2))
            print("mu ",mu," sigma ",sigma)
            plt.figure(figsize=(20,10))
            plt.subplot(1,2,1)
            sns.distplot(data[column]**(1/2), fit=norm, color="orange")
            plt.subplot(1,2,2)
            stats.probplot(data[column]**(1/2), plot = plt)
        except RuntimeError:
            pass
        except ValueError:
            pass
    else:
        try:
            print("Exponential - "+column)
            (mu,sigma)=norm.fit(data[column]**(1/1.2))
            print("mu ",mu," sigma ",sigma)
            plt.figure(figsize=(20,10))
            plt.subplot(1,2,1)
            sns.distplot(data[column]**(1/1.2), fit=norm, color="orange")
            plt.subplot(1,2,2)
            stats.probplot(data[column]**(1/1.2), plot = plt)
        except RuntimeError:
            pass
        except ValueError:
            pass

transformations = ['boxcox','log','reciprocal','sqroot','exp']

for x in transformations:
    trying_different_transformations('screen_size',x)

for x in transformations:
    trying_different_transformations('rear_camera_mp',x)

for x in transformations:
    trying_different_transformations('front_camera_mp',x)

for x in transformations:
    trying_different_transformations('battery',x)

for x in transformations:
    trying_different_transformations('weight',x)

for x in transformations:
    trying_different_transformations('days_used',x)

def skweness_correction(columnName):    
    if columnName == 'front_camera_mp' or columnName == 'screen_size' or columnName == 'battery':
        data[columnName], temp_params = boxcox(
        data[columnName]+1)
    elif columnName == 'weight':
        data[columnName] = 1/data[columnName].replace([np.inf, -np.inf], 0)
    elif columnName =='rear_camera_mp':
        data[columnName] = data[columnName]**(1/2)
    print('''After Correcting''')
    (mu, sigma) = norm.fit(data[columnName])
    print("Mu after correcting {} : {}, Sigma after correcting {} : {}".format(
        columnName.upper(), mu, columnName.upper(), sigma))
    plt.figure(figsize=(20, 10))
    plt.subplot(1,2,1)
    sns.distplot(data[columnName], fit=norm, color="orange")
    plt.title(columnName.upper() +
              " Distplot After Skewness Correction", color="black")
    plt.subplot(1,2,2)
    stats.probplot(data[columnName], plot = plt)
    plt.show();

skewColumnList1 = ['screen_size','rear_camera_mp','front_camera_mp','battery','weight']
for columns in skewColumnList1:
    skweness_correction(columns)

len(detect_outliers(cols2))

data = data.drop(detect_outliers(cols2),axis = 0).reset_index(drop = True)

data.shape[0]

data = data.drop(['os','device_brand'],axis=1)

data

dummies_year = pd.get_dummies(data['release_year'],drop_first=True)
data = pd.concat([data,dummies_year],axis=1)
data = data.drop('release_year',axis=1)

data

Y = data['normalized_used_price']
X = data.loc[:, data.columns != 'normalized_used_price']

X.columns = X.columns.astype(str)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,random_state=10)

X_train.head()

X_train.columns

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_test.shape[0]

##Linear Regression Model:-
linear_model = LinearRegression()
linear_model.fit(X_train,y_train)
X_train_prediction = linear_model.predict(X_train)
print("MSE : ",mean_squared_error(y_train,X_train_prediction))
print("R2 Score : ",r2_score(y_train,X_train_prediction))

cross_linear = cross_val_score(linear_model,X_train,y_train,scoring="neg_mean_squared_error",cv=5)
mean_cross_linear = np.mean(cross_linear)
print(mean_cross_linear)

#Ridge Regression Model:
ridge_model = Ridge()
ridge_model.fit(X_train,y_train)
X_train_pred_ridge = ridge_model.predict(X_train)
print("MSE : ",mean_squared_error(y_train,X_train_pred_ridge))
print("R2 Score : ",r2_score(y_train,X_train_pred_ridge))

cross_ridge = cross_val_score(ridge_model,X_train,y_train,scoring="neg_mean_squared_error",cv=5)
mean_cross_ridge = np.mean(cross_ridge)
print(mean_cross_ridge)

##Lasso regression Model
lasso_model = Lasso()
lasso_model.fit(X_train,y_train)
X_train_pred_lasso = lasso_model.predict(X_train)
print("MSE : ",mean_squared_error(y_train,X_train_pred_lasso))
print("R2 Score : ",r2_score(y_train,X_train_pred_lasso))

cross_lasso = cross_val_score(lasso_model,X_train,y_train,scoring="neg_mean_squared_error",cv=5)
mean_cross_lasso = np.mean(cross_lasso)
print(mean_cross_lasso)

##SVM Regressor
svmreg_model = SVR()
svmreg_model.fit(X_train,y_train)
X_train_pred_svmreg = svmreg_model.predict(X_train)
print("MSE : ",mean_squared_error(y_train,X_train_pred_svmreg))
print("R2 Score : ",r2_score(y_train,X_train_pred_svmreg))

cross_svmreg = cross_val_score(svmreg_model,X_train,y_train,scoring="neg_mean_squared_error",cv=5)
mean_cross_svmreg = np.mean(cross_svmreg)
print(mean_cross_svmreg)

#Decision Tree Regressor
dtree_model = DecisionTreeRegressor(max_depth=10)
dtree_model.fit(X_train,y_train)
X_train_pred_dtree = dtree_model.predict(X_train)
print("MSE : ",mean_squared_error(y_train,X_train_pred_dtree))
print("R2 Score : ",r2_score(y_train,X_train_pred_dtree))

cross_dtree = cross_val_score(dtree_model,X_train,y_train,scoring="neg_mean_squared_error",cv=5)
mean_cross_dtree = np.mean(cross_dtree)
print(mean_cross_dtree)

#RandomForest Regressor
rfr_model = RandomForestRegressor()
rfr_model.fit(X_train,y_train)
X_train_pred_rfr = rfr_model.predict(X_train)
print("MSE : ",mean_squared_error(y_train,X_train_pred_rfr))
print("R2 Score : ",r2_score(y_train,X_train_pred_rfr))

cross_rfr = cross_val_score(rfr_model,X_train,y_train,scoring="neg_mean_squared_error",cv=5)
mean_cross_rfr = np.mean(cross_rfr)
print(mean_cross_rfr)


X_test_pred_rfr = rfr_model.predict(X_test)
print("MSE : ",mean_squared_error(y_test,X_test_pred_rfr))
print("R2 Score : ",r2_score(y_test,X_test_pred_rfr))



pk.dump(rfr_model,open("Price_Predictor.pkl",'wb'))

pk.dump(sc,open("Scaler.pkl",'wb'))









