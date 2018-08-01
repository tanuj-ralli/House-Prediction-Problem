import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv('data/train.csv')
# Isnull = train.isnull().sum() / len(train) * 100
# Isnull = Isnull[Isnull > 0]
# Isnull.sort_values(inplace=True, ascending=False)
# print(Isnull)

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
        'ExterQual', 'ExterCond', 'HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
        'YrSold', 'MoSold', 'MSZoning', 'LandContour', 'LotConfig', 'Neighborhood',
        'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
        'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'Foundation', 'GarageType', 'MiscFeature',
        'SaleType', 'SaleCondition', 'Electrical', 'Heating', 'Utilities')

for col in cols:
    train[col] = train[col].fillna('None')

train['LotFrontage'] = train.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

for col in ['GarageYrBlt', 'GarageArea', 'GarageCars']:
    train[col] = train[col].fillna(int(0))

train['MasVnrArea'] = train['MasVnrArea'].fillna(int(0))

train['Electrical'] = train['Electrical'].fillna(train['Electrical']).mode()[0]

# Isnull = train.isnull().sum() / len(train) * 100
# Isnull = Isnull[Isnull > 0]
# Isnull.sort_values(inplace=True, ascending=False)
# print(Isnull)



for c in cols:
    lbl = LabelEncoder()
    lbl.fit(list(train[c].values))
    train[c] = lbl.transform(list(train[c].values))


y = train['SalePrice']
del train['SalePrice']

X = train.values
y = y.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
model = linear_model.LinearRegression()
model.fit(X_train, y_train)

print("Predict value " + str(model.predict([X_test[142]])))
print("Real value " + str(y_test[142]))
print("Accuracy on test set --> ", model.score(X_test, y_test)*100)

test = pd.read_csv('data/test.csv')
for col in cols:
    test[col] = test[col].fillna('None')

test['LotFrontage'] = test.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

for col in ['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'LotFrontage', 'BsmtHalfBath', 'BsmtFullBath']:
    test[col] = test[col].fillna(int(0))

test['MasVnrArea'] = test['MasVnrArea'].fillna(int(0))

test['Electrical'] = test['Electrical'].fillna(test['Electrical']).mode()[0]

for c in cols:
    lbl = LabelEncoder()
    lbl.fit(list(test[c].values))
    test[c] = lbl.transform(list(test[c].values))

X_t = test.values

for i in test['Id']:
    print("ID : " + str(i) + " Predict value " + str(model.predict([X_t[i-1461]])))
