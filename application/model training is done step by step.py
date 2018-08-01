import numpy as np
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv('/home/tanuj/house/application/data/train.csv')
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

for col in ['GarageYrBlt', 'GarageArea', 'GarageCars', 'MasVnrArea']:
    train[col] = train[col].fillna(int(0))

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
Y = y.values
Y = Y.reshape(Y.shape[0],1)
max_colx = (X.max(0))
max_colx = max_colx.reshape(1,max_colx.shape[0])
max_colx[max_colx == 0] = 1
X = np.divide(X,max_colx)
df = pd.DataFrame(X)
df.fillna(int(0))

max_coly = (Y.max(0))
max_coly = max_coly.reshape(1,max_coly.shape[0])
max_coly[max_coly == 0] = 1
Y = np.divide(Y,max_coly)

# df = pd.DataFrame(X)
# res = df.isnull().any().any()
# print(res)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=7)

def initialize_with_zeros(dim):
    w = np.random.rand(dim, 1)
    b = 0
    return w, b


def propagate(w, b, X, Y):
    m = X.shape[0]
    A = np.matmul(X, w) + b

    cost = (1 / (2 * m)) * np.sum(np.square(A - Y))
    dw = (1 / m) * (np.dot((A - Y).T, X).T)
    db = (1 / m) * (np.sum(A - Y))
    grads = {"dw": dw,
             "db": db}
    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - (learning_rate * dw)
        b = b - (learning_rate * db)

        if i % 100 == 0:
            costs.append(cost)

        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


def predict(w, b, X):
    w = w.reshape(X.shape[1], 1)

    Y_prediction = np.matmul(X, w) + b

    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=5, learning_rate=0.05, print_cost=False):
    w, b = initialize_with_zeros(X_train.shape[1])
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    w = parameters["w"]
    b = parameters["b"]

    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d

d = model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.07, print_cost = True)

#Input any set of X to predict the Housing Price
test = pd.read_csv('/home/tanuj/house/application/data/test.csv')
for col in cols:
    test[col] = test[col].fillna('None')

test['LotFrontage'] = test.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

for col in ['GarageYrBlt', 'GarageArea', 'MasVnrArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'LotFrontage', 'BsmtHalfBath', 'BsmtFullBath']:
    test[col] = test[col].fillna(int(0))

test['Electrical'] = test['Electrical'].fillna(test['Electrical']).mode()[0]

for c in cols:
    lbl = LabelEncoder()
    lbl.fit(list(test[c].values))
    test[c] = lbl.transform(list(test[c].values))

X = test.values
X = np.divide(X,max_colx)

w = d['w']
b = d['b']

Y_prediction = np.matmul(X,w) + b
Y_prediction = Y_prediction * max_coly
print(Y_prediction)