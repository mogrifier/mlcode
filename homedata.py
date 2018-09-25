import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

def preprocess(datafile):
    # read the data and store data in DataFrame titled melbourne_data
    house_data = pd.read_csv(datafile)
    # print a statistical summary of the data in Melbourne data
    print(house_data.describe())
    # examine the column names
    print(house_data.columns)
    # drop data with missing values - this was causing trouble with some data sets be deleting EVERYTHING!
    #house_data = house_data.dropna(axis=0)

    return house_data

def train(X, y):
    """
    y is the prediction target variable (the output). We want to predict price.
    x is the subset from original dataframe of features to be used for generating a
    prediction from the model
    """
    # examine statistics for the subset of feature column data
    print("data description")
    print(X.describe())
    # head the shows the top few rows of data. useful to avoid surprises.
    print("first 5 rows")
    print(X.head())

    # Define model. Specify a number for random_state to ensure same results each run
    model = DecisionTreeRegressor(random_state=1)

    # Fit model and print parameters
    print(model.fit(X, y))

    return model

def predict(model, input_data):
    # use model to make predictions
    print("Making predictions for the following houses:")
    print(input_data)
    val =  model.predict((input_data))
    return val

def randomforest_splittrainpredict(X, y):
    forest_model = RandomForestRegressor(random_state=1)
    train_X, test_X, train_y, test_y = train_test_split(X, y, random_state = 0)
    forest_model.fit(train_X, train_y)

    melb_preds = forest_model.predict(test_X)
    print(mean_absolute_error(test_y, melb_preds))    

def split_trainer(max_leaf_nodes, X, y):
    # returns 4 different arrays (2 pairs). Each one is a set of data for the model to use in making
    # predictions (X is for input). The y values are the values the model predictions wll be compared against.
    train_X, test_X, train_y, test_y = train_test_split(X, y, random_state = 0)
    # Define model
    melbourne_model = DecisionTreeRegressor(max_leaf_nodes = max_leaf_nodes, random_state=1)
    # Fit model
    melbourne_model.fit(train_X, train_y)

    # get predicted prices on validation data
    val_predictions = melbourne_model.predict(test_X)
    print("split trained model accuracy MAE")
    print(mean_absolute_error(test_y, val_predictions))

def xgb_split_trainer(X, y):
    # returns 4 different arrays (2 pairs). Each one is a set of data for the model to use in making
    # predictions (X is for input). The y values are the values the model predictions wll be compared against.
    train_X, test_X, train_y, test_y = train_test_split(X, y, random_state = 0)
    # Define model with xgboost and fit it
    gbm = xgb.XGBClassifier(max_depth=10, n_estimators=300, learning_rate=0.1, objective='reg:linear').fit(train_X, train_y)
    # get predicted prices on validation data
    val_predictions = gbm.predict(test_X)
    print("split trained xgb model accuracy MAE")
    print(mean_absolute_error(test_y, val_predictions))


def main():
    print("********** FIRST MODEL")
    # save filepath to variable for easier access
    melbourne_file_path = '../melb_data.csv'
    # read the data and store data in DataFrame titled melbourne_data
    melbourne_data = preprocess(melbourne_file_path)
    melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
    X = melbourne_data[melbourne_features]
    my_model = train(X, melbourne_data.Price)

    # use model to make predictions
    predicted_home_prices = predict(my_model, X)

    print(predicted_home_prices)
    print("model accuracy MAE")
    print(mean_absolute_error(melbourne_data.Price, predicted_home_prices))

    # now let's create a second model using different features
    # save filepath to variable for easier access
    print("********** SECOND MODEL")

    melbourne_file_path = '../train.csv'
    # read the data and store data in DataFrame titled melbourne_data
    melbourne_data2 = preprocess(melbourne_file_path)
    melbourne_features = ['LotArea','YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
    X2 = melbourne_data2[melbourne_features]
    # why droprows here? because this is a subset of the features you want. If unused columns had bad data, you don't care
    # since not using them. This way, you preserve more dta (potentially) fro train and test.
    X2 = X2.dropna(axis=0)

    my_model2 = train(X2, melbourne_data2.SalePrice)

    # use model to make predictions
    predict(my_model2, X2.head())
    # now how accurate? compare
    feature = ['SalePrice']
    price = melbourne_data2[feature]
    print(price.head())

    # now some test data
    testdataframe = preprocess("../test.csv")
    #now create a subset
    testdata = testdataframe[melbourne_features]
    results = predict(my_model2, testdata)
    print (results)
    """
    but what do you compare predictions to for accuracy checking? That is where the notion of splitting up
    your train data into training and testing sets. You need sales prices, in this example, to be able to
    compare the prediction to actual.
    """
    splitdataframe = preprocess("../train.csv")
    splitdata = splitdataframe[melbourne_features]
    split_trainer(5, splitdata, splitdataframe.SalePrice)

    print("with garage and more as a feature")
    # I added some features to improve the results. Only choose columns
    # with numerical values! You can play around with these and see how
    # the prediction results change.
    melbourne_features2 = ['LotArea','YearBuilt', 'GarageCars', 'OverallQual', 'OverallCond',
    'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
    splitdataframe = preprocess("../train.csv")
    splitdata = splitdataframe[melbourne_features2]

    for max_leaf_nodes in [5, 50, 500, 5000]:
        print("Max leaf nodes: %d"  %(max_leaf_nodes))
        split_trainer(max_leaf_nodes, splitdata, splitdataframe.SalePrice)

    # best was 50 leaf nodes. error at 26,021

    # now try a randomforest
    print("random forest prediction")
    randomforest_splittrainpredict(splitdata, splitdataframe.SalePrice)
    #error at 19,920 without tuning at all. better model.

    # and now xgboost
    xgb_split_trainer(splitdata, splitdataframe.SalePrice)

if __name__== "__main__":
  main()