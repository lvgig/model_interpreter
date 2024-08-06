import pandas as pd
import numpy as np  # type: ignore
import pytest
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import xgboost as xgb
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, BaggingRegressor
from sklearn.svm import LinearSVC
from model_interpreter.interpreter import ModelInterpreter


@pytest.fixture()
def MI(features):
    x = ModelInterpreter(feature_names=features)
    return x


# Dataframes
@pytest.fixture
def linear_df():
    """Initial df with feature and target.
    Used for Linear regression model, XGB model(tree based) and balanced bagging regressor"""
    df = pd.DataFrame(
        {
            "MedInc": [8.3252, 8.3014, 7.2574, 5.6431, 3.8462],
            "HouseAge": [41.0, 21.0, 52.0, 52.0, 52.0],
            "AveRooms": [6.984, 6.238, 8.288, 5.817, 6.281],
            "AveBedrms": [1.023, 0.978, 1.073, 1.073, 1.081],
            "Population": [322.0, 2401.0, 496.0, 558.0, 565.0],
            "AveOccup": [2.555, 2.109, 2.802, 2.547, 2.181],
            "MedHouseVal": [4.526, 3.585, 3.521, 3.413, 3.422],
        }
    )
    return df


@pytest.fixture
def classification_df():
    """Initial df with feature and target.
    Used for Balanced bagging classifier"""
    df = pd.DataFrame(
        {
            "MedInc": [8.3252, 8.3014, 7.2574, 5.6431, 3.8462],
            "HouseAge": [41.0, 21.0, 52.0, 52.0, 52.0],
            "AveRooms": [6.984, 6.238, 8.288, 5.817, 6.281],
            "AveBedrms": [1.023, 0.978, 1.073, 1.073, 1.081],
            "Population": [322.0, 2401.0, 496.0, 558.0, 565.0],
            "AveOccup": [2.555, 2.109, 2.802, 2.547, 2.181],
            "MedHouseVal": [1.0, 0.0, 1.0, 0.0, 1.0],
        }
    )
    return df


@pytest.fixture
def logic_df():
    """Initial df with feature and target.
    Used for Logistic regression model"""
    df = pd.DataFrame(
        {
            "a": [18, 22, 34, 12, 23],
            "b": [41, 21, 53, 52, 53],
            "c": [6.984, 6.238, 8.288, 5.817, 6.281],
            "d": [82, 42, 106, 104, 106],
        }
    )

    df["d"] = df["d"].astype("category")
    return df


@pytest.fixture
def one_hot_df():
    df = pd.DataFrame(
        {
            "MedInc": [8.3252, 8.3014, 7.2574, 5.6431, 3.8462],
            "AveRooms": [6.984, 6.238, 8.288, 5.817, 6.281],
            "AveBedrms": [1.023, 0.978, 1.073, 1.073, 1.081],
            "Population": [322.0, 2401.0, 496.0, 558.0, 565.0],
            "AveOccup": [2.555, 2.109, 2.802, 2.547, 2.181],
            "MedHouseVal": [4.526, 3.585, 3.521, 3.413, 3.422],
            "Over25_Over25": [1.0, 0.0, 1.0, 1.0, 1.0],
            "Over25_Under25": [0.0, 1.0, 0.0, 0.0, 0.0],
        }
    )
    return df


@pytest.fixture
def clustering_data():
    X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    return X


# Models
@pytest.fixture
def lin_model(linear_df, features):
    """linear_regression_model model that is trained on the above dataframe.
    Takes in the initial df which was set as a fixture"""
    targets = ["MedHouseVal"]
    X = linear_df[features]
    y = linear_df[targets]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    regr = linear_model.LinearRegression()
    regr_model = regr.fit(X_train, y_train)

    return regr_model


@pytest.fixture
def logic_model(logic_df):
    """logic_df model that is trained on the above dataframe.
    Takes in the initial df which was set as a fixture"""

    features = ["a", "b", "c", "d"]
    targets = ["d"]
    X = logic_df[features]
    y = logic_df[targets]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    logic_model = LogisticRegression(random_state=0).fit(X_train, y_train)

    return logic_model


@pytest.fixture
def tree_model(linear_df, features):
    """XGB model that is trained on the above dataframe.
    Takes in the initial df which was set as a fixture"""

    targets = ["MedHouseVal"]
    X = linear_df[features]
    y = linear_df[targets]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    dtrain = xgb.DMatrix(data=X_train, label=y_train)
    xgb_model = xgb.train(
        params={"seed": 1, "max_depth": 6, "min_child_weight": 20}, dtrain=dtrain
    )

    return xgb_model


@pytest.fixture
def balanced_bagging_classifier(classification_df, features):
    """Balanced bagging classifier model that is trained on the above dataframe.
    Takes in the initial df which was set as a fixture"""

    targets = ["MedHouseVal"]
    X = classification_df[features]
    y = classification_df[targets]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    bbc_model = BaggingClassifier(n_estimators=10, random_state=0).fit(
        X_train, y_train.values.flatten()
    )

    return bbc_model


@pytest.fixture
def balanced_bagging_regressor(linear_df, features):
    """Balanced bagging classifier model that is trained on the above dataframe.
    Takes in the initial df which was set as a fixture"""

    targets = ["MedHouseVal"]
    X = linear_df[features]
    y = linear_df[targets]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    bb_reg = BaggingRegressor(n_estimators=10, random_state=0).fit(X_train, y_train)
    return bb_reg


@pytest.fixture
def tree_model_RF(linear_df, features):
    """RF model that is trained on the above dataframe.
    Takes in the initial df which was set as a fixture.
    This RF is a multi-class classification problem"""

    np.random.seed(43)

    X = linear_df[features]
    y = np.random.randint(0, 3, len(X))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X_train, y_train)

    return clf


@pytest.fixture
def tree_model_RF_binary(linear_df, features):
    """RF model that is trained on the above dataframe.
    Takes in the initial df which was set as a fixture"""

    X = linear_df[features]
    y = np.random.randint(0, 2, len(X))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X_train, y_train)

    return clf


@pytest.fixture
def k_model(clustering_data):
    kmeans = KMeans(n_clusters=2, random_state=0).fit(clustering_data)

    return kmeans


@pytest.fixture
def DBS_model(clustering_data):
    clustering = DBSCAN(eps=3, min_samples=2).fit(clustering_data)
    return clustering


@pytest.fixture
def linearSVC_model(classification_df, features):
    targets = ["MedHouseVal"]
    X = classification_df[features]
    y = classification_df[targets]
    svc_model = LinearSVC().fit(X, y)
    return svc_model


@pytest.fixture
def tree_model_one_hot(one_hot_df, one_hot_features):
    """XGB model that is trained on the above dataframe.
    Takes in the initial df which was set as a fixture"""

    targets = ["MedHouseVal"]
    X = one_hot_df[one_hot_features]
    y = one_hot_df[targets]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    dtrain = xgb.DMatrix(data=X_train, label=y_train)
    xgb_model = xgb.train(
        params={"seed": 1, "max_depth": 6, "min_child_weight": 20}, dtrain=dtrain
    )

    return xgb_model


# X_train
@pytest.fixture
def lin_X_train(linear_df, features):
    """Returns X_train from the initial df.
    Takes in the initial df which was set as a fixture"""

    targets = ["MedHouseVal"]
    X = linear_df[features]
    y = linear_df[targets]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    return X_train


@pytest.fixture
def logic_X_train(logic_df):
    """Returns X_train from the initial df.
    Takes in the initial df which was set as a fixture"""

    features = ["a", "b", "c", "d"]
    targets = ["d"]
    X = logic_df[features]
    y = logic_df[targets]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    return X_train


@pytest.fixture
def kernel_X_train(classification_df, features):
    """Returns X_train from the initial df.
    Takes in the initial df which was set as a fixture"""

    targets = ["MedHouseVal"]
    X = classification_df[features]
    y = classification_df[targets]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    return X_train


# X_test
@pytest.fixture
def lin_X_test(linear_df, features):
    """Returns X_train from the initial df.
    Takes in the initial df which was set as a fixture"""

    targets = ["MedHouseVal"]
    X = linear_df[features]
    y = linear_df[targets]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    return X_test


@pytest.fixture
def log_X_test(logic_df, features):
    """Returns X_train from the initial df.
    Takes in the initial df which was set as a fixture"""

    targets = ["MedHouseVal"]
    X = logic_df[features]
    y = logic_df[targets]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    return X_test


@pytest.fixture
def one_hot_tree_X_test(one_hot_df, one_hot_features):
    """Returns X_train from the initial df.
    Takes in the initial df which was set as a fixture"""

    targets = ["MedHouseVal"]
    X = one_hot_df[one_hot_features]
    y = one_hot_df[targets]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    return X_test


@pytest.fixture
def kernel_X_test(classification_df, features):
    """Returns X_test from the initial df.
    Takes in the initial df which was set as a fixture"""

    targets = ["MedHouseVal"]
    X = classification_df[features]
    y = classification_df[targets]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    return X_train


# Misc.
@pytest.fixture
def invalid_x_train():
    return "GGWP"


@pytest.fixture
def single_row(lin_X_test, features):
    lin_single_row = lin_X_test[features].head(1)
    return lin_single_row


@pytest.fixture
def kernel_single_row(kernel_X_test, features):
    ker_single_row = kernel_X_test[features].head(1)
    return ker_single_row


@pytest.fixture
def features():
    features = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup"]
    return features


@pytest.fixture
def one_hot_features():
    features = [
        "MedInc",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Over25_Over25",
        "Over25_Under25",
    ]
    return features


@pytest.fixture
def output_one_hot_encoding():
    """One hot encoding"""
    output = [
        {"MedInc": (8.3014, 0.0)},
        {"AveRooms": (6.238, 0.0)},
        {"AveBedrms": (0.978, 0.0)},
        {"Population": (2401.0, 0.0)},
        {"AveOccup": (2.109, 0.0)},
        {"Over25": ("Under25", 0.0)},
    ]

    return output
