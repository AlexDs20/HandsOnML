#! python3
import os
import joblib       # to save the runs with different models .dump, .load
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn import model_selection
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler, \
                                StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from generic import fetch_data, load_data, display_info, plot_hist, split_train_test, \
        split_train_test_by_id, stratified_split, display_scores, confidence_interval


class CombinedAttributesDivide(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):                 # No *args, **kwargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        rooms_id = housing.columns.get_loc("total_rooms")
        bedrooms_id = housing.columns.get_loc("total_bedrooms")
        population_id = housing.columns.get_loc("population")
        households_id = housing.columns.get_loc("households")
        rooms_per_household = X[:, rooms_id] / X[:, households_id]
        population_per_household = X[:, population_id] / X[:, households_id]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_id] / X[:, rooms_id]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


def exploring_data(housing):
    # Display info about the data
    if False:
        display_info(housing)
        print('# of ocean_prox. categories: \n',
              housing["ocean_proximity"].value_counts(), '\n')
    if False:
        plot_hist(housing)

    # ------------------------------
    # Split Data
    # ------------------------------
    if False:
        # normal
        train, test = split_train_test(housing, 0.2)

        # by id
        housing_with_id = housing.reset_index()
        train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

        housing_with_id["id"] = housing["longitude"] * 10**3 + housing["latitude"]
        train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")

        # use scikit-learn (equivalent to split_train_test)
        train_set, test_set = model_selection.train_test_split(housing, test_size=0.2, random_state=42)

    if True:
        # if import to keep the distribution of income_cat
        bins = [0., 1.5, 3.0, 4.5, 6., np.inf]
        test_size = 0.2

        strat_train_set, strat_test_set = stratified_split(housing, cat="median_income", bins=bins, test_size=test_size)

        housing = strat_train_set.copy()

    # ------------------------------
    # Investigate Data
    # ------------------------------
    if False:
        housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
                     s=housing["population"]/100, label="population", figsize=(10, 7),
                     c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
                     )
        plt.legend()
        plt.show()

    # Correlation
    if False:
        corr_matrix = housing.corr()
        print(corr_matrix["median_house_value"].sort_values(ascending=False))

        # Plot correlation as scatter plots for diff attributes
        attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
        pd.plotting.scatter_matrix(housing[attributes], figsize=(12, 8))
        plt.show()

        housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
        plt.show()
        # -> reveals horizontal lines that we may want to remove

    # Attribute Combination
    if False:
        housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
        housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
        # housing["population_per_household"] = housing["population"]/housing["households"]

        if False:
            corr_matrix = housing.corr()
            print(corr_matrix["median_house_value"].sort_values(ascending=False))

    # Preparing Data for machine learning
    if False:
        housing = strat_train_set.drop("median_house_value", axis=1)
    # housing_labels = strat_train_set["median_house_value"].copy()

    # missing values: 3 possibilities
    if False:
        housing.dropna(subset=["total_bedrooms"])       # Get rid of the data
        housing.drop("total_bedrooms", axis=1)          # Get rid of the whole attribute
        median = housing["total_bedrooms"].median()
        housing["total_bedrooms"].fillna(median, inplace=True)  # set missing value to zero/median/mean

    if False:
        # Median of category cannot be calculated -> create a copy without that category
        housing_num = housing.drop("ocean_proximity", axis=1)
        imputer = SimpleImputer(strategy="median")
        imputer.fit(housing_num)
        print(imputer.statistics_, housing_num.median().values)

        # Transform data
        X = imputer.transform(housing_num)

        # Combines the fit and the transform in one action
        imputer.fit_transform(housing_num)

        # Recreate a new DataFrame
        housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)
        if False:
            print(housing_tr)

    # Handle Categorical and text
    if False:
        housing_cat = housing[["ocean_proximity"]]
        print(housing_cat.head(10))
        # convert Cat to number
        ordinal_encoder = OrdinalEncoder()
        housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
        print(ordinal_encoder.categories_)
        print(housing_cat_encoded[:10])
        # Problem with this is that 0 and 1 would be seen as close by the algo but not necessarily true
        # -> Prefer to OneHotEncode: 1 new category for the data per category, and for each of these it's either 1 or 0
        cat_encoder = OneHotEncoder()
        housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
        print(cat_encoder.categories_)
        print(housing_cat_1hot)

    if False:
        # This can be done with pandas directly
        housing = pd.get_dummies(housing, prefix='', prefix_sep='')

    # Custom transformers can be created (like OrdinalEncoder, OneHotEncoder, Imputer,...)
    # -> Create a class with fit() (returning itself), transform() and fit_transform() (not needed if TransformerMixin
    # used as a base class) and if BaseEstimator class -> get_params() and set_params()
    if False:
        attr_adder = CombinedAttributesDivide(housing, add_bedrooms_per_room=False)
        housing_extra_attribs = attr_adder.transform(housing.values)

    # Feature Scaling: fit on training data and then transform training and test set
    # 2 methods:    -> min-max scaling: normalization
    #               -> Standardization
    if False:
        scaler = MinMaxScaler()
        housing_scaled = scaler.fit_transform(housing_extra_attribs)
    if False:
        scaler = StandardScaler()
        housing_scaled = scaler.fit_transform(housing_extra_attribs)
        print(housing_scaled)

    # Pipeline: to organise all transformation of it in a simpler manner
    # On numerical attributes:
    if False:
        num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy="median")),
                ('attribs_adder', CombinedAttributesDivide(housing)),
                ('std_scaler', StandardScaler()),
                ])
        housing_num_tr = num_pipeline.fit_transform(housing_num)
        print(housing_num_tr)
    # To also take care of categorical attributes:
    if True:
        housing_num = housing.drop("ocean_proximity", axis=1)
        num_attribs = list(housing_num)
        cat_attribs = ["ocean_proximity"]
        num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy="median")),
                ('attribs_adder', CombinedAttributesDivide(housing)),
                ('std_scaler', StandardScaler()),
                ])
        full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(), cat_attribs)
            ])
        housing_prepared = full_pipeline.fit_transform(housing)
        print(housing_prepared)


def complete_pipeline(data):
    categories = ["ocean_proximity"]
    data_num = data.drop(categories, axis=1)
    num_attribs = list(data_num)
    num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('attribs_adder', CombinedAttributesDivide()),
            ('std_scaler', StandardScaler()),
            ])

    unique_values = ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(categories=[unique_values]), categories)
        ])

    return full_pipeline


def split_data(housing):
    # split the data and keep the distribution of income_cat
    bins = [0., 1.5, 3.0, 4.5, 6., np.inf]
    test_size = 0.2

    strat_train_set, strat_test_set = stratified_split(housing, cat="median_income", bins=bins, test_size=test_size)

    housing = strat_train_set.copy()

    # Preparing the data
    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"]

    return housing, housing_labels, strat_test_set.drop("median_house_value", axis=1), \
        strat_test_set["median_house_value"]


def train_linear_model(data, labels):
    model = LinearRegression()
    model.fit(data, labels)
    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, os.path.join('models', 'linear.pkl'))
    return model


def train_decision_tree_model(data, labels):
    model = DecisionTreeRegressor()
    model.fit(data, labels)
    # Save Model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, os.path.join('models', 'decision_tree.pkl'))
    return model


def train_random_forest_model(data, labels):
    model = RandomForestRegressor()
    model.fit(data, labels)
    # Save Model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, os.path.join('models', 'random_forest.pkl'))
    return model


def error_predictions(model, data, labels, plot=False):
    predictions = model.predict(data)

    mse = mean_squared_error(labels, predictions)
    rmse = np.sqrt(mse)

    if plot:
        plt.plot(np.array(labels), np.array(predictions), '.', alpha=0.4)
        plt.show()

    return rmse


def random_forest_search(model, data, labels):
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=250, stop=300, num=2)]
    # Number of features to consider at every split
    # max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    # max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    # max_depth.append(None)
    # Minimum number of samples required to split a node
    # min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    # min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    # bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators}

    # Search
    model_random = RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=1000,
                                      cv=2, verbose=2, random_state=42, n_jobs=-1)

    model_random.fit(data, labels)
    return model_random


if __name__ == '__main__':

    DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
    HOUSING_FILE = "housing.tgz"
    HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/" + HOUSING_FILE

    HOUSING_PATH = os.path.join("../data", "housing")
    HOUSING_DATA = os.path.join(HOUSING_PATH, "housing.csv")

    # ------------------------------
    # Downloading Data
    # ------------------------------
    if False:
        fetch_data(HOUSING_URL, HOUSING_PATH, HOUSING_FILE)

    # Loading Data
    housing = load_data(HOUSING_DATA)

    # exploring_data(housing)

    # Split Data
    train, train_labels, test, test_labels = split_data(housing)

    full_pipeline = complete_pipeline(train)
    train_prepared = full_pipeline.fit_transform(train)

    # Select and Train Model
    if False:
        model = train_linear_model(train_prepared, train_labels)
        model = train_decision_tree_model(train_prepared, train_labels)
        model = train_random_forest_model(train_prepared, train_labels)
    else:
        # model_string = 'linear.pkl'
        # model_string = 'decision_tree.pkl'
        model_string = 'random_forest.pkl'
        model = joblib.load(os.path.join('models', model_string))

    # Check Predictions (does not include treatment of the data (pipeline))
    rmse = error_predictions(model, train_prepared, train_labels)
    print(rmse)

    # Cross-Validation of model (split training set in smaller set and check the scores)
    if False:
        scores = cross_val_score(model, train_prepared, train_labels,
                                 scoring="neg_mean_squared_error",
                                 cv=10)         # 10 fold cross-validation: train 10-times the model (split data in 10)
        scores = np.sqrt(-scores)
        display_scores(scores)

    # Now need to fine tune the hyperparameters of the models -> Grid Search
    if False:
        param_grid = [
                {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90], 'max_features': [2, 4, 6, 8, 10, 12]},
                {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
                ]
        grid_search = GridSearchCV(model, param_grid, cv=5,
                                   scoring='neg_mean_squared_error',
                                   return_train_score=True)

        grid_search.fit(train_prepared, train_labels)
        # print(grid_search.best_params_)
        # print(grid_search.best_estimator_)

    # if hyperparameter space is large -> use randomized search: RandomizedSearchCV()
    if False:
        grid_search = random_forest_search(model, train_prepared, train_labels)
        joblib.dump(grid_search, os.path.join('models', 'best_model.pkl'))
        # print(grid_search.best_params_)     # {'n_estimators': 444, 'max_features': 'sqrt', 'bootstrap': False}
    else:
        grid_search = joblib.load(os.path.join('models', 'best_model.pkl'))

    # Another possibility is to combine models -> ensemble methods

    # Inspect the model -> gain insight
    if False:
        feature_importances = grid_search.best_estimator_.feature_importances_
        extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
        cat_encoder = full_pipeline.named_transformers_["cat"]      # gets the part of the pipeline called 'cat'
        cat_one_hot_attribs = list(cat_encoder.categories_[0])

        train_num = train.drop("ocean_proximity", axis=1)
        num_attribs = list(train_num)
        attributes = num_attribs + extra_attribs + cat_one_hot_attribs
        print(list(sorted(zip(feature_importances, attributes), reverse=True)))     # -> only INLAND cat matters

        # Try to look at errors and why it makes them:
        # - adding extra features, removing uninformative, cleaning outliers, ...

    # Evaluate the system on the Test Set
    if True:
        final_model = grid_search.best_estimator_

        X_test = test
        y_test = test_labels

        X_test_prepared = full_pipeline.transform(X_test)
        final_predictions = final_model.predict(X_test_prepared)

        final_mse = mean_squared_error(y_test, final_predictions)
        final_rmse = np.sqrt(final_mse)
        print(final_rmse)

        ci = 0.95
        interval = confidence_interval(ci, final_predictions, y_test)
        print(interval)
