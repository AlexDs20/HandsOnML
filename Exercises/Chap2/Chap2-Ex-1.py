from sklearn.impute import SimpleImputer
from pandas.plotting import scatter_matrix
from zlib import crc32
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit


# Load data
fileName = 'housing.csv'
datapath = os.path.join("..", "datasets", "housing", "")

housing = pd.read_csv(datapath + fileName)
housing.head()
housing.info()
housing.describe()
housing['ocean_proximity'].value_counts()

housing.hist(bins=30)

# See positions of measurements
var1 = 'latitude'
var2 = 'longitude'
var3 = 'median_house_value'
size = 5*housing[var3]/np.mean(housing[var3])
color = housing[var3]

h = plt.scatter(housing[var1], housing[var2], s=size,
                c=color, alpha=0.2)
plt.colorbar(h)
plt.title(var3)
plt.xlabel(var1)
plt.ylabel(var2)
plt.show()

# =====TEST SET======


def split_train_test(data, testRatio=0.2):
    np.random.seed(42)
    shuffleIndices = np.random.permutation(len(data))
    testSize = int(len(data) * testRatio)
    testIndices = shuffleIndices[:testSize]
    trainIndices = shuffleIndices[testSize:]
    return data.iloc[trainIndices], data.iloc[testIndices]


def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32


def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


# This selection will not select the same data if we add new data to the set
# trainData, testData = split_train_test(housing, 0.2)

# This selection will not select any of the previous test data after increasing size of the dataset

housing_with_id = housing.reset_index()   # adds an `index` column
trainSet, testSet = split_train_test_by_id(housing_with_id, 0.2, "index")

trainSet, testSet = train_test_split(housing, random_state=42, test_size=0.2)

housing["median_income"].hist()

housing["income_cat"] = pd.cut(housing["median_income"], bins=[
                               0., 1.5, 3., 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])

housing["income_cat"]
#   housing.drop(columns=["income_cat"])
#   housing.pop("income_cat")
housing["income_cat"].hist(density=True)

# Do stratified sampling (keep same proportion for the trainSet)
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing["income_cat"]):
    stratTrainSet = housing.loc[train_index]
    stratTestSet = housing.loc[test_index]

stratTrainSet["income_cat"].hist(density=True)
stratTestSet["income_cat"].hist(density=True)

stratTestSet["income_cat"].value_counts()/len(stratTestSet)
stratTrainSet["income_cat"].value_counts()/len(stratTrainSet)

stratTestSet.describe()
stratTrainSet.describe()

# Remove the category as it is not usefull
for set_ in (stratTrainSet, stratTestSet):
    set_.drop("income_cat", axis=1, inplace=True)


# only work on train set
housing = stratTrainSet.copy()
housing

# See positions of measurements
var1 = 'latitude'
var2 = 'longitude'
var3 = 'median_house_value'
size = housing["population"]/100
color = housing[var3]
#   color = (10000*housing["median_income"])/housing[var3]

plt.figure(9)
h = plt.scatter(housing[var1], housing[var2], s=size, label="Population",
                c=np.log10(color), alpha=0.4, cmap=plt.get_cmap("jet"))
plt.colorbar(h, label=var3)
plt.title(var3)
plt.legend()
plt.xlabel(var1)
plt.ylabel(var2)
plt.show()

# Correlation matrix for the different categories.
corr_matrix = housing.corr()
housing.columns

plt.figure(2)
plt.pcolor(corr_matrix)


# More detailed correlation plot for some fields

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]

scatter_matrix(housing[attributes], figsize=(12, 8))

plt.figure(3)
housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)

# Can create new attributes

housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]

# See positions of measurements
var1 = 'latitude'
var2 = 'longitude'
var3 = 'population_per_household'
size = housing["population"]/100
color = housing[var3]
#   color = (10000*housing["median_income"])/housing[var3]

plt.figure(9, figsize=(8, 8))
h = plt.scatter(housing[var1], housing[var2], s=size, label="Population",
                c=np.log10(color), alpha=0.4, cmap=plt.get_cmap("jet"))
plt.colorbar(h, label=var3)
plt.title(var3)
plt.legend()
plt.xlabel(var1)
plt.ylabel(var2)
plt.show()

corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

housing = stratTrainSet.drop("median_house_value", axis=1)
housing_labels = stratTrainSet["median_house_value"].copy()

# Create functions to clean the data
# Fix missing values: 3 possible ways:
if 0:           # Drop only those who don't have a total bedrooms value
    housing = housing.dropna(subset=["total_bedrooms"])
if 0:           # Drop the total bedroom attribue
    housing = housing.drop("total_bedrooms", axis=1)
if 1:           # the same median value should also be put in the test set!
    median = housing["total_bedrooms"].median()
    housing["total_bedrooms"].fillna(median, inplace=True)

housing.describe()

# -----------------
# Scikit has a way to handle missing values:

imputer = SimpleImputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)

imputer.fit(housing_num)
imputer.statistics_
housing_num.median().values

# replace the missing values using the imputer
X = imputer.transform(housing_num)
# X is a numpy array
housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing_num.index)
