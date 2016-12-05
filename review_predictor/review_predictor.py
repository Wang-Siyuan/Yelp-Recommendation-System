from pprint import pprint
import numpy as np
import itertools
from util import utility as u
from sklearn import linear_model
from autograd import grad
import math
import mord as m
from data_model import user as user_module
from data_model import review as review_module
from data_model import business as business_module
from data_model import data_aggregator as data_aggregator_module
from regression import logistic_regression as logistic_regression_module
from regression import ordinal_regression as ordinal_regression_module
from regression import quadratic_loss as quadratic_loss_module
from sklearn.decomposition import PCA
import pickle
import plotly.plotly as py
import plotly.graph_objs as go


np.set_printoptions(threshold=np.nan)
USER_DATA_SET_FILE_PATH = 'data_set/yelp_academic_dataset_user.json';

TRAINING_DATA_SET_SIZE = 1000
TEST_DATA_SET_SIZE = 1000
VALIDATION_DATA_SET_SIZE = 0

print('Started loading business, user, review data set. Step 1/5');
user = user_module.User(USER_DATA_SET_FILE_PATH);
user_data_dict = user.getUserDataDict()
with open('./business_data_dict.pickle', 'rb') as f:
    business_data_dict = pickle.load(f)
with open('./indexed_review_data.pickle', 'rb') as f:
    indexed_review_data = pickle.load(f)
with open('./top_user_review_count_dict.pickle', 'rb') as f:
    top_user_review_count_dict = pickle.load(f)
with open('./user_Id_to_business_Id_map.pickle', 'rb') as f:
    user_Id_to_business_Id_map = pickle.load(f)
print('Finished loading business, user, review data set. Step 1/5');


print('Started constructing X,Y data matrix. Step 2/5');
data_aggregator = data_aggregator_module.DataAggregator(TRAINING_DATA_SET_SIZE,TEST_DATA_SET_SIZE,VALIDATION_DATA_SET_SIZE)
(X_training, Y_training, X_test, Y_test) = data_aggregator.generateDataset(business_data_dict, user_data_dict, indexed_review_data, top_user_review_count_dict, user_Id_to_business_Id_map)
pprint(len(X_training))
pprint(len(Y_training))
print('Finished constructing X,Y data matrix. Step 3/5');


print('Started transforming X,Y data matrix. Step 3/5');
# Enter the transformation here
print('Finished transforming X,Y data matrix. Step 3/5');



print('Started fitting ML model. Step 4/5');
# model = logistic_regression_module.LogisticRegression()
# model = ordinal_regression_module.OrdinalRegression()
model = quadratic_loss_module.QuadraticLoss()
model.fit(X_training, Y_training)
print('Finished fitting ML model. Step 4/5');


print('Started analyzing in-sample/out-of-sample error/accuracy for the ML model. Step 5/5');
(error_val, accuracy) = model.accuracy_and_error(X_training, Y_training)
pprint("In sample error is: " + str(error_val) + ", accuracy is " + str(accuracy))
(error_val, accuracy) = model.accuracy_and_error(X_test, Y_test)
pprint("Out of sample error is: " + str(error_val) + ", accuracy is " + str(accuracy))
print('Finished analyzing in-sample/out-of-sample error/accuracy for the ML model. Step 5/5');
