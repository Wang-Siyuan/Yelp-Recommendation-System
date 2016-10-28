from pprint import pprint
import numpy as np
import itertools
from util import utility as u
from sklearn import linear_model
from model import user as user_module
from model import review as review_module
from model import business as business_module

np.set_printoptions(threshold=np.nan)
USER_DATA_SET_FILE_PATH = 'data_set/yelp_academic_dataset_user.json';
BUSINESS_DATA_SET_FILE_PATH = 'data_set/yelp_academic_dataset_business_restaurants_only.json';
REVIEW_DATA_SET_FILE_PATH = 'data_set/yelp_academic_dataset_review.json';
# REVIEW_DATA_SET_FILE_PATH = 'data_set/yelp_academic_dataset_review_test.json';


TRAINING_DATA_SET_SIZE = 40000;
VALIDATION_DATA_SET_SIZE = 40000;
TEST_DATA_SET_SIZE = 40000;
TOTAL_DATA_SET_SIZE = TRAINING_DATA_SET_SIZE + VALIDATION_DATA_SET_SIZE + TEST_DATA_SET_SIZE;
FEATURE_SIZE = 46;

print('Started loading business data set. Step 1/6');
business = business_module.Business(BUSINESS_DATA_SET_FILE_PATH);
print('Finished loading business data set. Step 1/6');


print('Started loading review data set. Step 2/6');
review = review_module.Review(REVIEW_DATA_SET_FILE_PATH, business.getBusinessDataDict());
print('Finished loading review data set. Step 2/6');
pprint('Important: ' + str(u.count_iterable(review.getReviewData())));


print('Started loading user data set. Step 3/6');
user = user_module.User(USER_DATA_SET_FILE_PATH, review.getUserIdToBusinessIdMap(), business.getBusinessDataDict());
print('Finished loading user data set. Step 3/6');


print('Started constructing X,Y data matrix. Step 4/6');
X = np.zeros((TRAINING_DATA_SET_SIZE, FEATURE_SIZE));
Y = np.zeros((TRAINING_DATA_SET_SIZE, 1));
X_validation = np.zeros((VALIDATION_DATA_SET_SIZE, FEATURE_SIZE));
Y_validation = np.zeros((VALIDATION_DATA_SET_SIZE, ));
X_test = np.zeros((TEST_DATA_SET_SIZE, FEATURE_SIZE));
Y_test = np.zeros((TEST_DATA_SET_SIZE, 1));

training_review_data = [];
validation_review_data = [];
test_review_data = [];


for i,review_data_entry in enumerate(review.getReviewData()):
	if i < TRAINING_DATA_SET_SIZE:
		training_review_data.append(review_data_entry);
	elif i < (TRAINING_DATA_SET_SIZE + VALIDATION_DATA_SET_SIZE):
		validation_review_data.append(review_data_entry);
	elif i < (TRAINING_DATA_SET_SIZE + VALIDATION_DATA_SET_SIZE + TEST_DATA_SET_SIZE):
		test_review_data.append(review_data_entry);
	else:
		break;
pprint('Training data size: ' + str(u.count_iterable(training_review_data)));
pprint('Validation data size: ' + str(u.count_iterable(validation_review_data)));
pprint('Test data size: ' + str(u.count_iterable(test_review_data)));


for i,review_data_entry in enumerate(validation_review_data):
	user_id = review_data_entry['user_id'];
	user_matrix = user.populate_user_data(user_id);
	business_id = review_data_entry['business_id'];
	business_matrix = business.populate_business_data(user, user_id, business_id);
	# print(user_matrix.shape);
	# print(business_matrix.shape);
	X_validation[i,:] = np.concatenate((user_matrix, business_matrix), axis=1);
	Y_validation[i] = review_data_entry['stars'];
# pprint(X_validation.sum(axis=0))
X_validation_normed = (X_validation - X_validation.mean(axis=0)) / X_validation.std(axis=0);

for i,review_data_entry in enumerate(training_review_data):
	user_id = review_data_entry['user_id'];
	user_matrix = user.populate_user_data(user_id);
	business_id = review_data_entry['business_id'];
	business_matrix = business.populate_business_data(user, user_id, business_id);
	# print(user_matrix.shape);
	# print(business_matrix.shape);
	X[i,:] = np.concatenate((user_matrix, business_matrix), axis=1);
	Y[i] = review_data_entry['stars'];
# pprint(X.sum(axis=0))
X_normed = (X - X.mean(axis=0)) / X.std(axis=0);

for i,review_data_entry in enumerate(test_review_data):
	# pprint(review_data_entry);
	user_id = review_data_entry['user_id'];
	user_matrix = user.populate_user_data(user_id);
	business_id = review_data_entry['business_id'];
	business_matrix = business.populate_business_data(user, user_id, business_id);
	# print(user_matrix.shape);
	# print(business_matrix.shape);
	X_test[i,:] = np.concatenate((user_matrix, business_matrix), axis=1);
	Y_test[i] = review_data_entry['stars'];
# pprint(X_test.sum(axis=0))
X_test_normed = (X_test - X_test.mean(axis=0)) / X_test.std(axis=0);

# pprint(X)
# pprint(X.sum(axis=0))
# pprint(Y)
# pprint(X_normed);
print('Finished constructing X,Y data matrix. Step 4/6');

print('Started fitting ML model. Step 5/6');
# Use cross validation to find the appropriate alpha
reg = linear_model.LassoCV(eps=0.01,n_alphas=3,n_jobs=-1,precompute=True,max_iter=100);
# pprint(X_validation_normed);
# pprint(Y_validation);
reg.fit (X_validation_normed, Y_validation);
chosen_alpha = reg.alpha_;

# Ridge Regression
# reg = linear_model.Ridge (alpha = .5)
# reg.fit (X_normed, Y);

# Lasso Regularizer
reg = linear_model.Lasso(alpha = chosen_alpha);
reg.fit (X_normed, Y); 
print(reg.coef_);
print(reg.intercept_);
print('Finished fitting ML model. Step 5/6');

print('Started predicting using ML model. Step 6/6');
in_sample_error = 0;
for i in range(0,TRAINING_DATA_SET_SIZE-1):
	in_sample_error += (reg.predict(X[i,:].reshape(1, -1)) - Y[i,0])**2;
in_sample_error /= TRAINING_DATA_SET_SIZE;
print('In sample error is: ' + str(in_sample_error));

out_of_sample_error = 0;
for i in range(0,TEST_DATA_SET_SIZE-1):
	out_of_sample_error += (reg.predict(X_test[i,:].reshape(1, -1)) - Y_test[i,0])**2;
out_of_sample_error /= TEST_DATA_SET_SIZE;
print('Out of sample error is: ' + str(out_of_sample_error));
print('Finished predicting using ML model. Step 6/6');
