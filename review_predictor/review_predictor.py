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
from sklearn.decomposition import PCA

import plotly.plotly as py
import plotly.graph_objs as go

#from matplotlib import pyplot
#from mpl_toolkits.mplot3d import axes3D

np.set_printoptions(threshold=np.nan)
USER_DATA_SET_FILE_PATH = 'data_set/yelp_academic_dataset_user.json';
BUSINESS_DATA_SET_FILE_PATH = 'data_set/yelp_academic_dataset_business_restaurants_only.json';
# REVIEW_DATA_SET_FILE_PATH = 'data_set/yelp_academic_dataset_review.json';
REVIEW_DATA_SET_FILE_PATH = 'data_set/yelp_academic_dataset_review_test.json';


TRAINING_DATA_SET_SIZE = 4000;
TEST_DATA_SET_SIZE = 4000;
TOTAL_DATA_SET_SIZE = TRAINING_DATA_SET_SIZE + TEST_DATA_SET_SIZE;
FEATURE_SIZE = 45;

print('Started loading business data set. Step 1/6');
business = business_module.Business(BUSINESS_DATA_SET_FILE_PATH);
# business.generateHistogram();
print('Finished loading business data set. Step 1/6');


print('Started loading review data set. Step 2/6');
review = review_module.Review(REVIEW_DATA_SET_FILE_PATH, business.getBusinessDataDict());
# review.generateStarsHistogram();
print('Finished loading review data set. Step 2/6');
print('Total restaurant review count after filtering: ' + str(u.count_iterable(review.getReviewData())));


print('Started loading user data set. Step 3/6');
user = user_module.User(USER_DATA_SET_FILE_PATH, review.getUserIdToBusinessIdMap(), business.getBusinessDataDict());
# user.generateStarsHistogram();
print('Finished loading user data set. Step 3/6');


print('Started constructing X,Y data matrix. Step 4/6');


all_review_data = [];

for i,review_data_entry in enumerate(review.getReviewData()):
    if i < TOTAL_DATA_SET_SIZE:
        all_review_data.append(review_data_entry);
    else:
        break;

print('Total data size: ' + str(u.count_iterable(all_review_data)));

X = np.zeros((len(all_review_data), FEATURE_SIZE));
Y = np.zeros((len(all_review_data), 1));
for i,review_data_entry in enumerate(all_review_data):
    user_id = review_data_entry['user_id'];
    user_matrix = user.populate_user_data(user_id);
    business_id = review_data_entry['business_id'];
    business_matrix = business.populate_business_data(user, user_id, business_id);
    # print(user_matrix.shape);
    # print(business_matrix.shape);
    X[i,:] = np.concatenate((user_matrix, business_matrix), axis=1);
    Y[i] = review_data_entry['stars'];


X_normed = (X - X.mean(axis=0)) / X.std(axis=0); #If we decided to normalize

#X_normed = X; #If we decided not to normalize
X_normed=np.ma.compress_cols(np.ma.masked_invalid(X_normed))
<<<<<<< 4db7f3384f2b492cd66313242eca4cbd1df092ff
pca = PCA(n_components= 3)
X_normed=pca.fit_transform(X_normed)

#Axes3D.scatter(X_normed[:,1],X_normed[:,2],X_normed[:,3], zdir='z', s=20)

trace = go.Scatter(
    x = X_normed[:,1],
    y = X_normed[:,2],
	z = X_normed[:,3],
    mode = 'markers'
)

data = [trace]

# Plot and embed in ipython notebook!
py.iplot(data, filename='basic-scatter')


U, s, V = np.linalg.svd(X_normed, full_matrices=False)
S = np.diag(s)
np.allclose(X_normed, np.dot(U[:,:], np.dot(S[:,:], V[:,:])))

print(X_normed.shape)
X_training = X_normed[0:TRAINING_DATA_SET_SIZE,:]
Y_training = Y[0:TRAINING_DATA_SET_SIZE,:]
X_test = X_normed[TRAINING_DATA_SET_SIZE:X_normed.shape[0],:]
Y_test = Y[TRAINING_DATA_SET_SIZE:X_normed.shape[0],:]
print(X_training.shape)
print(Y_training.shape)
print(X_test.shape)
print(Y_test.shape)

print('Finished constructing X,Y data matrix. Step 4/6');
print('Started fitting ML model. Step 5/6');


################ Ordinal Loss Approach #####################
c = m.OrdinalRidge();
c.fit(X_training, Y_training);

################ Vectorize Y and use sum of binary loss function as objective function #####################
# Y_training_vectorized = np.zeros((Y_training.shape[0],5))
# Y_test_vectorized = np.zeros((Y_test.shape[0],5))
# pprint(Y_training_vectorized.shape)
# for i in range(1,Y_training.shape[0]):
#     Y_training_vectorized[i,:] = u.convert_y_to_vector(Y_training[i])
#     Y_test_vectorized[i,:] = u.convert_y_to_vector(Y_test[i])


# def custom_binary_loss(y_single, y_predicted_single):
#     return math.log(1+math.exp(-10*y_single*y_predicted_single))

# def custom_vector_loss(y_vectorized, y_predicted_vectorized):
#     total_loss = 0
#     for i in range(0,4):
#         total_loss += custom_binary_loss(y_vectorized[0,i], y_predicted_vectorized[0,i])
#     return total_loss

# def custom_loss_by_vectorization(y, y_predicted):
#     return custom_vector_loss(u.convert_y_to_vector(y), u.convert_y_to_vector(y_predicted))

# def custom_training_data_total_loss(w):
#     # pprint("w")
#     # pprint(w)
#     # pprint(X_training)
#     all_predicted_y = np.dot(X_training,w)
#     # pprint(all_predicted_y.shape)
#     pprint(all_predicted_y)
#     total_loss = 0
#     for i in range(1,all_predicted_y.shape[0]):
#         predicted_y = all_predicted_y[i]
#         # pprint(all_predicted_y)
#         # pprint(Y_training[i,0])
#         # if predicted_y < 5:
#         pprint("lol");
#         pprint(predicted_y)
#         pprint(float(predicted_y))
#         # pprint(u.convert_y_to_vector(predicted_y))
#         total_loss += custom_loss_by_vectorization(Y_training[i,0], predicted_y)
#     return total_loss

# def predict(w,X):
#     y_predicted = np.dot(X,w)
#     min_val = 0
#     min_loss_val = sys.maxsize
#     for i in range(1,5):
#         if custom_loss_by_vectorization(i,y_predicted) < min_loss_val:
#             min_val = i
#             min_loss_val = custom_loss_by_vectorization(i,y_predicted)
#     return min_val


    
# gradient = grad(custom_training_data_total_loss)

# weights = np.random.rand(X_training.shape[1])
# step_size = 0.01
# max_iters = 100;
# # for i in range(max_iters):
#     # if i % 10 == 0:
# print('Iteration %-4d | Loss: %.4f' % (1, custom_training_data_total_loss(weights)))
# weights -= gradient(weights) * step_size





################ Purely Logistic Loss Approach #####################
# # Use cross validation to find the appropriate alpha
# model = linear_model.LogisticRegressionCV(
#         Cs=9
#         ,penalty='l2'
#         ,scoring='roc_auc'
#         ,cv=5
#         ,n_jobs=-1
#         ,max_iter=10000
#         ,fit_intercept=True
#         ,tol=10
#     );
# model.fit (X, Y.ravel());
# pprint(model.coef_);
# print('Finished fitting ML model. Step 5/6');

print('Started predicting using ML model. Step 6/6');

################ Vectorize Y and use sum of binary loss function as objective function #####################
# successCount = 0;
# for i in range(X_test.shape[0]):
#     if predict(weights, X_test[i]) == Y_test[i]:
#         successCount += 1
# pprint(successCount)
# pprint(successCount/X_test.shape[0])

################ Ordinal Loss Approach #####################
error_count = 0;
for i in range(1,X_test.shape[0]):
  Y_predicted = c.predict(X_test[i,:].reshape(1,-1))
  if Y_predicted != Y_test[i]:
      error_count += 1
      # pprint(str(Y_predicted) + " : " + str(Y_test[i]))
pprint(error_count);

################ Purely Logistic Loss Approach #####################
# in_sample_error = 0;
# for i in range(0,TRAINING_DATA_SET_SIZE-1):
#   predicted_review_result = model.predict(X[i,:].reshape(1, -1));
#   actual_review_result = Y[i,0];
#   if i < 50:
#       print(str(predicted_review_result) + ',' + str(actual_review_result));
#   in_sample_error += (predicted_review_result - actual_review_result)**2;
# in_sample_error /= TRAINING_DATA_SET_SIZE;
# print('In sample error is: ' + str(in_sample_error));
# print('Mean accuracy: ' + str(model.score(X,Y)));

# out_of_sample_error = 0;
# for i in range(0,TEST_DATA_SET_SIZE-1):
#   predicted_review_result = model.predict(X_test[i,:].reshape(1, -1));
#   actual_review_result = Y_test[i,0];
#   if i < 50:
#       print(str(predicted_review_result) + ',' + str(actual_review_result));
#   out_of_sample_error += (predicted_review_result - actual_review_result)**2;
# out_of_sample_error /= TEST_DATA_SET_SIZE;
# print('Out of sample error is: ' + str(out_of_sample_error));
# print('Mean accuracy: ' + str(model.score(X_test,Y_test)));
print('Finished predicting using ML model. Step 6/6');
