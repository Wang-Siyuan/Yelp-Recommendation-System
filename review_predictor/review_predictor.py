from pprint import pprint
import numpy as np
import itertools
from util import utility as u
from sklearn import linear_model
from autograd import grad
import math
import matplotlib
from matplotlib import style
style.use("ggplot")
from sklearn.cluster import KMeans
from collections import Counter
import pylab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mord as m
from data_model import user as user_module
from data_model import review as review_module
from data_model import business as business_module
from data_model import data_aggregator as data_aggregator_module
from regression import ordinal_logistic as ordinal_logistic_module
from regression import logistic_regression as logistic_regression_module
from regression import ordinal_regression as ordinal_regression_module
from regression import quadratic_loss as quadratic_loss_module
from regression import vectorized_output_regression as vectorized_output_regression_module
from sklearn.decomposition import PCA
import pickle



np.set_printoptions(threshold=np.nan)
USER_DATA_SET_FILE_PATH = 'data_set/yelp_academic_dataset_user.json';

TRAINING_DATA_SET_SIZE = 5000
TEST_DATA_SET_SIZE = 5000
VALIDATION_DATA_SET_SIZE = 0
FEATURE_SIZE = 8

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
#If we decided to normalize
X_training = (X_training - X_training.mean(axis=0)) / X_training.std(axis=0)
X_training=np.ma.compress_cols(np.ma.masked_invalid(X_training))
X_test = (X_test - X_test.mean(axis=0)) / X_test.std(axis=0)
X_test=np.ma.compress_cols(np.ma.masked_invalid(X_test))
pprint(X_training.shape)
pprint(X_test.shape)

U, s, V = np.linalg.svd(X_training, full_matrices=False)
S = np.diag(s)
np.allclose(X_training, np.dot(U[:,:], np.dot(S[:,:], V[:,:])))
print('Finished constructing X,Y data matrix. Step 3/5');


# pca = PCA(n_components= 7)
# X_training=pca.fit_transform(X_training)
# X_test=pca.fit_transform(X_test)

# fig = plt.figure()
# # ax3D = fig.add_subplot(111, projection='3d')
# #
# # ax3D.scatter(X_training[:,0],X_training[:,1],X_training[:,2], zdir='z', s=20)
# # plt.show()
# ax = fig.gca(projection='3d')
# ax.set_title('Visualization of input dataset X after PCA')
# ax.scatter(X_training[:,0], X_training[:,1], X_training[:,2],  # data
#            color='purple',                            # marker colour
#            marker='o',                                # marker shape
#            s=30                                       # marker size
#            )

# plt.show()                                            # render the plot



print('Started clustering with kmeans');

cluster_num = 3
X_overall = np.concatenate((X_training,X_test))
Y_overall = np.concatenate((Y_training,Y_test))
len_X_training = len(X_training)
kmeans = KMeans(n_clusters = cluster_num, random_state=0).fit(X_overall)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

labels_train = labels[:len_X_training]
labels_test = labels[len_X_training:]

print('Finishing kmeans , start assigning index to X_training')
X_training_1 = X_training[labels_train==0]
X_training_2 = X_training[labels_train==1]
X_training_3 = X_training[labels_train==2]
Y_training_1 = Y_training[labels_train==0]
Y_training_2 = Y_training[labels_train==1]
Y_training_3 = Y_training[labels_train==2]

print("size of X_training_1:"+str(len(X_training_1)))
print("size of X_training_2:"+str(len(X_training_2)))
print("size of X_training_3:"+str(len(X_training_3)))

# kmeans = KMeans(n_clusters = cluster_num, random_state=0).fit(X_test)
# centroids = kmeans.cluster_centers_
# labels = kmeans.labels_

X_test_1 = X_test[labels_test==0]
X_test_2 = X_test[labels_test==1]
X_test_3 = X_test[labels_test==2]
Y_test_1 = Y_test[labels_test==0]
Y_test_2 = Y_test[labels_test==1]
Y_test_3 = Y_test[labels_test==2]

print("size of X_test_1:"+str(len(X_test_1)))
print("size of X_test_2:"+str(len(X_test_2)))
print("size of X_test_3:"+str(len(X_test_3)))


# color = ["purple", "r", "b"]
# c = Counter(labels)
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.set_title('Visualization of Input X after kmeans clustering ')
# for i in range(len(X_training)):
#     ax.scatter(X_training[i][0], X_training[i][1], X_training[i][2], c=color[labels[i]])

# ax.scatter(centroids[:, 0],centroids[:, 1], centroids[:, 2], marker = "x", s=150, linewidths = 5, zorder = 100)
# plt.show()

model_1 = logistic_regression_module.LogisticRegression()
model_1.fit(X_training_1, Y_training_1)
pprint('Finished fitting model 1');

model_2 = logistic_regression_module.LogisticRegression()
model_2.fit(X_training_2, Y_training_2)
pprint('Finished fitting model 2');

model_3 = logistic_regression_module.LogisticRegression()
model_3.fit(X_training_3, Y_training_3)
pprint('Finished fitting model 3');

# model_1 = quadratic_loss_module.QuadraticLoss()
# model_1.fit(X_training_1, Y_training_1)
# pprint('Finished fitting model 1');
#
# model_2 = quadratic_loss_module.QuadraticLoss()
# model_2.fit(X_training_2, Y_training_2)
# pprint('Finished fitting model 2');
#
# model_3 = quadratic_loss_module.QuadraticLoss()
# model_3.fit(X_training_3, Y_training_3)
# pprint('Finished fitting model 3');


### fitting model using ordinal regression

# model_1 = ordinal_regression_module.OrdinalRegression()
# model_1.fit(X_training_1, Y_training_1)
# pprint('Finished fitting model 1');
#
# model_2 = ordinal_regression_module.OrdinalRegression()
# model_2.fit(X_training_2, Y_training_2)
# pprint('Finished fitting model 2');
#
# model_3 = ordinal_regression_module.OrdinalRegression()
# model_3.fit(X_training_3, Y_training_3)
# pprint('Finished fitting model 3');


# model_1 = vectorized_output_regression_module.VectorizedOutputRegression()
# model_1.fit(X_training_1, Y_training_1)
# pprint('Finished fitting model 1');
#
# model_2 = vectorized_output_regression_module.VectorizedOutputRegression()
# model_2.fit(X_training_2, Y_training_2)
# pprint('Finished fitting model 2');
#
# model_3 = vectorized_output_regression_module.VectorizedOutputRegression()
# model_3.fit(X_training_3, Y_training_3)
# pprint('Finished fitting model 3');

print('Finished fitting ML model. Step 4/5');


print('Started analyzing in-sample/out-of-sample error/accuracy for the ML model. Step 5/5');
(error_val_1, accuracy_1) = model_1.accuracy_and_error(X_training_1, Y_training_1)
(error_val_2, accuracy_2) = model_2.accuracy_and_error(X_training_2, Y_training_2)
(error_val_3, accuracy_3) = model_3.accuracy_and_error(X_training_3, Y_training_3)
accuracy=(accuracy_1*len(X_training_1)+accuracy_2*len(X_training_2)+accuracy_3*len(X_training_3))/len(X_training)
error = (error_val_1*len(X_training_1)+error_val_2*len(X_training_2)+error_val_3*len(X_training_3))/len(X_training)

pprint("In sample error is: " + str(error) + ", accuracy is " + str(accuracy))

(error_val_1, accuracy_1) = model_1.accuracy_and_error(X_test_1, Y_test_1)
(error_val_2, accuracy_2) = model_2.accuracy_and_error(X_test_2, Y_test_2)
(error_val_3, accuracy_3) = model_3.accuracy_and_error(X_test_3, Y_test_3)
accuracy=(accuracy_1*len(X_test_1)+accuracy_2*len(X_test_2)+accuracy_3*len(X_test_3))/len(X_test)
error = (error_val_1*len(X_test_1)+error_val_2*len(X_test_2)+error_val_3*len(X_test_3))/len(X_test)
pprint("Out of sample error is: " + str(error) + ", accuracy is " + str(accuracy))
print('Finished analyzing in-sample/out-of-sample error/accuracy for the ML model. Step 5/5');
