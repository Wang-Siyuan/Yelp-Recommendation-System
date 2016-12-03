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

pprint(u.str_2_int(True));
pprint(u.str_2_int("True"));
pprint(u.str_2_int("true"));
pprint(u.str_2_int(False));
pprint(u.str_2_int("False"));
pprint(u.str_2_int("false"));
pprint(u.str_2_int("a"));

pprint(u.convert_y_to_vector(1))
pprint(u.convert_y_to_vector(2))
pprint(u.convert_y_to_vector(3))
pprint(u.convert_y_to_vector(4))
pprint(u.convert_y_to_vector(5))

def custom_binary_loss(y_single, y_predicted_single):
    return math.log(1+math.exp(-10*y_single*y_predicted_single))

def custom_vector_loss(y_vectorized, y_predicted_vectorized):
    total_loss = 0
    for i in range(0,4):
        total_loss += custom_binary_loss(y_vectorized[0,i], y_predicted_vectorized[0,i])
    return total_loss

def custom_loss_by_vectorization(y, y_predicted):
    return custom_vector_loss(u.convert_y_to_vector(y), u.convert_y_to_vector(y_predicted))

def custom_training_data_total_loss(w):
    all_predicted_y = np.dot(X_training,w)
    total_loss = 0
    for i in range(1,all_predicted_y.shape[0]):
        predicted_y = all_predicted_y[i]
        total_loss += custom_loss_by_vectorization(Y_training[i,:], predicted_y)
    return total_loss

pprint(custom_binary_loss(1,1))
pprint(custom_binary_loss(1,-1))
pprint(custom_binary_loss(-1,1))
pprint(custom_binary_loss(-1,-1))
pprint("")
pprint(custom_vector_loss(np.array([1,1,1,1,1]).reshape(1,5),np.array([1,1,1,1,1]).reshape(1,5)))
pprint(custom_vector_loss(np.array([1,-1,-1,-1,-1]).reshape(1,5),np.array([1,-1,-1,-1,-1]).reshape(1,5)))
pprint(custom_vector_loss(np.array([1,-1,-1,-1,-1]).reshape(1,5),np.array([1,-1,-1,-1,-1]).reshape(1,5)))
pprint(custom_vector_loss(np.array([1,-1,-1,-1,-1]).reshape(1,5),np.array([1,1,-1,-1,-1]).reshape(1,5)))
pprint(custom_vector_loss(np.array([1,-1,-1,-1,-1]).reshape(1,5),np.array([1,1,1,-1,-1]).reshape(1,5)))
pprint(custom_vector_loss(np.array([1,-1,-1,-1,-1]).reshape(1,5),np.array([1,1,1,1,-1]).reshape(1,5)))
pprint(custom_vector_loss(np.array([1,-1,-1,-1,-1]).reshape(1,5),np.array([1,1,1,1,1]).reshape(1,5)))
pprint(custom_vector_loss(np.array([1,1,-1,-1,-1]).reshape(1,5),np.array([1,1,1,1,1]).reshape(1,5)))
pprint("")
X_training = np.array([1,2,3,4,5,6]).reshape(3,2);
w = np.random.rand(X_training.shape[1])
all_predicted_y = np.dot(X_training,w)
pprint(all_predicted_y)
pprint(type(all_predicted_y[0]))


