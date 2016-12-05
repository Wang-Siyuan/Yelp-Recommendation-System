from sklearn import linear_model
from util import utility as u
from autograd import grad
import numpy as np

class VectorizedOutputRegression:
	def __init__(self):
		self.weights = np.random.rand(X_training.shape[1])

	def fit(self, X, Y_training):
		Y_training_vectorized = np.zeros((Y_training.shape[0],5))
		pprint(Y_training_vectorized.shape)
		for i in range(1,Y_training.shape[0]):
    		Y_training_vectorized[i,:] = u.convert_y_to_vector(Y_training[i])

		gradient = grad(custom_training_data_total_loss)
		step_size = 0.01
		max_iters = 100;
		for i in range(max_iters):
		    if i % 10 == 0:
				print('Iteration %-4d | Loss: %.4f' % (1, custom_training_data_total_loss(self.weights)))
			self.weights -= gradient(self.weights) * step_size

	def accuracy_and_error(self, X_test, Y_test):
		Y_test_vectorized = np.zeros((Y_test.shape[0],5))
		for i in range(1,Y_test.shape[0]):
    		Y_test_vectorized[i,:] = u.convert_y_to_vector(Y_test[i])
		error_val = 0
		prediction_match_count = 0
		for i in range(0,X.shape[0]-1):
		  predicted_review_result = self.predict(self.weights, X[i,:].reshape(1, -1));
  		  actual_review_result = Y[i,0];
		  if u.convert_y_to_discrete_output(predicted_review_result) == actual_review_result = Y[i,0]:
		  	prediction_match_count += 1
		  in_sample_error += (predicted_review_result - actual_review_result)**2;
		in_sample_error /= X.shape[0];
		accuracy = prediction_match_count/X.shape[0]
		return (in_sample_error, accuracy)


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
    # pprint("w")
    # pprint(w)
    # pprint(X_training)
    all_predicted_y = np.dot(X_training,w)
    # pprint(all_predicted_y.shape)
    pprint(all_predicted_y)
    total_loss = 0
    for i in range(1,all_predicted_y.shape[0]):
        predicted_y = all_predicted_y[i]
        # pprint(all_predicted_y)
        # pprint(Y_training[i,0])
        # if predicted_y < 5:
        pprint("lol");
        pprint(predicted_y)
        pprint(float(predicted_y))
        # pprint(u.convert_y_to_vector(predicted_y))
        total_loss += custom_loss_by_vectorization(Y_training[i,0], predicted_y)
    return total_loss

def predict(w,X):
    y_predicted = np.dot(X,w)
    min_val = 0
    min_loss_val = sys.maxsize
    for i in range(1,5):
        if custom_loss_by_vectorization(i,y_predicted) < min_loss_val:
            min_val = i
            min_loss_val = custom_loss_by_vectorization(i,y_predicted)
    return min_val


