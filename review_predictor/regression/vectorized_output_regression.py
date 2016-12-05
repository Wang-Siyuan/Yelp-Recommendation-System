from sklearn import linear_model
from util import utility as u
import numpy as np
from pprint import pprint
import math
import sys


class VectorizedOutputRegression:
    def __init__(self,FEATURE_SIZE, X, Y):
        self.weights = np.array([ 0.47991533,  0.46137807,  0.41392526,  0.33557923,  0.13173285,
        0.24962059,  0.26108967])
        self.X = X
        self.Y = Y

    def fit(self, X, Y):
        Y_vectorized = np.zeros((Y.shape[0],5))
        for i in range(1,Y.shape[0]):
            Y_vectorized[i,:] = u.convert_y_to_vector(Y[i])
        step_size = 0.001
        max_iters = 40;
        w = self.weights;
        # pprint("gradient");
        for i in range(max_iters):
            if i % 10 == 0:
                print('Iteration %-4d | Loss: %.4f' % (i, self.custom_training_data_total_loss(w)))
                # pprint(np.dot(self.X,self.weights))
                # pprint(self.Y)
            # pprint(self.mygrad(self.weights))
            self.weights -= self.mygrad(self.weights) * step_size
            # pprint(self.weights)

    def accuracy_and_error(self, X_test, Y_test):
        Y_test_vectorized = np.zeros((Y_test.shape[0],5))
        for i in range(1,Y_test.shape[0]):
            Y_test_vectorized[i,:] = u.convert_y_to_vector(Y_test[i])
        error_val = 0
        prediction_match_count = 0
        error = 0;
        for i in range(0,X_test.shape[0]-1):
          predicted_review_result = self.predict(self.weights, X_test[i,:].reshape(1, -1));
          actual_review_result = Y_test[i,0];
          if u.convert_y_to_discrete_output(predicted_review_result) <= actual_review_result:
            prediction_match_count += 1
          error += (predicted_review_result - actual_review_result)**2;
        error /= X_test.shape[0];
        accuracy = prediction_match_count/X_test.shape[0]
        return (error, accuracy)


    def custom_binary_loss(self, y_single, y_predicted_single):
        return math.log(1+math.exp(-y_single*y_predicted_single))

    def custom_vector_loss(self, y_vectorized, y_predicted_vectorized):
        total_loss = 0
        for i in range(0,4):
            total_loss += self.custom_binary_loss(y_vectorized[0,i], y_predicted_vectorized[0,i])
        return total_loss

    def custom_loss_by_vectorization(self, y, y_predicted):
        return self.custom_vector_loss(u.convert_y_to_vector(y), u.convert_y_to_vector(y_predicted))

    def custom_training_data_total_loss(self, w):
        all_predicted_y = np.dot(self.X,w)
        total_loss = 0
        for i in range(1,all_predicted_y.shape[0]):
            predicted_y = all_predicted_y[i]
            # pprint(all_predicted_y)
            # pprint(Y_training[i,0])
            # if predicted_y < 5:
            # pprint(predicted_y)
            # pprint(float(predicted_y))
            new_Loss = self.custom_loss_by_vectorization(self.Y[i,0], predicted_y)
            total_loss += new_Loss
            # pprint(new_Loss)
        return total_loss

    def mygrad(self, w):
        delta_initial = 0.001
        gradient = np.zeros(7)
        for i in range(0,7):
            for multiplier in range(1,50):
                delta = delta_initial*(2**multiplier)
                # pprint(delta)
                if self.custom_training_data_total_loss(w+delta) != self.custom_training_data_total_loss(w):
                    gradient[i] = (self.custom_training_data_total_loss(w+delta) - self.custom_training_data_total_loss(w))/delta
                    break
                elif multiplier == 50:
                    pprint("reached maximum delta")
                    pprint(delta)
        return gradient

    def predict(self, w, X):
        y_predicted = np.dot(self.X,w)
        min_val = 0
        min_loss_val = sys.maxsize
        for i in range(1,5):
            if self.custom_loss_by_vectorization(i,y_predicted[i]) < min_loss_val:
                min_val = i
                min_loss_val = self.custom_loss_by_vectorization(i,y_predicted[i])
        return min_val


