from sklearn import linear_model
from util import utility as u
import mord as m
from sklearn.model_selection import train_test_split
import numpy as np
import math

class OrdinalRegression:
    # def __init__(self):
    def fit(self, X, Y):
      # best_alpha_list = {}
      # for counter in range(1,10):
      #   X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
      #   best_score = 0
      #   best_alpha = 0
      #   for i in range(-4,4):
      #     alpha_value = 10**(i)
      #     c = m.OrdinalRidge(
      #       alpha=alpha_value,
      #       fit_intercept=True,
      #       normalize=True,
      #       copy_X=True,
      #       max_iter=1000,
      #       tol=0.001,
      #       solver='auto')
      #     c.fit(X_train, y_train)
      #     score = c.score(X_test,y_test)
      #     if score > best_score:
      #       best_alpha = alpha_value
      #       best_score = score
      #       if best_alpha not in best_alpha_list:
      #         best_alpha_list[best_alpha] = 0
      #         best_alpha_list[best_alpha] += 1
      # alpha_to_use = 0
      # highest_frequency = 0
      # for best_alpha in best_alpha_list:
      #     if best_alpha_list[best_alpha] > highest_frequency:
      #         highest_frequency = best_alpha_list[best_alpha]
      #         alpha_to_use = best_alpha_list
      # self.alpha = alpha_to_use
      # print(self.alpha)
      # print(self.alpha)
      self.c = m.OrdinalRidge(
        alpha = 1,
        fit_intercept=True,
        normalize=True,
        copy_X=True,
        max_iter=1000,
        tol=0.001,
        solver='auto')
      self.c.fit(X, Y)


    def accuracy_and_error(self, X, Y):
        error_val = 0
        prediction_match_count = 0
        for i in range(0,X.shape[0]-1):
          predicted_review_result = self.c.predict(X[i,:].reshape(1, -1));
          actual_review_result = Y[i,0];
          if math.fabs(u.convert_y_to_discrete_output(predicted_review_result) - actual_review_result)<=2:
            prediction_match_count += 1
          error_val += (predicted_review_result - actual_review_result)**2;
        error_val /= X.shape[0];
        accuracy = prediction_match_count/X.shape[0]
        return (error_val, accuracy)
