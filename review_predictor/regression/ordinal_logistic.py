from sklearn import linear_model
from util import utility as u
# from minirank import logistic

class OrdinalLogistic:

    def fit(self, X, Y):
        self.w_, self.theta_ = logistic.ordinal_logistic_fit(X, y)

    def accuracy_and_error(self, X, Y):
        pred = logistic.ordinal_logistic_predict(w_, theta_, X)
        error_val = 0
        prediction_match_count = 0
        pprint(pred.shape)
        for i in range(0,X.shape[0]-1):
          predicted_review_result = pred[i,0]
          actual_review_result = Y[i,0]
          if u.convert_y_to_discrete_output(predicted_review_result) <= actual_review_result:
            prediction_match_count += 1
          in_sample_error += (predicted_review_result - actual_review_result)**2;
        in_sample_error /= X.shape[0];
        accuracy = prediction_match_count/X.shape[0]
        return (in_sample_error, accuracy)

