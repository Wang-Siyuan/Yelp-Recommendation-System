from sklearn import linear_model
from util import utility as u
from pprint import pprint

class LogisticRegression:
    def __init__(self):
        self.model = linear_model.LogisticRegressionCV(
        Cs=10
        ,penalty='l2'
        ,scoring='roc_auc'
        ,cv=3
        ,n_jobs=-1
        ,max_iter=1000
        ,fit_intercept=True
        ,tol=10)

    def fit(self, X, Y):
        self.model.fit(X, Y.ravel())
        pprint(self.model.C_)

    def accuracy_and_error(self, X, Y):
        error_val = 0
        prediction_match_count = 0
        for i in range(0,X.shape[0]-1):
          predicted_review_result = self.model.predict(X[i,:].reshape(1, -1));
          actual_review_result = Y[i,0];
          if u.convert_y_to_discrete_output(predicted_review_result) <= actual_review_result:
            prediction_match_count += 1
          error_val += (predicted_review_result - actual_review_result)**2;
        error_val /= X.shape[0];
        accuracy = prediction_match_count/X.shape[0]
        return (error_val, accuracy)
