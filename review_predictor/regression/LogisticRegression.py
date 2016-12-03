from sklearn import linear_model

class LogisticRegression:
	def __init__(self):
		self.model = linear_model.LogisticRegressionCV(
        Cs=9
        ,penalty='l2'
        ,scoring='roc_auc'
        ,cv=3
        ,n_jobs=-1
        ,max_iter=10000
        ,fit_intercept=True
        ,tol=10)

	def fit(self, X, Y):
		self.model.fit (X_training, Y_training.ravel())

	def accuracy_and_error(self, X, Y):
		in_sample_error = 0;
		for i in range(0,X.shape[0]-1):
		  predicted_review_result = model.predict(X[i,:].reshape(1, -1));
		  actual_review_result = Y[i,0];
		  in_sample_error += (predicted_review_result - actual_review_result)**2;
		in_sample_error /= X.shape[0];

