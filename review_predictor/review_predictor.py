import utility as u
from pprint import pprint
import numpy as np
import sys
import itertools
from sklearn import linear_model

user_data = u.parse_data_set('data-set/yelp_academic_dataset_user.json')
business_data = u.parse_data_set('data-set/yelp_academic_dataset_business.json')
review_data = u.parse_data_set('data-set/yelp_academic_dataset_review.json')

training_data_set_size = 20;
feature_size = 11;
X = np.zeros((training_data_set_size, feature_size));
Y = np.zeros((training_data_set_size, 1));

# user_data_sample = (user_data for user_data in range(1, 20))
for i,user_data_entry in enumerate(itertools.islice(user_data,20)):
	sys.stdout.write(str(i))
	pprint(user_data_entry)
	X[i,0] = user_data_entry['average_stars'];
	X[i,1] = u.get_nullable_attribute(user_data_entry['compliments'],'cool');
	X[i,2] = u.get_nullable_attribute(user_data_entry['compliments'],'hot');
	X[i,3] = u.get_nullable_attribute(user_data_entry['compliments'],'more');
	X[i,4] = u.get_nullable_attribute(user_data_entry['compliments'],'writer');
	X[i,5] = user_data_entry['fans'];
	X[i,6] = user_data_entry['review_count'];
	X[i,7] = user_data_entry['votes']['cool'];
	X[i,8] = user_data_entry['votes']['useful'];


for i,business_data_entry in enumerate(itertools.islice(business_data,20)):
	sys.stdout.write(str(i))
	pprint(business_data_entry)

	X[i,9] = u.get_nullable_attribute_with_str_2_int(business_data_entry['attributes'],'Accepts Credit Cards');
	X[i,10] = u.get_nullable_attribute_with_expected_value(business_data_entry['attributes'], 'Alcohol', 'full-bar');
	# X[i,9] = str_2_int(business_data_entry['attributes']['Accepts Credit Cards'])
	# X[i,9] = str_2_int(business_data_entry['attributes']['Accepts Credit Cards'])
	# X[i,9] = str_2_int(business_data_entry['attributes']['Accepts Credit Cards'])
	# X[i,9] = str_2_int(business_data_entry['attributes']['Accepts Credit Cards'])
	# X[i,9] = str_2_int(business_data_entry['attributes']['Accepts Credit Cards'])
	# X[i,9] = str_2_int(business_data_entry['attributes']['Accepts Credit Cards'])
	# X[i,9] = str_2_int(business_data_entry['attributes']['Accepts Credit Cards'])
	# X[i,9] = str_2_int(business_data_entry['attributes']['Accepts Credit Cards'])

	# X[i,11] = business_data_entry['review_count'];

for i,review_data_entry in enumerate(itertools.islice(review_data,20)):
	sys.stdout.write(str(i))
	pprint(review_data_entry)
	Y[i] = review_data_entry['stars'];

pprint(X)
pprint(Y)

reg = linear_model.Ridge (alpha = .5);
reg.fit (X, Y); 
print(reg.coef_);
print(reg.intercept_);

# pprint(next(user_data))
# pprint(next(business_data))
# pprint(next(review_data))

