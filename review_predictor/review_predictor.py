import utility as u
from pprint import pprint
import numpy as np
import sys
import itertools
from sklearn import linear_model

np.set_printoptions(threshold=np.nan)

user_data = u.parse_data_set('data-set/yelp_academic_dataset_user.json')
business_data = u.parse_data_set('data-set/yelp_academic_dataset_business_restaurants_only.json')
review_data = u.parse_data_set('data-set/yelp_academic_dataset_review.json')
alcohol_type_enumeration = ['full_bar', 'beer_and_wine'];

training_data_set_size = 20000;
feature_size = 46;

X = np.zeros((training_data_set_size, feature_size));
Y = np.zeros((training_data_set_size, 1));

user_dict = {};
for user_data_entry in user_data:
	user_dict[user_data_entry['user_id']] = user_data_entry;

business_dict = {}
for business_data_entry in business_data:
	business_dict[business_data_entry['business_id']] = business_data_entry;


def populate_user_data(row_num, feature_index_start, user_data_entry):
	X[row_num,feature_index_start] = user_data_entry['average_stars'];
	X[row_num,feature_index_start + 1] = u.get_nullable_attribute(user_data_entry['compliments'],'cool');
	X[row_num,feature_index_start + 2] = u.get_nullable_attribute(user_data_entry['compliments'],'hot');
	X[row_num,feature_index_start + 3] = u.get_nullable_attribute(user_data_entry['compliments'],'more');
	X[row_num,feature_index_start + 4] = u.get_nullable_attribute(user_data_entry['compliments'],'writer');
	X[row_num,feature_index_start + 5] = user_data_entry['fans'];
	X[row_num,feature_index_start + 6] = user_data_entry['review_count'];
	X[row_num,feature_index_start + 7] = user_data_entry['votes']['cool'];
	X[row_num,feature_index_start + 8] = user_data_entry['votes']['useful'];


def populate_business_data(row_num, feature_index_start, user_id, business_data_entry):
	business_attribute = business_data_entry['attributes'];
	X[row_num,feature_index_start] = u.get_nullable_attribute_with_str_2_int(business_attribute,'Accepts Credit Cards');

	X[row_num,feature_index_start + 1] = u.get_nullable_attribute_with_contained_by_enumeration(business_attribute, 'Alcohol', alcohol_type_enumeration);

	X[row_num,feature_index_start + 2] = u.get_nullable_attribute_and_check_for_boolean_sub_attribute(business_attribute,'Ambience','casual');
	X[row_num,feature_index_start + 3] = u.get_nullable_attribute_and_check_for_boolean_sub_attribute(business_attribute,'Ambience','classy');
	X[row_num,feature_index_start + 4] = u.get_nullable_attribute_and_check_for_boolean_sub_attribute(business_attribute,'Ambience','divey');
	X[row_num,feature_index_start + 5] = u.get_nullable_attribute_and_check_for_boolean_sub_attribute(business_attribute,'Ambience','hipster');
	X[row_num,feature_index_start + 6] = u.get_nullable_attribute_and_check_for_boolean_sub_attribute(business_attribute,'Ambience','intimate');
	X[row_num,feature_index_start + 7] = u.get_nullable_attribute_and_check_for_boolean_sub_attribute(business_attribute,'Ambience','romantic');
	X[row_num,feature_index_start + 8] = u.get_nullable_attribute_and_check_for_boolean_sub_attribute(business_attribute,'Ambience','touristy');
	X[row_num,feature_index_start + 9] = u.get_nullable_attribute_and_check_for_boolean_sub_attribute(business_attribute,'Ambience','trendy');
	X[row_num,feature_index_start + 10] = u.get_nullable_attribute_and_check_for_boolean_sub_attribute(business_attribute,'Ambience','upscale');

	X[row_num,feature_index_start + 11] = u.get_nullable_attribute_with_expected_value(business_attribute,'Attire','casual');
	X[row_num,feature_index_start + 12] = u.get_nullable_attribute_with_expected_value(business_attribute,'Attire','dressy');
	X[row_num,feature_index_start + 13] = u.get_nullable_attribute_with_expected_value(business_attribute,'Attire','formal');

	X[row_num,feature_index_start + 14] = u.get_nullable_attribute_with_str_2_int(business_attribute,'Caters');
	X[row_num,feature_index_start + 15] = u.get_nullable_attribute_with_str_2_int(business_attribute,'Delivery');
	X[row_num,feature_index_start + 16] = u.get_nullable_attribute_with_str_2_int(business_attribute,'Drive-Thru');

	X[row_num,feature_index_start + 17] = u.get_nullable_attribute_and_check_for_boolean_sub_attribute(business_attribute,'Good For','breakfast');
	X[row_num,feature_index_start + 18] = u.get_nullable_attribute_and_check_for_boolean_sub_attribute(business_attribute,'Good For','brunch');
	X[row_num,feature_index_start + 19] = u.get_nullable_attribute_and_check_for_boolean_sub_attribute(business_attribute,'Good For','dessert');
	X[row_num,feature_index_start + 20] = u.get_nullable_attribute_and_check_for_boolean_sub_attribute(business_attribute,'Good For','dinner');
	X[row_num,feature_index_start + 21] = u.get_nullable_attribute_and_check_for_boolean_sub_attribute(business_attribute,'Good For','latenight');
	X[row_num,feature_index_start + 22] = u.get_nullable_attribute_and_check_for_boolean_sub_attribute(business_attribute,'Good For','lunch');

	X[row_num,feature_index_start + 23] = u.get_nullable_attribute_with_str_2_int(business_attribute,'Good For Groups');
	X[row_num,feature_index_start + 24] = u.get_nullable_attribute_with_str_2_int(business_attribute,'Good for Kids');
	X[row_num,feature_index_start + 25] = u.get_nullable_attribute_with_str_2_int(business_attribute,'Has TV');
	X[row_num,feature_index_start + 26] = u.get_noise_level_num_value(business_attribute);
	X[row_num,feature_index_start + 27] = u.get_nullable_attribute_with_str_2_int(business_attribute,'Outdoor Seating');
	X[row_num,feature_index_start + 28] = u.get_nullable_attribute_with_boolean_dict(business_attribute, 'Parking');
	X[row_num,feature_index_start + 29] = u.get_nullable_attribute_with_str_2_int(business_attribute,'Price Range');
	X[row_num,feature_index_start + 30] = u.get_nullable_attribute_with_str_2_int(business_attribute,'Take-out');
	X[row_num,feature_index_start + 31] = u.get_nullable_attribute_with_str_2_int(business_attribute,'Takes Reservations');
	X[row_num,feature_index_start + 32] = u.get_nullable_attribute_with_str_2_int(business_attribute,'Waiter Service');
	X[row_num,feature_index_start + 33] = get_user_preference_business_category_correlation(user_id, business_data_entry['categories']);
	X[row_num,feature_index_start + 34] = business_data_entry['review_count'];
	X[row_num,feature_index_start + 35] = business_data_entry['stars'];
	X[row_num,feature_index_start + 36] = business_data_entry['review_count'];


def get_user_preference_business_category_correlation(user_id, categories):
	category_preferences_percentage_dict = user_category_preferences_percentage_dict[user_id];
	correlation = 0;
	for category in categories:
		if category in category_preferences_percentage_dict:
			correlation += category_preferences_percentage_dict[category];
	return correlation;

user_category_preferences = {};
restaurants_reviews = [];
for review_data_entry in itertools.islice(review_data,training_data_set_size*5):
	if review_data_entry['user_id'] in user_dict and review_data_entry['business_id'] in business_dict:
		restaurants_reviews.append(review_data_entry);

		business_data_entry = business_dict[review_data_entry['business_id']];
		restaurant_categories = business_data_entry['categories'];
		user_id = review_data_entry['user_id'];
		if user_id not in user_category_preferences:
			user_category_preferences[user_id] = {};
		category_preferences = user_category_preferences[user_id];
		for restaurants_category in restaurant_categories:
			if restaurants_category not in category_preferences:
				category_preferences[restaurants_category] = 0;
			category_preferences[restaurants_category] += 1;
# pprint(user_category_preferences);


user_category_preferences_percentage_dict = {};
for user_id in user_category_preferences:
	user_category_preferences_percetange_percentages = {};
	user_category_preferences_percentage_dict[user_id] = user_category_preferences_percetange_percentages;
	total_count = 0;
	category_count = user_category_preferences[user_id];
	for category in category_count:
		total_count += category_count[category];
	for category in category_count:
		user_category_preferences_percetange_percentages[category] = category_count[category]/total_count;

# pprint(user_category_preferences_percentage_dict);



for i,review_data_entry in enumerate(itertools.islice(restaurants_reviews,training_data_set_size)):
	user_id = review_data_entry['user_id'];
	user_data_entry = user_dict[user_id];
	populate_user_data(i, 0, user_data_entry);
	business_data_entry = business_dict[review_data_entry['business_id']];
	populate_business_data(i, 9, user_id, business_data_entry);
	# pprint(review_data_entry)
	Y[i] = review_data_entry['stars'];

# pprint(X)
pprint(X.sum(axis=0))
# pprint(Y)

X_normed = (X - X.mean(axis=0)) / X.std(axis=0)

pprint(X_normed);




reg = linear_model.Lasso(alpha = 0.01, max_iter=10000);
reg.fit (X_normed, Y); 
print(reg.coef_);
print(reg.intercept_);

# pprint(next(user_data))
# pprint(next(business_data))
# pprint(next(review_data))

