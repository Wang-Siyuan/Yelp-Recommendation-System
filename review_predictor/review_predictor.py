import utility as u
from pprint import pprint
import numpy as np
import sys
import itertools
from sklearn import linear_model

np.set_printoptions(threshold=np.nan)

user_data = u.parse_data_set('data-set/yelp_academic_dataset_user.json')
business_data = u.parse_data_set('data-set/yelp_academic_dataset_business.json')
review_data = u.parse_data_set('data-set/yelp_academic_dataset_review.json')

training_data_set_size = 1000;
feature_size = 79;
X = np.zeros((training_data_set_size, feature_size));
Y = np.zeros((training_data_set_size, 1));

user_dict = {};
for user_data_entry in user_data:
	user_dict[user_data_entry['user_id']] = user_data_entry;

business_dict = {}
for business_data_entry in business_data:
	business_dict[business_data_entry['business_id']] = business_data_entry;


def populate_user_data(feature_index_start, user_data_entry):
	X[i,feature_index_start] = user_data_entry['average_stars'];
	X[i,feature_index_start + 1] = u.get_nullable_attribute(user_data_entry['compliments'],'cool');
	X[i,feature_index_start + 2] = u.get_nullable_attribute(user_data_entry['compliments'],'hot');
	X[i,feature_index_start + 3] = u.get_nullable_attribute(user_data_entry['compliments'],'more');
	X[i,feature_index_start + 4] = u.get_nullable_attribute(user_data_entry['compliments'],'writer');
	X[i,feature_index_start + 5] = user_data_entry['fans'];
	X[i,feature_index_start + 6] = user_data_entry['review_count'];
	X[i,feature_index_start + 7] = user_data_entry['votes']['cool'];
	X[i,feature_index_start + 8] = user_data_entry['votes']['useful'];


def populate_business_data(feature_index_start, business_data_entry):
	X[i,feature_index_start] = u.get_nullable_attribute_with_str_2_int(business_data_entry['attributes'],'Accepts Credit Cards');
	if 'Alcohol' in business_data_entry['attributes'] and (business_data_entry['attributes']['Alcohol'] == 'full-bar' or business_data_entry['attributes']['Alcohol'] == 'beer_and_wine'):
		X[i,feature_index_start + 1] = 1;
	X[i,feature_index_start + 2] = u.get_nullable_attribute_with_expected_value(business_data_entry['attributes'],'Ambience','casual');
	X[i,feature_index_start + 3] = u.get_nullable_attribute_with_expected_value(business_data_entry['attributes'],'Ambience','classy');
	X[i,feature_index_start + 4] = u.get_nullable_attribute_with_expected_value(business_data_entry['attributes'],'Ambience','divey');
	X[i,feature_index_start + 5] = u.get_nullable_attribute_with_expected_value(business_data_entry['attributes'],'Ambience','hipster');
	X[i,feature_index_start + 6] = u.get_nullable_attribute_with_expected_value(business_data_entry['attributes'],'Ambience','intimate');
	X[i,feature_index_start + 7] = u.get_nullable_attribute_with_expected_value(business_data_entry['attributes'],'Ambience','romantic');
	X[i,feature_index_start + 8] = u.get_nullable_attribute_with_expected_value(business_data_entry['attributes'],'Ambience','touristy');
	X[i,feature_index_start + 9] = u.get_nullable_attribute_with_expected_value(business_data_entry['attributes'],'Ambience','trendy');
	X[i,feature_index_start + 10] = u.get_nullable_attribute_with_expected_value(business_data_entry['attributes'],'Ambience','upscale');
	X[i,feature_index_start + 11] = u.get_nullable_attribute_with_expected_value(business_data_entry['attributes'],'Attire','casual');
	X[i,feature_index_start + 12] = u.get_nullable_attribute_with_expected_value(business_data_entry['attributes'],'Attire','dressy');
	X[i,feature_index_start + 13] = u.get_nullable_attribute_with_expected_value(business_data_entry['attributes'],'Attire','formal');
	X[i,feature_index_start + 14] = u.get_nullable_attribute_with_str_2_int(business_data_entry['attributes'],'Caters');
	X[i,feature_index_start + 15] = u.get_nullable_attribute_with_str_2_int(business_data_entry['attributes'],'Delivery');
	X[i,feature_index_start + 16] = u.get_nullable_attribute_with_str_2_int(business_data_entry['attributes'],'Drive-Thru');
	X[i,feature_index_start + 17] = u.get_nullable_attribute_with_expected_value(business_data_entry['attributes'],'Good For','breakfast');
	X[i,feature_index_start + 18] = u.get_nullable_attribute_with_expected_value(business_data_entry['attributes'],'Good For','brunch');
	X[i,feature_index_start + 19] = u.get_nullable_attribute_with_expected_value(business_data_entry['attributes'],'Good For','dessert');
	X[i,feature_index_start + 20] = u.get_nullable_attribute_with_expected_value(business_data_entry['attributes'],'Good For','dinner');
	X[i,feature_index_start + 21] = u.get_nullable_attribute_with_expected_value(business_data_entry['attributes'],'Good For','latenight');
	X[i,feature_index_start + 22] = u.get_nullable_attribute_with_expected_value(business_data_entry['attributes'],'Good For','lunch');
	X[i,feature_index_start + 23] = u.get_nullable_attribute_with_str_2_int(business_data_entry['attributes'],'Good For Groups');
	X[i,feature_index_start + 24] = u.get_nullable_attribute_with_str_2_int(business_data_entry['attributes'],'Good for Kids');
	X[i,feature_index_start + 25] = u.get_nullable_attribute_with_str_2_int(business_data_entry['attributes'],'Has TV');
	X[i,feature_index_start + 26] = u.get_nullable_attribute_with_str_2_int(business_data_entry['attributes'],'Noise Level');
	X[i,feature_index_start + 27] = u.get_nullable_attribute_with_str_2_int(business_data_entry['attributes'],'Outdoor Seating');
	if 'Parking' in business_data_entry['attributes'] and (business_data_entry['attributes']['Parking']['garage'] == 'True' or business_data_entry['attributes']['Parking']['lot'] == 'True' or business_data_entry['attributes']['Parking']['street'] == 'True' or business_data_entry['attributes']['Parking']['valet'] == 'True' or business_data_entry['attributes']['Parking']['validated'] == 'True'):
		X[i,feature_index_start + 28] = 1;
	X[i,feature_index_start + 29] = u.get_nullable_attribute_with_str_2_int(business_data_entry['attributes'],'Price Range');
	X[i,feature_index_start + 30] = u.get_nullable_attribute_with_str_2_int(business_data_entry['attributes'],'Take-out');
	X[i,feature_index_start + 31] = u.get_nullable_attribute_with_str_2_int(business_data_entry['attributes'],'Takes Reservations');
	X[i,feature_index_start + 32] = u.get_nullable_attribute_with_str_2_int(business_data_entry['attributes'],'Waiter Service');
	# if u.str_2_int(business_data_entry['hours'][]
	categories = business_data_entry['categories'];
	if 'American (New)' in categories:
		X[i,feature_index_start + 34] = 1;
	if 'American (Traditional)' in categories:
		X[i,feature_index_start + 35] = 1;
	if 'Asian Fusion' in categories:
		X[i,feature_index_start + 36] = 1;
	if 'Barbeque' in categories:
		X[i,feature_index_start + 37] = 1;
	if 'Brazilian' in categories:
		X[i,feature_index_start + 38] = 1;
	if 'Breakfast & Brunch' in categories:
		X[i,feature_index_start + 39] = 1;
	if 'Buffets' in categories:
		X[i,feature_index_start + 40] = 1;
	if 'Cajun/Creole' in categories:
		X[i,feature_index_start + 41] = 1;
	if 'Caribbean' in categories:
		X[i,feature_index_start + 42] = 1;
	if 'Chinese' in categories:
		X[i,feature_index_start + 43] = 1;
	if 'Cuban' in categories:
		X[i,feature_index_start + 44] = 1;
	if 'French' in categories:
		X[i,feature_index_start + 45] = 1;
	if 'Greek' in categories:
		X[i,feature_index_start + 46] = 1;
	if 'Indian' in categories:
		X[i,feature_index_start + 47] = 1;
	if 'Italian' in categories:
		X[i,feature_index_start + 48] = 1;
	if 'Japanese' in categories:
		X[i,feature_index_start + 49] = 1;
	if 'Korean' in categories:
		X[i,feature_index_start + 50] = 1;
	if 'Malaysian' in categories:
		X[i,feature_index_start + 51] = 1;
	if 'Mediterranean' in categories:
		X[i,feature_index_start + 52] = 1;
	if 'Mexican' in categories:
		X[i,feature_index_start + 53] = 1;
	if 'Noodles' in categories:
		X[i,feature_index_start + 54] = 1;
	if 'Pizza' in categories:
		X[i,feature_index_start + 55] = 1;
	if 'Russian' in categories:
		X[i,feature_index_start + 56] = 1;
	if 'Brazilian' in categories:
		X[i,feature_index_start + 57] = 1;
	if 'Salad' in categories:
		X[i,feature_index_start + 58] = 1;
	if 'Seafood' in categories:
		X[i,feature_index_start + 59] = 1;
	if 'Soul Food' in categories:
		X[i,feature_index_start + 60] = 1;
	if 'Southern' in categories:
		X[i,feature_index_start + 61] = 1;
	if 'Spanish' in categories:
		X[i,feature_index_start + 62] = 1;
	if 'Taiwanese' in categories:
		X[i,feature_index_start + 63] = 1;
	if 'Vegan' in categories:
		X[i,feature_index_start + 64] = 1;
	if 'Vegetarian' in categories:
		X[i,feature_index_start + 65] = 1;
	if 'Vietnamese' in categories:
		X[i,feature_index_start + 66] = 1;
	X[i,feature_index_start + 67] = business_data_entry['review_count'];
	X[i,feature_index_start + 68] = business_data_entry['stars'];
	X[i,feature_index_start + 69] = business_data_entry['review_count'];


for i,review_data_entry in enumerate(itertools.islice(review_data,training_data_set_size)):
	user_data_entry = user_dict[review_data_entry['user_id']];
	populate_user_data(0,user_data_entry);
	business_data_entry = business_dict[review_data_entry['business_id']];
	populate_business_data(9,business_data_entry);
	pprint(review_data_entry)
	Y[i] = review_data_entry['stars'];

pprint(X)
pprint(Y)

reg = linear_model.Lasso(alpha = 0.2);
reg.fit (X, Y); 
print(reg.coef_);
print(reg.intercept_);

# pprint(next(user_data))
# pprint(next(business_data))
# pprint(next(review_data))
