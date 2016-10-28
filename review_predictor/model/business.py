from util import utility as u
from pprint import pprint
import numpy as np

ALCOHOL_TYPE_ENUMERATION = ['full_bar', 'beer_and_wine'];

class Business:

	numOfFeatures = 37;

	def __init__(self, data_set_file_path):
		self.business_data = u.parse_data_set(data_set_file_path);
		self.business_data_dict = self.indexBusinessData();

	def indexBusinessData(self):
		business_data_dict = {}
		for business_data_entry in self.business_data:
			business_data_dict[business_data_entry['business_id']] = business_data_entry;
		return business_data_dict;

	def populate_business_data(self, user, user_id, business_id):
		X = np.zeros((1,Business.numOfFeatures));
		business_data_entry = self.business_data_dict[business_id];
		business_attribute = business_data_entry['attributes'];
		X[0,0] = u.get_nullable_attribute_with_str_2_int(business_attribute,'Accepts Credit Cards');

		X[0,1] = u.get_nullable_attribute_with_contained_by_enumeration(business_attribute, 'Alcohol', ALCOHOL_TYPE_ENUMERATION);

		X[0,2] = u.get_nullable_attribute_and_check_for_boolean_sub_attribute(business_attribute,'Ambience','casual');
		X[0,3] = u.get_nullable_attribute_and_check_for_boolean_sub_attribute(business_attribute,'Ambience','classy');
		X[0,4] = u.get_nullable_attribute_and_check_for_boolean_sub_attribute(business_attribute,'Ambience','divey');
		X[0,5] = u.get_nullable_attribute_and_check_for_boolean_sub_attribute(business_attribute,'Ambience','hipster');
		X[0,6] = u.get_nullable_attribute_and_check_for_boolean_sub_attribute(business_attribute,'Ambience','intimate');
		X[0,7] = u.get_nullable_attribute_and_check_for_boolean_sub_attribute(business_attribute,'Ambience','romantic');
		X[0,8] = u.get_nullable_attribute_and_check_for_boolean_sub_attribute(business_attribute,'Ambience','touristy');
		X[0,9] = u.get_nullable_attribute_and_check_for_boolean_sub_attribute(business_attribute,'Ambience','trendy');
		X[0,10] = u.get_nullable_attribute_and_check_for_boolean_sub_attribute(business_attribute,'Ambience','upscale');

		X[0,11] = u.get_nullable_attribute_with_expected_value(business_attribute,'Attire','casual');
		X[0,12] = u.get_nullable_attribute_with_expected_value(business_attribute,'Attire','dressy');
		X[0,13] = u.get_nullable_attribute_with_expected_value(business_attribute,'Attire','formal');

		X[0,14] = u.get_nullable_attribute_with_str_2_int(business_attribute,'Caters');
		X[0,15] = u.get_nullable_attribute_with_str_2_int(business_attribute,'Delivery');
		X[0,16] = u.get_nullable_attribute_with_str_2_int(business_attribute,'Drive-Thru');

		X[0,17] = u.get_nullable_attribute_and_check_for_boolean_sub_attribute(business_attribute,'Good For','breakfast');
		X[0,18] = u.get_nullable_attribute_and_check_for_boolean_sub_attribute(business_attribute,'Good For','brunch');
		X[0,19] = u.get_nullable_attribute_and_check_for_boolean_sub_attribute(business_attribute,'Good For','dessert');
		X[0,20] = u.get_nullable_attribute_and_check_for_boolean_sub_attribute(business_attribute,'Good For','dinner');
		X[0,21] = u.get_nullable_attribute_and_check_for_boolean_sub_attribute(business_attribute,'Good For','latenight');
		X[0,22] = u.get_nullable_attribute_and_check_for_boolean_sub_attribute(business_attribute,'Good For','lunch');

		X[0,23] = u.get_nullable_attribute_with_str_2_int(business_attribute,'Good For Groups');
		X[0,24] = u.get_nullable_attribute_with_str_2_int(business_attribute,'Good for Kids');
		X[0,25] = u.get_nullable_attribute_with_str_2_int(business_attribute,'Has TV');
		X[0,26] = u.get_noise_level_num_value(business_attribute);
		X[0,27] = u.get_nullable_attribute_with_str_2_int(business_attribute,'Outdoor Seating');
		X[0,28] = u.get_nullable_attribute_with_boolean_dict(business_attribute, 'Parking');
		X[0,29] = u.get_nullable_attribute_with_str_2_int(business_attribute,'Price Range');
		X[0,30] = u.get_nullable_attribute_with_str_2_int(business_attribute,'Take-out');
		X[0,31] = u.get_nullable_attribute_with_str_2_int(business_attribute,'Takes Reservations');
		X[0,32] = u.get_nullable_attribute_with_str_2_int(business_attribute,'Waiter Service');
		X[0,33] = user.get_correlation_between_user_category_preferences_and_business_categories(user_id, business_data_entry['categories']);
		X[0,34] = business_data_entry['review_count'];
		X[0,35] = business_data_entry['stars'];
		X[0,36] = business_data_entry['review_count'];
		return X;

	def getBusinessDataDict(self):
		return self.business_data_dict;
