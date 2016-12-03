from util import utility as u
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt

ALCOHOL_TYPE_ENUMERATION = ['full_bar', 'beer_and_wine'];

class Business:

    # numOfFeatures = 36;
    numOfFeatures = 8;

    def __init__(self, data_set_file_path):
        self.business_data = u.parse_data_set(data_set_file_path);
        self.business_data_dict = self.indexBusinessData();

    def indexBusinessData(self):
        business_data_dict = {}
        for business_data_entry in self.business_data:
            business_data_dict[business_data_entry['business_Id']] = business_data_entry;
        return business_data_dict;

    def populate_business_data(self, user, user_Id, business_Id):
        X = np.zeros((1,Business.numOfFeatures));
        business_data_entry = self.business_data_dict[business_Id];
        business_attribute = business_data_entry['attributes'];

        X[0,0] = u.get_nullable_attribute_with_expected_value(business_attribute,'Attire','casual');
        X[0,1] = u.get_nullable_attribute_with_expected_value(business_attribute,'Attire','dressy');
        X[0,2] = u.get_nullable_attribute_with_expected_value(business_attribute,'Attire','formal');
        X[0,3] = u.get_noise_level_num_value(business_attribute);
        X[0,4] = business_data_entry['Price Range'];
        X[0,5] = u.get_nullable_attribute_with_str_2_int(business_attribute,'Waiter Service');
        X[0,6] = user.get_correlation_between_user_category_preferences_and_business_categories(user_Id, business_data_entry['categories']);
        X[0,7] = business_data_entry['stars'];

        # X[0,0] = u.get_nullable_attribute_with_str_2_int(business_attribute,'Accepts Credit Cards');

        # X[0,1] = u.get_nullable_attribute_with_contained_by_enumeration(business_attribute, 'Alcohol', ALCOHOL_TYPE_ENUMERATION);

        # X[0,2] = u.get_nullable_attribute_and_check_for_boolean_sub_attribute(business_attribute,'Ambience','casual');
        # X[0,3] = u.get_nullable_attribute_and_check_for_boolean_sub_attribute(business_attribute,'Ambience','classy');
        # X[0,4] = u.get_nullable_attribute_and_check_for_boolean_sub_attribute(business_attribute,'Ambience','divey');
        # X[0,5] = u.get_nullable_attribute_and_check_for_boolean_sub_attribute(business_attribute,'Ambience','hipster');
        # X[0,6] = u.get_nullable_attribute_and_check_for_boolean_sub_attribute(business_attribute,'Ambience','intimate');
        # X[0,7] = u.get_nullable_attribute_and_check_for_boolean_sub_attribute(business_attribute,'Ambience','romantic');
        # X[0,8] = u.get_nullable_attribute_and_check_for_boolean_sub_attribute(business_attribute,'Ambience','touristy');
        # X[0,9] = u.get_nullable_attribute_and_check_for_boolean_sub_attribute(business_attribute,'Ambience','trendy');
        # X[0,10] = u.get_nullable_attribute_and_check_for_boolean_sub_attribute(business_attribute,'Ambience','upscale');

        # X[0,11] = u.get_nullable_attribute_with_expected_value(business_attribute,'Attire','casual');
        # X[0,12] = u.get_nullable_attribute_with_expected_value(business_attribute,'Attire','dressy');
        # X[0,13] = u.get_nullable_attribute_with_expected_value(business_attribute,'Attire','formal');

        # X[0,14] = u.get_nullable_attribute_with_str_2_int(business_attribute,'Caters');
        # X[0,15] = u.get_nullable_attribute_with_str_2_int(business_attribute,'Delivery');
        # X[0,16] = u.get_nullable_attribute_with_str_2_int(business_attribute,'Drive-Thru');

        # X[0,17] = u.get_nullable_attribute_and_check_for_boolean_sub_attribute(business_attribute,'Good For','breakfast');
        # X[0,18] = u.get_nullable_attribute_and_check_for_boolean_sub_attribute(business_attribute,'Good For','brunch');
        # X[0,19] = u.get_nullable_attribute_and_check_for_boolean_sub_attribute(business_attribute,'Good For','dessert');
        # X[0,20] = u.get_nullable_attribute_and_check_for_boolean_sub_attribute(business_attribute,'Good For','dinner');
        # X[0,21] = u.get_nullable_attribute_and_check_for_boolean_sub_attribute(business_attribute,'Good For','latenight');
        # X[0,22] = u.get_nullable_attribute_and_check_for_boolean_sub_attribute(business_attribute,'Good For','lunch');

        # X[0,23] = u.get_nullable_attribute_with_str_2_int(business_attribute,'Good For Groups');
        # X[0,24] = u.get_nullable_attribute_with_str_2_int(business_attribute,'Good for Kids');
        # X[0,25] = u.get_nullable_attribute_with_str_2_int(business_attribute,'Has TV');
        # X[0,26] = u.get_noise_level_num_value(business_attribute);
        # X[0,27] = u.get_nullable_attribute_with_str_2_int(business_attribute,'Outdoor Seating');
        # X[0,28] = u.get_nullable_attribute_with_boolean_dict(business_attribute, 'Parking');
        # X[0,29] = business_attribute['Price Range'];
        # X[0,30] = u.get_nullable_attribute_with_str_2_int(business_attribute,'Take-out');
        # X[0,31] = u.get_nullable_attribute_with_str_2_int(business_attribute,'Takes Reservations');
        # X[0,32] = u.get_nullable_attribute_with_str_2_int(business_attribute,'Waiter Service');
        # X[0,33] = user.get_correlation_between_user_category_preferences_and_business_categories(user_Id, business_data_entry['categories']);
        # X[0,34] = business_data_entry['review_count'];
        # X[0,35] = business_data_entry['stars'];
        return X;

    def getBusinessDataDict(self):
        return self.business_data_dict;

    def generateHistogram(self):
        categories_count = {};
        total_categories_count = 0;
        for business_Id in self.business_data_dict:
            business_data_entry = self.business_data_dict[business_Id];
            # pprint(business_data_entry);
            business_categories = business_data_entry['categories'];
            for business_category in business_categories:
                if business_category != 'Restaurants':
                    if business_category not in categories_count:
                        categories_count[business_category] = 0;
                    categories_count[business_category] += 1;
                    total_categories_count += 1;
        fracs = [];
        labels = [];
        for category in categories_count:
            labels.append(category);
            fracs.append(categories_count[category]/total_categories_count);
        # pprint(fracs);
        # pprint(labels);

        patches, texts = plt.pie(fracs, startangle=90)
        plt.legend(patches, labels, loc="best")
        # Set aspect ratio to be equal so that pie is drawn as a circle.
        plt.axis('equal')
        plt.tight_layout()
        plt.show()