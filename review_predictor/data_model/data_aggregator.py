from util import utility as u
from pprint import pprint
import numpy as np

FEATURE_SIZE = 8
class DataAggregator:
    def __init__(self, training_set_size, test_set_size, validation_set_size):
        self.training_set_size = training_set_size
        self.test_set_size = test_set_size
        self.validation_set_size = validation_set_size
        self.total_size = training_set_size + test_set_size + validation_set_size

    def generateDataset(self, business_data_dict, user_data_dict, indexed_review_data, top_user_review_count_dict, user_Id_to_business_Id_map):
        X_training = np.array([], dtype=np.int64).reshape(0,FEATURE_SIZE)
        Y_training = np.array([], dtype=np.int64).reshape(0,1)
        X_test = np.array([], dtype=np.int64).reshape(0,FEATURE_SIZE)
        Y_test = np.array([], dtype=np.int64).reshape(0,1)
        for user_Id in top_user_review_count_dict:
            if X_training.shape[0] >= self.training_set_size and X_test.shape[0] >= self.test_set_size:
                break
            else:
                business_Ids = user_Id_to_business_Id_map[user_Id]
                for i, business_Id in enumerate(business_Ids):
                    input_matrix = self.populateInputMatrix(user_Id, user_data_dict, business_Id, user_Id_to_business_Id_map, business_data_dict)
                    output_matrix = self.populateOutputMatrix(user_Id, business_Id, indexed_review_data)
                    if i % 2 == 0:
                        X_training = np.vstack([X_training, input_matrix])
                        Y_training = np.vstack([Y_training, output_matrix])
                    else:
                        X_test = np.vstack([X_test, input_matrix])
                        Y_test = np.vstack([Y_test, output_matrix])
        return (X_training, Y_training, X_test, Y_test)

    def populateInputMatrix(self, user_Id, user_data_dict, business_Id, user_Id_to_business_Id_map, business_data_dict):
        input_matrix = np.zeros((1,FEATURE_SIZE))
        input_matrix[0,0] = self.getUserPreferenceBusinessAttributeCorrelation(user_Id, user_Id_to_business_Id_map, business_Id, business_data_dict, "Attire")
        input_matrix[0,1] = u.get_noise_level_num_value(business_data_dict[business_Id]['attributes']);
        input_matrix[0,2] = self.getUserPreferenceBusinessAttributeCorrelation(user_Id, user_Id_to_business_Id_map, business_Id, business_data_dict, "Price Range")
        input_matrix[0,3] = self.getUserPreferenceBusinessAttributeCorrelation(user_Id, user_Id_to_business_Id_map, business_Id, business_data_dict, "Waiter Service")
        input_matrix[0,4] = self.getUserPreferenceBusinessAttributeCorrelation(user_Id, user_Id_to_business_Id_map, business_Id, business_data_dict, "categories")
        input_matrix[0,5] = business_data_dict[business_Id]['stars']
        input_matrix[0,6] = user_data_dict[user_Id]['average_stars']
        input_matrix[0,7] = 1
        return input_matrix

    def populateOutputMatrix(self, user_Id, business_Id, indexed_review_data):
        output_matrix = np.zeros((1,1))
        output_matrix[0,0] = indexed_review_data[(user_Id,business_Id)]
        return output_matrix

    def getUserPreferenceBusinessAttributeCorrelation(self, user_Id, user_Id_to_business_Id_map, business_Id, business_data_dict, attribute_name):
        business_data_entry = business_data_dict[business_Id]
        if attribute_name not in business_data_entry and attribute_name not in business_data_entry['attributes']:
            # pprint(attribute_name)
            # pprint(business_data_entry)
            return 0
        else:
            correlation = -0.5
            user_preferences = self.calculateUserPreferences(user_Id, user_Id_to_business_Id_map, business_data_dict, attribute_name)
            if attribute_name in business_data_entry:
                attribute_val = business_data_entry[attribute_name]
                for attribute_val_item in attribute_val:
                    if attribute_val_item in user_preferences:
                        correlation += user_preferences[attribute_val_item]
            elif attribute_name in business_data_entry['attributes']:
                attribute_val = business_data_entry['attributes'][attribute_name]
                if attribute_val in user_preferences:
                    correlation += user_preferences[attribute_val]
            return correlation

    def calculateUserPreferences(self, user_Id, user_Id_to_business_Id_map, business_data_dict, attribute_name):
        user_preferences = {}
        business_Ids = user_Id_to_business_Id_map[user_Id]
        total_count = 0
        for business_Id in business_Ids:
            business_data = business_data_dict[business_Id]
            if attribute_name in business_data:
                attribute_val = business_data[attribute_name]
                for attribute_val_item in attribute_val:
                        total_count += 1
                        if attribute_val_item not in user_preferences:
                            user_preferences[attribute_val_item] = 1
                        else:
                            user_preferences[attribute_val_item] += 1
            elif attribute_name in business_data['attributes']:
                attribute_val = business_data['attributes'][attribute_name]
                total_count += 1
                if attribute_val not in user_preferences:
                    user_preferences[attribute_val] = 1
                else:
                    user_preferences[attribute_val] += 1
        for attribute_val in user_preferences:
            user_preferences[attribute_val] = user_preferences[attribute_val]/total_count
        return user_preferences







