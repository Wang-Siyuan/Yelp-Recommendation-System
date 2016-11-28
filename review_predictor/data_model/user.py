from util import utility as u
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

class User:

	numOfFeatures = 9;

	def __init__(self, data_set_file_path, user_id_to_business_id_map, business_data_dict):
		self.user_data = u.parse_data_set(data_set_file_path);
		self.user_data_dict = self.indexUserData();
		self.user_category_preferences = self.getUserCategoryPreferences(business_data_dict, user_id_to_business_id_map);

	def indexUserData(self):
		user_data_dict = {};
		for user_data_entry in self.user_data:
			user_data_dict[user_data_entry['user_id']] = user_data_entry;
		return user_data_dict;

	def getUserCategoryPreferences(self, business_data_dict, user_id_to_business_id_map):
		user_restaurant_category_count = {};
		for user_id in user_id_to_business_id_map:
			business_data_entry = business_data_dict[user_id_to_business_id_map[user_id]];
			restaurant_categories = business_data_entry['categories'];
			if user_id not in user_restaurant_category_count:
				user_restaurant_category_count[user_id] = {};
			category_preferences = user_restaurant_category_count[user_id];
			for restaurants_category in restaurant_categories:
				if restaurants_category not in category_preferences:
					category_preferences[restaurants_category] = 0;
				category_preferences[restaurants_category] += 1;

		user_category_preferences = {};
		for user_id in user_restaurant_category_count:
			user_category_preferences_percetange = {};
			user_category_preferences[user_id] = user_category_preferences_percetange;
			total_count = 0;
			category_count = user_restaurant_category_count[user_id];
			for category in category_count:
				total_count += category_count[category];
			for category in category_count:
				user_category_preferences_percetange[category] = category_count[category]/total_count;
		return user_category_preferences;

	def get_correlation_between_user_category_preferences_and_business_categories(self, user_id, categories):
		category_preferences_percentages = self.user_category_preferences[user_id];
		correlation = 0;
		for category in categories:
			if category in category_preferences_percentages:
				correlation += category_preferences_percentages[category];
		return correlation;

	def populate_user_data(self, user_id):
		user_data_entry = self.user_data_dict[user_id];
		X = np.zeros((1,User.numOfFeatures));
		X[0,0] = user_data_entry['average_stars'];
		X[0,1] = u.get_nullable_attribute(user_data_entry['compliments'],'cool');
		X[0,2] = u.get_nullable_attribute(user_data_entry['compliments'],'hot');
		X[0,3] = u.get_nullable_attribute(user_data_entry['compliments'],'more');
		X[0,4] = u.get_nullable_attribute(user_data_entry['compliments'],'writer');
		X[0,5] = user_data_entry['fans'];
		X[0,6] = user_data_entry['review_count'];
		X[0,7] = user_data_entry['votes']['cool'];
		X[0,8] = user_data_entry['votes']['useful'];
		return X;

	def generateStarsHistogram(self):
		review_counts = np.zeros((u.count_iterable(self.user_data_dict),1));
		for i,user_id in enumerate(self.user_data_dict):
			user_data_entry = self.user_data_dict[user_id];
			review_counts[i,0] = user_data_entry["review_count"];
		pl.hist(review_counts, bins=np.logspace(1, 4.0, 50));
		pl.gca().set_xscale("log")
		plt.title('User Review counts histogram.');
		pl.show()

		average_stars = np.zeros((u.count_iterable(self.user_data_dict),1));
		for i,user_id in enumerate(self.user_data_dict):
			user_data_entry = self.user_data_dict[user_id];
			average_stars[i,0] = user_data_entry["average_stars"];
		plt.hist(average_stars, bins=[0.5,1.5,2.5,3.5,4.5,5.5]);
		plt.title('User average stars histogram.');
		plt.show();

		fans_count = np.zeros((u.count_iterable(self.user_data_dict),1));
		for i,user_id in enumerate(self.user_data_dict):
			user_data_entry = self.user_data_dict[user_id];
			fans_count[i,0] = user_data_entry["fans"];
		pl.hist(fans_count, bins=np.logspace(1, 4.0, 50));
		pl.gca().set_xscale("log")
		pl.title('User fans count histogram');
		pl.show()
