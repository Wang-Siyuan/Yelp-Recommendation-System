from util import utility as u
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt

class Review:
	def __init__(self, data_set_file_path, business_data_dict):
		self.review_data = u.parse_data_set(data_set_file_path);
		self.review_data = self.filterRestaurantReview(business_data_dict);
		self.user_id_to_business_id_map = {};

	def filterRestaurantReview(self, business_data_dict):
		filtered_review_data = [];
		for review_data_entry in self.review_data:
			if review_data_entry['business_id'] in business_data_dict:
				filtered_review_data.append(review_data_entry);
		return filtered_review_data;

	def getUserIdToBusinessIdMap(self):
		if not self.user_id_to_business_id_map:	
			for review_data_entry in self.review_data:
				self.user_id_to_business_id_map[review_data_entry['user_id']] = review_data_entry['business_id'];
		return self.user_id_to_business_id_map;

	def getReviewData(self):
		return self.review_data;

	def generateStarsHistogram(self):
		stars = np.zeros((u.count_iterable(self.review_data),1));
		for i,review_data_entry in enumerate(self.review_data):
			stars[i,0] = review_data_entry["stars"];
		plt.hist(stars, bins=[0.5,1.5,2.5,3.5,4.5,5.5]);
		plt.title('Business stars histogram.');
		plt.show();



