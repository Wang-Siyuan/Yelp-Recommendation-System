from util import utility as u
from pprint import pprint

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


