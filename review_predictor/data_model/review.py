from util import utility as u
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt

class Review:
    def __init__(self, data_set_file_path, user_data_dict, business_data_dict):
        self.review_data = u.parse_data_set(data_set_file_path)
        self.review_data = self.filterRestaurantReview(user_data_dict, business_data_dict)
        self.top_users_review_count_dict = self.generateSortedTopUserReviewCountDict()
        self.user_Id_to_business_Id_map = self.generateUserIdToBusinessIdMap()
        self.indexedReviewData = self.indexReviewData()

    def filterRestaurantReview(self, user_data_dict, business_data_dict):
        filtered_review_data = [];
        for review_data_entry in self.review_data:
            if review_data_entry['business_Id'] in business_data_dict and review_data_entry['user_Id'] in user_data_dict:
                filtered_review_data.append(review_data_entry);
        return filtered_review_data;

    def indexReviewData(self):
        indexedReviewData = {}
        for review_data_entry in self.review_data:
            user_Id = review_data_entry['user_Id']
            if user_Id not in self.top_users_review_count_dict:
                continue;
            else:
                business_Id = review_data_entry['business_Id']
                indexedReviewData[(user_Id,business_Id)] = review_data_entry["stars"]
        return indexedReviewData

    def getIndexedReviewData(self):
        return self.indexedReviewData

    def generateSortedTopUserReviewCountDict(self):
        user_review_count_dict = {}
        for review_data_entry in self.review_data:
            user_Id = review_data_entry['user_Id']
            if user_Id not in user_review_count_dict:
                user_review_count_dict[user_Id] = 1
            else:
                user_review_count_dict[user_Id] += 1
        top_users_with_most_reviews = sorted(user_review_count_dict.keys(), key=lambda i: user_review_count_dict[i])[-100:]
        top_users_with_most_reviews.reverse();
        top_users_review_count_dict = {}
        for user_Id in top_users_with_most_reviews:
            top_users_review_count_dict[user_Id] = user_review_count_dict[user_Id]
        return top_users_review_count_dict

    def getSortedTopUserReviewCountDict(self):
        return self.top_users_review_count_dict

    def generateUserIdToBusinessIdMap(self):
        ret = {}
        for review_data_entry in self.review_data:
            user_Id = review_data_entry['user_Id']
            if user_Id not in self.top_users_review_count_dict:
                continue;
            if user_Id not in ret:
                ret[user_Id] = [review_data_entry['business_Id']]
            else:
                ret[user_Id].append(review_data_entry['business_Id'])
        return ret

    def getUserIdToBusinessIdMap(self):
        return self.user_Id_to_business_Id_map

    def getReviewData(self):
        return self.review_data;

    def generateStarsHistogram(self):
        stars = np.zeros((u.count_iterable(self.review_data),1));
        for i,review_data_entry in enumerate(self.review_data):
            stars[i,0] = review_data_entry["stars"];
        plt.hist(stars, bins=[0.5,1.5,2.5,3.5,4.5,5.5]);
        plt.title('Business stars histogram.');
        plt.show();



