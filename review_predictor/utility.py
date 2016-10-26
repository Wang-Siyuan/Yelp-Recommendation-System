import json
import codecs
from pprint import pprint


def parse_data_set()
    user_data = []
    with codecs.open('data-set/yelp_academic_dataset_user.json','rU','utf-8') as f:
        for line in f:
           yield json.loads(line)

