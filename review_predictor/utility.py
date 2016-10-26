import json
import codecs


def parse_data_set(file_path):
    with codecs.open(file_path,'rU','utf-8') as f:
        for line in f:
           yield json.loads(line)


def str_2_int(val):
	return int(val == 'true');

def get_nullable_attribute(attribute_dict, key):
	if key in attribute_dict:
		return attribute_dict[key];
	else:
		return 0;

def get_nullable_attribute_with_str_2_int(attribute_dict, key):
	if key in attribute_dict:
		return str_2_int(attribute_dict[key]);
	else:
		return 0;

def get_nullable_attribute_with_expected_value(attribute_dict, key, expected_value):
	if key in attribute_dict:
		if attribute_dict[key] == expected_value:
			return 1;
		else:
			return 0;
	else:
		return 0;