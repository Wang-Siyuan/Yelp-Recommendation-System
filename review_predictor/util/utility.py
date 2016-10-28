import json
import codecs
import numpy as np
import pprint

def parse_data_set(file_path):
    with codecs.open(file_path,'rU','utf-8') as f:
        for line in f:
           yield json.loads(line)

def write_data_set_to_file(data, file_path):
    with open(file_path, 'w') as outfile:
        json.dump(data, outfile)


def str_2_int(val):
    return int(val == True or val == 'true' or val == 'True');

def get_nullable_attribute(attribute_dict, attribute_key):
    if attribute_key in attribute_dict:
        return attribute_dict[attribute_key];
    else:
        return 0;

def get_nullable_attribute_with_str_2_int(attribute_dict, attribute_key):
    if attribute_key in attribute_dict:
        return str_2_int(attribute_dict[attribute_key]);
    else:
        return 0;

def get_nullable_attribute_with_expected_value(attribute_dict, attribute_key, expected_value):
    if attribute_key in attribute_dict:
        if attribute_dict[attribute_key] == expected_value:
            return 1;
        else:
            return 0;
    else:
        return 0;

def get_nullable_attribute_and_check_for_boolean_sub_attribute(attribute_dict, attribute_key, key_in_attribute_value):
    if attribute_key in attribute_dict:
        attribute_value = attribute_dict[attribute_key];
        if key_in_attribute_value in attribute_value and str_2_int(attribute_value[key_in_attribute_value]) == 1:
            return 1;
        else:
            return 0;
    else:
        return 0;

def get_nullable_attribute_with_contained_by_enumeration(attribute_dict, attribute_key, expected_value_list):
    if attribute_key in attribute_dict:
        if attribute_dict[attribute_key] in expected_value_list:
            return 1;
        else:
            return 0;
    else:
        return 0;

def get_nullable_attribute_with_boolean_dict(attribute_dict, key):
    if key in attribute_dict:
        attribute_value_dict = attribute_dict[key];
        for attribute_value_dict_key in attribute_value_dict:
            if str_2_int(attribute_value_dict[attribute_value_dict_key]) == 1:
                return 1;
        return 0;
    else:
        return 0;

def get_noise_level_num_value(attribute_dict):
    if 'Noise Level' not in attribute_dict:
        return 0;
    elif attribute_dict['Noise Level'] == 'very_loud':
        return -2;
    elif attribute_dict['Noise Level'] == 'loud':
        return -1;
    elif attribute_dict['Noise Level'] == 'average':
        return 0;
    elif attribute_dict['Noise Level'] == 'quiet':
        return 1;

def count_iterable(i):
    return sum(1 for e in i)

    