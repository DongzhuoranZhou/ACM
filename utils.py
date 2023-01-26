import yaml
from itertools import product
from collections import OrderedDict
import copy


def hyperparameters_combination(yaml_path=None):
    if yaml_path == None:
        yaml_path = "~/../configs/config_for_all_layers.yml"
    with open(yaml_path, "r") as file:
        context = yaml.load(file, Loader=yaml.FullLoader)
    list_of_arg_with_list = list()
    for arg in context.keys():
        if arg == "random_seeds":
            continue
        if type(context[arg]) == list:
            list_of_arg_with_list.append(arg)
    print("list_of_arg_with_list: ", list_of_arg_with_list)
    dict_of_arg_with_list = OrderedDict({arg_key: context[arg_key] for arg_key in list_of_arg_with_list})
    print("dict_of_arg_with_list: ", dict_of_arg_with_list)
    list_new_complete_config = list()
    for arg_values in product(*dict_of_arg_with_list.values()):
        new_arg_dict = {arg_key: value for arg_key, value in zip(list_of_arg_with_list, arg_values)}
        context.update(new_arg_dict)
        context_new = copy.deepcopy(context)
        list_new_complete_config.append(context_new)
    return list_new_complete_config

if __name__ == "__main__":
    hyperparameters_combination()