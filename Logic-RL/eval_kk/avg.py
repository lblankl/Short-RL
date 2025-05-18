import argparse
parser = argparse.ArgumentParser(description="avg")
parser.add_argument("--log_path", "-l", type=str, default="log.txt", help="Log file path")

# read all the {x}.json file   in the log_path every in this form: {'acc'：acc}

args = parser.parse_args()

log_path = args.log_path
import os
import json
import numpy as np

def read_json_files(log_path):
    json_files = []
    for file in os.listdir(log_path):
        if file.endswith(".json"):
            json_files.append(os.path.join(log_path, file))
    return json_files

def read_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def extract_acc(data):
    acc = []
    for key, value in data.items():
        if isinstance(value, dict) and 'acc' in value:
            acc.append(value['acc'])
    return acc
#calculate the average of the acc

def calculate_average(acc_list):
    if len(acc_list) == 0:
        return 0
    return np.mean(acc_list)

def main():
    json_files = read_json_files(log_path)
    all_acc = []
    for file in json_files:
        print(file)
        data = read_json_file(file)
       
        acc = data.get('acc', 0)
        all_acc.append(acc)
    average_acc = calculate_average(all_acc)
    print(f"Average accuracy: {average_acc}")
    #save the average acc to a json file
    with open(os.path.join(log_path, 'average_acc.json'), 'w') as f:
        json.dump({'average_acc': average_acc}, f)
if __name__ == "__main__":
    main()