import json

import argparse
parser = argparse.ArgumentParser(description='Process datasets for DeepScaler training')
parser.add_argument('--datap', default="./deepscaler/data/orz_math_57k_collected.json",
                   help='Local directory to save processed datasets')
parser.add_argument('--savepath', default="./deepscaler/data/orzmath/orz_math_57k_collected.json")
args = parser.parse_args()
datap = args.datap
data = json.load(open(datap))


newdata = [{'question': d[0]['value'], 'answer': d[1]['ground_truth']['value']} for d in data]
#save to a new json file
savepath=args.savepath
json.dump(newdata, open(savepath, 'w'))