
import argparse
import os
from vllm import LLM, SamplingParams

import gc
import tqdm
import torch
from diversity_metrics import DistinctNgrams,SentBert,SyntacticDiversity
import pandas as pd
#add model path and data path
parser = argparse.ArgumentParser(description='Diversity evaluation for Short-RL and Kimi')
parser.add_argument('--model_path', type=str, default="/volume/ailab4sci/txie/ydl/short_ablation2/VShortRL-logic1e-6-200-1/actor", help='path to the model')
parser.add_argument('--data_path', type=str, default="/volume/ailab4sci/txie/ydl/Short-RL/Logic-RL/data/kk/instruct/5ppl/test.parquet", help='path to the data')
parser.add_argument('--temp_dir', type=str, default="/volume/ailab4sci/txie/ydl/Short-RL/diversity-eval", help='path to the temp dir')
parser.add_argument('--rollout_n', type=int, default=8, help='number of rollouts')
parser.add_argument('--model_type', type=str, default="short_rl", help='model type: short_rl or kimi')
args = parser.parse_args()

datap=args.data_path
modeldir=args.model_path
rollout_n=args.rollout_n
temp_dir=args.temp_dir
model_type=args.model_type
#get all the folders in the directory  ## global_step_50


data=pd.read_parquet(datap)
config = {'n': rollout_n,'num_sets':1,'samples_per_set':rollout_n,'input_path':temp_dir}
Semantic=SentBert(config)
Lexical=DistinctNgrams(config)
Syntactic=SyntacticDiversity()

short_rl_folders = os.listdir(modeldir)



diversity_trend = pd.DataFrame(columns=['steps','Semantic','Lexical',"Syntactic","avg"])



sampling_params = SamplingParams(temperature=0.7, n=rollout_n,max_tokens=4096)
for folder in tqdm.tqdm(short_rl_folders, desc=model_type):
    steps= folder.split("_")[-1]
    steps= int(steps)
    modelpath= os.path.join(modeldir, folder)
    
    llm = LLM(model=modelpath,gpu_memory_utilization=0.4)
    ppl5_diversity_scorelist = []

    for i in tqdm.tqdm(range(len(data)), desc="5ppl"):
        prompt=data.iloc[i]['prompt'][0]['content']
        outputs = llm.generate(prompt, sampling_params)
        text_list = []
        for output in outputs:
            for i in range(rollout_n):
                generated_text = output.outputs[i].text
                print(f"Generated text: {generated_text!r}")
                if generated_text =='':
                    generated_text='</think>'
                text_list.append(generated_text)
        semantic_diversity_score = Semantic(0,text_list)
        Lexical_diversity_score  = Lexical(text_list)
        Syntactic_diversity_score= Syntactic(text_list)
        avg= (semantic_diversity_score + Lexical_diversity_score + Syntactic_diversity_score) / 3
        ppl5_diversity_scorelist.append({'steps': steps, 'Semantic': semantic_diversity_score, 'Lexical': Lexical_diversity_score, "Syntactic": Syntactic_diversity_score, 'avg': avg})
    length= len(ppl5_diversity_scorelist)
    #average the diversity score
    ppl5_diversity_score_avg = {'Semantic': [sum([d['Semantic'] for d in ppl5_diversity_scorelist]) / length], 'Lexical': [sum([d['Lexical'] for d in ppl5_diversity_scorelist]) / length], "Syntactic": [sum([d['Syntactic'] for d in ppl5_diversity_scorelist]) / length], 'steps': [steps],'avg': [sum([d['avg'] for d in ppl5_diversity_scorelist]) / length]}
    print(f"5ppl diversity score: {ppl5_diversity_score_avg}")
    diversity_trend = pd.concat([diversity_trend , pd.DataFrame(ppl5_diversity_score_avg)], ignore_index=True)

    del llm.llm_engine.model_executor.driver_worker
    gc.collect()
    torch.cuda.empty_cache()

#sort by steps and set index from 0 to n-1
diversity_trend = diversity_trend.sort_values(by='steps')
diversity_trend = diversity_trend.reset_index(drop=True)


#save to csv
diversity_trend.to_csv(f'{model_type}_diversity_trend.csv', index=False)
