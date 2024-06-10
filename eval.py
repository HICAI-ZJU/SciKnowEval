import json
import os
import argparse
from evaluation.metrics import *
from evaluation.define import get_task_func, get_task_data, reformat_result

import warnings
warnings.filterwarnings("ignore")


def main(args):
    # Load data
    print(">>>>>> Loading data...")
    f = open(args.data_path, 'r', encoding='utf-8')
    try:
        if args.data_path.endswith('.json'):
            data = json.load(f)
        else:
            data = [json.loads(line) for line in f.readlines()]
    except:
        raise ValueError("Failed to load data.")
    
    eval_func = get_task_func()
    task_data = get_task_data(data=data)
    if not os.path.exists(args.output_path):
        os.makedirs(os.path.dirname(args.output_path))

    # Evaluate
    results = {}
    for task, func in eval_func.items():
        if func == get_score_GPT4 and os.environ.get('OPENAI_API_KEY') is None:
            results[task] = None
            continue
        if func == get_score_GPT4:
            inputs = {'data': task_data[task], 'evaluator': args.gen_evaluator}
        elif func in [get_score_RE_triplets, get_score_RE_tuples]:
            inputs = {'data': task_data[task], 'word2vec_model_path': args.word2vec_model_path}
        else:
            inputs = {'data': task_data[task]}

        results[task] = func(**inputs)
    results = reformat_result(results)

    # Save results
    print(">>>>>> Saving results...")
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="path to the JSON file needed to be evaluated.")
    parser.add_argument("--word2vec_model_path", type=str, required=True, help="path to the word2vec model, e.g. GoogleNews-vectors-negative300.bin")
    parser.add_argument("--gen_evaluator", type=str, required=True, help="the OPENAI model used to evaluate generation tasks.")
    parser.add_argument("--output_path", type=str, default="./outputs.json", help="the path to save the evaluation results.")
    args = parser.parse_args()

    main(args)