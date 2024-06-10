import os
import yaml
from typing import List, Any, Dict
from tqdm import tqdm
from scipy.spatial.distance import cosine
from gensim.models import KeyedVectors
from evaluation.utils.relation_extraction import *
from evaluation.utils.process import load_word2vec_model
from evaluation.utils.generation import calculate_nltk_scores, calculate_smiles_metrics

import tiktoken
from evaluation.utils.openai_api import OpenAIChat
script_dir = os.path.dirname(os.path.abspath(__file__))


# For true or false questions
def get_single_score_TF(d: Dict[str, Any]) -> int:
    """
    Evaluate turue or false questions.

    Parameters:
    d (dict): A single task data.

    Returns:
    score: The score of the data, i.e. 1 if the answer is correct, 0 otherwise.
    """
    pred = d['response'].strip()
    answer = d['answer']
    if answer.lower() in pred.lower():
        return 1
    else:
        if "true" in pred.lower():
            return int('Yes' == answer)
        elif "false" in pred.lower():
            return int('No' == answer)
    return 0

# For multiple choice questions
def get_single_score_MCQ(d: Dict[str, Any]) -> int:
    """
    Evaluate multiple choice questions (mcq-4-choices or mcq-2-choices).

    Parameters:
    d (dict): A single task data.

    Returns:
    score: The score of the data, i.e. 1 if the answer is correct, 0 otherwise.
    """

    pred = d['response'].strip()
    answer = d['answerKey']

    correct_options = d['choices']['text'][d['choices']['label'].index(answer)]
    wrong_options = [v for k, v in zip(d['choices']['label'], d['choices']['text']) if k != answer]
    assert len(wrong_options) == 3 or len(wrong_options) == 1
    if len(pred) == 0:
        return 0
    elif len(pred) == 1:
        return pred == answer
    elif str(correct_options) in pred and sum([int(str(op) in pred) for op in wrong_options]) == 0:
        return 1
    else:
        # if there are other contents other than the correct answer
        for key in ['D', 'C', 'B', 'A']:
            if (f"{key}." in pred or f"{key}\n" in pred or f"{key})" in pred or f"{key} " in pred or f"\"{key}" in pred or f"{key}\"" in pred or pred[0] == key):
                return (key == answer)
    return 0

# overall: true or false + MCQ
def get_score_CLS(data: List[Dict[str, Any]]) -> float:
    if len(data) == 0:
        return 0
    correct = 0
    for d in data:
        if isinstance(d['type'], list):
            d['type'] = d['type'][0]
        if d['response'] is None:
            continue
        if 'mcq' in d['type']:
            correct += get_single_score_MCQ(d)
        elif 'true' in d['type']:
            correct += get_single_score_TF(d)
        else:
            raise ValueError(f"unknown type: {d['type']}")
    return float(correct / len(data))

def get_score_reaction(data: List[Dict[str, Any]]) -> float:
    correct = 0
    for d in data:
        if 'mcq' in d['type']:
            correct += get_single_score_MCQ(d)
        elif d['type'] == 'filling':
            if d['response'] is None:
                continue
            candidates = d['response'].strip().split('.')       # candidate smiles list
            for c in candidates:
                if c in d['answer']:
                    correct += 1
                    break
        else:
            raise ValueError(f"unknown type: {d['type']}")
    return float(correct / len(data))

def get_score_filling(data: List[Dict[str, Any]]) -> float:
    correct = 0
    for d in data:
        if d['response'] is None:
            continue
        pred = d['response'].strip()
        answer = d['answer'].strip()
        if answer in pred:
            correct += 1
    return float(correct / len(data))

def get_score_RE_tuples(word2vec_model_path: str, data: List[Dict[str, Any]]) -> float:
    """
    Evaluate the tuple extraction task.

    Parameters:
    data (list): A list of task data.

    Returns:
    score: The F1 score of the data.
    """

    word2vec_model = load_word2vec_model(model_path=word2vec_model_path)
    tuples_pred, chemical_pred, disease_pred, word2vec_dict1, word2count_dict1 = validate_format_and_extract_data_tuples(word2vec_model, [d['response'] for d in data])
    tuples_answer, chemical_answer, disease_answer, word2vec_dict2, word2count_dict2 = validate_format_and_extract_data_tuples(word2vec_model, [d['answer'] for d in data])
    
    word2vec_dict = {**word2vec_dict1, **word2vec_dict2}
    word2count_dict = {**word2count_dict1, **word2count_dict2}

    try:
        chemical_cos_f1_score = cos_f1_score(word2vec_dict, word2count_dict, chemical_pred, chemical_answer)
        diseases_cos_f1_score = cos_f1_score(word2vec_dict, word2count_dict, disease_pred, disease_answer)
        re_macro_f1_score = macro_f1_score_tuples(word2vec_dict, word2count_dict, tuples_pred, tuples_answer)
        task2_score = ((chemical_cos_f1_score + diseases_cos_f1_score)/2 + re_macro_f1_score)/2
        return task2_score
    except ValueError as e:
        raise ValueError(f"Error in task2_score: {e}")
    except Exception as e:
        raise ValueError(f"Unexpected error in task2_score: {e}")

def get_score_RE_triplets(word2vec_model_path: str, data: List[Dict[str, Any]]) -> float:
    """
    Evaluate the triplet extraction task.

    Parameters:
    data (list): A list of task data.

    Returns:
    score: The F1 score of the data.
    """

    word2vec_model = load_word2vec_model(model_path=word2vec_model_path)
    # extract data
    triplets_pred, drug_pred, relation_pred, word2vec_dict1, word2count_dict1 = validate_format_and_extract_data_triplets(word2vec_model, [d['response'] for d in data])
    triplets_answer, drug_answer, relation_answer, word2vec_dict2, word2count_dict2 = validate_format_and_extract_data_triplets(word2vec_model, [d['answer'] for d in data])
    word2vec_dict = {**word2vec_dict1, **word2vec_dict2}
    word2count_dict = {**word2count_dict1, **word2count_dict2}

    try:
        drug_cos_f1_score = cos_f1_score(word2vec_dict, word2count_dict, drug_pred, drug_answer)
        relation_cos_f1_score = cos_f1_score(word2vec_dict, word2count_dict, relation_pred, relation_answer)
        re_macro_f1_score = macro_f1_score_triplets(word2vec_dict, word2count_dict, triplets_pred, triplets_answer)
        total_f1_score = ((drug_cos_f1_score + relation_cos_f1_score)/2 + re_macro_f1_score)/2
        return total_f1_score
    except ValueError as e:
        raise ValueError(f"Error in total_f1_score: {e}")
    except Exception as e:
        raise ValueError(f"Unexpected error in total_f1_score: {e}")

def get_score_BLEU_ROUGE(data: List[Dict[str, Any]]) -> float:
    """
    Supports tasks that are suitable for BLEU, ROUGE, and METEOR scoring.

    Parameters:
    data (list): A list of task data.

    Returns:
    score: The score of the data.
    """

    tokenizer = tiktoken.encoding_for_model("gpt-4")
    pred_strs = [d['response'].strip() for d in data]
    ans_strs = [d['answer'].strip() for d in data]
    
    return calculate_nltk_scores(tokenizer, ans_strs, pred_strs)

def get_score_Mol_GEN(data: List[Dict[str, Any]]) -> float:
    """
    Supports the Molecular Generation task.
    
    Parameters:
    data (list): A list of task data.

    Returns:
    score: The score of the data.
    """

    pred_strs = [[sorted(d['response'].strip().split(), key=lambda x: len(x))[-1] if len(d['response'].strip().split()) > 1 else d['response'].strip()] for d in data]
    ans_strs = [[d['answer'].strip()] for d in data]

    return calculate_smiles_metrics(pred_strs,ans_strs)


def get_score_GPT4(data: List[Dict[str, Any]], task: str, evaluator: str = "gpt-4o") -> float:
    """
    Supports tasks that are suitable for GPT-4 scoring.

    Parameters:
    data (list): A list of task data.

    Returns:
    score: The score of the data.
    """

    try:
        data = [d for d in data if d['response'] != "" and d['answer'] != ""]
        if len(data) == 0:
            return 0
        score_mcq_dict = {
            'A': '0.5',
            'B': '0.75',
            'C': '1.0',
            'D': '0.25',
            'E': '0.0',
        }
        task_trans_dict = {
            'chemical_text_summary': 'text_summary',
            'biological_text_summary': 'text_summary',
            'chemical_harmful_QA': 'harmful_QA',
            'biological_harmful_QA': 'harmful_QA',
            'chemical_reagent_generation': 'reagent_generation',
            'biological_reagent_generation': 'reagent_generation',
            'chemical_procedure_generation': 'procedure_generation',
            'biological_procedure_generation': 'procedure_generation',
            'extract_doping': 'extract_doping',
        }
        prompts = yaml.load(open(os.path.join(script_dir, 'utils', 'prompts', 'prompt.yaml')), Loader=yaml.FullLoader)
        prompt = prompts[task_trans_dict[task]]
        messages = [
            [
                {"role": "system", "content": prompt['system']},
                {"role": "user", "content": prompt['user'].format(**d)},
            ] for d in data
        ]

        model = OpenAIChat(model_name=evaluator, max_tokens=64, temperature=0.0, top_p=1.0, response_format='text')
        batch_size = 100
        total_len = range(0, len(messages), batch_size)
        for index in tqdm(total_len, total=len(total_len), desc=f'eval on {task}'):
            responses_index = model.batch_run(messages[index:index+batch_size])
            for i in range(len(responses_index)):
                data[index+i]['eval'] = {'default': responses_index[i] if responses_index[i] != None else ''}
        if prompt['type'] == 'T/F':
            correct = sum([1 for d in data if 'Yes'.lower() in d['eval']['default'].lower()])
            return float(correct / len(data))
        elif prompt['type'] == 'score':
            data = [d for d in data if "Rating:" in d['eval']['default']]
            for i, d in enumerate(data):
                data[i]['eval']['score'] = int(d['eval']['default'].split("Rating:")[1].strip()[0])
            return sum([d['eval']['score'] for d in data]) / len(data)
        elif prompt['type'] == 'MCQ':
            data = [d for d in data if "(" in d['eval']['default']]
            for i, d in enumerate(data):
                data[i]['eval']['score'] = score_mcq_dict[d['eval']['default'].split('(')[1][0]]
            return sum([float(d['eval']['score']) for d in data]) / len(data)
        else:
            raise ValueError('prompt type {} not found'.format(prompt['type']))

    except:
        raise ValueError('task {} not found'.format(task))
