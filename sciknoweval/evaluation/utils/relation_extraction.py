from collections import Counter
import re
from typing import List, Tuple, Any, Dict
import numpy as np
import re
from evaluation.utils.process import same_entities, sentence_to_vec, cosine_similarity, cosine_similarity_2



def parse_tuples(tuple_str):
    """
    parse the binary tuple string and convert it to a list format.
    
    Parameters:
    param tuple_str (str): the binary tuple string, in the format of "(A1,B1),(A2,B2),..."
    
    Returns:
    list of tuples: the parsed binary tuple list
    """

    tuple_str = tuple_str.strip()
    if tuple_str[0] == '[':
        tuple_str = tuple_str[1:]
    if tuple_str[-1] == ']':
        tuple_str = tuple_str[:-1]
    # remove the quotes if there are any
    tuple_str = tuple_str.strip('"')
    tuple_str = tuple_str.lower()

    # ('A1','B1'), ('A2','B2'), ...
    if "('" in tuple_str:
        tuple_str = tuple_str.replace("('", "(").replace("','", ",").replace("', '", ", ").replace("')", ")")
    if "['" in tuple_str:
        tuple_str = tuple_str.replace("['", "[").replace("','", ",").replace("', '", ", ").replace("']", "]")
    
    # initialize the result list
    tuples = []
    
    
    # split by comma, but ignore commas in parentheses
    if '),' in tuple_str:
        tuples_list = re.split(r'\s*\),\s*\(', tuple_str)
        
        for tuple_str in tuples_list:
            # remove parentheses
            clean_tuple_str = tuple_str.replace('(', '').replace(')', '')
            # split by comma
            tuple_elements = clean_tuple_str.split(',')
            
            if len(tuple_elements) == 2:
                A, B = tuple_elements
                tuples.append((A.strip(), B.strip()))
            else:
                raise ValueError(f"Invalid tuple: {tuple_str}")
    elif '],' in tuple_str:
        tuples_list = re.split(r'\s*\],\s*\[', tuple_str)
        
        for tuple_str in tuples_list:
            # remove parentheses
            clean_tuple_str = tuple_str.replace('[', '').replace(']', '')
            # split by comma
            tuple_elements = clean_tuple_str.split(',')
            
            if len(tuple_elements) == 2:
                A, B = tuple_elements
                tuples.append((A.strip(), B.strip()))
            else:
                raise ValueError(f"Invalid tuple: {tuple_str}")
    
    return tuples

def parse_triplets(triplets_str):
    """
    parse the triplets string and convert it to a list format.
    
    Parameters:
    triplets_str (str): the triplets string, in the format of "(A1,B1,C1),(A2,B2,C2),..."
    
    Returns:
    list of tuples: the parsed triplets list.
    """

    triplets_str = triplets_str.strip()

    if triplets_str[0] == '[':
        triplets_str = triplets_str[1:]
    if triplets_str[-1] == ']':
        triplets_str = triplets_str[:-1]
    
    triplets_str = triplets_str.strip('"')
    triplets_str = triplets_str.lower()

    # ('A1','B1','C1'), ('A2','B2','C2'),...
    if "('" in triplets_str:
        triplets_str = triplets_str.replace("('", "(").replace("','", ",").replace("', '", ", ").replace("')", ")")
    if "['" in triplets_str:
        triplets_str = triplets_str.replace("['", "[").replace("','", ",").replace("', '", ", ").replace("']", "]")
    # initialize the result list
    triplets = []
    
    # split the string using regex, but ignore commas inside parentheses
    if '),' in triplets_str:
        triplets_list = re.split(r'\s*\),\s*\(', triplets_str)
        for triplet in triplets_list:
            # remove the parentheses
            clean_triplet = triplet.replace('(', '').replace(')', '')
            # split the elements
            triplet_elements = clean_triplet.split(',')
            if len(triplet_elements) == 3:
                A, B, C = triplet_elements
                triplets.append((A.strip(), B.strip(), C.strip()))
            else:
                raise ValueError(f"Invalid triplet: {triplet}")
    elif '],' in triplets_str:
        triplets_list = re.split(r'\s*\],\s*\[', triplets_str)
        for triplet in triplets_list:
            # remove the parentheses
            clean_triplet = triplet.replace('[', '').replace(']', '')
            # split the elements
            triplet_elements = clean_triplet.split(',')
            if len(triplet_elements) == 3:
                A, B, C = triplet_elements
                triplets.append((A.strip(), B.strip(), C.strip()))
            else:
                raise ValueError(f"Invalid triplet: {triplet}")
            
    return triplets


def validate_format_and_extract_data_tuples(model, data: List[str]) -> Tuple[List[List[Any]], List[str], List[str], Dict[Any, Any], Dict[Any, Any]]:
    """
    Extract (chemical, disease) 2-tuples from the data.
    
    Parameters:
    data (List[Dict]): readed JSONL data list.
    
    Returns:
    Tuple[bool, str, List[List[Any]], List[str], List[str], List[str]]:
    - the extracted (chemical, disease) tuples.
    - the extracted chemical list.
    - the extracted disease list.
    - the word2vec dict.
    - the word2count dict.
    """
    all_tuples = []
    list_chemical = []
    list_disease = []
    word2vec_dict = {} 
    word2count_dict = {}
    
    for item in data:
        tuples = [('', '')]
        try:
            tuple_str = item
            if tuple_str:
                tuples = parse_tuples(tuple_str)
        except:           
            tuples = [('', '')]
        finally:
            all_tuples.extend(tuples)
            for tuple in tuples:
                list_chemical.append(tuple[0])
                list_disease.append(tuple[1])
                word2vec_dict[tuple[0]] = sentence_to_vec(tuple[0], model)
                word2vec_dict[tuple[1]] = sentence_to_vec(tuple[1], model)
                word2count_dict[tuple[0]] = Counter(tuple[0])
                word2count_dict[tuple[1]] = Counter(tuple[1])
    list_chemical = sorted(set(list_chemical), key=str.lower)
    list_disease = sorted(set(list_disease), key=str.lower)
    
    return all_tuples, list_chemical, list_disease, word2vec_dict, word2count_dict

def validate_format_and_extract_data_triplets(model, data: List[str]) -> Tuple[List[List[Any]], List[str], List[str], Dict[Any, Any], Dict[Any, Any]]:
    """
    Extract (drugA, relationship, drugB) triplets from the data.

    Parameters:
    data (List[Dict]): readed JSONL data list.

    Returns:
    Tuple[List[List[Any]], List[str], List[str], List[str]]:
    - the extracted (drugA, relationship, drugB) triplets.
    - the extracted drug list.
    - the extracted relationship list.
    - the word2vec dict.
    - the word2count dict.
    """
    all_triplets = []
    list_drug = []
    list_drugA = []
    list_drugB = []
    list_relationship = []
    word2vec_dict = {} 
    word2count_dict = {}
    for item in data:
        triplets = [('', '', '')]
        try:
            triplets_str = item
            if triplets_str:
                triplets = parse_triplets(triplets_str)
        except:
            triplets = [('', '', '')]
        finally:
            # pre-process the sentence to vector operation to avoid redundant calculation
            all_triplets.extend(triplets)
            for triplet in triplets:
                list_drugA.append(triplet[0])
                list_drugB.append(triplet[2])
                list_relationship.append(triplet[1])
                word2vec_dict[triplet[0]] = sentence_to_vec(triplet[0], model)
                word2vec_dict[triplet[2]] = sentence_to_vec(triplet[2], model)
                word2vec_dict[triplet[1]] = sentence_to_vec(triplet[1], model)
                word2count_dict[triplet[0]] = Counter(triplet[0])
                word2count_dict[triplet[2]] = Counter(triplet[2])
                word2count_dict[triplet[1]] = Counter(triplet[1])
    
    list_drug = sorted(set(list_drugA + list_drugB), key=str.lower)
    list_relationship = sorted(set(list_relationship), key=str.lower)
    
    return all_triplets, list_drug, list_relationship, word2vec_dict, word2count_dict



def match_tuple(vec_dict, count_dict, compare:List[Any],bases:List[List[Any]]) -> bool:
    for base in bases:
        if compare[0] == base[0] and compare[1] == base[1] :
            found_match = True
            break
        elif same_entities(vec_dict, count_dict, compare[0], base[0], 0.95) and same_entities(vec_dict, count_dict, compare[1], base[1], 0.95):
            found_match = True
            break
        else:
            continue
    return found_match

def match_triplet(vec_dict, count_dict, compare:List[Any],bases:List[List[Any]]) -> bool:
    found_match = False
    for base in bases:
        if compare[0] == base[0] and compare[1] == base[1] and compare[2] == base[2] :
            found_match = True
            break
        elif same_entities(vec_dict, count_dict, compare[0], base[0], 0.95) and same_entities(vec_dict, count_dict, compare[1], base[1], 0.95) and same_entities(vec_dict, count_dict, compare[2], base[2], 0.95):
            found_match = True
            break
        else:
            continue
    return found_match

def cos_f1_score(vec_dict, count_dict, prediction: List[str], answers: List[str]) -> float:
    """
    计算基于余弦相似度的 F1 分数。
    Calculates F1 score based on cosine similarity.

    Parameters:
    prediction (List[str]): The predicted entity list.
    answers (List[str]): The true entity list.

    Returns:
    float: The F1 score.
    """
    try:
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        matched_ground_truth_tokens = set()
        similarity_threshold = 0.9  
        for pred_entity in prediction:
            found_match = False
            for idx, true_entity in enumerate(answers):
                similarity = min(cosine_similarity(vec_dict, pred_entity, true_entity), cosine_similarity_2(count_dict, pred_entity, true_entity))
                if similarity > similarity_threshold:
                    found_match = True
                    if idx not in matched_ground_truth_tokens:
                        true_positives += 1
                        matched_ground_truth_tokens.add(idx)
                    break
            if not found_match:
                false_positives += 1
        false_negatives = len(answers) - len(matched_ground_truth_tokens)
        
        # Calculate precision, recall, and F1 score
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return f1
    except ValueError as e:
        raise ValueError(f"Error calculating F1 score for NER: {e}")
    except Exception as e:
        raise ValueError(f"Unexpected error calculating F1 score for NER: {e}")

def macro_f1_score_triplets(vec_dict, count_dict, prediction: List[List[Any]], answers: List[List[Any]]) -> float:
    """
    Calculates macro-average F1 score for triplet extraction.

    Parameters:
    prediction (List[List[Any]]): The predicted triplet list.
    answers (List[List[Any]]): The true triplet list.

    Returns:
    float: The macro-average F1 score.
    """
    
    try:
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for pred_entity in prediction:
            if match_triplet(vec_dict, count_dict, pred_entity, answers):
                true_positives += 1
            else:
                false_positives += 1

        for true_entity in answers:
            if match_triplet(vec_dict, count_dict, true_entity, prediction) == False:
                false_negatives += 1

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return f1
    except ValueError as e:
        raise ValueError(f"Error calculating macro F1 score for triplet RE: {e}")
    except Exception as e:
        raise ValueError(f"Unexpected error calculating macro F1 score for triplet RE: {e}")

def macro_f1_score_tuples(vec_dict, count_dict, prediction: List[List[Any]], answers: List[List[Any]]) -> float:
    """
    Calculates macro-average F1 score for tuple extraction.

    Parameters:
    prediction (List[List[Any]]): The predicted tuple list.
    answers (List[List[Any]]): The true tuple list.

    Returns:
    float: The macro-average F1 score.
    """

    try:
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        for pred_entity in prediction:
            if match_tuple(vec_dict, count_dict, pred_entity, answers):
                true_positives += 1
            else:
                false_positives += 1

        for true_entity in answers:
            if match_tuple(vec_dict, count_dict, true_entity, prediction) == False:
                false_negatives += 1

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return f1
    except ValueError as e:
        raise ValueError(f"Error calculating macro F1 score for tuple RE: {e}")
    except Exception as e:
        raise ValueError(f"Unexpected error calculating macro F1 score for tuple RE: {e}")