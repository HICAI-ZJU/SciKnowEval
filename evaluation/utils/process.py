from collections import Counter
import math
import re
import json
from typing import List, Tuple, Any, Dict
import numpy as np
import re
import string
from scipy.spatial.distance import cosine
from gensim.models import KeyedVectors
import sys
import ast


def load_word2vec_model(model_path: str):
    try:
        return KeyedVectors.load_word2vec_format(model_path, binary=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{model_path}' not found.")
    except ValueError as e:
        raise ValueError(f"Error loading model from '{model_path}': {e}")


def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    if not s:
        raise ValueError("Input string is empty.")
    
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s

# 定义使用Word2Vec计算词间相似度的metric
# sentence -> vector
def sentence_to_vec(sentence: str, model):
    """
    将句子转换为词向量表示，并返回句子向量。
    transforms a sentence into a word vector representation

    Parameters:
    sentence (str): 输入的句子。
    model: 词向量模型。

    Returns:
    np.array: 句子向量。
    """
    
    words = sentence.split()
    word_vectors = []
    for word in words:
        if word in model:
            word_vectors.append(model[word])

    if not word_vectors:
        return np.zeros(model.vector_size)

    word_vectors = np.array(word_vectors)
    sentence_vector = word_vectors.mean(axis=0)
    return sentence_vector

def cosine_similarity(vect_dict, sentence1: str, sentence2: str):
    """
    Calculates the cosine similarity between two sentences.

    Parameters:
    vect_dict (dict): Dictionary of sentence vectors.
    sentence1 (str): First sentence.
    sentence2 (str): Second sentence.

    Returns:
    float: Cosine similarity between the two sentences.
    """

    try:
        vec1 = vect_dict[sentence1]
        vec2 = vect_dict[sentence2]
        return 1 - cosine(vec1, vec2)
    except ValueError as e:
        raise ValueError(f"Error calculating cosine similarity: {e}")
    except Exception as e:
        return 0  # 返回0表示找不到词

def cosine_similarity_2(count_dict, word1: str, word2: str) -> float:
    """
    Calculate the cosine similarity between two words.

    Parameters:
    count_dict (dict): Dictionary of word counts.
    word1 (str): First word.
    word2 (str): Second word.

    Returns:
    float: Cosine similarity between the two words.
    """

    try:
        vec1 = count_dict[word1]
        vec2 = count_dict[word2]

        dot_product = sum(vec1[word] * vec2[word] for word in vec1 if word in vec2)

        magnitude1 = math.sqrt(sum(vec1[word] ** 2 for word in vec1))
        magnitude2 = math.sqrt(sum(vec2[word] ** 2 for word in vec2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        else:
            return dot_product / (magnitude1 * magnitude2)
    except Exception as e:
        raise ValueError(f"Error calculating cosine similarity: {e}")

def same_entities(vect_dict, count_dict, word1: str, word2: str, threshold: float) -> bool:
    """
    判断两个词语是否表示相同的实体。
    Determine if two words are the same entity based on their similarity.

    Parameters:
    vect_dict (dict): Dictionary of word vectors.
    count_dict (dict): Dictionary of word counts.
    word1 (str): Word in first sentence.
    word2 (str): Word in second sentence.
    threshold (float): Similarity threshold.

    Returns:
    bool: True if the words are the same entity, False otherwise.
    """

    try:
        similarity_1 = cosine_similarity_2(count_dict, word1, word2)
        similarity_2 = cosine_similarity(vect_dict, word1, word2)

        return min(similarity_1, similarity_2) > threshold
    except ValueError as e:
        raise ValueError(f"Error comparing entities: {e}")
    except Exception as e:
        raise ValueError(f"Unexpected error comparing entities: {e}")