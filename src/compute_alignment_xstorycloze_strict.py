import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cosine
import argparse
from collections import defaultdict
from datasets import load_dataset #load_dataset from Huggingface

def load_correct_answer(args):
    lang_dict = {'arabic': 'ar', 
                 'english': 'en', 
                 'spanish': 'es',
                 'basque': 'eu',
                 'indonesian': 'id',
                 'burmese': 'my',
                 'russian': 'ru',
                 'telugu': 'te',
                 'chinese': 'zh',
                 'swahili': 'sw',
                 'hindi': 'hi'}
    lang_code = lang_dict[args.lang]
    hf = load_dataset("juletxara/xstory_cloze", lang_code)
    answer_right_ending = hf['eval']['answer_right_ending']
    answer = []
    for i in range(len(answer_right_ending)):
        answer.append(answer_right_ending[i])
    return answer

def compute_alignment(args, answer):

    ''' 
    Load embeddings
    
    '''

    with open(os.path.join(args.embedding_path, "english_choice1.pkl"), "rb") as pickle_file:
        english_choice1 = pickle.load(pickle_file)

    with open(os.path.join(args.embedding_path, "english_choice2.pkl"), "rb") as pickle_file:
        english_choice2 = pickle.load(pickle_file)

    with open(os.path.join(args.embedding_path, f"{args.lang}_choice1.pkl"), "rb") as pickle_file:
        lang_choice1 = pickle.load(pickle_file)
    
    with open(os.path.join(args.embedding_path, f"{args.lang}_choice2.pkl"), "rb") as pickle_file:
        lang_choice2 = pickle.load(pickle_file)


    def cosine_similarity(array1, array2):
        cosine_dist = cosine(array1, array2)
        cosine_similarity = 1 - cosine_dist
        return cosine_similarity
    

    english_choice1_formatted_lasttoken = defaultdict(dict)
    english_choice2_formatted_lasttoken = defaultdict(dict)
    lang_choice1_formatted_lasttoken = defaultdict(dict)
    lang_choice2_formatted_lasttoken = defaultdict(dict)
    
    english_choice1_formatted_weighted = defaultdict(dict)
    english_choice2_formatted_weighted = defaultdict(dict)
    lang_choice1_formatted_weighted = defaultdict(dict)
    lang_choice2_formatted_weighted = defaultdict(dict)

    binary_alignment_matrix_lasttoken = defaultdict(dict)
    binary_alignment_matrix_weighted = defaultdict(dict)


    
    # Compute alignment per layer for each sentence
    for layer in range(32):  # Iterate over layers
        for item in english_choice1[layer]:    
            english_choice1_formatted_lasttoken[layer][item['id']] = item['embd_lasttoken']
            english_choice1_formatted_weighted[layer][item['id']] = item['embd_weighted']

        for item in english_choice2[layer]:
            english_choice2_formatted_lasttoken[layer][item['id']] = item['embd_lasttoken']
            english_choice2_formatted_weighted[layer][item['id']] = item['embd_weighted']

        for item in lang_choice1[layer]:
            lang_choice1_formatted_lasttoken[layer][item['id']] = item['embd_lasttoken']
            lang_choice1_formatted_weighted[layer][item['id']] = item['embd_weighted']
    
        for item in lang_choice2[layer]:
            lang_choice2_formatted_lasttoken[layer][item['id']] = item['embd_lasttoken']
            lang_choice2_formatted_weighted[layer][item['id']] = item['embd_weighted']


    for layer in range(32):
        for idx in range(1511):
            cs_11_lasttoken = cosine_similarity(english_choice1_formatted_lasttoken[layer][idx+1], lang_choice1_formatted_lasttoken[layer][idx+1])
            cs_22_lasttoken = cosine_similarity(english_choice2_formatted_lasttoken[layer][idx+1], lang_choice2_formatted_lasttoken[layer][idx+1])
        
            cs_21_lasttoken = cosine_similarity(english_choice2_formatted_lasttoken[layer][idx+1], lang_choice1_formatted_lasttoken[layer][idx+1])
            cs_12_lasttoken = cosine_similarity(english_choice1_formatted_lasttoken[layer][idx+1], lang_choice2_formatted_lasttoken[layer][idx+1])
            
           

            cs_11_weighted = cosine_similarity(english_choice1_formatted_weighted[layer][idx+1], lang_choice1_formatted_weighted[layer][idx+1])
            cs_22_weighted = cosine_similarity(english_choice2_formatted_weighted[layer][idx+1], lang_choice2_formatted_weighted[layer][idx+1])

            cs_21_weighted = cosine_similarity(english_choice2_formatted_weighted[layer][idx+1], lang_choice1_formatted_weighted[layer][idx+1])
            cs_12_weighted = cosine_similarity(english_choice1_formatted_weighted[layer][idx+1], lang_choice2_formatted_weighted[layer][idx+1])

            if answer[idx]==1:
                if (cs_11_lasttoken > cs_22_lasttoken) & (cs_11_lasttoken> cs_21_lasttoken) & (cs_11_lasttoken> cs_12_lasttoken):
                    binary_alignment_matrix_lasttoken[idx][layer] = 1
                else:
                    binary_alignment_matrix_lasttoken[idx][layer] = 0
                
                if (cs_11_weighted > cs_22_weighted) & (cs_11_weighted> cs_21_weighted) & (cs_11_weighted> cs_12_weighted):
                    binary_alignment_matrix_weighted[idx][layer] = 1
                else:
                    binary_alignment_matrix_weighted[idx][layer] = 0

            if answer[idx]==2:
                if (cs_22_lasttoken > cs_11_lasttoken) & (cs_22_lasttoken> cs_21_lasttoken) & (cs_22_lasttoken> cs_12_lasttoken):
                    binary_alignment_matrix_lasttoken[idx][layer] = 1
                else:
                    binary_alignment_matrix_lasttoken[idx][layer] = 0

                if (cs_22_weighted > cs_11_weighted) & (cs_22_weighted> cs_21_weighted) & (cs_22_weighted> cs_12_weighted):
                    binary_alignment_matrix_weighted[idx][layer] = 1
                else:
                    binary_alignment_matrix_weighted[idx][layer] = 0

          
              
    binary_dict_data_lasttoken = {k: dict(v) for k, v in binary_alignment_matrix_lasttoken.items()}
    binary_dict_data_weighted = {k: dict(v) for k, v in binary_alignment_matrix_weighted.items()}
    os.makedirs(args.save_path, exist_ok=True)  # Create the directory if it doesn't exist

    # Write to JSON file
    
    with open(args.save_path+f'DALI_{args.lang}_lasttoken.json', 'w') as f:
        json.dump(binary_dict_data_lasttoken, f, indent=4)

    with open(args.save_path+f'DALI_{args.lang}_weighted.json', 'w') as f:
        json.dump(binary_dict_data_weighted, f, indent=4)
    
    
if __name__ == '__main__':
     parser = argparse.ArgumentParser(description='Process Arguments for experiments with the selected LLM on various datasets')
     parser.add_argument('--llm_name', type=str, default="Llama3.1", help='LLM name')
     parser.add_argument('--lang', type=str, default = 'arabic', help = 'language')
     parser.add_argument('--save_path', type=str, default='../alignment_outputs/Llama3.1/xstorycloze_dali_strict/')
     parser.add_argument('--embedding_path', type=str,default='/fs/nexus-scratch/kravisan/embeddings/Llama3.1/xstorycloze_dali/')
     args = parser.parse_args()
     answer = load_correct_answer(args)
     compute_alignment(args, answer)


