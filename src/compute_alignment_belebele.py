import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cosine
import argparse
from collections import defaultdict


def compute_alignment(args):

    ''' 
    Load embeddings
    
    '''

    with open(os.path.join(args.embedding_path, "eng_Latn_choice1.pkl"), "rb") as pickle_file:
        english_choice1 = pickle.load(pickle_file)

    with open(os.path.join(args.embedding_path, "eng_Latn_choice2.pkl"), "rb") as pickle_file:
        english_choice2 = pickle.load(pickle_file)

    with open(os.path.join(args.embedding_path, "eng_Latn_choice3.pkl"), "rb") as pickle_file:
        english_choice3 = pickle.load(pickle_file)

    with open(os.path.join(args.embedding_path, "eng_Latn_choice4.pkl"), "rb") as pickle_file:
        english_choice4 = pickle.load(pickle_file)

    with open(os.path.join(args.embedding_path, f"{args.lang}_choice1.pkl"), "rb") as pickle_file:
        lang_choice1 = pickle.load(pickle_file)
    
    with open(os.path.join(args.embedding_path, f"{args.lang}_choice2.pkl"), "rb") as pickle_file:
        lang_choice2 = pickle.load(pickle_file)
    
    with open(os.path.join(args.embedding_path, f"{args.lang}_choice3.pkl"), "rb") as pickle_file:
        lang_choice3 = pickle.load(pickle_file)

    with open(os.path.join(args.embedding_path, f"{args.lang}_choice4.pkl"), "rb") as pickle_file:
        lang_choice4 = pickle.load(pickle_file)


    def cosine_similarity(array1, array2):
        cosine_dist = cosine(array1, array2)
        cosine_similarity = 1 - cosine_dist
        return cosine_similarity
    

    english_choice1_formatted_lasttoken = defaultdict(dict)
    english_choice2_formatted_lasttoken = defaultdict(dict)
    english_choice3_formatted_lasttoken = defaultdict(dict)
    english_choice4_formatted_lasttoken = defaultdict(dict)


    lang_choice1_formatted_lasttoken = defaultdict(dict)
    lang_choice2_formatted_lasttoken = defaultdict(dict)
    lang_choice3_formatted_lasttoken = defaultdict(dict)
    lang_choice4_formatted_lasttoken = defaultdict(dict)
    
    english_choice1_formatted_weighted = defaultdict(dict)
    english_choice2_formatted_weighted = defaultdict(dict)
    english_choice3_formatted_weighted = defaultdict(dict)
    english_choice4_formatted_weighted = defaultdict(dict)


    lang_choice1_formatted_weighted = defaultdict(dict)
    lang_choice2_formatted_weighted = defaultdict(dict)
    lang_choice3_formatted_weighted = defaultdict(dict)
    lang_choice4_formatted_weighted = defaultdict(dict)

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

        for item in english_choice3[layer]:
            english_choice3_formatted_lasttoken[layer][item['id']] = item['embd_lasttoken']
            english_choice3_formatted_weighted[layer][item['id']] = item['embd_weighted']

        for item in english_choice4[layer]:
            english_choice4_formatted_lasttoken[layer][item['id']] = item['embd_lasttoken']
            english_choice4_formatted_weighted[layer][item['id']] = item['embd_weighted']

        for item in lang_choice1[layer]:
            lang_choice1_formatted_lasttoken[layer][item['id']] = item['embd_lasttoken']
            lang_choice1_formatted_weighted[layer][item['id']] = item['embd_weighted']
    
        for item in lang_choice2[layer]:
            lang_choice2_formatted_lasttoken[layer][item['id']] = item['embd_lasttoken']
            lang_choice2_formatted_weighted[layer][item['id']] = item['embd_weighted']

        for item in lang_choice3[layer]:
            lang_choice3_formatted_lasttoken[layer][item['id']] = item['embd_lasttoken']
            lang_choice3_formatted_weighted[layer][item['id']] = item['embd_weighted']

        for item in lang_choice4[layer]:
            lang_choice4_formatted_lasttoken[layer][item['id']] = item['embd_lasttoken']
            lang_choice4_formatted_weighted[layer][item['id']] = item['embd_weighted']


    for layer in range(32):
        for idx in range(100):
            cs_11_weighted = cosine_similarity(english_choice1_formatted_weighted[layer][idx+1], lang_choice1_formatted_weighted[layer][idx+1])
            cs_22_weighted = cosine_similarity(english_choice2_formatted_weighted[layer][idx+1], lang_choice2_formatted_weighted[layer][idx+1])
            cs_33_weighted = cosine_similarity(english_choice3_formatted_weighted[layer][idx+1], lang_choice3_formatted_weighted[layer][idx+1])
            cs_44_weighted = cosine_similarity(english_choice4_formatted_weighted[layer][idx+1], lang_choice4_formatted_weighted[layer][idx+1])
    
            cs_12_weighted = cosine_similarity(english_choice1_formatted_weighted[layer][idx+1], lang_choice2_formatted_weighted[layer][idx+1])
            cs_21_weighted = cosine_similarity(english_choice2_formatted_weighted[layer][idx+1], lang_choice1_formatted_weighted[layer][idx+1])
            cs_13_weighted= cosine_similarity(english_choice1_formatted_weighted[layer][idx+1], lang_choice3_formatted_weighted[layer][idx+1])
            cs_31_weighted= cosine_similarity(english_choice3_formatted_weighted[layer][idx+1], lang_choice1_formatted_weighted[layer][idx+1])
            cs_14_weighted= cosine_similarity(english_choice1_formatted_weighted[layer][idx+1], lang_choice4_formatted_weighted[layer][idx+1])
            cs_41_weighted= cosine_similarity(english_choice4_formatted_weighted[layer][idx+1], lang_choice1_formatted_weighted[layer][idx+1])
            cs_23_weighted= cosine_similarity(english_choice2_formatted_weighted[layer][idx+1], lang_choice3_formatted_weighted[layer][idx+1])
            cs_32_weighted= cosine_similarity(english_choice3_formatted_weighted[layer][idx+1], lang_choice2_formatted_weighted[layer][idx+1])
            cs_24_weighted= cosine_similarity(english_choice2_formatted_weighted[layer][idx+1], lang_choice4_formatted_weighted[layer][idx+1])
            cs_42_weighted= cosine_similarity(english_choice4_formatted_weighted[layer][idx+1], lang_choice2_formatted_weighted[layer][idx+1])
            cs_34_weighted= cosine_similarity(english_choice3_formatted_weighted[layer][idx+1], lang_choice4_formatted_weighted[layer][idx+1])
            cs_43_weighted= cosine_similarity(english_choice4_formatted_weighted[layer][idx+1], lang_choice3_formatted_weighted[layer][idx+1])
            
            max_non_aligned_weighted = max(cs_12_weighted,cs_21_weighted,cs_13_weighted,cs_31_weighted,cs_14_weighted,cs_41_weighted,cs_23_weighted,cs_32_weighted,cs_24_weighted,cs_42_weighted,cs_34_weighted,cs_43_weighted)

            cs_11_lasttoken = cosine_similarity(english_choice1_formatted_lasttoken[layer][idx+1], lang_choice1_formatted_lasttoken[layer][idx+1])
            cs_22_lasttoken = cosine_similarity(english_choice2_formatted_lasttoken[layer][idx+1], lang_choice2_formatted_lasttoken[layer][idx+1])
            cs_33_lasttoken = cosine_similarity(english_choice3_formatted_lasttoken[layer][idx+1], lang_choice3_formatted_lasttoken[layer][idx+1])
            cs_44_lasttoken = cosine_similarity(english_choice4_formatted_lasttoken[layer][idx+1], lang_choice4_formatted_lasttoken[layer][idx+1])
            
            
            cs_12_lasttoken = cosine_similarity(english_choice1_formatted_lasttoken[layer][idx+1], lang_choice2_formatted_lasttoken[layer][idx+1])
            cs_21_lasttoken = cosine_similarity(english_choice2_formatted_lasttoken[layer][idx+1], lang_choice1_formatted_lasttoken[layer][idx+1])
            cs_13_lasttoken= cosine_similarity(english_choice1_formatted_lasttoken[layer][idx+1], lang_choice3_formatted_lasttoken[layer][idx+1])
            cs_31_lasttoken= cosine_similarity(english_choice3_formatted_lasttoken[layer][idx+1], lang_choice1_formatted_lasttoken[layer][idx+1])
            cs_14_lasttoken= cosine_similarity(english_choice1_formatted_lasttoken[layer][idx+1], lang_choice4_formatted_lasttoken[layer][idx+1])
            cs_41_lasttoken= cosine_similarity(english_choice4_formatted_lasttoken[layer][idx+1], lang_choice1_formatted_lasttoken[layer][idx+1])
            cs_23_lasttoken= cosine_similarity(english_choice2_formatted_lasttoken[layer][idx+1], lang_choice3_formatted_lasttoken[layer][idx+1])
            cs_32_lasttoken= cosine_similarity(english_choice3_formatted_lasttoken[layer][idx+1], lang_choice2_formatted_lasttoken[layer][idx+1])
            cs_24_lasttoken= cosine_similarity(english_choice2_formatted_lasttoken[layer][idx+1], lang_choice4_formatted_lasttoken[layer][idx+1])
            cs_42_lasttoken= cosine_similarity(english_choice4_formatted_lasttoken[layer][idx+1], lang_choice2_formatted_lasttoken[layer][idx+1])
            cs_34_lasttoken= cosine_similarity(english_choice3_formatted_lasttoken[layer][idx+1], lang_choice4_formatted_lasttoken[layer][idx+1])
            cs_43_lasttoken= cosine_similarity(english_choice4_formatted_lasttoken[layer][idx+1], lang_choice3_formatted_lasttoken[layer][idx+1])

            max_non_aligned_lasttoken = max(cs_12_lasttoken,cs_21_lasttoken,cs_13_lasttoken,cs_31_lasttoken,cs_14_lasttoken,cs_41_lasttoken,cs_23_lasttoken,cs_32_lasttoken,cs_24_lasttoken,cs_42_lasttoken,cs_34_lasttoken,cs_43_lasttoken)


            if (cs_11_lasttoken > max_non_aligned_lasttoken) & (cs_22_lasttoken > max_non_aligned_lasttoken) & (cs_33_lasttoken > max_non_aligned_lasttoken) & (cs_44_lasttoken > max_non_aligned_lasttoken):
                binary_alignment_matrix_lasttoken[idx][layer] = 1
            else:
                binary_alignment_matrix_lasttoken[idx][layer] = 0

            if (cs_11_weighted > max_non_aligned_weighted) & (cs_22_weighted > max_non_aligned_weighted) & (cs_33_weighted > max_non_aligned_weighted) & (cs_44_weighted > max_non_aligned_weighted):
                binary_alignment_matrix_weighted[idx][layer] = 1
            else:
                binary_alignment_matrix_weighted[idx][layer] = 0
    


   
    binary_dict_data_lasttoken = {int(k)+1: dict(v) for k, v in binary_alignment_matrix_lasttoken.items()}
    binary_dict_data_weighted = {int(k)+1: dict(v) for k, v in binary_alignment_matrix_weighted.items()}
    os.makedirs(args.save_path, exist_ok=True)  # Create the directory if it doesn't exist

    # Write to JSON file
    
    with open(args.save_path+f'DALI_{args.lang}_lasttoken.json', 'w') as f:
        json.dump(binary_dict_data_lasttoken, f, indent=4)

    with open(args.save_path+f'DALI_{args.lang}_weighted.json', 'w') as f:
        json.dump(binary_dict_data_weighted, f, indent=4)
    
    
if __name__ == '__main__':
     parser = argparse.ArgumentParser(description='Process Arguments for experiments with the selected LLM on various datasets')
     parser.add_argument('--llm_name', type=str, default="Llama3.1", help='LLM name')
     parser.add_argument('--lang', type=str, default = 'hin_Deva', help = 'language')
     parser.add_argument('--save_path', type=str, default='../alignment_outputs/Llama3.1/belebele_dali/')
     parser.add_argument('--embedding_path', type=str,default='/fs/nexus-scratch/kravisan/embeddings/Llama3.1/belebele_dali/')
     args = parser.parse_args()

     compute_alignment(args)


