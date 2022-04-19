from torch import embedding
from tqdm import tqdm

from utils import bert, common, eda

import os
os.environ['NO_PROXY'] = 'huggingface.co'

import numpy as np
from utils.augmentation import compute_kernel_bias, transform_and_normalize

def cache_bert_aug(
    dataset_path,
    aug_output_path,
    aug_type,
):

    input_path = f"{dataset_path}/train.txt"

    bert_as_dict = {}
    aug_method = eda.get_aug_method(aug_type)

    lines = open(input_path, 'r', encoding='utf-8').readlines()

    for line in tqdm(lines):
        parts = line.strip().split('\t')
        sentence = parts[1]
        augmented_sentences = aug_method(
            sentence,
            n_aug = 4,
            alpha = 0.1,
        )

        embeddings = []
        for s in [sentence] + augmented_sentences:
            embeddings.append(bert.get_embedding(s))
        bert_as_dict[sentence] = embeddings
    
    common.save_pickle(aug_output_path, bert_as_dict)
    print(f"augmented bert as dict len {len(bert_as_dict)} saved in {aug_output_path}")


def cache_bert_noaug(
    dataset_path,
    output_path,
    whiten = False,
    ):

    input_paths = [
        f"{dataset_path}/train.txt",
        f"{dataset_path}/test.txt",
    ]

    bert_as_dict = {}
    lines = []
    for input_path in input_paths:
        newlines = open(input_path, 'r', encoding='utf-8').readlines()
        lines += newlines

    # part of identifying unique sentence num
    # for line in tqdm(lines):
    
    # 1: Solution 1 for Normal Bert
    for line in tqdm(lines):
        parts = line.split('\t')
        sentence = parts[1].strip()
        embedding = bert.get_embedding(sentence)
        bert_as_dict[sentence] = embedding
    
    # TODO: Updated 1 for CNN-like cache
    # a long sentence can be represented as a lot of short n-gram, we can make use of the n-gram(can be shrinked)
    # to train the classification model, the core is, how to shrink the size n-gram and how to use bert the embed
    # the n-gram(sum or weighted sum)

    # 2: Solution 2 for Sbert
    # lines_list = []
    # for line in lines:
    #     parts = line.strip().split('\t')
    #     sentence = parts[1]
    #     lines_list.append(sentence)
    # embeddings = bert.get_embedding(lines_list)
    # for i, line in enumerate(lines):
    #     parts = line.strip().split('\t')
    #     sentence = parts[1]
    #     bert_as_dict[sentence] = embeddings[i,:]
    
    if whiten:
        embedding_all = np.empty([1,768]) # TODO: Replace with embed_size param
        for sentence, embedding_np in bert_as_dict.items():
            embedding_np = embedding_np.reshape(1,768)
            embedding_all = np.append(embedding_all, embedding_np, axis=0)
        kernel, bias = compute_kernel_bias(embedding_all, n_components=768)
        for sentence, embedding_np in bert_as_dict.items():
            embedding_np = transform_and_normalize(embedding_np, kernel, bias)
            bert_as_dict[sentence] = embedding_np.flatten()
    
    common.save_pickle(output_path, bert_as_dict)
    print(f"bert as dict len {len(bert_as_dict)} saved in {output_path}")


if __name__ == "__main__":

    output_folder = "alberts"

    for dataset_name in [
        # "20news",
        # "agnews",
        # "dbpedia",
        # "yahoo",
        "snips",
        # "fewrel",
        # "huff",
        # "sst2", 
        # "sst1", 
        # "subj", 
        # "cov", 
        # "trec", 
        # "clinc"
    ]:

        cache_bert_noaug(
            dataset_path = f"full-datasets/{dataset_name}",
            output_path = f"{output_folder}/{dataset_name}_noaug.pkl",
            whiten=False,
        )

        # for aug_type in ['delete', 'synonym', 'insert', 'swap']: #['backtrans', 'delete', 'synonym', 'insert', 'swap'] 
        #     cache_bert_aug(
        #         dataset_path = f"full-datasets/{dataset_name}",
        #         aug_output_path = f"{output_folder}/{dataset_name}_trainaug_{aug_type}.pkl",
        #         aug_type = aug_type,
        #     )