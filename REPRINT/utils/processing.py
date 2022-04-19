import math
import numpy as np
from sklearn.utils import shuffle


def get_x_y_test(txt_path, bert_as_dict):
    lines = open(txt_path, encoding='utf-8').readlines()

    x = np.zeros((len(lines), 768))
    y = np.zeros((len(lines), ))

    for i, line in enumerate(lines):
        parts = line.strip().split('\t')
        label = int(parts[0])
        string = parts[1]
        try:
            assert string.strip() in bert_as_dict
        except:
            assert string in bert_as_dict
        try:
            embedding = bert_as_dict[string.strip()]
        except:
            embedding = bert_as_dict[string]
        x[i, :] = embedding
        y[i] = label
    
    x, y = shuffle(x, y, random_state = 0)
    return x, y

def get_label_to_embedding_list_unbalanced(
    txt_path, 
    bert_as_dict,
    label_to_n,
    aug_type='none',
):
    lines = open(txt_path, "r", encoding='utf-8').readlines()

    label_to_embedding_list = {}
    for i, line in enumerate(lines):
        parts = line.strip().split('\t')
        label = int(parts[0])
        sentence = parts[1]
        try:
            assert sentence.strip() in bert_as_dict
        except:
            assert sentence in bert_as_dict
        try:
            embedding = bert_as_dict[sentence.strip()]
        except:
            embedding = bert_as_dict[sentence]

        if label in label_to_embedding_list:
            if aug_type in ['synonym', 'insert', 'delete', 'swap', 'backtrans']:
                if len(label_to_embedding_list[label]) < 5 * label_to_n[label]:
                        label_to_embedding_list[label] += embedding
            else:
                if len(label_to_embedding_list[label]) < label_to_n[label]:
                    label_to_embedding_list[label].append(embedding)
        else:
            if aug_type in ['synonym', 'insert', 'delete', 'swap', 'backtrans']:
                label_to_embedding_list[label] = embedding
            else:
                label_to_embedding_list[label] = [embedding]
            
    label_to_embedding_np = {label: np.stack(embedding_list, axis=0) for label, embedding_list in label_to_embedding_list.items()}
    return label_to_embedding_np

def transform_to_xy(label_to_embedding_np, soft_label_dict = None, upspl=False):

    target_n_class = max(embedding_np.shape[0] for embedding_np in label_to_embedding_np.values())

    x_list = []
    y_list = []

    if soft_label_dict is not None:
        for label, embedding_np in label_to_embedding_np.items():

            embedding_np_target_n_class = [embedding_np]
            label_n_class = [soft_label_dict[label]]
            
            num_copies = math.ceil((target_n_class - embedding_np.shape[0]) / embedding_np.shape[0])
            if num_copies >= 1:
                for _ in range(num_copies):
                    embedding_np_target_n_class.append(embedding_np)
                    label_n_class.append(soft_label_dict[label])
            
            embedding_np_target_n_class = np.concatenate(embedding_np_target_n_class, axis=0)
            embedding_np_target_n_class = embedding_np_target_n_class[:target_n_class]
            label_n_class = np.concatenate(label_n_class, axis=0)
            label_n_class = label_n_class[:target_n_class]

            x_list.append(embedding_np_target_n_class)
            y_list.append(label_n_class)
    elif upspl:
        num_dup = int(len(label_to_embedding_np)/2) + 1
        for label, embedding_np in label_to_embedding_np.items():
    
            embedding_np_target_n_class = [embedding_np]
            
            num_copies = math.ceil((target_n_class - embedding_np.shape[0]) / embedding_np.shape[0])
            if num_copies >= 1:
                for _ in range(num_copies):
                    embedding_np_target_n_class.append(embedding_np)
            
            embedding_np_target_n_class = np.concatenate(embedding_np_target_n_class, axis=0)
            embedding_np_target_n_class = embedding_np_target_n_class[:target_n_class]
            y_n_class = np.stack([label for _ in range(target_n_class)])
            
            embedding_np_target_n_class = np.concatenate([embedding_np_target_n_class for _ in range(num_dup)], axis=0)
            y_n_class = np.concatenate([y_n_class for _ in range(num_dup)], axis=0)

            x_list.append(embedding_np_target_n_class)
            y_list.append(y_n_class)
    else:
        for label, embedding_np in label_to_embedding_np.items():

            embedding_np_target_n_class = [embedding_np]
            
            num_copies = math.ceil((target_n_class - embedding_np.shape[0]) / embedding_np.shape[0])
            if num_copies >= 1:
                for _ in range(num_copies):
                    embedding_np_target_n_class.append(embedding_np)
            
            embedding_np_target_n_class = np.concatenate(embedding_np_target_n_class, axis=0)
            embedding_np_target_n_class = embedding_np_target_n_class[:target_n_class]
            y_n_class = np.stack([label for _ in range(target_n_class)])

            x_list.append(embedding_np_target_n_class)
            y_list.append(y_n_class)

    x = np.concatenate(x_list)
    y = np.concatenate(y_list)
    
    x, y = shuffle(x, y, random_state = 0)
    return x, y

def transform_to_xy_smote(label_to_embedding_np):
    from imblearn.over_sampling import SMOTE
    
    x_list = []
    y_list = []
    min_n = 99999
    for label, embedding_np in label_to_embedding_np.items():

        embedding_np_target_n_class = embedding_np
        y_n_class = np.stack([label for _ in range(embedding_np.shape[0])])

        x_list.append(embedding_np_target_n_class)
        y_list.append(y_n_class)
        min_n = min(min_n, embedding_np.shape[0])
    
    x = np.concatenate(x_list)
    y = np.concatenate(y_list)
    
    sm = SMOTE(random_state=42, k_neighbors=min_n-1)
    x, y = sm.fit_resample(x, y)
    x, y = shuffle(x, y, random_state = 0)
    
    num_dup = int(len(label_to_embedding_np)/2) + 1
    x = np.concatenate([x for _ in range(num_dup)], axis=0)
    y = np.concatenate([y for _ in range(num_dup)], axis=0)
    
    return x, y
