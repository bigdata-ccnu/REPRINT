import numpy as np
from sklearn.preprocessing import OneHotEncoder
from .bert import get_embedding
from .eda import get_aug_method

def extrapolate_augmentation(
    label_to_embedding_np,
    original_weight = 1,
    num_extrapolations = 999,
    ):
    
    label_to_augmented = {}

    label_to_avg = {}

    for label, embedding_np in label_to_embedding_np.items():
        
        avg_embedding = np.mean(embedding_np, axis=0)
        label_to_avg[label] = avg_embedding
    
    for current_label, current_embedding_np in label_to_embedding_np.items():
        current_label_avg_embedding = label_to_avg[current_label]
        label_to_augmented[current_label] = []

        weight_multiplier = original_weight if current_label % 2 == 0 else 1
        label_to_augmented[current_label] = [current_embedding_np for _ in range(weight_multiplier)] 
        for other_label, other_embedding_np in label_to_embedding_np.items():
            if current_label != other_label and abs(current_label - other_label) <= num_extrapolations / 2:
                
                other_label_avg_embedding = label_to_avg[other_label]
                diff = other_label_avg_embedding - current_label_avg_embedding

                augmented_data = other_embedding_np - diff
                label_to_augmented[current_label].append(augmented_data)
    
    label_to_augmented = {k: np.concatenate(v) for k, v in label_to_augmented.items()}
    return label_to_augmented

def eda(label_to_n, txt_path,):
    lines = open(txt_path, "r", encoding='utf-8').readlines()
    min_n = min(label_to_n.values())
    max_n = max(label_to_n.values())
    multiplier = int(max_n/min_n) - 1
    
    label_to_embedding_list = {}
    label_count = {k:0 for k in label_to_n.keys()}
    for i, line in enumerate(lines):
        parts = line.strip().split('\t')
        label = int(parts[0])
        sentence = parts[1]
        label_count[label] += 1
        if label_count[label] > label_to_n[label]:
            continue
        else:
            aug_types = ['synonym', 'insert', 'delete', 'swap']
            # randomly choose one augmentation type in aug_types
            augmented_sentences = []
            for _ in range(multiplier):
                aug_type = np.random.choice(aug_types)
                aug_method = get_aug_method(aug_type)
                augmented_sentences += aug_method(
                    sentence,
                    n_aug = 1,
                    alpha = 0.1,
                )
            for s in [sentence] + augmented_sentences:
                if label in label_to_embedding_list:
                    label_to_embedding_list[label].append(get_embedding(s))
                else:
                    label_to_embedding_list[label] = [get_embedding(s)]
    label_to_embedding_np = {label: np.stack(emb, axis=0) for label, emb in label_to_embedding_list.items()}
    return label_to_embedding_np

def tmix(
    label_to_embedding_np,
    num_copies = None,
):

    n_class = len(label_to_embedding_np)
    num_copies = int(n_class/2) if num_copies is None else num_copies
    max_n_class = max(embedding_np.shape[0] for embedding_np in label_to_embedding_np.values())

    x_list = []
    y_list = []
    for label, embedding_np in label_to_embedding_np.items():
        multiplier = max(int(max_n_class / embedding_np.shape[0]), 1)
        embedding_np_target_n_class = np.concatenate([embedding_np for _ in range(multiplier)])
        y_n_class = np.stack([label for _ in range(embedding_np.shape[0]*multiplier)])

        x_list.append(embedding_np_target_n_class)
        y_list.append(y_n_class)
    x = np.concatenate(x_list)
    y = np.concatenate(y_list)

    lamb = np.random.beta(0.2,0.2)
    lamb = max(lamb, 1-lamb)
    x_ori = np.concatenate([x for _ in range(num_copies+1)])
    y_ori = np.concatenate([y for _ in range(num_copies+1)])
    idx_rerange = np.random.choice(np.arange(x.shape[0]), num_copies*x.shape[0], replace=True)
    x_pairs = np.concatenate([x, x[idx_rerange,:]])
    y_pairs = np.concatenate([y, y[idx_rerange]])
    enc = OneHotEncoder()
    augmented_x = lamb * x_ori + (1-lamb) * x_pairs
    y_ori = y_ori.reshape(-1,1)
    y_pairs = y_pairs.reshape(-1,1)
    augmented_y = lamb * enc.fit_transform(y_ori) + (1-lamb) * enc.fit_transform(y_pairs)
    
    return augmented_x, augmented_y

def pcarotate_augmentation(
    label_to_embedding_np,
    original_weight=1,
    num_extrapolations=999,
    number_of_basis=None,  # number of basis to consider
    fix_ratio = False,
    ratio = None,
    soft_label=False,
    ):
    
    label_to_augmented = {}
    out_soft_label_dict = {}
    basis_dict = {}
    proj_dict = {}
    mu_dict = {}
    ratios_dict = {}

    for label, embedding_np in label_to_embedding_np.items():
        
        basis_, proj_, mu_X_, ratios = center_and_pca_updated(embedding_np, number_of_basis, fix_ratio = fix_ratio, ratio=ratio)
        basis_dict[label] = basis_
        proj_dict[label] = proj_
        mu_dict[label] = mu_X_
        ratios_dict[label] = ratios
    
    if soft_label:
        enc = OneHotEncoder()
        enc.fit(np.array([i for i in range(len(label_to_embedding_np))]).reshape(-1,1))
    
    for current_label, current_embedding_np in label_to_embedding_np.items():

        target_number = current_embedding_np.shape[0]  # number of samples in target class
        current_basis = basis_dict[current_label]
        current_proj = proj_dict[current_label]
        current_mu = mu_dict[current_label]
        label_to_augmented[current_label] = []
        out_soft_label_dict[current_label] = []

        weight_multiplier = original_weight if current_label % 2 == 0 else 1
        label_to_augmented[current_label] = [current_embedding_np for _ in range(weight_multiplier)]
        if soft_label:
            ori_onehot_label = enc.transform(np.array([current_label for _ in range(target_number)]).reshape(-1,1)).toarray()
            out_soft_label_dict[current_label].append(ori_onehot_label)

        for other_label, other_embedding_np in label_to_embedding_np.items():
            if current_label != other_label and abs(current_label - other_label) <= num_extrapolations / 2:

                source_number = other_embedding_np.shape[0]  # number of samples in source class
                other_basis = basis_dict[other_label]
                other_proj = proj_dict[other_label]
                other_mu = mu_dict[other_label]

                current_index = np.random.choice(np.arange(target_number), source_number) # uniform sampling
                sampled_current_proj = current_proj[current_index]
                augmented_data = other_embedding_np - other_mu - other_proj + sampled_current_proj + current_mu
                
                if soft_label:
                    out_soft_label = generate_soft_label(other_basis, other_label, current_basis, current_label, n_class=len(label_to_embedding_np))
                    out_soft_label = np.repeat(out_soft_label, repeats = augmented_data.shape[0], axis=0)
                    out_soft_label_dict[current_label].append(out_soft_label)
                label_to_augmented[current_label].append(augmented_data)
    
    label_to_augmented = {k: np.concatenate(v) for k, v in label_to_augmented.items()}
    
    if soft_label:
        out_soft_label_dict = {k: np.concatenate(v) for k, v in out_soft_label_dict.items()}
        return label_to_augmented, out_soft_label_dict
    else:
        return label_to_augmented

def within_extra_augmentation(
    label_to_embedding_np,
    num_copies = None,
):

    n_class = len(label_to_embedding_np)
    num_copies = int(n_class/2) if num_copies is None else num_copies
    max_n_class = max(embedding_np.shape[0] for embedding_np in label_to_embedding_np.values())

    label_to_augmented = {}
    for label, embedding_np in label_to_embedding_np.items():
        multiplier = max(int(max_n_class / embedding_np.shape[0]), 1)
        label_to_augmented[label] = [np.concatenate([embedding_np for _ in range(multiplier)])]
        for _ in range(int(num_copies * multiplier)):
            embedding_np_noise = np.array(embedding_np, copy=True)
            np.random.shuffle(embedding_np_noise)
            augmented_data = embedding_np + (embedding_np - embedding_np_noise) / 2
            label_to_augmented[label].append(augmented_data)
    
    label_to_augmented = {k: np.concatenate(v) for k, v in label_to_augmented.items()}
    return label_to_augmented

def linear_delta_augmentation(
    label_to_embedding_np,
    num_copies = None,
):

    n_class = len(label_to_embedding_np)
    num_copies = int(n_class/2) if num_copies is None else num_copies
    max_n_class = max(embedding_np.shape[0] for embedding_np in label_to_embedding_np.values())

    label_to_augmented = {}
    for label, embedding_np in label_to_embedding_np.items():
        multiplier = max(int(max_n_class / embedding_np.shape[0]), 1)
        label_to_augmented[label] = [np.concatenate([embedding_np for _ in range(multiplier)])]
        for _ in range(int(num_copies * multiplier)):
            embedding_np_noise_1 = np.array(embedding_np, copy=True)
            np.random.shuffle(embedding_np_noise_1)
            embedding_np_noise_2 = np.array(embedding_np, copy=True)
            np.random.shuffle(embedding_np_noise_2)
            augmented_data = embedding_np + (embedding_np_noise_1 - embedding_np_noise_2)
            label_to_augmented[label].append(augmented_data)
    
    label_to_augmented = {k: np.concatenate(v) for k, v in label_to_augmented.items()}
    return label_to_augmented

def gaussian_augmentation(
    label_to_embedding_np,
    num_copies = None,
):

    n_class = len(label_to_embedding_np)
    num_copies = int(n_class/2) if num_copies is None else num_copies
    max_n_class = max(embedding_np.shape[0] for embedding_np in label_to_embedding_np.values())

    label_to_augmented = {}
    for label, embedding_np in label_to_embedding_np.items():
        multiplier = max(int(max_n_class / embedding_np.shape[0]), 1)
        label_to_augmented[label] = [np.concatenate([embedding_np for _ in range(multiplier)])]
        for _ in range(int(num_copies * multiplier)):
            noise = np.random.normal(0, 0.1, size = embedding_np.shape)
            augmented_data = embedding_np + noise
            label_to_augmented[label].append(augmented_data)
    
    label_to_augmented = {k: np.concatenate(v) for k, v in label_to_augmented.items()}
    return label_to_augmented


from sklearn.decomposition import PCA
def center_and_pca_updated(X, n_comp=None, fix_ratio=False, ratio=None):
    mu_X = np.mean(X, axis=0)
    X = X - mu_X
    if fix_ratio:
        pca = PCA(n_components=min(X.shape[1], X.shape[0]))
        pca.fit(X)
        explained_ratio_sum = np.cumsum(pca.explained_variance_ratio_)
        ratios = pca.explained_variance_ratio_[:10]
        ratio_keep = pca.explained_variance_ratio_[np.where(explained_ratio_sum<=ratio)]
        L_k = ratio_keep.shape[0]
        basis = pca.components_[:L_k,:]
    else:
        n_comp = min(X.shape[1], n_comp)
        pca = PCA(n_components=min(X.shape[1], X.shape[0]))
        pca.fit(X)
        ratios = pca.explained_variance_ratio_[:10]
        pca = PCA(n_components=n_comp)
        pca.fit(X)
        basis = pca.components_

    proj = np.matmul(np.matmul(X, np.transpose(basis)), basis)
    return basis, proj, mu_X, ratios


def generate_soft_label(source_basis, source_label, target_basis, target_label, n_class=None, epsilon=0.01):
    dim = source_basis.shape[1]
    determinant_source = (np.linalg.det((np.identity(dim) - np.matmul(np.transpose(source_basis), source_basis))))
    determinant_target = (np.linalg.det(np.matmul(np.transpose(target_basis), target_basis)))
    s_label = np.zeros(n_class)
    if determinant_source > 0 and determinant_target >0:
        lambda_value = determinant_source/(determinant_source + determinant_target + epsilon)
        s_label[source_label] = lambda_value
        s_label[target_label] = 1-lambda_value
    else:
        s_label[target_label] = 1
    return np.array(s_label).reshape(1, n_class)
