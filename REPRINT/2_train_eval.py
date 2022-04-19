import numpy as np
from statistics import mean, stdev
from tqdm import tqdm

from utils import augmentation, common, processing, svm
import argparse

from sklearn.preprocessing import OneHotEncoder

parser = argparse.ArgumentParser(description='params of ratio')
parser.add_argument("--fix_ratio", type=bool, default=False, required=False,
                    help="If we use Fixed Ratio")
parser.add_argument("--ratio", type=float, default=None, required=False,
                    help="Fixed Ratio")
parser.add_argument("--number_of_basis", type=int, default=None, required=False,
                    help="Number of basis")
parser.add_argument("--soft_label", type=bool, default=False, required=False,
                    help="If we use soft_label")

dataset_to_n1 = {  # dataset name and number of training samples in each class
    'snips': 1800,
    'agnews': 2000, # 1900 test per class, 30000 train per class, 4 class
    'dbpedia': 2500, # 5000 test per class, too large, 40000 train per class, 14 class
    'yahoo': 3000, # 6000 test per class, too large, 140000 train per class, 10 class
}

num_labels = {  # dataset name and number of classes
    'snips': 7,
    'agnews': 4,
    'dbpedia': 14,
    'yahoo': 10,
}

def generate_label_to_n(n_0, n_1, num_labels=200):
    d = {}
    for i in range(0, num_labels, 2):
        d[i] = n_0
        d[i+1] = n_1
    return d

def run_experiments(
    aug_type,
    input_folder,
    dataset_name,
    noaug_bert,
    test_x,
    test_y,
    label_to_n,
    n_seeds = 10,
    args = None,
    ):
    
    train_acc_list = []
    acc_list = []
    for seed in tqdm(range(n_seeds)):

        bert_as_dict = noaug_bert
        label_to_embedding_np = processing.get_label_to_embedding_list_unbalanced(
            txt_path = f"{input_folder}/{dataset_name}/train_s{seed}.txt",
            bert_as_dict = bert_as_dict,
            label_to_n = label_to_n,
            aug_type = aug_type,
        )
        train_y_soft01 = None
        train_y_soft02 = None
        train_y_soft03 = None
        train_y_soft04 = None
        train_y_soft05 = None
        
        if "-extrapolate" in aug_type:
            label_to_embedding_np = augmentation.extrapolate_augmentation(
                label_to_embedding_np,
                num_extrapolations = int(aug_type.split('-')[0]),
            )
        elif "-pcapolate" in aug_type:
            if args.soft_label:
                label_to_embedding_np01, train_y_soft01 = augmentation.pcarotate_augmentation(
                    label_to_embedding_np,
                    num_extrapolations = int(aug_type.split('-')[0]),
                    number_of_basis=args.number_of_basis,
                    fix_ratio=args.fix_ratio,
                    ratio=args.ratio,
                    soft_label=args.soft_label,
                )
                label_to_embedding_np02, train_y_soft02 = augmentation.pcarotate_augmentation(
                    label_to_embedding_np,
                    num_extrapolations = int(aug_type.split('-')[0]),
                    number_of_basis=args.number_of_basis,
                    fix_ratio=args.fix_ratio,
                    ratio=args.ratio,
                    soft_label=args.soft_label,
                )
                label_to_embedding_np03, train_y_soft03 = augmentation.pcarotate_augmentation(
                    label_to_embedding_np,
                    num_extrapolations = int(aug_type.split('-')[0]),
                    number_of_basis=args.number_of_basis,
                    fix_ratio=args.fix_ratio,
                    ratio=args.ratio,
                    soft_label=args.soft_label,
                )
                label_to_embedding_np04, train_y_soft04 = augmentation.pcarotate_augmentation(
                    label_to_embedding_np,
                    num_extrapolations = int(aug_type.split('-')[0]),
                    number_of_basis=args.number_of_basis,
                    fix_ratio=args.fix_ratio,
                    ratio=args.ratio,
                    soft_label=args.soft_label,
                )
                label_to_embedding_np05, train_y_soft05 = augmentation.pcarotate_augmentation(
                    label_to_embedding_np,
                    num_extrapolations = int(aug_type.split('-')[0]),
                    number_of_basis=args.number_of_basis,
                    fix_ratio=args.fix_ratio,
                    ratio=args.ratio,
                    soft_label=args.soft_label,
                )
            else:
                label_to_embedding_np01 = augmentation.pcarotate_augmentation(
                    label_to_embedding_np,
                    num_extrapolations = int(aug_type.split('-')[0]),
                    number_of_basis=args.number_of_basis,
                    fix_ratio=args.fix_ratio,
                    ratio=args.ratio,
                    soft_label=args.soft_label,
                )
                label_to_embedding_np02 = augmentation.pcarotate_augmentation(
                    label_to_embedding_np,
                    num_extrapolations = int(aug_type.split('-')[0]),
                    number_of_basis=args.number_of_basis,
                    fix_ratio=args.fix_ratio,
                    ratio=args.ratio,
                    soft_label=args.soft_label,
                )
                label_to_embedding_np03 = augmentation.pcarotate_augmentation(
                    label_to_embedding_np,
                    num_extrapolations = int(aug_type.split('-')[0]),
                    number_of_basis=args.number_of_basis,
                    fix_ratio=args.fix_ratio,
                    ratio=args.ratio,
                    soft_label=args.soft_label,
                )
                label_to_embedding_np04 = augmentation.pcarotate_augmentation(
                    label_to_embedding_np,
                    num_extrapolations = int(aug_type.split('-')[0]),
                    number_of_basis=args.number_of_basis,
                    fix_ratio=args.fix_ratio,
                    ratio=args.ratio,
                    soft_label=args.soft_label,
                )
                label_to_embedding_np05 = augmentation.pcarotate_augmentation(
                    label_to_embedding_np,
                    num_extrapolations = int(aug_type.split('-')[0]),
                    number_of_basis=args.number_of_basis,
                    fix_ratio=args.fix_ratio,
                    ratio=args.ratio,
                    soft_label=args.soft_label,
                )
        elif "tmix" in aug_type:
            train_x_tmix, train_y_tmix = augmentation.tmix(label_to_embedding_np)
        elif aug_type == "eda":
            label_to_embedding_np = augmentation.eda(label_to_n, txt_path=f"{input_folder}/{dataset_name}/train_s{seed}.txt",)             
        elif aug_type == "linear-delta":
            label_to_embedding_np = augmentation.linear_delta_augmentation(label_to_embedding_np)
        elif aug_type == "within-extra":
            label_to_embedding_np = augmentation.within_extra_augmentation(label_to_embedding_np)
        elif aug_type == "gaussian":
            label_to_embedding_np = augmentation.gaussian_augmentation(label_to_embedding_np)

        if aug_type == 'none':
            train_x, train_y = processing.transform_to_xy(label_to_embedding_np, upspl=True)
        elif aug_type == 'smote':
            train_x, train_y = processing.transform_to_xy_smote(label_to_embedding_np)
        elif 'pcapolate' in aug_type:
            train_x01, train_y01 = processing.transform_to_xy(label_to_embedding_np01, train_y_soft01)
            train_x02, train_y02 = processing.transform_to_xy(label_to_embedding_np02, train_y_soft02)
            train_x03, train_y03 = processing.transform_to_xy(label_to_embedding_np03, train_y_soft03)
            train_x04, train_y04 = processing.transform_to_xy(label_to_embedding_np04, train_y_soft04)
            train_x05, train_y05 = processing.transform_to_xy(label_to_embedding_np05, train_y_soft05)
        else:
            train_x, train_y = processing.transform_to_xy(label_to_embedding_np)

        if "tmix" in aug_type:
            train_acc, acc = svm.train_eval_lr(
                train_x_tmix, train_y_tmix,
                test_x, test_y,
                num_seeds = 1,
                soft_label=True,
            )
            train_acc_list.append(train_acc)
            acc_list.append(acc)
        elif 'pcapolate' in aug_type:
            train_acc1, acc1 = svm.train_eval_lr(
                train_x01, train_y01,
                test_x, test_y,
                num_seeds = 1,
                soft_label=args.soft_label,
            )
            train_acc2, acc2 = svm.train_eval_lr(
                train_x02, train_y02,
                test_x, test_y,
                num_seeds = 1,
                soft_label=args.soft_label,
            )
            train_acc3, acc3 = svm.train_eval_lr(
                train_x03, train_y03,
                test_x, test_y,
                num_seeds = 1,
                soft_label=args.soft_label,
            )
            train_acc4, acc4 = svm.train_eval_lr(
                train_x04, train_y04,
                test_x, test_y,
                num_seeds = 1,
                soft_label=args.soft_label,
            )
            train_acc5, acc5 = svm.train_eval_lr(
                train_x05, train_y05,
                test_x, test_y,
                num_seeds = 1,
                soft_label=args.soft_label,
            )
            train_acc_list.append(train_acc1)
            acc_list.append(acc1)
            train_acc_list.append(train_acc2)
            acc_list.append(acc2)
            train_acc_list.append(train_acc3)
            acc_list.append(acc3)
            train_acc_list.append(train_acc4)
            acc_list.append(acc4)
            train_acc_list.append(train_acc5)
            acc_list.append(acc5)
        else:
            train_acc, acc = svm.train_eval_lr(
                train_x, train_y,
                test_x, test_y,
                num_seeds = 1,
            )
            train_acc_list.append(train_acc)
            acc_list.append(acc)

    print(f"{dataset_name} at {min(label_to_n.values())} to {max(label_to_n.values())} with aug {aug_type}: mean={mean(train_acc_list)*100:.1f}_{mean(acc_list)*100:.1f}, stdev={stdev(acc_list)*100:.1f}")


if __name__ == "__main__":
    args, _ = parser.parse_known_args()
    print(args)

    input_folder = "full-datasets"
    for dataset_name in [
        # 'sst1', 
        # 'sst2',
        # 'agnews',
        'dbpedia',
        # 'yahoo',
        # 'snips',
        # '20news',
        # 'huff',
        # 'fewrel',
    ]:

        noaug_bert = common.load_pickle(f"berts/{dataset_name}_noaug.pkl")
        test_x, test_y = processing.get_x_y_test(  # encoding and get test set
           txt_path=f"{input_folder}/{dataset_name}/test.txt",
           bert_as_dict=noaug_bert,
        )

        n_1 = dataset_to_n1[dataset_name]
        for label_to_n in [
            generate_label_to_n(n_0=16, n_1=n_1),
            generate_label_to_n(n_0=32, n_1=n_1),
            generate_label_to_n(n_0=64, n_1=n_1),
            generate_label_to_n(n_0=128, n_1=n_1),
            generate_label_to_n(n_0=256, n_1=n_1),
            # generate_label_to_n(n_0=n_1, n_1=n_1),
        ]:

            for aug_type in [
                # 'none',
                # 'smote',
                # 'tmix',
                # 'eda',
                # 'gaussian',
                # 'linear-delta',
                # 'within-extra',
                # '999-extrapolate',
                '999-pcapolate',
            ]:

                run_experiments(
                    aug_type=aug_type,
                    input_folder=input_folder,
                    dataset_name=dataset_name,
                    noaug_bert=noaug_bert,
                    test_x=test_x,
                    test_y=test_y,
                    label_to_n=label_to_n,
                    n_seeds=5,
                    args=args,
                )