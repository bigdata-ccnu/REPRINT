import random
random.seed(42)


def create_shuffled_training_sets(
    dataset_folder, 
    dataset,
    num_seeds = 10,
):

    input_train_path = f"{dataset_folder}/{dataset}/train.txt"

    lines = open(input_train_path, 'r', encoding='utf-8').readlines()
    class_to_lines = {}

    for line in lines:
        _class = int(line.strip().split('\t')[0])
        if _class in class_to_lines:
            class_to_lines[_class].append(line)
        else:
            class_to_lines[_class] = [line]

    for seed in range(num_seeds):
        output_train_path = f"{dataset_folder}/{dataset}/train_s{seed}.txt"

        with open(output_train_path, "w", encoding='utf-8') as writer:
            for _class, lines in class_to_lines.items():
                lines_copy = lines
                random.shuffle(lines_copy)
                for line in lines_copy:
                    writer.write(line)


if __name__ == "__main__":

    dataset_folder = "full-datasets"
    dataset = "dbpedia"
    create_shuffled_training_sets(
        dataset_folder = dataset_folder,
        dataset = dataset,
    )
