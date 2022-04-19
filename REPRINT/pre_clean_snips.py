def clean_snips(
    input_txt,
    input_labels,
    output_file,
):
    txt_lines = open(input_txt, 'r', encoding='utf-8').readlines()
    labels = open(input_labels, 'r', encoding='utf-8').readlines()

    labels_set = list(set(labels))
    label_to_class_num = {label:i for i, label in enumerate(labels_set)}
    
    with open(output_file, "w", encoding='utf-8') as writer:
        for label, line in zip(labels, txt_lines):
            output_line = f"{label_to_class_num[label]}\t{line.strip()}\n"
            writer.write(output_line)


if __name__ == "__main__":

    for split in ['train', 'test']:
        clean_snips(
            input_txt = f"full-datasets/snips_raw/{split}/seq.in",
            input_labels = f"full-datasets/snips_raw/{split}/label.txt",
            output_file = f"full-datasets/snips/{split}.txt",
        )