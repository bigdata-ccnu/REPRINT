def clean_agnews(
    input_csv,
    output_file,
):
    txt_lines = open(input_csv, 'r', encoding='utf-8').readlines()

    with open(output_file, "w", encoding='utf-8') as writer:
        for line in txt_lines:
            lst = line.split(",")
            lb = int(lst[0])
            line = " ".join([tok.strip().strip('"').strip("'").strip(" ") for tok in lst[1:]])
            output_line = f"{lb}\t{line.strip()}\n"
            writer.write(output_line)


if __name__ == "__main__":

    for split in ['train', 'test']:
        clean_agnews(
            input_csv = f"full-datasets/agnews_raw/{split}.csv",
            output_file = f"full-datasets/agnews/{split}.txt",
        )