def clean_yahoo(
    input_csv,
    output_file,
):
    txt_lines = open(input_csv, 'r', encoding='utf-8').readlines()

    num_class = [0 for _ in range(10)]
    max_num_per_class = 6000

    with open(output_file, "w", encoding='utf-8') as writer:
        for line in txt_lines:
            lst = line.split(",")
            lb = int(lst[0].strip('"'))
            if num_class[lb-1] == max_num_per_class:
                continue
            else:
                line = " ".join([tok.strip().strip('"').strip("'").strip(" ") for tok in lst[1:]])
                output_line = f"{lb}\t{line.strip()}\n"
                writer.write(output_line)
                num_class[lb-1] += 1


if __name__ == "__main__":

    for split in ['train', 'test']:
        clean_yahoo(
            input_csv = f"full-datasets/yahoo_raw/{split}.csv",
            output_file = f"full-datasets/yahoo/{split}.txt",
        )