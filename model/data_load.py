import numpy as np
def load_language_data(data_dir="data"):
    sentences = []
    labels = []

    language_map = {
        "english.txt": 0,
        "french.txt": 1,
        "german.txt": 2
    }

    for filename, label in language_map.items():
        path = f"{data_dir}/{filename}"
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                sentence = line.strip()
                if sentence == "":
                    continue
                sentences.append(sentence)
                labels.append(label)

    return sentences, np.array([labels])