import os
import pandas as pd

from utils.preprocess_text import preprocess_text


def prepare_data(root_path, dataset):
    if dataset == 'odeuropa':

        folder = os.path.join(root_path, "Odeuropa/benchmarks_and_corpora/benchmarks/EN/webanno")
        cleansed_path = os.path.join(folder, "odeuropa_preprocessed.csv")

        # Load preprocessed data if already exist
        if os.path.isfile(cleansed_path):
            df = pd.read_csv(cleansed_path)

        else:

            print("Preprocessing Data")

            dataframes = []
            for subfolder in sorted(os.listdir(folder)):
                subfolder_path = subfolder
                tables = []
                subfolder_path = os.path.join(folder, subfolder_path)
                for file in os.scandir(subfolder_path):
                    if file.is_file():
                        annotation_path = os.path.join(subfolder_path, file.name)

                        table = pd.read_table(annotation_path, comment='#', error_bad_lines=False, engine="python",
                                              header=None, quoting=3, quotechar=None)
                        table = table.rename(columns={0: "token_id", 1: "char_range", 2: "token", 3: "ref_type"})

                        if not 'ref_type' in table:
                            continue
                        table['filename'] = annotation_path.split('/')[-2]
                        tables.append(table)

                dataframes.append(pd.concat(tables, ignore_index=True, axis=0))

            refs = pd.concat(dataframes)

            refs['sentence_id'] = refs['token_id'].apply(lambda x: x.split('-')[0])
            refs = refs.drop_duplicates(subset=['filename', 'sentence_id', 'token_id'])
            refs = refs[~refs['ref_type'].isna()]
            refs['ref_type'] = refs['ref_type'].apply(lambda x: 'O' if x == '_' else '1')
            refs = refs.reset_index(drop='true')

            sentences = refs.groupby(['filename', 'sentence_id'], as_index=False).agg({'token': ' '.join})
            encodings = refs.groupby(['filename', 'sentence_id'], as_index=False).agg({'ref_type': ' '.join}).rename(
                columns={'ref_type': 'labels'})

            encodings['labels'] = encodings['labels'].apply(mark_beginning_and_intermediate_token)

            df = pd.merge(sentences, encodings, on=["filename", "sentence_id"])
            df = df.rename(columns={'token': 'text'})
            df['contains_ref'] = df['labels'].apply(contains_ref)

            df = preprocess_text(df)
            df.to_csv(cleansed_path)

        print(df)

    elif dataset == 'auditory':
        data = None

    return df


def contains_ref(labels):
    labels = labels.split()
    for label in labels:
        if label != "O":
            return True
    return False


def get_ref_type(ref_type):
    if '[' in ref_type:
        return str(ref_type).split('[')[0]
    else:
        return ref_type


def mark_beginning_and_intermediate_token(label):
    tokens = label.split()

    for i in range(len(tokens)):
        if tokens[i] != 'O':
            tokens[i] = 'B-' + tokens[i]

    for i in range(len(tokens) - 1):
        if tokens[i] != 'O' and tokens[i + 1] != 'O':
            if tokens[i + 1].replace("B-", '') in tokens[i]:
                tokens[i + 1] = tokens[i + 1].replace("B-", "I-")

    return ' '.join(tokens)
