import os
import os.path
from os.path import dirname
import pandas as pd

import torch.utils.data
import tensorflow as tf
from sklearn.utils import shuffle

from .utils import *
from FedWeit.utils import write_file, load_file


########################################################################################################################

def get_mixture(opt, fixed_order=False, base_dir=None):
    data = {}
    taskcla = []
    size = []  # [32, 32, 3]
    splits = ['train', 'test']

    idata = opt.dataset
    if not fixed_order:
        idata = list(shuffle(idata, random_state=opt.random_seed_data_gen))
    chkpt_save_folder = os.path.expanduser(os.path.join(base_dir, 'data', 'binary_mixture'))
    if not os.path.isdir(chkpt_save_folder):
        os.makedirs(chkpt_save_folder)

    if True or any([not os.path.exists(chkpt_save_folder + f'/data/binary_mixture/{dataset_name}_train_x.bin') for dataset_name in
                    idata]):
        # Pre-load
        for dataset_name in idata:
            dat = {}
            labels = set()
            max_length = 0
            for split in splits:
                dat[split] = NlpDataset(opt, os.path.join(base_dir, 'data', dataset_name), train_dev_test=split)
                max_length = max(max_length, dat[split].get_max_num_words())
                labels.update(list(dat[split].get_labels()))
            labels = sorted(labels)
            size.append(max_length)
            for split in splits:
                dat[split].set_labels(labels)

            tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="<UNK>")
            tokenizer.fit_on_texts([sent for _, dat_split in dat.items() for sent in dat_split.get_data()])
            # dat['maxlen'] = max_length

            save_filepath = os.path.abspath(
                os.path.join(dirname(dirname(dirname(__file__))), "Resources",
                             f"glove_{opt.embedding_dim}_{dataset_name}.npz"))
            print(f"save_filepath {save_filepath}")
            known_ids = set()

            # if not os.path.isfile(save_filepath):
            def create_embedding_matrix(filepath, word_index, embedding_dim: int = 300):
                words_found = 0
                # Adding again 1 because of reserved 0 index
                vocab_size = len(word_index) + 1
                np.random.seed(opt.random_seed_data_gen)
                embedding_matrix = np.random.uniform(-0.5 / vocab_size, 0.5 / vocab_size,
                                                     [vocab_size, embedding_dim])
                # embedding_matrix_known_words = []
                with open(filepath) as f:
                    for line in f:
                        word, *vector = line.split()
                        if word in word_index:
                            idx = word_index[word]
                            try:
                                embedding_matrix[idx] = np.array(vector, dtype=np.float32)[: embedding_dim]
                                # embedding_matrix_known_words.append(embedding_matrix[idx])
                                words_found += 1
                                known_ids.add(idx)
                            except ValueError:
                                continue
                words_not_found = vocab_size - 1 - words_found
                print(f"Glove: words_not_found: {words_not_found}, words_found: {words_found}")
                return embedding_matrix

            glove_filename = {100: "glove.6B.100d.txt", 200: "glove.6B.200d.txt",
                              300: "glove.840B.300d.txt"}[opt.embedding_dim]
            glove_filepath = os.path.abspath(
                join(dirname(dirname(dirname(__file__))), "Resources", glove_filename))
            embedding_matrix = create_embedding_matrix(
                glove_filepath, tokenizer.word_index, embedding_dim=opt.embedding_dim)
            print(f"Glove save path: {save_filepath}")
            np.savez_compressed(save_filepath, embeddings=embedding_matrix)
            # np.savez_compressed(save_filepath + "_", embeddings=embedding_matrix_known_words)

            data[dataset_name] = {"name": dataset_name, "ncla": len(labels), "labels": list(labels)}
            write_file(chkpt_save_folder, f'{dataset_name}_stats.json', data[dataset_name])
            for split in splits:
                dat[split].transform_data(tokenizer, max_length=max_length)
                loader = torch.utils.data.DataLoader(dat[split], batch_size=1, shuffle=False)
                data[dataset_name][split] = {'x': [], 'y': [], 'x_vector': []}
                unk_documents, total_documents = 0, 0
                for document, target in loader:
                    document = document.numpy()[0]
                    data[dataset_name][split]['x'].append(document)
                    data[dataset_name][split]['y'].append(target.numpy())

                    word_count = 0
                    word_embedding_mean_vector = np.zeros(opt.embedding_dim)
                    for idx in document:
                        if idx in known_ids:
                            word_count += 1
                            word_embedding_mean_vector += embedding_matrix[idx]
                    if word_count > 0:
                        word_embedding_mean_vector = word_embedding_mean_vector / word_count
                    else:
                        print(f"Unk Document: {document}")
                        unk_documents += 1
                    total_documents += 1
                    data[dataset_name][split]['x_vector'].append(word_embedding_mean_vector)
                print(f"{unk_documents} / {total_documents} documents had no known Words; given 0 embedding")

            for s in splits:
                torch.save(data[dataset_name][s]['x'], os.path.join(chkpt_save_folder, f'{dataset_name}_{s}_x.bin'))
                torch.save(data[dataset_name][s]['y'], os.path.join(chkpt_save_folder, f'{dataset_name}_{s}_y.bin'))

    else:
        # Load binary files
        for dataset_name in idata:
            data[dataset_name] = load_file(chkpt_save_folder, f'{dataset_name}_stats.json')
            for split in splits:
                data[dataset_name][split] = {'x': [], 'y': []}
                data[dataset_name][split]['x'] = torch.load(os.path.join(chkpt_save_folder, f'{dataset_name}_{s}_x.bin'))
                data[dataset_name][split]['y'] = torch.load(os.path.join(chkpt_save_folder, f'{dataset_name}_{s}_y.bin'))

    n = 0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n

    return data, taskcla, size


########################################################################################################################
class NlpDataset(torch.utils.data.Dataset):
    def __init__(self, opt, root: str, train_dev_test: str):
        self.root = os.path.expanduser(root)
        self.opt = opt

        if train_dev_test == "train":
            self.filename = "training.txt"
        # elif train_dev_test == "valid":
        #     self.filename = "validation.txt"
        elif train_dev_test == "test":
            self.filename = "test.txt"
        file_path = os.path.join(root, self.filename)
        # print(f"root: {root} filename: {self.filename}\n "
        #       f"opt.mixture_dir: {opt.mixture_dir} opt.dataset_folders: {opt.dataset_folders}")
        try:
            header_line = True if open(file_path, "r").readline() == 'LABEL\tDOCUMENT\n' else False
            preprocess_data = True if "\t" not in open(file_path, "r").readline() else False
        except UnicodeDecodeError:
            header_line = True if open(file_path, "rb").readline().decode() == 'LABEL\tDOCUMENT\n' else False
            preprocess_data = True if b"\t" not in open(file_path, "rb").readline() else False

        read_csv_params = dict(delimiter="\t", engine="python", quotechar=None,
                               quoting=3, error_bad_lines=False, warn_bad_lines=True, na_filter=False)
        if preprocess_data:
            with open(file_path, "rb") as r:
                sentences = r.readlines()
            for sentence_index in range(len(sentences)):
                sentence = sentences[sentence_index]
                space_index = sentence.index(b" ")
                sentences[sentence_index] = sentence[:space_index] + b"\t" + sentence[space_index + 1:]
            with open(file_path, "wb") as w:
                w.write(b"".join(sentences))

        if header_line:
            self.data = pd.read_csv(file_path, sep="\t", **read_csv_params)
        else:
            self.data = pd.read_csv(os.path.join(root, self.filename), sep="\t", header=None,
                                    names=["LABEL", "DOCUMENT"], **read_csv_params)

        self.labels = self.data["LABEL"]
        self.data = self.data["DOCUMENT"]
        self.max_num_words = max([document.count(" ") + 1 for document in self.data])

        unique_labels = self.labels.unique().tolist()
        self.labels_dict = {label: index for index, label in enumerate(unique_labels)}

    def get_labels(self):
        return self.labels_dict.keys()

    def set_labels(self, labels):
        self.labels_dict = {label: index for index, label in enumerate(labels)}
        print(f"set_labels_dict: {self.labels_dict}")

    def get_data(self):
        return self.data

    def get_max_num_words(self):
        return self.max_num_words

    # {'earn', 'grain', 'trade', 'money-fx', 'crude', 'acq', 'interest', 'ship'}
    def transform_data(self, tokenizer, max_length: int):
        self.data = tokenizer.texts_to_sequences(self.data)
        # if else currently redundant
        if self.filename == "training.txt":
            self.data = tf.keras.preprocessing.sequence.pad_sequences(self.data, maxlen=max_length, padding='post')
        else:
            self.data = tf.keras.preprocessing.sequence.pad_sequences(self.data, maxlen=max_length, padding='post')

    def __getitem__(self, index):
        """
        Args: index (int): Index
        Returns: tuple: (image, target) where target is index of the target class.
        """
        text, target = self.data[index], self.labels_dict[self.labels.iloc[index]]
        return text, target

    def __len__(self):
        return len(self.data)
