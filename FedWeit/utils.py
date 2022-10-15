import codecs
import json
import os
import pickle
import pprint
import random
from datetime import datetime

import numpy as np

# from FedWeit.models.utils import bhattacharya_distance, bhatta_dist

pp = pprint.PrettyPrinter(indent=4)


def syslog(pid, message, logger=None):
    if pid == -2:
        worker = 'server'
    elif pid == -1:
        worker = 'data'
    else:
        worker = 'client:' + str(pid)
    output = '[%s][%s] %s' % (datetime.now().strftime("%Y/%m/%d-%H:%M:%S"), worker, message)
    if logger:
        logger.info(output)
    else:
        print(output)


def random_shuffle(seed, ls, secondary_ls=None):
    random.seed(seed)
    random.shuffle(ls)
    if secondary_ls:
        random.seed(seed)
        random.shuffle(secondary_ls)


def random_sample(seed, ls, num_pick):
    random.seed(seed)
    return random.sample(ls, num_pick)


def obj_to_pickle_string(x):
    return codecs.encode(pickle.dumps(x), "base64").decode()


def pickle_string_to_obj(s):
    return pickle.loads(codecs.decode(s.encode(), "base64"))


# def get_serialized_weights(weights):
#     return json.dumps([w.tolist() for w in weights])
#
#
# def get_weights_from_string(str_weights):
#     weights = json.loads(str_weights)
#     return [np.array(w) for w in weights]


def write_file(filepath, filename, data):
    if not os.path.isdir(filepath):
        os.makedirs(filepath)
    with open(os.path.join(filepath, filename), 'w+') as outfile:
        json.dump(data, outfile, indent="\t", default=convert_numpy_int)


def load_file(filepath, filename):
    with open(os.path.join(filepath, filename), 'r') as infile:
        return json.loads(infile.read())


def np_save(base_dir, filename, data):
    if not os.path.isdir(base_dir):
        os.makedirs(base_dir)
    np.save(os.path.join(base_dir, filename), data)


def pickle_save(base_dir, filename, data):
    with open(os.path.join(base_dir, filename), 'wb') as out:
        pickle.dump(data, out, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(path):
    with open(path, 'rb') as out:
        loaded = pickle.load(out)
    return loaded


def load_weights(path):
    return np.load(path, allow_pickle=True)


# def compare_numpys(w1, w2):
#     a = np.abs(np.subtract(w1, w2))
#     b = np.ravel(a[0])
#     c = np.ravel(a[1])
#     d = np.sum(b) + np.sum(c)
#     if d == 0:
#         print('weights are equal')
#     else:
#         print('weights are not equal')


def compare_(path):
    filenames = ['post-weights-1.npy', 'pre-weights-0.npy', 'reuters8_10.npy', 'pre-weights-1.npy',
                 'x_train.npy', 'post-weights-0.npy', 'post-weights-2.npy', 'pre-weights-2.npy']
    for filename in filenames:
        print(filename)
        a = np.load(os.path.join(path, "1", filename), allow_pickle=True).item()
        b = np.load(os.path.join(path, "2", filename), allow_pickle=True).item()
        equal = True
        for key in a:
            print(key)
            print(a[key])
            equal = str(a[key]) == str(b[key])
            if not equal:
                break
        print(f"Equal: {equal}")


def convert_numpy_int(o):
    """
    Numpy int objects cannot be json dumped directly. This is a simple script to convert them back to basic int

    @param o: numpy int object
    @return:
    """
    if isinstance(o, np.int64):
        return int(o)
    raise TypeError


# def get_setting(opt):
#     setting = ''
#     if opt.model == 0:
#         setting += '-cnn'
#     elif opt.model == 1:
#         setting += '-apd'
#     if opt.federated:
#         setting += '-fcl'
#     else:
#         setting += '-cl'
#     return setting
#
#
# def get_model(value, model):
#     if model == 'cnn':
#         return value == 0
#     elif model == 'apd':
#         return value == 1
#     elif model == 'abc_l2t':
#         return value == 2
#     elif model == 'abc_apd':
#         return value == 3
#     elif model == 'ewc':
#         return value == 4
#     else:
#         SystemExit('SystemExit: no proper model was given. see help: -h')
