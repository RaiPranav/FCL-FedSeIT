import tensorflow as tf

from FedWeit.utils import *
from FedWeit.models.global_model import GlobalModel
from FedWeit.models.utils import get_conv_fc_layers


class GlobalFedWeIT(GlobalModel):
    def __init__(self, opt, logger):
        print("Global init")
        super(GlobalFedWeIT, self).__init__(opt, logger)
        self.opt = opt
        self.current_task = 0
        # self.input_shape = 964 if self.opt.dataset == ['reuters8'] else 48
        self.input_shape = {'TMN': 48, 'reuters8': 964, 'TREC6': 37, 'TREC50': 37, 'subj': 120, 'polarity': 59,
                            'imdb': 5, 'AGnews': 195}[self.opt.dataset[0]]

        self.task_doc_embeddings = {}
        self.cids_tids = set()
        self.logger = logger

        fs = 128
        cnn_shapes = [(3, self.opt.embedding_dim, fs), (4, self.opt.embedding_dim, fs), (5, self.opt.embedding_dim, fs)]

        if self.opt.base_architect == 0:
            raise NotImplementedError
        elif self.opt.base_architect == 1:
            raise NotImplementedError
        elif self.opt.base_architect == 2:
            if self.opt.concatenate_aw_kbs:
                raise NotImplementedError("concat_aw not valid for Kim et al approach!")
            self.shapes = cnn_shapes
            if self.opt.fedweit_dense:
                self.shapes += [(fs * self.opt.num_clients * 3, fs * 3)]
        elif self.opt.base_architect == 3:
            num_adapts = self.opt.et_top_k if self.opt.embedding_transfer else self.opt.num_clients
            self.shapes = cnn_shapes
            if self.opt.project_adaptives:
                if self.opt.concatenate_aw_kbs:
                    self.shapes += [(fs * num_adapts, fs)]
                else:
                    self.shapes += [(fs, fs)]
                self.shapes += [(fs * 2 * 3, fs * 3)]
            else:
                if self.opt.concatenate_aw_kbs:
                    self.shapes += [(fs * (num_adapts + 1) * 3, fs * 3)]
                else:
                    self.shapes += [(fs * 2 * 3, fs * 3)]

        self.conv_layers, self.fc_layers = get_conv_fc_layers(self.shapes)
        self.initialize_weights()

        print("Global init done")

    def initialize_weights(self):
        print("initialising weights")
        self.weights = []
        self.client_adapts = {}
        self.initializer = tf.keras.initializers.VarianceScaling(seed=self.opt.random_seed)
        for i in range(len(self.shapes)):
            if self.opt.dense_detached and i in self.fc_layers:
                self.weights.append(None)
                continue
            if self.opt.project_adaptives and i == self.fc_layers[0]:
                self.weights.append(None)
                continue
            weight = self.initializer(self.shapes[i])
            self.logger.info(f"Shared Weights init; layer {i}; mean: {tf.math.reduce_mean(tf.abs(weight)).numpy()} "
                             f"std: {tf.math.reduce_std(tf.abs(weight)).numpy()}")
            weight = weight.numpy().tolist()
            self.weights.append(weight)
        print(f"initialising weights done")

    def get_weights(self):
        print("#### Global: get_weights")
        return self.weights

    def get_adapts(self):
        print("#### Global: get_adapts")
        return self.client_adapts, self.task_doc_embeddings

    def set_weights(self, weights):
        print("#### Global: set_weights")
        self.weights = weights

    def set_adapts(self, client_adapts, cids_tids: list = None):
        print("#### Global: set_adapts")
        self.cids_tids.update(cids_tids)
        if not self.task_doc_embeddings:
            self.client_adapts = client_adapts
        else:
            for response_index, (cid, tid) in enumerate(cids_tids):
                if (cid, tid) not in self.task_doc_embeddings:
                    self.task_doc_embeddings[(cid, tid)] = self.task_doc_embeddings[(cid, tid)]
                self.client_adapts[(cid, tid)] = client_adapts[response_index]

    def get_current_task_number(self) -> int:
        tid = max([resp_tid for _, resp_tid in self.cids_tids]) if self.cids_tids else 0
        return tid

    def update_weights(self, responses):
        print("#### Global: update_weights")
        # embedding_transfer = True if 'vector_train' in responses[0] else False
        client_both = [pickle_string_to_obj(resp['client_both']) for resp in responses]
        # pdb.set_trace()

        client_w = [cb[0] for cb in client_both]
        client_a = [cb[1] for cb in client_both]
        client_sizes = [resp['train_size'] for resp in responses]
        client_masks = [resp['client_masks'] for resp in responses] if self.opt.sparse_comm else []

        fed_methods = {0: 'FedAvg', 1: 'FedProx'}
        self.logger.info(f"Applying global averaging: {fed_methods[self.opt.fed_method]}")
        if self.opt.fed_method == 0:
            self.apply_federated_average(client_w, client_sizes, client_masks)
        elif self.opt.fed_method == 1:
            self.apply_federated_prox(client_w, client_sizes, client_masks)

        if self.opt.server_sparse_comm:
            self.sparse_communication()
        self.calculate_comm_costs(self.get_weights())

        cids_tids = [(resp['client_id'], resp['task_id']) for resp in responses]
        self.set_adapts(client_a, cids_tids=cids_tids)

    def get_info(self):
        info = {
            'shapes': self.shapes,
            'input_shape': self.input_shape,
            'shared_params': self.weights
        }
        print(f"Getting info")
        return info
