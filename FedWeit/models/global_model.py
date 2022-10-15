import tensorflow as tf

from FedWeit.utils import *


class GlobalModel(object):
    def __init__(self, opt, logger):
        self.opt = opt
        self.logger = logger
        self.comm_ratio = []

    def initialize_weights(self):
        raise NotImplementedError()

    def get_weights(self):
        raise NotImplementedError()

    def set_weights(self, weights):
        raise NotImplementedError()

    def update_weights(self, responses):
        raise NotImplementedError()

    def get_info(self):
        raise NotImplementedError()

    def apply_federated_average(self, client_weights, client_sizes, client_masks=[]):
        new_weights = [np.zeros_like(w) for w in self.get_weights()]
        if self.opt.sparse_comm:
            epsi = 1e-15
            client_masks = tf.ragged.constant(client_masks, dtype=tf.float32)
            client_sizes = [tf.math.multiply(m, client_sizes[i]) for i, m in enumerate(client_masks)]
            total_sizes = epsi
            for _cs in client_sizes:
                total_sizes += _cs
            for c_idx, c_weights in enumerate(client_weights):  # by client
                # c_weights = pickle_string_to_obj(c_weights)
                for lidx, l_weights in enumerate(c_weights):  # by layer
                    ratio = client_sizes[c_idx][lidx] / total_sizes[lidx]
                    if len(new_weights[lidx].shape) == 0:
                        continue
                    new_weights[lidx] += tf.math.multiply(l_weights, ratio).numpy()
                # print('mean_weights_layer_%d: %.8f'%(lidx, np.mean(np.abs(new_weights[lidx]))))
        else:
            total_size = np.sum(client_sizes)
            for c in range(len(client_weights)):  # by client
                # _client_weights = pickle_string_to_obj(client_weights[c])
                _client_weights = client_weights[c]
                for i in range(len(new_weights)):  # by layer
                    if len(new_weights[i].shape) == 0:
                        continue
                    new_weights[i] += _client_weights[i] * float(client_sizes[c] / total_size)
        self.set_weights(new_weights)

    def apply_federated_prox(self, client_weights, client_sizes, client_masks=[]):
        new_weights = [np.zeros_like(w) for w in self.get_weights()]
        if self.opt.sparse_comm:
            epsi = 1e-15
            client_masks = tf.ragged.constant(client_masks, dtype=tf.float32)
            total_sizes = epsi
            for _cs in client_masks:
                total_sizes += _cs
            for c_idx, c_weights in enumerate(client_weights):  # by client
                # c_weights = pickle_string_to_obj(c_weights)
                for lidx, l_weights in enumerate(c_weights):  # by layer
                    if len(new_weights[lidx].shape) == 0:
                        continue
                    ratio = 1 / total_sizes[lidx]
                    new_weights[lidx] += tf.math.multiply(l_weights, ratio).numpy()
        else:
            total_size = len(client_sizes)
            for c in range(len(client_weights)):  # by client
                # _client_weights = pickle_string_to_obj(client_weights[c])
                _client_weights = client_weights[c]
                for i in range(len(new_weights)):  # by layer
                    if len(new_weights[i].shape) == 0:
                        continue
                    new_weights[i] += _client_weights[i] * float(1 / total_size)
        self.set_weights(new_weights)

    def sparse_communication(self):
        ww = self.get_weights()
        new_weights = [np.zeros_like(w) for w in ww]
        for tid, theta_g in enumerate(ww):
            g_sort = np.sort(np.abs(theta_g), axis=None)
            thr_g = g_sort[-int((1 - self.opt.sparse_broad_rate) * len(g_sort))]
            selected = np.where(np.abs(theta_g) >= thr_g, theta_g, np.zeros(theta_g.shape))
            new_weights[tid] = selected

        self.set_weights(new_weights)

    def calculate_comm_costs(self, new_weights):
        # ToDo Optimisation?: new_weights always comes via self.get_weights(), so identical to current_weights
        current_weights = self.get_weights()
        num_base_params = 0
        for i, weights in enumerate(current_weights):
            if len(weights.shape) == 0:
                continue
            params = 1
            for d in np.shape(weights):
                params *= d
            num_base_params += params
        #     print('sw_{}: {}'.format(i, params))
        # print('num_base_params:', num_base_params)
        num_active_params = 0
        for i, nw in enumerate(new_weights):
            if len(nw.shape) == 0:
                continue
            actives = tf.not_equal(nw, tf.zeros_like(nw))
            actives = tf.reduce_sum(tf.cast(actives, tf.float32))
            num_active_params += actives.numpy()
        # print('pruned_sw_{}: {}'.format(i, actives.numpy()))
        self.comm_ratio.append(num_active_params / num_base_params)
        # when opt.sparse_communication in parser.py is True, then output < 1. Else 1, it seems
        syslog(-2, 'communication cost ratio: %.3f' % (num_active_params / num_base_params), self.logger)

    def write_current_status(self):
        write_file(self.opt.log_dir, 'server.txt', {
            'comm_ratio': self.comm_ratio,
        })
