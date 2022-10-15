import math
from copy import deepcopy
from os.path import join, abspath, dirname
from typing import List

from scipy.spatial.distance import cosine
import tensorflow as tf
import numpy as np

from FedWeit.models.fedweit.fedweit_layers import DecomposableConv, DecomposableDense
from FedWeit.models.fedweit.fedweit_plus_layers import DecomposableConvLocal, DecomposableConvAwAdd, DecomposableConvAwSimple, \
    DecomposableDenseSimple
from FedWeit.models.local_model import LocalModel
from FedWeit.models.utils import get_conv_fc_layers
from FedWeit.utils import syslog, load_weights

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class LocalFedWeIT(LocalModel):
    def __init__(self, client_id, data_info, opt, logger):
        super(LocalFedWeIT, self).__init__(client_id, data_info, opt, logger)
        self.opt = opt
        self.client_id = client_id
        self.data_info = data_info
        self.shared_aw = {}  # locally grouped adpative weights
        self.cent_list = []
        self.local_id = -1

        self.trainable_variables = []
        self.frozen_weights = dict()
        self.num_adapts = self.opt.et_top_k if self.opt.embedding_transfer else int(self.opt.num_clients * self.opt.frac_clients)

        self.logger = logger
        tf.random.set_seed(self.opt.random_seed)

    def initialize(self, model_info):
        # Gets called only on task 1 for each client
        self.logger.info("local initialize")
        self.models = []
        self.model_info = model_info
        self.maxlen = self.model_info['input_shape']
        self.shapes = self.model_info['shapes']
        self.conv_layers, self.fc_layers = get_conv_fc_layers(self.shapes)

        # logger.info(f"self.model_info['input_shape'] {self.model_info['input_shape']}")
        self.current_lr = self.opt.lr
        self.optimizer = self.get_optimizer(self.current_lr)

        # seed = self.opt.random_seed + self.client_id + self.current_task
        self.initializer = tf.keras.initializers.VarianceScaling(seed=self.opt.random_seed)
        # Different seed for adaptive, because global shared weights have seed random_seed too
        self.initializer_adaptive = tf.keras.initializers.VarianceScaling(seed=self.opt.random_seed + 42)
        self.variables = {
            'mask': {},
            'bias': {},
            'adaptive': {},
            'from_kb': {},
            'atten': {}
        }
        if self.opt.dense_detached or self.opt.project_adaptives:
            self.variables['fully_local'] = {}
        if self.opt.federated:  # initialized by global server,
            # Default
            self.variables['shared'] = []
            for i in range(len(self.shapes)):
                if self.opt.dense_detached and i in self.fc_layers:
                    self.variables['shared'].append(None)
                    continue
                if self.opt.project_adaptives and i == self.fc_layers[0]:
                    self.variables['shared'].append(None)
                    continue
                self.variables['shared'].append(tf.Variable(self.model_info['shared_params'][i], trainable=True,
                                                            name=f'global/layer_{i}/sw'))
        else:  # initialized locally
            # raise Exception("Check non self.federated settings")
            self.variables['shared'] = [
                tf.Variable(self.initializer(self.shapes[i]), trainable=True,
                            name='global/layer_{}/sw'.format(i)) for i in range(len(self.shapes))]
        self.build_model()
        # self.opt.load_weights default = False
        if self.opt.load_weights:
            weights = load_weights(self.opt.load_weights_dir)
            self.model.set_weights(weights)
        self.logger.info("local initialize done")

    def init_on_new_task(self):
        # Does not trigger on 1st task, but every task after that
        # logger.info("init_on_new_task")
        if self.opt.continual:
            # default
            self.build_model()
            self.recover_prev_theta()
        else:
            self.set_weights(self.model_info['shared_params'])
            self.build_model()
        self.current_lr = self.opt.lr
        self.optimizer = self.get_optimizer(self.current_lr)

    def l1_pruning(self, weights, hyp):
        hard_threshold = tf.cast(tf.greater(tf.abs(weights), hyp), tf.float32)
        return tf.multiply(weights, hard_threshold)

    def get_weights(self):
        self.logger.info("#### Local: get_weights")
        if self.opt.sparse_comm:
            hard_threshold = []
            sw_pruned = []
            masks = self.variables['mask'][self.current_task]
            for lid, sw in enumerate(self.variables['shared']):
                if self.opt.federated:
                    sw_pruned.append(sw.numpy())
                    hard_threshold.append(masks[lid][0])
                    continue

                mask = masks[lid][0]
                m_sorted = tf.sort(tf.keras.backend.flatten(tf.abs(mask)))
                thres = m_sorted[math.floor(len(m_sorted) * (self.opt.client_sparsity))]
                m_bianary = tf.cast(tf.greater(tf.abs(mask), thres), tf.float32).numpy().tolist()
                hard_threshold.append(m_bianary)
                # m_pruned = tf.multiply(mask, hard_threshold[-1]).numpy()
                sw_pruned.append(sw.numpy() * m_bianary)
            self.calculate_comm_costs(sw_pruned)
            return sw_pruned, hard_threshold
        else:
            # Default
            return [sw.numpy() for sw in self.variables['shared'] if sw is not None]

    def get_adapts(self):
        self.logger.info("#### Local: get_adapts")
        adps = []
        for adps_layers_index in self.variables['adaptive'][self.current_task]:
            adps += self.variables['adaptive'][self.current_task][adps_layers_index]
        params = [self.l1_pruning(adp, self.opt.l1_hyp).numpy() for adp in adps]
        return params

    def set_weights(self, new_weights):
        self.logger.info("#### Local: set_weights")
        # if self.opt.project_adaptives:
        #     final_conv_layer = self.conv_layers[-1]
        #     new_weights.insert(final_conv_layer + 1, None)
        for i, w in enumerate(new_weights):
            if w is None:
                continue
            if type(w) == np.ndarray:
                if len(w.shape) == 0:
                    continue

            sw = self.get_variable('shared', i)
            residuals = tf.cast(tf.equal(w, tf.zeros_like(w)), dtype=tf.float32)
            sw.assign(sw * residuals + w)

    def log_adapts_similarity_with_kb(self):
        for _l in range(len(self.shapes)):
            if self.opt.dense_detached and _l in self.fc_layers:
                continue
            local_adaptive = self.get_variable(var_type='adaptive', layer_idx=_l, task_idx=self.current_task)
            adp_kb = self.get_variable('from_kb', _l, self.current_task)

            cosines = []
            for adaptive_idx in range(adp_kb.shape[-1]):
                adp_kb_ = adp_kb[..., adaptive_idx]
                cosines.append(1 - cosine(tf.reshape(local_adaptive, [-1]), tf.reshape(adp_kb_, [-1])))
            # cosines = [ for adp_kb_ in adp_kb]
            cosines = [round(cosine_val, 4) for cosine_val in cosines]
            self.logger.info(f"Adaptive Similarity; Client {self.client_id} Round {self.current_round} Layer {_l}: {cosines}")

    def set_adapts(self, new_weights):
        self.logger.info("#### Local: set_adapts")
        if not self.opt.fedweit or self.opt.disable_alphas:
            return
        for _l in range(len(self.shapes)):
            if self.opt.dense_detached and _l in self.fc_layers:
                continue
            adp_kb = self.get_variable('from_kb', _l, self.current_task)
            new_w = np.zeros_like(adp_kb.numpy())
            if len(adp_kb.shape) == 5:
                for _c in range(len(new_weights)):
                    new_w[:, :, :, :, _c] = new_weights[_c][_l]
            else:
                # default
                for _c in range(len(new_weights)):
                    if len(new_weights[_c][_l].shape) == 2:
                        new_w[:, :, _c] = new_weights[_c][_l]
                    else:
                        new_w[:, :, :, _c] = new_weights[_c][_l]
            adp_kb.assign(new_w)

    def set_alphas(self, alphas: List[int]):
        """
        Adapts have been set already; now set alphas

        @param alphas:
        @return:
        """
        # alphas = [alpha / 10 for alpha in alphas]
        self.logger.info(f"Received{len(alphas)} alphas; "
                         f"{len(self.shapes) - len(alphas)} will be randomly initialised")
        for _l in range(len(self.shapes)):
            if self.opt.dense_detached and _l in self.fc_layers:
                continue
            alphas_layer = self.get_variable('atten', _l, self.current_task)
            alphas_layer_value = np.array(alphas + list(alphas_layer.numpy()[len(alphas):]))
            alphas_layer.assign(alphas_layer_value)
            self.logger.info(f"NEW Attentions of client {self.client_id}, task {self.current_task},"
                             f"layer {_l}:\n {alphas_layer_value}\n")

    def get_variable(self, var_type, layer_idx, task_idx=None, additional_index_bias: int = None):
        if var_type == 'shared':
            return self.variables[var_type][layer_idx]
        elif additional_index_bias:
            # This happens if model == 5 and using concatentation (only for bias!)
            return self.variables[var_type][task_idx][layer_idx][additional_index_bias]
        else:
            return self.variables[var_type][task_idx][layer_idx][0]

    def generate_mask(self, mask):
        return tf.keras.activations.sigmoid(mask)

    def get_optimizer(self, current_lr):
        # return tf.keras.optimizers.SGD(learning_rate=current_lr)
        return tf.keras.optimizers.Adam(learning_rate=current_lr)  # 0.001

    def build_model(self):
        self.logger.info(f"build_model with architecture: {self.opt.base_architect}")
        if self.opt.base_architect == 0:  # LeNet
            raise NotImplementedError
        elif self.opt.base_architect == 1:
            raise NotImplementedError
        elif self.opt.base_architect == 2:
            model = self.build_cnn_kim()
        elif self.opt.base_architect == 3:
                model = self.build_cnn_parallel()
        self.logger.info(f"Model Summary\n{model.summary()}")

        prev_variables = ['mask', 'bias']
        if self.opt.fedweit:
            prev_variables += ['adaptive']
            if (self.opt.embedding_transfer and self.opt.et_alphas_trainable) or not self.opt.embedding_transfer:
                prev_variables += ['atten']
        if self.opt.dense_detached:
            prev_variables += ['fully_local']
        if self.opt.foreign_adaptives_trainable:
            prev_variables += ['from_kb']
        self.trainable_variables = [sw for sw in self.variables['shared'] if sw is not None]

        for tid in range(self.current_task + 1):
            if not self.opt.continual:
                if tid < self.current_task:
                    continue
            for lid in range(len(self.shapes)):
                for pvar in prev_variables:
                    if pvar == 'bias' and tid < self.current_task:
                        continue
                    if 'atten' in pvar and (self.opt.embedding_transfer and not self.opt.et_alphas_trainable):
                        continue
                    if pvar in ['atten', 'from_kb'] and (tid == 0 or tid != self.current_task):
                        continue
                    if self.opt.dense_detached:
                        if lid in self.fc_layers and pvar not in ["fully_local", "bias"]:
                            continue
                        if lid in self.conv_layers and pvar == "fully_local":
                            continue
                    if self.opt.project_adaptives:
                        if lid == self.fc_layers[0] and pvar not in ["fully_local", "bias"]:
                            continue
                        if lid in self.conv_layers and pvar == "fully_local":
                            continue
                    if pvar == "bias":
                        self.trainable_variables += self.variables[pvar][tid][lid]
                    else:
                        self.trainable_variables.append(self.get_variable(pvar, lid, tid))
        # print(self.trainable_variables)
        # exit()
        self.models.append(model)
        self.logger.info("build_model done")

    def build_cnn_kim(self):
        # self.conv_layers, self.fc_layers = self.opt.shape["conv"], self.opt.shape["fc"]
        inputs = tf.keras.Input(self.maxlen, dtype=tf.int32)
        glove_load_filepath = abspath(join(dirname(dirname(dirname(dirname(__file__)))), "Resources",
                                           f"glove_{self.opt.embedding_dim}_{self.opt.dataset[0]}.npz"))
        glove = np.load(glove_load_filepath)['embeddings']
        embedding_layer = tf.keras.layers.Embedding(glove.shape[0], glove.shape[1], weights=[glove],
                                                    trainable=self.opt.trainable, input_length=self.maxlen)(inputs)
        pooled_outputs = []
        for i in self.conv_layers:
            pre_max_pool = DecomposableConv(
                name='layer_{}'.format(i), rank=1,
                filters=self.model_info['shapes'][i][-1], kernel_size=self.model_info['shapes'][i][0],
                # strides     = (1, self.model_info['shapes'][i][1]),
                fedweit=self.opt.fedweit, mask_bool=self.opt.mask,
                strides=1, dilation_rate=1, padding='valid', activation='relu',
                l1_hyp=self.opt.l1_hyp, mask_hyp=self.opt.mask_hyp,
                shared=self.variables['shared'][i], adaptive=self.create_variable('adaptive', i),
                from_kb=self.create_variable('from_kb', i), atten=self.create_variable('atten', i),
                bias=self.create_variable('bias', i), use_bias=True,
                mask=self.generate_mask(self.create_variable('mask', i)))(embedding_layer)
            # x = tf.nn.lrn(x, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
            conv_layer = tf.keras.layers.MaxPooling1D(pool_size=self.maxlen - self.model_info['shapes'][i][0] + 1,
                                                      strides=1, padding='valid')(pre_max_pool)  # valid same
            pooled_outputs.append(conv_layer)
        pooled_ = tf.concat(pooled_outputs, -1)
        dense_layer = tf.keras.layers.Flatten()(pooled_)
        dense_layer = tf.keras.layers.Dropout(self.opt.dropout)(dense_layer)

        for i in self.fc_layers:
            if not self.opt.fedweit:
                raise NotImplementedError("Dense Decomposable - Non fedweit not yet implemented")
            dense_layer = DecomposableDense(
                name='layer_{}'.format(i), units=self.model_info['shapes'][i][-1], input_dim=dense_layer.shape[-1],
                l1_hyp=self.opt.l1_hyp, mask_hyp=self.opt.mask_hyp,
                shared=self.variables['shared'][i], adaptive=self.create_variable('adaptive', i),
                from_kb=self.create_variable('from_kb', i), atten=self.create_variable('atten', i),
                bias=self.create_variable('bias', i), use_bias=True,
                mask=self.generate_mask(self.create_variable('mask', i)))(dense_layer)
            dense_layer = tf.keras.activations.relu(dense_layer)
        outputs = tf.keras.layers.Dense(self.num_classes[-1], activation='softmax',
                                        name='task_{}/head'.format(self.current_task))(dense_layer)
        return tf.keras.Model(inputs=inputs,
                              outputs=[inputs, embedding_layer, pre_max_pool, conv_layer, dense_layer, outputs])

    def build_cnn_parallel(self):
        inputs = tf.keras.Input(self.maxlen, dtype=tf.int32)
        glove_load_filepath = abspath(join(dirname(dirname(dirname(dirname(__file__)))), "Resources",
                                           f"glove_{self.opt.embedding_dim}_{self.opt.dataset[0]}.npz"))
        glove = np.load(glove_load_filepath)['embeddings']
        embedding_layer = tf.keras.layers.Embedding(glove.shape[0], glove.shape[1], weights=[glove],
                                                    trainable=self.opt.trainable, input_length=self.maxlen)(inputs)
        # aw_mode = "concat" if self.opt.concatenate_aw_kbs else "add"

        pooled_outputs, pooled_outputs_aw_kbs = [], []
        for layer_idx in self.conv_layers:
            pre_max_pool_local = DecomposableConvLocal(
                name=f'layer_l{layer_idx}', rank=1,
                filters=self.model_info['shapes'][layer_idx][-1], kernel_size=self.model_info['shapes'][layer_idx][0],
                strides=1, dilation_rate=1, padding='valid', activation='relu',
                l1_hyp=self.opt.l1_hyp, mask_hyp=self.opt.mask_hyp,
                shared=self.variables['shared'][layer_idx], adaptive=self.create_variable('adaptive', layer_idx),
                bias=self.create_variable('bias', layer_idx), use_bias=True,
                mask=self.generate_mask(self.create_variable('mask', layer_idx)))(embedding_layer)
            conv_layer_local = tf.keras.layers.MaxPooling1D(pool_size=self.maxlen - self.model_info['shapes'][layer_idx][0] + 1,
                                                            strides=1, padding='valid')(pre_max_pool_local)  # valid same

            additional_index = 1
            pooled_outputs.append(conv_layer_local)
            pooled_outputs_aw_kbs_this_cnn = []
            if self.opt.concatenate_aw_kbs:
                self.create_variable('from_kb', layer_idx)
                self.create_variable('atten', layer_idx)
                for adapt_index in range(self.num_adapts):
                    pre_max_pool_aw = DecomposableConvAwSimple(
                        name=f'layer_aw{layer_idx}_{adapt_index}', rank=1,
                        filters=self.model_info['shapes'][layer_idx][-1], kernel_size=self.model_info['shapes'][layer_idx][0],
                        strides=1, dilation_rate=1, padding='valid', l1_hyp=self.opt.l1_hyp,
                        from_kb=self.get_variable('from_kb', layer_idx, self.current_task)[..., adapt_index],
                        atten=self.get_variable('atten', layer_idx, self.current_task)[..., adapt_index],
                        bias=self.create_variable('bias', layer_idx, additional_index=additional_index), use_bias=True)(embedding_layer)
                    # ToDo Valid only for ReLu
                    additional_index += 1
                    conv_layer_aw = tf.keras.layers.MaxPooling1D(pool_size=self.maxlen - self.model_info['shapes'][layer_idx][0] + 1,
                                                                 strides=1, padding='valid')(pre_max_pool_aw)
                    # conv_layer_aw = tf.keras.activations.relu(conv_layer_aw)
                    pooled_outputs_aw_kbs_this_cnn.append(conv_layer_aw)
            else:
                pre_max_pool_aw = DecomposableConvAwAdd(
                    name=f'layer_aw{layer_idx}', rank=1,
                    filters=self.model_info['shapes'][layer_idx][-1], kernel_size=self.model_info['shapes'][layer_idx][0],
                    strides=1, dilation_rate=1, padding='valid',
                    l1_hyp=self.opt.l1_hyp, from_kb=self.create_variable('from_kb', layer_idx), atten=self.create_variable('atten', layer_idx),
                    bias=self.create_variable('bias', layer_idx, additional_index=additional_index), use_bias=True)(embedding_layer)
                additional_index += 1
                conv_layer_aw = tf.keras.layers.MaxPooling1D(pool_size=self.maxlen - self.model_info['shapes'][layer_idx][0] + 1,
                                                             strides=1, padding='valid')(pre_max_pool_aw)  # valid same
                # conv_layer_aw = tf.keras.activations.relu(conv_layer_aw)
                pooled_outputs_aw_kbs_this_cnn.append(conv_layer_aw)
            if self.opt.project_adaptives:
                # ToDo
                #     1. Correct shape load in create_variable
                # 2. check trainable_variables
                # 3. check model summary
                # 4. Look at debug flow for projective
                pooled_outputs_aw_kbs_this_cnn = tf.concat(pooled_outputs_aw_kbs_this_cnn, -1)
                dense_layer_projection = tf.keras.layers.Flatten()(pooled_outputs_aw_kbs_this_cnn)

                layer_idx_projection = self.fc_layers[0]
                dense_layer_projection = DecomposableDenseSimple(
                    name=f'layer_{layer_idx_projection}_{layer_idx}', units=self.model_info['shapes'][layer_idx_projection][-1],
                    input_dim=dense_layer_projection.shape[-1], l1_hyp=self.opt.l1_hyp, mask_hyp=self.opt.mask_hyp,
                    fully_local=self.create_variable('fully_local', layer_idx_projection, additional_index=layer_idx),
                    bias=self.create_variable('bias', layer_idx_projection, additional_index=layer_idx),
                    use_bias=True)(dense_layer_projection)
                dense_layer_projection = tf.keras.activations.relu(dense_layer_projection)
                # pooled_outputs += [dense_layer_projection]
                pooled_outputs_aw_kbs.append(dense_layer_projection)
            else:
                pooled_outputs_aw_kbs_this_cnn = [tf.keras.activations.relu(layer) for layer in pooled_outputs_aw_kbs_this_cnn]
                # pooled_outputs += pooled_outputs_aw_kbs_this_cnn
                pooled_outputs_aw_kbs += pooled_outputs_aw_kbs_this_cnn
            del pooled_outputs_aw_kbs_this_cnn

        # each layer is by default (None, 1 , 128)
        # print(pooled_outputs)
        pooled_ = tf.concat(pooled_outputs, -1)
        pooled_ = tf.keras.layers.Flatten()(pooled_)
        pooled_aw_kbs_ = tf.concat(pooled_outputs_aw_kbs, -1)
        pooled_aw_kbs_ = tf.keras.layers.Flatten()(pooled_aw_kbs_)

        dense_layer = tf.concat([pooled_, pooled_aw_kbs_], -1)
        dense_layer = tf.keras.layers.Flatten()(dense_layer)
        dense_layer = tf.keras.layers.Dropout(self.opt.dropout)(dense_layer)

        for layer_idx in self.fc_layers:
            if self.opt.project_adaptives and layer_idx == self.fc_layers[0]:
                continue
            if not self.opt.fedweit:
                raise NotImplementedError("Dense Decomposable - Non fedweit not yet implemented")
            if self.opt.dense_detached:
                dense_layer = DecomposableDenseSimple(
                    name=f'layer_{layer_idx}', units=self.model_info['shapes'][layer_idx][-1], input_dim=dense_layer.shape[-1],
                    l1_hyp=self.opt.l1_hyp, mask_hyp=self.opt.mask_hyp,
                    fully_local=self.create_variable('fully_local', layer_idx),
                    bias=self.create_variable('bias', layer_idx), use_bias=True)(dense_layer)
            else:
                dense_layer = DecomposableDense(
                    name=f'layer_{layer_idx}', units=self.model_info['shapes'][layer_idx][-1], input_dim=dense_layer.shape[-1],
                    l1_hyp=self.opt.l1_hyp, mask_hyp=self.opt.mask_hyp,
                    shared=self.variables['shared'][layer_idx], adaptive=self.create_variable('adaptive', layer_idx),
                    from_kb=self.create_variable('from_kb', layer_idx), atten=self.create_variable('atten', layer_idx),
                    bias=self.create_variable('bias', layer_idx), use_bias=True,
                    mask=self.generate_mask(self.create_variable('mask', layer_idx)))(dense_layer)
            dense_layer = tf.keras.activations.relu(dense_layer)
        outputs = tf.keras.layers.Dense(self.num_classes[-1], activation='softmax',
                                        name='task_{}/head'.format(self.current_task))(dense_layer)
        return tf.keras.Model(inputs=inputs,
                              outputs=[inputs, embedding_layer, (pre_max_pool_local, pre_max_pool_aw),
                                       (conv_layer_local, conv_layer_aw), dense_layer, outputs])

    def create_variable(self, var_type, layer_index, additional_index: int = 0):
        if self.current_task not in self.variables[var_type]:
            self.variables[var_type][self.current_task] = {}
        if layer_index not in self.variables[var_type][self.current_task]:
            self.variables[var_type][self.current_task][layer_index] = []

        if var_type == 'adaptive':
            # derived from shared here; is same as shared when first round starts?
            trainable = False if self.opt.disable_adaptive else True
            if self.opt.disable_adaptive:
                init_value = np.zeros(self.model_info['shapes'][layer_index], dtype=np.float32)
            elif self.opt.adaptive_random:
                init_value = self.initializer_adaptive((self.model_info['shapes'][layer_index]))
            else:
                init_value = self.variables['shared'][layer_index].numpy() / 5
        elif var_type == 'fully_local':
            # while len(self.variables[var_type][self.current_task]) < layer_index:
            #     self.variables[var_type][self.current_task].append(None)
            trainable = True
            init_value = self.initializer(self.model_info['shapes'][layer_index])

        elif var_type == 'from_kb':
            if self.current_task > 0:
                trainable = True if self.opt.foreign_adaptives_trainable else False
            else:
                trainable = False
            _shape = list(self.model_info['shapes'][layer_index])

            _shape.append(self.num_adapts)
            _shape = tuple(_shape)
            init_value = np.zeros(_shape, dtype=np.float32)
        elif var_type == 'atten':
            print('@@@@@@@@@@@@@@ %d x %d' % (self.num_adapts, self.current_task))
            if self.current_task > 0:
                trainable = True
            else:
                trainable = False
            if self.opt.disable_alphas:
                init_value = np.zeros(self.num_adapts, dtype=np.float32)
                trainable = False
            else:
                init_value = self.initializer((self.num_adapts,))

            # self.logger.info(f"Attention of client {self.client_id}, task {self.current_task}, layer {i}:\n {init_value}\n")
        elif var_type == 'bias':
            # if shape=(3,300,128) then this has 128 length vector
            trainable = True
            init_value = self.initializer((self.model_info['shapes'][layer_index][-1],))

        elif var_type == 'mask':
            # if shape=(3,300,128) then this has 128 length vector
            trainable = True
            init_value = self.initializer((self.model_info['shapes'][layer_index][-1],))
        else:
            raise ValueError(f"Invalid var_type {var_type}")

        # if additional_index:
        #     layer_index = f"{layer_index}_{additional_index}"
        var_name = f'task_{self.current_task}/layer_{layer_index}_{additional_index}/{var_type}'
        if var_type in ['from_kb', 'atten'] and not self.opt.fedweit:
            var = None
        else:
            var = tf.Variable(init_value, trainable=trainable, name=var_name)
        self.logger.info(f"{var_type} of layer {layer_index}; trainable: {trainable}")
        self.logger.info(f'\tmean: {tf.math.reduce_mean(tf.abs(init_value)).numpy()}, '
                         f'std:{tf.math.reduce_std(tf.abs(init_value)).numpy()}')
        self.variables[var_type][self.current_task][layer_index].append(var)
        if len(self.variables[var_type][self.current_task][layer_index]) != additional_index + 1:
            raise BufferError(f"{var_type}_{self.current_task}_{layer_index} has "
                              f"{len(self.variables[var_type][self.current_task][layer_index])} but additonal_index is "
                              f"{additional_index}")

        if var_type == 'from_kb' and trainable:
            if 'from_kb' not in self.frozen_weights:
                self.frozen_weights['from_kb'] = dict()
            self.frozen_weights['from_kb'][(self.current_task, layer_index)] = tf.Variable(deepcopy(
                init_value), trainable=False, name="frozen_" + var_name)
        return var

    def recover_prev_theta(self):
        # before shared params are updated by federated weights from server.
        self.theta = {}
        for i in range(len(self.model_info['shapes'])):
            if self.opt.dense_detached and i in self.fc_layers:
                continue
            self.theta[i] = {}
            sw = self.get_variable(var_type='shared', layer_idx=i)
            for j in range(self.current_task):
                pmask = self.get_variable(var_type='mask', layer_idx=i, task_idx=j)
                g_pmask = self.generate_mask(pmask)
                if self.opt.disable_adaptive:
                    #################################################
                    # recover the masked version of the sw at this layer and task
                    theta_t = sw * g_pmask
                    self.theta[i][j] = theta_t.numpy()
                #################################################
                else:
                    paw = self.get_variable(var_type='adaptive', layer_idx=i, task_idx=j)
                    if self.opt.is_hierarchy and j < self.local_id:
                        group_info = self.assign_list[i][j]
                        is_single_group = np.sum(np.equal(self.assign_list[i], group_info)) == 1
                        if not is_single_group:
                            local_shared = self.shared_aw[self.local_id]  # trainable false
                            full_paw = local_shared + paw
                    elif j >= self.local_id:
                        is_single_group = True
                    if is_single_group:
                        full_paw = paw
                    #################################################
                    theta_t = sw * g_pmask + full_paw
                    self.theta[i][j] = theta_t.numpy()
                #################################################

    def loss(self, model, y_true, y_pred):
        weight_decay = 0
        # Weight_decay (l2) = mask + adaptive local + Shared weights
        sparseness, approx_loss, sparseness_mask = 0, 0, 0
        from_kb_drift_loss = 0
        loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        pure_loss = deepcopy(loss)
        for layer_idx in self.conv_layers:
            if self.opt.foreign_adaptives_trainable and self.current_task > 0:
                from_kb = self.get_variable(var_type='from_kb', layer_idx=layer_idx, task_idx=self.current_task)
                frozen_from_kb_weights = self.frozen_weights['from_kb'][(self.current_task, layer_idx)]
                from_kb_drift_loss += self.opt.foreign_adaptives_trainable_drift_constant * \
                                      tf.nn.l2_loss(from_kb - frozen_from_kb_weights)
                # if tf.reduce_sum(from_kb - frozen_from_kb_weights) != 0:
                #     a=0
                # if layer_idx == 0 and self.current_task > 0:
                #     b=0

            sw = self.get_variable(var_type='shared', layer_idx=layer_idx)
            if self.opt.mask:  # Default
                mask = self.get_variable(var_type='mask', layer_idx=layer_idx, task_idx=self.current_task)
                g_mask = self.generate_mask(mask)
                weight_decay += self.opt.wd_rate * tf.nn.l2_loss(mask)
                if not self.opt.disable_sparsity:  # Default
                    if self.opt.sparse_comm:
                        sparseness_mask += self.opt.l1_mask_hyp * tf.reduce_sum(tf.abs(mask))
                    else:  # Default
                        sparseness += self.opt.mask_hyp * tf.reduce_sum(tf.abs(mask))
            if not self.opt.disable_adaptive:  # Default
                aw = self.get_variable(var_type='adaptive', layer_idx=layer_idx, task_idx=self.current_task)
                weight_decay += self.opt.wd_rate * tf.nn.l2_loss(aw)
                if not self.opt.disable_sparsity:  # Default
                    sparseness += self.opt.l1_hyp * tf.reduce_sum(tf.abs(aw))
            if self.current_task == 0 or not self.opt.continual:  # Default
                weight_decay += self.opt.wd_rate * tf.nn.l2_loss(sw)
            else:  # Default
                for j in range(self.current_task):
                    pmask = self.get_variable(var_type='mask', layer_idx=layer_idx, task_idx=j)
                    g_pmask = self.generate_mask(pmask)
                    # if self.opt.disable_adaptive:
                    #     #################################################
                    #     theta_t = sw * g_pmask
                    #     a_l2 = tf.nn.l2_loss(theta_t - self.theta[i][j])
                    #     approx_loss += self.opt.approx_hyp * a_l2
                    # #################################################
                    # else:
                    paw = self.get_variable(var_type='adaptive', layer_idx=layer_idx, task_idx=j)
                    if self.opt.is_hierarchy and j < self.local_id:
                        group_info = self.assign_list[layer_idx][j]
                        is_single_group = np.sum(np.equal(self.assign_list[layer_idx], group_info)) == 1
                        if not is_single_group:
                            local_shared = self.shared_aw[self.local_id]  # trainable false
                            full_paw = local_shared + paw
                    elif j >= self.local_id:  # = -1 for some reason          # Default
                        is_single_group = True
                    if is_single_group:  # Default
                        full_paw = paw
                    #################################################
                    if self.opt.mask:  # Default
                        theta_t = sw * g_pmask + full_paw
                    else:
                        theta_t = sw + full_paw
                    a_l2 = tf.nn.l2_loss(theta_t - self.theta[layer_idx][j])
                    approx_loss += self.opt.approx_hyp * a_l2
                    #################################################
                    if not self.opt.disable_sparsity:  # Default
                        sparseness += self.opt.l1_hyp * tf.reduce_sum(tf.abs(paw))

        for layer_idx in self.fc_layers:
            if self.opt.dense_detached:
                fl = self.get_variable(var_type='fully_local', layer_idx=layer_idx, task_idx=self.current_task)
                weight_decay += self.opt.wd_rate * tf.nn.l2_loss(fl)
                if not self.opt.disable_sparsity:
                    sparseness += self.opt.l1_hyp * tf.reduce_sum(tf.abs(fl))
                continue
            if self.opt.foreign_adaptives_trainable and self.current_task > 0:
                from_kb = self.get_variable(var_type='from_kb', layer_idx=layer_idx, task_idx=self.current_task)
                frozen_from_kb_weights = self.frozen_weights['from_kb'][(self.current_task, layer_idx)]
                from_kb_drift_loss += self.opt.foreign_adaptives_trainable_drift_constant * \
                                      tf.nn.l2_loss(from_kb - frozen_from_kb_weights)

            sw = self.get_variable(var_type='shared', layer_idx=layer_idx)
            if self.opt.mask:
                mask = self.get_variable(var_type='mask', layer_idx=layer_idx, task_idx=self.current_task)
                weight_decay += self.opt.wd_rate * tf.nn.l2_loss(mask)
                if not self.opt.disable_sparsity:
                    if self.opt.sparse_comm:
                        sparseness_mask += self.opt.l1_mask_hyp * tf.reduce_sum(tf.abs(mask))
                    else:
                        sparseness += self.opt.mask_hyp * tf.reduce_sum(tf.abs(mask))
            if not self.opt.disable_adaptive:
                aw = self.get_variable(var_type='adaptive', layer_idx=layer_idx, task_idx=self.current_task)
                weight_decay += self.opt.wd_rate * tf.nn.l2_loss(aw)
                if not self.opt.disable_sparsity:
                    sparseness += self.opt.l1_hyp * tf.reduce_sum(tf.abs(aw))
            if self.current_task == 0 or not self.opt.continual:
                weight_decay += self.opt.wd_rate * tf.nn.l2_loss(sw)
            else:
                # approx_loss += self.opt.approx_hyp * tf.nn.l2_loss(sw-sw.numpy())
                for j in range(self.current_task):
                    pmask = self.get_variable(var_type='mask', layer_idx=layer_idx, task_idx=j)
                    g_pmask = self.generate_mask(pmask)
                    gaw_list = []
                    # if self.opt.disable_adaptive:
                    #     #################################################
                    #     theta_t = sw * g_pmask
                    #     a_l2 = tf.nn.l2_loss(theta_t - self.theta[i][j])
                    #     approx_loss += self.opt.approx_hyp * a_l2
                    # #################################################
                    # else:
                    paw = self.get_variable(var_type='adaptive', layer_idx=layer_idx, task_idx=j)
                    if self.opt.is_hierarchy and j < self.local_id:
                        group_info = self.assign_list[layer_idx][j]
                        is_single_group = np.sum(np.equal(self.assign_list[layer_idx], group_info)) == 1
                        if not is_single_group:
                            local_shared = self.shared_aw[self.local_id]  # trainable false
                            full_paw = local_shared + paw
                        # local_shared_list.append(local_shared)
                    elif self.opt.is_hierarchy and j >= self.local_id:
                        is_single_group = True
                    if is_single_group:
                        full_paw = paw
                    #################################################
                    if self.opt.mask:
                        theta_t = sw * g_pmask + full_paw
                    else:
                        theta_t = sw + full_paw
                    a_l2 = tf.nn.l2_loss(theta_t - self.theta[layer_idx][j])
                    approx_loss += self.opt.approx_hyp * a_l2
                    #################################################
                    if not self.opt.disable_sparsity:
                        sparseness += self.opt.l1_hyp * tf.reduce_sum(tf.abs(paw))
        loss += weight_decay + sparseness + approx_loss + sparseness_mask + from_kb_drift_loss
        return loss, (pure_loss, approx_loss, weight_decay, sparseness, sparseness_mask)

    def calculate_capacity(self):
        num_active_params = 0
        num_base_params = 0
        for dims in self.model_info['shapes']:
            params = 1
            for d in dims:
                params *= d
            num_base_params += params
        for nc in self.num_classes:
            num_base_params += self.model_info['shapes'][-1][-1] * nc

        top_most = self.models[-1].get_layer('task_{}/head'.format(self.current_task))
        top_most_kernel = top_most.kernel
        top_most_bias = top_most.bias

        var_list = self.trainable_variables.copy()
        var_list += [top_most_kernel, top_most_bias]
        for var in var_list:
            # print('var_name: %s'%var.name)
            if 'adaptive' in var.name:
                if not self.opt.disable_sparsity:
                    var = self.l1_pruning(var, self.opt.l1_hyp)
            actives = tf.not_equal(var, tf.zeros_like(var))
            actives = tf.reduce_sum(tf.cast(actives, tf.float32))
            num_active_params += actives.numpy()

        self.mem_ratio.append(num_active_params / num_base_params)
        syslog(self.client_id, 'memory capacity ratio: %.3f' % (num_active_params / num_base_params), self.logger)

    def calculate_comm_costs(self, sw_pruned):
        num_base_params = 0
        for i, sw in enumerate(self.variables['shared']):
            params = 1
            for d in sw.shape:
                params *= d
            num_base_params += params
        # print('sw_{}: {}'.format(i, params))
        # print('num_base_params:', num_base_params)

        num_active_params = 0
        for i, pruned in enumerate(sw_pruned):
            actives = tf.not_equal(pruned, tf.zeros_like(pruned))
            actives = tf.reduce_sum(tf.cast(actives, tf.float32))
            num_active_params += actives.numpy()
        # print('pruned_sw_{}: {}'.format(i, actives.numpy()))

        self.comm_ratio.append(num_active_params / num_base_params)
        syslog(self.client_id, 'communication cost ratio: %.3f' % (num_active_params / num_base_params), self.logger)
