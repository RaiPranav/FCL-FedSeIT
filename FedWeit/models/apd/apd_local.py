import math

from FedWeit.utils import *
from FedWeit.models.local_model import LocalModel
from FedWeit.models.apd.apd_layers import *


class LocalAPD(LocalModel):

    def __init__(self, client_id, data_info, opt):
        super(LocalAPD, self).__init__(client_id, data_info, opt)
        self.opt = opt
        self.client_id = client_id
        self.data_info = data_info
        self.shared_aw = {}  # locally grouped adpative weights
        self.cent_list = []
        self.local_id = -1

    def initialize(self, model_info):
        self.models = []
        self.model_info = model_info
        self.current_lr = self.opt.lr
        self.optimizer = self.get_optimizer(self.current_lr)
        self.initializer = tf.keras.initializers.VarianceScaling()
        self.variables = {
            'mask': {},
            'bias': {},
            'adaptive': {},
            'local': {}}
        if self.opt.federated:
            self.variables['shared'] = [tf.Variable(self.model_info['shared_params'][i],  # initialized by global server,
                                                    trainable=True, name='global/layer_{}/sw'.format(i))
                                        for i in range(len(self.model_info['shapes']))]
        else:
            self.variables['shared'] = [tf.Variable(self.initializer(self.model_info['shapes'][i]),  # initialized locally
                                                    trainable=True, name='global/layer_{}/sw'.format(i))
                                        for i in range(len(self.model_info['shapes']))]
        self.build_model()
        if self.opt.load_weights:
            weights = load_weights(self.opt.load_weights_dir)
            self.model.set_weights(weights)

    def init_on_new_task(self):
        if self.opt.continual:
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
        if self.opt.sparse_comm:
            # masks = [self.generate_mask(mask) for mask in self.variables['mask'][self.current_task]]
            # for m in masks:
            #     print('mean:', tf.math.reduce_mean(tf.abs(m)).numpy(), 'std:', tf.math.reduce_std(tf.abs(m)).numpy())
            #     print('================================================')
            hard_threshold = []
            sw_pruned = []
            masks = self.variables['mask'][self.current_task]
            for lid, sw in enumerate(self.variables['shared']):
                mask = masks[lid]
                m_sorted = tf.sort(tf.keras.backend.flatten(tf.abs(mask)))
                thres = m_sorted[math.floor(len(m_sorted) * (1 - self.opt.sparse_comm_rate))]
                m_bianary = tf.cast(tf.greater(tf.abs(mask), thres), tf.float32).numpy().tolist()
                hard_threshold.append(m_bianary)
                # m_pruned = tf.multiply(mask, hard_threshold[-1]).numpy()
                sw_pruned.append(sw.numpy() * m_bianary)
            self.calculate_comm_costs(sw_pruned)
            return sw_pruned, hard_threshold
        else:
            return [sw.numpy() for sw in self.variables['shared']]

    def set_weights(self, new_weights):
        for i, w in enumerate(new_weights):
            sw = self.get_variable('shared', i)
            if self.opt.sparse_comm:
                residuals = tf.cast(tf.equal(w, tf.zeros_like(w)), dtype=tf.float32)
                sw.assign(sw * residuals + w)
            else:
                sw.assign(w)

    def get_variable(self, var_type, layer_idx, task_idx=None):
        if var_type == 'shared':
            return self.variables[var_type][layer_idx]
        else:
            return self.variables[var_type][task_idx][layer_idx]

    def generate_mask(self, mask):
        return tf.keras.activations.sigmoid(mask)

    def get_optimizer(self, current_lr):
        return tf.keras.optimizers.SGD(learning_rate=current_lr)
        # return tf.keras.optimizers.Adam(learning_rate=current_lr) # 0.001

    def build_model(self):
        if self.opt.base_architect == 0:  # LeNet
            model = self.build_modified_LeNet()
        elif self.opt.base_architect == 1:
            model = self.build_modified_AlexNet()
        prev_variables = ['mask', 'bias'] if self.opt.disable_adaptive else ['mask', 'bias', 'adaptive']
        self.trainable_variables = [sw for sw in self.variables['shared']]
        for tid in range(self.current_task + 1):
            if not self.opt.continual:
                if tid < self.current_task:
                    continue
            for lid in range(len(self.model_info['shapes'])):
                for pvar in prev_variables:
                    if pvar == 'bias' and tid < self.current_task:
                        continue
                    self.trainable_variables.append(self.get_variable(pvar, lid, tid))
        # model.summary()
        # print('self.trainable_variables:', [v.name for v in self.trainable_variables])
        # print('-----------------------------------------------------------------')
        self.models.append(model)

    def build_modified_LeNet(self):
        self.conv_layers = [0, 1]
        self.fc_layers = [2, 3]
        inputs = x = tf.keras.Input(self.model_info['input_shape'])
        for i in self.conv_layers:
            x = DecomposableConv(
                name='layer_{}'.format(i),
                filters=self.model_info['shapes'][i][-1],
                kernel_size=(5, 5),
                strides=(1, 1),
                padding='same',
                activation='relu',
                l1_hyp=self.opt.l1_hyp,
                mask_hyp=self.opt.mask_hyp,
                shared=self.variables['shared'][i],
                adaptive=self.create_variable('adaptive', i),
                bias=self.create_variable('bias', i), use_bias=True,
                mask=self.generate_mask(self.create_variable('mask', i)))(x)
            x = tf.nn.lrn(x, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
            x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        x = tf.keras.layers.Flatten()(x)
        for i in self.fc_layers:
            x = DecomposableDense(
                name='layer_{}'.format(i),
                units=self.model_info['shapes'][i][-1],
                input_dim=x.shape[-1],
                l1_hyp=self.opt.l1_hyp,
                mask_hyp=self.opt.mask_hyp,
                shared=self.variables['shared'][i],
                adaptive=self.create_variable('adaptive', i),
                bias=self.create_variable('bias', i), use_bias=True,
                mask=self.generate_mask(self.create_variable('mask', i)))(x)
            x = tf.keras.activations.relu(x)
        x = tf.keras.layers.Dense(self.num_classes[-1], activation='softmax',
                                  name='task_{}/head'.format(self.current_task))(x)
        return tf.keras.Model(inputs=inputs, outputs=x)

    def build_modified_AlexNet(self):
        self.conv_layers = [0, 1, 2]
        self.fc_layers = [3, 4]
        inputs = x = tf.keras.Input(self.model_info['input_shape'])
        for i in self.conv_layers:
            x = DecomposableConv(
                name='layer_{}'.format(i),
                filters=self.model_info['shapes'][i][-1],
                kernel_size=(self.model_info['shapes'][i][0], self.model_info['shapes'][i][1]),
                strides=(1, 1),
                padding='same',
                activation='relu',
                l1_hyp=self.opt.l1_hyp,
                mask_hyp=self.opt.mask_hyp,
                shared=self.variables['shared'][i],
                adaptive=self.create_variable('adaptive', i),
                bias=self.create_variable('bias', i), use_bias=True,
                mask=self.generate_mask(self.create_variable('mask', i)))(x)
            if i < 2:
                x = tf.keras.layers.Dropout(0.2)(x)
            else:
                x = tf.keras.layers.Dropout(0.5)(x)
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Flatten()(x)
        for i in self.fc_layers:
            x = DecomposableDense(
                name='layer_{}'.format(i),
                units=self.model_info['shapes'][i][-1],
                input_dim=self.model_info['shapes'][i][0],
                activation='relu',
                l1_hyp=self.opt.l1_hyp,
                mask_hyp=self.opt.mask_hyp,
                shared=self.variables['shared'][i],
                adaptive=self.create_variable('adaptive', i),
                bias=self.create_variable('bias', i), use_bias=True,
                mask=self.generate_mask(self.create_variable('mask', i)))(x)
            x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(self.num_classes[-1], activation='softmax',
                                  name='task_{}/head'.format(self.current_task))(x)
        return tf.keras.Model(inputs=inputs, outputs=x)

    def create_variable(self, var_type, i):
        if self.current_task not in self.variables[var_type]:
            self.variables[var_type][self.current_task] = []
        if var_type == 'adaptive':
            trainable = False if self.opt.disable_adaptive else True
            init_value = np.zeros(self.model_info['shapes'][i], dtype=np.float32) \
                if self.opt.disable_adaptive else self.variables['shared'][i].numpy() / 2
        else:
            trainable = True
            init_value = self.initializer((self.model_info['shapes'][i][-1],))
        var = tf.Variable(
            init_value,
            trainable=trainable,
            name='task_{}/layer_{}/{}'.format(self.current_task, i, var_type))
        self.variables[var_type][self.current_task].append(var)
        return var

    def recover_prev_theta(self):
        # before shared params are updated by federated weights from server.
        self.theta = {}
        for i in range(len(self.model_info['shapes'])):
            self.theta[i] = {}
            sw = self.get_variable(var_type='shared', layer_idx=i)
            for j in range(self.current_task):
                pmask = self.get_variable(var_type='mask', layer_idx=i, task_idx=j)
                g_pmask = self.generate_mask(pmask)
                if self.opt.disable_adaptive:
                    #################################################
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
        sparseness, approx_loss, sparseness_mask = 0, 0, 0
        loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        for i in self.conv_layers:
            sw = self.get_variable(var_type='shared', layer_idx=i)
            mask = self.get_variable(var_type='mask', layer_idx=i, task_idx=self.current_task)
            g_mask = self.generate_mask(mask)
            weight_decay += self.opt.wd_rate * tf.nn.l2_loss(mask)
            if not self.opt.disable_sparsity:
                if self.opt.sparse_comm:
                    sparseness_mask += self.opt.l1_mask_hyp * tf.reduce_sum(tf.abs(mask))
                else:
                    sparseness += self.opt.mask_hyp * tf.reduce_sum(tf.abs(mask))
            if not self.opt.disable_adaptive:
                aw = self.get_variable(var_type='adaptive', layer_idx=i, task_idx=self.current_task)
                weight_decay += self.opt.wd_rate * tf.nn.l2_loss(aw)
                if not self.opt.disable_sparsity:
                    sparseness += self.opt.l1_hyp * tf.reduce_sum(tf.abs(aw))
            if self.current_task == 0 or not self.opt.continual:
                weight_decay += self.opt.wd_rate * tf.nn.l2_loss(sw)
            else:
                for j in range(self.current_task):
                    pmask = self.get_variable(var_type='mask', layer_idx=i, task_idx=j)
                    g_pmask = self.generate_mask(pmask)
                    if self.opt.disable_adaptive:
                        #################################################
                        theta_t = sw * g_pmask
                        a_l2 = tf.nn.l2_loss(theta_t - self.theta[i][j])
                        approx_loss += self.opt.approx_hyp * a_l2
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
                        a_l2 = tf.nn.l2_loss(theta_t - self.theta[i][j])
                        approx_loss += self.opt.approx_hyp * a_l2
                        #################################################
                        if not self.opt.disable_sparsity:
                            sparseness += self.opt.l1_hyp * tf.reduce_sum(tf.abs(paw))
        for i in self.fc_layers:
            sw = self.get_variable(var_type='shared', layer_idx=i)
            mask = self.get_variable(var_type='mask', layer_idx=i, task_idx=self.current_task)
            weight_decay += self.opt.wd_rate * tf.nn.l2_loss(mask)
            if not self.opt.disable_sparsity:
                if self.opt.sparse_comm:
                    sparseness_mask += self.opt.l1_mask_hyp * tf.reduce_sum(tf.abs(mask))
                else:
                    sparseness += self.opt.mask_hyp * tf.reduce_sum(tf.abs(mask))
            if not self.opt.disable_adaptive:
                aw = self.get_variable(var_type='adaptive', layer_idx=i, task_idx=self.current_task)
                weight_decay += self.opt.wd_rate * tf.nn.l2_loss(aw)
                if not self.opt.disable_sparsity:
                    sparseness += self.opt.l1_hyp * tf.reduce_sum(tf.abs(aw))
            if self.current_task == 0 or not self.opt.continual:
                weight_decay += self.opt.wd_rate * tf.nn.l2_loss(sw)
            else:
                for j in range(self.current_task):
                    pmask = self.get_variable(var_type='mask', layer_idx=i, task_idx=j)
                    g_pmask = self.generate_mask(pmask)
                    gaw_list = []
                    if self.opt.disable_adaptive:
                        #################################################
                        theta_t = sw * g_pmask
                        a_l2 = tf.nn.l2_loss(theta_t - self.theta[i][j])
                        approx_loss += self.opt.approx_hyp * a_l2
                        #################################################
                    else:
                        paw = self.get_variable(var_type='adaptive', layer_idx=i, task_idx=j)
                        if self.opt.is_hierarchy and j < self.local_id:
                            group_info = self.assign_list[i][j]
                            is_single_group = np.sum(np.equal(self.assign_list[i], group_info)) == 1
                            if not is_single_group:
                                local_shared = self.shared_aw[self.local_id]  # trainable false
                                full_paw = local_shared + paw
                                local_shared_list.append(local_shared)
                        elif self.opt.is_hierarchy and j >= self.local_id:
                            is_single_group = True
                        if is_single_group:
                            full_paw = paw
                        #################################################
                        theta_t = sw * g_pmask + full_paw
                        a_l2 = tf.nn.l2_loss(theta_t - self.theta[i][j])
                        approx_loss += self.opt.approx_hyp * a_l2
                        #################################################
                        if not self.opt.disable_sparsity:
                            sparseness += self.opt.l1_hyp * tf.reduce_sum(tf.abs(paw))

        loss += weight_decay + sparseness + approx_loss + sparseness_mask
        return loss

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
            if 'adaptive' in var.name:
                if not self.opt.disable_sparsity:
                    var = self.l1_pruning(var, self.opt.l1_hyp)
            actives = tf.not_equal(var, tf.zeros_like(var))
            actives = tf.reduce_sum(tf.cast(actives, tf.float32))
            num_active_params += actives.numpy()

        self.mem_ratio.append(num_active_params / num_base_params)
        syslog(self.client_id, 'memory capacity ratio: %.3f' % (num_active_params / num_base_params))

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
        syslog(self.client_id, 'communication cost ratio: %.3f' % (num_active_params / num_base_params))

    # def k_means_clustering(self, prv_cent=None, is_decomposed=False):
    #     assign_list = []
    #     get_cents = []
    #     _k = int(self.opt.k_centroides * (self.current_task+1) / self.opt.clustering_iter)

    #     if hasattr(self, 'assign_list'):
    #         is_decomposed = True

        # for i in range(4):
        #     full_aw = []
        #     only_aw = []
        #     for tid in range(self.current_task+1):
        #         get_aw = self.get_variable('adaptive', i, tid)
        #         only_aw.append(get_aw)
        #
        #         if is_decomposed and len(self.assign_list[0]) > tid:
        #             local_id = self.assign_list[i][tid]
        #             # NOTE not single group
        #             if np.sum(local_id == np.array(self.assign_list[i])) > 1:
        #                 local_shared = self.get_variable() ('id%d_local%d'%(len(self.assign_list[0]), local_id),
        #                                                     'aw_group/layer%d'%i, False, reuse=True)
        #                 full_aw.append(get_aw + local_shared)
        #                 print('layer%d,  local%d + task%d_aw'%(i, local_id, tid))
    #                 else:
    #                     full_aw.append(get_aw)
    #                     print('layer%d,  single_group%d_task%d_aw'%(i, local_id, tid))
    #             else:
    #                 full_aw.append(get_aw)
    #                 print('layer%d,  single_task%d_aw'%(i, tid))

    #         rs_aws = tf.random_shuffle(full_aw)
    #         _slice = rs_aws.get_shape().as_list()
    #         _slice[0] = _k
    #         if prv_cent:
    #             #start from prv centroides
    #             slc = tf.slice(rs_aws, np.zeros_like(_slice), _slice)
    #             slc = self.sess.run(slc)
    #             n_prv_cents = prv_cent[-1][i].shape[0]
    #             slc[:n_prv_cents] = prv_cent[-1][i]
    #             centroides = tf.Variable(slc)
    #         else:
    #             centroides = tf.Variable(tf.slice(rs_aws, np.zeros_like(_slice), _slice))

    #         expanded_vectors = tf.expand_dims(full_aw, 0)
    #         expanded_centroides = tf.expand_dims(centroides, 1)

            # dims = np.arange(len(full_aw[0].shape)) + 2
            # assignments = tf.argmin(tf.reduce_sum(tf.square(tf.subtract(expanded_vectors, expanded_centroides)), dims), 0)
            # means = tf.concat([tf.reduce_mean(tf.gather(full_aw, tf.reshape(tf.where( tf.equal(assignments, _c)),[1,-1])),
            #                                   reduction_indices=[1]) for _c in range(_k)], 0)

    #         update_centroides = tf.assign(centroides, means)
    #         self.variable_initialization()
    #         for step in range(100):
    #            _, centroid_values, assignment_values = self.sess.run([update_centroides, centroides, assignments])

    #         print(' [*] k-means clustering of layer %d : %s'%(i, assignment_values.tolist()))
    #         self.aw_consolidation(_k, i, np.array(full_aw), np.array(only_aw), assignment_values)
    #         assign_list.append(assignment_values.tolist())
    #         get_cents.append(centroid_values)
    #     self.assign_list = assign_list
    #     return get_cents

    # def aw_consolidation(self, _k, layer_id, full_aws, only_aws, cluster_info):
    #     op_list = []
    #     for _c in range(_k):
    #         is_group = cluster_info == _c
    #         group_aws = full_aws[is_group]

    #         if np.sum(is_group) == 1:
    #             print(' [*] #%d single group'%_c)
    #             full_single_aw = group_aws[0]
    #             only_single_aw = only_aws[is_group][0]

    #             if full_single_aw.name != only_single_aw.name:
    #                 print(' [*] Single but different! %s <=> %s'%(full_single_aw.name, only_single_aw.name))
    #                 op_list.append(only_single_aw.assign(full_single_aw))

    #         else:
    #             # NOTE
    #             # Current measure is median. try to mean!
    #             e_max = tf.reduce_max(group_aws.tolist(), 0)
    #             e_min = tf.reduce_min(group_aws.tolist(), 0)

    #             e_gap = e_max-e_min
    #             e_nind = tf.cast(tf.greater(e_gap, self.c.e_gap_hyp), tf.float32)
    #             e_ind = tf.abs(1-e_nind)

    #             var_type = 'gaw'
    #             new_aw = tf.Variable(
    #                             tf.reduce_mean([e_max, e_min], 0) * e_ind, 
    #                             trainable=trainable,  
    #                             name='local_{}/layer_{}/{}'.format(_c, layer_id, var_type))
    #             self.variables[var_type][_c].append(var)

    #             local_capacity = tf.reduce_sum(tf.cast(tf.not_equal(e_ind, tf.zeros_like(e_ind)), tf.int32))
    #             print(' [*] #%d local_shared_elements: %d/%d'%(_c, local_capacity, np.prod(new_aw.get_shape().as_list())))

    #             for _f_aws, _o_aws in zip(group_aws.tolist(), only_aws[is_group].tolist()):
    #                 op_list.append(_o_aws.assign(_f_aws * e_nind))
