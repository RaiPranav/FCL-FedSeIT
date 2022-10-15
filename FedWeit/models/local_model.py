from typing import List
import numpy as np
from os.path import join
from copy import deepcopy

import tensorflow_addons as tfa
import tensorflow as tf

from FedWeit.models.utils import process_confusion_matrix, plot_learning_curve
from FedWeit.utils import write_file, syslog
from FedWeit.utils import np_save


class LocalModel(object):
    def __init__(self, client_id, data_info, opt, logger):

        self.models = None
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if len(gpus) > 0:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 2.5)])

        self.opt = opt
        self.client_id = client_id
        self.data_info = data_info

        self.logger = logger
        self.temp_vars = {}

        # gpus = tf.config.list_physical_devices('GPU')
        # self.logger.info(f"\n GPUs detected: {gpus}")
        # if gpus:
        #     # Create 2 virtual GPUs with 1GB memory each
        #     try:
        #         tf.config.set_logical_device_configuration(
        #             gpus[0],
        #             [tf.config.LogicalDeviceConfiguration(memory_limit=102400),
        #              tf.config.LogicalDeviceConfiguration(memory_limit=102400)])
        #         logical_gpus = tf.config.list_logical_devices('GPU')
        #         self.logger.info(f"{len(gpus)}, Physical GPU, {len(logical_gpus)} Logical GPUs")
        #     except RuntimeError as e:
        #         # Virtual devices must be set before GPUs have been initialized
        #         self.logger.info(e)

        self.log_dir = ''
        self.tasks = []
        self.mem_ratio = []
        self.comm_ratio = []
        self.num_train_list = []
        self.num_test_list = []
        self.num_valid_list = []
        self.x_test_list = []
        self.y_test_list = []

        self.performance_epoch = {}
        self.performance_watch_acc = {}
        self.performance_watch_f1_macro = {}
        self.loss_watch = {}
        self.classwise_acc_watch = {}
        self.classwise_f1_watch = {}
        self.pure_loss, self.approx_loss, self.weight_decay_loss, self.sparse_loss, self.sparse_mask_loss = \
            {}, {}, {}, {}, {}
        for metric_dict in [self.performance_watch_acc, self.performance_watch_f1_macro, self.loss_watch,
                            self.classwise_acc_watch, self.classwise_f1_watch,
                            self.pure_loss, self.approx_loss, self.weight_decay_loss, self.sparse_loss, self.sparse_mask_loss]:
            for split_name in ['train', 'valid', 'test']:
                metric_dict[split_name] = {task_id: dict() for task_id in range(self.opt.num_pick_tasks)}

        self.performance_final = {}
        self.performance_final_f1_macro = {}
        self.num_classes = []

        self.options = {}
        self.early_stop = False
        self.test_batch_step = 10
        self.filename = 'client-' + str(self.client_id) + '.json'
        self.metrics = {
            'valid_lss': tf.keras.metrics.Mean(name='valid_lss'),
            'train_lss': tf.keras.metrics.Mean(name='train_lss'),
            'test_lss': tf.keras.metrics.Mean(name='test_lss'),
            'valid_acc': tf.keras.metrics.CategoricalAccuracy(name='valid_acc'),
            'train_acc': tf.keras.metrics.CategoricalAccuracy(name='train_acc'),
            'test_acc': tf.keras.metrics.CategoricalAccuracy(name='test_acc'),
            'valid_f1_macro': tfa.metrics.F1Score(num_classes=self.opt.num_classes, average="macro",
                                                  name="valid_f1_macro"),
            'train_f1_macro': tfa.metrics.F1Score(num_classes=self.opt.num_classes, average="macro",
                                                  name="train_f1_macro"),
            'test_f1_macro': tfa.metrics.F1Score(num_classes=self.opt.num_classes, average="macro",
                                                 name="test_f1_macro"),
            # 'valid_prec': tfa.metrics.F1Score(num_classes=self.opt.num_classes, average="macro",
            #                                       name="valid_prec"),
            # 'train_prec': tfa.metrics.F1Score(num_classes=self.opt.num_classes, average="macro",
            #                                       name="train_prec"),
            # 'test_prec': tfa.metrics.F1Score(num_classes=self.opt.num_classes, average="macro",
            #                                      name="test_prec"),
            # 'valid_rec': tfa.metrics.F1Score(num_classes=self.opt.num_classes, average="macro",
            #                                   name="valid_rec"),
            # 'train_rec': tfa.metrics.F1Score(num_classes=self.opt.num_classes, average="macro",
            #                                   name="train_rec"),
            # 'test_rec': tfa.metrics.F1Score(num_classes=self.opt.num_classes, average="macro",
            #                                  name="test_rec"),
            'valid_cm': tfa.metrics.MultiLabelConfusionMatrix(num_classes=self.opt.num_classes, name='valid_cm'),
            'train_cm': tfa.metrics.MultiLabelConfusionMatrix(num_classes=self.opt.num_classes, name='train_cm'),
            'test_cm': tfa.metrics.MultiLabelConfusionMatrix(num_classes=self.opt.num_classes, name='test_cm'),
            'valid_pure_lss': tf.keras.metrics.Mean(name='valid_pure_lss'),
            'train_pure_lss': tf.keras.metrics.Mean(name='train_pure_lss'),
            'test_pure_lss': tf.keras.metrics.Mean(name='test_pure_lss'),
            'valid_approx_lss': tf.keras.metrics.Mean(name='valid_approx_lss'),
            'train_approx_lss': tf.keras.metrics.Mean(name='train_approx_lss'),
            'test_approx_lss': tf.keras.metrics.Mean(name='test_approx_lss'),
            'valid_weight_decay_lss': tf.keras.metrics.Mean(name='valid_weight_decay_lss'),
            'train_weight_decay_lss': tf.keras.metrics.Mean(name='train_weight_decay_lss'),
            'test_weight_decay_lss': tf.keras.metrics.Mean(name='test_weight_decay_lss'),
            'valid_sparse_lss': tf.keras.metrics.Mean(name='valid_sparse_lss'),
            'train_sparse_lss': tf.keras.metrics.Mean(name='train_sparse_lss'),
            'test_sparse_lss': tf.keras.metrics.Mean(name='test_sparse_lss'),
            'valid_sparse_mask_lss': tf.keras.metrics.Mean(name='valid_sparse_mask_lss'),
            'train_sparse_mask_lss': tf.keras.metrics.Mean(name='train_sparse_mask_lss'),
            'test_sparse_mask_lss': tf.keras.metrics.Mean(name='test_sparse_mask_lss'),
        }

    def loss(self, model, y_true, y_pred):
        # must be implemented in the child class
        raise NotImplementedError()

    def get_optimizer(self, lr):
        # must be implemented in the child class
        raise NotImplementedError()

    def get_weights(self):
        # must be implemented in the child class
        raise NotImplementedError()

    def set_weights(self):
        # must be implemented in the child class
        raise NotImplementedError()

    def initialize(self, model_info):
        # must be implemented in the child class
        raise NotImplementedError()

    def build_model(self, model_info):
        # must be implemented in the child class
        raise NotImplementedError()

    def init_on_new_task(self):
        # must be implemented in the child class
        raise NotImplementedError()

    def log_batch_performance(self, split, batch, loss, y_batch, y_pred):
        precision_decimal = 4
        metric = tf.keras.metrics.Mean()(loss)
        result = round(float(metric), precision_decimal)

        separator = "\t"
        self.logger.info(f"Client_{self.client_id}_{split}\tBatch_{batch}{separator}loss:{result}")
        for metric_name, metric in [("acc", tf.keras.metrics.CategoricalAccuracy()),
                                    ("f1_macro",
                                     tfa.metrics.F1Score(num_classes=self.opt.num_classes, average="macro"))]:
            result = round(float(metric(y_batch, y_pred).numpy()), precision_decimal)
            self.logger.info(f"Client_{self.client_id}_{split}\tBatch_{batch}{separator}{metric_name}:{result}")
            metric.reset_states()
        cm = tfa.metrics.MultiLabelConfusionMatrix(num_classes=self.opt.num_classes)
        # print(y_batch, y_pred)
        cm = cm(y_batch, tf.one_hot(tf.math.argmax(y_pred, axis=1), depth=y_pred.shape[1])).numpy()
        classwise_acc, classwise_f1 = process_confusion_matrix(cm)

        self.logger.info(f"Client_{self.client_id}_{split}\tBatch_{batch}{separator}Classwise_acc:{classwise_acc}")
        self.logger.info(f"Client_{self.client_id}_{split}\tBatch_{batch}{separator}Classwise_f1:{classwise_f1}")

        separator += "\t"
        vlss, vpure_lss, [vacc, vf1_macro, cm] = self.validate(self.current_task)
        for metric_name, result in [("loss", vlss), ("pure_loss", vpure_lss), ("acc", vacc),
                                    ("f1_macro", vf1_macro)]:
            result = round(result, precision_decimal)
            self.logger.info(f"Client_{self.client_id}_valid\tBatch_{batch}{separator}{metric_name}:{result}")
        cm = cm.numpy()
        classwise_acc, classwise_f1 = process_confusion_matrix(cm)
        self.logger.info(f"Client_{self.client_id}_valid\tBatch_{batch}{separator}Classwise_acc:{classwise_acc}")
        self.logger.info(f"Client_{self.client_id}_valid\tBatch_{batch}{separator}Classwise_f1:{classwise_f1}")

    def train_step(self, model, x, y):
        tf.keras.backend.set_learning_phase(1)
        with tf.GradientTape() as tape:
            outputs_all = model(x)
            y_pred = outputs_all[-1]

            loss, losses = self.loss(model, y, y_pred)
            pure_loss = losses[0]
        gradients = tape.gradient(loss, self.trainable_variables)
        if self.opt.model == 1:  # APD
            gradients *= 10  # from the original source
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return y_pred, pure_loss, loss

    def train_one_round(self, current_round, count_rounds, is_last=False):
        """

        @param current_round: 0 to num_tasks_per_client x max_rounds_per_task - 1
        @param count_rounds: 1 to max_rounds_per_task. Reset to 1 for every new task
        @param is_last: is last round or not
        @return:
        """
        self.current_round = current_round
        self.count_rounds = count_rounds
        model = self.models[self.current_task]

        # np_save(self.opt.log_dir, f'x_train.npy', self.x_train)
        self.log_adapts_similarity_with_kb()
        np_save(self.opt.log_dir, f'pre-weights-{self.client_id}.npy', self.get_weights())
        for w in self.get_weights():
            w = tf.abs(tf.Variable(w))
            self.logger.info(f'Logging sw mean: {tf.math.reduce_mean(w).numpy()}, std:{tf.math.reduce_std(w).numpy()}')

        nun_epochs = self.opt.num_epochs
        if self.opt.converge_first_round_epochs and self.count_rounds == 1:
            self.logger.info("First round under 'converge-first' conditions starting. setting lr patience")
            nun_epochs = self.opt.converge_first_round_epochs
            self.temp_vars["lr_patience"] = self.opt.lr_patience
            self.opt.current_lr_patience = self.opt.lr_patience = self.opt.converge_first_round_epochs * 2

            for _l in range(len(self.model_info['shapes'])):
                adp_kb = self.get_variable('from_kb', _l, self.current_task)
                # 3/4/5, 300, 128, 3
                self.temp_vars[f"from_kb_{_l}"] = deepcopy(adp_kb)
                new_aw_kb_this_round = tf.zeros_like(adp_kb)
                adp_kb.assign(new_aw_kb_this_round)

        if not self.opt.old_early_stopping_enabled:
            # default
            self.lowest_lss = np.inf
            self.current_lr_patience = self.opt.lr_patience
            if self.opt.reset_current_lr:
                # Resetting current learning rate every round. Not default.
                self.current_lr = self.opt.lr
                self.optimizer = self.get_optimizer(self.current_lr)

        for epoch in range(nun_epochs):
            self.current_epoch = epoch
            self.current_batch = 0
            for i in range(0, len(self.x_train), self.opt.batch_size):  # train
                self.current_batch += 1
                if 0 < self.opt.num_examples <= i + self.opt.batch_size:
                    x_batch = self.x_train[i: self.opt.num_examples]
                    y_batch = self.y_train[i: self.opt.num_examples]
                    y_pred, pure_loss, loss = self.train_step(model, x_batch, y_batch)
                    # self.log_batch_performance("train", self.current_batch, loss, y_batch, y_pred)
                    break
                else:
                    x_batch = self.x_train[i: i + self.opt.batch_size]
                    y_batch = self.y_train[i: i + self.opt.batch_size]
                    y_pred, pure_loss, loss = self.train_step(model, x_batch, y_batch)
                # self.log_batch_performance("train", self.current_batch, loss, y_batch, y_pred)
            model = self.get_model(self.current_task)

            # train eval and validation eval
            y_pred_train, loss_train, other_losses_train = self.predict_labels(self.x_train, self.y_train, model)
            train_lss, train_acc, _, other_losses_train = self.measure_split_performance("train", self.y_train,
                                                                                         y_pred_train, loss_train,
                                                                                         other_losses_train)
            y_pred_valid, loss_valid, other_losses_valid = self.predict_labels(self.x_valid, self.y_valid, model)
            vlss, vacc, _, other_losses_valid = self.measure_split_performance("valid", self.y_valid, y_pred_valid,
                                                                               loss_valid, other_losses_valid)

            message = f"[client:{self.client_id}] task:{self.current_task} ({self.tasks[self.current_task]}), " \
                      f"round:{self.current_round} (cnt:{self.count_rounds}), epoch:{self.current_epoch}, "
            message_train = message + f"train_lss:{round(train_lss, 4)}, train_acc:{round(train_acc, 4)}" \
                                      f" ({', '.join(map(str, [round(other_loss, 4) for other_loss in other_losses_train]))})" \
                                      f" (task_{self.current_task})"
            message_valid = message + f"valid_lss:{round(vlss, 4)}, valid_acc:{round(vacc, 4)} " \
                                      f"({', '.join(map(str, [round(other_loss, 4) for other_loss in other_losses_valid]))})" \
                                      f" (task_{self.current_task})"
            self.logger.info(message_train)
            self.logger.info(message_valid)

            # Epoch-level evalation for test
            is_last_epoch = (epoch == nun_epochs - 1)
            self.evaluate(is_last=(is_last and is_last_epoch))

            # Adapt lr
            if vlss < self.lowest_lss:
                self.lowest_lss = vlss
                self.current_lr_patience = self.opt.lr_patience
            else:
                self.current_lr_patience -= 1
                if self.current_lr_patience <= 0:
                    self.current_lr /= self.opt.lr_factor
                    syslog(self.client_id, 'task:%d, round:%d (cnt:%d), drop lr => %.10f'
                           % (self.current_task, self.current_round, self.count_rounds, self.current_lr), self.logger)
                    if self.current_lr < self.opt.lr_min:
                        syslog(self.client_id, 'task:%d, round:%d (cnt:%d), early stop, reached minium lr (%.10f)'
                               % (self.current_task, self.current_round, self.count_rounds, self.current_lr),
                               self.logger)
                        self.early_stop = True
                        break
                    self.current_lr_patience = self.opt.lr_patience
                    self.optimizer = self.get_optimizer(self.current_lr)

        if self.opt.converge_first_round_epochs and self.count_rounds == 1:
            self.logger.info("First round under 'converge-first' conditions finished. Resetting lr")
            self.opt.current_lr_patience = self.opt.lr_patience = self.temp_vars["lr_patience"]
            self.lowest_lss = np.inf
            self.current_lr = self.opt.lr
            self.optimizer = self.get_optimizer(self.current_lr)
            self.early_stop = False

            for _l in range(len(self.model_info['shapes'])):
                adp_kb = self.get_variable('from_kb', _l, self.current_task)
                adp_kb.assign(self.temp_vars[f"from_kb_{_l}"])
                del self.temp_vars[f"from_kb_{_l}"]

        np_save(self.opt.log_dir, f'post-weights-{self.client_id}.npy', self.get_weights())
        for w in self.get_weights():
            w = tf.abs(tf.Variable(w))
            self.logger.info(f'POST\tLogging sw mean: {tf.math.reduce_mean(w).numpy()}, std:{tf.math.reduce_std(w).numpy()}')

    def measure_split_performance(self, split_name, y_true, y_pred_all, loss_total, other_losses, tid=None):
        tid = self.current_task if type(tid) != int else tid
        losses = [f'{split_name}_lss', f'{split_name}_pure_lss', f'{split_name}_approx_lss', f'{split_name}_weight_decay_lss',
                  f'{split_name}_sparse_lss', f'{split_name}_sparse_mask_lss']
        other_metrics = [f'{split_name}_acc', f'{split_name}_f1_macro', f'{split_name}_cm']

        self.add_performance_to_tensor(losses, other_metrics, loss_total, other_losses, y_true[: len(y_pred_all)],
                                       y_pred_all)
        loss, (pure_lss, approx_lss, weight_decay_lss, sparse_lss, sparse_lss_mask), [acc, f1_macro, cm] = \
            self.measure_performance(losses, other_metrics)
        classwise_acc, classwise_f1 = process_confusion_matrix(cm)

        # print(split_name, tid, self.current_task, self.current_round)
        # print(self.loss_watch[split_name][tid])
        # print(self.performance_watch_acc)
        for metric_dict in [self.performance_watch_acc, self.performance_watch_f1_macro, self.loss_watch,
                            self.classwise_acc_watch, self.classwise_f1_watch,
                            self.pure_loss, self.approx_loss, self.weight_decay_loss, self.sparse_loss, self.sparse_mask_loss]:
            if self.current_round not in metric_dict[split_name][tid]:
                metric_dict[split_name][tid][self.current_round] = []

        self.loss_watch[split_name][tid][self.current_round].append(loss)
        self.performance_watch_acc[split_name][tid][self.current_round].append(acc)
        self.performance_watch_f1_macro[split_name][tid][self.current_round].append(f1_macro)
        self.classwise_acc_watch[split_name][tid][self.current_round].append(str(classwise_acc))
        self.classwise_f1_watch[split_name][tid][self.current_round].append(str(classwise_f1))

        self.pure_loss[split_name][tid][self.current_round].append(pure_lss)
        self.approx_loss[split_name][tid][self.current_round].append(approx_lss)
        self.weight_decay_loss[split_name][tid][self.current_round].append(weight_decay_lss)
        self.sparse_loss[split_name][tid][self.current_round].append(sparse_lss)
        self.sparse_mask_loss[split_name][tid][self.current_round].append(sparse_lss_mask)
        # print(self.performance_watch_acc, "\n\n")
        return loss, acc, f1_macro, (pure_lss, approx_lss, weight_decay_lss, sparse_lss, sparse_lss_mask)

    def predict_labels(self, x, y, model):
        y_pred_all, loss_total = None, None
        pure_loss_total, approx_loss_total, weight_decay_total, sparseness_total, sparseness_mask_total = \
            None, None, None, None, None
        tf.keras.backend.set_learning_phase(0)

        for i in range(0, len(self.x_valid), self.opt.batch_size):
            x_batch = x[i: i + self.opt.batch_size]
            y_batch = y[i: i + self.opt.batch_size]
            y_pred_batch = model(x_batch)[-1]
            loss, losses = self.loss(model, y_batch, y_pred_batch)

            pure_loss_total = tf.concat((pure_loss_total, losses[0]), axis=0) if tf.is_tensor(
                pure_loss_total) else losses[0]
            approx_loss_total = tf.concat((approx_loss_total, [losses[1]]), axis=0) if tf.is_tensor(
                approx_loss_total) else [losses[1]]
            weight_decay_total = tf.concat((weight_decay_total, [losses[2]]), axis=0) if tf.is_tensor(
                weight_decay_total) else [losses[2]]
            sparseness_total = tf.concat((sparseness_total, [losses[3]]), axis=0) if tf.is_tensor(
                sparseness_total) else [losses[3]]
            sparseness_mask_total = tf.concat((sparseness_mask_total, [losses[4]]), axis=0) if tf.is_tensor(
                sparseness_mask_total) else [losses[4]]
            loss_total = tf.concat((loss_total, loss), axis=0) if tf.is_tensor(loss_total) else loss
            y_pred_all = tf.concat((y_pred_all, y_pred_batch), axis=0) if tf.is_tensor(y_pred_all) else y_pred_batch
        return y_pred_all, loss_total, (pure_loss_total, approx_loss_total, weight_decay_total, sparseness_total,
                                        sparseness_mask_total)

    def evaluate(self, is_last=False):
        for tid in range(len(self.x_test_list)):
            # if tid == self.current_task or tid in self.opt.watch_tasks or is_last:
            # lss, pure_lss, [acc, f1_macro, cm] = self._evaluate(self.x_test_list[tid], self.y_test_list[tid], head_idx=tid)
            model = self.get_model(tid)
            y_pred_test, loss_test, other_losses_test = self.predict_labels(self.x_test_list[tid], self.y_test_list[tid], model)
            # self.logger.info(f"Sending evaluate cid {self.client_id} tid: {tid}")
            lss, acc, f1_macro, other_losses = self.measure_split_performance("test", self.y_test_list[tid], y_pred_test,
                                                                              loss_test, other_losses_test, tid=tid)

            if tid == self.current_task:
                if tid not in self.performance_epoch:
                    self.performance_epoch[tid] = []  # initialize
                self.performance_epoch[tid].append(acc)
            if is_last:
                if tid not in self.performance_final:
                    self.performance_final[tid] = []  # initialize
                self.performance_final_f1_macro[tid] = []
                self.performance_final[tid].append(acc)
                self.performance_final_f1_macro[tid].append(f1_macro)

            # other_losses = [round(other_loss, 4) for other_loss in other_losses]
            message = f"[client:{self.client_id}] task:{self.current_task} ({self.tasks[tid]}), " \
                      f"round:{self.current_round} (cnt:{self.count_rounds}), epoch:{self.current_epoch}, " \
                      f"test_lss:{round(lss, 4)}, test_acc:{round(acc, 4)}"
            message += f" ({', '.join(map(str, [round(other_loss, 4) for other_loss in other_losses]))}) (task_{tid})"
            self.logger.info(message)

    def add_performance_to_tensor(self, losses, metrics_names: List[str], loss, other_losses, y_true, y_pred):
        lss_name, pure_lss_name, approx_lss_name, weight_decay_lss_name, sparse_lss_name, sparse_mask_lss_name = losses
        self.metrics[lss_name](loss)

        pure_loss, approx_loss, weight_decay, sparseness, sparseness_mask = other_losses
        self.metrics[pure_lss_name](pure_loss)
        self.metrics[approx_lss_name](approx_loss)
        self.metrics[weight_decay_lss_name](weight_decay)
        self.metrics[sparse_lss_name](sparseness)
        self.metrics[sparse_mask_lss_name](sparseness_mask)

        for metric_name in metrics_names:
            if "_cm" in metric_name:
                y_pred_discrete = tf.one_hot(tf.math.argmax(y_pred, axis=1), depth=y_pred.shape[1], dtype=np.int32)
                self.metrics[metric_name](y_true, y_pred_discrete)
            else:
                self.metrics[metric_name](y_true, y_pred)

    def measure_performance(self, losses, metric_names: List[str]):
        lss_name, pure_lss_name, approx_lss_name, weight_decay_lss_name, sparse_lss_name, sparse_mask_lss_name = losses
        lss = float(self.metrics[lss_name].result())  # tensor to float
        self.metrics[lss_name].reset_states()

        pure_lss = float(self.metrics[pure_lss_name].result())  # tensor to float
        approx_lss = float(self.metrics[approx_lss_name].result())  # tensor to float
        weight_decay_lss = float(self.metrics[weight_decay_lss_name].result())  # tensor to float
        sparse_lss = float(self.metrics[sparse_lss_name].result())  # tensor to float
        sparse_lss_mask = float(self.metrics[sparse_mask_lss_name].result())  # tensor to float
        self.metrics[pure_lss_name].reset_states()
        self.metrics[approx_lss_name].reset_states()
        self.metrics[weight_decay_lss_name].reset_states()
        self.metrics[sparse_lss_name].reset_states()
        self.metrics[sparse_mask_lss_name].reset_states()

        metrics = []
        for metric_name in metric_names:
            if "_cm" in metric_name:
                metric = self.metrics[metric_name].result()
            else:
                metric = float(self.metrics[metric_name].result())  # tensor to float
            self.metrics[metric_name].reset_states()
            metrics.append(metric)
        return lss, (pure_lss, approx_lss, weight_decay_lss, sparse_lss, sparse_lss_mask), metrics

    def set_task(self, task_id, data):
        train = data['train']
        test = data['test']
        valid = data['valid']

        self.x_train = np.array([tup[0] for tup in train])
        self.y_train = np.array([tup[1] for tup in train])
        # self.vector_train = np.array([tup[2] for tup in train])
        self.x_valid = np.array([tup[0] for tup in valid])
        self.y_valid = np.array([tup[1] for tup in valid])
        self.x_test = np.array([tup[0] for tup in test])
        self.y_test = np.array([tup[1] for tup in test])

        # self.logger.info(f"test_task first 100:\n{np.sum(self.y_test, axis=0)}")
        # self.logger.info(f"train_task: \n{list(self.y_test)}")

        if self.opt.et_use_clusters:
            self.doc_embeddings = np.array(data['clusters_centers'])
        else:
            self.doc_embeddings = np.array(data['doc_embeddings'])
        self.doc_embeddings_original_count = len(data['doc_embeddings'])

        self.x_test_list.append(self.x_test)
        self.y_test_list.append(self.y_test)

        self.early_stop = False
        self.lowest_lss = np.inf
        self.current_lr = self.opt.lr
        self.current_lr_patience = self.opt.lr_patience

        self.current_task = task_id
        self.tasks.append(data['name'])
        self.num_classes.append(len(data['classes']))

        self.train_size_per_class = data['size_per_class']
        self.num_train_list.append(len(self.x_train))
        self.num_test_list.append(len(self.x_test))
        self.num_valid_list.append(len(self.x_valid))
        if self.current_task > 0:
            self.init_on_new_task()

    def get_model(self, head_idx, test=False):
        # if False:
        # 	types = ['mask', 'bias', 'adaptive', 'adaptive_kb', 'atten']
        # 	for lid in range(4):
        # 		for _v in types:
        # 			vv = self.get_variable(_v, lid, head_idx)
        # 			print('tid %d, lid %d, vartype: %s, meanabs:%.6f'%(head_idx, lid, _v, np.mean(np.abs(vv.numpy()))))
        return self.models[head_idx]

    def get_test_size(self, task_idx):
        if task_idx == -1:  # if total
            total = [len(x) for x in self.x_test_list]
            return np.sum(total)
        else:
            return len(self.x_test_list[task_idx])

    def get_options(self):
        if len(self.options.keys()) > 0:
            return self.options
        else:
            self.options = {k: v for k, v in vars(self.opt).items()}
            return self.options

    def write_current_performances(self, pre_train: bool = False):
        self.logger.info("Writing current performance")
        performance_final = {}
        for tid in self.performance_final.keys():
            performance_final[tid] = (self.performance_final[tid],
                                      self.performance_final_f1_macro[tid])
        filename = self.filename if not pre_train else self.filename[:-5] + "_pre.json"

        plot_learning_curve(self.tasks, self.performance_watch_acc['test'], title=f"{self.client_id}_acc", x_axis_name="epochs",
                            y_axis_name="Accuracy Score", save_folder=join(self.log_dir, "Learning Curves"))

        write_file(self.log_dir, filename, {
            'client_id': self.client_id,
            'task_info': self.tasks,
            'performance': self.performance_epoch,
            'losses': self.loss_watch,
            'performance_watch_acc': self.performance_watch_acc,
            'performance_watch_f1_macro': self.performance_watch_f1_macro,
            'performance_final_acc_f1macro': self.performance_final,
            'mem_ratio': self.mem_ratio,
            'comm_ratio': self.comm_ratio,
            'options': self.get_options(),
            'data_info': self.data_info,
            'num_examples': {
                'train': self.num_train_list,
                'test': self.num_test_list,
                'valid': self.num_valid_list,
            }
        })
        filename = self.filename[:-5] + "_supp.json" if not pre_train else self.filename[:-5] + "_supp_pre.json"
        write_file(self.log_dir, filename, {
            'client_id': self.client_id,
            'classwise_acc_watch': self.classwise_acc_watch,
            'classwise_f1_watch': self.classwise_f1_watch,
            'pure_loss': self.pure_loss,
            'approx_loss': self.approx_loss,
            'weight_decay_loss': self.weight_decay_loss,
            'sparse_loss': self.sparse_loss,
            'sparse_mask_loss': self.sparse_mask_loss,
            'options': self.get_options(),
            'data_info': self.data_info,
            'task_info': self.tasks,
            'num_examples': {
                'train': self.num_train_list,
                'test': self.num_test_list,
                'valid': self.num_valid_list,
            }
        })
