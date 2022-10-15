import logging
import sys
from socketIO_client import SocketIO, LoggingNamespace

import tensorflow as tf

from FedWeit.data.generator import DataGenerator
from FedWeit.models.fedweit.fedweit_local import LocalFedWeIT
from FedWeit.utils import *

class Client:
    def __init__(self, opt):
        self.opt = opt
        self.current_task = -1
        self.count_rounds = 0
        self.recent_s_rounds = -1
        self.socketio_client = SocketIO(opt.host_ip, opt.host_port, LoggingNamespace)
        self.add_handlers()

    def get_local_model(self):
        # if get_model(self.opt.model, model='cnn'):
        #         return LocalCNN(self.client_id, self.data_info, self.opt)
        # elif get_model(self.opt.model, model='apd'):
        #         return LocalAPD(self.client_id, self.data_info, self.opt)
        if self.opt.model == 3 or self.opt.model == 5:
            return LocalFedWeIT(self.client_id, self.data_info, self.opt, self.logger)
        # elif get_model(self.opt.model, model='ewc'):
        #         return LocalEWC(self.client_id, self.data_info, self.opt)

    def get_next_task(self):
        self.current_task += 1
        self.count_rounds = 0
        data = self.data_generator.get_task(self.current_task)
        self.current_task_name = data['name']

        self.current_task_info = {'name': data['name'], 'classes': data['classes'],
                                  'size_per_class_split': data['size_per_class_split']}
        self.local_model.set_task(self.current_task, data)

        # Transfer task vectors, if enabled
        if self.opt.embedding_transfer:
            self.socketio_client.emit('client_embedding_vector',
                                      {'cid': self.client_id, 'tid': self.current_task,
                                       'doc_embeddings': obj_to_pickle_string(self.local_model.doc_embeddings),
                                       'task_original_count': self.local_model.doc_embeddings_original_count,
                                       'task_info': obj_to_pickle_string(self.current_task_info)}
                                      )
        syslog(self.client_id, f'the next task ({self.current_task_name}) has been loaded', self.logger)

    def stop(self):
        syslog(self.client_id, 'learning all tasks has been done.', self.logger)
        self.socketio_client.emit('client-stop', {'client_id': self.client_id})
        syslog(self.client_id, 'done.', self.logger)
        os._exit(0)

    def add_handlers(self):
        def client_request_init(*args):
            req = args[0]
            self.client_id = req['client_id']
            self.opt.log_dir = req['log_dir']
            self.opt.weights_dir = req['weights_dir']
            print(f"logging to {self.opt.log_dir}/client_{self.client_id}.log")

            self.logger = logging.getLogger(f"client_{self.client_id}")
            formatter_client = logging.Formatter('%(asctime)s : %(message)s')
            fileHandler_ = logging.FileHandler(f"{self.opt.log_dir}/client_{self.client_id}.log", mode='w')

            fileHandler_.setFormatter(formatter_client)
            self.logger.setLevel(logging.INFO)
            self.logger.addHandler(fileHandler_)
            streamHandler_client = logging.StreamHandler()
            streamHandler_client.setFormatter(formatter_client)
            self.logger.addHandler(streamHandler_client)

            self.logger.info("on_request_init")
            self.logger.info(f"\nUsing GPU {len(tf.config.list_physical_devices('GPU'))}\n")

            self.data_generator = DataGenerator(self.client_id, self.opt, self.logger, tasks_so_far=req['tasks_so_far'])
            if self.opt.exhaust_tasks:
                self.logger.info(f"Client {self.client_id} reporting its tasks")
                self.socketio_client.emit('client_report_tasks', {
                    "client_id": self.client_id, "tasks": self.data_generator.get_tasks()
                })
            self.data_info = self.data_generator.get_info()
            self.local_model = self.get_local_model()

            self.get_next_task()
            self.local_model.log_dir = req['log_dir']
            self.local_model.initialize(req['model_info'])

            self.logger.info(f"Command line instructions: {str(sys.argv)}")
            self.logger.info(f"Client Info: {self.data_info}")
            self.socketio_client.emit('client_ready', {'client_id': self.client_id})

        def client_request_train(*args):
            self.logger.info(f"client_request_train")
            req = args[0]

            self.count_rounds += 1
            syslog(self.client_id, 'task:%d, round:%d (cnt:%d), train one round'
                   % (self.current_task, req['server_round'], self.count_rounds), self.logger)

            is_last_task = (self.current_task == self.data_info['num_tasks'] - 1)
            is_last_round = (self.count_rounds % self.opt.num_rounds == 0 and self.count_rounds != 0)
            is_last = is_last_task and is_last_round
            if self.opt.socket_test:
                resp = {'client_id': self.client_id, 'client_round': req['server_round']}
            elif not req['comm']:
                self.logger.info("not reg['comm']")
                # if (self.count_rounds == 1 or (self.count_rounds == 2 and self.opt.embedding_transfer)) \
                #         and req['server_weights'] is not None:
                if self.count_rounds == 1 and req['server_weights'] is not None:
                    weights_both = pickle_string_to_obj(req['server_weights'])
                    self.local_model.set_adapts(weights_both[1])
                    if self.opt.embedding_transfer:
                        if self.opt.et_init_alphas:
                            self.local_model.set_alphas(pickle_string_to_obj(req["adaptive_scores"]))

                self.local_model.train_one_round(req['server_round'], self.count_rounds, is_last)
                syslog(self.client_id, 'training only', self.logger)
            else:
                if self.opt.federated:
                    if req['server_round'] == 0:
                        client_w = pickle_string_to_obj(req['server_weights'])
                        self.local_model.set_weights(client_w)
                    else:
                        weights_both = pickle_string_to_obj(req['server_weights'])
                        self.local_model.set_weights(weights_both[0])
                        # if self.count_rounds == 1 or (self.count_rounds == 2 and self.opt.embedding_transfer):
                        if self.count_rounds == 1:
                            self.local_model.set_adapts(weights_both[1])
                            if self.opt.embedding_transfer:
                                if self.opt.et_init_alphas:
                                    self.local_model.set_alphas(pickle_string_to_obj(req["adaptive_scores"]))

                ########################################################################
                self.local_model.train_one_round(req['server_round'], self.count_rounds, is_last)
                ########################################################################
                atten_variable = self.local_model.get_variable(var_type='atten', layer_idx=0,
                                                               task_idx=self.current_task)
                self.logger.info(f'cid: {self.client_id} task:{self.current_task}, round:{self.count_rounds}')
                self.logger.info(f"Attention value: mean: {tf.math.reduce_mean(tf.abs(atten_variable)).numpy()} "
                                 f"std: {tf.math.reduce_std(tf.abs(atten_variable)).numpy()}\n{atten_variable}")

                resp = {'client_id': int(self.client_id), 'task_id': int(self.current_task), 'client_round': int(req['server_round'])}
                if self.opt.federated:
                    if self.opt.sparse_comm:
                        weights, masks = self.local_model.get_weights()
                        adps = self.local_model.get_adapts()
                        resp['client_masks'] = masks
                    else:
                        # Default
                        weights = self.local_model.get_weights()
                        adps = self.local_model.get_adapts()

                    resp['client_both'] = obj_to_pickle_string([weights, adps])
                    resp['train_size_per_class'] = list(map(int, self.local_model.train_size_per_class))
                    resp['train_size'] = len(self.local_model.x_train)
                    resp['current_task'] = int(self.current_task)
                    resp['last_round'] = self.local_model.early_stop or is_last_round
                    resp['early_stop'] = self.local_model.early_stop
                    # if self.opt.embedding_transfer:
                    #         resp['vector_train'] = obj_to_pickle_string(self.local_model.vector_train)
                    self.resp = resp

            if not self.opt.model == 0:
                self.local_model.calculate_capacity()
            self.local_model.write_current_performances()
            if req['comm']:
                self.socketio_client.emit('client_update', resp)  # to the server only
            else:
                self.socketio_client.emit('client-train-done', {'client_id': self.client_id})
            if is_last_round or self.local_model.early_stop:
                if self.opt.save_weights:
                    np_save(self.opt.weights_dir, f'client{self.client_id}-task{self.current_task}-weights.npy',
                            self.local_model.variables)
                    syslog(self.client_id, '%dth weights has been saved.' % self.current_task, self.logger)
                if is_last_task:
                    metrics_update = {}
                    metrics_xls = {}
                    for metrics_name, metrics in [("acc", self.local_model.performance_watch_acc),
                                                  ("f1_macro", self.local_model.performance_watch_f1_macro),
                                                  ("loss", self.local_model.loss_watch)]:
                        metrics_xls[metrics_name] = metrics
                        for split in metrics:
                            metrics_update[f"{split}_{metrics_name}"] = []
                            for task_idx in metrics[split]:
                                last_round_task_idx = max(metrics[split][task_idx].keys())

                                metrics_update[f"{split}_{metrics_name}"].append(
                                    metrics[split][task_idx][last_round_task_idx][-1])
                    self.logger.info(f"Client {self.client_id} updating mlflow")
                    self.socketio_client.emit('update_mlflow', self.client_id, metrics_update, metrics_xls)
                    if self.local_model.early_stop:
                        self.local_model.evaluate(is_last=True)
                    self.stop()
                else:
                    self.get_next_task()

        def on_request_train_all(*args):
            self.logger.info(f"on_request_train_all")
            for i in range(self.data_info['num_tasks']):
                for j in range(self.opt.num_rounds):
                    self.count_rounds += 1
                    syslog(self.client_id, 'task:%d, round:%d, train one round'
                           % (self.current_task, self.count_rounds), self.logger)

                    is_last_task = (self.current_task == self.data_info['num_tasks'] - 1)
                    is_last_round = (self.count_rounds % self.opt.num_rounds == 0 and self.count_rounds != 0)
                    is_last = is_last_task and is_last_round
                    r = (self.count_rounds - 1) % self.opt.num_rounds
                    self.local_model.train_one_round(r, self.count_rounds, is_last)

                    if not self.opt.model == 0:
                        self.local_model.calculate_capacity()
                    self.local_model.write_current_performances()
                    if is_last_round or self.local_model.early_stop:
                        if is_last_task:
                            if self.local_model.early_stop:
                                self.local_model.evaluation(is_last=True)
                            self.stop()
                        else:
                            if self.opt.save_weights:
                                save_weights(self.opt.weights_dir, 'client%d-task%d-weights.npy'
                                             % (self.client_id, self.current_task), self.local_model.variables)
                                syslog(self.client_id, '%dth weights has been saved.' % self.current_task, self.logger)
                            self.get_next_task()

        # register handlers
        print("adding client handlers")
        self.socketio_client.on('client_request_init', client_request_init)
        self.socketio_client.on('client_request_train', client_request_train)
        self.socketio_client.on('train_all', on_request_train_all)
        self.socketio_client.emit('single_client_started', {})
        self.socketio_client.wait()
