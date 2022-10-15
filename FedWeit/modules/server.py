import logging
import threading
import time
from copy import deepcopy
from os.path import dirname
from pprint import pformat
from xlwt import Workbook
from os.path import join, abspath, realpath

import mlflow
from flask import Flask, request
from flask_socketio import *
import webbrowser

# from models.cnn.cnn_global import GlobalCNN
# from models.ewc.ewc_global import GlobalEWC
# from models.apd.apd_global import GlobalAPD
from scipy.spatial.distance import cosine

from FedWeit.models.fedweit.fedweit_global import GlobalFedWeIT
from FedWeit.models.utils import get_conv_fc_layers
from FedWeit.utils import *
from FedWeit.data.generator import DataGenerator


class Server:
    def __init__(self, opt):
        self.opt = opt
        self.sid2cid = {}
        self.responses = []
        self.ready_clients = set()
        self.early_stopped_responses = []
        # self.task_vector_embeddings = {}
        self.task_score_dict = {}
        self.task_info = {}

        self.trained_clients = []

        self.server_id = -2
        self.client_id = -1
        self.tasks_in_clients = set()
        self.tasks_exhausted = False

        self.current_round = -1
        self.new_task_started = True
        self.has_begun = False
        # opt.frac_clients = fraction of clients to participate per round. default = 1.0
        self.num_waitings = round(opt.num_clients * opt.frac_clients)
        self.num_stopped_clients = 0
        self.metrics_mlflow = {}
        self.metrics = {}

        self.lock = threading.Lock()
        self.lock_train = threading.Lock()

        logger_server = logging.getLogger("server_logger")
        formatter = logging.Formatter('%(asctime)s : %(message)s')
        fileHandler = logging.FileHandler(f"{self.opt.log_dir}/server.log", mode='w')
        fileHandler.setFormatter(formatter)
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)
        logger_server.setLevel(logging.DEBUG)
        logger_server.addHandler(fileHandler)
        logger_server.addHandler(streamHandler)
        self.logger = logger_server

        logger = logging.getLogger('werkzeug')
        self.server = Flask(__name__)
        self.server.logger = logger
        self.socketio = SocketIO(self.server)
        self.server.logger.setLevel(logging.ERROR)

        if not os.path.isdir(self.opt.log_dir):
            os.makedirs(self.opt.log_dir)

        self.global_model = self.get_global_model()
        shapes = self.global_model.get_info()['shapes']
        conv_layers, fc_layers = get_conv_fc_layers(shapes)
        self.opt.shapes = {
            "shapes": shapes, "conv": conv_layers, "fc": fc_layers
        }

        self.listener = threading.Thread(target=self.listen)
        self.listener.daemon = True
        self.listener.start()

        self.add_handlers()

    def run(self):
        self.logger.info(f"Command line instructions: {str(sys.argv)}")
        self.logger.info(f"\n\n\nStarting server on port {self.opt.host_port}")
        write_file(self.opt.log_dir, "opt.json", vars(self.opt))
        self.socketio.run(self.server, host=self.opt.host_ip, port=self.opt.host_port)

    def get_global_model(self):
        # if get_model(self.opt.model, model='cnn'):
        #         return GlobalCNN(self.opt)
        # elif get_model(self.opt.model, model='apd'):
        #         return GlobalAPD(self.opt)
        if self.opt.model == 3 or self.opt.model == 5:
            return GlobalFedWeIT(self.opt, self.logger)
        # elif get_model(self.opt.model, model='ewc'):
        #         return GlobalEWC(self.opt)

    def listen(self):
        syslog(self.server_id, 'started to listen', self.logger)
        # is_wait = True
        is_train = False
        if not self.opt.federated:
            self.logger.info(f"self.ready_clients {self.ready_clients} self.opt.num_clients {self.opt.num_clients}")
            while len(self.ready_clients) <= self.opt.num_clients:
                if len(self.ready_clients) == self.opt.num_clients:
                    for sid in self.ready_clients:
                        self.socketio.emit('train_all', {}, room=sid)
                        self.socketio.sleep(1)
                    is_train = True
                if len(self.ready_clients) <= 0 and is_train:
                    self.stop()
                time.sleep(2)
        while True:
            if self.has_begun:
                if len(self.responses) + len(self.early_stopped_responses) >= self.num_waitings:
                    self.new_task_started = False
                    last_round_tasks = [response['last_round'] for response in
                                        self.responses + self.early_stopped_responses]
                    task_ids = [response['current_task'] for response in
                                self.responses + self.early_stopped_responses]
                    self.logger.info(f"last_round_tasks: {last_round_tasks}")

                    assert len(set(task_ids)) <= 1, "task_ids mismatch between responses"
                    if all([last_round_task for last_round_task in last_round_tasks]) and len(
                            last_round_tasks) >= 1:
                        self.new_task_started = True
                    if self.new_task_started:
                        if task_ids[-1] == self.opt.num_pick_tasks - 1:
                            self.stop()

                    if self.opt.federated and not self.opt.socket_test:
                        syslog(self.server_id, 'update global weights', self.logger)
                        with self.lock:
                            responses = self.responses + self.early_stopped_responses
                            self.global_model.update_weights(responses)
                            # self.adapts_cids = [resp['client_id'] for resp in self.responses + self.early_stopped_responses]
                        if self.opt.sparse_comm:
                            self.global_model.write_current_status()

                    # current round done
                    if len(self.responses) + len(self.early_stopped_responses) + len(
                            self.trained_clients) >= self.opt.num_clients:
                        # train new round
                        with self.lock:
                            self.responses = []  # reset
                            if self.new_task_started:
                                # train new task
                                self.early_stopped_responses = []

                        with self.lock_train:
                            self.trained_clients = []  # reset
                        self.train_next_round()

                    if len(self.ready_clients) <= 0:
                        self.stop()
            else:
                if len(self.ready_clients) == self.opt.num_clients:  # >=self.num_waitings:
                    self.logger.info("Proceeding to train_next_round")
                    self.has_begun = True
                    self.train_next_round()
            time.sleep(1)

    def train_next_round(self):
        # np_save(self.opt.log_dir, f'pre-weights.npy',
        #         self.global_model.get_weights())

        self.current_round += 1
        self.logger.info(f"train_next_round, current_round: {self.current_round}")
        # How many clients are selected depends on opt.frac_clients. default = 1.0
        try:
            random.seed(self.opt.random_seed + self.client_id + self.global_model.get_current_task_number())
            self.selected_clients = random.sample(list(self.ready_clients), self.num_waitings)
        except ValueError:
            # At least one other client must have early stopped in another round
            self.selected_clients = self.ready_clients
        self.logger.info(f"round:{self.current_round}, request train one round ({self.get_clients(self.selected_clients)})")

        req = {'server_round': self.current_round}
        # Get the weights
        if self.opt.federated and not self.opt.socket_test:  # self.opt.socket_test default=False
            if self.current_round > 0:
                get_w = self.global_model.get_weights()
                get_a, _ = self.global_model.get_adapts()
                req['server_weights'] = np.array([get_w, get_a])
            else:
                req['server_weights'] = self.global_model.get_weights()
        elif not self.opt.federated:
            req['server_weights'] = None

        early_stopped_clients = [response['client_id'] for response in self.early_stopped_responses]
        cid2sid = {value: key for key, value in self.sid2cid.items()}
        early_stopped_clients = [cid2sid[cid] for cid in early_stopped_clients]

        req['comm'] = True
        i = 0
        while len(self.responses) + len(self.early_stopped_responses) < self.num_waitings:
            if len(self.responses) >= i:
                # 5 clients are called at once, perhaps as buffer?
                sids = list(self.selected_clients)[i: i + 5]
                tid = self.global_model.get_current_task_number()
                if self.new_task_started and self.current_round > 0:
                    tid += 1
                for sid in sids:
                    if sid in early_stopped_clients:
                        continue
                    req_client = deepcopy(req)
                    if self.opt.federated and not self.opt.socket_test and self.current_round > 0 \
                            and self.opt.embedding_transfer and self.global_model.task_doc_embeddings:
                        get_w, client_adapts = req_client['server_weights']
                        cid = self.sid2cid[sid]
                        self.logger.info(f"Getting top adapts via task vector embeddings for client {cid}")
                        self.logger.info(f"Total client adapts under consideration: {len(client_adapts.keys())}")
                        client_adapts, adaptive_scores = self.get_top_adapts(client_adapts, ignore_cid=cid,
                                                                             ignore_tid=tid, top_k=self.opt.et_top_k)
                        if self.opt.et_init_alphas:
                            self.logger.info(f"Total alphas sent: {len(adaptive_scores)}")
                            req_client['adaptive_scores'] = obj_to_pickle_string(adaptive_scores)
                        req_client['server_weights'] = np.array([get_w, client_adapts])

                    req_client['server_weights'] = obj_to_pickle_string(req_client['server_weights'])
                    self.logger.info(f"Emitting for selected client sid {sid} cid {self.get_clients([sid])}")
                    self.socketio.emit('client_request_train', req_client, room=sid)
                    self.socketio.sleep(1)
                i += 5
            time.sleep(0.5)

        req['comm'] = False
        if self.current_round == 0 or self.current_round % self.opt.num_rounds != 0:
            req['server_weights'] = None

        syslog(self.server_id, "Transmit training request to not comm. clients", self.logger)
        i = 0
        only_train_clients = self.ready_clients - set(self.selected_clients)
        while len(self.trained_clients) + len(self.early_stopped_responses) < \
                self.opt.num_clients - self.num_waitings:
            if len(self.trained_clients) >= i:
                sids = list(only_train_clients)[i: i + 5]
                for sid in sids:
                    if sid in early_stopped_clients:
                        continue
                    self.logger.info(f"Emitting for only_train client sid {sid}")
                    self.socketio.emit('client_request_train', req, room=sid)  # to the specific client
                    self.socketio.sleep(0.5)
                i += 5
            time.sleep(0.5)
        # self.socketio.emit('client_request_train', req) # broadcast to all

        time.sleep(20)

    def log_mlflow(self):
        time.sleep(15)
        mlflow_dir = abspath(dirname(dirname(dirname(__file__))))
        mlflow.set_tracking_uri(f"file://{join(mlflow_dir, 'mlruns')}")
        client = mlflow.tracking.MlflowClient()

        # expt_name = f"{self.opt.task_pool}_{self.opt.et_top_k}_{self.opt.split_option}_nt_{self.opt.num_pick_tasks}"
        expt_name = f"{self.opt.task_pool}_l_{self.opt.approx_hyp}"
        if self.opt.et_top_k > 0:
            expt_name += f"_et__{self.opt.et_top_k}"
        if self.opt.converge_first_round_epochs or self.opt.adaptive_random or self.opt.reset_current_lr:
            expt_name += "_ablative"
        experiment = mlflow.get_experiment_by_name(expt_name)
        if not experiment:
            self.logger.info("Creating Experiment")
            experiment = mlflow.create_experiment(expt_name)
        else:
            self.logger.info("Loaded Experiment")
            experiment = experiment.experiment_id

        if self.opt.fedweit or self.opt.task_type in ['iid']:
            run_name = self.opt.task_type
        else:
            run_name = "FedProx" if self.opt.fed_method == 1 else "FedAvg"
            if self.opt.mask:
                run_name += "_Masked"
        if self.opt.debug:
            run_name = "Debug_" + run_name
        run = client.create_run(experiment, tags={"mlflow.runName": run_name})

        run_id = run.info.run_id
        client.log_param(run_id, "_run_id", run_id)
        # client.log_param(run_id, "epochs", self.opt.num_epochs)
        # client.log_param(run_id, "rounds", self.opt.num_rounds)
        # client.log_param(run_id, "num_clients", self.opt.num_clients)

        # client.log_param(run_id, "embedding_dim", self.opt.embedding_dim)
        # model_shape = self.global_model.get_info()['shapes']
        # conv_layers, fc_layers = get_conv_fc_layers(model_shape)
        # conv_layers = [model_shape[i] for i in conv_layers]
        # fc_layers = [model_shape[i] for i in fc_layers]
        # client.log_param(run_id, "conv_layers", str(conv_layers))
        # client.log_param(run_id, "fc_layers", str(fc_layers))
        client.log_param(run_id, "model_architecture", f"{self.opt.model}_{self.opt.base_architect}")

        # client.log_param(run_id, "lr", self.opt.lr)
        # client.log_param(run_id, "num_classes", self.opt.num_classes)
        # client.log_param(run_id, "split_option", self.opt.split_option)

        # client.log_param(run_id, "num_tasks", self.opt.num_pick_tasks)
        # client.log_param(run_id, "gen_num_tasks", self.opt.gen_num_tasks)

        client.log_param(run_id, "converge_first_round_epochs", self.opt.converge_first_round_epochs)
        client.log_param(run_id, "adaptive_random", self.opt.adaptive_random)
        client.log_param(run_id, "reset_lr", self.opt.reset_current_lr)
        client.log_param(run_id, "embedding_transfer", self.opt.embedding_transfer)
        if self.opt.embedding_transfer:
            # client.log_param(run_id, "et_top_k", self.opt.et_top_k)
            client.log_param(run_id, "et_method", self.opt.et_task_similarity_method)
            client.log_param(run_id, "et_use_clusters", self.opt.et_use_clusters)
            client.log_param(run_id, "et_init_alphas", self.opt.et_init_alphas)
            client.log_param(run_id, "et_alphas_trainable", self.opt.et_alphas_trainable)

        model_type = "FedWeIT" if self.opt.base_architect == 2 else "FedPrIT"

        client.log_param(run_id, "lambda_2", self.opt.approx_hyp)
        # client.log_param(run_id, "disable_alphas", self.opt.disable_alphas)
        # client.log_param(run_id, "foreign_adaptives_trainable_constant", None if not self.opt.foreign_adaptives_trainable else
        #                  self.opt.foreign_adaptives_trainable_drift_constant)

        fedweit_dense = "FedWeIT Dense" if model_type == "FedWeIT" and self.opt.fedweit_dense else ""
        client.log_param(run_id, "fedweit_dense", fedweit_dense)
        # client.log_param(run_id, "concatenate_aw_kbs", self.opt.concatenate_aw_kbs)
        shared = "Shared" if not self.opt.dense_detached else ""
        client.log_param(run_id, "dense_shared", shared)
        project = "Project" if self.opt.project_adaptives else ""
        client.log_param(run_id, "project_adaptives", project)

        if model_type != "FedWeIT":
            model_type = (model_type + shared + project).strip()
        else:
            model_type = "FedWeIT Dense" if model_type == "FedWeIT" and self.opt.fedweit_dense else model_type
        client.log_param(run_id, "Model Type", model_type)

        # client.log_param(run_id, "old_early_stopping_enabled", self.opt.old_early_stopping_enabled)
        client.log_param(run_id, "seeds", ", ".join([str(self.opt.random_seed), str(self.opt.random_seed_task_alloc)]))

        # client.log_param(run_id, "approx_hyp", self.opt.approx_hyp)
        # client.log_param(run_id, "l1_hyp", self.opt.l1_hyp)
        # client.log_param(run_id, "mask_hyp", self.opt.mask_hyp)
        # client.log_param(run_id, "l1_mask_hyp", self.opt.l1_mask_hyp)

        for key in self.metrics_mlflow:
            self.logger.info(f"Logging metric {key} with {len(self.metrics_mlflow[key])} elements")
            score = sum(self.metrics_mlflow[key]) / len(self.metrics_mlflow[key])
            self.logger.info(f"\t{key}: {score}")
            client.log_metric(run_id, key, score)

        client.log_artifacts(run_id, self.opt.log_dir)
        if os.path.exists(self.opt.weights_dir):
            client.log_artifacts(run_id, self.opt.weights_dir, artifact_path="weights")

        data_dir = DataGenerator(self.server_id, self.opt, self.logger, set()).base_dir
        for filename in os.listdir(data_dir):
            filepath = os.path.join(data_dir, filename)
            if os.path.isdir(filepath):
                client.log_artifacts(run_id, filepath, artifact_path=f"data/{filename}")
            else:
                if filename.endswith(".npy") and not self.opt.log_data_mlflow:
                    continue
                client.log_artifact(run_id, filepath, artifact_path="data")

        # client.log_artifacts(run_id, local_dir=temp_model_path)
        client.set_terminated(run_id)
        self.logger.info(f"Mlflow logged to run_id {run_id} of expt {expt_name}")
        webbrowser.open('file://' + realpath(join(mlflow_dir, 'mlruns', experiment, run_id)))

    def stop(self):
        time.sleep(10)
        syslog(self.server_id, 'all clients are done.', self.logger)

        write_file(self.opt.log_dir, "opt.json", vars(self.opt))

        wb = Workbook()
        for split in ['test', 'valid', 'train']:
            sheet = wb.add_sheet(split)
            row, column = 0, 0
            metrics_split = ['acc', 'f1_macro']
            # Dictionary to store all values for this cell, so they can be avg later
            for metric_name in metrics_split:
                metrics_values = dict()
                column_names_values = dict()
                sheet.write(row, column, f"{split}_{metric_name}")
                for cid in range(len(self.metrics)):
                    row_client = row + cid + 2
                    sheet.write(row_client, column, f"Client_{cid}")

                    try:
                        for task_idx in sorted(self.metrics[cid][metric_name][split].keys()):
                            for server_round in self.metrics[cid][metric_name][split][task_idx].keys():
                                column_client = column + int(server_round) + 1
                                column_names_values[(row + 1, column_client)] = f"t_{task_idx}_r_{server_round}"

                                value = self.metrics[cid][metric_name][split][task_idx][server_round][-1]
                                if (row_client, column_client) in metrics_values:
                                    metrics_values[(row_client, column_client)].append(value)
                                else:
                                    metrics_values[(row_client, column_client)] = [value]
                    except KeyError as err:
                        self.logger.error(pformat(self.metrics))
                        self.logger.error(err)
                        exit()

                for row_client, column_client in metrics_values.keys():
                    sheet.write(row_client, column_client,
                                sum(metrics_values[(row_client, column_client)]) /
                                len(metrics_values[(row_client, column_client)]))
                for _, column_names_id in column_names_values.keys():
                    sheet.write(row + 1, column_names_id, column_names_values[(row + 1, column_names_id)])

                columns_clients = set(column_client for _, column_client in metrics_values.keys())
                for column_clients in columns_clients:
                    values_column = [metrics_values[(row + cid + 2, column_clients)]
                                     for cid in range(len(self.metrics))
                                     if (row + cid + 2, column_clients) in metrics_values]
                    values_column = [sum(client_val) / len(client_val) for client_val in values_column]
                    sheet.write(row + len(self.metrics) + 2, column_clients,
                                sum(values_column) / len(values_column))
                row += len(self.metrics) + 5
        wb.save(os.path.join(self.opt.log_dir, 'Metrics.xlsx'))

        if self.opt.save_weights:
            self.logger.info(f"All clients done; saving weights in {self.opt.weights_dir}")
            np_save(self.opt.weights_dir, f'round-{self.current_round}-aggr-weights.npy',
                    self.global_model.get_weights())
            syslog(self.server_id, 'final aggregated weights has been saved.', self.logger)

        self.log_mlflow()
        syslog(self.server_id, f'done. {self.opt.log_dir}', self.logger)
        os._exit(0)  # thread

    def get_clients(self, sids):
        c = [self.sid2cid[sid] for sid in sids]
        return ','.join(map(str, c))

    def get_top_adapts(self, client_adapts: dict, ignore_cid: int, ignore_tid: int, top_k: int = 3):
        task_doc_embeddings_this_client = self.global_model.task_doc_embeddings[(ignore_cid, ignore_tid)]["doc_embeddings"]
        cids_tids = list(client_adapts.keys())

        task_vectors = [self.global_model.task_doc_embeddings[(cid, tid)] for cid, tid in cids_tids]
        client_adapts = list(client_adapts.values())
        try:
            ignore_cid_tid_index = cids_tids.index((ignore_cid, ignore_tid))
            cids_tids = cids_tids[:ignore_cid_tid_index] + cids_tids[ignore_cid_tid_index + 1:]
            task_vectors = task_vectors[:ignore_cid_tid_index] + task_vectors[ignore_cid_tid_index + 1:]
            client_adapts = client_adapts[:ignore_cid_tid_index] + client_adapts[ignore_cid_tid_index + 1:]
        except ValueError:
            # This is a new task; so nothing to ignore
            pass

        def similarity_between_tasks_step(task_vector_1, task_vector_2, threshold=0.7):
            # [n1 x n2, 1] vector
            cosine_values = [1 - cosine(doc1, doc2) for doc1 in task_vector_1 for doc2 in
                             task_vector_2]
            score = sum(cosine_value > threshold for cosine_value in cosine_values) / len(cosine_values)
            return score

        def similarity_between_tasks_rectified_linear(task_vector_1, task_vector_2, threshold=0.7):
            cosine_values = [1 - cosine(doc1, doc2) for doc1 in task_vector_1 for doc2 in
                             task_vector_2]
            bin_vals, bins = np.histogram(cosine_values, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            score = sum([cosine_value for cosine_value in cosine_values if cosine_value > threshold]) / len(cosine_values)

            self.logger.info(f"Rectifided Linear Similarity Score {score} with threshold {threshold}\n{bin_vals}")
            return score

        def similarity_between_tasks_linear(task_vector_1, task_vector_2):
            cosine_values = [1 - cosine(doc1, doc2) for doc1 in task_vector_1 for doc2 in
                             task_vector_2]
            score = sum(cosine_values) / len(cosine_values)
            return score

        def similarity_between_tasks_count_rectified_linear(task_vector_1, task_vector_2, threshold=0.7):
            cosine_values = [1 - cosine(doc1, doc2) for doc1 in task_vector_1 for doc2 in
                             task_vector_2]
            bin_vals, bins = np.histogram(cosine_values, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            all_valid_cosine_values = [cosine_value for cosine_value in cosine_values if cosine_value > threshold]
            score = sum(all_valid_cosine_values) / len(all_valid_cosine_values)

            self.logger.info(f"Rectifided Linear Similarity Score {score} with threshold {threshold}\n{bin_vals}")
            return score

        # def similarity_between_tasks_bhattacharya_single(task_vector_1, task_vector_2):
        #     # [n1 x n2, 1] vector
        #     score = bhattacharya_distance(task_vector_1, task_vector_2)
        #     return score

        scores = []
        ordering_counts = []
        self.logger.info(f"Task Sim Method: {self.opt.et_task_similarity_method}")
        for task_vector_index, other_task_vector in enumerate(task_vectors):
            other_task_vector, other_task_vector_length = other_task_vector["doc_embeddings"], \
                                                          other_task_vector["task_original_count"]
            cid_tid_this_task = (ignore_cid, ignore_tid)
            cid_tid_other_task = cids_tids[task_vector_index]
            if (cid_tid_this_task, cid_tid_other_task) not in self.task_score_dict:
                self.logger.info(f"Task-Task similarity between {cid_tid_this_task} and {cid_tid_other_task}")
                # score = similarity_metric_between_tasks_linear(task_vector_this_client, other_task_vector)
                if self.opt.et_task_similarity_method.startswith("rectified_linear"):
                    score = similarity_between_tasks_rectified_linear(task_doc_embeddings_this_client, other_task_vector)
                # elif self.opt.et_task_similarity_method == "bhatt":
                #     score = similarity_between_tasks_bhattacharya_single(task_doc_embeddings_this_client, other_task_vector)
                elif self.opt.et_task_similarity_method.startswith("linear"):
                    score = similarity_between_tasks_linear(task_doc_embeddings_this_client, other_task_vector)
                elif self.opt.et_task_similarity_method.startswith("step"):
                    score = similarity_between_tasks_step(task_doc_embeddings_this_client, other_task_vector)
                elif self.opt.et_task_similarity_method.startswith("count_rectified_linear"):
                    score = similarity_between_tasks_count_rectified_linear(task_doc_embeddings_this_client, other_task_vector)
                else:
                    raise ValueError(f"Invalid et_task_similairity_method {self.opt.et_task_similarity_method}")
                self.task_score_dict[(cid_tid_this_task, cid_tid_other_task)] = \
                    self.task_score_dict[(cid_tid_other_task, cid_tid_this_task)] = score
            scores.append(self.task_score_dict[(cid_tid_this_task, cid_tid_other_task)])
            ordering_counts.append(other_task_vector_length)

        # 0.39, 0.439
        self.logger.info(f"\n\nAnchor Task, cid: {ignore_cid}, tid: {ignore_tid}: "
                         f"{pformat(self.task_info[(ignore_cid, ignore_tid)], indent=4)}")
        if cids_tids:
            if "_ordered" in self.opt.et_task_similarity_method:
                scores, ordering_counts, client_adapts, task_vectors, cids_tids = (list(ls) for ls in zip(
                    *sorted(zip(scores, ordering_counts, client_adapts, task_vectors, cids_tids),
                            key=lambda x: (-x[1], -x[0]))[:top_k]))
                self.logger.info(f"Top counts of compared tasks: {ordering_counts}")
            else:
                scores, client_adapts, task_vectors, cids_tids = (list(ls) for ls in zip(
                    *sorted(zip(scores, client_adapts, task_vectors, cids_tids), key=lambda x: -x[0])[:top_k]))
        if "_normalised" in self.opt.et_task_similarity_method:
            self.logger.info(f"Top cosine scores before normalisation: {scores}")
            scores = [float(score) / sum(scores) for score in scores]

        self.logger.info(f"Top cosine scores: {scores}")
        self.logger.info(f"Top (client_id, task_id): {cids_tids}")
        for (cid, tid), score in zip(cids_tids, scores):
            # self.logger.info(f"Score {score} for task:\n{json.dumps(self.task_info[(cid, tid)])}")
            self.logger.info(f"Score {score} for task:\n{pformat(self.task_info[(cid, tid)], indent=4)}")
        return client_adapts, scores

    def add_handlers(self):
        @self.socketio.on('update_mlflow')
        def update_mlflow(client_id, metrics_mlflow, metrics):
            with self.lock:
                self.logger.info(f"Updating Mlflow matrix for client {client_id}")
                if not self.metrics_mlflow:
                    self.metrics_mlflow = metrics_mlflow
                else:
                    for metric_name in metrics_mlflow:
                        self.metrics_mlflow[metric_name] += metrics_mlflow[metric_name]
            self.metrics[client_id] = metrics

        @self.socketio.on('single_client_started')
        def handle_start(resp):
            time.sleep(2)
            with self.lock:
                self.client_id += 1
                cid = self.client_id
                self.sid2cid[request.sid] = cid
                self.logger.info(f"Initialise-emit for client {cid} sid: {str(request.sid)}")
                message = {
                    'client_id': cid, 'model_info': self.global_model.get_info(), 'log_dir': self.opt.log_dir,
                    'weights_dir': self.opt.weights_dir, 'tasks_so_far': list(self.tasks_in_clients)
                }
                # to the specific client
                request_sent = self.socketio.emit('client_request_init', message, room=request.sid)
                self.logger.info(f"{request_sent}")
                if self.opt.exhaust_tasks:
                    total_tasks_limit = (self.client_id + 1) * self.opt.num_pick_tasks
                    while len(self.tasks_in_clients) < total_tasks_limit and not self.tasks_exhausted:
                        time.sleep(1)
                        pass

        @self.socketio.on('client_report_tasks')
        def handle_report_tasks(resp):
            self.logger.info(f"Client {resp['client_id']} reporting its tasks: {resp['tasks']}")
            tasks_count = len(self.tasks_in_clients)
            self.tasks_in_clients.update(resp["tasks"])
            self.logger.info(f"Old tasks_list: {tasks_count}; New tasks_list: {len(self.tasks_in_clients)}")
            if len(self.tasks_in_clients) != tasks_count + self.opt.num_pick_tasks:
                self.tasks_exhausted = True

        @self.socketio.on('client_ready')
        def handle_ready(resp):
            syslog(self.server_id, 'client:%d is ready' % (resp['client_id']), self.logger)
            with self.lock:
                self.ready_clients.add(request.sid)

        @self.socketio.on('client_update')
        def handle_update(resp):
            """
            Client returns w and a ; append this response to self.responses
            """
            if resp['client_round'] < self.current_round:
                syslog(self.server_id, 'round:%d, receive outdated updates from client:%d (client-round:%d). Ignored.'
                       % (self.current_round, resp['client_id'], resp['client_round']), self.logger)
            else:
                syslog(self.server_id, 'round:%d, receive trained weights from client:%d.'
                       % (self.current_round, resp['client_id']), self.logger)  # , sys.getsizeof(resp)
                with self.lock:
                    time.sleep(2)
                    if resp['early_stop']:
                        self.early_stopped_responses.append(resp)
                    else:
                        self.responses.append(resp)

        @self.socketio.on('client-train-done')
        def handle_train_done(resp):
            syslog(self.server_id, 'client:%d training done!' % (resp['client_id']), self.logger)
            with self.lock_train:
                # ToDo Question: How does request.sid work?
                self.trained_clients.append(request.sid)

        @self.socketio.on('client-stop')
        def handle_stop(resp):
            syslog(self.server_id, 'round:%d, client:%d has been stopped ' % (self.current_round, resp['client_id']),
                   self.logger)
            with self.lock:
                self.ready_clients.remove(request.sid)
                self.num_stopped_clients += 1

        @self.socketio.on('client_embedding_vector')
        def handle_embedding_vector(resp):
            # Only called if embedding_transfer = True and new task loaded at client side
            syslog(self.server_id, f'round:{self.current_round + 1}, sid:{request.sid} '
                                   f'initialised new task', self.logger)
            cid, tid = resp['cid'], resp['tid']
            self.global_model.task_doc_embeddings[(cid, tid)] = {
                'doc_embeddings': pickle_string_to_obj(resp['doc_embeddings']),
                'task_original_count': resp['task_original_count']
            }
            self.task_info[(cid, tid)] = pickle_string_to_obj(resp['task_info'])
