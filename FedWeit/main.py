import os
from os.path import dirname, join, abspath
import datetime
# import multiprocessing
import time

from FedWeit.modules.launcher import Launcher
from FedWeit.models.utils import fix_mlflow_uris
from FedWeit.parser import Parser
from FedWeit.utils import pp


def main(opt):
    opt = get_settings(opt)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    # if opt.worker_type == "both":
    # 	for worker_type in ["server", "client"]:
    # 		p = multiprocessing.Process(target=launch, args=(deepcopy(opt), worker_type))
    # 		p.start()
    # 		time.sleep(2)
    # else:
    launch(opt, opt.worker_type)


def launch(opt, worker_type: str):
    opt.worker_type = worker_type
    # print(opt.worker_type)
    # exit()
    pp.pprint(opt)
    launcher = Launcher(opt)
    if opt.worker_type == 'server':
        fix_mlflow_uris(opt)
        # exit()
        if not os.path.exists(opt.log_dir):
            os.makedirs(opt.log_dir)
        launcher.run_server()
    elif opt.worker_type == 'client':
        time.sleep(1)
        launcher.run_clients()
    elif opt.worker_type == 'data':
        if not os.path.exists(opt.log_dir):
            os.makedirs(opt.log_dir)
        launcher.generate_data()
    else:
        exit('SystemExit: no proper worker type was given.')


def get_settings(opt):
    now = datetime.datetime.now().strftime("%d.%m__%H:%M:%S")
    opt.task_pool = opt.dataset_folders
    logs_folder = join(dirname(dirname(__file__)), 'outputs', 'logs', opt.task_pool)
    if opt.debug:
        logs_folder = join(logs_folder, "debug")

    opt.socket_test = False
    opt.disable_adaptive = False  # deprecated
    opt.disable_sparsity = False  # deprecated
    if opt.worker_type == 'data':
        opt.log_dir = abspath(join(logs_folder, 'data_gen', now))
    else:
        if opt.embedding_transfer:
            clusters_description = f"k_{opt.et_top_k}"
            if opt.et_use_clusters:
                clusters_description += f"_clust_{opt.et_max_cluster_centroids}"
            alphas_description = f"alphas_{opt.et_init_alphas}_atrain_{opt.et_alphas_trainable}"
            logs_folder = join(logs_folder, clusters_description, alphas_description)
        else:
            expt_description = "FedWeit"
            logs_folder = join(logs_folder, expt_description)
        opt.log_dir = abspath(join(logs_folder, now))
    opt.weights_dir = abspath(join(dirname(dirname(__file__)), 'outputs', 'weights', opt.task_pool, now))

    opt.num_examples = -1
    # opt.debug = True
    opt.fedweit = True
    # exit()

    if opt.task_type == "stl":
        opt.split_option = 'iid'
        opt.num_pick_tasks = 1
        opt.num_classes = {'TMN': 8, 'reuters8': 7, 'TREC6': 6}[opt.dataset_folders]
        # opt.num_classes = 8 if opt.dataset_folders == 'reuters8' else 7
        opt.num_clients = 1
        opt.num_rounds = 1  # 10, 5
        opt.num_epochs = 250
        if opt.debug:
            opt.num_rounds = 1  # 10, 5
            opt.num_epochs = 5
    elif opt.task_type == "federated":
        opt.num_classes = {
            'TMN': 4, 'reuters8': 4, 'TREC6': 4, 'TREC50': 4, 'subj': 2, 'polarity': 2, 'AGnews': 4
        }[opt.dataset_folders]  # 8			Determines how many classes per task
        if opt.debug:
            opt.num_pick_tasks = 2
            opt.num_rounds, opt.num_epochs = 2, 1
            opt.num_clients = 1
            opt.num_examples = 40
        else:
            # opt.num_pick_tasks = 5  # How many tasks per client 			False: (-1 if use all tasks)
            opt.num_rounds, opt.num_epochs = 10, 50
            opt.num_clients = 3

        if opt.split_option == "overlapped":
            opt.gen_num_tasks = 8 if not opt.gen_num_tasks else opt.gen_num_tasks
        elif opt.split_option == "non_iid":
            opt.gen_num_tasks = opt.num_clients * opt.num_pick_tasks if not opt.gen_num_tasks else opt.gen_num_tasks

    # opt.task_pool = ", ".join(opt.dataset_folders) if type(opt.dataset_folders) == list else opt.dataset_folders
    # If merging datasets in future, change this
    opt.mixture_filename = f'saved_mixture_{opt.task_pool}.npy'
    opt.dataset = [opt.dataset_folders]
    # print(f"dataset: {opt.dataset}")
    if opt.fedweit:
        opt.mask = True
    return opt


if __name__ == '__main__':
    main(Parser().parse())
