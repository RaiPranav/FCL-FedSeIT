import argparse
from os.path import dirname, join

ROOT_PATH = dirname(dirname(__file__))
# Setting --dense-detached disables the [shared] setting mentioned in the thesis document

class Parser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.set_basic_arguments()
        self.set_arguments()

    def set_basic_arguments(self):
        self.parser.add_argument('-w', '--worker-type', type=str, default='server', help='worker type (server, client, data)')
        self.parser.add_argument('--task-type', type=str, default="federated", help='stl or federated', nargs='*')
        self.parser.add_argument('--federated', type=bool, default=True, help='federated learning, otherwise continual learning')
        self.parser.add_argument('--continual', type=bool, default=True, help='continual learning, otherwise single task')
        self.parser.add_argument('--old-early-stopping-enabled', action='store_true',
                                 help="If True, use FedWeit's early stopping technique")
        self.parser.add_argument('--reset-current-lr', action='store_true', help="If True, reset current lr every round")
        self.parser.add_argument('--embedding-dim', type=int, default=300, help='GloVe Embedding size')

        self.parser.add_argument('--sparse-comm', type=bool, default=False, help='sparse communication')
        self.parser.add_argument('--server-sparse-comm', type=bool, default=False, help='sparse communication')
        self.parser.add_argument('--client-sparsity', type=float, default=0.7,
                                 help='sparsity ratio of client-side communication. if 0.7 then client sends 30% global weights')
        self.parser.add_argument('--sparse-broad-rate', type=float, default=0.7,
                                 help='sparsity ratio of broadcasting. if 0.7 then server sends 30% of global weights')
        self.parser.add_argument('--task-pool', type=int, default=8,
                                 help='task pool: 0 (Hetero-8), 1 (NonIID-58), 2 (Overalpped-50) ')

        self.parser.add_argument('--data-dir', type=str, default=join(ROOT_PATH, 'data', 'tasks'), help='data path')
        self.parser.add_argument('--mixture-dir', type=str, default=join(ROOT_PATH, 'data', 'mixture_loader'),
                                 help='mixture data path')
        self.parser.add_argument('-g', '--gpu', type=str, default="-1", help='gpu id')
        self.parser.add_argument('-t', '--manual-tasks', type=str, default=[], help='task name to experiment', nargs='*')
        self.parser.add_argument('-d', '--dataset-folders', type=str, default="reuters8",
                                 help='Dataset folder name under data/mixture_loader/data. Current=reuters8, TMN, TREC6')
        self.parser.add_argument('--load-weights', type=bool, default=False, help='True, if want to load global weights')
        self.parser.add_argument('--load-weights-dir', type=str, default='', help='load weights path')
        self.parser.add_argument('--disable-mask', dest="mask", action='store_false',
                                 help='If True, use shared_weights * mask for theta calc')

        self.parser.add_argument('--model', type=int, default=3, help='model: 0(L2T), 1(APD), 2(ABC) 3(Kim et al) 4(ours)')
        self.parser.add_argument('--base-architect', type=int, default=2, help='architecture: 0(LeNet), 1(AlexNet) 2(Kim et al)')
        self.parser.add_argument('--trainable', action='store_true', help='Embedding layer trainable')
        self.parser.add_argument('--batch-size', type=int, default=64, help='batch size')
        # self.parser.add_argument('--maxlen', type=int, default=60, help='Max length for cnn')

    def set_arguments(self):
        self.parser.add_argument('--fedweit', type=bool, default=True, help='If True, use adaptive weights from other clients')
        self.parser.add_argument('--split-option', type=str, default="overlapped",
                                 help='What kind of tasks to split out of dataset(s) iid / non_iid / overlapped')

        self.parser.add_argument('--embedding-transfer', action='store_true',
                                 help='If True, use Embedding transfer using task vectors')
        self.parser.add_argument('--et-top-k', type=int, default=0,
                                 help='If using Embedding transfer, how many top adaptive matrices to send. Must be specified')
        self.parser.add_argument('--et-init-alphas', action='store_true',
                                 help='If using Embedding Transfer, initialise attention alphas to the Task-Task sim score')
        self.parser.add_argument('--et-alphas-trainable', action='store_true',
                                 help='If using Embedding Transfer, and True, set alphas as trainable always')
        self.parser.add_argument('--et-task-similarity-method', type=str, default="rectified_linear",
                                 help='If using Embedding Transfer, how to compare task-task similarity.'
                                      'rectified_linear / bhatt / linear / step')
        self.parser.add_argument('--et-use-clusters', action='store_true',
                                 help="If using ET, and set to true, use cluster centroids instead of full Doc Embeddings")
        self.parser.add_argument('--et-max-cluster-centroids', type=int, default=200,
                                 help="If using ET with clusters, set cluster size for while clustering")
        self.parser.add_argument('--et-clustering-algorithm', type=str, default="kmedoids", help='kmeans / kmedoids')

        self.parser.add_argument('--debug', action='store_true', help='If True, run on fewer epochs')
        self.parser.add_argument('--save-weights', type=bool, default=False, help='True, if want to save global weights')
        self.parser.add_argument('--log-data-mlflow', action='store_true',
                                 help='If True, log npy data files to mlflow (~120 Mb for R8_4_8)')
        self.parser.add_argument('--host-port', type=int, default=5023, help='host port num')

        self.parser.add_argument('--fed-method', type=int, default=0, help='0(FedAvg), 1(FedProx)')
        self.parser.add_argument('--num-clients', type=int, default=3, help='number of clients')
        self.parser.add_argument('--frac-clients', type=float, default=1, help='fraction of clients per round')
        self.parser.add_argument('--num-rounds', type=int, default=2, help='number of rounds')
        self.parser.add_argument('--num-epochs', type=int, default=5, help='number of epochs')
        self.parser.add_argument('--gen-num-tasks', type=int, help='number of tasks per client')
        self.parser.add_argument('--num-pick-tasks', type=int, default=5, help='Num of tasks done per client')

        self.parser.add_argument('--concatenate-aw-kbs', action='store_true',
                                 help='If True, concatenate incoming adaptive weights instead of alpha-weighted sum')
        # project-adaptives -> Add a detached dense layer after ONLY the foreign adaptives processing
        self.parser.add_argument('--project-adaptives', action='store_true',
                                 help='If True, apply a linear / dense projection on concated/added adaptives')

        self.parser.add_argument('--disable-alphas', action='store_true', help='If True, set alphas to 0 and non trainable')
        self.parser.add_argument('--lambda-two', dest="approx_hyp", type=float,
                                 help='Lambda2 of FedWeit eqn. [catastrophic forgetting]')
        self.parser.add_argument('--adaptive-random', action='store_true',
                                 help='If True, randomly init the adpative weights. Else, base it on shared weights '
                                      '(as in FedWeit)')
        # dense-detached -> ALL dense layers are detached
        self.parser.add_argument('--dense-detached', action='store_true',
                                 help='If True, the dense layer weights (only model 5 for now) will not be sent to the server. '
                                      'This will be a simple/default/normal dense layer!')
        self.parser.add_argument('--fedweit-dense', action='store_true',
                                 help='If True, a dense layer added to the end of FedWeIT')
        self.parser.add_argument('--foreign-adaptives-trainable', action='store_true',
                                 help="If True, other tasks' adaptives (with attention) are trainable")

        self.parser.add_argument('--converge-first-round-epochs', type=int, default=None,
                                 help="If provided with a number, each task's first round will first run for this many epochs"
                                      "with adaptives disabled, after which optimiser will be reset. Next rounds are normal")
        self.parser.add_argument('--exhaust-tasks', action='store_true',
                                 help='If True, first try to exhaust all available tasks before repeating')

        self.parser.add_argument('--random-seed', type=int, default=1, help='Global random seed')
        self.parser.add_argument('--random-seed-task-alloc', type=int, default=1, help='Global random seed')
        self.parser.add_argument('--random-seed-kmeans', type=int, default=42, help='Global random seed')
        self.parser.add_argument('--random-seed-data-gen', type=int, default=42, help='Global random seed')

    def parse(self):
        args, unparsed = self.parser.parse_known_args()
        if len(unparsed) != 0:
            raise Exception(f'Unknown argument: {unparsed}')
        args.host_ip = '0.0.0.0'
        return args
