import time
import multiprocessing
import logging
import shutil
from os.path import abspath, join

from FedWeit.utils import *
from FedWeit.modules.server import Server
from FedWeit.modules.client import Client
from FedWeit.modules.config import Config
from FedWeit.data.generator import DataGenerator


class Launcher:
    def __init__(self, args):
        self.opt = args
        self.processes = []

    def run_server(self):
        config = Config(self.opt)
        self.opt = config.get_options()
        self.server = Server(self.opt)
        self.server.run()

    def run_clients(self):
        # logger.info("run_clients")
        if self.opt.gpu == '-1':
            for cid in range(self.opt.num_clients):
                config = Config(self.opt)
                self.opt = config.get_options()
                self.spawn_client_process()
                time.sleep(20)
        else:
            config = Config(self.opt)
            self.opt = config.get_options()
            Client(self.opt)
            # self.spawn_client_process()
        # atexit.register(self.kill_processes)

    def spawn_client_process(self):
        p = multiprocessing.Process(target=self.init_client, args=())
        p.start()
        self.processes.append(p)
        print('client started with pid %d' % (p.pid))

    def init_client(self):
        Client(self.opt)

    def kill_processes(self):
        for p in self.processes:
            pid = p.pid
            p.terminate()
            print('[notice] pid: %d has been terminated' % (pid))

    def generate_data(self):
        start_time = time.time()

        logger = logging.getLogger(f"data_generator")
        formatter_client = logging.Formatter('%(asctime)s : %(message)s')
        fileHandler_ = logging.FileHandler(f"{self.opt.log_dir}/data_generator.log", mode='w')
        fileHandler_.setFormatter(formatter_client)
        logger.setLevel(logging.INFO)
        logger.addHandler(fileHandler_)
        streamHandler_client = logging.StreamHandler()
        streamHandler_client.setFormatter(formatter_client)
        logger.addHandler(streamHandler_client)

        data_generator = DataGenerator(-1, self.opt, logger, tasks_so_far=set())
        data_generator.generate_data()
        data_info = data_generator.get_info()

        write_file(self.opt.log_dir, "opt_data.json", vars(self.opt))
        write_file(data_generator.base_dir, "opt_data.json", vars(self.opt))
        shutil.copyfile(abspath(join(self.opt.log_dir, "data_generator.log")),
                        abspath(join(data_generator.base_dir, "data_generator.log")))
        syslog(-1, 'done (%d sec)' % (time.time() - start_time), logger)
