class Config:
    def __init__(self, opt):
        self.opt = opt
        if self.opt.federated:
            if self.opt.model == 0:  # FedAvg L2T
                if self.opt.base_architect == 0:  # LeNet
                    self.fedavg_l2t_lenet()
                elif self.opt.base_architect == 1:  # AlexNet
                    self.fedavg_l2t_alexnet()
            elif self.opt.model == 1:  # FedAvg APD
                if self.opt.base_architect == 0:  # LeNet
                    if self.opt.sparse_comm:
                        self.fedavg_apd_lenet_sparse_comm()
                    else:
                        self.fedavg_apd_lenet()
                elif self.opt.base_architect == 1:  # AlexNet
                    if self.opt.sparse_comm:
                        self.fedavg_adp_alexnet_sparse_comm()
                    else:
                        self.fedavg_adp_alexnet()
            elif self.opt.model == 2:  # abc-l2t
                self.fedavg_abc_l2t_lenet()
            elif self.opt.model == 3:  # abc-apd
                # Original default
                self.fedavg_abc_apd_kim()
            elif self.opt.model == 4:  # EWC
                self.fedavg_l2t_lenet()
            elif self.opt.model == 5:  # EWC
                self.opt.base_architect = 3
                self.fedavg_abc_apd_parallel()
        elif self.opt.continual:
            if self.opt.model == 0:  # CNN
                if self.opt.base_architect == 0:  # LeNet
                    self.lenet()
                elif self.opt.base_architect == 1:  # AlexNet
                    self.alexnet()
            elif self.opt.model == 1:  # APD
                if self.opt.base_architect == 0:  # LeNet
                    self.apd_lenet()
                elif self.opt.base_architect == 1:  # AlexNet
                    self.apd_alexnet()
            elif self.opt.model == 4:  # EWC
                self.lenet()
        else:  # Single Task Learning
            if self.opt.model == 0:
                if self.opt.base_architect == 0:  # LeNet
                    self.lenet()
                elif self.opt.base_architect == 1:  # AlexNet
                    self.alexnet()
            elif self.opt.model == 1:
                if self.opt.base_architect == 0:
                    self.apd_lenet()
                elif self.opt.base_architect == 1:
                    self.apd_alexnet()
            else:
                pass

    def get_options(self):
        return self.opt

    # Original Default
    def fedavg_abc_apd_kim(self):
        # self.opt.server_sparse      = False
        self.opt.dropout = 0.3
        self.opt.lr = 1e-4
        self.opt.lr_patience = 3
        self.opt.lr_factor = 3
        self.opt.lr_min = 1e-10
        self.opt.wd_rate = 1e-4
        self.opt.l1_hyp = 1e-3
        self.opt.mask_hyp = 0
        self.opt.l1_mask_hyp = 4e-4
        self.opt.e_gap_hyp = 1e-2
        if not self.opt.approx_hyp:
            self.opt.approx_hyp = 100.
        self.opt.k_centroides = 2
        self.opt.clustering_iter = 5  # task
        self.opt.is_hierarchy = False

    # Our Default
    def fedavg_abc_apd_parallel(self):
        # self.opt.server_sparse      = False
        self.opt.dropout = 0.3
        self.opt.lr = 1e-4
        self.opt.lr_patience = 3
        self.opt.lr_factor = 3
        self.opt.lr_min = 1e-10
        self.opt.wd_rate = 1e-4
        self.opt.l1_hyp = 1e-3
        self.opt.mask_hyp = 0
        self.opt.l1_mask_hyp = 4e-4
        self.opt.e_gap_hyp = 1e-2
        if not self.opt.approx_hyp:
            self.opt.approx_hyp = 100.
        self.opt.k_centroides = 2
        self.opt.clustering_iter = 5  # task
        self.opt.is_hierarchy = False

        self.opt.foreign_adaptives_trainable_drift_constant = 1e-1

    def fedavg_abc_l2t_lenet(self):
        self.opt.lr = 1e-4
        self.opt.lr_patience = 3
        self.opt.lr_factor = 3
        self.opt.lr_min = 1e-10
        self.opt.l1_hyp = 2e-3
        self.opt.l1_mask_hyp = 4e-4

    def lenet(self):
        print('config: lenet')
        self.opt.lr = 1e-4
        self.opt.lr_patience = 3
        self.opt.lr_factor = 3
        self.opt.lr_min = 1e-10
        self.opt.l2_lambda = 1e-2

    def apd_lenet(self):
        print('config: apd_lenet')
        self.opt.lr = 1e-4
        self.opt.lr_patience = 3
        self.opt.lr_factor = 3
        self.opt.lr_min = 1e-10
        self.opt.wd_rate = 1e-4
        self.opt.l1_hyp = 4e-3
        self.opt.e_gap_hyp = 1e-2
        self.opt.mask_hyp = 0
        self.opt.approx_hyp = 100.
        self.opt.k_centroides = 2
        self.opt.clustering_iter = 5  # task
        self.opt.is_hierarchy = False

    def fedavg_l2t_lenet(self):
        print('config: fedavg_l2t_lenet')
        self.opt.lr = 1e-4
        self.opt.lr_patience = 3
        self.opt.lr_factor = 3
        self.opt.lr_min = 1e-10
        self.opt.l2_lambda = 1e-2

    def fedavg_lenet_sparse_comm(self):
        print('config: fedavg_lenet_sparse_comm')
        self.opt.lr = 1e-4
        self.opt.lr_patience = 3
        self.opt.lr_factor = 3
        self.opt.lr_min = 1e-10
        self.opt.wd_rate = 1e-4
        self.opt.l1_hyp = 1e-3
        self.opt.mask_hyp = 0
        self.opt.l1_mask_hyp = 4e-4
        self.opt.e_gap_hyp = 1e-2
        self.opt.approx_hyp = 100.
        self.opt.k_centroides = 2
        self.opt.clustering_iter = 5  # task
        self.opt.is_hierarchy = False

    def fedavg_apd_lenet(self):
        print('config: fedavg_apd_lenet')
        self.opt.lr = 1e-4
        self.opt.lr_patience = 3
        self.opt.lr_factor = 3
        self.opt.lr_min = 1e-10
        self.opt.wd_rate = 1e-4
        self.opt.l1_hyp = 1e-3
        self.opt.e_gap_hyp = 1e-2
        self.opt.mask_hyp = 0
        self.opt.approx_hyp = 100.
        self.opt.k_centroides = 2
        self.opt.clustering_iter = 5  # task
        self.opt.is_hierarchy = False

    def fedavg_apd_lenet_sparse_comm(self):
        print('config: fedavg_apd_lenet_sparse_comm')
        self.opt.lr = 1e-4
        self.opt.lr_patience = 3
        self.opt.lr_factor = 3
        self.opt.lr_min = 1e-10
        self.opt.wd_rate = 1e-4
        self.opt.l1_hyp = 1e-3
        self.opt.mask_hyp = 0
        self.opt.l1_mask_hyp = 4e-4
        self.opt.e_gap_hyp = 1e-2
        self.opt.approx_hyp = 100.
        self.opt.k_centroides = 2
        self.opt.clustering_iter = 5  # task
        self.opt.is_hierarchy = False

    def alexnet(self):
        print('config: alexnet')
        self.opt.lr = 1e-4
        self.opt.lr_patience = 3
        self.opt.lr_factor = 3
        self.opt.lr_min = 1e-10
        self.opt.l2_lambda = 1e-2

    def apd_alexnet(self):
        print('config: apd_alexnet')
        self.opt.lr = 1e-4
        self.opt.lr_patience = 3
        self.opt.lr_factor = 3
        self.opt.lr_min = 1e-10
        self.opt.wd_rate = 1e-4
        self.opt.l1_hyp = 1e-3
        self.opt.e_gap_hyp = 1e-2
        self.opt.mask_hyp = 0
        self.opt.approx_hyp = 100.
        self.opt.k_centroides = 2
        self.opt.clustering_iter = 5  # task
        self.opt.is_hierarchy = False

    def fedavg_l2t_alexnet(self):
        print('config: fedavg_l2t_alexnet')
        self.opt.lr = 1e-4
        self.opt.lr_patience = 3
        self.opt.lr_factor = 3
        self.opt.lr_min = 1e-10
        self.opt.l2_lambda = 1e-2

    def fedavg_adp_alexnet(self):
        print('config: fedavg_adp_alexnet')
        self.opt.lr = 1e-4
        self.opt.lr_patience = 3
        self.opt.lr_factor = 3
        self.opt.lr_min = 1e-10
        self.opt.wd_rate = 1e-4
        self.opt.l1_hyp = 1e-3
        self.opt.e_gap_hyp = 1e-2
        self.opt.mask_hyp = 0
        self.opt.approx_hyp = 100.
        self.opt.k_centroides = 2
        self.opt.clustering_iter = 5  # task
        self.opt.is_hierarchy = False

    def fedavg_adp_alexnet_sparse_comm(self):
        print('config: fedavg_adp_alexnet_sparse_comm')
        self.opt.lr = 1e-3 / 3
        self.opt.lr_patience = 3
        self.opt.lr_factor = 3
        self.opt.lr_min = 1e-10
        self.opt.wd_rate = 1e-4
        self.opt.l1_hyp = 1e-3
        self.opt.mask_hyp = 0
        self.opt.l1_mask_hyp = 4e-4
        self.opt.e_gap_hyp = 1e-2
        self.opt.approx_hyp = 100.
        self.opt.k_centroides = 2
        self.opt.clustering_iter = 5  # task
        self.opt.is_hierarchy = False
