import tensorflow as tf
from FedWeit.models.global_model import GlobalModel


class GlobalAPD(GlobalModel):

    def __init__(self, opt):
        super(GlobalAPD, self).__init__(opt)
        self.opt = opt
        self.current_task = 0
        self.input_shape = (32, 32, 3)
        if self.opt.base_architect == 0:
            self.shapes = [
                (5, 5, 3, 20),
                (5, 5, 20, 50),
                (3200, 800),
                (800, 500),
            ]
        elif self.opt.base_architect == 1:
            self.shapes = [
                (4, 4, 3, 64),
                (3, 3, 64, 128),
                (2, 2, 128, 256),
                (4096, 512),
                (512, 512),
            ]
        self.initialize_weights()

    def initialize_weights(self):
        self.weights = []
        self.initializer = tf.keras.initializers.VarianceScaling(seed=self.opt.random_seed)
        for i in range(len(self.shapes)):
            self.weights.append(self.initializer(self.shapes[i]).numpy().tolist())

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights

    def update_weights(self, responses):
        client_weights = [resp['client_weights'] for resp in responses]
        client_sizes = [resp['train_size'] for resp in responses]
        client_masks = [resp['client_masks'] for resp in responses] if self.opt.sparse_comm else []
        self.apply_federated_average(client_weights, client_sizes, client_masks)
        self.calculate_comm_costs(self.get_weights())

    def get_info(self):
        return {
            'shapes': self.shapes,
            'input_shape': self.input_shape,
            'shared_params': self.weights
        }
