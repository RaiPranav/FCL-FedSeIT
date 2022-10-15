import os
import yaml
# import numpy as np
# import tensorflow as tf
# from math import sqrt
from os.path import abspath, dirname, join, exists
from os import makedirs
import matplotlib.pyplot as plt


# from tensorflow.python.keras.metrics import Precision, Recall
# from scipy.stats import gaussian_kde


def plot_learning_curve(tasks_names, performance_json: dict, title: str, x_axis_name: str, y_axis_name: str, save_folder: str):
    fig = plt.figure()
    ax = fig.add_subplot()
    # ax.set_aspect('equal')
    ax.grid()

    ax.set_xlabel(x_axis_name)
    ax.set_ylabel(y_axis_name)
    ax.set_title(title)

    colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    colours = colours * 3
    total_epochs = 0
    markersize, linewidth = 3, 1.3
    for task_num, task_name, colour in zip(sorted(performance_json.keys()), tasks_names, colours):
        values_all = performance_json[task_num].values()
        values_all = [value for values_round in values_all for value in values_round]
        if task_num == 0:
            total_epochs = len(values_all)
        x_coordinates = list(range(total_epochs - len(values_all) + 1, total_epochs + 1))

        ax.plot(x_coordinates, values_all, 'o-', color=colour, label=task_name, markersize=markersize, linewidth=linewidth)
        # ax.plot(x_coordinates, values_all, 'o-', color=colour, label=task_name)

    ax.legend(loc="best")
    if not exists(save_folder):
        makedirs(save_folder)

    fig.savefig(f"{join(save_folder, title)}.png")
    print(f"Figure saved to {join(save_folder, title)}.png")


def fix_mlflow_uris(opt):
    mlflow_dir = abspath(dirname(dirname(dirname(__file__))))

    try:
        expts = os.listdir(join(mlflow_dir, "mlruns"))
    except FileNotFoundError:
        return
    if ".trash" in expts:
        expts.remove(".trash")
    for folder in expts:
        expt_folder = join(mlflow_dir, "mlruns", folder)
        with open(join(expt_folder, 'meta.yaml'), 'r') as f:
            expt_config = yaml.load(f, Loader=yaml.FullLoader)
        expt_config['artifact_location'] = "file://" + expt_folder
        with open(join(expt_folder, 'meta.yaml'), 'w') as f:
            f.write(yaml.dump(expt_config))

        run_folders = os.listdir(expt_folder)
        run_folders.remove("meta.yaml")
        for run_folder in run_folders:
            run_folder_meta_file = join(expt_folder, run_folder, "meta.yaml")
            with open(run_folder_meta_file, 'r') as f:
                run_config = yaml.load(f, Loader=yaml.FullLoader)
            run_config['artifact_uri'] = expt_config['artifact_location'] + "/" + \
                                         run_config['artifact_uri'].split("/")[-2] + "/artifacts"
            with open(run_folder_meta_file, 'w') as f:
                f.write(yaml.dump(run_config))


def get_conv_fc_layers(shape):
    conv_layers = []
    fc_layers = []
    for layer_idx, layer_shape in enumerate(shape):
        if len(layer_shape) == 3:
            conv_layers.append(layer_idx)
        else:
            fc_layers.append(layer_idx)
    return conv_layers, fc_layers


def process_confusion_matrix(cm):
    precision_decimal = 4

    total_instances = sum(sum(cm[0]))
    classwise_acc, classwise_f1 = [], []
    classwise_prec, classwise_recall = [], []
    for class_idx in range(cm.shape[0]):
        tn, tp = cm[class_idx][1][1], cm[class_idx][0][0]
        fp, fn = cm[class_idx][0][1], cm[class_idx][1][0]
        prec, recall = tp / (tp + fp), tp / (tp + fn)

        classwise_acc.append(round(float((tp + tn) / total_instances), precision_decimal))

        classwise_prec.append(prec)
        classwise_recall.append(recall)
        classwise_f1.append(round(float(2 * prec * recall / (prec + recall)), precision_decimal))
    # macro_prec = sum(classwise_prec) / len(classwise_prec)
    # macro_recall = sum(classwise_recall) / len(classwise_recall)
    # print("macro F1", 2 * macro_prec * macro_recall / (macro_prec + macro_recall))
    return classwise_acc, classwise_f1

# class F1_Score(tf.keras.metrics.Metric):
#     def __init__(self, name='f1_score', **kwargs):
#         super().__init__(name=name, **kwargs)
#         self.f1 = self.add_weight(name='f1', initializer='zeros')
#         self.precision_fn = Precision(thresholds=0.5)
#         self.recall_fn = Recall(thresholds=0.5)
#
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         p = self.precision_fn(y_true, y_pred)
#         r = self.recall_fn(y_true, y_pred)
#         # since f1 is a variable, we use assign
#         self.f1.assign(2 * ((p * r) / (p + r + 1e-6)))
#
#     def result(self):
#         return self.f1
#
#     def reset_states(self):
#         # we also need to reset the state of the precision and recall objects
#         self.precision_fn.reset_states()
#         self.recall_fn.reset_states()
#         self.f1.assign(0)
#
# def bhatta_dist(X1, X2, method='continuous'):
#     #Calculate the Bhattacharyya distance between X1 and X2. X1 and X2 should be 1D numpy arrays representing the same
#     # feature in two separate classes.
#     def get_density(x, cov_factor=0.1):
#         #Produces a continuous density function for the data in 'x'. Some benefit may be gained from adjusting the cov_factor.
#         density = gaussian_kde(x)
#         density.covariance_factor = lambda:cov_factor
#         density._compute_covariance()
#         return density
#     #Combine X1 and X2, we'll use it later:
#     cX = np.concatenate((X1,X2))
#     if method == 'continuous':
#         ###Use a continuous density function to calculate the coefficient (This is the most consistent, but also slightly slow):
#         N_STEPS = 200
#         #Get density functions:
#         d1 = get_density(X1)
#         d2 = get_density(X2)
#         #Calc coeff:
#         xs = np.linspace(min(cX),max(cX),N_STEPS)
#         bht = 0
#         for x in xs:
#             p1 = d1(x)
#             p2 = d2(x)
#             bht += sqrt(p1*p2)*(max(cX)-min(cX))/N_STEPS
#     else:
#         raise ValueError("The value of the 'method' parameter does not match any known method")
#
#     ###Lastly, convert the coefficient into distance:
#     if bht==0:
#         bht = float('Inf')
#     else:
#         bht = -np.log(bht)
#     return bht
#
#
# def bhattacharya_distance(x1, x2):
#     """
#
#     @param x1:
#     @param x2:
#     @param method:
#     @return:
#     """
#     x1, x2 = x1 * 100, x2 * 100
#     u1 = np.mean(x1, axis=0)
#     u2 = np.mean(x2, axis=0)
#     mean_difference = u1 - u2
#
#     sigma1 = np.cov(np.transpose(x1))
#     sigma2 = np.cov(np.transpose(x2))
#     sigma = (sigma1 + sigma2) / 2
#     # sigma2=sigma1 + np.random.multivariate_normal([0, 0, 0, 0], np.identity(4) / 10000, (4, 4))[0]
#
#     det_1 = np.linalg.det(sigma1)
#     det_2 = np.linalg.det(sigma2)
#     det = np.linalg.det(sigma)
#
#     # epsilon = 1e-15
#     # db_ = np.log(det * det / (det_1 * det_2)) / 4
#     db_ = np.log(det / (np.sqrt(det_1 * det_2))) / 2
#     # db_ = np.sqrt(det_1) * np.sqrt(det_2)
#     # db_ = np.log(det / db_) / 2
#     # if np.isnan(db_):
#     #     db_ = epsilon
#     # if np.isnan(db_):
#     #     epsilon = 1e-13
#     #     db_ = np.log((epsilon + det) / (np.sqrt(epsilon + det_1) * np.sqrt(epsilon + det_2))) / 2
#
#     db = np.transpose(mean_difference).dot(np.linalg.inv(sigma))
#     db = db.dot(mean_difference) / 8
#
#     return db + db_
