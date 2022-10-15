import numpy as np
# from copy import deepcopy
# from tqdm import tqdm
from os.path import join, exists
from os import makedirs

# import torch
from sklearn.cluster import KMeans
# from sklearn_extra.cluster import KMedoids
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import cosine

import matplotlib.pyplot as plt


def analyse_dataset(dataset):
    return

def print_model_report(model):
    def human_format(num):
        magnitude = 0
        while abs(num) >= 1000:
            magnitude += 1
            num /= 1000.0
        return '%.1f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])

    print('-' * 100)
    print(model)
    print('Dimensions =', end=' ')
    count = 0
    for p in model.parameters():
        print(p.size(), end=' ')
        count += np.prod(p.size())
    print()
    print('Num parameters = %s' % (human_format(count)))
    print('-' * 100)
    return count


def plot_diagram(data, diagram_name: str, plotting_model_name_graph: str = "pca",
                 n_components_graph: int = 3, colours: list = None, vector_info_for_graph: list = None,
                 save_folder: str = ""):
    colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'] if not colours else colours
    if plotting_model_name_graph == "tsne":
        plotting_model = TSNE(n_components_graph, random_state=5212)
    elif plotting_model_name_graph == "pca":
        plotting_model = PCA(n_components=n_components_graph, random_state=5212)
    components_all = plotting_model.fit_transform(data)

    fig = plt.figure()
    pca_x, pca_y = [tup[0] for tup in components_all], [tup[1] for tup in components_all]
    if n_components_graph == 3:
        pca_z = [tup[2] for tup in components_all]
        ax = fig.add_subplot(projection='3d')
        ax.scatter(pca_x, pca_y, pca_z, color=colours)
    else:
        ax = fig.add_subplot()
        ax.set_aspect('equal')
        ax.scatter(pca_x, pca_y, color=colours)
    if vector_info_for_graph:
        for i, label in enumerate(vector_info_for_graph):
            plt.annotate(label, (pca_x[i], pca_y[i]))

    if plotting_model_name_graph == "tsne":
        labels = [f"TSNE_{i + 1}" for i in range(n_components_graph)]
    else:
        labels = [f"PCA_{i + 1}_{round(plotting_model.explained_variance_ratio_[i] * 100, 5)}"
                  for i in range(n_components_graph)]
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    if n_components_graph == 3:
        ax.set_zlabel(labels[2])
    # ax.legend()
    if not exists(save_folder):
        makedirs(save_folder)
    fig.savefig(join(save_folder, f"{diagram_name}.png"))


def cluster(opt, data, k: int = 200):
    # k = min(k, len(data))
    if len(data) < 200:
        return data

    km = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=3, verbose=1, random_state=opt.random_seed_kmeans)
    km.fit(data)

    if opt.et_clustering_algorithm == "kmeans":
        centres = km.cluster_centers_
    else:
        # km = KMedoids(n_clusters=k, init='k-medoids++', max_iter=100, random_state=opt.random_seed)
        # km.fit(data)
        # centres = km.cluster_centers_
        centres, medoid_indices = [], []
        for centre_index, centre in enumerate(km.cluster_centers_):
            cluster_member_indices = np.where(km.labels_ == centre_index)
            cluster_members_data = data[cluster_member_indices]

            distances = np.array([1 - cosine(centre, doc) for doc in cluster_members_data])
            try:
                centres.append(cluster_members_data[np.argmin(distances)])
            except ValueError as err:
                # This cluster has no members. <k clusters will return
                pass
            # medoid_indices.append(np.argmin(distances))
            # medoid = data[medoid_indices[-1]]
            # centres.append(medoid)
        centres = np.array(centres)
    return centres


########################################################################################################################

#
# def get_model(model):
#     return deepcopy(model.state_dict())
#
#
# def set_model_(model, state_dict):
#     model.load_state_dict(deepcopy(state_dict))
#     return
#
#
# def freeze_model(model):
#     for param in model.parameters():
#         param.requires_grad = False
#     return


########################################################################################################################


# def compute_conv_output_size(Lin, kernel_size, stride=1, padding=0, dilation=1):
#     return int(np.floor((Lin + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1))


########################################################################################################################


# def compute_mean_std_dataset(dataset):
#     # dataset already put ToTensor
#     mean = 0
#     std = 0
#     loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
#     for image, _ in loader:
#         mean += image.mean(3).mean(2)
#     mean /= len(dataset)
#
#     mean_expanded = mean.view(mean.size(0), mean.size(1), 1, 1).expand_as(image)
#     for image, _ in loader:
#         std += (image - mean_expanded).pow(2).sum(3).sum(2)
#
#     std = (std / (len(dataset) * image.size(2) * image.size(3) - 1)).sqrt()
#
#     return mean, std


########################################################################################################################


# def fisher_matrix_diag(t, x, y, model, criterion, sbatch=20):
#     # Init
#     fisher = {}
#     for n, p in model.named_parameters():
#         fisher[n] = 0 * p.data
#     # Compute
#     model.train()
#     for i in tqdm(range(0, x.size(0), sbatch), desc='Fisher diagonal', ncols=100, ascii=True):
#         b = torch.LongTensor(np.arange(i, np.min([i + sbatch, x.size(0)])))
#         # b=torch.LongTensor(np.arange(i,np.min([i+sbatch,x.size(0)]))).cuda()
#         images = torch.autograd.Variable(x[b], volatile=False)
#         target = torch.autograd.Variable(y[b], volatile=False)
#         # Forward and backward
#         model.zero_grad()
#         outputs = model.forward(images)
#         loss = criterion(t, outputs[t], target)
#         loss.backward()
#         # Get gradients
#         for n, p in model.named_parameters():
#             if p.grad is not None:
#                 fisher[n] += sbatch * p.grad.data.pow(2)
#     # Mean
#     for n, _ in model.named_parameters():
#         fisher[n] = fisher[n] / x.size(0)
#         fisher[n] = torch.autograd.Variable(fisher[n], requires_grad=False)
#     return fisher


########################################################################################################################

# def cross_entropy(outputs, targets, exp=1, size_average=True, eps=1e-5):
#     out = torch.nn.functional.softmax(outputs)
#     tar = torch.nn.functional.softmax(targets)
#     if exp != 1:
#         out = out.pow(exp)
#         out = out / out.sum(1).view(-1, 1).expand_as(out)
#         tar = tar.pow(exp)
#         tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)
#     out = out + eps / out.size(1)
#     out = out / out.sum(1).view(-1, 1).expand_as(out)
#     ce = -(tar * out.log()).sum(1)
#     if size_average:
#         ce = ce.mean()
#     return ce


########################################################################################################################


# def set_req_grad(layer, req_grad):
#     if hasattr(layer, 'weight'):
#         layer.weight.requires_grad = req_grad
#     if hasattr(layer, 'bias'):
#         layer.bias.requires_grad = req_grad
#     return


########################################################################################################################
