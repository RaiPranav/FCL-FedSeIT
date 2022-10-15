import glob
from os.path import join
from typing import Set

from sklearn.model_selection import train_test_split
from sklearn import metrics
from scipy import spatial
import tensorflow as tf

from FedWeit.utils import *
from FedWeit.data.utils import cluster, plot_diagram
from FedWeit.data.mixture import get_mixture


class DataGenerator:
    def __init__(self, client_id, opt, logger, tasks_so_far: Set[str]):
        self.opt = opt
        self.client_id = client_id  # 0, 1, ...
        # self.train_valid_test_splits = (0.7, 0.1, 0.2)  # train, test.py, valid
        if self.opt.split_option == 'non_iid' or self.opt.split_option == 'overlapped':
            self.base_dir = join(self.opt.data_dir, self.opt.split_option,
                                 f'num_classes_{self.opt.num_classes}_num_tasks_{self.opt.gen_num_tasks}')
        elif self.opt.split_option == 'iid':
            self.base_dir = join(self.opt.data_dir, self.opt.split_option, 'num_clients_' + str(self.opt.num_clients))

        self.logger = logger
        self.info = {
            'num_tasks': 0,
            'datasets': ', '.join(self.opt.dataset)
        }
        if self.client_id > -1:
            self.tasks = []
            if len(self.opt.manual_tasks) > 0:
                self.logger.info(f"Manual Tasks")
                self.tasks = self.opt.manual_tasks  # [task + '.npy' for task in self.opt.manual_tasks]
                syslog(self.client_id, 'tasks are manually set: {}'.format(', '.join(self.opt.manual_tasks)),
                       self.logger)
            else:
                if self.opt.split_option == 'non_iid' or self.opt.split_option == 'overlapped':
                    self.logger.info(f"split_option: {self.opt.split_option}")
                    # candidate_tasks = []
                    for dataset in self.opt.dataset:
                        path = join(self.base_dir, dataset + '_*')
                        self.tasks += [os.path.basename(p) for p in glob.glob(path) if not p.endswith(".json")]
                    self.tasks = sorted(self.tasks)
                    self.logger.info(f"Full tasks: {self.tasks}")
                    random_shuffle(self.client_id * self.opt.random_seed_task_alloc + self.opt.num_pick_tasks, self.tasks)
                    if self.opt.exhaust_tasks:
                        new_tasks = [task for task in self.tasks if task not in tasks_so_far][: self.opt.num_pick_tasks]
                        more_tasks_needed = max(0, self.opt.num_pick_tasks - len(new_tasks))
                        new_tasks += [task for task in self.tasks if task not in new_tasks][: more_tasks_needed]
                        self.tasks = new_tasks
                    else:
                        self.tasks = self.tasks[: self.opt.num_pick_tasks]
                    self.logger.info(f"cid {self.client_id} | tasks {self.tasks}")
                elif self.opt.split_option == 'iid':
                    for dataset in self.opt.dataset:
                        path = join(self.base_dir, dataset + '_' + str(self.client_id) + '*')
                        self.tasks += [os.path.basename(p) for p in glob.glob(path) if not p.endswith(".json")]
                    self.logger.info(f"Tasks before shuffling {self.client_id} tasks: {self.tasks} Path: {path}")
                    random_shuffle(self.client_id + self.opt.random_seed_task_alloc, self.tasks)
                self.logger.info(f"ClientTasks {self.client_id} tasks: {self.tasks}")
            self.logger.info(f"self.tasks length final: {len(self.tasks)}")
            self.info['num_tasks'] = len(self.tasks)

    def get_info(self):
        return self.info

    def get_tasks(self):
        return self.tasks

    def generate_data(self):
        saved_mixture_filepath = join(self.opt.mixture_dir, 'saved', self.opt.mixture_filename)
        syslog(self.client_id, f'saved_mixture_filepath: {saved_mixture_filepath}', self.logger)

        if not os.path.exists(saved_mixture_filepath):
            syslog(self.client_id, 'generating & processing mixture data', self.logger)
            mixture = get_mixture(opt=self.opt, base_dir=self.opt.mixture_dir, fixed_order=True)
            np_save(join(self.opt.mixture_dir, 'saved'), self.opt.mixture_filename, mixture)
        else:
            syslog(self.client_id, f'loading mixture data: {saved_mixture_filepath}', self.logger)
            mixture = np.load(saved_mixture_filepath, allow_pickle=True)
        self.generate_tasks(mixture)

    def generate_tasks(self, mixture):
        syslog(self.client_id, 'generating tasks with the given options', self.logger)
        self.task_cnt = -1

        syslog(self.client_id, f'self.opt.dataset: {self.opt.dataset}', self.logger)
        for d in self.opt.dataset:
            mixture_index = list(mixture[0].keys()).index(d)
            self._generate_tasks(d, mixture[0][d], mixture[2][mixture_index])

    def _generate_tasks(self, dataset, data, maxlen_dataset: int):
        x_train = np.array(data['train']['x'])
        y_train = np.array(data['train']['y'])
        doc_embeddings_all_train = np.array(data['train']['x_vector'])
        x_test = np.array(data['test']['x'])
        y_test = np.array(data['test']['y'])
        # vector_test = np.array(data['test.py']['x_vector'])
        labels_names = data['labels']
        del data

        labels = np.unique(y_test)
        labels_names_dict = {label: label_name for label, label_name in zip(labels, labels_names)}
        random_shuffle(self.opt.random_seed_data_gen, labels, labels_names)
        # train_label_to_x, test_label_to_x = {}, {}
        train_indices, test_indices = {}, {}
        doc_embeddings_info, doc_embeddings_all_tasks = [], []
        cluster_centres_all_tasks = []
        for label in labels:
            train_indices[label] = np.where(y_train == label)[0]
            test_indices[label] = np.where(y_test == label)[0]
            # train_label_to_x[label] = x_train[train_indices[label]]
            # test_label_to_x[label] = x_test[test_indices[label]]

        # Log full dataset's distribution
        distribution_json = {}
        for split, distribtution in [("train", train_indices), ("test", test_indices)]:
            distribution_json[split] = {}
            for key in distribtution:
                distribution_json[split][int(key)] = len(distribtution[key])
        write_file(filepath=self.base_dir, filename=f"{dataset}_distribution.json", data=distribution_json)

        syslog(self.client_id, f"_generate_tasks split_option: {self.opt.split_option}", self.logger)
        # if self.opt.split_option == 'non_iid':  # NonIID
        #     labels_per_task = [labels[i: i + self.opt.num_classes] for i in range(0, len(labels), self.opt.num_classes)]
        if self.opt.split_option == 'iid':
            indices = np.arange(len(x_train))
            # random_shuffle(self.opt.global_random_seed, indices)  # globally same order
            for cid in range(self.opt.num_clients):
                self.task_cnt += 1
                offset = round(len(indices) / self.opt.num_clients)
                idx = indices[cid * offset: (cid + 1) * offset]
                x_task = x_train[idx]
                y_task = y_train[idx]
                doc_embeddings_task = doc_embeddings_all_train[idx]
                syslog(self.client_id, 'task: %d, dataset: %s, classes: %s, instances: %d'
                       % (self.task_cnt, dataset, ','.join(map(str, labels)), len(x_task)), self.logger)
                self._save_task((x_task, x_test), (y_task, y_test), doc_embeddings_task, [],
                                labels, dataset, cid, labels_names_dict, maxlen_dataset=maxlen_dataset)
        elif self.opt.split_option == 'non_iid' or self.opt.split_option == 'overlapped':
            labels_per_task = []
            for index in range(self.opt.gen_num_tasks):
                random.seed(self.opt.random_seed_data_gen + index)
                labels_per_task.append(np.array(random.sample(list(labels), self.opt.num_classes)))
            label_map = {}
            for labels in labels_per_task:
                for label in labels:
                    if label not in label_map:
                        label_map[label] = 0
                    label_map[label] += 1
            label_map_non_iid_counter = {label: 0 for label in label_map.keys()}

            for t, task in enumerate(labels_per_task):
                self.task_cnt += 1
                if self.opt.split_option == 'non_iid':
                    def split_instance_non_iid(c, idx):
                        fraction_size = int(len(idx) / label_map[c])
                        if label_map_non_iid_counter[c] < label_map[c] - 1:
                            upper_limit = fraction_size * (label_map_non_iid_counter[c] + 1)
                        elif label_map_non_iid_counter[c] == label_map[c] - 1:
                            upper_limit = len(idx)
                        else:
                            raise IndexError("Dataset has already been exhausted!")
                        idx = idx[fraction_size * label_map_non_iid_counter[c]: upper_limit]
                        label_map_non_iid_counter[c] += 1
                        return idx

                    idx = np.concatenate([split_instance_non_iid(c, train_indices[c]) for c in task], axis=0)
                else:
                    def split_instance_overlapped(c, idx):
                        idx = idx[:round(len(idx) / label_map[c])]
                        label_map[c] -= 1
                        return idx

                    idx = np.concatenate([split_instance_overlapped(c, train_indices[c]) for c in task], axis=0)
                x_task = x_train[idx]
                y_task = y_train[idx]
                doc_embeddings_task = doc_embeddings_all_train[idx]
                doc_embeddings_all_tasks.append(doc_embeddings_task)
                # self.opt.et_clustering_algorithm = "kmeans"
                cluster_centres_all_tasks.append(cluster(self.opt, doc_embeddings_task, k=self.opt.et_max_cluster_centroids))
                self.logger.info(f"Task {task} has {len(cluster_centres_all_tasks[-1])} valid clusters")
                doc_embeddings_info.append(f"{t}: {str(task)}, {len(x_task)}")

                test_idx = np.concatenate([test_indices[c] for c in task], axis=0)
                x_test_task = x_test[test_idx]
                y_test_task = y_test[test_idx]
                syslog(self.client_id, 'task: %d, dataset: %s, classes: %s'
                       % (self.task_cnt, dataset, ','.join(map(str, task))), self.logger)
                self._save_task((x_task, x_test_task), (y_task, y_test_task), doc_embeddings_task, cluster_centres_all_tasks[-1],
                                task, dataset, t, labels_names_dict, maxlen_dataset=maxlen_dataset)

            # -----------------------------------------------------------------------------------------
            # Avg labels' predictions FULL DATASET (not tasks!)
            unique_label_idxs = list(range(len(labels_names)))
            avg_label_doc_embeddings = [doc_embeddings_all_train[np.where(y_train == label_idx)[0]]
                                        for label_idx in unique_label_idxs]
            self.logger.info(f"\n\nSizes: "
                             f"{[len(avg_label_doc_embedding) for avg_label_doc_embedding in avg_label_doc_embeddings]}")
            avg_label_doc_embeddings = [np.sum(avg_label_doc_embedding, axis=0) / len(avg_label_doc_embedding)
                                        for avg_label_doc_embedding in avg_label_doc_embeddings]
            cosine_matrix = [[1 - spatial.distance.cosine(doc_embedding, avg_label_doc_embedding)
                              for doc_embedding in doc_embeddings_all_train]
                             for avg_label_doc_embedding in avg_label_doc_embeddings]
            cosine_matrix = np.array(cosine_matrix)
            matrix_pred = np.argmax(cosine_matrix, axis=0)
            cm = metrics.confusion_matrix(y_train, matrix_pred)
            f1 = metrics.f1_score(y_train, matrix_pred, average='macro')
            acc = metrics.accuracy_score(y_train, matrix_pred)
            report = metrics.classification_report(y_train, matrix_pred)
            self.logger.info(f"Confusion matrix: \n{cm}")
            self.logger.info(f"F1 macro: {f1}")
            self.logger.info(f"Acc: {acc}")
            self.logger.info(f"report: \n{report}")

            # -----------------------------------------------------------------------------------------
            # Plotting
            n_components, plotting_model_name = [(2, "pca"), (2, "tsne"), (3, "pca"), (3, "tsne")][0]
            # ToDo Minor Expand on colours for large num_tasks
            colours_primary = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'] * 3
            save_folder = join(self.base_dir, dataset)

            # Tasks' Clusters from Doc Embeddings
            for task_num in range(len(cluster_centres_all_tasks)):
                cluster_centres_task = cluster_centres_all_tasks[task_num]
                doc_embeddings_task = doc_embeddings_all_tasks[task_num]
                data_task = np.vstack((doc_embeddings_task, cluster_centres_task))
                colours_task = ['b' for _ in range(len(doc_embeddings_task))] + ['r' for _ in range(len(cluster_centres_task))]
                plot_diagram(data=data_task, colours=colours_task,
                             diagram_name=f"{plotting_model_name}_{n_components}_{task_num}.png",
                             plotting_model_name_graph=plotting_model_name, n_components_graph=n_components,
                             save_folder=join(save_folder, f"{self.opt.et_clustering_algorithm}_Tasks_Clusters"))

            # Avg tasks / centres only
            avg_task_vectors = [np.sum(task_vectors, axis=0) / len(task_vectors) for task_vectors in doc_embeddings_all_tasks]
            plot_diagram(data=avg_task_vectors, colours=[['r'] * len(avg_task_vectors)][0],
                         vector_info_for_graph=doc_embeddings_info,
                         diagram_name=f"{plotting_model_name}_{n_components}_avg.png",
                         plotting_model_name_graph=plotting_model_name, n_components_graph=n_components,
                         save_folder=save_folder)
            colours_primary = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'] * (int(len(y_train) / 8) + 1)
            # All data
            colours_all_data = [colours_primary[y[0]] for y in y_train]
            plot_diagram(data=doc_embeddings_all_train, colours=colours_all_data,
                         diagram_name=f"{plotting_model_name}_{n_components}_all.png",
                         plotting_model_name_graph=plotting_model_name, n_components_graph=n_components,
                         save_folder=save_folder)
            # All tasks
            # colours_all_tasks = [[colours_primary] * len(samples_in_task)
            #                      for colour_primary, samples_in_task in zip(colours_primary, doc_embeddings_all_tasks)]
            # colours_all_tasks = [colour[0] for colours_task in colours_all_tasks for colour in colours_task]
            # plot_diagram(data=[task_vector for task_vectors in doc_embeddings_all_tasks for task_vector in task_vectors],
            #              colours=colours_all_tasks, diagram_name=f"{plotting_model_name}_{n_components}_tasks.png",
            #              plotting_model_name_graph=plotting_model_name, n_components_graph=n_components,
            #              save_folder=save_folder)

    def _save_task(self, x, y, doc_embeddings_task, cluster_centres_task, labels, dataset, tid, labels_names_dict,
                   validation_fraction: float = 1 / 8, maxlen_dataset: int = 60):
        (x_task, x_test) = x
        (y_task, y_test) = y
        labels = sorted(labels)
        labels_names = [labels_names_dict[label] for label in labels]

        indices = list(range(len(x_task)))
        task_indices, valid_indices, y_task, y_valid = train_test_split(indices, y_task,
                                                                        test_size=validation_fraction,
                                                                        random_state=self.opt.random_seed_data_gen,
                                                                        stratify=y_task, shuffle=True)
        x_valid = x_task[valid_indices]
        x_task = x_task[task_indices]
        # vector_valid = vector_task[valid_indices]
        # vector_task = vector_task[task_indices]

        labels_task, count_task_np = np.unique(y_task, return_counts=True)
        labels_valid, count_valid_np = np.unique(y_valid, return_counts=True)
        labels_test, count_test_np = np.unique(y_test, return_counts=True)
        count_task, count_valid, count_test = [], [], []
        for label in labels:
            if label in labels_task:
                count_task.append(count_task_np[np.where(labels_task == label)][0])
            else:
                count_task.append(0)
            if label in labels_valid:
                count_valid.append(count_valid_np[np.where(labels_valid == label)][0])
            else:
                count_valid.append(0)
            if label in labels_test:
                count_test.append(count_test_np[np.where(labels_test == label)][0])
            else:
                count_test.append(0)

        self.logger.info(f"Stratified Train Valid splitting with frac: {validation_fraction}")
        self.logger.info(np.round(count_task / sum(count_task), 4))
        self.logger.info(np.round(count_valid / sum(count_valid), 4))
        self.logger.info(np.round(count_test / sum(count_test), 4))

        size_per_class = []
        size_per_class_split = {'train': [], 'valid': [], 'test': []}
        for split, label_ids, label_counts in [("train", labels_task, count_task), ("valid", labels_valid, count_valid),
                                               ("test", labels_test, count_test)]:
            # label_ids, label_counts = np.unique(y, return_counts=True)
            # for label_id, label_count in zip(label_ids, label_counts):
            #     size_per_class_split[split].append(int(label_count))
            size_per_class_split[split] = list(map(int, label_counts))
        for label in range(len(labels)):
            # print(size_per_class_split)
            size_per_class.append(size_per_class_split['train'][label] + size_per_class_split['valid'][label] +
                                  size_per_class_split['test'][label])

        task_idx_list = [np.where(y_task[:] == c)[0] for c in labels]
        valid_idx_list = [np.where(y_valid[:] == c)[0] for c in labels]
        test_idx_list = [np.where(y_test[:] == c)[0] for c in labels]
        for i, (train_idx, valid_idx, test_idx) in enumerate(zip(task_idx_list, valid_idx_list, test_idx_list)):
            y_task[train_idx] = i  # reset classes id
            y_valid[valid_idx] = i
            y_test[test_idx] = i

        y_task = tf.keras.utils.to_categorical(y_task, len(labels))
        y_valid = tf.keras.utils.to_categorical(y_valid, len(labels))
        y_test = tf.keras.utils.to_categorical(y_test, len(labels))

        pairs_train = list(zip(x_task, y_task))
        pairs_valid = list(zip(x_valid, y_valid))
        pairs_test = list(zip(x_test, y_test))

        filename = '{}_{}'.format(dataset, tid)
        _data = {
            'train': pairs_train,
            'test': pairs_test,
            'valid': pairs_valid,
            'doc_embeddings': doc_embeddings_task,
            'clusters_centers': cluster_centres_task,
            'classes': labels,
            'name': filename,
            'size_per_class': size_per_class,
            'size_per_class_split': size_per_class_split,
            'maxlen': maxlen_dataset
        }
        np_save(base_dir=self.base_dir, filename=filename, data=_data)

        stats = {
            'train_length': len(_data['train']),
            'valid_length': len(_data['valid']),
            'test_length': len(_data['test']),
            'classes': str(labels),
            'labels_names': labels_names,
            'name': filename,
            'size_per_class': size_per_class,
            'size_per_class_split': size_per_class_split,
            'maxlen': maxlen_dataset
        }
        write_file(filepath=self.base_dir, filename=filename + ".json", data=stats)

    def get_task(self, task_id):
        task = np.load(join(self.base_dir, self.tasks[task_id]), allow_pickle=True)
        return task.item()
