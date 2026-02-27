import torch
import numpy as np
import torch.nn
import os

from src.utils import parse_yaml
from src.codes_q import create_bivariate_bicycle_codes
from src.build_circuit import build_circuit
from src.gnn_models import GNN_7_1head
from src.graph_representation import sample_syndromes, get_node_features, get_edges
from src.utils import get_adjacency_matr_from_check_matrices
from torch_geometric.data import Data, Batch
from multiprocessing import Pool, cpu_count
import wandb
os.environ["WANDB_SILENT"] = "True"

from pathlib import Path
from datetime import datetime
import time

class Decoder:
    def __init__(self, yaml_config=None, script_name=None):
        # load settings and initialise state
        paths, model_settings, graph_settings, training_settings = parse_yaml(yaml_config)
        self.save_dir = Path(paths["save_dir"])
        self.save_model_dir = Path(paths["save_model_dir"])
        self.saved_model_path = paths["saved_model_path"]
        self.model_settings = model_settings
        self.graph_settings = graph_settings
        self.training_settings = training_settings

        self.wandb_log = self.training_settings["wandb"]

        self.code_size = self.graph_settings["code_size"]
        self.train_error_rate = self.graph_settings["train_error_rate"]
        self.test_error_rate = self.graph_settings["test_error_rate"]
        # note: the self edges are included in the sorting, so m_nearest nodes is
        # effectively reduced by 1
        self.m_nearest_nodes = self.graph_settings["m_nearest_nodes"] + 1
        self.sigmoid = torch.nn.Sigmoid()

        # current training status
        self.epoch = training_settings["current_epoch"]
        if training_settings["device"] == "cuda":
            self.device = torch.device(
                training_settings["device"] if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device("cpu")

        # create a dictionary saving training metrics
        training_history = {}
        training_history["epoch"] = self.epoch
        training_history["train_loss"] = []
        training_history["train_accuracy"] = []
        training_history["test_loss"] = []
        training_history["test_accuracy"] = []
        training_history["val_loss"] = []
        training_history["val_accuracy"] = []

        self.training_history = training_history
        
        # only keep best found weights
        self.optimal_weights = None

        # instantiate model and optimizer
        self.model = GNN_7_1head(
            hidden_channels_GCN=model_settings["hidden_channels_GCN"],
            hidden_channels_MLP=model_settings["hidden_channels_MLP"],
            num_classes=model_settings["num_classes"]
            ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters())

        print(f'Running with a learning rate of {training_settings["lr"]}.')
        # generate a unique name to not overwrite other models
        current_datetime = datetime.now().strftime("%y%m%d_%H%M%S")
        name = ("n" +
                str(graph_settings["code_size"]) +
                '_' + current_datetime +
                '_' + script_name)
        save_path = self.save_dir / (name + ".pt")
        self.save_name = name

        # make sure we did not create an existing name
        if save_path.is_file():
            save_path = self.save_dir / (name + "_1.pt")

        save_model_path = self.save_model_dir / (name + "_model.pt")

        self.save_model_path = save_model_path
        self.save_path = save_path

        # check if model should be loaded
        if training_settings["resume_training"]:
            self.load_trained_model()

    def save_model_w_training_settings(self):
        # make sure the save folder exists, else create it
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_model_dir.mkdir(parents=True, exist_ok=True)

        attributes = {
            "training_history": self.training_history,
            "graph_settings": self.graph_settings,
            "training_settings": self.training_settings,
            "model_settings": self.model_settings,
        }

        attributes_model = {
            "training_history": self.training_history,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "graph_settings": self.graph_settings,
            "training_settings": self.training_settings,
            "model_settings": self.model_settings,
        }

        torch.save(attributes, self.save_path)
        torch.save(attributes_model, self.save_model_path)

    def load_trained_model(self):
        model_path = Path(self.saved_model_path)
        saved_attributes = torch.load(model_path, map_location=self.device, weights_only=False)

        # update attributes and load model with trained weights
        self.training_history = saved_attributes["training_history"]

        self.epoch = saved_attributes["training_history"]["epoch"] + 1
        self.model.load_state_dict(saved_attributes["model"])
        self.optimizer.load_state_dict(saved_attributes["optimizer"])

    def initialise_simulations(self, error_rate):
        # simulation settings (l, m, A_x_pows, A_y_pows, B_x_pows, B_y_pows, name=None)
        # [[72,12,6]]
        if self.code_size == 72:
            k, d = 12, 6
            code, A_list, B_list = create_bivariate_bicycle_codes(6, 6, [3], [1,2], [1,2], [3])

        # [[90,8,10]]
        elif self.code_size == 90:
            k, d = 8, 10
            code, A_list, B_list = create_bivariate_bicycle_codes(15, 3, [9], [1,2], [2,7], [0])

        # [[108,8,10]]
        elif self.code_size == 108:
            k, d = 8, 10
            code, A_list, B_list = create_bivariate_bicycle_codes(9, 6, [3], [1,2], [1,2], [3])

        # [[144,12,12]]
        elif self.code_size == 144:
            k, d = 12, 12
            code, A_list, B_list = create_bivariate_bicycle_codes(12, 6, [3], [1,2], [1,2], [3])

        # [[288,12,18]]
        elif self.code_size == 288:
            k, d = 12, 18
            code, A_list, B_list = create_bivariate_bicycle_codes(12, 12, [3], [2,7], [1,2], [3])

        else:
            print("!!!Please enter valid quantum code!!!")
        
        # check if number of classes and k allign:
        if k != self.model_settings["num_classes"]:
            print("!!!Please check the number of logical qubits!!!")

        # build the circuit, standard: d_t = code distance
        self.d_t = d
        self.n_stabilizers = self.code_size
        # find the number of X(Z)-stabilizers:
        self.n_Z_stabilizers = int(self.n_stabilizers / 2)
        # 72(dt+1) Z stabilizers and 72(dt-1) X stabilizers
        # find the adjacency matrix of the code:
        self.adj_code = get_adjacency_matr_from_check_matrices(code.hz, code.hx, 6).astype(np.float32)

        # distinguish between training and testing:
        if error_rate.__class__ == float:
            circuit = build_circuit(code, A_list, B_list, 
                                    p=error_rate, # physical error rate
                                    num_repeat=self.d_t, # usually set to code distance
                                    z_basis=True,   # whether in the z-basis or x-basis
                                    use_both=True, # whether use measurement results 
                                    #in both basis to decode one basis
                                   )
            self.compiled_sampler = circuit.compile_detector_sampler()
        
        elif error_rate.__class__ == list:
            compiled_samplers = []
            for p in error_rate:
                circuit = build_circuit(code, A_list, B_list, 
                                    p=p, # physical error rate
                                    num_repeat=self.d_t, # usually set to code distance
                                    z_basis=True,   # whether in the z-basis or x-basis
                                    use_both=True)
                sampler = circuit.compile_detector_sampler()
                compiled_samplers.append(sampler)
            self.compiled_sampler = compiled_samplers
    

    def evaluate_test_set(self, batch, n_trivial_syndromes, loss_fun, n_samples):
        correct_preds = 0
        # loop over batches
        with torch.no_grad():
            out = self.model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)

            prediction = (self.sigmoid(out.detach()) > 0.5).long()
            target = batch.y.long()
            correct_preds += int(((prediction == target).sum(dim=1) == 
                                  self.model_settings["num_classes"]).sum().item())
            val_loss = loss_fun(out, batch.y)
        val_accuracy = (correct_preds + n_trivial_syndromes) / (n_samples)
        return val_loss, val_accuracy
    

    def get_batch_of_graphs(self, syndromes, y):
        n_graphs = syndromes.shape[0]
        repeated_arguments = []
        for i in range(n_graphs):
            repeated_arguments.append((syndromes[i, :, :], y[i, :], 
                                       self.n_Z_stabilizers, self.d_t, 
                                       self.adj_code, self.m_nearest_nodes))
        #  create batches in parallel:
        with Pool(processes = (cpu_count() - 1)) as pool:
            batch = pool.starmap(generate_graph, repeated_arguments)
        batch = [Data(x=torch.from_numpy(item[0]),
                      edge_index=torch.from_numpy(item[1]),
                      edge_attr=torch.from_numpy(item[2]),
                      y=torch.from_numpy(item[3]))
                for item in batch]
        batch = Batch.from_data_list(batch)
        batch = batch.to(self.device)
        return batch
        

    def train(self):
        print("==============TRAINING==============")
        time_start = time.perf_counter()
        # training settings
        current_epoch = self.epoch
        n_epochs = self.training_settings["epochs"]
        dataset_size = self.training_settings["dataset_size"]
        validation_set_size = self.training_settings["validation_set_size"]
        test_set_size = self.training_settings["test_set_size"]
        batch_size = self.training_settings["batch_size"]
        n_batches = dataset_size // batch_size
        loss_fun = torch.nn.BCEWithLogitsLoss()
        sigmoid = torch.nn.Sigmoid()

        # Learning rate:
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.training_settings["lr"]

        # generate test set
        self.initialise_simulations(self.test_error_rate)
        syndromes, y, n_trivial_test = sample_syndromes(test_set_size, self.compiled_sampler,
                                                       self.n_Z_stabilizers, self.n_stabilizers, self.d_t)
        test_set = self.get_batch_of_graphs(syndromes, y)
        # normalize the node features:
        mean_d_t = self.d_t / 2
        test_set.x[:, 0] = (test_set.x[:, 0] - mean_d_t) / (mean_d_t)
        mean_space = (self.n_stabilizers - 1) / 2
        test_set.x[:, 1] = (test_set.x[:, 1] - mean_space) / mean_space


        # TIMING:
        time_sample = 0.
        time_fit = 0.
        time_write = 0.

        sample_start = time.perf_counter()
        self.initialise_simulations(self.train_error_rate)

        # generate validation set
        syndromes, y, n_trivial = sample_syndromes(validation_set_size, self.compiled_sampler,
                                                       self.n_Z_stabilizers, self.n_stabilizers, self.d_t)
        validation_set = self.get_batch_of_graphs(syndromes, y)
        # normalize the node features:
        mean_d_t = self.d_t / 2
        validation_set.x[:, 0] = (validation_set.x[:, 0] - mean_d_t) / (mean_d_t)
        mean_space = (self.n_stabilizers - 1) / 2
        validation_set.x[:, 1] = (validation_set.x[:, 1] - mean_space) / mean_space

        # the first batch of the training dataset:
        syndromes, y, n_trivial = sample_syndromes(batch_size, self.compiled_sampler,
                                                       self.n_Z_stabilizers, self.n_stabilizers, self.d_t)
        train_batch = self.get_batch_of_graphs(syndromes, y)
        time_sample += (time.perf_counter() - sample_start)

        # INITIALIZE WANDBE
        if self.wandb_log:
            wandb.init(project="IBM_codes_alvis", name = self.save_name, config = {
                **self.model_settings, **self.graph_settings, **self.training_settings})
    
        for epoch in range(current_epoch, n_epochs):
            train_loss = 0
            epoch_n_graphs = 0
            epoch_n_correct = 0

            for _ in range(n_batches):
                # forward/backward pass
                # normalize the node features:
                mean_d_t = self.d_t / 2
                train_batch.x[:, 0] = (train_batch.x[:, 0] - mean_d_t) / (mean_d_t)
                mean_space = (self.n_stabilizers - 1) / 2
                train_batch.x[:, 1] = (train_batch.x[:, 1] - mean_space) / mean_space

                fit_start = time.perf_counter()
                self.optimizer.zero_grad()
                out = self.model(train_batch.x, train_batch.edge_index, 
                                 train_batch.batch, train_batch.edge_attr)
                loss = loss_fun(out, train_batch.y)
                loss.backward()
                self.optimizer.step()

                # update loss and accuracies
                prediction = (sigmoid(out.detach()) > 0.5).long()
                target = train_batch.y.long()
                epoch_n_correct += int(((prediction == target).sum(dim=1) == 
                            self.model_settings["num_classes"]).sum().item())
                train_loss += loss.item() * batch_size
                epoch_n_graphs += batch_size
                time_fit += (time.perf_counter() - fit_start)

                sample_start = time.perf_counter()
                # replace the batch:
                syndromes, y, n_trivial = sample_syndromes(batch_size, self.compiled_sampler,
                                                       self.n_Z_stabilizers, self.n_stabilizers, self.d_t)
                train_batch = self.get_batch_of_graphs(syndromes, y)
                time_fit += (time.perf_counter() - fit_start)

            # train
            train_loss /= epoch_n_graphs
            train_accuracy = epoch_n_correct / (dataset_size)

            # validation (set the n_trivial syndromes to 0)
            val_loss, val_accuracy = self.evaluate_test_set(validation_set, 
                                                            0,
                                                            loss_fun,
                                                            validation_set_size)

            # test (count trivial syndromes as correct predictions)
            test_loss, test_accuracy = self.evaluate_test_set(test_set, 
                                                            n_trivial_test,
                                                            loss_fun,
                                                            test_set_size)
            print(f'Epoch: {epoch}, Loss: {train_loss:.4f}, Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}, Test Acc: {test_accuracy:.4f}')

            write_start = time.perf_counter()
            # save training attributes after every epoch
            self.training_history["epoch"] = epoch
            self.training_history["train_loss"].append(train_loss)
            self.training_history["val_loss"].append(val_loss)
            self.training_history["test_loss"].append(test_loss)
            self.training_history["train_accuracy"].append(train_accuracy)
            self.training_history["val_accuracy"].append(val_accuracy)
            self.training_history["test_accuracy"].append(test_accuracy)
            self.save_model_w_training_settings()

            # Log training and testing metrics to wandb
            if self.wandb_log:
                metrics = {'loss': train_loss, 'accuracy': train_accuracy, 
                           'val accuracy': val_accuracy, 'test accuracy': test_accuracy}
                wandb.log(metrics)
            write_end = time.perf_counter()
            time_write += (write_end - write_start)
        
        runtime = time.perf_counter()-time_start
        print('Training completed after {:.1f}:{:.1f}:{:.1f}'.format(*divmod(divmod(
            runtime, 60)[0], 60), *divmod(runtime, 60)[::-1]))
        
        print(f'Sampling and Graphing: {time_sample:.0f}s')
        print(f'Fitting: {time_fit:.0f}s')
        print(f'Writing: {time_write:.0f}s')
    
    def test(self):
        print("==============TESTING==============")
        time_start = time.perf_counter()
        loss_fun = torch.nn.BCEWithLogitsLoss()

        self.initialise_simulations(self.test_error_rate)
        batch_size = self.training_settings["acc_test_batch_size"]
        n_test_batches = self.training_settings["acc_test_size"] // batch_size

        test_accuracy = 0
        n_trivial_syndromes = 0
        for i in range(n_test_batches):
            # generate test batch
            syndromes, y, n_trivial_test = sample_syndromes(batch_size, self.compiled_sampler,
                                                       self.n_Z_stabilizers, self.n_stabilizers, self.d_t)
            test_set = self.get_batch_of_graphs(syndromes, y)
            # normalize the node features:
            mean_d_t = self.d_t / 2
            test_set.x[:, 0] = (test_set.x[:, 0] - mean_d_t) / (mean_d_t)
            mean_space = (self.n_stabilizers - 1) / 2
            test_set.x[:, 1] = (test_set.x[:, 1] - mean_space) / mean_space
            test_loss_batch, test_accuracy_batch = self.evaluate_test_set(test_set, 
                                                            n_trivial_test,
                                                            loss_fun,
                                                            batch_size)
            print(f'Accuracy: {test_accuracy_batch:.6f} Trivials: {n_trivial_test} No. Samples: {batch_size}')
            test_accuracy += test_accuracy_batch
            n_trivial_syndromes += n_trivial_test

        test_accuracy = test_accuracy / n_test_batches
        print(f'Test Acc: {test_accuracy}, tested on {n_test_batches * batch_size} '
              f'samples, of which {n_trivial_syndromes} trivial samples.')
        self.training_history["test_accuracy"].append(test_accuracy)

        runtime = time.perf_counter()-time_start
        print('Testing completed after {:.1f}:{:.1f}:{:.1f}'.format(*divmod(divmod(
            runtime, 60)[0], 60), *divmod(runtime, 60)[::-1]))


    def run(self):
        if self.training_settings["resume_training"]:
            print(f'Loading model {self.saved_model_path}')
        print(f'Running on code size {self.code_size}.')
        if self.training_settings['run_training']:
            if self.training_settings['run_test']:
                self.train()
                self.test()
                # only save final test accuracy if trained before
                self.save_model_w_training_settings()
            else:
                self.train()
        else:
            self.test()

def generate_graph(syndromes, y, n_Z_stabilizers, d_t, adj_code, top_m):
    node_features = get_node_features(syndromes, n_Z_stabilizers, d_t)
    edge_index, edge_attr = get_edges(node_features, adj_code, top_m)
    return [node_features, edge_index, edge_attr, y.reshape(1, -1).astype(np.float32)]