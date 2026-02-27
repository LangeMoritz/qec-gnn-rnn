import numpy as np
import torch
import scipy

def sample_syndromes(n_shots, compiled_sampler, n_Z_stabilizers, n_stabilizers, d_t):
    # distinguish between training and testing:
    if compiled_sampler.__class__ == list:
        # sample for each error rate:
        n_trivial_syndromes = 0
        detection_events_list, observable_flips_list = [], []
        n_shots_one_p = n_shots // len(compiled_sampler)
        for sampler in compiled_sampler:
            # repeat each experiments multiple times to get enough non-empty:
            detections_one_p, observable_flips_one_p = [], []
            while len(detections_one_p) < n_shots_one_p:
            # 72(dt+1) Z stabilizers and 72(dt-1) X stabilizers
                detection_events, observable_flips = sampler.sample(
                shots=n_shots_one_p,
                separate_observables=True)
                # sums over the detectors to check if we have a parity change
                shots_w_flips = np.sum(detection_events, axis=1) != 0
                # save only data for measurements with non-empty syndromes
                # but count how many trivial (identity) syndromes we have
                n_trivial_syndromes += np.invert(shots_w_flips).sum()
                detections_one_p.extend(detection_events[shots_w_flips, :])
                observable_flips_one_p.extend(observable_flips[shots_w_flips, :])
            # if there are more non-empty syndromes than necessary
            detection_events_list.append(detections_one_p[:n_shots_one_p])
            observable_flips_list.append(observable_flips_one_p[:n_shots_one_p])
        # interleave lists to mix error rates: 
        # [sample(p1), sample(p2), ..., sample(p_n), sample(p1), sample(p2), ...]
        # instead of [sample(p1), sample(p1), ..., sample(p_2), sample(p2), ...]
        detection_events_list = [val for tup in zip(*detection_events_list) for val in tup]
        observable_flips_list = [val for tup in zip(*observable_flips_list) for val in tup]
    else:
        # 72(dt+1) Z stabilizers and 72(dt-1) X stabilizers
        detection_events_list, observable_flips_list = compiled_sampler.sample(
            shots=n_shots,
            separate_observables=True)
        # sums over the detectors to check if we have a parity change
        shots_w_flips = np.sum(detection_events_list, axis=1) != 0
        # save only data for measurements with non-empty syndromes
        # but count how many trivial (identity) syndromes we have
        n_trivial_syndromes = np.invert(shots_w_flips).sum()
        detection_events_list = detection_events_list[shots_w_flips, :]
        observable_flips_list = observable_flips_list[shots_w_flips, :]
    
    # num_samples, d_t, num_stabilizers, last perfect syndromes first:
    detection_events = np.array(detection_events_list)
    non_trivials = detection_events.shape[0]
    detection_events = np.hstack((detection_events[:, -n_Z_stabilizers:],
                                  detection_events[:, :-n_Z_stabilizers]))
    detection_events = detection_events.reshape((non_trivials, d_t, n_stabilizers))
    # make an array from the list:
    observable_flips = np.array(observable_flips_list)
    return detection_events, observable_flips, n_trivial_syndromes

def get_node_features(syndromes, n_Z_stabilizers, d_t):
    # 72(dt+1) Z stabilizers and 72(dt-1) X stabilizers
    # syndromes come as (d_t, n_stabilizers)
    # perfect last round comes first 
    # get the nonzero entries (node features):
    defect_inds = np.nonzero(syndromes)
    node_features = np.transpose(np.array(defect_inds))
    # assign the correct (d_t) time coordinate to the last perfect stabilizers
    # which currently stand in first position:
    check_first = node_features[:, 0] == 0
    check_last = node_features[:, 1] < n_Z_stabilizers
    stabilizers_from_last_round = check_first & check_last
    node_features[stabilizers_from_last_round, 0] = d_t
    # assign the correct ([0, ..., n_Z_stabilizers]) index to the first stabilizers:
    first_stabilizers = (node_features[:, 0] == 0) & (node_features[:, 1] >=
                                                       n_Z_stabilizers)
    node_features[first_stabilizers, 1] -= n_Z_stabilizers
    # note: syndromes come as:
    # [[Z_1_dt+1, ..., Z_n_dt+1, Z_1_1, ..., Z_n_1], [Z_1_2, ..., Z_n_2, X_1_2, ..., X_n_2],
    # ..., [Z_1_dt, ..., Z_n_dt, X_1_dt, ..., X_n_dt]]
    # [time, space, [stabilizer type]]:
    node_features = node_features.astype(np.float32)

    return node_features

def get_edges(node_features, adj_code, m_nearest_nodes):
    n_nodes = node_features.shape[0]
    # create edge indices for a fully connected graph
    edge_index = np.array(np.meshgrid(np.arange(n_nodes), np.arange(n_nodes), 
                                      indexing='ij')).reshape(2, -1)
    # convert indices to check coordinates:
    coordinate_list = np.vstack((node_features[edge_index[0, :], 1], 
                                 node_features[edge_index[1, :], 1])).astype(np.uint16)
    # covert in order to index the adjacency matrix:
    # compute the distances between the nodes from adjacency matrix:
    space_dist = adj_code[tuple(coordinate_list)].reshape(n_nodes, n_nodes)
    # get the temporal distance of any pair of nodes
    time_dist = np.abs(np.subtract.outer(node_features[:, 0], node_features[:, 0]))
    # get supremum norm of time and space distance:
    Adj = np.maximum(space_dist, time_dist)

    # only keep the edges that are as close or closer as the m_nearest node for 
    # each node (row by row)
    # Get the partition indices for the top m_nearest_nodes in each row
    # Find the m-th largest value for each row
    # do this only if there are more than m_nearest nodes in the graph:
    if m_nearest_nodes < n_nodes:
        thresholds = np.partition(-Adj, -m_nearest_nodes, axis=1)[:, -m_nearest_nodes]
        # Create a mask to retain all values >= threshold
        mask = -Adj >= thresholds[:, None]

        # Apply the mask to the adjacency matrix
        Adj = Adj * mask
    # note: the self edges are included in the sorting, so m_nearest nodes is
    # effectively reduced by 1

    Adj = np.maximum(Adj, Adj.T) # Make sure for each edge i->j there is edge j->i
    n_edges = np.count_nonzero(Adj) # Get number of edges

    # get the edge indices:
    edge_index = np.nonzero(Adj)
    edge_attr = Adj[edge_index].reshape(n_edges, 1)
    edge_index = np.array(edge_index)
    edge_attr = 1 / edge_attr ** 2

    return edge_index.astype(np.int64), edge_attr.astype(np.float32)
