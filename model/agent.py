import os
import sys
import ctypes
import torch
import numpy as np
from train_value_network import FNN

class Agent(object):
    """ Agent for dispatching and reposition """

    def __init__(self, **kwargs):
        """ Load your trained model and initialize the parameters """
        dirname = os.path.dirname(__file__)
        ckpt_path = os.path.join(dirname, 'state_dict.pkl')
        self.value_network = FNN(input_dim=2+1, output_dim=1, hidden_dim=128)
        self.value_network.load_state_dict(torch.load(os.path.join(dirname, ckpt_path)))
        self.value_network.eval()

        hung_lib = "hungnpmc.so" if sys.platform == "darwin" else "hungnp.so"
        hung_lib = os.path.join(dirname, hung_lib)
        self.hung = ctypes.cdll.LoadLibrary(hung_lib)

        self.gamma = 0.9
        self.TIME_INTERVAL = 10 * 60

    def dispatch(self, dispatch_observ):
        """ Compute the assignment between drivers and passengers at each time step
        :param dispatch_observ: a list of dict, the key in the dict includes:
          order_id, int
          driver_id, int
          order_driver_distance, float
          order_start_location, a list as [lng, lat], float
          order_finish_location, a list as [lng, lat], float
          driver_location, a list as [lng, lat], float
          timestamp, int
          order_finish_timestamp, int
          day_of_week, int
          reward_units, float
          pick_up_eta, float

        :return: a list of dict, the key in the dict includes:
          order_id and driver_id, the pair indicating the assignment
        """
        if len(dispatch_observ) == 0:
            return 0
        edges = []
        orders, orders_inv = {}, {}
        drivers, drivers_inv = {}, {}
        for pair in dispatch_observ:
            order_id, driver_id, distance = pair['order_id'], pair['driver_id'], pair['order_driver_distance']
            reward = pair['reward_units']
            duration = pair['order_finish_timestamp'] - pair['timestamp']
            state = torch.tensor([[pair['timestamp'], *pair['driver_location']]])
            next_state = torch.tensor([[pair['order_finish_timestamp'], *pair['order_finish_location']]])

            v0 = self.value_network(state)
            v1 = self.value_network(next_state)
            done_prob = self.finish_prob(distance)
            if done_prob <= 0:
                continue

            gamma = self.gamma ** (duration // self.TIME_INTERVAL)
            complete_reward = reward + gamma * v1 - v0
            expected_reward = done_prob * complete_reward

            if order_id not in orders:
                idx = len(orders)
                orders[order_id] = idx
                orders_inv[idx] = order_id
            if driver_id not in drivers:
                idx = len(drivers)
                drivers[driver_id] = idx
                drivers_inv[idx] = driver_id

            edge = {}
            edge['weight'] = expected_reward
            edge['order_id'] = order_id
            edge['driver_id'] = driver_id
            edge['distance'] = distance
            edges.append(edge)

        weights = np.zeros((len(orders), len(drivers)))
        for edge in edges:
            i = orders[edge['order_id']]
            j = drivers[edge['driver_id']]
            weights[i,j] = edge['weight']

        matched_pairs = self.hungarian(weights)
        dispatch_action = [{'order_id': orders_inv[i], 'driver_id': drivers_inv[j]}
                           for i, j in matched_pairs]
        return dispatch_action

    def finish_prob(self, distance):
        return 0.01 * np.exp(np.log(20)/2000. * distance)

    def hungarian(self, weights):
        n, m = weights.shape
        if n > m:
            # Hungarian only takes a wide matrix
            matched_pairs = self.hungarian(weights.T)
            return [(col, row) for row, col in matched_pairs]

        cols = -np.ones(n, dtype=np.int32)  # matched column indices
        cols = cols.ctypes.data_as(ctypes.c_void_p)
        dataptr = weights.ctypes.data_as(ctypes.c_void_p)
        self.hung.MaxProfMatching(dataptr, n, m, cols)
        array_pointer = ctypes.cast(cols, ctypes.POINTER(ctypes.c_int * n))
        np_arr = np.frombuffer(array_pointer.contents, dtype=np.int32, count=n)
        matched_pairs = [(row, col) for row, col in enumerate(np_arr)]
        return matched_pairs


    def reposition(self, repo_observ):
        """ Compute the reposition action for the given drivers
        :param repo_observ: a dict, the key in the dict includes:
          timestamp: int
          driver_info: a list of dict, the key in the dict includes:
              driver_id: driver_id of the idle driver in the treatment group, int
              grid_id: id of the grid the driver is located at, str
          day_of_week: int

        :return: a list of dict, the key in the dict includes:
          driver_id: corresponding to the driver_id in the od_list
          destination: id of the grid the driver is repositioned to, str
        """
        repo_action = []
        for driver in repo_observ['driver_info']:
            # the default reposition is to let drivers stay where they are
            repo_action.append({'driver_id': driver['driver_id'], 'destination': driver['grid_id']})
        return repo_action
