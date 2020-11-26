import os
import sys
import ctypes
import torch
import numpy as np

from km import KM
from train_value_network import FNN, FNN_config

class Agent(object):
    """ Agent for dispatching and reposition """

    def __init__(self, **kwargs):
        """ Load your trained model and initialize the parameters """
        dirname = os.path.dirname(__file__)
        ckpt_path = os.path.join(dirname, 'state_dict.pkl')
        self.value_network = FNN(**FNN_config)
        self.value_network.load_state_dict(torch.load(os.path.join(dirname, ckpt_path)))
        self.value_network.eval() # TODO: remove if update

        self.GAMMA = 0.9
        self.TIME_SLOT = 10 * 60

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
            return []

        order_id2loc, driver_id2loc = {}, {}
        order_loc2id, driver_loc2id = {}, {}
        count_order, count_driver = 0, 0
        for pair in dispatch_observ:
            #if self.get_finish_prob(pair) <= 0: # TODO: may eliminate an order/driver?
            #    continue
            if pair['order_id'] not in order_id2loc:
                order_id2loc[pair['order_id']] = count_order
                order_loc2id[count_order] = pair['order_id']
                count_order += 1
            if pair['driver_id'] not in driver_id2loc:
                driver_id2loc[pair['driver_id']] = count_driver
                driver_loc2id[count_driver] = pair['driver_id']
                count_driver += 1

        weights = torch.zeros(len(order_id2loc), len(driver_id2loc) + len(order_id2loc))
        for pair in dispatch_observ:
            i = order_id2loc[pair['order_id']]
            j = driver_id2loc[pair['driver_id']]
            weights[i, j] = self.get_weight(pair)
        raw_assign = KM(weights).run()[:len(driver_id2loc)]
        dispatch_action = []
        for i, item in enumerate(raw_assign):
            if item != -1:
                dispatch_action.append({'order_id': order_loc2id[item.item()],
                                        'driver_id': driver_loc2id[i]})

        # TODO: value network update
        #loss = self.value_network.criterion(predicted_value, target_value)
        #loss.backward()
        #self.value_network.

        return dispatch_action

    def get_weight(self, pair):
        order_finish_location = pair['order_finish_location']
        driver_location = pair['driver_location']
        timestamp = pair['timestamp']
        order_finish_timestamp = pair['order_finish_timestamp']
        reward_units = pair['reward_units']
        v1 = self.value_network(torch.tensor([timestamp, *driver_location]))
        v0 = self.value_network(torch.tensor([order_finish_timestamp, *order_finish_location]))
        dt = (order_finish_timestamp - timestamp) / self.TIME_SLOT
        finish_prob = self.get_finish_prob(pair)
        reward = finish_prob * self.get_discounted_reward(dt, reward_units)
        advantage = reward + self.GAMMA**dt * v1 - v0
        return advantage

    def get_discounted_reward(self, dt, reward_units):
        reward_per_slot = reward_units / dt
        discounted_reward = reward_per_slot * (1 - self.GAMMA**dt) / (1 - self.GAMMA)
        return discounted_reward

    def get_finish_prob(self, pair):
        distance = pair['order_driver_distance']
        prob = 0.01 * np.exp(np.log(20)/2000. * distance)
        return min(max(prob, 0), 1)

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
