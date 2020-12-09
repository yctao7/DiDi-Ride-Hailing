import os
import sys
import ctypes
import torch
import numpy as np
import time
from km import KM
# from train_value_network import FNN, FNN_config
# from train_Q_table import FNN

# The network used to generate the value for a specific time (date&time)
class FNN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        """Value Network example
            Args:
                input_dim (int): `state` dimension.
                    `state` is 2-D tensor of shape (n, input_dim)
                output_dim (int): Number of actions.
                    value is 2-D tensor of shape (n, output_dim)
                hidden_dim (int): Hidden dimension in fc layer
        """
        super(FNN, self).__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU()
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU()
        )

        self.final = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns a state_value
            Args:
                x (torch.Tensor): `State` 2-D tensor of shape (n, input_dim)
            Returns:
                torch.Tensor: state_value, 2-D tensor of shape (n, output_dim)
        """
        x = self.layer1(x)
        x = self.layer2(x)    # comment this line to reduce the hidden layer number from 2 to 1
        x = self.final(x)

        return x

class Agent(object):
    """ Agent for dispatching and reposition """

    def __init__(self, **kwargs):
        """ Load your trained model and initialize the parameters """
        self.LEARNING_RATE = 0.01
        self.GAMMA = 0.9
        self.TIME_SLOT = 10 * 60
        self.max_lng = 104.3
        self.min_lng = 103.7
        self.max_lat = 30.9
        self.min_lat = 30.4
        # dirname = os.path.dirname(__file__)
        # ckpt_path = os.path.join(dirname, 'state_dict.pkl')
        # self.value_network = FNN(**FNN_config)
        # self.value_network.load_state_dict(torch.load(os.path.join(dirname, ckpt_path)))
        # self.value_network.eval() # TODO: remove if update
        # self.Q_table = loc_table()
        self.order_NN = FNN(input_dim=4+6, hidden_dim=64, output_dim=1)
        self.idle_NN = FNN(input_dim=4, hidden_dim=64, output_dim=1)
        self.order_optim = torch.optim.Adam(self.order_NN.parameters(), lr=self.LEARNING_RATE)
        self.idle_optim = torch.optim.Adam(self.idle_NN.parameters(), lr=self.LEARNING_RATE)
        self.criterion = torch.nn.MSELoss()
        self.order_batch = {'state': [], 'action': [], 'target_value': []}
        self.idle_batch = {'state': [], 'target_value': []}
        self.driver_recorder = []   # a list of dict, the key in the dict includes:
                                    # driver_id: int
                                    # this_state: an n*4 numpy array (x, y, date, t)
                                    # this_action: an n*6 numpy array (start_x, start_y, finish_x, finish_y, finish_date, finish_t)
                                    # this_reward: an n*1 numpy array
                                    # next_state: an n*4 numpy array (x, y, date, t)
                                    # next_action: an n*6 numpy array (start_x, start_y, finish_x, finish_y, finish_date, finish_t)
                                    # this_timestamp: an n*1 numpy array (timestamp)
        self.record_num = 0         # the number of data in the driver recorder


    def location_normalization(self, lng, lat):
        x = (lng - self.min_lng) / (self.max_lng - self.min_lng)
        y = (lat - self.min_lat) / (self.max_lat - self.min_lat)
        return x, y

    def add_driver_recorder(self, driver_id, driver_location, timestamp, order_start_location=None, order_finish_location=None, order_finish_timestamp=None, reward_units=None):
        '''
        Add records to the recorder of a certain driver
        :param driver_id: int
        :param driver_location: a list as [lng, lat], float
        :param timestamp: int
        :param order_start_location: a list as [lng, lat], float
        :param order_finish_location: a list as [lng, lat], float
        :param order_finish_timestamp: int
        :param reward_units: float
        :return:
        '''
        driver_x, driver_y = self.location_normalization(driver_location[0], driver_location[1])
        tmp = time.gmtime(timestamp)
        date = tmp[6]
        t = tmp[3] * 6 + tmp[4] / 10
        if order_start_location is not None:
            order_finish_time = time.gmtime(order_finish_timestamp)
            order_finish_date = order_finish_time[6]
            order_finish_t = order_finish_time[3] * 6 + order_finish_time[4] / 10
            order_start_x, order_start_y = self.location_normalization(order_start_location[0], order_start_location[1])
            order_finish_x, order_finish_y = self.location_normalization(order_finish_location[0], order_finish_location[1])
            reward_units = self.get_discounted_reward((order_finish_timestamp-timestamp)/(10*60), reward_units)
        else:
            order_finish_date = order_finish_t = order_start_x = order_start_y = order_finish_x = order_finish_y = np.nan
            reward_units = 0
        driver_index = -1
        for i in range(len(self.driver_recorder)):
            if self.driver_recorder[i]['driver_id'] == driver_id:
                driver_index = i
                break
        if driver_index == -1:
            self.driver_recorder.append({'driver_id': driver_id,
                                         'this_state': np.empty((0, 4)),
                                         'this_action': np.empty((0, 6)),
                                         'this_reward': np.empty((0, 1)),
                                         'next_state': np.empty((0, 4)),
                                         'next_action': np.empty((0, 6)),
                                         'this_timestamp': np.empty((0, 1))
                                         })
        # wrap the SARSA information
        this_state = np.array([[driver_x, driver_y, date, t]])
        this_action = np.array([[order_start_x, order_start_y, order_finish_x, order_finish_y, order_finish_date, order_finish_t]])
        this_reward = np.array([[reward_units]])
        next_state = np.array([[order_finish_x, order_finish_y, order_finish_date, order_finish_t]])
        next_action = np.array([[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]])
        # update last step
        if self.driver_recorder[driver_index]['next_state'].shape[0] > 0:
            self.driver_recorder[driver_index]['next_state'][-1] = this_state
            self.driver_recorder[driver_index]['next_action'][-1] = this_action
        # update the first step (order/idle)
        self.driver_recorder[driver_index]['this_state'] = np.concatenate([self.driver_recorder[driver_index]['this_state'], this_state])
        self.driver_recorder[driver_index]['this_action'] = np.concatenate([self.driver_recorder[driver_index]['this_action'], this_action])
        self.driver_recorder[driver_index]['this_reward'] = np.concatenate([self.driver_recorder[driver_index]['this_reward'], this_reward])
        self.driver_recorder[driver_index]['next_state'] = np.concatenate([self.driver_recorder[driver_index]['next_state'], next_state])
        self.driver_recorder[driver_index]['next_action'] = np.concatenate([self.driver_recorder[driver_index]['next_action'], next_action])
        self.driver_recorder[driver_index]['this_timestamp'] = np.concatenate([self.driver_recorder[driver_index]['this_timestamp'], np.array([[timestamp]])])
        self.record_num += 1
        # update the second step (idle)
        if order_start_location is not None:
            this_state = next_state
            this_action = next_action
            this_reward = np.array([[0]])
            next_state = np.array([[np.nan, np.nan, np.nan, np.nan]])
            next_action = np.array([[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]])
            self.driver_recorder[driver_index]['this_state'] = np.concatenate([self.driver_recorder[driver_index]['this_state'], this_state])
            self.driver_recorder[driver_index]['this_action'] = np.concatenate([self.driver_recorder[driver_index]['this_action'], this_action])
            self.driver_recorder[driver_index]['this_reward'] = np.concatenate([self.driver_recorder[driver_index]['this_reward'], this_reward])
            self.driver_recorder[driver_index]['next_state'] = np.concatenate([self.driver_recorder[driver_index]['next_state'], next_state])
            self.driver_recorder[driver_index]['next_action'] = np.concatenate([self.driver_recorder[driver_index]['next_action'], next_action])
            self.driver_recorder[driver_index]['this_timestamp'] = np.concatenate([self.driver_recorder[driver_index]['this_timestamp'], np.array([[order_finish_timestamp]])])
            self.record_num += 1

    def batch_update(self):
        '''
        Transfer the training data from driver recorder to batch
        :return:
        '''
        for driver in self.driver_recorder:
            for i in range(len(driver['this_state'])):
                if i == len(driver['this_state'])-1:
                    break
                this_state = driver['this_state'][i]
                this_action = driver['this_action'][i]
                reward = driver['this_reward'][i]
                next_state = driver['next_state'][i]
                next_action = driver['next_action'][i]

                dt = driver['this_timestamp'][i + 1] - driver['this_timestamp'][i]
                if not np.isnan(next_action[0]):
                    next_value = self.order_NN(torch.tensor(np.concatenate([next_state, next_action]), dtype=torch.float)).detach().numpy()
                else:
                    next_value = self.idle_NN(torch.tensor(next_state, dtype=torch.float)).detach().numpy()
                target_value = reward + self.GAMMA ** (dt / (10 * 60)) * next_value
                if not np.isnan(this_action[0]):
                    self.order_batch['state'].append(this_state)
                    self.order_batch['action'].append(this_action)
                    self.order_batch['target_value'].append(target_value)
                else:
                    self.idle_batch['state'].append(this_state)
                    self.idle_batch['target_value'].append(target_value)
        self.driver_recorder = []
        self.record_num = 0

    def NN_update(self):
        order_state_action = torch.tensor(np.concatenate([self.order_batch['state'], self.order_batch['action']], 1), dtype=torch.float)
        idle_state = torch.tensor(self.idle_batch['state'], dtype=torch.float)
        order_target_value = torch.tensor(self.order_batch['target_value'], dtype=torch.float)
        idle_target_value = torch.tensor(self.idle_batch['target_value'], dtype=torch.float)
        if len(order_state_action) > 0:
            order_loss = self.criterion(self.order_NN(order_state_action), order_target_value)
            self.order_optim.zero_grad()
            order_loss.backward()
            self.order_optim.step()
        if len(idle_target_value) > 0:
            idle_loss = self.criterion(self.idle_NN(idle_state), idle_target_value)
            self.idle_optim.zero_grad()
            idle_loss.backward()
            self.idle_optim.step()

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

        self.update_value_network(dispatch_observ, dispatch_action)

        return dispatch_action

    def update_value_network(self, dispatch_observ, dispatch_action):
        # collect data to the driver recorder
        driver_set = set()
        for pair in dispatch_observ:
            driver_set.add(pair['driver_id'])
        # add the information of the drivers who have been assigned orders to driver recorder
        for pair in dispatch_action:
            driver_id = pair['driver_id']
            order_id = pair['order_id']
            driver_set.remove(driver_id)
            for order in dispatch_observ:
                if order['order_id'] == order_id and order['driver_id'] == driver_id:
                    break
            param = {
                'driver_id': driver_id,
                'driver_location': order['driver_location'],
                'timestamp': order['timestamp'],
                'order_start_location': order['order_start_location'],
                'order_finish_location': order['order_finish_location'],
                'order_finish_timestamp': order['order_finish_timestamp'],
                'reward_units': order['reward_units']
            }
            self.add_driver_recorder(**param)
        # add the information of the drivers who have been idle to driver recorder
        for driver_id in driver_set:
            for order in dispatch_observ:
                if order['driver_id'] == driver_id:
                    break
            driver_location = order['driver_location']
            timestamp = order['timestamp']
            self.add_driver_recorder(driver_id, driver_location, timestamp)

        # Add data from driver recorder to batch and perform parameter update
        if True:
        # if self.record_num >= 128:
            self.batch_update()
            self.NN_update()

    def get_weight(self, pair):
        driver_location = pair['driver_location']
        timestamp = pair['timestamp']
        order_start_location = pair['order_start_location']
        order_finish_location = pair['order_finish_location']
        order_finish_timestamp = pair['order_finish_timestamp']
        driver_x, driver_y = self.location_normalization(driver_location[0], driver_location[1])
        order_start_x, order_start_y = self.location_normalization(order_start_location[0], order_start_location[1])
        order_finish_x, order_finish_y = self.location_normalization(order_finish_location[0], order_finish_location[1])
        order_start_time = time.gmtime(timestamp)
        order_start_date = order_start_time[6]
        order_start_t = order_start_time[3] * 6 + order_start_time[4] / 10
        order_finish_time = time.gmtime(order_finish_timestamp)
        order_finish_date = order_finish_time[6]
        order_finish_t = order_finish_time[3] * 6 + order_finish_time[4] / 10
        state = torch.tensor([driver_x, driver_y, order_start_date, order_start_t], dtype=torch.float)
        action = torch.tensor([order_start_x, order_start_y, order_finish_x, order_finish_y, order_finish_date, order_finish_t], dtype=torch.float)
        state_action = torch.cat([state, action])
        value = self.order_NN(state_action)
        return value

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