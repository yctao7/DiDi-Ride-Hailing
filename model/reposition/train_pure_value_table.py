import torch
import numpy as np
import time
import os
import pandas as pd
import pickle

def read_file(*filenames):
    '''
    Load the data file and return a big numpy array that contains all information except order ID
    :param filenames: the tuple in which the name strings of the files are
    :return all_data_array: a numpy array
    '''
    if os.path.exists("all_data_array.npy"):
        return np.load("all_data_array.npy")
    all_data_array = np.zeros([0, 7])
    for filename in filenames:
        print('reading file:', filename)
        file = open(filename)
        data = file.readlines()
        data_list = []
        for line in data:
            tmp = line.split(',')
            for i in range(1, 8):
                tmp[i] = eval(tmp[i])
            data_list.append(tmp)
        file.close()
        data_array = np.zeros([len(data_list), 7])
        for i in range(len(data_list)):
            data_array[i, :] = np.array(data_list[i][1:8])
        all_data_array = np.concatenate([all_data_array, data_array])
    np.save("all_data_array.npy", all_data_array)
    return all_data_array

def decode_state(state):
    '''
    Decode the input state to the network
    :param state: an n*3 numpy array, 3 columns are timestamp, longtitude, latitude respectively
    :return new_state: an n*4 numpy array, the first two columns are day_of_week and the number of 10-minute that have passed
    '''
    date = np.zeros([state.shape[0], 1])
    t = np.zeros([state.shape[0], 1])
    for i in range(state.shape[0]):
        tmp = time.gmtime(state[i, 0])
        date[i, 0] = tmp[6]
        t[i, 0] = tmp[3]*6+tmp[4]/10
    date = np.array(date)
    t = np.array(t)
    new_state = np.concatenate([date, t, state[:, 1:3]], axis=1)
    return new_state


# The class used to discretize the map into square grids and storage the time table at the square grids
class loc_table(object):
    def __init__(self, load_file=None):
        self.grids = pd.read_csv('hexagon_grid_table.csv',
                                 names=['grid_id', 'lng1', 'lat1', 'lng2', 'lat2', 'lng3', 'lat3', 'lng4', 'lat4',
                                        'lng5', 'lat5', 'lng6', 'lat6'])
        self.grids['lng'] = (self.grids['lng1'] + self.grids['lng2'] + self.grids['lng3'] + self.grids['lng4'] +
                             self.grids['lng5'] + self.grids['lng6']) / 6
        self.grids['lat'] = (self.grids['lat1'] + self.grids['lat2'] + self.grids['lat3'] + self.grids['lat4'] +
                             self.grids['lat5'] + self.grids['lat6']) / 6
        self.grids = self.grids.drop(4183)
        self.row_num, self.col_num = 100, 100
        self.date_num, self.time_num = 7, int(86400/(10*60))
        self.max_lng = 104.3
        self.min_lng = 103.7
        self.max_lat = 30.9
        self.min_lat = 30.4
        self.step_size_lng = (self.max_lng - self.min_lng) / self.col_num
        self.step_size_lat = (self.max_lat - self.min_lat) / self.row_num
        self.grid_table = pd.DataFrame(data=None, index=range(self.row_num), columns=range(self.col_num))
        self.table_table = [[None for i in range(self.col_num)] for j in range(self.row_num)] 

        self.create_grid_table()
        self.create_table_table()

        if load_file is not None:
            self.load_table_table(load_file)

    def create_grid_table(self):
        '''
        Create a table that contains the grid IDs in square grids
        :return grid_table: a pd dataframe
        '''
        for k in range(self.grids.shape[0]):
            for l in range(1, 7):
                x, y = self.look_up(self.grids.iloc[k, 2 * l - 1], self.grids.iloc[k, 2 * l])
                if x is None:
                    continue
                if type(self.grid_table.loc[x, y]) is float:
                    self.grid_table.loc[x, y] = [self.grids.iloc[k, 0]]
                else:
                    self.grid_table.loc[x, y].append(self.grids.iloc[k, 0])
        self.grid_table.to_csv('grid_table.csv', encoding='gbk')
        return self.grid_table

    def create_table_table(self):
        '''
        Create a table that contains the time table (from date&time to value) in square grids
        :return NN_table: a second order list
        '''
        for i in range(self.row_num):
            for j in range(self.col_num):
                self.table_table[i][j] = [[0.0 for i in range(self.time_num)] for j in range(self.date_num)]
        return self.table_table

    def look_up(self, lng, lat):
        '''
        Given the longitude and the latitude of a location, return the index of the location in the square grid table
        :param lng, lat: two float
        :return xlabel, ylabel: two int
        '''
        if lng <= self.min_lng or lng >= self.max_lng or lat <= self.min_lat or lat >= self.max_lat:
            return None, None
        xlabel = int((lng - self.min_lng) / self.step_size_lng)
        ylabel = int((lat - self.min_lat) / self.step_size_lat)
        if xlabel == self.col_num:
            xlabel -= 1
        if ylabel == self.row_num:
            ylabel -= 1
        return xlabel, ylabel

    def storage_table_table(self):
        f = open('table_table.pkl', 'wb')
        pickle.dump(self.table_table, f, -1)
        f.close()

    def load_table_table(self, filename):
        f = open(filename, 'rb')
        self.table_table = pickle.load(f)
        f.close()

    def look_up_value(self, lng, lat, timestamp):
        '''
        Return the values of the given state
        :param lng, lat, timestamp: three scalar
        :return value: an n-dim list
        '''
        state = np.array([[timestamp, lng, lat]])
        state = decode_state(state)
        date, t, lng, lat = state[0]
        x, y = self.look_up(lng, lat)
        if x is None:
            return 0
        value = self.table_table[int(x)][int(y)][int(date)][int(t)]

        return value



# The training agent
class Agent(object):
    def __init__(
        self,
        gamma=0.9,
        batch_size=20000
    ):
        self.batch_size = batch_size
        self.gamma = gamma
        self.table = loc_table()
        self.criterion = torch.nn.MSELoss()

    def discount_reward(self, reward, dt):
        '''
        Modify the reward using the method from the paper
        :param reward: original reward
        :param dt: order duraction (second)
        :return modified_reward: reward after modified
        '''
        time_slot = 10*60   # 10 minutes
        num_slot = dt / time_slot   # could be a float
        reward_per_slot = reward / num_slot
        modified_reward = reward_per_slot * (1 - self.gamma**num_slot) / (1 - self.gamma)

        return modified_reward

    def update_param(self, batch):
        '''
        Use a batch data to take one update of NN parameters
        :param batch: a numpy array with shape batch_size*7 (remove order ID)
        :return:
        '''
        this_state = decode_state(batch[:, [0, 2, 3]])
        next_state = decode_state(batch[:, [1, 4, 5]])
        dt = batch[:, 1] - batch[:, 0]  # unit: second
        time_slot = 10 * 60             # 10 minutes
        num_slot = dt / time_slot
        reward = batch[:, 6]
        reward = self.discount_reward(reward, dt)
        for i in range(batch.shape[0]):
            this_x, this_y = self.table.look_up(this_state[i, 2], this_state[i, 3])
            next_x, next_y = self.table.look_up(next_state[i, 2], next_state[i, 3])
            this_date, this_time = int(this_state[i, 0]), int(this_state[i, 1])
            # print('this_date:', this_date, 'this_time:', this_time)
            next_date, next_time = int(next_state[i, 0]), int(next_state[i, 1])
            if this_x is None or next_x is None:
                continue
            predict_value = self.table.table_table[this_x][this_y][this_date][this_time]
            self.table.table_table[this_x][this_y][this_date][this_time] = reward[i] + self.gamma ** num_slot[i] * self.table.table_table[next_x][next_y][next_date][next_time]
            after_predict_value = self.table.table_table[this_x][this_y][this_date][this_time]
        MSE = self.criterion(torch.tensor(predict_value), torch.tensor(after_predict_value))
        print('MSE between updation:', MSE.item(), 'Value sample:', predict_value)
        # convergence_flag = (MSE < 1e-3)
        convergence_flag = False
        return convergence_flag

    def train(self, *filename):
        data_array = read_file(*filename)
        max_iteration = 15000
        for iter in range(max_iteration):
            print('iter:', iter)
            sample_index = np.random.choice(data_array.shape[0], self.batch_size, replace=False)
            batch = data_array[sample_index]
            convergence_flag = self.update_param(batch)
            if convergence_flag:
                break
            if iter % 300 == 0:
                self.table.storage_table_table()


if __name__ == '__main__':
    Dummy_Agent = Agent()
    filepath = os.path.dirname(__file__) + '/data4/total_ride_request'
    file_list = os.listdir(filepath)
    file_list = file_list[1:]
    for i in range(len(file_list)):
        file_list[i] = filepath + '/' + file_list[i]
    Dummy_Agent.train(*file_list)
