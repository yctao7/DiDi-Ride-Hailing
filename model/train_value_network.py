import os
import torch
import numpy as np

DATA_DIR = os.environ.get('DATA')
ORDER_FILE = 'data4/total_ride_request/order_20161101'
WORKING_DIR = os.path.dirname(__file__)

def read_file(filename):
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

    return data_array


# The network used to generate the value for a certain state (time&location)
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
        x = self.layer2(x)
        x = self.final(x)

        return x

class Agent(object):
    def __init__(
        self,
        learning_rate=1e-3,
        gamma=0.9,
        batch_size=2000
    ):
        # 4-dim input: location (latitude, longitude), time, day of week
        # 1-dim output: scalar value
        self.value_network = FNN(input_dim=2+1, output_dim=1, hidden_dim=128)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.value_network.parameters(), lr=learning_rate)
        self.batch_size = batch_size
        self.gamma = gamma

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
        state = torch.tensor(batch[:, [0, 2, 3]], dtype=torch.float)
        next_state = torch.tensor(batch[:, [1, 4, 5]], dtype=torch.float)
        dt = batch[:, 1] - batch[:, 0]  # unit: second
        time_slot = 10 * 60 # 10 minutes
        num_slot = dt / time_slot
        reward = batch[:, 6]
        reward = torch.tensor(self.discount_reward(reward, dt), dtype=torch.float)

        predict_value = self.value_network.forward(state)
        gamma = torch.tensor(self.gamma ** num_slot, dtype=torch.float)
        target_value = reward + gamma * self.value_network(next_state).squeeze()
        loss = self.criterion(predict_value, target_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        after_predict_value = self.value_network.forward(state)
        MSE = self.criterion(predict_value, after_predict_value)
        value_sample = self.value_network.forward(state[0]).item()
        return loss.item(), MSE.item(), value_sample

    def train(self, filename):
        data_array = read_file(filename)
        max_iteration = 1500
        for i in range(max_iteration):
            sample_index = np.random.choice(data_array.shape[0], self.batch_size, replace=False)
            batch = data_array[sample_index]
            loss, mse, value_sample = self.update_param(batch)
            if i % 20 == 0:
                print('Iteration {}, loss {}, MSE betw updates {}, value sample {}'.format(
                      i, loss, mse, value_sample))

    def storage_network(self):
        torch.save(self.value_network.state_dict(),
                   os.path.join(WORKING_DIR, 'state_dict.pkl'))

if __name__ == '__main__':
    Dummy_Agent = Agent()
    Dummy_Agent.train(os.path.join(DATA_DIR, ORDER_FILE))
    Dummy_Agent.storage_network()
