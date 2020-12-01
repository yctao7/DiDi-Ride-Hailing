import os
import collections
from glob import glob
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

#DATA_DIR = '../data/' #os.environ.get('DATA')
DATA_DIR = '/home/zeyusun/work/RL/data/'
#ORDER_FILES = sorted(glob(os.path.join(DATA_DIR, 'data4/total_ride_request/order_*')))
ORDER_FILES = ['/home/zeyusun/work/RL/DiDi-Ride-Hailing/model/order_20161101'] #truncated file for debug
FNN_config = {
    'input_dim': 2+1,
    'output_dim': 1,
    'hidden_dim': 64,
}


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
        self.layer3 = torch.nn.Sequential(
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
        x = self.layer3(x)
        x = self.final(x)

        return x

class OrderDataset(Dataset):
    def __init__(self, order_files, batch_size=None, transform=None):
        self.batch_size = batch_size or 1
        self.transform = transform
        dfs = []
        for order_file in tqdm(order_files):
            dfs.append(pd.read_csv(order_file,
                 names=['id', 't1', 't2', 'lon1', 'lat1', 'lon2', 'lat2', 'r'],
                 index_col=['id']))
        df = pd.concat(dfs)
        #df['duration'] = df['t2'] - df['t1']
        #t1 = df['t1'].apply(pd.to_datetime)
        #df['t1'] = df['t1'].apply(self.parse_timestamp)
        #df['t2'] = df['t2'].apply(self.parse_timestamp)
        self.df = df

    def parse_timestamp(self, timestamp):
        tm = pd.Timestamp(timestamp, unit='s', tz='Asia/Shanghai')
        return (tm.hour * 60 + tm.minute) * 60 + tm.second

    def __getitem__(self, idx):
        indices = slice(idx * self.batch_size, (idx+1) * self.batch_size)
        reward = self.df['r'].iloc[indices].to_numpy()

        state = self.df[['t1','lon1','lat1']].iloc[indices]
        next_state = self.df[['t2','lon2','lat2']].iloc[indices]
        duration = (self.df['t2'].iloc[indices] - self.df['t1'].iloc[indices]).to_numpy()
        state['t1'] = state['t1'].apply(self.parse_timestamp)
        next_state['t2'] = next_state['t2'].apply(self.parse_timestamp)
        if self.transform:
            state = self.transform(state.to_numpy())
            next_state = self.transform(next_state.to_numpy())
        return state, next_state, duration, reward

    def __len__(self):
        return len(self.df) // self.batch_size

class Trainer:
    def __init__(self,
                 learning_rate=1e-1,
                 **FNN_config):
        self.network = FNN(**FNN_config)
        self.learning_rate = learning_rate
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)

        self.TIME_SLOT = 10 * 60.
        self.GAMMA = 0.9

        self.history = collections.defaultdict(list)
        self.sample_states = self.get_sample_states()
        self.sample_values = self.get_sample_values()

    def train(self, dataloader):
        state_grid = self.get_grid()
        for i, batch in tqdm(enumerate(dataloader)):
            states, next_states, durations, rewards = batch
            gamma = self.GAMMA ** (durations / self.TIME_SLOT)
            next_values = self.network(next_states).squeeze()
            target_values = rewards + gamma * next_values
            predicted_values = self.network(states).squeeze()
            loss = self.criterion(predicted_values, target_values)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.history['target_values'] = target_values.detach().tolist()
            self.history['predicted_values'] = predicted_values.detach().tolist()
            self.history['rewards'] = rewards
            loss = loss.item()
            self.history['loss'].append(loss)
            alpha = 0.1
            self.history['loss_smooth'].append(
                loss if len(self.history['loss_smooth']) == 0
                else (1-alpha) * self.history['loss_smooth'][-1] + alpha * loss)

            self.get_sample_values()
            if i % 1 == 0:
                self.plot()

    def plot(self):
        """Plot the training progresses."""
        from IPython.display import clear_output
        import matplotlib.pyplot as plt

        clear_output(True)
        plt.figure(figsize=(16, 5))

        plt.subplot(131)
        plt.title('Loss')
        plt.plot(self.history['loss'], alpha=0.5)
        plt.plot(self.history['loss_smooth'])
        plt.xlabel('Updating steps')

        plt.subplot(132)
        plt.plot(self.history['target_values'], self.history['predicted_values'],
                 'o',  label='predicted', alpha=0.8)
        xmin, xmax = [f(self.history['target_values']) for f in [min, max]]
        ymin, ymax = [f(self.history['predicted_values']) for f in [min, max]]
        plt.plot([xmin,xmax], [ymin,ymax])
        plt.plot(self.history['target_values'], self.history['rewards'],
                 'o', label='rewards', alpha=0.8)
        plt.xlabel('target_values')
        plt.legend()

        plt.subplot(133)
        plt.imshow(self.sample_values, origin='lower')
        plt.show()
        #plt.close()

    def save(self, filename):
        torch.save(self.network.state_dict(), filename)

    def get_sample_states(self):
        N_lon, N_lat, N_t = 20, 20, 25
        lon_lim = (103, 106) #103.78, 104.26
        lat_lim = (28, 32) #30.45, 30.88
        lon_range = np.linspace(*lon_lim, N_lon)
        lat_range = np.linspace(*lat_lim, N_lat)
        #t_range = np.linspace(0, 86400, N_t)
        t_range = [0]
        states = torch.tensor(
            [[t, x, y] for t in t_range
                       for x in lon_range
                       for y in lat_range], dtype=torch.float32)
        norm_states = normalize(states)
        return norm_states

    def get_sample_values(self):
        sample_states = self.get_sample_states()
        sample_states

    def visualize(self, filename=None):
        import contextily as ctx
        import numpy as np
        import matplotlib.pyplot as plt

        norm_states = self.get_grid()
        #fig, ax = plt.subplots()
        plt.imshow(values[0], cmap='coolwarm', alpha=0.4, origin='lower')
        plt.colorbar()
        #ax = gdf.plot(figsize=(10,10), column='value', cmap='coolwarm', alpha=0.4)
        # TODO: bug
        #ctx.add_basemap(ax, crs="EPSG:4326", zoom=12)#,
        #                #source=ctx.providers.Stamen.TonerLite)
        if filename is not None:
            plt.savefig(filename, dpi=300)

def normalize(states):
    states = torch.tensor(states, dtype=torch.float32)
    mean = torch.tensor([43200, 104.1, 30.6])
    std = torch.tensor([24941, 0.1, 0.1])
    x = (states - mean) / (std + 1e-16)
    return  x

def main(batch_size, num_workers, learning_rate):
    transform = normalize 
    dataset = OrderDataset(ORDER_FILES[:1], transform=transform,
                           batch_size=batch_size)
    dataloader = DataLoader(dataset,
                            batch_size=None, # disable automatic batching
                            batch_sampler=None, # disable automatic batching
                            num_workers=num_workers)
                            #pin_memory=True,
                            #drop_last=True)
    trainer = Trainer(learning_rate=learning_rate,
                      **FNN_config)
    trainer.train(dataloader)
    trainer.save('state_dict.pkl')
    #trainer.visualize('values.png')

if __name__ == '__main__':
    batch_size = 200
    num_workers = 0
    learning_rate = 1e-3
    main(batch_size, num_workers, learning_rate)
