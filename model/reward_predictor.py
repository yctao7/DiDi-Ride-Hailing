import os
from glob import glob
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

DATA_DIR = os.environ.get('DATA')
ORDER_FILES = sorted(glob(os.path.join(DATA_DIR, 'data4/total_ride_request/order_*')))
#ORDER_FILES = ['order_20161101'] #truncated file for debug
FNN_config = {
    'input_dim': 2+1,
    'output_dim': 1,
    'hidden_dim': 32,
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

class OrderDataset(Dataset):
    def __init__(self, order_files, transform=None):
        self.transform = transform
        dfs = []
        for order_file in order_files:
            dfs.append(pd.read_csv(order_file,
                 names=['id', 't1', 't2', 'lon1', 'lat1', 'lon2', 'lat2', 'r'],
                 index_col=['id']))
        df = pd.concat(dfs)
        df['duration'] = df['t2'] - df['t1']
        df['t1'] = df['t1'].apply(self.parse_timestamp)
        df['t2'] = df['t2'].apply(self.parse_timestamp)
        self.df = df

    def parse_timestamp(self, timestamp):
        tm = pd.Timestamp(timestamp, unit='s', tz='Asia/Shanghai')
        return (tm.hour * 60 + tm.minute) * 60 + tm.second

    def __getitem__(self, idx):
        reward = self.df['r'].iloc[idx]
        duration = self.df['duration'].iloc[idx]
        state = self.df[['t1','lon1','lat1']].iloc[idx].to_numpy()
        next_state = self.df[['t2','lon2','lat2']].iloc[idx].to_numpy()
        if self.transform:
            state = self.transform(state)
            next_state = self.transform(next_state)
        return state, next_state, duration, reward

    def __len__(self):
        return len(self.df)

class Trainer:
    def __init__(self,
                 learning_rate=1e-3,
                 **FNN_config):
        self.network = FNN(**FNN_config)
        self.learning_rate = learning_rate
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)

        self.TIME_SLOT = 10 * 60.
        self.GAMMA = 0.9

    def train(self, dataloader):
        for i, batch in enumerate(dataloader):
            states, next_states, durations, rewards = batch
            gamma = self.GAMMA ** (durations / self.TIME_SLOT)
            next_values = self.network(next_states).squeeze()
            target_values = rewards + gamma * next_values
            predicted_values = self.network(states).squeeze()
            loss = self.criterion(predicted_values, target_values)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def save(self, filename):
        torch.save(self.network.state_dict(), filename)

    def visualize(self, filename=None):
        from itertools import product
        import contextily as ctx
        import numpy as np
        import matplotlib.pyplot as plt

        N_lon, N_lat, N_t = 20, 20, 25
        lon_range = np.linspace(103.78, 104.26, N_lon)
        lat_range = np.linspace(30.45, 30.88, N_lat)
        t_range = np.linspace(0, 86400, N_t)
        states = torch.tensor(
            [[t, x, y] for t in t_range
                       for x in lon_range
                       for y in lat_range], dtype=torch.float32)
        values = self.network(states).detach().numpy()
        values = values.reshape(N_t, N_lon, N_lat)
        ax = plt.imshow(values[0], cmap='coolwarm', alpha=0.4)
        #ax = gdf.plot(figsize=(10,10), column='value', cmap='coolwarm', alpha=0.4)
        # TODO: bug
        ctx.add_basemap(ax, crs="EPSG:4326", zoom=12,
                        source=ctx.providers.Stamen.TonerLite)
        if filename is not None:
            plt.savefig(filename, dpi=300)
            

def normalize(states):
    states = torch.tensor(states, dtype=torch.float32)
    mean = torch.tensor([43200, 104.1, 30.6])
    std = torch.tensor([24941, 0.1, 0.1])
    x = (states - mean) / (std + 1e-16)
    return  x

if __name__ == '__main__':
    batch_size = 2000
    num_workers = 0
    learning_rate = 1e-3

    transform = normalize 
    dataset = OrderDataset(ORDER_FILES[:1], transform=transform)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            #pin_memory=True,
                            drop_last=True)
    trainer = Trainer(learning_rate=learning_rate,
                      **FNN_config)
    trainer.train(dataloader)
    trainer.save('state_dict.pkl')
    trainer.visualize('values.png')
