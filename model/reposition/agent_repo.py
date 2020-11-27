import torch
import numpy as np
import pandas as pd
# from train_value_network import FNN
from train_pure_value_table import *

class Agent(object):
    def __init__(self):
        self.max_dist = 5*60*3  # the maximum distance a driver can go between two reposition
        self.vel = 3
        # self.value_network = torch.load('value_network.pkl')
        self.table = loc_table(load_file='table_table.pkl')
        print('Reposition algorithm initiated.')

    def dispatch(self,  dispatch_observ):
        pass

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
            grid_id = driver['grid_id']
            timestamp = repo_observ['timestamp']
            driver_info = self.table.grids[self.table.grids['grid_id']==grid_id]
            lng, lat = driver_info['lng'].item(), driver_info['lat'].item()
            x, y = self.table.look_up(lng, lat)
            max_value = 0
            destination_id = grid_id
            print('Start repositioning new driver')
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if x+dx < 0 or x+dx > 99 or y+dy < 0 or y+dy > 99:
                        continue
                    grid_id_list = self.table.grid_table.iloc[x+dx, y+dy]
                    #print('{} grids in square ({}, {})'.format(grid_id_list, dx, dy))
                    if type(grid_id_list) is not float:
                        for tmp_id in grid_id_list:
                            tmp_info = self.table.grids[self.table.grids['grid_id']==tmp_id]
                            dest_lng, dest_lat = tmp_info['lng'].item(), tmp_info['lat'].item()
                            dist = (((dest_lng-lng)*96000)**2 + ((dest_lat-lat)*111000)**2)**0.5
                            dt = int(dist / self.vel)
                            if dt > 5*60:
                                continue
                            value = self.table.look_up_value(dest_lng, dest_lat, timestamp+dt)
                            if value > max_value:
                                max_value = value
                                destination_id = tmp_id

            # repo_action.append({'driver_id': driver['driver_id'], 'destination': driver['grid_id']})
            repo_action.append({'driver_id': driver['driver_id'], 'destination': destination_id})

        return repo_action

if __name__ == '__main__':
    Dummy = Agent()
    repo_observ = {"timestamp": 1477962000, "driver_info": [{"driver_id": 0, "grid_id": "8f2a0ba14e0965b7"}, {"driver_id": 2, "grid_id": "ce76b98e88cc9213"}, {"driver_id": 6, "grid_id": "d8ce6475afb7dcf6"}, {"driver_id": 9, "grid_id": "8a1313c9a61cacb2"}], "day_of_week": 2}
    x = Dummy.reposition(repo_observ)
    