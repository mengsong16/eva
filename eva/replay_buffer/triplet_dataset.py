import random
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader as TorchDataLoader
from collections import OrderedDict

class Triplet(object):
    def __init__(self, s1, s2, s3):
        self.data = np.array([s1, s2, s3])
    
    def __hash__(self):
        return hash(np.array2string(self.data))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        
        return np.array_equal(self.data, other.data)


class TripletBuffer(object):
    """
    Unlimited triplet buffer
    """
    def __init__(self):
        self.buffer = OrderedDict()
    
    def __len__(self):
        return len(self.buffer)

    # s1, s2, s3 are numpy array states
    def is_triplet(self, s1, s2, s3):
        if np.array_equal(s1, s2) or np.array_equal(s1, s3) or np.array_equal(s2, s3):
            return False
        else:
            return True    

    def add_triplet(self, s1, s2, s3):
        if self.is_triplet(s1, s2, s3):
            trip = Triplet(s1,s2,s3)
            if trip not in self.buffer:
                self.buffer[trip] = None
    
    # return a list of Triplets
    def get_all_triplets(self):
        return list(self.buffer.keys())

    def print_all_triplets(self):
        triplets = self.get_all_triplets()
        for trip in triplets:
            print(trip.data)    

class TripletDataset(TorchDataset):
    
    def __init__(self, triplets):
        super(TripletDataset, self).__init__()
        self.triplets = triplets #  a list of Triplet
        #self.size = int(batch_size) * int(num_batches)

    def __len__(self):
        return len(self.triplets)

    def get_target(self):
        pass    

    def __getitem__(self, idx):
        # get triplet
        if torch.is_tensor(idx):
            idx = idx.tolist()[0]

        # randomly sample a triplet from the list
        triplet = random.choice(self.triplets)

        # convert to torch tensor
        sample = {
            'anchor': torch.tensor(triplet.data[0], dtype=torch.float), 
            'positive': torch.tensor(triplet.data[1], dtype=torch.float),
            'negative': torch.tensor(triplet.data[2], dtype=torch.float) 
        }        
        return sample

if __name__ == "__main__": 
    buffer = TripletBuffer()
    #print(buffer.get_all_triplets())
    buffer.add_triplet(np.array([1,1]), np.array([2,2]), np.array([3,3]))
    #buffer.print_all_triplets()
    #print(buffer.__len__())
    buffer.add_triplet(np.array([1,1]), np.array([2,2]), np.array([3,3]))
    #buffer.print_all_triplets()
    buffer.add_triplet(np.array([1,1]), np.array([1,1]), np.array([3,3]))
    #buffer.print_all_triplets()
    buffer.add_triplet(np.array([1,1]), np.array([3,3]), np.array([3,3]))
    buffer.add_triplet(np.array([3,3]), np.array([3,3]), np.array([3,3]))
    buffer.add_triplet(np.array([1,1]), np.array([3,3]), np.array([1,1]))
    buffer.add_triplet(np.array([-2,1]), np.array([2,2]), np.array([3,3]))
    buffer.add_triplet(np.array([-2,1]), np.array([2,2]), np.array([3,3]))
    buffer.add_triplet(np.array([100,1]), np.array([102,2]), np.array([99,0]))
    buffer.add_triplet(np.array([101,1]), np.array([102,2]), np.array([99,0]))
    #buffer.print_all_triplets()

    dataset = TripletDataset(triplets=buffer.get_all_triplets())
    print("Dataset size (number of triplets): %d"%len(dataset))

    train_dataset_loader = TorchDataLoader(dataset, 
                                    batch_size=2, 
                                    shuffle=True, 
                                    num_workers=0)

    print("Dataloader size (number of batches): %d"%len(train_dataset_loader))

    for batch_idx, triplet_batch in enumerate(train_dataset_loader): 
        print("---------------- batch %d ------------------"%batch_idx)
        print('anchor:')
        print(triplet_batch['anchor']) 
        print('positive:')
        print(triplet_batch['positive']) 
        print('negative:')
        print(triplet_batch['negative'])

    print("---------------------------------")

