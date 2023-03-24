import torch, dgl
import numpy as np


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, setType='train', neg_data=None):
        super(MyDataset, self).__init__()
        if setType not in ['train', 'val', 'test']:
            raise ValueError('Invalid setType {}'.format(setType))

        self.n_user = dataset['userCount']
        self.n_item = dataset['itemCount']
        self.n_category = dataset['categoryCount']
        self.training = setType == 'train'

        if self.training:
            uids, iids = dataset['train'].nonzero()
            self.data = np.stack((uids, iids), axis=1).astype(np.int64)
            self.actSet = set((uid, iid) for uid, iid in zip(uids, iids))
        else:
            uids, iids = dataset[setType].nonzero()
            data = []
            for uid, pos in zip(uids, iids):
                data.append((uid, pos))
                for neg in neg_data[uid]:
                    data.append((uid, neg))
            self.data = np.array(data, dtype=np.int64)

    def neg_sample(self):
        assert self.training
        self.neg_data = np.random.randint(low=0, high=self.n_item, size=len(self.data), dtype=np.int64)
        for i in range(len(self.data)):
            uid = self.data[i][0]
            iid = self.neg_data[i]
            while (uid, iid) in self.actSet:
                iid = np.random.randint(low=0, high=self.n_item)
            self.neg_data[i] = iid

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user, item = self.data[idx]
        if self.training:
            return user, item, self.neg_data[idx]
        else:
            return user, item


def prepare_dgl_graph(args, dataset):
    """
        indexed from 0
        [base_u, base_u + n_user)
        [base_i, base_i + n_item)
        [base_c, base_c + n_category)
        base_u starts from 0

        etype:
            0 u-u
            1 i-u
            2 u-i
            3 c-i
            4 i-c
    """
    src, dst, etype = [], [], []
    bu = 0
    bi = bu + dataset['userCount']
    bc = bi + dataset['itemCount']
    num_nodes = bc + dataset['categoryCount']

    """ social network """
    uids, fids = dataset['trust'].nonzero()
    src += (bu + uids).tolist()
    dst += (bu + fids).tolist()
    etype += [0] * dataset['trust'].nnz

    """ user-item interactions """
    uids, iids = dataset['train'].nonzero()
    src += (bi + iids).tolist()
    dst += (bu + uids).tolist()
    etype += [1] * dataset['train'].nnz
    src += (bu + uids).tolist()
    dst += (bi + iids).tolist()
    etype += [2] * dataset['train'].nnz

    """ item-categories relations """
    iids, cids = dataset['category'].nonzero()
    src += (bc + cids).tolist()
    dst += (bi + iids).tolist()
    etype += [3] * dataset['category'].nnz
    src += (bi + iids).tolist()
    dst += (bc + cids).tolist()
    etype += [4] * dataset['category'].nnz

    graph = dgl.graph((src, dst), num_nodes=num_nodes)
    graph.edata['type'] = torch.LongTensor(etype)

    return graph
