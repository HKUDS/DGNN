import torch, pickle, time, os
import numpy as np
from options import parse_args
from torch.utils.data import DataLoader
from model import HGMN, BPRLoss
from data_utils import prepare_dgl_graph, MyDataset
from utils import load_data, load_model, save_model, fix_random_seed_as
from tqdm import tqdm

class Model():
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
        self.dataset = load_data(args.data_path)
        val_neg_data = load_data(args.val_neg_path)
        test_neg_data = load_data(args.test_neg_path)
        trainset = MyDataset(self.dataset, 'train')
        valset   = MyDataset(self.dataset, 'val',  val_neg_data)
        testset  = MyDataset(self.dataset, 'test', test_neg_data)
        self.trainloader = DataLoader(
            dataset=trainset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )
        self.valloader = DataLoader(
            dataset=valset,
            batch_size=args.test_batch_size * 101,
            shuffle=False,
            num_workers=args.num_workers
        )
        self.testloader = DataLoader(
            dataset=testset,
            batch_size=args.test_batch_size * 101,
            shuffle=False,
            num_workers=args.num_workers
        )
        self.graph = prepare_dgl_graph(args, self.dataset).to(self.device)
        self.model = HGMN(args, self.dataset['userCount'], self.dataset['itemCount'], self.dataset['categoryCount'])
        self.model = self.model.to(self.device)
        self.criterion = BPRLoss(args.reg)
        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters()}
        ], lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=args.decay_step,
            gamma=args.decay
        )
        if args.checkpoint:
            load_model(self.model, args.checkpoint, self.optimizer)


    def train(self):
        args = self.args

        best_hr, best_ndcg, best_epoch, wait = 0, 0, 0, 0
        start_time = time.time()
        for self.epoch in range(1, args.n_epoch + 1):
            epoch_losses = self.train_one_epoch(self.trainloader, self.graph)
            print('epoch {} done! elapsed {:.2f}.s, epoch_losses {}'.format(
                self.epoch, time.time() - start_time, epoch_losses
            ), flush=True)

            hr, ndcg = self.validate(self.testloader, self.graph)
            cur_best = hr + ndcg > best_hr + best_ndcg
            if cur_best:
                best_hr, best_ndcg, best_epoch = hr, ndcg, self.epoch
                wait = 0
            else:
                wait += 1
            print('+ epoch {} tested, elapsed {:.2f}s, N@{}: {:.4f}, R@{}: {:.4f}'.format(
                self.epoch, time.time() - start_time, args.topk, ndcg, args.topk, hr
            ), flush=True)

            if args.model_dir and cur_best:
                desc = f'{args.dataset}_hid_{args.n_hid}_layer_{args.n_layers}_mem_{args.mem_size}_' + \
                       f'lr_{args.lr}_reg_{args.reg}_decay_{args.decay}_step_{args.decay_step}_batch_{args.batch_size}'
                perf = '' # f'N/R_{ndcg:.4f}/{hr:.4f}'
                fname = f'{args.desc}_{desc}_{perf}.pth'
                save_model(self.model, os.path.join(args.model_dir, fname), self.optimizer)

            if wait >= args.patience:
                print(f'Early stop at epoch {self.epoch}, best epoch {best_epoch}')
                break

        print(f'Best N@{args.topk} {best_ndcg:.4f}, R@{args.topk} {best_hr:.4f}', flush=True)


    def train_one_epoch(self, dataloader, graph):
        self.model.train()

        epoch_losses = [0] * 2
        dataloader.dataset.neg_sample()
        tqdm_dataloader = tqdm(dataloader)

        for iteration, batch in enumerate(tqdm_dataloader):
            user_idx, pos_idx, neg_idx = batch

            rep, user_pool = self.model(graph)
            user = rep[user_idx] + user_pool[user_idx]
            pos  = rep[self.model.n_user + pos_idx]
            neg  = rep[self.model.n_user + neg_idx]
            pos_preds = self.model.predict(user, pos)
            neg_preds = self.model.predict(user, neg)
            loss, losses = self.criterion(pos_preds, neg_preds, user, pos, neg)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_losses = [x + y for x, y in zip(epoch_losses, losses)]
            tqdm_dataloader.set_description('Epoch {}, loss: {:.4f}'.format(self.epoch, loss.item()))

        if self.scheduler is not None:
            self.scheduler.step()

        epoch_losses = [sum(epoch_losses)] + epoch_losses
        return epoch_losses


    def calc_hr_and_ndcg(self, preds, topk):
        preds = preds.reshape(-1, 101)
        labels = torch.zeros_like(preds)
        labels[:, 0] = 1
        _, indices = preds.topk(topk)
        hits = labels.gather(1, indices)
        hrs = hits.sum(1).tolist()
        weights = 1 / torch.log2(torch.arange(2, 2 + topk).float()).to(hits.device)
        ndcgs = (hits * weights).sum(1).tolist()
        return hrs, ndcgs


    def validate(self, dataloader, graph):
        self.model.eval()
        hrs, ndcgs = [], []

        with torch.no_grad():
            tqdm_dataloader = tqdm(dataloader)
            for iteration, batch in enumerate(tqdm_dataloader, start=1):
                user_idx, item_idx = batch

                rep, user_pool = self.model(graph)
                user = rep[user_idx] + user_pool[user_idx]
                item  = rep[self.model.n_user + item_idx]
                preds = self.model.predict(user, item)

                preds_hrs, preds_ndcgs = self.calc_hr_and_ndcg(preds, self.args.topk)
                hrs += preds_hrs
                ndcgs += preds_ndcgs

        return np.mean(hrs), np.mean(ndcgs)


    def test(self):
        load_model(self.model, args.checkpoint)
        self.model.eval()

        with torch.no_grad():
            rep, user_pool = self.model(self.graph)

            """ Save embeddings """
            user_emb = (rep[:self.model.n_user] + user_pool).cpu().numpy()
            item_emb = rep[self.model.n_user: self.model.n_user + self.model.n_item].cpu().numpy()
            with open(f'HGMN-{self.args.dataset}-embeds.pkl', 'wb') as f:
                pickle.dump({'user_embed': user_emb, 'item_embed': item_emb}, f)

            """ Save results """
            tqdm_dataloader = tqdm(self.testloader)
            uids, hrs, ndcgs = [], [], []
            for iteration, batch in enumerate(tqdm_dataloader, start=1):
                user_idx, item_idx = batch

                user = rep[user_idx] + user_pool[user_idx]
                item  = rep[self.model.n_user + item_idx]
                preds = self.model.predict(user, item)

                preds_hrs, preds_ndcgs = self.calc_hr_and_ndcg(preds, self.args.topk)
                hrs += preds_hrs
                ndcgs += preds_ndcgs
                uids += user_idx[::101].tolist()

            with open(f'HGMN-{self.args.dataset}-test.pkl', 'wb') as f:
                pickle.dump({uid: (hr, ndcg) for uid, hr, ndcg in zip(uids, hrs, ndcgs)}, f)


if __name__ == "__main__":
    args = parse_args()
    fix_random_seed_as(args.seed)

    app = Model(args)

    app.train()
