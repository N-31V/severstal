import torch
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from work_with_data import train_val_dataloader, SteelDataset


def load_model(model):
    with open(model, 'rb') as f:
        model = pickle.load(f)
    return model


class ModelToolkit:

    def __init__(self, model, name, num_workers=8, batch_size=8):
        self.model = model
        self.name = name
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.lr = 5e-4

        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            print(self.device, torch.cuda.get_device_name(0))
        else:
            self.device = torch.device('cpu')
            print(self.device)
        self.model = model.to(self.device)

        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=3,
                                                                    verbose=True)
        self.epoch = 0
        self.best_loss = float('inf')
        self.losses = {phase: [] for phase in ['train', 'val']}
        self.scores = {phase: Meter() for phase in ['train', 'val']}
        torch.backends.cudnn.benchmark = True
        self.dataloaders = train_val_dataloader(
            data_folder='./input/train_images',
            df_path='./input/train.csv',
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def forward(self, images, targets):
        images = images.to(self.device)
        masks = targets.to(self.device)
        outputs = self.model(images)
        loss = self.criterion(outputs, masks)
        return loss, outputs

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.epoch += 1
            self.run_epoch('train')
            with torch.no_grad():
                val_loss = self.run_epoch('val')
                self.scheduler.step(val_loss)
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                print('******** New optimal found, saving state ********')
                self.save_model()
            print()
        self.plot_scores()

    def predict(self, path='./input/test_images'):
        dataloader = DataLoader(
            SteelDataset(path),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )
        self.model.eval()
        test_predictions = []
        test_images_id = []
        for images, targets, images_id in tqdm(dataloader):
            images = images.to(self.device)
            with torch.no_grad():
                outputs = self.model(images)
            test_predictions.extend(torch.sigmoid(outputs).data.cpu().numpy())
            test_images_id.extend(images_id)
            torch.cuda.empty_cache()
        return test_images_id, test_predictions

    def save_model(self):
        file_name = 'models/{}-{}-{:.4f}.pickle'.format(self.name, self.epoch, self.best_loss)
        print('saving model with name: "{}"'.format(file_name))
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)

    def run_epoch(self, phase):
        meter = Meter()
        start = time.strftime('%H:%M:%S')
        print(f'Starting epoch: {self.epoch} | phase: {phase} | ⏰: {start}')
        self.model.train(phase == 'train')
        dataloader = self.dataloaders[phase]
        running_loss = 0.0
        total_batches = len(dataloader)
        tk0 = tqdm(dataloader, total=total_batches)
        self.optimizer.zero_grad()
        for itr, batch in enumerate(tk0):
            images, targets, images_id = batch
            loss, outputs = self.forward(images, targets)
            if phase == 'train':
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            running_loss += loss.item()
            outputs = outputs.detach().cpu()
            meter.metrics(torch.sigmoid(outputs), targets)
            tk0.set_postfix(loss=(running_loss / (itr + 1)))
        epoch_loss = running_loss / total_batches
        '''logging the metrics at the end of an epoch'''
        dice, iou, dice_pos, iou_pos, neg = meter.get_mean_metrics()
        print('Loss: %0.4f | dice: %0.4f | IoU: %0.4f  | dice_pos: %0.4f | IoU_pos: %0.4f | dice&IoU_neg: %0.4f' % (
            epoch_loss, dice, iou, dice_pos, iou_pos, neg))
        self.scores[phase].append_metrics(dice, iou, dice_pos, iou_pos, neg)
        self.losses[phase].append(epoch_loss)
        torch.cuda.empty_cache()
        return epoch_loss

    def plot_scores(self):
        pl = 1
        plt.figure(figsize=(15, 5))
        plt.subplot(2, 3, pl)
        self.plot_score(self.losses['train'], self.losses['val'], 'loss')
        for train, val, name in zip(self.scores['train'], self.scores['val'],
                                    ['dice', 'iou', 'dice_pos', 'iou_pos', 'neg']):
            pl += 1
            plt.subplot(2, 3, pl)
            self.plot_score(train, val, name)

    @staticmethod
    def plot_score(train, val, name):
        plt.figure(figsize=(15, 5))
        plt.plot(train, label=f'train {name}')
        plt.plot(val, label=f'val {name}')
        plt.title(f'{name} plot')
        plt.xlabel('Epoch')
        plt.ylabel(f'{name}')
        plt.legend()
        plt.show()


class Meter:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.dice_scores = []
        self.dice_pos_scores = []
        self.iou_scores = []
        self.iou_pos_scores = []
        self.neg_scores = []

    def get_mean_metrics(self):
        dice = np.mean(self.dice_scores)
        iou = np.mean(self.iou_scores)
        dice_pos = np.mean(self.dice_pos_scores)
        iou_pos = np.mean(self.iou_pos_scores)
        neg = np.mean(self.neg_scores)
        return dice, iou, dice_pos, iou_pos, neg

    def append_metrics(self, dice, iou, dice_pos, iou_pos, neg):
        self.dice_scores.append(dice)
        self.iou_scores.append(iou)
        self.dice_pos_scores.append(dice_pos)
        self.iou_pos_scores.append(iou_pos)
        self.neg_scores.append(neg)

    def get_metrics(self):
        dice = self.dice_scores
        iou = self.iou_scores
        dice_pos = self.dice_pos_scores
        iou_pos = self.iou_pos_scores
        neg = self.neg_scores
        return dice, iou, dice_pos, iou_pos, neg

    def metrics(self, probability, truth):
        """Calculates dice and iou of positive and negative images seperately"""
        """probability and truth must be torch tensors"""
        batch_size = len(truth)
        with torch.no_grad():
            probability = probability.view(batch_size, -1)
            truth = truth.view(batch_size, -1)
            assert (probability.shape == truth.shape)

            p = (probability > self.threshold)
            t = (truth > 0.5)

            neg_index = torch.nonzero(t.sum(-1) == 0)
            pos_index = torch.nonzero(t.sum(-1) >= 1)

            neg = (p.float().sum(-1) == 0).float()
            dice_pos = 2 * (p & t).sum(-1) / ((p.float() + t.float()).sum(-1))
            iou_pos = (p & t).sum(-1).float() / ((p | t).sum(-1))

            self.dice_scores.extend(np.vstack((neg[neg_index], dice_pos[pos_index])).tolist())
            self.dice_pos_scores.extend(dice_pos[pos_index].tolist())
            self.iou_scores.extend(np.vstack((neg[neg_index], iou_pos[pos_index])).tolist())
            self.iou_pos_scores.extend(iou_pos[pos_index].tolist())
            self.neg_scores.extend(neg[neg_index].tolist())
