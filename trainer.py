from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np
from tqdm.notebook import tqdm #for jupyter notebook environment. If in script or command line environment, from tqdm import tqdm
import matplotlib.pyplot as plt
import torch

class Trainer:
    def __init__(self, model, optimizer, train_loader, test_loader, device):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

    def train_one_epoch(self, epoch):
        self.model.train()
        self.model.to(self.device)
        ground_truths, predictions, losses = [], [], []
        with tqdm(total = len(self.train_loader), desc = "Epoch {} for training".format(epoch), unit="batch") as data_iter:
            for i, data in enumerate(self.train_loader):
                data = data.to(self.device)    
                self.optimizer.zero_grad()
                output, loss = self.model(data, data.batch)
                loss.backward()
                self.optimizer.step()
                y_true = data.y.cpu().detach().numpy()
                y_true_binarized = label_binarize(y_true, classes=[0, 1, 2, 3])
                y_pred = output.cpu().detach().numpy()
                auc_micro = roc_auc_score(y_true_binarized, y_pred, average = 'micro')
                data_iter.set_postfix(train_loss = round(loss.item(), 2), train_auc = round(auc_micro, 2), valid_loss = None, valid_auc = None)
                data_iter.update(1)
                ground_truths.extend(list(y_true_binarized))
                predictions.extend(list(y_pred))
                losses.append(loss.item())
        epoch_loss = sum(losses)/len(losses)
        epoch_auc_micro = roc_auc_score(np.array(ground_truths), np.array(predictions), average = 'micro')
        return epoch_loss, epoch_auc_micro

    def valid_one_epoch(self, epoch, loss, auc):
        ground_truths, predictions, losses = [], [], []
        correct = 0
        total = 0
        self.model.to('cpu')
        self.model.eval()
        with tqdm(total = len(self.test_loader), desc = "Epoch {} for testing".format(epoch), unit = 'batch') as data_iter:
            data_iter.set_postfix(train_loss = round(loss, 2), train_auc = round(auc, 2), valid_loss = 'TBD', valid_auc = 'TBD', accuracy = 'TBD')
            with torch.no_grad():
                for data in self.test_loader: #test_loader (val_loader) is the input of the Trainer class.
                    output, vloss = self.model(data, data.batch)
                    _, pred = torch.max(output, dim = 1)
                    total += data.y.size(0)
                    correct += (pred == data.y).sum().item()
                    y_true = data.y.cpu().detach().numpy()
                    y_true_binarized = label_binarize(y_true, classes=[0, 1, 2, 3])
                    y_pred = output.cpu().detach().numpy()
                    ground_truths.extend(list(y_true_binarized))
                    predictions.extend(list(y_pred))
                    losses.append(vloss.item())
                    data_iter.update(1)
            epoch_auc_micro = roc_auc_score(np.array(ground_truths), np.array(predictions), average = 'micro')
            epoch_loss = sum(losses)/len(losses)
            accuracy = round(100*correct/total, 2)
            data_iter.set_postfix(train_loss = round(loss, 2), train_auc = round(auc, 2), valid_loss = round(epoch_loss, 2), valid_auc = round(epoch_auc_micro, 2), accuracy = accuracy)
        return epoch_loss, epoch_auc_micro, accuracy

    def train(self, epochs):
        train_loss, train_auc, valid_loss, valid_auc, valid_acc = [], [], [], [], []
        for epoch in range(epochs):
            tloss, tauc = self.train_one_epoch(epoch)
            train_loss.append(tloss)
            train_auc.append(tauc)
            vloss, vauc, vacc = self.valid_one_epoch(epoch, tloss, tauc)
            valid_loss.append(vloss)
            valid_auc.append(vauc)
            valid_acc.append(vacc)
        return train_loss, train_auc, valid_loss, valid_auc, valid_acc

    def predict(self, test_loader, n_classes): #test_loader (val_loader) is the external input for the predict function.
        predictions = []
        ground_truths = []
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        correct = 0
        total = 0
        self.model.to('cpu')
        self.model.eval()
        data_iter = tqdm(total = len(test_loader), desc = 'Prediction', unit = 'batch')
        with torch.no_grad():
            for data in test_loader:
                output, _ = self.model(data, data.batch)
                _, pred = torch.max(output, dim = 1)
                total += data.y.size(0)
                correct += (pred == data.y).sum().item()
                y_true = data.y.cpu().detach().numpy()
                y_true_binarized = label_binarize(y_true, classes=[0, 1, 2, 3])
                y_pred = output.cpu().detach().numpy()
                predictions.extend(list(y_pred))
                ground_truths.extend(list(y_true_binarized))
                data_iter.set_postfix(stage="testing on-going")
                data_iter.update(1)
        data_iter.set_postfix(stage="testing done!")
        data_iter.close()
        
        #After finishing the prediction job, plot the ROC curve for each class and the average (micro) roc.
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(np.array(ground_truths)[:, i], np.array(predictions)[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        fpr["micro"], tpr["micro"], _ = roc_curve(np.array(ground_truths).ravel(), np.array(predictions).ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # Plot ROC curve for each class
        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"], color = 'deeppink', linestyle = ':', linewidth = 4, label = 'micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]))

        colors = ['aqua', 'darkorange', 'cornflowerblue', 'green']
        for i, color in enumerate(colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic to Multi-class')
        plt.legend(loc="lower right")
        plt.show()
        accuracy = round(100*correct/total, 2)
        print('Accuracy: {}%'.format(accuracy))

        return predictions, ground_truths