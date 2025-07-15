from sklearn.metrics import mean_squared_error, r2_score
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
        with tqdm(total = len(self.train_loader), desc = "Epoch {} - training".format(epoch), unit="batch") as data_iter:
            for i, data in enumerate(self.train_loader):
                data = data.to(self.device)    
                self.optimizer.zero_grad()
                output, loss = self.model(data, data.batch)
                loss.backward()
                self.optimizer.step()

                y_true = data.y.cpu().detach().numpy().flatten()
                y_pred = output.cpu().detach().numpy().flatten()

                ground_truths.extend(y_true)
                predictions.extend(y_pred)
                losses.append(loss.item())

                if i%10 == 0 or i == len(self.train_loader) - 1:
                    data_iter.set_postfix(train_loss = round(loss.item(), 4), train_rmse = round(mean_squared_error(ground_truths, predictions, squared=False), 4), train_r2 = round(r2_score(ground_truths, predictions), 4), valid_loss = None, valid_rmse = None, valid_r2 = None)
                data_iter.update(1)
                
        epoch_loss = round(sum(losses)/len(losses), 4)
        epoch_rmse, epoch_r2 = round(mean_squared_error(ground_truths, predictions, squared=False), 4), round(r2_score(ground_truths, predictions), 4)
        print("Epoch {} - Training Loss {}, R2 {}, RMSE {}.".format(epoch, epoch_loss, epoch_r2, epoch_rmse))
        return epoch_loss, epoch_rmse, epoch_r2

    def valid_one_epoch(self, epoch, train_loss, train_rmse, train_r2):
        ground_truths, predictions, losses = [], [], []
        correct = 0
        total = 0
        self.model.to('cpu')
        self.model.eval()
        with tqdm(total = len(self.test_loader), desc = "Epoch {} - Validation".format(epoch), unit = 'batch') as data_iter:
            data_iter.set_postfix(train_loss = round(train_loss, 2), train_rmse = round(train_rmse, 2), train_r2 = round(train_r2, 2), valid_loss = 'TBD', valid_rmse = 'TBD', valid_r2 = 'TBD')
            with torch.no_grad():
                for data in self.test_loader:
                    output, vloss = self.model(data, data.batch)

                    y_true = data.y.cpu().detach().numpy().flatten()
                    y_pred = output.cpu().detach().numpy().flatten()

                    ground_truths.extend(y_true)
                    predictions.extend(y_pred)
                    losses.append(vloss.item())
                    
                    data_iter.update(1)

            epoch_loss = np.mean(losses)
            epoch_rmse = mean_squared_error(ground_truths, predictions, squared=False)
            epoch_r2 = r2_score(ground_truths, predictions)

            data_iter.set_postfix(train_loss=round(train_loss, 4), train_r2=round(train_r2, 4),
                         valid_loss=round(epoch_loss, 4), valid_r2=round(epoch_r2, 4),
                         valid_rmse=round(epoch_rmse, 4))
        
        print("Epoch {} - Validation Loss {}, R2 {}, RMSE {}.".format(epoch, epoch_loss, epoch_r2, epoch_rmse))

        return epoch_loss, epoch_rmse, epoch_r2

    def train(self, epochs):
        train_loss, train_rmse, train_r2 = [], [], []
        valid_loss, valid_rmse, valid_r2 = [], [], []
        for epoch in range(epochs):
            tloss, trmse, tr2 = self.train_one_epoch(epoch)
            train_rmse.append(trmse)
            train_r2.append(tr2)
            train_loss.append(tloss)
            vloss, vrmse, vr2 = self.valid_one_epoch(epoch, tloss, trmse, tr2)
            valid_loss.append(vloss)
            valid_rmse.append(vrmse)
            valid_r2.append(vr2)
        return train_loss, train_rmse, train_r2, valid_loss, valid_rmse, valid_r2

    def predict(self, test_loader):
        predictions = []
        ground_truths = []
        self.model.to('cpu')
        self.model.eval()
        
        with torch.no_grad():
            data_iter = tqdm(test_loader, total = len(test_loader), desc = 'Predicting')
            for data in data_iter:
                output, _ = self.model(data, data.batch)

                y_true = data.y.cpu().detach().numpy().flatten()
                y_pred = output.cpu().detach().numpy().flatten()

                predictions.extend(y_pred)
                ground_truths.extend(y_true)
        data_iter.set_postfix(stage="testing")
        data_iter.close()

        predictions = np.array(predictions)
        ground_truths = np.array(ground_truths)

        rmse = mean_squared_error(ground_truths, predictions, squared=False)
        r2 = r2_score(ground_truths, predictions)

        print(f"Test RMSE: {rmse:.4f}")
        print(f"Test R²:   {r2:.4f}")
        
        plt.figure()
        plt.scatter(ground_truths, predictions, alpha=0.5, c='blue', label='Predictions')
        plt.plot([min(ground_truths), max(ground_truths)],
                [min(ground_truths), max(ground_truths)],
                'r--', label='Ideal (y = x)')
        plt.xlabel('Actual logS')
        plt.ylabel('Predicted logS')
        plt.title('Prediction vs Actual (logS)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return predictions, ground_truths