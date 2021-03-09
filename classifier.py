import torch
import torchvision
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

torch.manual_seed(0)
np.random.seed(10)


def load_data(csv_path):
    """
    Read diabetes data from csv.
    """
    # Read data.
    df = pd.read_csv(csv_path)

    # Numerically encode.
    df = df.replace('Yes', 1)
    df = df.replace('No', 0)
    df = df.replace('Male', 1)
    df = df.replace('Female', 0)
    df = df.replace('Positive', 1)
    df = df.replace('Negative', 0)

    # Numpify and shuffle
    data = df.to_numpy()
    x = data[:, :-1]

    y = data[:, -1]
    shuffle_perm = np.random.permutation(x.shape[0])
    x = x[shuffle_perm]
    y = y[shuffle_perm]

    # Preprocess data.
    pass

    # Split into training and testing partitions.
    train_x = x[:int(0.5 * x.shape[0]), :]
    train_y = y[:int(0.5 * y.shape[0])]
    test_x = x[int(0.5 * x.shape[0]):, :]
    test_y = y[int(0.5 * y.shape[0]):]

    return {
        'x': x,
        'y': y,
        'train_x': train_x,
        'train_y': train_y,
        'test_x': test_x,
        'test_y': test_y
    }


class Classifier(torch.nn.Module):
    """
    Neural network classifier for the diabetes dataset.
    """

    def __init__(self, hdim1):
        super(Classifier, self).__init__()
        self.input = torch.nn.Linear(16, hdim1)
        self.fc1 = torch.nn.Linear(hdim1, 1)
        self.out_activation = torch.nn.Sigmoid()
        self.hidden_activation = torch.nn.ReLU()

    def forward(self, x):
        z = self.input(x)
        z = self.hidden_activation(z)
        logits = self.fc1(z)
        return logits


def train():
    # Hyper-parameters.
    epochs = 10000
    batch_size = 64

    # Model and data.
    model = Classifier(hdim1=32).cuda()
    # data = load_data('datasets/diabetes_data_upload.csv')
    synthetic_data = load_data('datasets/synthetic.csv')
    train_x = synthetic_data['x']
    train_y = synthetic_data['y']
    # train_x = np.concatenate((data['train_x'], synthetic_data['x']), axis=0)
    # train_y = np.concatenate((data['train_y'], synthetic_data['y']), axis=0)
    # train_x = data['train_x']
    # train_y = data['train_y']

    # Loss and optimizers.
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # Train loop.
    for i in range(epochs):
        for batch_idx in range(0, train_x.shape[0] - batch_size, batch_size):
            # Get batch.
            x = torch.tensor(train_x[batch_idx: batch_idx + batch_size]).float().cuda()
            y = torch.tensor(train_y[batch_idx: batch_idx + batch_size]).float().cuda()

            # Zero out accumulated gradients.
            optimizer.zero_grad()

            # Forward.
            logits = model(x)
            logits = torch.squeeze(logits)

            # Loss.
            loss = criterion(logits, y)
            loss.backward()

            # Backward.
            optimizer.step()

            # Logging.
            y_true = y.detach().cpu().numpy()
            y_pred = model.out_activation(logits).detach().cpu().numpy()
            y_pred = np.round(y_pred)
            acc = accuracy_score(y_true, y_pred)
            print("epoch: {} | batch: {} | loss: {} | batch accuracy: {}".format(
                i, batch_idx, loss, acc
            ))
    return model


def test(model):
    # Model and data.
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    data = load_data('datasets/diabetes_data_upload.csv')

    # Metrics variables.
    batch_size = 32

    # Test.
    logits = model(torch.tensor(data['test_x'].astype('float32')).cuda().float())
    y_pred = np.round(model.out_activation(logits).detach().cpu().numpy())
    y_true = data['test_y'].astype('float32')
    metrics = classification_report(y_true, y_pred)
    print(metrics)


def main():
    model = train()
    test(model)


if __name__ == '__main__':
    main()
