import torch
import torchvision
import pandas as pd
import numpy as np
from sklearn.metrics import jaccard_score

np.random.seed(0)


def load_mnist_data(split='train', batch_size=None):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = torchvision.datasets.MNIST(root='datasets',
                                         train=(split == 'train'),
                                         download=True,
                                         transform=transforms)
    dataset = torch.utils.data.Subset(dataset, [i for i in range(len(dataset)) if dataset[i][1] < 2])
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'))
    return loader


def load_diabetes_data(file=None, label=None, batch_size=None):
    """
    Read diabetes data from csv.
    """
    # Read data.
    df = pd.read_csv(file)

    # Numerically encode.
    df = df.replace('Yes', 1)
    df = df.replace('No', -1)
    df = df.replace('Male', 1)
    df = df.replace('Female', -1)
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
    age_range = np.max(x[:, 0]) - np.min(x[:, 0])
    scaled_ages = np.expand_dims(-1 + ((x[:, 0] - np.min(x[:, 0])) * 2) / age_range, axis=1)
    x = np.hstack((scaled_ages, x[:, 1:]))

    # Only take a subset of data if label is provided.
    if label is not None:
        assert label == 1 or label == 0
        index = [i for i in range(len(y)) if y[i] == label]
        x = x[index]

    # Tensorify.
    dataset = torch.utils.data.TensorDataset(torch.tensor(x))
    dataset = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    return dataset


def compare_fake_real():
    real = pd.read_csv('datasets/diabetes_data_upload.csv')
    fake = pd.read_csv('datasets/synthetic.csv')
    real = real.replace('Yes', 1)
    real = real.replace('No', 0)
    real = real.replace('Male', 1)
    real = real.replace('Female', 0)
    real = real.replace('Positive', 1)
    real = real.replace('Negative', 0)

    fake = fake.replace('Yes', 1)
    fake = fake.replace('No', 0)
    fake = fake.replace('Male', 1)
    fake = fake.replace('Female', 0)
    fake = fake.replace('Positive', 1)
    fake = fake.replace('Negative', 0)

    real = real.to_numpy()
    fake = fake.to_numpy()

    real = ([real[i] for i in range(real.shape[0])])
    fake = ([fake[i] for i in range(fake.shape[0])])
    repeats = []
    for f in fake:
        for r in real:
            if np.array_equal(r, f):
                count.append(f)
    print(repeats)


def merge_synthetic_data(pos, neg):
    """
    This function merges data and unnorm age
    pos_path: generated samples with postive labels 
    neg_path: generated samples with negative labels paths
    """
    # Read data and get age stats for normalization.
    df = pd.read_csv('datasets/diabetes_data_upload.csv')
    data = df.to_numpy()
    x = data[:, :-1]

    min_found = np.min(x[:, 0])
    age_range = np.max(x[:, 0]) - np.min(x[:, 0])
    pos[0] = pos[0].apply(lambda scaled_ages: np.round(min_found + ((scaled_ages + 1) * age_range) / 2, decimals=2))
    neg[0] = neg[0].apply(lambda scaled_ages: np.round(min_found + ((scaled_ages + 1) * age_range) / 2, decimals=2))
    combined = pd.concat([pos, neg])

    # Changing the 1 and 0 to labels such as Yes, No etc
    combined.columns = df.columns

    for index in combined.columns[2:16]:
        combined[index].replace(to_replace=1.0, value='Yes', inplace=True)
        combined[index].replace(to_replace=-1.0, value='No', inplace=True)

    combined[combined.columns[1]].replace(to_replace=1.0, value='Male', inplace=True)
    combined[combined.columns[1]].replace(to_replace=-1.0, value='Female', inplace=True)
    combined[combined.columns[16]].replace(to_replace=1.0, value='Positive', inplace=True)
    combined[combined.columns[16]].replace(to_replace=0.0, value='Negative', inplace=True)

    combined.to_csv('datasets/synthetic.csv', index=False)

    print(combined.head())


if __name__ == '__main__':
    compare_fake_real()
