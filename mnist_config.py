config = {
    'exp': '6',
    'device': 'gpu',
    'optim': 'Adam',
    'lr': 0.0002,
    'epochs': 100,
    'batch_size': 256,
    'save_interval': 1,
    'networks': {
        'discriminator': {
            'in_dim': 784,
            'h_dims': [600, 300],
        },
        'generator': {
            'z_dim': 100,
            'in_dim': 100,
            'h_dims': [300, 600],
            'out_dim': 784,
        }
    }
}
