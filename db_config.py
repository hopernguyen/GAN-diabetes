config = {
    'exp': '1',
    'ckpt_name': 'generator.pt',
    'device': 'gpu',
    'optim': 'Adam',
    'lr': 0.0002,
    'epochs': 1000,
    'batch_size': 16,
    'save_interval': 1,
    'networks': {
        'discriminator': {
            'in_dim': 16,
            'h_dims': [20],
        },
        'generator': {
            'z_dim': 100,
            'in_dim': 100,
            'h_dims': [20],
            'out_dim': 16,
        }
    }
}
