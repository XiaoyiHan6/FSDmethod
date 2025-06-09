from torch import optim


def get_optimizer(cfg, model):
    optimizer = cfg['Optimize']['optim']
    lr = cfg['Optimize']['lr']

    # 1.
    if optimizer == 'sgd':
        # lr = 0.001
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    # 2.
    elif optimizer == 'asgd':
        optimizer = optim.ASGD(model.parameters(), lr=lr, weight_decay=5e-4)
    # 3.
    elif optimizer == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=lr)
    # 4.
    elif optimizer == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=lr)
    # 5.
    elif optimizer == 'adam':
        # lr = 5e-4
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.95, 0.999), eps=5e-4)
    # 6.
    elif optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.95, 0.999), eps=5e-4)
    # 7.
    elif optimizer == 'radam':
        optimizer = optim.RAdam(model.parameters(), lr=lr, betas=(0.95, 0.999), eps=5e-4)
    # 8.
    elif optimizer == 'adamax':
        optimizer = optim.Adamax(model.parameters(), lr=lr, betas=(0.95, 0.999), eps=5e-4)
    # 9.
    elif optimizer == 'nadam':
        optimizer = optim.NAdam(model.parameters(), lr=lr, betas=(0.95, 0.999), eps=5e-4)
    # 10.
    elif optimizer == 'sparseadam':
        optimizer = optim.SparseAdam(model.parameters(), lr=lr, betas=(0.95, 0.999), eps=5e-4)
    # 11.
    elif optimizer == 'rprop':
        optimizer = optim.Rprop(model.parameters(), lr=lr)
    # 12.
    elif optimizer == 'rmsprop':
        # lr = 0.001
        optimizer = optim.RMSprop(model.parameters(), lr=lr, momentum=0.9, eps=5e-4)
    else:
        raise NotImplementedError
    return optimizer
