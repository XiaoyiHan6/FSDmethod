from torch import optim


def get_optimizer(cfg, model):
    optimizer = cfg['Optimize']['optim']
    lr = cfg['Optimize']['lr']

    # 1.
    # not recommended
    if optimizer == 'sgd':
        # lr = 0.001
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4) # 1e-4
    # 2.
    # not recommended
    elif optimizer == 'asgd':
        optimizer = optim.ASGD(model.parameters(), lr=lr, weight_decay=1e-4)
    # 3.
    # not recommended
    elif optimizer == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=lr)
    # 4.
    elif optimizer == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=lr)
    # 5.
    elif optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-5)
    # 6.
    elif optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-5, weight_decay=1e-2)
    # 7.
    elif optimizer == 'radam':
        optimizer = optim.RAdam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    # 8.
    elif optimizer == 'adamax':
        optimizer = optim.Adamax(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    # 9.
    elif optimizer == 'nadam':
        optimizer = optim.NAdam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    # 10.
    elif optimizer == 'sparseadam':
        optimizer = optim.SparseAdam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    # 11.
    # not recommended
    elif optimizer == 'rprop':
        optimizer = optim.Rprop(model.parameters(), lr=lr)
    # 12.
    elif optimizer == 'rmsprop':
        # lr = 0.001
        optimizer = optim.RMSprop(model.parameters(), lr=lr, momentum=0.9, eps=1e-8)
    else:
        raise NotImplementedError
    return optimizer
