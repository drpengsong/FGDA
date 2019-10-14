from solver import mySolver

config = {
    'batch_size': [16, 10],
    'epoch': 100,
    'lr': 1e-3,
    'weight_decay': 5e-4,
    'momentum': 0.9,

    'lr_f': 10,
    'lr_c': 5,
    'lr_d': 2,

    'FC/D': 9,

    'domain': 1.0,
    'class': 0.1,

    'ori' : 0.9,


    'random' : False

}

solver = mySolver(config)
ret = solver.train()
