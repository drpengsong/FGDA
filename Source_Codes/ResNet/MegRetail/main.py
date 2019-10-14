from solver import mySolver

config = {
    'batch_size': [16, 5],
    'epoch': 200,
    'lr': 1e-4,
    'weight_decay': 5e-4,
    'momentum': 0.9,

    'lr_f': 10,
    'lr_c': 5,
    'lr_d': 2,

    'FC/D': 10,

    'domain': 1.0,
    'class': -1.0,
    'ori': 0.3,

    'random': True,

    'source': 'sku',
    'target': 'shelf',

}

# config = {
#     'batch_size': [16, 96],
#     'epoch': 100,
#     'lr': 1e-4,
#     'weight_decay': 5e-4,
#     'momentum': 0.9,
#
#     'lr_f': 10,
#     'lr_c': 5,
#     'lr_d': 2,
#
#     'FC/D': 10,
#
#     'domain': 0.2,
#     'class': 0.01,
#     'ori': 0.9,
#
#     'random': False,
#
#     'source': 'sku',
#     'target': 'web',
#
# }

# config = {
#     'batch_size': [16, 12],
#     'epoch': 100,
#     'lr': 1e-4,
#     'weight_decay': 5e-4,
#     'momentum': 0.9,
#
#     'lr_f': 10,
#     'lr_c': 5,
#     'lr_d': 2,
#
#     'FC/D': 10,
#
#     'domain': 0.5,
#     'class': -0.1,
#     'ori': 0.8,
#
#     'random': False,
#
#     'source': 'shelf',
#     'target': 'sku',
#
#
# }

# config = {
#     'batch_size': [14, 128],
#     'epoch': 100,
#     'lr': 1e-4,
#     'weight_decay': 5e-4,
#     'momentum': 0.9,
#
#     'lr_f': 10,
#     'lr_c': 5,
#     'lr_d': 2,
#
#     'FC/D': 10,
#
#     'domain': 0.9,
#     'class': -0.1,
#     'ori': 0.9,
#
#     'random': False,
#
#     'source': 'shelf',
#     'target': 'web',
#
# }

# config = {
#     'batch_size': [128, 5],
#     'epoch': 100,
#     'lr': 1e-4,
#     'weight_decay': 5e-4,
#     'momentum': 0.9,
#
#     'lr_f': 10,
#     'lr_c': 5,
#     'lr_d': 2,
#
#     'FC/D': 10,
#
#     'domain': 0.1,
#     'class': 0.01,
#     'ori': 0.8,
#
#     'random': True,
#
#     'source': 'web',
#     'target': 'sku',
#
# }
# config = {
#     'batch_size': [256, 6],
#     'epoch': 100,
#     'lr': 1e-4,
#     'weight_decay': 5e-4,
#     'momentum': 0.9,
#
#     'lr_f': 10,
#     'lr_c': 5,
#     'lr_d': 2,
#
#     'FC/D': 10,
#
#     'domain': 1.0,
#     'class': 0.001,
#     'ori': 0.9,
#
#     'random': False,
#
#     'source': 'web',
#     'target': 'shelf',
#
# }

solver = mySolver(config)
solver.train()
