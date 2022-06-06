import torch.optim as optim

from koala import VanillaKOALA, MomentumKOALA


def _adadelta(parameters, **kwargs):
    return optim.Adadelta(parameters, **kwargs)


def _adagrad(parameters, **kwargs):
    return optim.Adagrad(parameters, **kwargs)


def _adam(parameters, **kwargs):
    return optim.Adam(parameters, **kwargs)


def _koala_v(parameters, **kwargs):
    return VanillaKOALA(parameters, **kwargs)


def _koala_m(parameters, **kwargs):
    return MomentumKOALA(parameters, **kwargs)


def _rmsprop(parameters, **kwargs):
    return optim.RMSprop(parameters, **kwargs)


def _sgd(parameters, **kwargs):
    return optim.SGD(parameters, **kwargs)


optimizers = {
    'adadelta': _adadelta,
    'adagrad': _adagrad,
    'adam': _adam,
    'koala-v': _koala_v,
    'koala-m': _koala_m,
    'rmsprop': _rmsprop,
    'sgd': _sgd
}
