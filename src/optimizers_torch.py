import torch
import torch.optim as optim

# For this project, we can wrap standard PyTorch optimizers or implement custom ones
# Here is a helper to get optimizers easily
def get_optimizer(name, model_params, lr=0.01, **kwargs):
    if name == 'sgd':
        return optim.SGD(model_params, lr=lr, momentum=0)
    elif name == 'momentum':
        return optim.SGD(model_params, lr=lr, momentum=kwargs.get('momentum', 0.9))
    elif name == 'nesterov':
        return optim.SGD(model_params, lr=lr, momentum=kwargs.get('momentum', 0.9), nesterov=True)
    elif name == 'adam':
        return optim.Adam(model_params, lr=lr)
    elif name == 'rmsprop':
        return optim.RMSprop(model_params, lr=lr)
    elif name == 'adagrad':
        return optim.Adagrad(model_params, lr=lr)
    else:
        raise ValueError(f"Optimizer {name} not supported")

