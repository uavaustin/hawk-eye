""" An expotential moving average implementation for PyTorch. TensorFlow has an
implementation here:
https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage """

import copy

import torch


class Ema:
    def __init__(self, model: torch.nn.Module, decay: float = 0.995):
        self.decay = decay
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        self.ema_model = copy.deepcopy(model)

        # Make sure the model is on the cpu. Since we aren't training this model
        # directly, we want to free up as much space on the gpu.
        self.ema_model.cpu()
        self.ema_model.eval()

    def update(self, model: torch.nn.Module) -> None:
        """ Pass in the base model in order to update the ema model's parameters. """

        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        # Loop over the state dictionary of the incoming model and update the ema model.
        with torch.no_grad():
            model_state_dict = model.state_dict()
            for key, shadow_val in self.ema_model.state_dict().items():
                model_val = model_state_dict[key].detach().cpu()
                shadow_val.copy_(
                    shadow_val * self.decay + (1.0 - self.decay) * model_val
                )
