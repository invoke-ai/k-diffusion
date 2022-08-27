import math

import torch
from torch import nn

from . import sampling, utils


class VDenoiser(nn.Module):
    """A v-diffusion-pytorch model wrapper for k-diffusion."""

    def __init__(self, inner_model):
        super().__init__()
        self.inner_model = inner_model
        self.sigma_data = 1.

    def get_scalings(self, sigma):
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = -sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_skip, c_out, c_in

    def sigma_to_t(self, sigma):
        return sigma.atan() / math.pi * 2

    def t_to_sigma(self, t):
        return (t * math.pi / 2).tan()

    def loss(self, input, noise, sigma, **kwargs):
        c_skip, c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        noised_input = input + noise * utils.append_dims(sigma, input.ndim)
        model_output = self.inner_model(noised_input * c_in, self.sigma_to_t(sigma), **kwargs)
        target = (input - c_skip * noised_input) / c_out
        return (model_output - target).pow(2).flatten(1).mean(1)

    def forward(self, input, sigma, **kwargs):
        c_skip, c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        return self.inner_model(input * c_in, self.sigma_to_t(sigma), **kwargs) * c_out + input * c_skip


class DiscreteSchedule(nn.Module):
    """A mapping between continuous noise levels (sigmas) and a list of discrete noise
    levels."""

    def __init__(self, sigmas, quantize):
        super().__init__()
        self.register_buffer('sigmas', sigmas)
        self.quantize = quantize

    def get_sigmas(self, n=None):
        if n is None:
            return sampling.append_zero(self.sigmas.flip(0))
        t_max = len(self.sigmas) - 1
        t = torch.linspace(t_max, 0, n, device=self.sigmas.device)
        return sampling.append_zero(self.t_to_sigma(t))

    def sigma_to_t(self, sigma, quantize=None):
        quantize = self.quantize if quantize is None else quantize
        dists = torch.abs(sigma - self.sigmas[:, None])
        if quantize:
            return torch.argmin(dists, dim=0).view(sigma.shape)
        topk_indices = torch.topk(dists, dim=0, k=2, largest=False).indices
        topk_indices_device=topk_indices.device

        # TODO: revert this once MPS supports aten::sort.values_stable.
        # we're transferring the topk indices to CPU, sorting them, then transferring the result (sort_values) back to GPU.
        # it's fine to sort on-CPU, because it's a wee little 2x2 matrix.
        # PYTORCH_ENABLE_MPS_FALLBACK=1 would do the same thing. but I want us to be able to run without that.
        # so that we find out any time a fallback is required, and can review whether it's consequential.
        must_sort_on_cpu = topk_indices_device.type == 'mps'
        topk_indices = topk_indices.cpu() if must_sort_on_cpu else topk_indices

        sort_values = torch.sort(topk_indices, dim=0).values
        sort_values = sort_values.to(topk_indices_device) if must_sort_on_cpu else sort_values

        low_idx, high_idx = sort_values
        low, high = self.sigmas[low_idx], self.sigmas[high_idx]
        w = (low - sigma) / (low - high)
        w = w.clamp(0, 1)
        t = (1 - w) * low_idx + w * high_idx
        return t.view(sigma.shape)

    def t_to_sigma(self, t):
        t = t.float()
        low_idx, high_idx, w = t.floor().long(), t.ceil().long(), t-(t*t.sgn()).floor()*t.sgn() if t.device.type == 'mps' else t.frac()
        return (1 - w) * self.sigmas[low_idx] + w * self.sigmas[high_idx]


class DiscreteEpsDDPMDenoiser(DiscreteSchedule):
    """A wrapper for discrete schedule DDPM models that output eps (the predicted
    noise)."""

    def __init__(self, model, alphas_cumprod, quantize):
        super().__init__(((1 - alphas_cumprod) / alphas_cumprod) ** 0.5, quantize)
        self.inner_model = model
        self.sigma_data = 1.

    def get_scalings(self, sigma):
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_out, c_in

    def get_eps(self, *args, **kwargs):
        return self.inner_model(*args, **kwargs)

    def loss(self, input, noise, sigma, **kwargs):
        c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        noised_input = input + noise * utils.append_dims(sigma, input.ndim)
        eps = self.get_eps(noised_input * c_in, self.sigma_to_t(sigma), **kwargs)
        return (eps - noise).pow(2).flatten(1).mean(1)

    def forward(self, input, sigma, **kwargs):
        c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        eps = self.get_eps(input * c_in, self.sigma_to_t(sigma), **kwargs)
        return input + eps * c_out


class OpenAIDenoiser(DiscreteEpsDDPMDenoiser):
    """A wrapper for OpenAI diffusion models."""

    def __init__(self, model, diffusion, quantize=False, has_learned_sigmas=True, device='cpu'):
        alphas_cumprod = torch.tensor(diffusion.alphas_cumprod, device=device, dtype=torch.float32)
        super().__init__(model, alphas_cumprod, quantize=quantize)
        self.has_learned_sigmas = has_learned_sigmas

    def get_eps(self, *args, **kwargs):
        model_output = self.inner_model(*args, **kwargs)
        if self.has_learned_sigmas:
            return model_output.chunk(2, dim=1)[0]
        return model_output


class CompVisDenoiser(DiscreteEpsDDPMDenoiser):
    """A wrapper for CompVis diffusion models."""

    def __init__(self, model, quantize=False, device='cpu'):
        super().__init__(model, model.alphas_cumprod, quantize=quantize)

    def get_eps(self, *args, **kwargs):
        return self.inner_model.apply_model(*args, **kwargs)
