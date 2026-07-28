from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F

from torchdiffeq import odeint
from einops import rearrange, repeat


def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def append_dims(t, ndims):
    shape = t.shape
    return t.reshape(*shape, *((1,) * ndims))


class MSELoss(Module):
    def forward(self, pred, target, padding_mask=None):
        if padding_mask is None:
            return F.mse_loss(pred, target)
        loss = F.mse_loss(pred, target, reduction='none')
        mask = ~padding_mask  # padding_mask는 패딩인 곳이 True라서, 반대로 뒤집어서 진짜 데이터인 곳만 loss에 반영
        mask = mask.unsqueeze(-1).float()
        return (loss * mask).mean(dim=-1).sum() / (mask.sum() + 1e-8)


class RectifiedFlowDecoder(Module):
    """
    x0(noise)에서 x1(data)까지 선형보간 경로(x_t = t*x1 + (1-t)*x0)를 학습하는 flow matching 디코더.
    predict='flow' + MSE loss만 쓰는, 이 프로젝트 실사용 경로만 남긴 버전
    (noise_schedule/consistency-FM(EMA)/immiscible flow/sampling clipping/데이터 정규화 훅 등은 제거함).
    """

    def __init__(
        self,
        model: Module,
        odeint_kwargs: dict = dict(
            atol=1e-5,
            rtol=1e-5,
            method='midpoint',
        ),
        data_shape: Tuple[int, ...] | None = None,
    ):
        super().__init__()
        self.net = model
        self.loss_fn = MSELoss()
        self.odeint_kwargs = odeint_kwargs
        self.data_shape = data_shape

    @property
    def device(self):
        return next(self.net.parameters()).device

    def predict_flow(self, noised, *, times, y, padding_mask=None, text_embedding=None):
        """noised(x_t), times(t), y(조건, z)를 받아 self.net으로 flow(속도)를 예측"""
        batch = noised.shape[0]

        times = rearrange(times, '... -> (...)')
        if times.numel() == 1:
            times = repeat(times, '1 -> b', b=batch)

        return self.net(noised, times=times, y=y, padding_mask=padding_mask, text_embedding=text_embedding)

    @torch.no_grad()
    def sample(
        self,
        y,  # 조건 (z)
        batch_size=1,
        steps=16,
        noise=None,  # x0. 안 주면 순수 가우시안 노이즈에서 시작 (refinement에서는 decoder(z)+noise를 넘김)
        padding_mask=None,
        text_embedding=None,
        data_shape: Tuple[int, ...] | None = None,
    ):
        was_training = self.training
        self.eval()

        data_shape = default(data_shape, self.data_shape)
        assert exists(data_shape), 'you need to either pass in a `data_shape` or have trained at least with one forward'

        def ode_fn(t, x):
            return self.predict_flow(x, times=t, y=y, padding_mask=padding_mask, text_embedding=text_embedding)

        noise = default(noise, torch.randn((batch_size, *data_shape), device=self.device))

        times = torch.linspace(0., 1., steps, device=self.device)
        trajectory = odeint(ode_fn, noise, times, **self.odeint_kwargs)
        sampled_data = trajectory[-1]

        self.train(was_training)
        return sampled_data

    def forward(self, data, y, padding_mask=None, text_embedding=None, noise: Tensor | None = None):
        batch, *data_shape = data.shape
        self.data_shape = default(self.data_shape, data_shape)

        # x0 - noise (안 주면 순수 가우시안), x1 - data
        noise = default(noise, torch.randn_like(data))

        times = torch.rand(batch, device=self.device)
        padded_times = append_dims(times, data.ndim - 1)

        # Algorithm 2: x1*t + x0*(1-t) - t=0이면 noise, t=1이면 data
        noised = padded_times * data + (1. - padded_times) * noise
        target_flow = data - noise  # 직선 경로의 속도(상수) = 정답 flow

        pred_flow = self.predict_flow(noised, times=padded_times, y=y, padding_mask=padding_mask, text_embedding=text_embedding)

        return self.loss_fn(pred_flow, target_flow, padding_mask=padding_mask)
