"""
Shared optimizer, the parameters in the optimizer will shared in the multiprocessors.
"""

import torch


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(())
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

    def __setstate__(self, state):
        super().__setstate__(state)
        # Spawned workers need their own Adam bias-correction counters.
        for param_state in self.state.values():
            if 'step' in param_state:
                param_state['step'] = param_state['step'].clone()
