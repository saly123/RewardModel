import math
import torch
from torch.optim.optimizer import Optimizer


class AdamWeightDecayOptimizer(Optimizer):
    """ Implements Adam algorithm with weight decay fix.
    """

    def __init__(self, params, lr=1e-4, global_step=0, warmup_steps=1e4, num_train_step=1e6, betas=(0.9, 0.999),
                 eps=1e-6, weight_decay=0.0, correct_bias=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, global_step=global_step,
                        warmup_steps=warmup_steps, num_train_step=num_train_step, correct_bias=correct_bias)
        super(AdamWeightDecayOptimizer, self).__init__(params, defaults)
        group = self.param_groups[0]
        self.lr = group["lr"]
        self.global_step = group["global_step"]
        self.warmup_steps = group["warmup_steps"]
        self.num_train_step = group["num_train_step"]

    def learning_rate_warmup_and_decay(self, init_lr, warmup_steps, num_train_step, global_step):
        if global_step < warmup_steps:
            ratio = global_step / warmup_steps
        else:
            ratio = (num_train_step - global_step) / (num_train_step - warmup_steps)
        return init_lr * ratio

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            group['lr'] = self.learning_rate_warmup_and_decay(self.lr, self.warmup_steps, self.num_train_step,
                                                              self.global_step)
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(1.0 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1.0 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                step_size = group['lr']
                if group['correct_bias']:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state['step']
                    bias_correction2 = 1.0 - beta2 ** state['step']
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group['weight_decay'] > 0.0:
                    p.data.add_(-group['lr'] * group['weight_decay'], p.data)

        self.global_step += 1

        return loss
