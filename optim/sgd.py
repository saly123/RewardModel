from torch.optim.optimizer import Optimizer


class SGD(Optimizer):

    def __init__(self, params, lr=1e-4, global_step=0, warmup_steps=1e4, num_train_step=1e6, weight_decay=0.0):
        defaults = dict(lr=lr, global_step=global_step, warmup_steps=warmup_steps, num_train_step=num_train_step,
                        weight_decay=weight_decay)
        super(SGD, self).__init__(params, defaults)
        group = self.param_groups[0]
        self.lr = group['lr']
        self.global_step = group['global_step']
        self.warmup_steps = group["warmup_steps"]
        self.num_train_step = group['num_train_step']

    def learning_rate_warmup_and_decay(self, init_lr, warmup_steps, num_train_step, global_step):
        if global_step < warmup_steps:
            ratio = global_step / warmup_steps
        else:
            ratio = (num_train_step - global_step) / (num_train_step - warmup_steps)
        return init_lr * ratio

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            group['lr'] = self.learning_rate_warmup_and_decay(self.lr, self.warmup_steps, self.num_train_step,
                                                              self.global_step)

            for p in group['params']:
                if p.grad is None:
                    continue
                p_data_fp32 = p.data.float()
                grad = p.grad.data.float()

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p_data_fp32)

                p_data_fp32.add_(-group['lr'], grad)
                p.data.copy_(p_data_fp32)

        self.global_step += 1

        return loss
