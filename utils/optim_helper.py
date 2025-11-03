import torch
import logging
import math
import importlib
from torch.nn.utils import clip_grad_norm_
from utils.misc_helper import build_cls_instance
logger = logging.getLogger('global')

class GradClipper(object):
    """
    Clip gradients by
    1. provided max_norm
    2. average total_norm from watch_iter iterations
    """

    def __init__(self, max_norm=0, norm_type=2, watch_iter=0):
        assert (max_norm > 0 and watch_iter == 0) or (max_norm == 0 and watch_iter > 0)
        self.watch_iter = watch_iter
        self.last_iter = 0
        self.max_norm = max_norm
        self.norm_type = norm_type

    def update_max_norm(self, parameters):
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(self.norm_type)
            total_norm += param_norm ** self.norm_type
        total_norm = total_norm ** (1. / self.norm_type)
        logger.info('total_norm:{}'.format(total_norm))
        self.max_norm += total_norm
        self.last_iter += 1

    def clip_grad(self, parameters):
        parameters = [p for p in parameters if p.requires_grad and p.grad is not None]
        if self.last_iter < self.watch_iter:
            self.update_max_norm(parameters)
            if self.last_iter == self.watch_iter:
                self.max_norm /= self.watch_iter
                logger.info('max_norm:{}'.format(self.max_norm))
        else:
            clip_grad_norm_(parameters, self.max_norm, self.norm_type)

def build_optimizer(cfg_optim, model):
    if 'groups' not in cfg_optim:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
    else:
        default_lr = float(cfg_optim['kwargs']['lr'])
        trainable_params = model.group_params_by_names(cfg_optim['groups'], default_lr)

    cfg_optim['kwargs']['params'] = trainable_params
    optim_type = cfg_optim['type']
    if '.' in optim_type:
        module_name, cls_name = optim_type.rsplit('.', 1)
    else:
        module_name, cls_name = 'torch.optim', optim_type
    if cls_name == 'PreciseAdjustAdamW':
        x = model.named_modules()
        for name, module in model.named_modules():
            if name.split('.')[-1] == 'clip':
                cfg_optim['kwargs']['clip'] = module
                break
    module = importlib.import_module(module_name)
    cls = getattr(module, cls_name)
    optimizer = cls(**cfg_optim['kwargs'])
    logger.info('build optimizer done')
    return optimizer

def warmup_cosine_annealing_lr(step, total_steps, warmup_steps):
    """
    Cosine annealing learning rate with warmup.
    return lambda1 as multiplier for base learning rate.
    """
    # Warmup
    if step < warmup_steps:
        lambda1 =  step / warmup_steps 
    # Cosine Annealing 
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        lambda1 = (1 + math.cos(math.pi * progress)) / 2
    return lambda1

def build_scheduler(cfg_sch, optimizer, last_epoch):
    cfg_sch['kwargs']['optimizer'] = optimizer
    cfg_sch['kwargs']['last_epoch'] = last_epoch
    if cfg_sch['type'] == "LambdaLR":
        # linear warmup + cosine annealing
        warmup_epoch = cfg_sch['kwargs']['warmup_epoch']
        T_max = cfg_sch['kwargs']['max_epoch']
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, 
                       lr_lambda=lambda step: warmup_cosine_annealing_lr(step, T_max, warmup_epoch), last_epoch=last_epoch, verbose=True)
    else:    
        lr_scheduler = build_cls_instance(torch.optim.lr_scheduler, cfg_sch)
    return lr_scheduler

def build_grad_clipper(cfg_clipper):
    if cfg_clipper is None:
        return None
    clipper = GradClipper(**cfg_clipper)
    logger.info('build gradcliper done')
    return clipper

