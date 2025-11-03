import logging

logger = logging.getLogger('global')

def calculate_parameters(model):
    total_num  = sum(p.numel() for p in model.parameters())
    trainable_num  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('Number of parameters: total: {}M, trainable: {}M'.format(round(total_num/1e6,1), round(trainable_num/1e6,1)))