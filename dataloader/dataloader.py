import logging
import importlib
from torch.utils.data import DataLoader, WeightedRandomSampler, RandomSampler
from torch.utils.data._utils.collate import default_collate

logger = logging.getLogger('global')
def build_dataloader(cfg, training):
    def get_dataset_cls(cfg):
        module_name, cls_name = cfg['type'].rsplit('.', 1)
        module = importlib.import_module(module_name)
        return getattr(module, cls_name)
    def get_sampler(dataset, cfg):
        if not cfg['train']['data_balance']:
            return RandomSampler(dataset, replacement=False)
        #use first score_key to data balance
        score_round = dataset.metas_combine[cfg['score_key_list'][0]].round()
        score_counts = score_round.value_counts()
        weights_map = 1 / score_counts
        score_weights = score_round.map(weights_map).to_numpy()
        sampler = WeightedRandomSampler(score_weights, len(score_weights), replacement=True)
        return sampler
    def flat_pair_collate(batch):
        batch = [item for pair in batch for item in pair]
        return default_collate(batch)

    action = 'train' if training else 'test'
    is_pair = cfg[action].get('is_pair', False)
    pad_mode = cfg[action].get('pad_mode', 'rightdown')
    logger.info("is_pair is set to {}, the data will shuffle pairedly when training if is_pair is True".format(is_pair))
    batch_size = cfg[action].get('batch_size', 16)
    logger.info('%s: building dataloader from %s' % (action, cfg[action]['meta_file_list']))
    dataset_cls = get_dataset_cls(cfg)
    dataset = dataset_cls(training=training,
                          images_dir_list=cfg[action]['images_dir_list'],
                          meta_file_list=cfg[action]['meta_file_list'],
                          score_key_list=cfg['score_key_list'],
                          resize_size=cfg['resize_size'],
                          is_pair=is_pair,
                          pad_mode=pad_mode)

    sampler = get_sampler(dataset, cfg) if training else None

    loader = DataLoader(dataset,
                        batch_size=batch_size if not is_pair else batch_size // 2,
                        num_workers=cfg[action].get('workers', 4),
                        pin_memory=True, sampler=sampler,
                        collate_fn=None if not is_pair else flat_pair_collate)
    return loader

if __name__ == '__main__':
    import yaml
    cfg = yaml.load(open('../experiments/exp36_swin_facenet_ca_pseudo/config.yml'), Loader=yaml.Loader)
    train_loader = build_dataloader(cfg['dataset'], training=True)

    for i, data in enumerate(train_loader):
        print(data)
    
    for i, data in enumerate(train_loader):
        print(data)
