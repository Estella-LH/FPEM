import logging
import os
import shutil
import torch
import matplotlib.pyplot as plt

logger = logging.getLogger('global')


class Saver:
    def __init__(self, save_dir, model, cfg_path=None):
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.save_meta(model)
        if cfg_path:
            cfg_name = os.path.basename(cfg_path)
            dst_path = os.path.join(self.save_dir, cfg_name)
            shutil.copy(cfg_path, dst_path)

    def remove_prefix(self, state_dict, prefix):
        """Old style model is stored with all names of parameters share common prefix 'module.'"""
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}
    
    def add_prefix(self, state_dict, prefix):
        """The params loaded from pretrain model share common prefix 'backbone.' in new model"""
        return {prefix+key: value for key, value in state_dict.items()}

    def get_model_from_ckpt(self, ckpt_paths):
        """Get model state_dict from checkpoint"""
        ckpt_paths = ckpt_paths if isinstance(ckpt_paths, list) else [ckpt_paths]
        state_dict = {}
        for ckpt_path in ckpt_paths:
            assert os.path.exists(ckpt_path), 'No such file %s' % ckpt_paths
            logger.info('load checkpoint from {}'.format(ckpt_path))
            ckpt_dict = torch.load(ckpt_path, map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device()))
            if 'model' in ckpt_dict:
                _state_dict = ckpt_dict['model']
            elif 'state_dict' in ckpt_dict:
                _state_dict = ckpt_dict['state_dict']
            elif 'state_dict_backbone' in ckpt_dict:
                _state_dict = ckpt_dict['state_dict_backbone']
                _state_dict = self.add_prefix(_state_dict, 'backbone.')
            else:
                _state_dict = ckpt_dict
            state_dict.update(_state_dict)
        
        state_dict = self.remove_prefix(state_dict, 'module.')
        return state_dict

    def restore_optimizer(self, optimizer, ckpt_path):
        """Get optimizer from checkpoint"""
        assert os.path.exists(ckpt_path), f'No such file: {ckpt_path}'
        logger.info('restore optimizer from {}'.format(ckpt_path))
        device = torch.cuda.current_device()
        ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage.cuda(device))
        epoch = ckpt['epoch']
        optimizer.load_state_dict(ckpt['optimizer'])
        return optimizer, epoch

    def save(self, best, epoch, **kwargs):
        """Save model checkpoint for one epoch"""
        os.makedirs(self.save_dir, exist_ok=True)
        # Assume we warmup for a epochs and training a+b epochs in total,
        # then our checkpoints are named of ckpt_e{-a+1}.pth ~ ckpt_e{b}.pth
        if best:
            ckpt_path = os.path.join(self.save_dir, 'ckpt_e_best.pth')
        else:
            ckpt_path = os.path.join(self.save_dir, 'ckpt_e_{}.pth'.format(epoch))
        kwargs['epoch'] = epoch
        torch.save(kwargs, ckpt_path)
        return ckpt_path

    def save_meta(self, model):
        """Save model structure"""
        os.makedirs(self.save_dir, exist_ok=True)
        meta_path = os.path.join(self.save_dir, 'ckpt_meta.txt')
        with open(meta_path, 'w') as fid:
            fid.write(str(model))

def plot_scatter(labels_all, preds_all, idx, results_dir, epoch):
    """
    plot scatter diagram
    """
    # set font format
    plt.rcParams.update({'axes.titlesize': 14, 'axes.titleweight': 'bold',
                        'axes.labelsize': 14, 'axes.labelweight': 'bold',
                        'xtick.labelsize': 13, 'ytick.labelsize': 13})        
    # plot scatter diagram          
    plt.scatter(labels_all[:,idx].flatten().tolist(), preds_all[:,idx].flatten().tolist())
    max_value = max(max(labels_all[:,idx].flatten().tolist()), max(preds_all[:,idx].flatten().tolist()))
    plt.plot([0, max_value], [0, max_value], color='black',linestyle='--',linewidth=1,label='Diagonal Reference')
    plt.xlabel('MOS')
    plt.ylabel('Predicted MOS')
    plt.title('Scatter Diagram: MOS vs Predicted MOS')
    plt.legend()
    plt.savefig(os.path.join(results_dir, "{}epoch_scatter_diagram.png".format(epoch)))