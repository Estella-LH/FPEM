import copy
import importlib
import logging
from collections import OrderedDict
import torch.nn as nn

logger = logging.getLogger('global')


class ModelHelper(nn.Module):
    """Build model from cfg"""

    def __init__(self, cfg):
        super(ModelHelper, self).__init__()
        for cfg_subnet in cfg:
            mname = cfg_subnet['name']
            mtype = cfg_subnet['type']
            kwargs = cfg_subnet['kwargs']
            module = self.build(mtype, kwargs)
            self.add_module(mname, module)

    def build(self, mtype, kwargs):
        module_name, cls_name = mtype.rsplit('.', 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, cls_name)
        return cls(**kwargs)

    def forward(self, x):
        for submodule in self.children():
            x = submodule(x)
        return x

    def check_share_keys_size(self, other_state_dict, shared_keys, model_keys):
        new_state_dict = OrderedDict()
        for k in shared_keys:
            if k in model_keys:
                if self.state_dict()[k].shape == other_state_dict[k].shape:
                    new_state_dict[k] = other_state_dict[k]
        return new_state_dict

    def fusion_ac_weights(self):
        for module in self.modules():
            for _, v in ac_cfg.items():
                if isinstance(module, v['block']):
                    module._fusion_weights()

    def load(self, other_state_dict, strict=False):
        """
        1. load resume model or pretained model
        2. load pretrained clssification model
        """
        logger.info("try to load the whole resume model or pretrained model...")
        model_keys = self.state_dict().keys()
        other_keys = other_state_dict.keys()
        shared_keys, unexpected_keys, missing_keys \
            = self.check_keys(model_keys, other_keys, 'model')
        # check shared_keys size
        new_state_dict = self.check_share_keys_size(
            other_state_dict, shared_keys, model_keys)
        shared_keys, unexpected_keys, missing_keys \
            = self.check_keys(model_keys, new_state_dict.keys(), 'model')
        self.load_state_dict(new_state_dict, strict=strict)

        num_share_keys = len(shared_keys)
        if num_share_keys == 0:
            logger.info(
                'failed to load the whole model directly,'
                'try to load each part seperately...'
            )
            for mname, module in self.named_children():
                module.load_state_dict(other_state_dict, strict=strict)
                module_keys = module.state_dict().keys()
                other_keys = other_state_dict.keys()

                # check and display info module by module
                shared_keys, unexpected_keys, missing_keys, \
                    = self.check_keys(module_keys, other_keys, mname)
                self.display_info(mname, shared_keys, unexpected_keys, missing_keys)
                num_share_keys += len(shared_keys)
                if num_share_keys == len(other_keys):
                    break
        else:
            self.display_info("model", shared_keys, unexpected_keys, missing_keys)
        self.fusion_ac_weights()
        return num_share_keys

    def check_keys(self, own_keys, other_keys, own_name):
        own_keys = set(own_keys)
        other_keys = set(other_keys)
        shared_keys = own_keys & other_keys
        unexpected_keys = other_keys - own_keys
        missing_keys = own_keys - other_keys
        return shared_keys, unexpected_keys, missing_keys

    def display_info(self, mname, shared_keys, unexpected_keys, missing_keys):
        info = "load {}:{} shared keys, {} unexpected keys, {} missing keys.".format(
            mname, len(shared_keys), len(unexpected_keys), len(missing_keys))

        if len(missing_keys) > 0:
            info += "\nmissing keys are as follows:\n    {}".format("\n    ".join(missing_keys))
        logger.info(info)
        
    def trainable_submodule(self, name_list:list):
        """select the trainbale submodule

        Args:
            name_list (list): submodul needed to be fixed
        """
        fixed_submodule = []
        trainable_submodule = []
        for name, module in self.named_children():
            for subname, submodule in module.named_children():
                for trainable_name in name_list:
                    if trainable_name not in subname:
                        for param in submodule.parameters():
                            param.requires_grad = False
                        fixed_submodule.append(name + '.' + subname)
                    else:
                        trainable_submodule.append(name + '.' + subname)
                        
        logger.info("fixed submodule : \n" + str(fixed_submodule))
        logger.info("trainable submodule : \n" + str(trainable_submodule))
        
    def group_params_by_names(self, groups, default_lr=1e-4):
            """Return trainable_params with different param groups using diffrent lr

            Args:
                groups (dict): config['groups']
            """
            # initialize param_list_by_name by config['groups']
            param_list_by_name = {}
            configed_name_list = []
            for cfg_ in groups:
                name = cfg_['name']
                lr = cfg_.get('lr', default_lr)
                train = cfg_.get('trainable', True)
                param_list_by_name[name] = {'params':[], 'lr':lr, 'train':train}
                configed_name_list.append(name)
            param_list_by_name['left'] = {'params':[], 'lr':default_lr, 'train':True}
            
            left_list = []  # store left params not mentioned in groups
            trainable_params = [] # store trainable params
            # iterate model submodule with configed_name_list
            for _, module_ in self.named_children():
                for module_name, submodule in module_.named_children():
                    matched_name = next((name for name in configed_name_list if name in module_name), None)
                    if matched_name:
                        cur_list = [param for param in submodule.parameters()]
                        param_list_by_name[matched_name]['params'] += cur_list
                        logger.info("Params of module: {} grouped into matched prefix: {}" .format(module_name,matched_name))
                    else: 
                        # if module not in config['groups'], its params will append to left
                        left_list += [param for param in submodule.parameters()]
                        logger.info("Params of module: {} grouped into 'left'" .format(module_name))

            # combine all unconfiged params into left_list
            param_list_by_name['left']['params'] = left_list
            
            for name, cfg_ in param_list_by_name.items():
                param_list,train,lr = cfg_['params'],cfg_['train'],cfg_['lr']
                if train:
                    params_group = [{'params': param_list, 'lr': float(lr)}]
                    trainable_params += params_group
                    logger.info("Trainable params: {} with lr {}".format(name, lr))
                else:
                    # frozen params will not be trained
                    for param in param_list:
                        param.requires_grad = False
                    logger.info("Frozen params: {}".format(name))
                
            logger.info("Params are diveded into groups according to config: \n" + str(groups))    
            return trainable_params

    def trainable_module(self, name_list:list):
        """select the trainbale module

        Args:
            name_list (list): submodul needed to be fixed
        """
        fixed_submodule = []
        trainable_submodule = []
        for name, module in self.named_children():
            for trainable_name in name_list:
                if trainable_name not in name:
                    for param in module.parameters():
                        param.requires_grad = False
                    fixed_submodule.append(name)
                else:
                    trainable_submodule.append(name)
                        
        logger.info("fixed submodule : \n" + str(fixed_submodule))
        logger.info("trainable submodule : \n" + str(trainable_submodule))

ac_cfg = {
}
