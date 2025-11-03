import torch
import torch.nn as nn
import numpy as np
from scipy import stats
import importlib
    
def build_losser(cfg):
    if '.' in cfg['loss']['type']:
        module_name, cls_name = cfg['loss']['type'].rsplit('.', 1)
    else:
        module_name, cls_name = 'torch.nn', cfg['loss']['type']
    module = importlib.import_module(module_name)
    cls = getattr(module, cls_name)
    return cls(**cfg['loss'].get('kwargs', {}))

def fit_function(predict, mos):
    predict = predict.cpu().detach().numpy()
    mos = mos.cpu().detach().numpy()
    params = np.polyfit(predict, mos, deg=3)
    p1 = np.poly1d(params)
    predict_fitted = p1(predict)
    return predict_fitted

def accuracy(predict, mos, margin=0.0):
    predict = predict
    mos = mos
    pred_errors = predict[0::2] - predict[1::2]
    mos_errors = mos[0::2] - mos[1::2]
    cnt = [1 if (pred_error*mos_error > 0 and abs(pred_error) > margin) else 0 for pred_error, mos_error in zip(pred_errors, mos_errors)]
    return np.mean(cnt)

def cal_metric(x: torch.Tensor, y:torch.Tensor, type) -> float:
    x = x.squeeze().cpu().detach().numpy()
    y = y.squeeze().cpu().detach().numpy()
    assert x.ndim == 1 and y.ndim == 1
    if type == 'srcc' or type == 'srocc':
        res, _ = stats.spearmanr(x, y)
    elif type == 'plcc':
        res, _ = stats.pearsonr(x, y)
    elif type == 'krcc' or type == 'krocc':
        res, _ = stats.kendalltau(x, y)
    elif type == 'somersd':
        res = stats.somersd(x, y).statistic
    elif type == 'mse':
        res = np.mean((x - y) ** 2)
    elif type == 'rmse':
        res = np.sqrt(np.mean((x - y) ** 2))
    elif type == 'mae':
        res = np.mean(np.abs(x - y))
    elif type == 'acc':
        res = accuracy(x, y)
    else:
        raise NotImplementedError
    return res

def get_metrics(cfg, x, y):
    res = []
    # input shape must be (sample_num, score_num)
    label_types = cfg['dataset']['score_key_list']
    if cfg['dataset']['train'].get('is_pair', False):
        if 'votes' in label_types:
            label_types.remove('votes')
        if 't_test' in label_types:
            label_types.remove('t_test')
    assert x.shape[1] == len(label_types)
    for idx, score_key in enumerate(label_types):
        x_slice, y_slice = x[:, idx], y[:, idx]
        res.append([cal_metric(x_slice, y_slice, metric) for metric in cfg['dataset']['metric_list']])
    return res

# def get_metrics(cfg, x, y):
#     # print(x.shape, y.shape)
#     res = []
#     # input shape must be (sample_num, score_num)
#     label_types = cfg['dataset']['score_key_list']
#     if y.shape[-1] != 1:
#         y = 1 * y[:, 0] + 2 * y[:, 1] + 3 * y[:, 2] + \
#             4 * y[:, 3] + 5 * y[:, 4]
#     x_slice, y_slice = x[:, 0], y
#     res.append([cal_metric(x_slice, y_slice, metric) for metric in cfg['dataset']['metric_list']])
#     return res

def get_acc_from_double_stimuli(loader, preds_all, idx, pred_thd, gt_thd):

    pair_names_l = [ l for l in loader.dataset.metas_combine['img'][::2]]
    pair_names_r = [ r for r in loader.dataset.metas_combine['img'][1::2]]

    def get_value(x, margin):
        assert type(x) != str, "x is string"
        if x > margin:
            return '1|0'
        elif x < -margin:
            return '0|1'
        else:
            return '0|0'
        
    preds_all = preds_all[:, idx].flatten()
    preds_even = preds_all[::2]
    preds_odd = preds_all[1::2]
    preds_diff = preds_even - preds_odd
    preds_final = [get_value(x, pred_thd) for x in preds_diff]
        
    if isinstance(gt_thd, int):
        labels_all = loader.dataset.metas_combine['votes']
        labels_even = torch.tensor([l for l in labels_all[::2]])
        labels_odd = torch.tensor([r for r in labels_all[1::2]])
        labels_diff = labels_even - labels_odd
        gts_final = [get_value(x, gt_thd) for x in labels_diff]
    else:
        gts_t = loader.dataset.metas_combine['t_test']
        gts_t_even = gts_t[::2]
        gts_t_odd = gts_t[1::2]
        gts_final = [f'{int(t_e)}|{int(t_o)}' for t_e, t_o in zip(gts_t_even, gts_t_odd)]
        l_avg = gts_t_even.mean()
        r_avg = gts_t_odd.mean()
        
    res = [1 if p==g else 0 for p,g in zip(preds_final, gts_final)]
    pair_num = len(preds_all)//2
    acc = sum(res)/pair_num
    
    l_cnt, r_cnt = 0, 0
    for vote in preds_final:
        l,r = vote.split('|')
        if l == '1':
            l_cnt += 1
        if r == '1':
            r_cnt += 1
    p_l_avg = l_cnt / pair_num
    p_r_avg = r_cnt / pair_num

    l_cnt, r_cnt = 0, 0
    for vote in gts_final:
        l,r = vote.split('|')
        if l == '1':
            l_cnt += 1
        if r == '1':
            r_cnt += 1
    l_avg = l_cnt / pair_num
    r_avg = r_cnt / pair_num

    acc_list = ['mean','',f'{p_l_avg}|{1-p_l_avg-p_r_avg}|{p_r_avg}',f'{l_avg}|{1-l_avg-r_avg}|{r_avg}',acc]

    return pair_names_l, pair_names_r, preds_final, gts_final, res, acc_list

if __name__ == '__main__':
    # x = torch.tensor([5, 6, 7, 8])
    # y = torch.tensor([2, 1, 4, 3])
    x = torch.tensor([[8], [7], [5], [6]])
    y = torch.tensor([[2], [1], [4], [3]])
    print(cal_metric(x, y, 'srcc'))
    print(cal_metric(x, y, 'plcc'))
    print(cal_metric(x, y, 'mse'))
    print(cal_metric(x, y, 'mae'))
    print(cal_metric(x, y, 'acc'))