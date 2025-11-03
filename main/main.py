import os, argparse, logging
import yaml
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from dataloader.dataloader import build_dataloader
from utils.log_helper import init_log
from utils.misc_helper import set_random_seed, format_cfg
from utils.model_helper import ModelHelper
from utils.optim_helper import build_grad_clipper, build_optimizer, build_scheduler
from utils.saver_helper import Saver, plot_scatter
from utils.loss_helper import build_losser, get_metrics, get_acc_from_double_stimuli
from utils.info_helper import calculate_parameters
# from utils.deploy_helper import export_onnx


parser = argparse.ArgumentParser()
parser.add_argument('--exp_root', type=str, required=True, help='the directory of experiments')
parser.add_argument("--train", help="use training config, else use testing config", action="store_true")
parser.add_argument('--calc_params', help='calculate number of parameters', action='store_true')
parser.add_argument('--export_onnx_path', help='path where the exported ONNX model will be saved', type=str, default=None, required=False)
args = parser.parse_args()

init_log('global', logging.INFO)
logger = logging.getLogger('global')

def main():
    # Set random seed
    set_random_seed(20200626)

    # Load cfg
    config_path = os.path.join(args.exp_root, 'config.yml')
    assert (os.path.exists(config_path)), "invalid config file: {}".format(config_path)
    cfg = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    logger.info("Running with config:\n{}".format(format_cfg(cfg)))

    # Set GPU ID
    gpu_id = cfg['trainer']['gpu_id']
    assert isinstance(gpu_id, int)
    torch.cuda.set_device(gpu_id)
    logger.info("Running on GPU: %d" % gpu_id)

    # Build model
    model = ModelHelper(cfg['net']).cuda()

    # Build saver
    cfg_saver = cfg.get('saver', dict())
    save_dir = os.path.join(args.exp_root, 'save_dir')
    log_dir = os.path.join(args.exp_root, 'log')
    saver = Saver(save_dir, model, config_path)

    # Load resume_model or pretrain_model if any
    if cfg_saver.get('resume_model', None):
        state_dict = saver.get_model_from_ckpt(cfg_saver['resume_model'])
        model.load(state_dict)
    elif cfg_saver.get('pretrain_model', None):
        state_dict = saver.get_model_from_ckpt(cfg_saver['pretrain_model'])
        model.load(state_dict)
    else:
        logger.info('no checkpoint provided to initialize the model')

    if args.calc_params:
        calculate_parameters(model)
        return

    # if args.export_onnx_path:
    #     export_onnx(model, cfg, args.export_onnx_path)
    #     return

    # Build grad_clipper
    grad_clipper = build_grad_clipper(cfg['trainer'].get('grad_clipper', None))

    # Build optimizer
    cfg_trainer = cfg['trainer']
    cfg_optim = cfg_trainer['optimizer']
    optimizer = build_optimizer(cfg_optim, model)

    # Restore optimizer and start_epoch if resume
    last_epoch = -1
    if cfg_saver.get('resume_model', None):
        optimizer, last_epoch = saver.restore_optimizer(optimizer, cfg_saver['resume_model'])
        logger.info('resume optimizer from checkpoint')
    logger.info('build saver and try to load trained model done')

    cfg_sch = cfg_trainer['lr_scheduler']
    lr_scheduler = build_scheduler(cfg_sch, optimizer, last_epoch)

    # Train
    if args.train:
        writer = SummaryWriter(os.path.join(log_dir, 'train'))
        train(model, cfg, lr_scheduler, writer, saver, grad_clipper)
    else:
        test(model, cfg, last_epoch)

def train(model, cfg, lr_scheduler, writer, saver, grad_clipper=None):
    # Set optimization and deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Build train/val dataloader
    train_loader = build_dataloader(cfg['dataset'], training=True)
    val_loader = build_dataloader(cfg['dataset'], training=False)
    resize_sizes = len(cfg['dataset']['resize_size'])

    # Model
    model.cuda().train()

    # Build Loss
    losser = build_losser(cfg)

    last_epoch = lr_scheduler.last_epoch
    start_epoch = last_epoch + 1
    max_epoch = cfg['trainer']['max_epoch']
    total_iter = last_epoch * len(train_loader)

    for cur_epoch in tqdm(range(start_epoch, max_epoch + 1)):
        for idx, inputs in enumerate(train_loader):
            total_iter += 1
            model.train()
            image_inputs = [inputs[f'image_input_{i+1}'].cuda() for i in range(resize_sizes)]
            preds = model(tuple(image_inputs))
            # preds = preds[1] if isinstance(preds, tuple) else preds
            labels = inputs['label'].cuda()
            loss = losser(preds, labels)
            model.zero_grad()
            loss.backward()
            if grad_clipper:
                grad_clipper.clip_grad(model.parameters())

            lr_scheduler.optimizer.step()

            # validation
            if cur_epoch % cfg['saver']['val_epoch'] == 0 and idx == len(train_loader) - 1:
                model.eval()
                preds_list, labels_list = [], []
                for val_inputs in val_loader:
                    image_val_inputs = [val_inputs[f'image_input_{i+1}'].cuda() for i in range(resize_sizes)]
                    preds = model(tuple(image_val_inputs))
                    preds = preds[0].detach().cpu() if isinstance(preds, tuple) else preds.detach().cpu()
                    labels = val_inputs['label'].detach().cpu()
                    preds_list.append(preds)
                    labels_list.append(labels)
                preds_all = torch.cat(preds_list, dim=0)
                labels_all = torch.cat(labels_list, dim=0)
                res = get_metrics(cfg, preds_all, labels_all)
                for i, score_key in enumerate(cfg['dataset']['score_key_list']):
                    for j, metric in enumerate(cfg['dataset']['metric_list']):
                        writer.add_scalar('scalar/%s_%s' % (score_key, metric), res[i][j], cur_epoch)
                        if metric == 'srocc' or metric == 'srcc':
                            logger.info('val: ep: {} - {}: srocc: {}'.format(cur_epoch, score_key, res[i][j]))
                # for j, metric in enumerate(cfg['dataset']['metric_list']):
                #     writer.add_scalar('scalar/%s_%s' % ('mos_clean', metric), res[0][j], cur_epoch)
                #     if metric == 'srocc' or metric == 'srcc':
                #         logger.info('val: ep: {} - {}: srocc: {}'.format(cur_epoch, 'mos_clean', res[0][j]))

                for idx, param_groups in enumerate(lr_scheduler.optimizer.param_groups):
                    base_lr = param_groups['lr']
                    writer.add_scalar(f'Learning_Rate_Group: {idx}', base_lr, cur_epoch)

            # display metric
            if total_iter % cfg['saver']['display_interval'] == 0:
                log = 'ep: {} it: {} - loss: {}'.format(cur_epoch, total_iter, loss)
                logger.info(log)

            # summary metric
            if total_iter % cfg['saver']['summary_interval'] == 0:
                writer.add_scalar('scalar/loss', loss, total_iter)

        lr_scheduler.step()


        #save ckpt
        if cur_epoch % cfg['saver']['save_epoch'] == 0:
            saver.save(best=False,
                       epoch=cur_epoch,
                       state_dict=model.state_dict(),
                       optimizer=lr_scheduler.optimizer.state_dict())

def test(model, cfg, epoch):
    # Set optimization and deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Build test dataloader
    loader = build_dataloader(cfg['dataset'], training=False)
    resize_sizes = len(cfg['dataset']['resize_size'])

    # Model
    model.cuda().eval()

    with torch.set_grad_enabled(False):
        preds_list, labels_list = [], []

        for inputs in loader:
            image_inputs = [inputs[f'image_input_{i+1}'].cuda() for i in range(resize_sizes)]
            preds = model(tuple(image_inputs))
            preds = preds[0] if isinstance(preds, tuple) else preds
            labels = inputs['label'].cuda()
            preds_list.append(preds)
            labels_list.append(labels)

        preds_all = torch.cat(preds_list, dim=0)
        labels_all = torch.cat(labels_list, dim=0)
        metric_res = get_metrics(cfg, preds_all, labels_all)

        #dump result
        results_dir = os.path.join(args.exp_root, 'results')
        os.makedirs(results_dir, exist_ok=True)
        result_path = os.path.join(results_dir, "{}epoch_result.xlsx".format(epoch))
        with pd.ExcelWriter(result_path, engine='openpyxl') as writer:
            #metrics such as srocc
            df_metric = pd.DataFrame(metric_res, index=cfg['dataset']['score_key_list'], columns=cfg['dataset']['metric_list'])
            df_metric.to_excel(writer, sheet_name='metrics')
            logger.info('result:\n %s' % df_metric.to_string())
            #detail info
            dump_name = cfg['dataset']['score_key_list'][0]
            idx = cfg['dataset']['score_key_list'].index(dump_name)
            df = pd.DataFrame({'img': loader.dataset.metas_combine['img'],
                               'preds': preds_all[:, idx].flatten().tolist(),
                               'labels': labels_all[:, idx].flatten().tolist()})
            df.to_excel(writer, sheet_name='single_stimuli_detail')

            # plot scatter diagram for mos distribution
            if cfg['saver'].get('plot_scatter', False):
                plot_scatter(labels_all, preds_all, idx, results_dir, epoch)

            if cfg['dataset']['test'].get('threshold', False):
                # get thresholds from config
                pred_thd = cfg['dataset']['test']['threshold'][0]
                gt_thd = cfg['dataset']['test']['threshold'][1]

                pair_names_l, pair_names_r, preds_final, gts_final, res, acc_list = get_acc_from_double_stimuli(loader, preds_all, idx, pred_thd, gt_thd)
                
                df = pd.DataFrame({'Left': pair_names_l, 'Right': pair_names_r,
                                f'preds_{pred_thd}': preds_final, f'labels_diff={gt_thd}': gts_final,
                                f'res_{pred_thd}_{gt_thd}': res
                                })
                df.loc[len(df)] = acc_list
                df.to_excel(writer, sheet_name='double_stimuli_detail')

                # add acc to metrics
                metric_res.append(acc_list[-1])
                df_metric[f'acc_{pred_thd}_{gt_thd}'] = acc_list[-1]
                df_metric.to_excel(writer, sheet_name='metrics')


if __name__ == '__main__':
    main()