import logging
import onnx
import torch
import warnings
import traceback

logger = logging.getLogger('global')
warnings.filterwarnings("ignore", message="The shape inference of prim::Constant type is missing")
warnings.filterwarnings("ignore", message="Converting a tensor to a Python number might cause the trace to be incorrect")

def export_onnx(model, cfg, onnx_path):
    #ONNX export
    try:
        model.eval()
        logger.info('Starting ONNX export with onnx %s...' % onnx.__version__)
        dummy_input = (tuple(torch.randn(1, 3, size, size, device='cuda') for size in cfg['dataset']['resize_size']),)
        input_names = ['input_%d' % i for i in range(len(cfg['dataset']['resize_size']))]
        output_names = ['output_0']
        dynamic_axes = {'input_0': {0: 'batch_size'},
                        'input_1': {0: 'batch_size'},
                        'output_0': {0: 'batch_size'}}
        torch.onnx.export(model, dummy_input, onnx_path,
                          verbose=False,
                          opset_version=13,
                          input_names=input_names,
                          output_names=output_names,
                          do_constant_folding=True,
                          dynamic_axes=dynamic_axes)
        logger.info('ONNX export success, saved as %s' % onnx_path)

    except Exception as e:
        logger.info('ONNX export failed: {}'.format(e))
        traceback.print_exc()
