import torch
import paddle
import numpy

from nystromformer_paddle import nystromformer_paddle, nystromformer_config, nystromformer_torch
import transformers

import random
import reprod_log
import os


def fake_data(save_dir=None):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    diff_helper = reprod_log.ReprodDiffHelper()
    try:
        input_data = diff_helper.load_info(save_dir + '../forward_compare/fake_data.npy')
        log = reprod_log.ReprodLogger()
        log.add('input_ids', input_data['input_ids'])
        log.add('token_type_ids', input_data['token_type_ids'])
        log.add('attention_mask', input_data['attention_mask'])
        try:
            input_data = diff_helper.load_info(save_dir + '../metric_compare/fake_data.npy')
            log.add('labels', input_data['labels'][:2])
        except FileNotFoundError:
            print('Please first run metric_compare.py')
        log.save(save_dir + 'fake_data.npy')
    except FileNotFoundError:
        print('Please first run forward_compare.py')


def main():
    compare_dir = './loss_compare/'
    fake_data(save_dir=compare_dir)
    diff_helper = reprod_log.ReprodDiffHelper()
    input_data = diff_helper.load_info(compare_dir + 'fake_data.npy')

    model_torch = transformers.NystromformerForSequenceClassification.from_pretrained('uw-madison/nystromformer-512')
    model_paddle = nystromformer_paddle.NystromformerForSequenceClassification(
        nystromformer_config.NystromformerConfig()
    )
    model_paddle.nystromformer.load_dict(paddle.load('../pretrained_files/nystromformer_model.params'))
    for param_torch, param in zip(model_torch.classifier.parameters(), model_paddle.classifier.parameters()):
        if param.ndim == 2:
            param.set_value(param_torch.transpose(1, 0).detach().numpy())
        else:
            param.set_value(param_torch.detach().numpy())
    model_torch.eval()
    model_paddle.eval()
    output_torch = model_torch(
        input_ids=torch.tensor(input_data['input_ids']),
        token_type_ids=torch.tensor(input_data['token_type_ids']),
        attention_mask=torch.tensor(input_data['attention_mask']),
        labels=torch.tensor(input_data['labels'])
    )
    output_paddle = model_paddle(
        input_ids=paddle.to_tensor(input_data['input_ids']),
        token_type_ids=paddle.to_tensor(input_data['token_type_ids']),
        attention_mask=paddle.to_tensor(input_data['attention_mask']),
        labels=paddle.to_tensor(input_data['labels'])
    )

    log_output_torch, log_output_paddle = reprod_log.ReprodLogger(), reprod_log.ReprodLogger()
    log_output_torch.add('output_loss', output_torch.loss.detach().numpy())
    log_output_paddle.add('output_loss', output_paddle['loss'].numpy())
    log_output_torch.save(compare_dir + 'loss_torch.npy')
    log_output_paddle.save(compare_dir + 'loss_paddle.npy')
    info_torch = diff_helper.load_info(compare_dir + 'loss_torch.npy')
    info_paddle = diff_helper.load_info(compare_dir + 'loss_paddle.npy')
    diff_helper.compare_info(info_torch, info_paddle)
    diff_helper.report(diff_method='mean', diff_threshold=1e-6, path=compare_dir + 'loss_diff_log.txt')


if __name__ == '__main__':
    random.seed(0)
    os.environ['PYTHONHASHSEED'] = str(0)
    paddle.device.set_device('cpu')
    numpy.random.seed(0)
    paddle.seed(0)
    torch.manual_seed(0)
    main()
