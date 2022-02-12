import random
import paddle
import torch
import numpy
import transformers
from nystromformer_paddle import nystromformer_paddle, nystromformer_config

import reprod_log
import os


def fake_data(save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(save_dir + '../loss_compare/fake_data.npy'):
        print('Please first run loss_compare.py')
        exit(0)
    os.system('cp ' + save_dir + '../loss_compare/fake_data.npy ' + save_dir + 'fake_data.npy')


def backward_torch(input_data, log):
    input_torch = {key: torch.tensor(value) for key, value in input_data.items()}
    model_torch = transformers.NystromformerForSequenceClassification.from_pretrained('uw-madison/nystromformer-512')
    model_torch.eval()
    optimizer = torch.optim.AdamW(
        model_torch.parameters(),
        lr=lr,
        betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01
    )
    for epoch in range(epochs):
        output = model_torch(**input_torch)
        loss = output.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(loss)
        log.add('loss' + str(epoch), loss.cpu().detach().numpy())
    return model_torch.classifier


def backward_paddle(input_data, model_torch_head, log):
    input_paddle = {key: paddle.to_tensor(value) for key, value in input_data.items()}
    model_paddle = nystromformer_paddle.NystromformerForSequenceClassification(
        nystromformer_config.NystromformerConfig()
    )
    for param_torch, param in zip(model_torch_head.parameters(), model_paddle.classifier.parameters()):
        if param.ndim == 2:
            param.set_value(param_torch.transpose(1, 0).detach().numpy())
        else:
            param.set_value(param_torch.detach().numpy())
    model_paddle.nystromformer.load_dict(paddle.load('../pretrained_files/nystromformer_model.params'))
    model_paddle.eval()
    optimizer = paddle.optimizer.AdamW(
        parameters=model_paddle.parameters(),
        learning_rate=lr,
        beta1=0.9, beta2=0.999, epsilon=1e-6, weight_decay=0.01
    )
    for epoch in range(epochs):
        output = model_paddle(**input_paddle)
        loss = output['loss']
        loss.backward()
        optimizer.step()
        optimizer.clear_gradients()
        print(loss)
        log.add('loss' + str(epoch), loss.cpu().numpy())


def main():
    compare_dir = './backward_compare/'
    fake_data(save_dir=compare_dir)
    diff_helper = reprod_log.ReprodDiffHelper()
    input_data = diff_helper.load_info(compare_dir + 'fake_data.npy')
    log_output_torch, log_output_paddle = reprod_log.ReprodLogger(), reprod_log.ReprodLogger()
    model_torch_head = backward_torch(input_data, log_output_torch)
    log_output_torch.save(compare_dir + 'bp_align_torch.npy')
    backward_paddle(input_data, model_torch_head, log_output_paddle)
    log_output_paddle.save(compare_dir + 'bp_align_paddle.npy')
    info_torch = diff_helper.load_info(compare_dir + 'bp_align_torch.npy')
    info_paddle = diff_helper.load_info(compare_dir + 'bp_align_paddle.npy')
    diff_helper.compare_info(info_torch, info_paddle)
    diff_helper.report(diff_method='mean', diff_threshold=2e-4, path=compare_dir + 'bp_align_diff_log.txt')


if __name__ == '__main__':
    random.seed(0)
    paddle.device.set_device('cpu')
    numpy.random.seed(0)
    paddle.seed(0)
    torch.manual_seed(0)

    lr = 0.01
    epochs = 3
    main()
