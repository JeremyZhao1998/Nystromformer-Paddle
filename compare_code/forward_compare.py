import random
import os

import numpy
import paddle
import torch
import transformers

import nystromformer_paddle.nystromformer_paddle as nystromformer_paddle
import nystromformer_paddle.nystromformer_config as nystromformer_config

import reprod_log

random.seed(0)
os.environ['PYTHONHASHSEED'] = str(0)
numpy.random.seed(0)
torch.manual_seed(0)
paddle.seed(0)
paddle.device.set_device('cpu')


def get_data_numpy(input_data):
    return {
        key: value.detach().numpy() if hasattr(value, 'detach') else value
        for key, value in input_data.items()
    }


def get_data_torch(input_data):
    return {key: torch.tensor(input_data[key]) for key in input_data}


def get_data_paddle(input_data):
    return {key: paddle.to_tensor(input_data[key]) for key in input_data}


def compare_nystromformer_embeddings(input_data, model_torch, model_paddle):
    input_torch, input_paddle = get_data_torch(input_data), get_data_paddle(input_data)

    """params_torch = {i: param for i, param in enumerate(model_torch.parameters())}
    for i, param in enumerate(model_paddle.parameters()):
        param.set_value(params_torch[i].detach().numpy())
    buffers_torch = {i: buffer for i, buffer in enumerate(model_torch.buffers())}
    for i, buffer in enumerate(model_paddle.buffers()):
        buffer.set_value(buffers_torch[i].detach().numpy())"""

    model_paddle.set_state_dict(paddle.load('../pretrained_files/nystromformer_embeddings.params'))

    output_torch = model_torch(**input_torch)
    output_paddle = model_paddle(**input_paddle)

    # paddle.save(model_paddle.state_dict(), '../pretrained_files/nystromformer_embeddings.params')

    return output_torch.detach().numpy()


def compare_nystromformer_encoder(input_data, model_torch, model_paddle):
    input_torch, input_paddle = get_data_torch(input_data), get_data_paddle(input_data)

    """params_torch = {i: param for i, param in enumerate(model_torch.parameters())}
    for i, param in enumerate(model_paddle.parameters()):
        if param.ndim == 2:
            param.set_value(params_torch[i].transpose(1, 0).detach().numpy())
        else:
            param.set_value(params_torch[i].detach().numpy())"""

    model_paddle.set_state_dict(paddle.load('../pretrained_files/nystromformer_encoder.params'))

    output_torch = model_torch(**input_torch)
    output_paddle = model_paddle(**input_paddle)

    # paddle.save(model_paddle.state_dict(), '../pretrained_files/nystromformer_encoder.params')

    return get_data_numpy(output_torch)


def compare_nystromformer_model(input_data, model_torch, model_paddle):
    input_torch, input_paddle = get_data_torch(input_data), get_data_paddle(input_data)

    """embeddings_input_data = {
        'input_ids': input_data['input_ids'],
        'token_type_ids': input_data['token_type_ids']
    }
    embedding_output = \
        compare_nystromformer_embeddings(embeddings_input_data, model_torch.embeddings, model_paddle.embeddings)

    input_shape = input_torch['input_ids'].size()
    device = input_torch['input_ids'].device
    extended_attention_mask = \
        model_torch.get_extended_attention_mask(input_torch['attention_mask'], input_shape, device).detach().numpy()
    encoder_input = {
        'hidden_states': embedding_output,
        'attention_mask': extended_attention_mask,
        'output_attentions': model_torch.config.output_attentions,
        'output_hidden_states': model_torch.config.output_hidden_states
    }
    return compare_nystromformer_encoder(encoder_input, model_torch.encoder, model_paddle.encoder)"""

    """model_paddle.embeddings.load_dict(paddle.load('../pretrained_files/nystromformer_embeddings.params'))
    model_paddle.encoder.load_dict(paddle.load('../pretrained_files/nystromformer_encoder.params'))"""

    model_paddle.set_state_dict(paddle.load('../pretrained_files/nystromformer_model.params'))

    output_torch = model_torch(**input_torch)
    output_paddle = model_paddle(**input_paddle)
    # paddle.save(model_paddle.state_dict(), '../pretrained_files/nystromformer_model.params')


def compare_nystromformer_for_sequence_classification(input_data, model_torch, model_paddle):
    input_torch, input_paddle = get_data_torch(input_data), get_data_paddle(input_data)
    model_paddle.nystromformer.load_dict(paddle.load('../pretrained_files/nystromformer_model.params'))
    params_torch = {i: param for i, param in enumerate(model_torch.classifier.parameters())}
    for i, param in enumerate(model_paddle.classifier.parameters()):
        if param.ndim == 2:
            param.set_value(params_torch[i].transpose(1, 0).detach().numpy())
        else:
            param.set_value(params_torch[i].detach().numpy())
    output_torch = model_torch(**input_torch)
    output_paddle = model_paddle(**input_paddle)
    return output_torch, output_paddle


def fake_data(batch_size=2, max_len=510, save_path=None):
    input_ids, token_type_ids, attention_mask = [], [], []
    for i in range(batch_size):
        seq_len = numpy.random.randint(low=10, high=max_len - 2)
        inputs = numpy.concatenate(
            [
                numpy.random.randint(low=4, high=30000, size=seq_len + 2, dtype=numpy.int64),
                numpy.zeros(max_len - seq_len - 2, dtype=numpy.int64)
            ],
            axis=0
        )
        inputs[0], inputs[seq_len + 1] = 2, 3
        input_ids.append(inputs)
        token_types = numpy.zeros_like(inputs, dtype=numpy.int64)
        token_type_ids.append(token_types)
        attentions = numpy.concatenate(
            [numpy.ones(seq_len + 2, dtype=numpy.int64), numpy.zeros(max_len - seq_len - 2, dtype=numpy.int64)],
            axis=0
        )
        attention_mask.append(attentions)
    input_ids, token_type_ids, attention_mask = \
        numpy.stack(input_ids, axis=0), numpy.stack(token_type_ids), numpy.stack(attention_mask)
    if save_path is not None:
        log = reprod_log.ReprodLogger()
        log.add('input_ids', input_ids)
        log.add('token_type_ids', token_type_ids)
        log.add('attention_mask', attention_mask)
        log.save(save_path)


def main():
    compare_dir = './forward_compare/'
    fake_data(save_path=compare_dir + 'fake_data.npy')
    diff_helper = reprod_log.ReprodDiffHelper()
    input_data = diff_helper.load_info(compare_dir + 'fake_data.npy')

    model_torch = transformers.NystromformerForSequenceClassification.from_pretrained('uw-madison/nystromformer-512')
    config_paddle = nystromformer_config.NystromformerConfig()
    model_paddle = nystromformer_paddle.NystromformerForSequenceClassification(config_paddle)
    model_torch.eval()
    model_paddle.eval()
    output_torch, output_paddle = \
        compare_nystromformer_for_sequence_classification(input_data, model_torch, model_paddle)

    log_output_torch, log_output_paddle = reprod_log.ReprodLogger(), reprod_log.ReprodLogger()
    log_output_torch.add('output_logits', output_torch.logits.detach().numpy())
    log_output_paddle.add('output_logits', output_paddle['logits'].numpy())
    log_output_torch.save(compare_dir + 'forward_torch.npy')
    log_output_paddle.save(compare_dir + 'forward_paddle.npy')
    info_torch = diff_helper.load_info(compare_dir + 'forward_torch.npy')
    info_paddle = diff_helper.load_info(compare_dir + 'forward_paddle.npy')
    diff_helper.compare_info(info_torch, info_paddle)
    diff_helper.report(diff_method='mean', diff_threshold=1e-6, path=compare_dir + 'forward_diff_log.txt')


if __name__ == '__main__':
    main()
