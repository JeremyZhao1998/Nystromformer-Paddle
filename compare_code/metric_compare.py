import paddle
import torch
import numpy

import reprod_log

from paddle.metric import Precision, Recall
from nystromformer_paddle import utils


def get_f1_score_paddle(logits, labels):
    precision, recall = Precision(), Recall()
    precision.reset()
    recall.reset()
    utils.update_metrics(logits, labels, [precision, recall])
    return utils.get_f1_score(precision, recall)


def get_f1_score_torch(logits, labels):
    predictions = logits.argmax(-1).cpu().tolist()
    labels = labels.cpu().data.tolist()
    predictions, labels = numpy.asarray(predictions), numpy.asarray(labels)
    tp = numpy.sum((labels == 1) * (predictions == 1))
    fp = numpy.sum((labels == 0) * (predictions == 1))
    tn = numpy.sum((labels == 0) * (predictions == 0))
    fn = numpy.sum((labels == 1) * (predictions == 0))
    return 2 * tp / (2 * tp + fp + fn + 1e-10)


def fake_data(batch_size=64, save_path=None):
    logits = numpy.random.normal(0, 1, size=(batch_size, 2)).astype(numpy.float32)
    labels = numpy.random.randint(0, 2, size=(batch_size,)).astype(numpy.int64)
    if save_path is not None:
        log = reprod_log.ReprodLogger()
        log.add('logits', logits)
        log.add('labels', labels)
        log.save(save_path)


def main():
    compare_dir = './metric_compare/'
    fake_data(save_path=compare_dir + 'fake_data.npy')
    diff_helper = reprod_log.ReprodDiffHelper()
    input_data = diff_helper.load_info(compare_dir + 'fake_data.npy')
    logits_torch, labels_torch = torch.tensor(input_data['logits']), torch.tensor(input_data['labels'])
    logits_paddle, labels_paddle = paddle.to_tensor(input_data['logits']), paddle.to_tensor(input_data['labels'])
    f1_score_torch = get_f1_score_torch(logits_torch, labels_torch)
    f1_score_paddle = get_f1_score_paddle(logits_paddle, labels_paddle)

    log_output_torch, log_output_paddle = reprod_log.ReprodLogger(), reprod_log.ReprodLogger()
    log_output_torch.add('f1_score', numpy.asarray(f1_score_torch))
    log_output_paddle.add('f1_score', numpy.asarray(f1_score_paddle))
    log_output_torch.save(compare_dir + 'metric_torch.npy')
    log_output_paddle.save(compare_dir + 'metric_paddle.npy')
    info_torch = diff_helper.load_info(compare_dir + 'metric_torch.npy')
    info_paddle = diff_helper.load_info(compare_dir + 'metric_paddle.npy')
    diff_helper.compare_info(info_torch, info_paddle)
    diff_helper.report(diff_method='mean', diff_threshold=1e-6, path=compare_dir + 'metric_diff_log.txt')


if __name__ == '__main__':
    paddle.device.set_device('cpu')
    numpy.random.seed(0)
    paddle.seed(0)
    torch.manual_seed(0)
    main()
