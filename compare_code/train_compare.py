import os
import reprod_log
import numpy


def main():
    compare_dir = './train_compare/'
    if not os.path.exists(compare_dir):
        os.mkdir(compare_dir)
    if not os.path.exists('../fine_tune_log.npy'):
        print('Please first do fine-tune by running run.py and get fine_tune_log.npy file.')
        exit(0)
    diff_helper = reprod_log.ReprodDiffHelper()
    f1_scores = diff_helper.load_info('../fine_tune_log.npy')
    f1_scores = [score.tolist() for score in f1_scores.values()]
    f1_score = max(f1_scores)
    train_log = reprod_log.ReprodLogger()
    train_log.add('f1_score', numpy.asarray(f1_score))
    train_log.save(compare_dir + 'train_align_paddle.npy')
    benchmark_log = reprod_log.ReprodLogger()
    benchmark_log.add('f1_score', numpy.asarray(0.930))
    benchmark_log.save(compare_dir + 'train_align_benchmark.npy')
    info_benchmark = diff_helper.load_info(compare_dir + 'train_align_benchmark.npy')
    info_paddle = diff_helper.load_info(compare_dir + 'train_align_paddle.npy')
    diff_helper.compare_info(info_benchmark, info_paddle)
    diff_helper.report(diff_method='mean', diff_threshold=0.005, path=compare_dir + 'train_align_diff_log.txt')


if __name__ == '__main__':
    main()
