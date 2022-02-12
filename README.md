# Nystromformer-Paddle
Reproducing Nystromformer(A Nyström-Based Algorithm for Approximating Self-Attention, AAAI 2021) based on PaddlePaddle.

Nystromformer official repo: https://github.com/mlpen/Nystromformer

Nystromformer paper link: https://arxiv.org/pdf/2102.03902v3.pdf

## Sub-directories

- paper_summary: Brief summary of the idea of the Nystromformer paper.
- nystromformer_paddle: Code of the model reproduced based on paddlepaddle.
- pretrained_files: The pretrained tokenizer file and model parameters. Note that we transferred the pretrained weight provided by huggingface(which is also transferred from the weights provided by the original paper repo) into paddlepaddle format.
- data: The directory which contains data files. Currently only the tokenized IMDB dataset.
- compare_code: We compared our reproduced model with the huggingface pytorch version, showing that the model structure and the training procedure are exactly the same. Pytorch is needed when running the code in this directory.

## Requirements

- paddlepaddle >= 2.2.0
- transformers >= 4.16.0.dev0
- datasets >= 1.18.1
- reprod_log (Installation see https://github.com/WenmuZhou/reprod_log)

## Get started

Clone our repo, and under Nystromformer-Paddle directory, simply run:

```
python run.py
```

This file fine-tunes the pretrained model with IMDB dataset and gets the expected result on validation set.

To change the configuration of the training, simply modify line 16-22 in run.py

