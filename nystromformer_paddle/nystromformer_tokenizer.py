from paddlenlp.transformers import BertTokenizer


class NystromformerTokenizer(BertTokenizer):

    resource_files_names = {"vocab_file": "vocab.txt"}
    pretrained_resource_files_map = {
        "vocab_file": {
            "nystromformer-base":
                "<where to download vocab.txt>",
        }
    }
    pretrained_init_configuration = {
        "nystromformer-base": {
            "do_lower_case": True
        },
    }
