import numpy as np
from numpy.lib import stride_tricks
import torch

from dnabert.transformers.configuration_bert import BertConfig
from dnabert.transformers.modeling_bert import BertForSequenceClassification
from dnabert.transformers.tokenization_bert import BertTokenizer


def kmer_pre_tokenize(seq, k):
    """ Pre-tokenize a sequence into k-mers.

    Args:
        seq (str): DNA sequence
        k (int): k-mer size
    
    Returns: array of k-mers
    """
    seq = np.fromiter(seq, (np.compat.unicode, 1))
    # words = stride_tricks.sliding_window_view(seq, window_shape=k)
    return np.apply_along_axis(
        lambda row: row.astype('|S1').tostring().decode('utf-8'),
        axis=1,
        arr=stride_tricks.sliding_window_view(seq, window_shape=k)
    )



if __name__ == '__main__':
    bert_config = BertConfig.from_pretrained(
        'dnabert/5-new-12w-0/',
        num_labels=2
    )
    tokenizer = BertTokenizer.from_pretrained(
        'dnabert/5-new-12w-0/',
        from_tf=False,
        do_lower_case=False
    )
    model = BertForSequenceClassification.from_pretrained(
        'dnabert/5-new-12w-0/',
        from_tf=False,
        config=bert_config,
    )

    seq = 'ATCCTNAATATAATAT'

    pre_tokens = kmer_pre_tokenize(seq, 5)
    pt_string = ' '.join(pre_tokens)

    tokens = tokenizer.tokenize(pt_string)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    features = tokenizer.encode_plus(
        pre_tokens.tolist(),
        is_split_into_words=True,
        return_tensors='pt',
    )

    