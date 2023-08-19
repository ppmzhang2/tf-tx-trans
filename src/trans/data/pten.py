"""Loaders for translation dataset from Portuguese to English in plain text."""
import tensorflow as tf
import tensorflow_datasets as tfds
from transformers import BertTokenizer

from trans import cfg

# BPE tokenizers from HuggingFace
TKR_PT_NAME = "neuralmind/bert-base-portuguese-cased"
TKR_EN_NAME = "bert-base-uncased"

# Dataset parameters
N_OBS_TR = 51785  # number of training observations
N_OBS_VA = 1193  # number of validation observations
N_OBS_TE = 1803  # number of test observations

# internal parameters
_BUFFER_SIZE = 20000  # buffer size for shuffling
_SLICE_CTX = slice(0, cfg.SEQ_LEN)  # slice for EN (query) and PT (context)
_SLICE_LBL = slice(1, cfg.SEQ_LEN + 1)  # slice for EN (label)


def get_tkr(name: str) -> BertTokenizer:
    """Load BPE tokenizer for Portuguese-English translation task."""
    return BertTokenizer.from_pretrained(name)


def encode_txt(
    pt: tf.Tensor,
    en: tf.Tensor,
    pt_tkr: BertTokenizer,
    en_tkr: BertTokenizer,
    *,
    integer: bool,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Encode Portuguese-English text to either token IDs or string tokens.

    Args:
        pt (tf.Tensor): Portuguese text.
        en (tf.Tensor): English text.
        pt_tkr (BertTokenizer): BPE tokenizer for Portuguese.
        en_tkr (BertTokenizer): BPE tokenizer for English.
        integer (bool, optional): whether to return integer token IDs or
            string tokens. Defaults to True.

    Returns:
        tuple[tf.Tensor, tf.Tensor, tf.Tensor]: token IDs or string tokens for
            Portuguese and English text, and English label.
    """

    def _txt2id(
        pt: tf.Tensor,
        en: tf.Tensor,
    ) -> tuple[list[int], list[int], list[int]]:
        pt_ids = pt_tkr.encode(pt.numpy().decode("utf-8"),
                               add_special_tokens=True)
        en_ids = en_tkr.encode(en.numpy().decode("utf-8"),
                               add_special_tokens=True)
        return pt_ids[_SLICE_CTX], en_ids[_SLICE_CTX], en_ids[_SLICE_LBL]

    def _txt2tk(
        pt: tf.Tensor,
        en: tf.Tensor,
    ) -> tuple[list[str], list[str], list[str]]:
        pt_tokens = pt_tkr.tokenize(pt.numpy().decode("utf-8"))
        en_tokens = en_tkr.tokenize(en.numpy().decode("utf-8"))
        return pt_tokens[_SLICE_CTX], en_tokens[_SLICE_CTX], en_tokens[
            _SLICE_LBL]

    if integer:
        return tf.py_function(_txt2id, [pt, en],
                              [tf.int32, tf.int32, tf.int32])
    return tf.py_function(_txt2tk, [pt, en], [tf.string, tf.string, tf.string])


def to_ragged(
    pt_tokens: tf.Tensor,
    en_tokens: tf.Tensor,
    en_labels: tf.Tensor,
) -> tuple[tf.RaggedTensor, tf.RaggedTensor, tf.RaggedTensor]:
    """Convert tokenized text to ragged tensors.

    Args:
        pt_tokens (tf.Tensor): token IDs for Portuguese text.
        en_tokens (tf.Tensor): token IDs for English text.
        en_labels (tf.Tensor): token IDs for English label.

    Returns:
        tuple[tf.RaggedTensor, tf.RaggedTensor, tf.RaggedTensor]: ragged
            tensors for Portuguese and English text, and English label.
    """
    return (
        tf.RaggedTensor.from_tensor(tf.expand_dims(pt_tokens, axis=-1)),
        tf.RaggedTensor.from_tensor(tf.expand_dims(en_tokens, axis=-1)),
        tf.RaggedTensor.from_tensor(tf.expand_dims(en_labels, axis=-1)),
    )


def make_batch(
    ds: tf.data.Dataset,
    pt_tkr: BertTokenizer,
    en_tkr: BertTokenizer,
    *,
    integer: bool,
    ragged: bool,
) -> tf.data.Dataset:
    """Make batches of Portuguese-English translation dataset.

    Args:
        ds (tf.data.Dataset): Portuguese-English translation dataset.
        pt_tkr (BertTokenizer): BPE tokenizer for Portuguese.
        en_tkr (BertTokenizer): BPE tokenizer for English.
        integer (bool): whether to return integer token IDs or string tokens.
        ragged (bool, optional): whether to return ragged tensors.

    Returns:
        tf.data.Dataset: batched dataset.
    """

    def _encoder(
        pt: tf.Tensor,
        en: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        return encode_txt(pt, en, pt_tkr, en_tkr, integer=integer)

    res = ds.shuffle(_BUFFER_SIZE).map(
        _encoder,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if ragged:
        return res.map(
            to_ragged,
            num_parallel_calls=tf.data.AUTOTUNE,
        ).batch(
            cfg.BATCH_SIZE,
            drop_remainder=True,
        ).prefetch(tf.data.AUTOTUNE)

    return res.padded_batch(
        cfg.BATCH_SIZE,
        padded_shapes=([None], [None], [None]),
        padding_values=(0, 0, 0),
        drop_remainder=True,
    ).prefetch(tf.data.AUTOTUNE)


def load_train_valid(
    *,
    ragged: bool = False,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tfds.core.DatasetInfo]:
    """Load Portuguese-English translation dataset.

    Returns:
        tuple[tf.data.Dataset, tf.data.Dataset, tfds.core.DatasetInfo]:
            training and validation datasets, and dataset info.
    """
    integer = True
    (ds_tr, ds_va), info = tfds.load(
        "ted_hrlr_translate/pt_to_en",
        split=["train", "validation"],
        shuffle_files=True,
        with_info=True,
        as_supervised=True,
    )
    pt_tkr, en_tkr = get_tkr(TKR_PT_NAME), get_tkr(TKR_EN_NAME)

    return (
        make_batch(ds_tr, pt_tkr, en_tkr, integer=integer, ragged=ragged),
        make_batch(ds_va, pt_tkr, en_tkr, integer=integer, ragged=ragged),
        info,
    )


def id2txt_en(ids_en: tf.Tensor) -> str:
    """Convert token IDs back to English text."""
    tkr_en = get_tkr(TKR_EN_NAME)
    txt_en = tkr_en.decode(ids_en, skip_special_tokens=True)
    return txt_en
