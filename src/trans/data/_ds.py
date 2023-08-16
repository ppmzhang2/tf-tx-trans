"""Dataset prcoessing for Portuguese-English translation task."""
import tensorflow as tf
import tensorflow_datasets as tfds
from transformers import BertTokenizer

# BPE tokenizers from HuggingFace
TKR_PT_NAME = "neuralmind/bert-base-portuguese-cased"
TKR_EN_NAME = "bert-base-uncased"


def get_tkr(name: str) -> BertTokenizer:
    """Load BPE tokenizer for Portuguese-English translation task."""
    return BertTokenizer.from_pretrained(name)


def txt2tk(pt: tf.Tensor, en: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Tokenize Portuguese and English text with TensorFlow Datasets."""

    def _txt2tk(pt: tf.Tensor, en: tf.Tensor) -> tuple[list[str], list[str]]:
        """Get tokens for Portuguese and English text."""
        pt_tkr, en_tkr = get_tkr(TKR_PT_NAME), get_tkr(TKR_EN_NAME)
        pt_tokens = pt_tkr.tokenize(pt.numpy().decode("utf-8"))
        en_tokens = en_tkr.tokenize(en.numpy().decode("utf-8"))
        return pt_tokens, en_tokens

    return tf.py_function(_txt2tk, [pt, en], [tf.string, tf.string])


def txt2id(pt: tf.Tensor, en: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Encode Portuguese and English text with BPE token IDs."""

    def _txt2id(pt: tf.Tensor, en: tf.Tensor) -> tuple[list[int], list[int]]:
        """Get token IDs for Portuguese and English text."""
        pt_tkr, en_tkr = get_tkr(TKR_PT_NAME), get_tkr(TKR_EN_NAME)
        pt_ids = pt_tkr.encode(pt.numpy().decode("utf-8"),
                               add_special_tokens=True)
        en_ids = en_tkr.encode(en.numpy().decode("utf-8"),
                               add_special_tokens=True)
        return pt_ids, en_ids

    return tf.py_function(_txt2id, [pt, en], [tf.int32, tf.int32])


def to_ragged(
    pt_tokens: tf.Tensor,
    en_tokens: tf.Tensor,
) -> tuple[tf.RaggedTensor, tf.RaggedTensor]:
    """Convert tokenized text to ragged tensors."""
    return (
        tf.RaggedTensor.from_tensor(tf.expand_dims(pt_tokens, axis=-1)),
        tf.RaggedTensor.from_tensor(tf.expand_dims(en_tokens, axis=-1)),
    )


def id2txt_en(ids_en: tf.Tensor) -> str:
    """Convert token IDs back to English text."""
    tkr_en = get_tkr(TKR_EN_NAME)
    txt_en = tkr_en.decode(ids_en, skip_special_tokens=True)
    return txt_en


def load_train_valid(
    batch_size: int
) -> tuple[tf.data.Dataset, tf.data.Dataset, tfds.core.DatasetInfo]:
    """Load Portuguese-English translation dataset.

    Returns:
        tuple[tf.data.Dataset, tf.data.Dataset, tfds.core.DatasetInfo]:
            training and validation datasets, and dataset info.
    """
    (ds_tr, ds_va), info = tfds.load(
        "ted_hrlr_translate/pt_to_en",
        split=["train", "validation"],
        shuffle_files=True,
        with_info=True,
        as_supervised=True,
    )

    ds_tr_ = ds_tr.map(
        txt2id,
        num_parallel_calls=tf.data.AUTOTUNE,
    ).map(
        to_ragged,
        num_parallel_calls=tf.data.AUTOTUNE,
    ).batch(batch_size)
    ds_va_ = ds_va.map(
        txt2id,
        num_parallel_calls=tf.data.AUTOTUNE,
    ).map(
        to_ragged,
        num_parallel_calls=tf.data.AUTOTUNE,
    ).batch(batch_size)

    return ds_tr_, ds_va_, info


if __name__ == "__main__":
    BATCH_SIZE = 2
    train_data, val_data, _ = load_train_valid(batch_size=BATCH_SIZE)
    for pt, en in train_data.take(1):
        print(pt)
        print(en)

    ids_en = tf.constant(
        [101, 2021, 2054, 2065, 2009, 2020, 3161, 1029, 102],
        dtype=tf.int32,
    )
    txt_en = id2txt_en(ids_en)
    print(txt_en)
