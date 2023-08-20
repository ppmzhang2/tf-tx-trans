# Transformer-based Language Translation Model with TensorFlow 2

## Environment

Install TensorFlow 2.0 and other dependencies:

```bash
# CUDA 12.1 + cuDNN 8.9 + Python 3.11
conda env create -f conda-cu12-py311.yaml
# Apple Silicon + Python 3.11
conda env create -f conda-apple-py311.yaml
```

Uninstall:

```bash
conda env remove --name py311-apple-tftxtrans
```

Note that environment variables such as `LD_LIBRARY_PATH` must be set properly
on a GPU machine:

```fish
set -Ux CUDNN_PATH $CONDA_PREFIX/lib/python3.1/site-packages/nvidia/cudnn
set -Ux LD_LIBRARY_PATH $LD_LIBRARY_PATH $CONDA_PREFIX/lib $CUDNN_PATH/lib
set -Ux XLA_FLAGS --xla_gpu_cuda_data_dir=$CONDA_PREFIX
```

## References

- [The Original Transformer Paper](https://arxiv.org/abs/1706.03762)
- The well documented TensorFlow
  [Transformer Tutorial](https://www.tensorflow.org/text/tutorials/transformer)
