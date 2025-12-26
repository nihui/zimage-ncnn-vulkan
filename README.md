# Z-Image ncnn

:exclamation: :exclamation: :exclamation: This software is in the early development stage, it may bite your cat

ncnn implementation of Z-Image image generater.

z-image-ncnn uses [ncnn project](https://github.com/Tencent/ncnn) as the universal neural network inference framework.

## About Z-Image

Z-Image: An Efficient Image Generation Foundation Model with Single-Stream Diffusion Transformer

https://github.com/Tongyi-MAI/Z-Image

## Usages

TBA

## Build from Source

TBA

## Sample Images

```
prompt="風的彷徨."
size=1024x1024
steps=9
seed=77
```

![zimage](images/77.jpg)

```
prompt="風的彷徨."
size=1024x1024
steps=9
seed=777
```

![zimage](images/777.jpg)

## Original Z-Image Project

- https://github.com/Tongyi-MAI/Z-Image

## Other Open-Source Code Used

- https://github.com/Tencent/ncnn for fast neural network inference on ALL PLATFORMS
- https://github.com/futz12/ncnn_llm for BPE tokenizer
