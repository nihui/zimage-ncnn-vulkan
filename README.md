# Z-Image ncnn Vulkan

:exclamation: :exclamation: :exclamation: This software is in the early development stage, it may bite your cat

ncnn implementation of Z-Image image generater.

zimage-ncnn-vulkan uses [ncnn project](https://github.com/Tencent/ncnn) as the universal neural network inference framework.

## About Z-Image

Z-Image: An Efficient Image Generation Foundation Model with Single-Stream Diffusion Transformer

https://github.com/Tongyi-MAI/Z-Image

## Usages

Currently, this program requires a Vulkan capable graphics card with at least 16GB of VRAM to run.

This program requires code from the ncnn git master branch to achieve correct results and optimal performance.

Further performance and video memory usage optimizations are underway. Please stay tuned :)

### prepare model files

https://huggingface.co/nihui-szyl/z-image-ncnn/tree/main/z-image-turbo

## Build from Source

1. Clone this project with all submodules

```shell
git clone https://github.com/nihui/zimage-ncnn-vulkan.git
cd zimage-ncnn-vulkan
git submodule update --init --recursive --depth 1
```

2. Build with CMake

```shell
mkdir build
cd build
cmake ../src
cmake --build . -j 4
```

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
- https://github.com/webmproject/libwebp for encoding and decoding Webp images on ALL PLATFORMS
- https://github.com/libjpeg-turbo/libjpeg-turbo for encoding and decoding JPEG images on ALL PLATFORMS
- https://github.com/pnggroup/libpng for encoding and decoding PNG images on ALL PLATFORMS
- https://github.com/zlib-ng/zlib-ng for encoding and decoding PNG images on ALL PLATFORMS
