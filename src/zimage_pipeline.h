// z-image implemented with ncnn library

#ifndef ZIMAGE_PIPELINE_H
#define ZIMAGE_PIPELINE_H

#include <memory>
#include <stdint.h>

#include "filesystem_utils.h"
#include "zimage.h"

class ZImagePipeline
{
public:
    path_t prompt;
    path_t negative_prompt;
    path_t outpath;
    path_t controlpath;
    path_t model;

    int width = 1024;
    int height = 1024;
    int steps = -1;
    int seed = 0;
    int gpuid = 233;
    int batch = 1;
    float control_scale = 1.f;

    int load();
    int generate() const;

private:
    bool loaded = false;
    int loaded_gpuid = 233;
    float guidance_scale = 0.f;
    float scheduler_shift = 3.f;
    uint32_t heap_budget = 0;
    ncnn::Option opt;

    std::unique_ptr<ZImage::Tokenizer> tokenizer;
};

#endif // ZIMAGE_PIPELINE_H
