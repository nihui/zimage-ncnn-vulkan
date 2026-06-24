// z-image implemented with ncnn library

#ifndef LANPAINT_H
#define LANPAINT_H

#include <memory>
#include <stdint.h>
#include <vector>

#include "filesystem_utils.h"
#include "mat.h"
#include "net.h"
#include "zimage.h"

class LanPaintPipeline
{
public:
    path_t prompt;
    path_t negative_prompt;
    path_t outpath;
    path_t inputpath;
    path_t maskpath;
    path_t controlpath;
    path_t model;

    int width = 1024;
    int height = 1024;
    bool size_set = false;
    int steps = -1;
    int seed = 0;
    int gpuid = 233;
    int batch = 1;
    float control_scale = 1.f;

    int outpaint[4] = {0, 0, 0, 0};
    bool outpaint_set = false;

    int lanpaint_steps = 5;
    float lanpaint_lambda = 16.f;
    float lanpaint_step_size = 0.2f;
    float lanpaint_beta = 1.f;
    float lanpaint_friction = 15.f;
    int lanpaint_early_stop = 1;
    bool lanpaint_prompt_first = false;
    bool preserve_known = false;

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

    int prepare_input(int& width, int& height, bool& input_enabled, ncnn::Mat& source_canvas, ncnn::Mat& paint_mask_pixels) const;
    int encode_input(
        const ncnn::Option& opt,
        int width,
        int height,
        int vae_tile_width,
        int vae_tile_height,
        bool input_enabled,
        const ncnn::Mat& source_canvas,
        const ncnn::Mat& paint_mask_pixels,
        ncnn::Mat& source_latent,
        ncnn::Mat& source_x,
        ncnn::Mat& paint_mask_x,
        ncnn::Mat& known_mask_x) const;
    int sample(
        std::vector<ncnn::Mat>& latents,
        const std::vector<ncnn::Mat>& noise_latents,
        bool input_enabled,
        const ncnn::Mat& source_x,
        const ncnn::Mat& paint_mask_x,
        const ncnn::Mat& known_mask_x,
        const ncnn::Mat& control_x,
        const ncnn::Option& opt,
        const std::vector<float>& sigmas,
        const ncnn::Mat& t_embeds,
        const ncnn::Mat& x_cos,
        const ncnn::Mat& x_sin,
        const ncnn::Mat& neg_x_cos,
        const ncnn::Mat& neg_x_sin,
        const ncnn::Mat& cap_refine,
        const ncnn::Mat& unified_cos,
        const ncnn::Mat& unified_sin,
        const ncnn::Mat& neg_cap_refine,
        const ncnn::Mat& neg_unified_cos,
        const ncnn::Mat& neg_unified_sin,
        bool apply_cfg,
        float guidance_scale,
        bool control_enabled,
        float control_scale,
        int steps) const;
    void composite_known_pixels(ncnn::Mat& outimage, const ncnn::Mat& source_canvas, const ncnn::Mat& paint_mask_pixels) const;
};

#endif // LANPAINT_H
