// z-image implemented with ncnn library

#include "lanpaint.h"

#include <algorithm>
#include <math.h>
#include <random>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "image_io.h"
#include "zimage.h"

static float mask_pixel_value(const ncnn::Mat& image, int x, int y)
{
    const int c = image.elempack;
    const unsigned char* p = (const unsigned char*)image.data + ((size_t)y * image.w + x) * c;
    if (c == 1)
        return p[0] > 127 ? 1.f : 0.f;

    int v = 0;
    if (c >= 3)
        v = (int)p[0] + (int)p[1] + (int)p[2];
    else
        v = (int)p[0] * 3;

    if (c == 4 && p[3] == 0)
        return 0.f;

    return v > 127 * 3 ? 1.f : 0.f;
}

static float outpaint_inner_overlap_value(int x, int y, int image_w, int image_h, const int outpaint[4], int overlap)
{
    if (overlap <= 0)
        return 0.f;

    const int left = outpaint[0];
    const int top = outpaint[1];
    const int right = left + image_w - 1;
    const int bottom = top + image_h - 1;

    int distance = overlap;
    if (outpaint[0] > 0)
        distance = std::min(distance, x - left);
    if (outpaint[1] > 0)
        distance = std::min(distance, y - top);
    if (outpaint[2] > 0)
        distance = std::min(distance, right - x);
    if (outpaint[3] > 0)
        distance = std::min(distance, bottom - y);

    if (distance >= overlap)
        return 0.f;

    return 1.f;
}

static void image_to_ncnn_rgb_float(const ncnn::Mat& image, ncnn::Mat& out)
{
    out.create(image.w, image.h, 3);

    const int c = image.elempack;
    for (int y = 0; y < image.h; y++)
    {
        float* rptr = out.channel(0).row(y);
        float* gptr = out.channel(1).row(y);
        float* bptr = out.channel(2).row(y);
        for (int x = 0; x < image.w; x++)
        {
            const unsigned char* p = (const unsigned char*)image.data + ((size_t)y * image.w + x) * c;
            rptr[x] = (float)p[0] / 127.5f - 1.f;
            gptr[x] = (float)(c >= 2 ? p[1] : p[0]) / 127.5f - 1.f;
            bptr[x] = (float)(c >= 3 ? p[2] : p[0]) / 127.5f - 1.f;
        }
    }
}

static void make_latent_mask_from_pixels(const ncnn::Mat& paint_pixels, int width, int height, ncnn::Mat& paint_latent_mask)
{
    const int latent_w = width / 8;
    const int latent_h = height / 8;

    paint_latent_mask.create(latent_w, latent_h);
    for (int y = 0; y < latent_h; y++)
    {
        float* row = paint_latent_mask.row(y);
        int sy = std::min(height - 1, y * 8 + 4);
        const float* srcrow = paint_pixels.row(sy);
        for (int x = 0; x < latent_w; x++)
        {
            int sx = std::min(width - 1, x * 8 + 4);
            row[x] = srcrow[sx] > 0.5f ? 1.f : 0.f;
        }
    }
}

static void patchify_mask(const ncnn::Mat& latent_mask, ncnn::Mat& x_mask)
{
    const int patch_size = 2;
    const int channels = 16;
    const int num_patches_w = latent_mask.w / patch_size;
    const int num_patches_h = latent_mask.h / patch_size;

    x_mask.create(patch_size * patch_size * channels, num_patches_w * num_patches_h);

    for (int py = 0; py < num_patches_h; py++)
    {
        for (int px = 0; px < num_patches_w; px++)
        {
            const float m00 = latent_mask.row(py * 2)[px * 2];
            const float m01 = latent_mask.row(py * 2)[px * 2 + 1];
            const float m10 = latent_mask.row(py * 2 + 1)[px * 2];
            const float m11 = latent_mask.row(py * 2 + 1)[px * 2 + 1];

            float* outptr = x_mask.row(py * num_patches_w + px);
            for (int q = 0; q < channels; q++) *outptr++ = m00;
            for (int q = 0; q < channels; q++) *outptr++ = m01;
            for (int q = 0; q < channels; q++) *outptr++ = m10;
            for (int q = 0; q < channels; q++) *outptr++ = m11;
        }
    }
}

static void invert_mask(const ncnn::Mat& mask, ncnn::Mat& inverted)
{
    inverted.create(mask.w, mask.h);
    for (int i = 0; i < mask.total(); i++)
    {
        inverted[i] = 1.f - mask[i];
    }
}

static int predict_velocity_pair(
    const ZImage::AllXEmbedder& all_x_embedder,
    const ZImage::NoiseRefiner& noise_refiner,
    const ZImage::UnifiedRefiner& unified_refiner,
    const ZImage::AllFinalLayer& all_final_layer,
    const ncnn::Mat& x,
    const ncnn::Mat& x_cos,
    const ncnn::Mat& x_sin,
    const ncnn::Mat& t_embed,
    const ncnn::Mat& cap_refine,
    const ncnn::Mat& unified_cos,
    const ncnn::Mat& unified_sin,
    const ncnn::Mat& neg_cap_refine,
    const ncnn::Mat& neg_unified_cos,
    const ncnn::Mat& neg_unified_sin,
    bool apply_cfg,
    float guidance_scale,
    bool want_big,
    float big_guidance_scale,
    ncnn::Mat& velocity,
    ncnn::Mat& velocity_big)
{
    ncnn::Mat x_embed;
    all_x_embedder.process(x, x_embed);

    ncnn::Mat x_embed_refine;
    noise_refiner.process(x_embed, x_cos, x_sin, t_embed, x_embed_refine);

    ncnn::Mat unified_embed;
    ZImage::concat_along_h(x_embed_refine, cap_refine, unified_embed);

    ncnn::Mat unified;
    unified_refiner.process(unified_embed, unified_cos, unified_sin, t_embed, unified);

    ncnn::Mat pos_final;
    all_final_layer.process(unified, t_embed, pos_final);

    ncnn::Mat neg_final;
    if (apply_cfg)
    {
        ncnn::Mat neg_unified_embed;
        ZImage::concat_along_h(x_embed_refine, neg_cap_refine, neg_unified_embed);

        ncnn::Mat neg_unified;
        unified_refiner.process(neg_unified_embed, neg_unified_cos, neg_unified_sin, t_embed, neg_unified);

        all_final_layer.process(neg_unified, t_embed, neg_final);
    }

    velocity.create(x.w, x.h);
    if (want_big)
        velocity_big.create(x.w, x.h);

    const int total = x.total();
    for (int i = 0; i < total; i++)
    {
        const float pos = pos_final[i];
        if (apply_cfg)
        {
            const float neg = neg_final[i];
            velocity[i] = pos + guidance_scale * (pos - neg);
            if (want_big)
                velocity_big[i] = pos + big_guidance_scale * (pos - neg);
        }
        else
        {
            velocity[i] = pos;
            if (want_big)
                velocity_big[i] = pos;
        }
    }

    return 0;
}

static int predict_velocity_pair_controlled(
    const ZImage::AllXEmbedder& all_x_embedder,
    const ZImage::NoiseRefiner& noise_refiner,
    const ZImage::UnifiedRefiner& unified_refiner,
    const ZImage::AllFinalLayer& all_final_layer,
    const ZImage::ControlRefiner& control_refiner,
    const ZImage::ControlUnified& control_unified,
    const ncnn::Mat& control_x,
    const ncnn::Mat& x,
    const ncnn::Mat& x_cos,
    const ncnn::Mat& x_sin,
    const ncnn::Mat& neg_x_cos,
    const ncnn::Mat& neg_x_sin,
    const ncnn::Mat& t_embed,
    const ncnn::Mat& cap_refine,
    const ncnn::Mat& unified_cos,
    const ncnn::Mat& unified_sin,
    const ncnn::Mat& neg_cap_refine,
    const ncnn::Mat& neg_unified_cos,
    const ncnn::Mat& neg_unified_sin,
    bool apply_cfg,
    float guidance_scale,
    float control_scale,
    bool want_big,
    float big_guidance_scale,
    ncnn::Mat& velocity,
    ncnn::Mat& velocity_big)
{
    ncnn::Mat x_embed;
    all_x_embedder.process(x, x_embed);

    ncnn::Mat hint0;
    ncnn::Mat hint1;
    ncnn::Mat control_context;
    if (control_refiner.process(control_x, x_embed, x_cos, x_sin, t_embed, hint0, hint1, control_context) != 0)
        return -1;

    ncnn::Mat x_embed_refine;
    if (noise_refiner.process_controlled(x_embed, x_cos, x_sin, t_embed, hint0, hint1, control_scale, x_embed_refine) != 0)
        return -1;

    ncnn::Mat neg_x_embed_refine;
    ncnn::Mat neg_control_context;
    if (apply_cfg)
    {
        ncnn::Mat neg_hint0;
        ncnn::Mat neg_hint1;
        if (control_refiner.process(control_x, x_embed, neg_x_cos, neg_x_sin, t_embed, neg_hint0, neg_hint1, neg_control_context) != 0)
            return -1;
        if (noise_refiner.process_controlled(x_embed, neg_x_cos, neg_x_sin, t_embed, neg_hint0, neg_hint1, control_scale, neg_x_embed_refine) != 0)
            return -1;
    }

    ncnn::Mat unified_embed;
    ZImage::concat_along_h(x_embed_refine, cap_refine, unified_embed);

    ncnn::Mat control_unified_embed;
    ZImage::concat_along_h(control_context, cap_refine, control_unified_embed);

    ncnn::Mat unified_hint0;
    ncnn::Mat unified_hint10;
    ncnn::Mat unified_hint20;
    if (control_unified.process(control_unified_embed, unified_embed, unified_cos, unified_sin, t_embed, unified_hint0, unified_hint10, unified_hint20) != 0)
        return -1;

    ncnn::Mat unified;
    if (unified_refiner.process_controlled(unified_embed, unified_cos, unified_sin, t_embed, unified_hint0, unified_hint10, unified_hint20, control_scale, unified) != 0)
        return -1;

    ncnn::Mat pos_final;
    all_final_layer.process(unified, t_embed, pos_final);

    ncnn::Mat neg_final;
    if (apply_cfg)
    {
        ncnn::Mat neg_unified_embed;
        ZImage::concat_along_h(neg_x_embed_refine, neg_cap_refine, neg_unified_embed);

        ncnn::Mat neg_control_unified_embed;
        ZImage::concat_along_h(neg_control_context, neg_cap_refine, neg_control_unified_embed);

        ncnn::Mat neg_unified_hint0;
        ncnn::Mat neg_unified_hint10;
        ncnn::Mat neg_unified_hint20;
        if (control_unified.process(neg_control_unified_embed, neg_unified_embed, neg_unified_cos, neg_unified_sin, t_embed, neg_unified_hint0, neg_unified_hint10, neg_unified_hint20) != 0)
            return -1;

        ncnn::Mat neg_unified;
        if (unified_refiner.process_controlled(neg_unified_embed, neg_unified_cos, neg_unified_sin, t_embed, neg_unified_hint0, neg_unified_hint10, neg_unified_hint20, control_scale, neg_unified) != 0)
            return -1;

        all_final_layer.process(neg_unified, t_embed, neg_final);
    }

    velocity.create(x.w, x.h);
    if (want_big)
        velocity_big.create(x.w, x.h);

    const int total = x.total();
    for (int i = 0; i < total; i++)
    {
        const float pos = pos_final[i];
        if (apply_cfg)
        {
            const float neg = neg_final[i];
            velocity[i] = pos + guidance_scale * (pos - neg);
            if (want_big)
                velocity_big[i] = pos + big_guidance_scale * (pos - neg);
        }
        else
        {
            velocity[i] = pos;
            if (want_big)
                velocity_big[i] = pos;
        }
    }

    return 0;
}

int LanPaintPipeline::load()
{
    if (model.find(PATHSTR("z-image-turbo")) != path_t::npos)
    {
        guidance_scale = 0.f;
        scheduler_shift = 3.f;
    }
    else if (model.find(PATHSTR("z-image")) != path_t::npos)
    {
        guidance_scale = 1.f;
        scheduler_shift = 6.f;
    }
    else
    {
        fprintf(stderr, "unknown model dir type\n");
        return -1;
    }

    loaded_gpuid = gpuid;
    if (loaded_gpuid == 233)
    {
        loaded_gpuid = ncnn::get_default_gpu_index();
    }

    opt = ncnn::Option();
    opt.vulkan_device_index = loaded_gpuid;
    opt.use_vulkan_compute = loaded_gpuid >= 0;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_packed = true;
    opt.use_bf16_storage = true;
    // opt.use_mapped_model_loading = true;

    // disable winograd for reducing vae vram usage
    opt.use_winograd_convolution = false;

    heap_budget = loaded_gpuid >= 0 ? ncnn::get_gpu_device(loaded_gpuid)->get_heap_budget() : 0;

    tokenizer.reset(new ZImage::Tokenizer(model));

    loaded = true;
    return 0;
}

int LanPaintPipeline::prepare_input(int& width, int& height, bool& input_enabled, ncnn::Mat& source_canvas, ncnn::Mat& paint_mask_pixels) const
{
    input_enabled = false;
    source_canvas.release();
    paint_mask_pixels.release();

    if (inputpath.empty())
    {
        if (!maskpath.empty() || outpaint_set)
        {
            fprintf(stderr, "--mask/--outpaint requires -i input-image\n");
            return -1;
        }

        return 0;
    }

    ncnn::Mat input_image;
    if (ImageIO::load_image(inputpath, input_image) != 0)
    {
#if _WIN32
        fwprintf(stderr, L"load input image %ls failed\n", inputpath.c_str());
#else
        fprintf(stderr, "load input image %s failed\n", inputpath.c_str());
#endif
        return -1;
    }

    ncnn::Mat mask_image;
    bool has_mask = false;
    if (!maskpath.empty())
    {
        if (ImageIO::load_image(maskpath, mask_image) != 0)
        {
#if _WIN32
            fwprintf(stderr, L"load mask image %ls failed\n", maskpath.c_str());
#else
            fprintf(stderr, "load mask image %s failed\n", maskpath.c_str());
#endif
            return -1;
        }
        has_mask = true;
    }

    if (outpaint_set)
    {
        for (int i = 0; i < 4; i++)
        {
            if (outpaint[i] < 0)
            {
                fprintf(stderr, "outpaint values must be >= 0\n");
                return -1;
            }
        }

        const int canvas_w = input_image.w + outpaint[0] + outpaint[2];
        const int canvas_h = input_image.h + outpaint[1] + outpaint[3];

        if (size_set && (width != canvas_w || height != canvas_h))
        {
            fprintf(stderr, "explicit image-size must match outpaint canvas size %d x %d\n", canvas_w, canvas_h);
            return -1;
        }

        source_canvas.create(canvas_w, canvas_h, (size_t)3u, 3);
        memset(source_canvas.data, 0, (size_t)canvas_w * canvas_h * 3);

        paint_mask_pixels.create(canvas_w, canvas_h);
        paint_mask_pixels.fill(1.f);

        if (has_mask && (mask_image.w != input_image.w || mask_image.h != input_image.h))
        {
            fprintf(stderr, "mask size must match input image size for outpaint\n");
            return -1;
        }

        const int src_c = input_image.elempack;
        const int outpaint_overlap = std::min(20, std::max(0, std::min(input_image.w, input_image.h) / 2));
        for (int y = 0; y < input_image.h; y++)
        {
            for (int x = 0; x < input_image.w; x++)
            {
                const int dx = x + outpaint[0];
                const int dy = y + outpaint[1];
                const unsigned char* src = (const unsigned char*)input_image.data + ((size_t)y * input_image.w + x) * src_c;
                unsigned char* dst = (unsigned char*)source_canvas.data + ((size_t)dy * canvas_w + dx) * 3;
                dst[0] = src[0];
                dst[1] = src_c >= 2 ? src[1] : src[0];
                dst[2] = src_c >= 3 ? src[2] : src[0];

                float paint = outpaint_inner_overlap_value(dx, dy, input_image.w, input_image.h, outpaint, outpaint_overlap);
                if (has_mask)
                    paint = std::max(paint, mask_pixel_value(mask_image, x, y));
                paint_mask_pixels.row(dy)[dx] = paint;
            }
        }

        width = canvas_w;
        height = canvas_h;
    }
    else
    {
        if (!has_mask)
        {
            fprintf(stderr, "input image requires -k/--mask or --outpaint\n");
            return -1;
        }

        if (mask_image.w != input_image.w || mask_image.h != input_image.h)
        {
            fprintf(stderr, "mask size must match input image size\n");
            return -1;
        }

        if (size_set && (width != input_image.w || height != input_image.h))
        {
            fprintf(stderr, "explicit image-size must match input image size %d x %d\n", input_image.w, input_image.h);
            return -1;
        }

        source_canvas = input_image;
        paint_mask_pixels.create(input_image.w, input_image.h);
        paint_mask_pixels.fill(0.f);
        for (int y = 0; y < input_image.h; y++)
        {
            float* row = paint_mask_pixels.row(y);
            for (int x = 0; x < input_image.w; x++)
            {
                row[x] = mask_pixel_value(mask_image, x, y);
            }
        }

        width = input_image.w;
        height = input_image.h;
    }

    input_enabled = true;
    return 0;
}

int LanPaintPipeline::encode_input(
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
    ncnn::Mat& known_mask_x) const
{
    if (!input_enabled)
        return 0;

    ncnn::Mat source_image_float;
    image_to_ncnn_rgb_float(source_canvas, source_image_float);

    const bool use_vae_tiled = vae_tile_width < width || vae_tile_height < height;

    ZImage::VAEEncoder vae_encoder;
    vae_encoder.load(model, use_vae_tiled, opt);
    if (use_vae_tiled)
    {
        vae_encoder.process_tiled(source_image_float, vae_tile_width, vae_tile_height, source_latent);
    }
    else
    {
        vae_encoder.process(source_image_float, source_latent);
    }

    if (source_latent.w != width / 8 || source_latent.h != height / 8 || source_latent.c != 16)
    {
        fprintf(stderr, "vae encoder output size mismatch, got %d x %d x %d expected %d x %d x 16\n",
                source_latent.w, source_latent.h, source_latent.c, width / 8, height / 8);
        return -1;
    }

    ZImage::patchify(source_latent, source_x);

    ncnn::Mat paint_latent_mask;
    make_latent_mask_from_pixels(paint_mask_pixels, width, height, paint_latent_mask);
    patchify_mask(paint_latent_mask, paint_mask_x);
    invert_mask(paint_mask_x, known_mask_x);

    return 0;
}

int LanPaintPipeline::sample(
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
    int steps) const
{
    ZImage::AllXEmbedder all_x_embedder;
    all_x_embedder.load(model, opt);

    ZImage::NoiseRefiner noise_refiner;
    noise_refiner.load(model, opt);

    ZImage::UnifiedRefiner unified_refiner;
    unified_refiner.load(model, opt);

    ZImage::AllFinalLayer all_final_layer;
    all_final_layer.load(model, opt);

    ZImage::ControlRefiner control_refiner;
    ZImage::ControlUnified control_unified;
    if (control_enabled)
    {
        if (control_refiner.load(model, opt) != 0)
        {
            fprintf(stderr, "load control refiner failed\n");
            return -1;
        }
        if (control_unified.load(model, opt) != 0)
        {
            fprintf(stderr, "load control unified failed\n");
            return -1;
        }
    }

    for (int b = 0; b < batch; b++)
    {
        ncnn::Mat x;
        ZImage::patchify(latents[b], x);

        ncnn::Mat noise_x;
        if (input_enabled)
            ZImage::patchify(noise_latents[b], noise_x);

        std::mt19937 inner_rng(seed + b * 1000003 + 1777);
        std::normal_distribution<float> inner_normal(0.f, 1.f);

        for (int z = 0; z < steps; z++)
        {
            ncnn::Mat t_embed = t_embeds.row_range(z, 1).clone();

            if (input_enabled)
            {
                const float sigma = sigmas[z];
                const float dt = sigmas[z + 1] - sigmas[z];
                const int total = x.total();

                for (int i = 0; i < total; i++)
                {
                    const float paint = paint_mask_x[i];
                    const float known = known_mask_x[i];
                    const float known_xt = sigma * noise_x[i] + (1.f - sigma) * source_x[i];
                    x[i] = x[i] * paint + known_xt * known;
                }

                int inner_steps = lanpaint_steps;
                if (steps - z <= lanpaint_early_stop || sigma <= 1e-6f)
                    inner_steps = 0;

                if (inner_steps > 0)
                {
                    const float one_minus_sigma = 1.f - sigma;
                    const float denom_flow = one_minus_sigma * one_minus_sigma + sigma * sigma;
                    const float abt = denom_flow > 1e-20f ? (one_minus_sigma * one_minus_sigma) / denom_flow : 0.f;
                    const float one_minus_abt = std::max(1e-6f, 1.f - abt);
                    const float sqrt_abt = sqrtf(std::max(0.f, abt));
                    const float vp_scale = sqrt_abt + sqrtf(std::max(0.f, 1.f - abt));
                    const float inner_step_size = lanpaint_step_size * (1.f - abt);
                    const float dt_x = inner_step_size;
                    const float dt_y = inner_step_size * lanpaint_beta;
                    const float A_x = 1.f / one_minus_abt;
                    const float A_y = (1.f + lanpaint_lambda) / one_minus_abt;
                    const float big_guidance_scale = lanpaint_prompt_first ? -0.5f : guidance_scale;

                    ncnn::Mat x_t(x.w, x.h);
                    for (int i = 0; i < total; i++)
                    {
                        x_t[i] = x[i] * vp_scale;
                    }

                    for (int inner = 0; inner < inner_steps; inner++)
                    {
                        ncnn::Mat x_flow(x.w, x.h);
                        for (int i = 0; i < total; i++)
                        {
                            x_flow[i] = x_t[i] / vp_scale;
                        }

                        ncnn::Mat velocity;
                        ncnn::Mat velocity_big;
                        if (control_enabled)
                        {
                            if (predict_velocity_pair_controlled(all_x_embedder, noise_refiner, unified_refiner, all_final_layer,
                                                                 control_refiner, control_unified, control_x,
                                                                 x_flow, x_cos, x_sin, neg_x_cos, neg_x_sin, t_embed, cap_refine, unified_cos, unified_sin,
                                                                 neg_cap_refine, neg_unified_cos, neg_unified_sin,
                                                                 apply_cfg, guidance_scale, control_scale, true, big_guidance_scale, velocity, velocity_big) != 0)
                                return -1;
                        }
                        else
                        {
                            if (predict_velocity_pair(all_x_embedder, noise_refiner, unified_refiner, all_final_layer,
                                                      x_flow, x_cos, x_sin, t_embed, cap_refine, unified_cos, unified_sin,
                                                      neg_cap_refine, neg_unified_cos, neg_unified_sin,
                                                      apply_cfg, guidance_scale, true, big_guidance_scale, velocity, velocity_big) != 0)
                                return -1;
                        }

                        for (int i = 0; i < total; i++)
                        {
                            const float x0 = x_flow[i] + sigma * velocity[i];
                            const float x0_big = x_flow[i] + sigma * velocity_big[i];
                            const float score_x = x0 - x_t[i];
                            const float score_y = -(1.f + lanpaint_lambda) * (x_t[i] - source_x[i]) + lanpaint_lambda * (x_t[i] - x0_big);
                            const float paint = paint_mask_x[i];
                            const float known = known_mask_x[i];
                            const float score = score_x * paint + score_y * known;
                            const float x0_from_score = x_t[i] + score;
                            const float A = A_x * paint + A_y * known;
                            const float step_dt = dt_x * paint + dt_y * known;

                            if (step_dt <= 0.f)
                                continue;

                            const float C = (sqrt_abt * x0_from_score - x_t[i]) / one_minus_abt + A * x_t[i];
                            const float A_dt = A * step_dt;
                            const float exp_neg = expf(-A_dt);
                            const float k = fabsf(A) < 1e-8f ? step_dt : (-expm1f(-A_dt)) / A;
                            const float k2 = fabsf(A) < 1e-8f ? step_dt : (-expm1f(-2.f * A_dt)) / (2.f * A);
                            const float mean = exp_neg * x_t[i] + k * C;
                            const float var = std::max(0.f, 2.f * k2);
                            x_t[i] = mean + inner_normal(inner_rng) * sqrtf(var);
                        }
                    }

                    for (int i = 0; i < total; i++)
                    {
                        x[i] = x_t[i] / vp_scale;
                    }
                }

                if (fabsf(dt) > 1e-8f)
                {
                    ncnn::Mat velocity;
                    ncnn::Mat velocity_big;
                    if (control_enabled)
                    {
                        if (predict_velocity_pair_controlled(all_x_embedder, noise_refiner, unified_refiner, all_final_layer,
                                                             control_refiner, control_unified, control_x,
                                                             x, x_cos, x_sin, neg_x_cos, neg_x_sin, t_embed, cap_refine, unified_cos, unified_sin,
                                                             neg_cap_refine, neg_unified_cos, neg_unified_sin,
                                                             apply_cfg, guidance_scale, control_scale, false, guidance_scale, velocity, velocity_big) != 0)
                            return -1;
                    }
                    else
                    {
                        if (predict_velocity_pair(all_x_embedder, noise_refiner, unified_refiner, all_final_layer,
                                                  x, x_cos, x_sin, t_embed, cap_refine, unified_cos, unified_sin,
                                                  neg_cap_refine, neg_unified_cos, neg_unified_sin,
                                                  apply_cfg, guidance_scale, false, guidance_scale, velocity, velocity_big) != 0)
                            return -1;
                    }

                    for (int i = 0; i < total; i++)
                    {
                        float corrected_velocity = velocity[i];
                        if (sigma > 1e-6f)
                        {
                            const float paint = paint_mask_x[i];
                            const float known = known_mask_x[i];
                            float x0 = x[i] + sigma * velocity[i];
                            x0 = x0 * paint + source_x[i] * known;
                            corrected_velocity = (x0 - x[i]) / sigma;
                        }
                        x[i] = x[i] - dt * corrected_velocity;
                    }
                }
                else
                {
                    for (int i = 0; i < total; i++)
                    {
                        const float paint = paint_mask_x[i];
                        const float known = known_mask_x[i];
                        x[i] = x[i] * paint + source_x[i] * known;
                    }
                }
            }
            else
            {
                ncnn::Mat velocity;
                ncnn::Mat velocity_big;
                if (control_enabled)
                {
                    if (predict_velocity_pair_controlled(all_x_embedder, noise_refiner, unified_refiner, all_final_layer,
                                                         control_refiner, control_unified, control_x,
                                                         x, x_cos, x_sin, neg_x_cos, neg_x_sin, t_embed, cap_refine, unified_cos, unified_sin,
                                                         neg_cap_refine, neg_unified_cos, neg_unified_sin,
                                                         apply_cfg, guidance_scale, control_scale, false, guidance_scale, velocity, velocity_big) != 0)
                        return -1;
                }
                else
                {
                    if (predict_velocity_pair(all_x_embedder, noise_refiner, unified_refiner, all_final_layer,
                                              x, x_cos, x_sin, t_embed, cap_refine, unified_cos, unified_sin,
                                              neg_cap_refine, neg_unified_cos, neg_unified_sin,
                                              apply_cfg, guidance_scale, false, guidance_scale, velocity, velocity_big) != 0)
                        return -1;
                }

                const float dt = sigmas[z + 1] - sigmas[z];
                const int total = x.total();
                for (int i = 0; i < total; i++)
                {
                    x[i] = x[i] - dt * velocity[i];
                }
            }

            if (batch > 1)
                fprintf(stderr, "step %d/%d of image %d/%d done\n", z + 1, steps, b + 1, batch);
            else
                fprintf(stderr, "step %d/%d done\n", z + 1, steps);
        }

        ZImage::unpatchify(x, latents[b]);
    }

    return 0;
}

void LanPaintPipeline::composite_known_pixels(ncnn::Mat& outimage, const ncnn::Mat& source_canvas, const ncnn::Mat& paint_mask_pixels) const
{
    if (outimage.w != source_canvas.w || outimage.h != source_canvas.h)
        return;

    const int blend_overlap = 9;
    const int blend_radius = blend_overlap / 2;

    ncnn::Mat dilated_mask(source_canvas.w, source_canvas.h);
    for (int y = 0; y < source_canvas.h; y++)
    {
        float* dstrow = dilated_mask.row(y);
        for (int x = 0; x < source_canvas.w; x++)
        {
            float value = 0.f;
            for (int ky = -blend_radius; ky <= blend_radius && value == 0.f; ky++)
            {
                const int sy = y + ky;
                if (sy < 0 || sy >= source_canvas.h)
                    continue;

                const float* srcrow = paint_mask_pixels.row(sy);
                for (int kx = -blend_radius; kx <= blend_radius; kx++)
                {
                    const int sx = x + kx;
                    if (sx < 0 || sx >= source_canvas.w)
                        continue;

                    if (srcrow[sx] > 0.5f)
                    {
                        value = 1.f;
                        break;
                    }
                }
            }
            dstrow[x] = value;
        }
    }

    float kernel[blend_overlap * blend_overlap];
    {
        const float sigma = (float)(blend_overlap - 1) / 4.f;
        const float denom = 2.f * sigma * sigma;
        float sum = 0.f;
        for (int y = 0; y < blend_overlap; y++)
        {
            const int yy = y - blend_radius;
            for (int x = 0; x < blend_overlap; x++)
            {
                const int xx = x - blend_radius;
                const float v = expf(-((float)(xx * xx + yy * yy)) / denom);
                kernel[y * blend_overlap + x] = v;
                sum += v;
            }
        }

        for (int i = 0; i < blend_overlap * blend_overlap; i++)
            kernel[i] /= sum;
    }

    ncnn::Mat blend_mask(source_canvas.w, source_canvas.h);
    for (int y = 0; y < source_canvas.h; y++)
    {
        float* dstrow = blend_mask.row(y);
        const float* paintrow = paint_mask_pixels.row(y);
        for (int x = 0; x < source_canvas.w; x++)
        {
            if (paintrow[x] > 0.5f)
            {
                dstrow[x] = 1.f;
                continue;
            }

            float sum = 0.f;
            for (int ky = -blend_radius; ky <= blend_radius; ky++)
            {
                const int sy = y + ky;
                if (sy < 0 || sy >= source_canvas.h)
                    continue;

                const float* srcrow = dilated_mask.row(sy);
                for (int kx = -blend_radius; kx <= blend_radius; kx++)
                {
                    const int sx = x + kx;
                    if (sx < 0 || sx >= source_canvas.w)
                        continue;

                    sum += srcrow[sx] * kernel[(ky + blend_radius) * blend_overlap + (kx + blend_radius)];
                }
            }

            dstrow[x] = sum;
        }
    }

    const int source_c = source_canvas.elempack;
    unsigned char* outptr = (unsigned char*)outimage.data;

    for (int y = 0; y < source_canvas.h; y++)
    {
        const float* maskrow = blend_mask.row(y);
        for (int x = 0; x < source_canvas.w; x++)
        {
            const float paint = maskrow[x];
            if (paint >= 1.f)
                continue;

            const unsigned char* src = (const unsigned char*)source_canvas.data + ((size_t)y * source_canvas.w + x) * source_c;
            unsigned char* dst = outptr + ((size_t)y * source_canvas.w + x) * 3;
#if _WIN32
            const unsigned char src0 = source_c >= 3 ? src[2] : src[0];
            const unsigned char src1 = source_c >= 2 ? src[1] : src[0];
            const unsigned char src2 = src[0];
#else
            const unsigned char src0 = src[0];
            const unsigned char src1 = source_c >= 2 ? src[1] : src[0];
            const unsigned char src2 = source_c >= 3 ? src[2] : src[0];
#endif
            const float known = 1.f - paint;
            dst[0] = (unsigned char)(dst[0] * paint + src0 * known + 0.5f);
            dst[1] = (unsigned char)(dst[1] * paint + src1 * known + 0.5f);
            dst[2] = (unsigned char)(dst[2] * paint + src2 * known + 0.5f);
        }
    }
}

int LanPaintPipeline::generate() const
{
    if (!loaded)
    {
        fprintf(stderr, "pipeline is not loaded\n");
        return -1;
    }

    int width = this->width;
    int height = this->height;
    int steps = this->steps;

    bool input_enabled = false;
    ncnn::Mat source_canvas;
    ncnn::Mat paint_mask_pixels;
    ncnn::Mat source_latent;
    ncnn::Mat source_x;
    ncnn::Mat paint_mask_x;
    ncnn::Mat known_mask_x;

    if (prepare_input(width, height, input_enabled, source_canvas, paint_mask_pixels) != 0)
        return -1;

    // assert width % 16 == 0
    // assert height % 16 == 0
    // assert (width / 16) * (height / 16) >= 32

    if (width % 16 != 0 || height % 16 != 0)
    {
        fprintf(stderr, "width and height must be multiple of 16 but got %d and %d\n", width, height);
        return -1;
    }

    if ((width / 16) * (height / 16) < 32)
    {
        fprintf(stderr, "(width / 16) * (height / 16) must be >= 32\n");
        return -1;
    }

    if (width > 2048 || height > 2048)
    {
        fprintf(stderr, "width and height must be <= 2048 but got %d and %d\n", width, height);
        return -1;
    }

    if (batch <= 0)
    {
        fprintf(stderr, "batch must be > 0 but got %d\n", batch);
        return -1;
    }

    if (model.find(PATHSTR("z-image-turbo")) != path_t::npos)
    {
        if (steps == -1)
            steps = 9;
    }
    else if (model.find(PATHSTR("z-image")) != path_t::npos)
    {
        if (steps == -1)
            steps = 50;
    }
    else
    {
        fprintf(stderr, "unknown model dir type\n");
        return -1;
    }

    const bool control_enabled = !controlpath.empty() && control_scale != 0.f;
    if (control_enabled && model.find(PATHSTR("z-image-turbo")) == path_t::npos)
    {
        fprintf(stderr, "control image currently requires z-image-turbo model\n");
        return -1;
    }

#if _WIN32
    fwprintf(stderr, L"prompt = %ls\n", prompt.c_str());
    fwprintf(stderr, L"negative-prompt = %ls\n", negative_prompt.c_str());
    fwprintf(stderr, L"output-path = %ls\n", outpath.c_str());
    fwprintf(stderr, L"input-image = %ls\n", inputpath.c_str());
    fwprintf(stderr, L"mask-image = %ls\n", maskpath.c_str());
    fwprintf(stderr, L"model = %ls\n", model.c_str());
    if (control_enabled)
        fwprintf(stderr, L"control-image = %ls\n", controlpath.c_str());
#else
    fprintf(stderr, "prompt = %s\n", prompt.c_str());
    fprintf(stderr, "negative-prompt = %s\n", negative_prompt.c_str());
    fprintf(stderr, "output-path = %s\n", outpath.c_str());
    fprintf(stderr, "input-image = %s\n", inputpath.c_str());
    fprintf(stderr, "mask-image = %s\n", maskpath.c_str());
    fprintf(stderr, "model = %s\n", model.c_str());
    if (control_enabled)
        fprintf(stderr, "control-image = %s\n", controlpath.c_str());
#endif
    fprintf(stderr, "image-size = %d x %d\n", width, height);
    fprintf(stderr, "steps = %d\n", steps);
    fprintf(stderr, "seed = %d\n", seed);
    fprintf(stderr, "gpu-id = %d\n", loaded_gpuid);
    fprintf(stderr, "batch = %d\n", batch);
    if (control_enabled)
        fprintf(stderr, "control-scale = %g\n", control_scale);
    fprintf(stderr, "lanpaint = steps:%d lambda:%g step-size:%g beta:%g friction:%g early-stop:%d prompt-mode:%s preserve-known:%d\n",
            lanpaint_steps, lanpaint_lambda, lanpaint_step_size, lanpaint_beta, lanpaint_friction, lanpaint_early_stop,
            lanpaint_prompt_first ? "prompt" : "image", preserve_known ? 1 : 0);

    const bool apply_cfg = guidance_scale > 0.f;

    ncnn::Option opt = this->opt;
    const uint32_t heap_usage_transformer = 13000 + (width * height / (1024 * 1024)) * 1000;
    if (heap_budget < heap_usage_transformer)
    {
        // enable the magic option for low vram graphics  :P
        opt.use_weights_in_host_memory = true;
    }
    NCNN_LOGE("low_vram = %d", opt.use_weights_in_host_memory);

    int vae_tile_width = 0;
    int vae_tile_height = 0;
    {
        int max_tile_area = (int)((float)heap_budget / 6000 * 1024 * 1024);
        ZImage::get_optimal_tile_size(width, height, max_tile_area, &vae_tile_width, &vae_tile_height);

        NCNN_LOGE("vae_tile_size = %d x %d", vae_tile_width, vae_tile_height);
    }

    if (batch > 1)
    {
        path_t filename = get_file_name_without_extension(outpath);
        path_t ext = get_file_extension(outpath);
#if _WIN32
        fwprintf(stderr, L"batch generation enabled. output-path will be %ls-0.%ls %ls-1.%ls %ls-2.%ls ...\n", filename.c_str(), ext.c_str(), filename.c_str(), ext.c_str(), filename.c_str(), ext.c_str());
#else
        fprintf(stderr, "batch generation enabled. output-path will be %s-0.%s %s-1.%s %s-2.%s ...\n", filename.c_str(), ext.c_str(), filename.c_str(), ext.c_str(), filename.c_str(), ext.c_str());
#endif
    }

    ncnn::Mat control_x;
    if (control_enabled)
    {
        ncnn::Mat control_image;
        if (ImageIO::load_image(controlpath, control_image) != 0)
        {
#if _WIN32
            fwprintf(stderr, L"load control image %ls failed\n", controlpath.c_str());
#else
            fprintf(stderr, "load control image %s failed\n", controlpath.c_str());
#endif
            return -1;
        }

        if (control_image.w != width || control_image.h != height)
        {
            fprintf(stderr, "control image size must match final canvas %d x %d but got %d x %d\n",
                    width, height, control_image.w, control_image.h);
            return -1;
        }

        const bool use_vae_tiled = vae_tile_width < width || vae_tile_height < height;
        if (ZImage::prepare_control_x(control_image, model, use_vae_tiled, vae_tile_width, vae_tile_height, opt, control_x) != 0)
            return -1;
    }

    // tokenizer
    std::vector<int> input_ids;
    std::vector<int> neg_input_ids;
    {
        tokenizer->encode(prompt, input_ids);

        if (apply_cfg)
        {
            tokenizer->encode(negative_prompt, neg_input_ids);
        }
    }

    // text encoder
    ncnn::Mat cap;
    ncnn::Mat neg_cap;
    {
        ZImage::TextEncoder text_encoder;

        text_encoder.load(model, opt);

        text_encoder.process(input_ids, cap);

        if (apply_cfg)
        {
            text_encoder.process(neg_input_ids, neg_cap);
        }
    }

    // prepare latents
    std::vector<ncnn::Mat> latents(batch);
    std::vector<ncnn::Mat> noise_latents(batch);
    for (int b = 0; b < batch; b++)
    {
        ZImage::generate_latent(width, height, seed + b, latents[b]);
        noise_latents[b] = latents[b].clone();
    }

    if (encode_input(opt, width, height, vae_tile_width, vae_tile_height, input_enabled, source_canvas, paint_mask_pixels, source_latent, source_x, paint_mask_x, known_mask_x) != 0)
        return -1;

    const int patch_size = 2;
    const int num_patches_w = latents[0].w / patch_size;
    const int num_patches_h = latents[0].h / patch_size;

    fprintf(stderr, "num_patches = %d x %d\n", num_patches_w, num_patches_h);

    ncnn::Mat x_cos;
    ncnn::Mat x_sin;
    ncnn::Mat cap_cos;
    ncnn::Mat cap_sin;
    ncnn::Mat unified_cos;
    ncnn::Mat unified_sin;
    ZImage::generate_x_freqs(num_patches_w, num_patches_h, cap.h, x_cos, x_sin);
    ZImage::generate_cap_freqs(cap.h, cap_cos, cap_sin);
    ZImage::concat_along_h(x_cos, cap_cos, unified_cos);
    ZImage::concat_along_h(x_sin, cap_sin, unified_sin);

    ncnn::Mat neg_x_cos;
    ncnn::Mat neg_x_sin;
    ncnn::Mat neg_cap_cos;
    ncnn::Mat neg_cap_sin;
    ncnn::Mat neg_unified_cos;
    ncnn::Mat neg_unified_sin;
    if (apply_cfg)
    {
        ZImage::generate_x_freqs(num_patches_w, num_patches_h, neg_cap.h, neg_x_cos, neg_x_sin);
        ZImage::generate_cap_freqs(neg_cap.h, neg_cap_cos, neg_cap_sin);
        ZImage::concat_along_h(neg_x_cos, neg_cap_cos, neg_unified_cos);
        ZImage::concat_along_h(neg_x_sin, neg_cap_sin, neg_unified_sin);
    }

    // cap_embedder
    ncnn::Mat cap_embed;
    ncnn::Mat neg_cap_embed;
    {
        ZImage::CapEmbedder cap_embedder;

        cap_embedder.load(model, opt);

        cap_embedder.process(cap, cap_embed);

        if (apply_cfg)
        {
            cap_embedder.process(neg_cap, neg_cap_embed);
        }
    }

    // context_refiner
    ncnn::Mat cap_refine;
    ncnn::Mat neg_cap_refine;
    {
        ZImage::ContextRefiner context_refiner;

        context_refiner.load(model, opt);

        context_refiner.process(cap_embed, cap_cos, cap_sin, cap_refine);

        if (apply_cfg)
        {
            context_refiner.process(neg_cap_embed, neg_cap_cos, neg_cap_sin, neg_cap_refine);
        }
    }

    // prepare timesteps
    std::vector<float> sigmas;
    std::vector<float> timesteps;
    ZImage::prepare_timestamps(steps, scheduler_shift, sigmas, timesteps);

    // t_embedder
    ncnn::Mat t_embeds;
    {
        ZImage::TEmbedder t_embedder;

        t_embedder.load(model, opt);

        t_embedder.process(timesteps, t_embeds);
    }

    if (sample(latents, noise_latents, input_enabled, source_x, paint_mask_x, known_mask_x, control_x, opt, sigmas, t_embeds,
           x_cos, x_sin, neg_x_cos, neg_x_sin, cap_refine, unified_cos, unified_sin,
           neg_cap_refine, neg_unified_cos, neg_unified_sin,
           apply_cfg, guidance_scale, control_enabled, control_scale, steps) != 0)
        return -1;

    // vae decode and save image
    {
        const bool use_vae_tiled = vae_tile_width < width || vae_tile_height < height;

        ZImage::VAEDecoder vae_decoder;

        vae_decoder.load(model, use_vae_tiled, opt);

        for (int b = 0; b < batch; b++)
        {
            // vae decode
            ncnn::Mat outimage;
            {
                const float vae_scaling_factor = 0.3611f;
                const float vae_shift_factor = 0.1159f;

                for (int i = 0; i < latents[b].total(); i++)
                {
                    latents[b][i] = latents[b][i] / vae_scaling_factor + vae_shift_factor;
                }

                if (use_vae_tiled)
                {
                    vae_decoder.process_tiled(latents[b], vae_tile_width, vae_tile_height, outimage);
                }
                else
                {
                    vae_decoder.process(latents[b], outimage);
                }
            }

            if (input_enabled && preserve_known)
            {
                composite_known_pixels(outimage, source_canvas, paint_mask_pixels);
            }

            if (batch > 1)
            {
                fprintf(stderr, "vae of image %d/%d done\n", b + 1, batch);
            }
            else
            {
                fprintf(stderr, "vae done\n");
            }

            int success = ImageIO::save_image(outpath, outimage, b, batch);
            if (!success)
            {
#if _WIN32
                fwprintf(stderr, L"encode image %ls failed\n", outpath.c_str());
#else
                fprintf(stderr, "encode image %s failed\n", outpath.c_str());
#endif
            }
        }
    }

    return 0;
}
