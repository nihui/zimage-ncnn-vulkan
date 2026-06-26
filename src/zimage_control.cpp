// z-image implemented with ncnn library

#include "zimage.h"

#include <stdio.h>
#include <string.h>

namespace ZImage {

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

static int same_shape(const ncnn::Mat& a, const ncnn::Mat& b)
{
    return a.dims == b.dims && a.w == b.w && a.h == b.h && a.d == b.d && a.c == b.c && a.elempack == b.elempack;
}

static int add_scaled_inplace(ncnn::Mat& a, const ncnn::Mat& b, float scale)
{
    if (a.empty() || b.empty())
    {
        fprintf(stderr, "control hint is empty\n");
        return -1;
    }

    if (!same_shape(a, b) || a.total() != b.total())
    {
        fprintf(stderr, "control hint shape mismatch, base %d x %d x %d x %d pack%d total%zu, hint %d x %d x %d x %d pack%d total%zu\n",
                a.w, a.h, a.d, a.c, a.elempack, a.total(), b.w, b.h, b.d, b.c, b.elempack, b.total());
        return -1;
    }

    if (a.elemsize != (size_t)sizeof(float) * a.elempack || b.elemsize != (size_t)sizeof(float) * b.elempack)
    {
        fprintf(stderr, "control hint expects fp32 Mat but got elemsize %zu and %zu\n", a.elemsize, b.elemsize);
        return -1;
    }

    if (scale == 0.f)
        return 0;

    const int stride = a.w * a.elempack;

    if (a.dims == 1)
    {
        float* aptr = a;
        const float* bptr = b;
        const int total = a.w * a.elempack;
        for (int i = 0; i < total; i++)
            aptr[i] += bptr[i] * scale;
    }
    else if (a.dims == 2)
    {
        for (int y = 0; y < a.h; y++)
        {
            float* aptr = a.row(y);
            const float* bptr = b.row(y);
            for (int i = 0; i < stride; i++)
                aptr[i] += bptr[i] * scale;
        }
    }
    else if (a.dims == 3)
    {
        for (int q = 0; q < a.c; q++)
        {
            ncnn::Mat ach = a.channel(q);
            const ncnn::Mat bch = b.channel(q);
            for (int y = 0; y < a.h; y++)
            {
                float* aptr = ach.row(y);
                const float* bptr = bch.row(y);
                for (int i = 0; i < stride; i++)
                    aptr[i] += bptr[i] * scale;
            }
        }
    }
    else if (a.dims == 4)
    {
        for (int q = 0; q < a.c; q++)
        {
            ncnn::Mat ach = a.channel(q);
            const ncnn::Mat bch = b.channel(q);
            for (int z = 0; z < a.d; z++)
            {
                ncnn::Mat adepth = ach.depth(z);
                const ncnn::Mat bdepth = bch.depth(z);
                for (int y = 0; y < a.h; y++)
                {
                    float* aptr = adepth.row(y);
                    const float* bptr = bdepth.row(y);
                    for (int i = 0; i < stride; i++)
                        aptr[i] += bptr[i] * scale;
                }
            }
        }
    }
    else
    {
        fprintf(stderr, "unsupported control hint dims %d\n", a.dims);
        return -1;
    }

    return 0;
}

static int add_scaled(const ncnn::Mat& a, const ncnn::Mat& b, float scale, ncnn::Mat& out)
{
    out = a.clone();
    if (out.empty())
    {
        fprintf(stderr, "control base tensor is empty\n");
        return -1;
    }

    return add_scaled_inplace(out, b, scale);
}

static path_t get_control_model_dir(const path_t& model)
{
    if (model.find(PATHSTR("z-image-control")) != path_t::npos)
        return model;

    const path_t turbo_dirname = PATHSTR("z-image-turbo");
    const size_t turbo_pos = model.rfind(turbo_dirname);
    if (turbo_pos != path_t::npos)
    {
        path_t control_model = model;
        control_model.replace(turbo_pos, turbo_dirname.size(), PATHSTR("z-image-control"));
        return control_model;
    }

    return model;
}

int prepare_control_x(
    const ncnn::Mat& control_image,
    const path_t& model,
    bool use_vae_tiled,
    int vae_tile_width,
    int vae_tile_height,
    const ncnn::Option& opt,
    ncnn::Mat& control_x)
{
    if (control_image.empty())
    {
        fprintf(stderr, "control image is empty\n");
        return -1;
    }

    ncnn::Mat control_image_float;
    image_to_ncnn_rgb_float(control_image, control_image_float);

    ncnn::Mat control_latent;
    {
        VAEEncoder vae_encoder;
        if (vae_encoder.load(model, use_vae_tiled, opt) != 0)
            return -1;

        if (use_vae_tiled)
        {
            if (vae_encoder.process_tiled(control_image_float, vae_tile_width, vae_tile_height, control_latent) != 0)
                return -1;
        }
        else
        {
            if (vae_encoder.process(control_image_float, control_latent) != 0)
                return -1;
        }
    }

    if (control_latent.c != 16)
    {
        fprintf(stderr, "control vae latent channel mismatch, got %d expected 16\n", control_latent.c);
        return -1;
    }

    ncnn::Mat control_latent_33(control_latent.w, control_latent.h, 33);
    for (int q = 0; q < 16; q++)
    {
        const ncnn::Mat src = control_latent.channel(q);
        ncnn::Mat dst = control_latent_33.channel(q);
        for (int y = 0; y < control_latent.h; y++)
        {
            const float* srcptr = src.row(y);
            float* dstptr = dst.row(y);
            memcpy(dstptr, srcptr, (size_t)control_latent.w * sizeof(float));
        }
    }
    for (int q = 16; q < 33; q++)
    {
        ncnn::Mat dst = control_latent_33.channel(q);
        dst.fill(0.f);
    }

    // ZImage::patchify uses latent.c when computing token width, so the same
    // helper handles this 33-channel ControlNet latent without a separate path.
    patchify(control_latent_33, control_x);
    if (control_x.w != 132)
    {
        fprintf(stderr, "control patch dim mismatch, got %d expected 132\n", control_x.w);
        return -1;
    }

    return 0;
}

int ControlRefiner::load(const path_t& model, const ncnn::Option& opt)
{
    return load(model, path_t(), opt);
}

int ControlRefiner::load(const path_t& model, const path_t& control_model_dir, const ncnn::Option& opt)
{
    const path_t control_model = control_model_dir.empty() ? get_control_model_dir(model) : control_model_dir;
    path_t parampath = control_model + PATHSTR("/z_image_control_refiner.ncnn.param");
    path_t modelpath = control_model + PATHSTR("/z_image_control_refiner.ncnn.bin");
    parampath = sanitize_filepath(parampath);
    modelpath = sanitize_filepath(modelpath);

    control_refiner.opt = opt;
    if (control_refiner.load_param(parampath.c_str()) != 0)
        return -1;
    if (control_refiner.load_model(modelpath.c_str()) != 0)
        return -1;

    return 0;
}

int ControlRefiner::process(
    const ncnn::Mat& control_x,
    const ncnn::Mat& x_embed,
    const ncnn::Mat& x_cos,
    const ncnn::Mat& x_sin,
    const ncnn::Mat& t_embed,
    ncnn::Mat& hint0,
    ncnn::Mat& hint1,
    ncnn::Mat& control_context) const
{
    ncnn::Extractor ex = control_refiner.create_extractor();

    ex.input("in0", control_x);
    ex.input("in1", x_embed);
    ex.input("in2", x_cos);
    ex.input("in3", x_sin);
    ex.input("in4", t_embed);

    if (ex.extract("out0", hint0) != 0)
        return -1;
    if (ex.extract("out1", hint1) != 0)
        return -1;
    if (ex.extract("out2", control_context) != 0)
        return -1;

    return 0;
}

int ControlUnified::load(const path_t& model, const ncnn::Option& opt)
{
    return load(model, path_t(), opt);
}

int ControlUnified::load(const path_t& model, const path_t& control_model_dir, const ncnn::Option& opt)
{
    const path_t control_model = control_model_dir.empty() ? get_control_model_dir(model) : control_model_dir;
    path_t parampath = control_model + PATHSTR("/z_image_control_unified.ncnn.param");
    path_t modelpath = control_model + PATHSTR("/z_image_control_unified.ncnn.bin");
    parampath = sanitize_filepath(parampath);
    modelpath = sanitize_filepath(modelpath);

    control_unified.opt = opt;
    if (control_unified.load_param(parampath.c_str()) != 0)
        return -1;
    if (control_unified.load_model(modelpath.c_str()) != 0)
        return -1;

    return 0;
}

int ControlUnified::process(
    const ncnn::Mat& control_unified_embed,
    const ncnn::Mat& unified_embed,
    const ncnn::Mat& unified_cos,
    const ncnn::Mat& unified_sin,
    const ncnn::Mat& t_embed,
    ncnn::Mat& hint0,
    ncnn::Mat& hint1,
    ncnn::Mat& hint2) const
{
    std::vector<ncnn::Mat> hints;
    if (process(control_unified_embed, unified_embed, unified_cos, unified_sin, t_embed, hints) != 0)
        return -1;
    if (hints.size() != 3)
    {
        fprintf(stderr, "control unified expected 3 hints but got %d\n", (int)hints.size());
        return -1;
    }

    hint0 = hints[0];
    hint1 = hints[1];
    hint2 = hints[2];
    return 0;
}

int ControlUnified::process(
    const ncnn::Mat& control_unified_embed,
    const ncnn::Mat& unified_embed,
    const ncnn::Mat& unified_cos,
    const ncnn::Mat& unified_sin,
    const ncnn::Mat& t_embed,
    std::vector<ncnn::Mat>& hints) const
{
    ncnn::Extractor ex = control_unified.create_extractor();

    ex.input("in0", control_unified_embed);
    ex.input("in1", unified_embed);
    ex.input("in2", unified_cos);
    ex.input("in3", unified_sin);
    ex.input("in4", t_embed);

    const std::vector<const char*>& output_names = control_unified.output_names();
    hints.resize(output_names.size());
    for (size_t i = 0; i < output_names.size(); i++)
    {
        if (ex.extract(output_names[i], hints[i]) != 0)
            return -1;
    }

    return 0;
}

int NoiseRefiner::process_controlled(
    const ncnn::Mat& x_embed,
    const ncnn::Mat& x_cos,
    const ncnn::Mat& x_sin,
    const ncnn::Mat& t_embed,
    const ncnn::Mat& hint0,
    const ncnn::Mat& hint1,
    float control_scale,
    ncnn::Mat& x_embed_refine) const
{
    if (control_scale == 0.f)
        return process(x_embed, x_cos, x_sin, t_embed, x_embed_refine);

    static const char noise_refiner_hint_blob[] = "63";

    ncnn::Extractor ex = noise_refiner.create_extractor();

    ex.input("in0", x_embed);
    ex.input("in1", x_cos);
    ex.input("in2", x_sin);
    ex.input("in3", t_embed);

    ncnn::Mat h0;
    if (ex.extract(noise_refiner_hint_blob, h0) != 0)
        return -1;

    if (add_scaled(h0, hint0, control_scale, h0) != 0)
        return -1;
    if (ex.input(noise_refiner_hint_blob, h0) != 0)
        return -1;
#if NCNN_VULKAN
    if (ex.input(noise_refiner_hint_blob, ncnn::VkMat()) != 0)
        return -1;
#endif

    if (ex.extract("out0", x_embed_refine) != 0)
        return -1;

    if (add_scaled(x_embed_refine, hint1, control_scale, x_embed_refine) != 0)
        return -1;

    return 0;
}

int UnifiedRefiner::process_controlled(
    const ncnn::Mat& unified_embed,
    const ncnn::Mat& unified_cos,
    const ncnn::Mat& unified_sin,
    const ncnn::Mat& t_embed,
    const ncnn::Mat& hint0,
    const ncnn::Mat& hint10,
    const ncnn::Mat& hint20,
    float control_scale,
    ncnn::Mat& unified) const
{
    std::vector<ncnn::Mat> hints(3);
    hints[0] = hint0;
    hints[1] = hint10;
    hints[2] = hint20;
    return process_controlled(unified_embed, unified_cos, unified_sin, t_embed, hints, control_scale, unified);
}

int UnifiedRefiner::process_controlled(
    const ncnn::Mat& unified_embed,
    const ncnn::Mat& unified_cos,
    const ncnn::Mat& unified_sin,
    const ncnn::Mat& t_embed,
    const std::vector<ncnn::Mat>& hints,
    float control_scale,
    ncnn::Mat& unified) const
{
    if (control_scale == 0.f)
        return process(unified_embed, unified_cos, unified_sin, t_embed, unified);

    if (hints.empty())
    {
        fprintf(stderr, "control unified hints are empty\n");
        return -1;
    }

    static const char* hint_blob_ids_lite[] = {"203", "703", "1203"};
    static const char* hint_blob_ids_full[] = {"203", "303", "403", "503", "603", "703", "803", "903", "1003", "1103", "1203", "1303", "1403", "1503", "1603"};

    const char** hint_blob_ids = 0;
    if (hints.size() == sizeof(hint_blob_ids_lite) / sizeof(hint_blob_ids_lite[0]))
    {
        hint_blob_ids = hint_blob_ids_lite;
    }
    else if (hints.size() == sizeof(hint_blob_ids_full) / sizeof(hint_blob_ids_full[0]))
    {
        hint_blob_ids = hint_blob_ids_full;
    }
    else
    {
        fprintf(stderr, "unsupported control unified hint count %d\n", (int)hints.size());
        return -1;
    }

    ncnn::Extractor ex = unified_refiner.create_extractor();

    ex.input("in0", unified_embed);
    ex.input("in1", unified_cos);
    ex.input("in2", unified_sin);
    ex.input("in3", t_embed);

    for (size_t i = 0; i < hints.size(); i++)
    {
        const char* blob_name = hint_blob_ids[i];

        ncnn::Mat h;
        if (ex.extract(blob_name, h) != 0)
            return -1;
        if (add_scaled(h, hints[i], control_scale, h) != 0)
            return -1;
        if (ex.input(blob_name, h) != 0)
            return -1;
#if NCNN_VULKAN
        if (ex.input(blob_name, ncnn::VkMat()) != 0)
            return -1;
#endif
    }

    if (ex.extract("out0", unified) != 0)
        return -1;

    return 0;
}

} // namespace ZImage
