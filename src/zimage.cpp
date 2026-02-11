// zimage implemented with ncnn library

#include "zimage.h"

#if _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#include <random>

// ncnn
#include "layer.h"
#include "layer_type.h"
#include "paramdict.h"
#include "modelbin.h"

namespace ZImage {

void generate_latent(int width, int height, int seed, ncnn::Mat& latent)
{
    const int latents_width = width / 8;
    const int latents_height = height / 8;

    latent.create(latents_width, latents_height, 16);

    std::mt19937 gen(seed);

    float mean = 0.f;
    float stddev = 1.f;
    std::normal_distribution<float> dist(mean, stddev);

    for (int i = 0; i < latent.total(); i++)
    {
        latent[i] = dist(gen);
    }
}

void rope_embedder(const ncnn::Mat& ids, ncnn::Mat& out_cos, ncnn::Mat& out_sin)
{
    // 确保预计算已完成
    const float theta = 256.0f;
    const int axes_dims[3] = {32, 48, 48};
    const int axes_lens[3] = {1024, 512, 512};

    ncnn::Mat freqs_cos[3];
    ncnn::Mat freqs_sin[3];
    {
        for (int i = 0; i < 3; i++)
        {
            int d = axes_dims[i];
            int end = axes_lens[i];
            int half_dim = d / 2;

            // 创建缓存 Mat, w=特征维(d/2), h=时间步(end)
            ncnn::Mat cos_mat(half_dim, end);
            ncnn::Mat sin_mat(half_dim, end);

            for (int t = 0; t < end; t++)
            {
                float* cos_ptr = cos_mat.row(t);
                float* sin_ptr = sin_mat.row(t);

                for (int j = 0; j < half_dim; j++)
                {
                    // Python: freqs = 1.0 / (theta ** (torch.arange(0, d, 2) / d))
                    // 指数部分: (2 * j) / d
                    float exponent = (float)(2 * j) / (float)d;
                    float freq = 1.0f / std::pow(theta, exponent);

                    // Python: angles = torch.outer(timestep, freqs)
                    float angle = (float)t * freq;

                    cos_ptr[j] = std::cos(angle);
                    sin_ptr[j] = std::sin(angle);
                }
            }

            freqs_cos[i] = cos_mat;
            freqs_sin[i] = sin_mat;
        }
    }

    int seqlen = ids.h; // ncnn Mat 的 h 在这里对应 seqlen

    int total_dim = axes_dims[0] + axes_dims[1] + axes_dims[2];
    int total_half_dim = total_dim / 2;

    // 分配输出 Mat
    out_cos.create(total_half_dim, seqlen);
    out_sin.create(total_half_dim, seqlen);

    // 遍历每一个 sequence step
    for (int i = 0; i < seqlen; i++)
    {
        // 获取当前步的索引 tuple [t, h, w]
        const int* id_ptr = ids.row<const int>(i);

        // 获取输出 Mat 当前行的指针
        float* out_c_ptr = out_cos.row(i);
        float* out_s_ptr = out_sin.row(i);

        int current_offset = 0;

        // 遍历 3 个轴 (axes)
        for (int axis = 0; axis < 3; axis++)
        {
            // 获取对应轴的索引值
            int idx = (int)id_ptr[axis];

            // 简单的边界保护
            if (idx < 0) idx = 0;
            if (idx >= axes_lens[axis]) idx = axes_lens[axis] - 1;

            // 获取该轴的预计算表
            const ncnn::Mat& cache_c = freqs_cos[axis];
            const ncnn::Mat& cache_s = freqs_sin[axis];

            // 该轴的特征长度
            int half_dim = axes_dims[axis] / 2;

            // 从缓存中复制数据到输出
            const float* src_c = cache_c.row(idx);
            const float* src_s = cache_s.row(idx);

            memcpy(out_c_ptr + current_offset, src_c, half_dim * sizeof(float));
            memcpy(out_s_ptr + current_offset, src_s, half_dim * sizeof(float));

            // 更新偏移量，准备拼接下一个轴的数据
            current_offset += half_dim;
        }
    }
}

void generate_x_freqs(int num_patches_w, int num_patches_h, int cap_len, ncnn::Mat& x_cos, ncnn::Mat& x_sin)
{
    const int num_patches = num_patches_w * num_patches_h;
    const int start_t = cap_len + 1;

    ncnn::Mat x_pos_ids(3, num_patches);
    for (int py = 0; py < num_patches_h; py++)
    {
        for (int px = 0; px < num_patches_w; px++)
        {
            int* p = x_pos_ids.row<int>(py * num_patches_w + px);
            p[0] = start_t;
            p[1] = py;
            p[2] = px;
        }
    }

    rope_embedder(x_pos_ids, x_cos, x_sin);
}

void generate_cap_freqs(int cap_len, ncnn::Mat& cap_cos, ncnn::Mat& cap_sin)
{
    ncnn::Mat cap_pos_ids(3, cap_len);
    for (int i = 0; i < cap_len; i++)
    {
        int* p = cap_pos_ids.row<int>(i);
        p[0] = 1 + i;
        p[1] = 0;
        p[2] = 0;
    }

    rope_embedder(cap_pos_ids, cap_cos, cap_sin);
}

void concat_along_h(const ncnn::Mat& a, const ncnn::Mat& b, ncnn::Mat& out)
{
    const int w = a.w;
    out.create(w, a.h + b.h);
    memcpy(out, a, w * a.h * sizeof(float));
    memcpy(out.row(a.h), b, w * b.h * sizeof(float));
}

void prepare_timestamps(int steps, float shift, std::vector<float>& sigmas, std::vector<float>& timesteps)
{
    sigmas.resize(steps + 1);
    timesteps.resize(steps);
    for (int i = 0; i < steps; i++)
    {
        float sigma = 1.f - i * (1.f / (steps - 1));

        sigma = shift * sigma / (1 + (shift - 1) * sigma);

        sigmas[i] = sigma;
        timesteps[i] = 1.f - sigma;
    }

    sigmas[steps] = 0.f;
}

void patchify(const ncnn::Mat& latent, ncnn::Mat& x)
{
    const int patch_size = 2;

    const int num_patches_w = latent.w / patch_size;
    const int num_patches_h = latent.h / patch_size;
    const int num_patches = num_patches_w * num_patches_h;

    const int channels = latent.c;

    x.create(patch_size * patch_size * channels, num_patches);

    for (int py = 0; py < num_patches_h; py++)
    {
        for (int px = 0; px < num_patches_w; px++)
        {
            float* outptr = x.row(py * num_patches_w + px);

            for (int q = 0; q < channels; q++)
            {
                *outptr++ = latent.channel(q).row(py * 2)[px * 2];
            }
            for (int q = 0; q < channels; q++)
            {
                *outptr++ = latent.channel(q).row(py * 2)[px * 2 + 1];
            }
            for (int q = 0; q < channels; q++)
            {
                *outptr++ = latent.channel(q).row(py * 2 + 1)[px * 2];
            }
            for (int q = 0; q < channels; q++)
            {
                *outptr++ = latent.channel(q).row(py * 2 + 1)[px * 2 + 1];
            }
        }
    }
}

void unpatchify(const ncnn::Mat& x, ncnn::Mat& latent)
{
    const int patch_size = 2;

    const int num_patches_w = latent.w / patch_size;
    const int num_patches_h = latent.h / patch_size;
    const int num_patches = num_patches_w * num_patches_h;

    const int channels = latent.c;

    for (int py = 0; py < num_patches_h; py++)
    {
        for (int px = 0; px < num_patches_w; px++)
        {
            const float* ptr = x.row(py * num_patches_w + px);

            for (int q = 0; q < channels; q++)
            {
                latent.channel(q).row(py * 2)[px * 2] = *ptr++;
            }
            for (int q = 0; q < channels; q++)
            {
                latent.channel(q).row(py * 2)[px * 2 + 1] = *ptr++;
            }
            for (int q = 0; q < channels; q++)
            {
                latent.channel(q).row(py * 2 + 1)[px * 2] = *ptr++;
            }
            for (int q = 0; q < channels; q++)
            {
                latent.channel(q).row(py * 2 + 1)[px * 2 + 1] = *ptr++;
            }
        }
    }
}

// Compute optimal tile size for splitting an image of (width x height) into tiles.
//
// Constraints:
//   - width and height are always multiples of 16.
//   - tile_size_w and tile_size_h must be multiples of 16.
//   - tile_size_w * tile_size_h <= max_tile_area_size.
//   - The tile aspect ratio should be as close as possible to width:height.
//   - The tiling should evenly divide the image to avoid very small leftover tiles.
static int align_up_16(int val)
{
    return ((val + 15) / 16) * 16;
}

void get_optimal_tile_size(int width, int height, int max_tile_area, int* tile_width, int* tile_height)
{
    // If the whole image fits in one tile, just return the original size.
    if ((long long)width * height <= max_tile_area)
    {
        *tile_width = width;
        *tile_height = height;
        return;
    }

    double input_ratio = (double)width / (double)height;
    double best_score  = -1.0;
    int    best_tw     = 16;
    int    best_th     = 16;

    // Enumerate the number of horizontal splits (nx) and vertical splits (ny).
    // For each (nx, ny) pair, compute the candidate tile size, then score it.
    //
    // tile_w = ceil(width  / nx)  aligned up to 16
    // tile_h = ceil(height / ny)  aligned up to 16
    //
    // Maximum possible nx = width/16, ny = height/16 (tile at least 16x16).
    int max_nx = width  / 16;
    int max_ny = height / 16;

    for (int nx = 1; nx <= max_nx; nx++)
    {
        // Candidate tile width: distribute width into nx pieces, align up to 16.
        int tw = align_up_16((width + nx - 1) / nx);
        if (tw > width)
            tw = width;
        if (tw < 16)
            tw = 16;

        for (int ny = 1; ny <= max_ny; ny++)
        {
            int th = align_up_16((height + ny - 1) / ny);
            if (th > height)
                th = height;
            if (th < 16)
                th = 16;

            // Area constraint.
            if ((long long)tw * th > max_tile_area)
                continue;

            // --- Scoring ---
            //
            // 1. Aspect-ratio similarity (higher is better, max = 1.0):
            //    ratio_score = min(r1/r2, r2/r1)  where r1 = tw/th, r2 = width/height
            double tile_ratio  = (double)tw / (double)th;
            double ratio_score = (tile_ratio < input_ratio)
                                     ? tile_ratio / input_ratio
                                     : input_ratio / tile_ratio;

            // 2. Utilization: how well does the tiling cover the image without
            //    creating tiny leftover strips?
            //
            //    actual_nx = ceil(width / tw),  actual_ny = ceil(height / th)
            //    last_w = width  - (actual_nx - 1) * tw
            //    last_h = height - (actual_ny - 1) * th
            //
            //    We want last_w / tw and last_h / th to be as close to 1 as
            //    possible (perfect division), or at least not too small.
            //    util = (last_w / tw) * (last_h / th)   range (0, 1]
            int actual_nx = (width  + tw - 1) / tw;
            int actual_ny = (height + th - 1) / th;
            int last_w = width  - (actual_nx - 1) * tw;
            int last_h = height - (actual_ny - 1) * th;
            if (last_w <= 0)
                last_w = tw;
            if (last_h <= 0)
                last_h = th;
            double util_w = (double)last_w / (double)tw;
            double util_h = (double)last_h / (double)th;
            double util   = util_w * util_h;

            // 3. Tile area efficiency: prefer larger tiles (fewer tiles total)
            //    as long as they satisfy the constraints.
            //    area_score = (tw * th) / max_tile_area   range (0, 1]
            double area_score = (double)((long long)tw * th) / (double)max_tile_area;

            // Combined score (weights chosen empirically):
            //   - utilization is the most important (avoid tiny scraps)
            //   - ratio preservation is second
            //   - area efficiency is third (prefer fewer, bigger tiles)
            double score = 0.45 * util + 0.35 * ratio_score + 0.20 * area_score;

            if (score > best_score)
            {
                best_score = score;
                best_tw = tw;
                best_th = th;
            }
        }
    }

    *tile_width = best_tw;
    *tile_height = best_th;
}

Tokenizer::Tokenizer() : bpe(BpeTokenizer::LoadFromFiles("vocab.txt", "merges.txt", SpecialTokensConfig{}, false, true, true))
{
    bpe.AddAdditionalSpecialToken("<|endoftext|>");
    bpe.AddAdditionalSpecialToken("<|im_start|>");
    bpe.AddAdditionalSpecialToken("<|im_end|>");
}

#if _WIN32
static std::string wstring_to_utf8_string(const std::wstring& wstr)
{
    int size_needed = WideCharToMultiByte(CP_UTF8, 0, wstr.data(), (int)wstr.size(), NULL, 0, NULL, NULL);
    std::string result(size_needed, 0);
    WideCharToMultiByte(CP_UTF8, 0, wstr.data(), (int)wstr.size(), &result[0], size_needed, NULL, NULL);
    return result;
}

static std::wstring utf8_string_to_wstring(const std::string& str)
{
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, str.data(), (int)str.size(), NULL, 0);
    std::wstring result(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, str.data(), (int)str.size(), &result[0], size_needed);
    return result;
}
#endif

int Tokenizer::encode(const path_t& prompt, std::vector<int>& ids) const
{
#if _WIN32
    std::string prompt_utf8 = wstring_to_utf8_string(prompt);
#else
    std::string prompt_utf8 = prompt;
#endif

    std::string message = std::string("<|im_start|>user\n") + prompt_utf8 + std::string("<|im_end|>\n<|im_start|>assistant\n");

    ids = bpe.encode(message, false, false);

    return 0;
}

int TextEncoder::load(const ncnn::Option& opt)
{
    text_encoder.opt = opt;
    text_encoder.load_param("z_image_turbo_text_encoder.ncnn.param");
    text_encoder.load_model("z_image_turbo_text_encoder.ncnn.bin");

    return 0;
}

static void generate_rope_embed_cache(int seqlen, int embed_dim, int position_id, ncnn::Mat& cos_cache, ncnn::Mat& sin_cache)
{
    const float rope_theta = 1000000;
    const float attention_factor = 1.f;

    // prepare inv_freq
    std::vector<float> inv_freq(embed_dim / 2);
    for (int i = 0; i < embed_dim / 2; i++)
    {
        inv_freq[i] = 1.f / powf(rope_theta, (float)(i * 2) / embed_dim);
    }

    cos_cache.create(embed_dim / 2, seqlen);
    sin_cache.create(embed_dim / 2, seqlen);

    for (int i = 0; i < seqlen; i++)
    {
        float* cos_ptr = cos_cache.row(i);
        float* sin_ptr = sin_cache.row(i);

        for (int j = 0; j < embed_dim / 2; j++)
        {
            const int pos = position_id + i;
            const float t = pos * inv_freq[j];
            const float cos_val = cosf(t);
            const float sin_val = sinf(t);
            *cos_ptr++ = cos_val;
            *sin_ptr++ = sin_val;
        }
    }
}

int TextEncoder::process(const std::vector<int>& input_ids, ncnn::Mat& cap)
{
    const int input_id_count = (int)input_ids.size();

    ncnn::Mat input_ids_mat(input_id_count);
    memcpy(input_ids_mat, input_ids.data(), input_ids.size() * sizeof(int));

    ncnn::Mat attention_mask(input_id_count, input_id_count);
    attention_mask.fill(-1e38f);
    for (int i = 0; i < input_ids.size(); i++)
    {
        float* row = attention_mask.row(i);
        for (int j = 0; j < i + 1; j++)
        {
            row[j] = 0.f;
        }
    }

    ncnn::Mat cos_cache;
    ncnn::Mat sin_cache;
    generate_rope_embed_cache(input_id_count, 128, 0, cos_cache, sin_cache);

    ncnn::Extractor ex = text_encoder.create_extractor();

    ex.input("in0", input_ids_mat);
    ex.input("in1", attention_mask);
    ex.input("in2", cos_cache);
    ex.input("in3", sin_cache);

    ex.extract("out0", cap);

    return 0;
}

int CapEmbedder::load(const ncnn::Option& opt)
{
    cap_embedder.opt = opt;
    cap_embedder.load_param("z_image_turbo_transformer_cap_embedder.ncnn.param");
    cap_embedder.load_model("z_image_turbo_transformer_cap_embedder.ncnn.bin");

    return 0;
}

int CapEmbedder::process(const ncnn::Mat& cap, ncnn::Mat& cap_embed)
{
    ncnn::Extractor ex = cap_embedder.create_extractor();

    ex.input("in0", cap);

    ex.extract("out0", cap_embed);

    return 0;
}

int ContextRefiner::load(const ncnn::Option& opt)
{
    context_refiner.opt = opt;
    context_refiner.load_param("z_image_turbo_transformer_context_refiner.ncnn.param");
    context_refiner.load_model("z_image_turbo_transformer_context_refiner.ncnn.bin");

    return 0;
}

int ContextRefiner::process(const ncnn::Mat& cap_embed, const ncnn::Mat& cap_cos, const ncnn::Mat& cap_sin, ncnn::Mat& cap_refine)
{
    ncnn::Extractor ex = context_refiner.create_extractor();

    ex.input("in0", cap_embed);
    ex.input("in1", cap_cos);
    ex.input("in2", cap_sin);

    ex.extract("out0", cap_refine);

    return 0;
}

int TEmbedder::load(const ncnn::Option& opt)
{
    t_embedder.opt = opt;
    t_embedder.load_param("z_image_turbo_transformer_t_embedder.ncnn.param");
    t_embedder.load_model("z_image_turbo_transformer_t_embedder.ncnn.bin");

    return 0;
}

int TEmbedder::process(const std::vector<float>& timesteps, ncnn::Mat& t_embeds)
{
    const float t_scale = 1000.f;

    const int steps = (int)timesteps.size();

    ncnn::Mat t_mat(1, steps);
    for (int z = 0; z < steps; z++)
    {
        t_mat.row(z)[0] = timesteps[z] * t_scale;
    }

    ncnn::Extractor ex = t_embedder.create_extractor();

    ex.input("in0", t_mat);

    ex.extract("out0", t_embeds);

    return 0;
}

int AllXEmbedder::load(const ncnn::Option& opt)
{
    all_x_embedder.opt = opt;
    all_x_embedder.load_param("z_image_turbo_transformer_all_x_embedder.ncnn.param");
    all_x_embedder.load_model("z_image_turbo_transformer_all_x_embedder.ncnn.bin");

    return 0;
}

int AllXEmbedder::process(const ncnn::Mat& x, ncnn::Mat& x_embed)
{
    ncnn::Extractor ex = all_x_embedder.create_extractor();

    ex.input("in0", x);

    ex.extract("out0", x_embed);

    return 0;
}

int NoiseRefiner::load(const ncnn::Option& opt)
{
    noise_refiner.opt = opt;
    noise_refiner.load_param("z_image_turbo_transformer_noise_refiner.ncnn.param");
    noise_refiner.load_model("z_image_turbo_transformer_noise_refiner.ncnn.bin");

    return 0;
}

int NoiseRefiner::process(const ncnn::Mat& x_embed, const ncnn::Mat& x_cos, const ncnn::Mat& x_sin, const ncnn::Mat& t_embed, ncnn::Mat& x_embed_refine)
{
    ncnn::Extractor ex = noise_refiner.create_extractor();

    ex.input("in0", x_embed);
    ex.input("in1", x_cos);
    ex.input("in2", x_sin);
    ex.input("in3", t_embed);

    ex.extract("out0", x_embed_refine);

    return 0;
}

int UnifiedRefiner::load(const ncnn::Option& opt)
{
    unified_refiner.opt = opt;
    unified_refiner.load_param("z_image_turbo_transformer_unified.ncnn.param");
    unified_refiner.load_model("z_image_turbo_transformer_unified.ncnn.bin");

    return 0;
}

int UnifiedRefiner::process(const ncnn::Mat& unified_embed, const ncnn::Mat& unified_cos, const ncnn::Mat& unified_sin, const ncnn::Mat& t_embed, ncnn::Mat& unified)
{
    ncnn::Extractor ex = unified_refiner.create_extractor();

    ex.input("in0", unified_embed);
    ex.input("in1", unified_cos);
    ex.input("in2", unified_sin);
    ex.input("in3", t_embed);

    ex.extract("out0", unified);

    return 0;
}

int AllFinalLayer::load(const ncnn::Option& opt)
{
    all_final_layer.opt = opt;
    all_final_layer.load_param("z_image_turbo_transformer_all_final_layer.ncnn.param");
    all_final_layer.load_model("z_image_turbo_transformer_all_final_layer.ncnn.bin");

    return 0;
}

int AllFinalLayer::process(const ncnn::Mat& unified, const ncnn::Mat& t_embed, ncnn::Mat& unified_final)
{
    ncnn::Extractor ex = all_final_layer.create_extractor();

    ex.input("in0", unified);
    ex.input("in1", t_embed);

    ex.extract("out0", unified_final);

    return 0;
}

// 0 = inference
// 1 = inference and collect mean and var
// 2 = inference with collected mean and var
static thread_local int g_vae_tiled_groupnorm_state = 0;
static thread_local std::vector< std::vector<float> > g_means;
static thread_local std::vector< std::vector<float> > g_vars;
static thread_local int g_groupnorm_count = 0;

class VAETiledGroupNorm : public ncnn::Layer
{
public:
    VAETiledGroupNorm();

    virtual int load_param(const ncnn::ParamDict& pd);
    virtual int load_model(const ncnn::ModelBin& mb);
    virtual int forward_inplace(ncnn::Mat& bottom_top_blob, const ncnn::Option& opt) const;

public:
    // param
    int group;
    int channels;
    float eps;
    int affine;

    // model
    ncnn::Mat gamma_data;
    ncnn::Mat beta_data;

    int g_meanvar_index;
};

DEFINE_LAYER_CREATOR(VAETiledGroupNorm)

VAETiledGroupNorm::VAETiledGroupNorm()
{
    one_blob_only = true;
    support_inplace = true;

    g_meanvar_index = g_groupnorm_count;
    g_groupnorm_count++;
}

int VAETiledGroupNorm::load_param(const ncnn::ParamDict& pd)
{
    group = pd.get(0, 1);
    channels = pd.get(1, 0);
    eps = pd.get(2, 0.001f);
    affine = pd.get(3, 1);

    return 0;
}

int VAETiledGroupNorm::load_model(const ncnn::ModelBin& mb)
{
    if (affine == 0)
        return 0;

    gamma_data = mb.load(channels, 1);
    if (gamma_data.empty())
        return -100;

    beta_data = mb.load(channels, 1);
    if (beta_data.empty())
        return -100;

    return 0;
}

static void groupnorm(float* ptr, float& mean, float& var, const float* gamma_ptr, const float* beta_ptr, float eps, int channels, int size, size_t cstep)
{
    if (g_vae_tiled_groupnorm_state == 0 || g_vae_tiled_groupnorm_state == 1)
    {
        float sum = 0.f;
        for (int q = 0; q < channels; q++)
        {
            const float* ptr0 = ptr + cstep * q;
            for (int i = 0; i < size; i++)
            {
                sum += ptr0[i];
            }
        }

        mean = sum / (channels * size);

        float sqsum = 0.f;
        for (int q = 0; q < channels; q++)
        {
            const float* ptr0 = ptr + cstep * q;
            for (int i = 0; i < size; i++)
            {
                float v = ptr0[i] - mean;
                sqsum += v * v;
            }
        }

        var = sqsum / (channels * size);
    }

    float a = 1.f / sqrtf(var + eps);
    float b = -mean * a;

    if (gamma_ptr && beta_ptr)
    {
        for (int q = 0; q < channels; q++)
        {
            float* ptr0 = ptr + cstep * q;
            const float gamma = gamma_ptr[q];
            const float beta = beta_ptr[q];
            for (int i = 0; i < size; i++)
            {
                ptr0[i] = (ptr0[i] * a + b) * gamma + beta;
            }
        }
    }
    else
    {
        for (int q = 0; q < channels; q++)
        {
            float* ptr0 = ptr + cstep * q;
            for (int i = 0; i < size; i++)
            {
                ptr0[i] = ptr0[i] * a + b;
            }
        }
    }
}

int VAETiledGroupNorm::forward_inplace(ncnn::Mat& bottom_top_blob, const ncnn::Option& opt) const
{
    const int dims = bottom_top_blob.dims;
    const int channels_g = channels / group;

    std::vector<float> means;
    std::vector<float> vars;
    if (g_vae_tiled_groupnorm_state == 1)
    {
        means.resize(group);
        vars.resize(group);
    }
    if (g_vae_tiled_groupnorm_state == 2)
    {
        means = g_means[g_meanvar_index];
        vars = g_vars[g_meanvar_index];
    }

    if (dims == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int g = 0; g < group; g++)
        {
            float mean;
            float var;
            if (g_vae_tiled_groupnorm_state == 2)
            {
                mean = means[g];
                var = vars[g];
            }

            ncnn::Mat bottom_top_blob_g = bottom_top_blob.range(g * channels_g, channels_g);
            const float* gamma_ptr = affine ? (const float*)gamma_data + g * channels_g : 0;
            const float* beta_ptr = affine ? (const float*)beta_data + g * channels_g : 0;
            groupnorm(bottom_top_blob_g, mean, var, gamma_ptr, beta_ptr, eps, channels_g, 1, 1);

            if (g_vae_tiled_groupnorm_state == 1)
            {
                means[g] = mean;
                vars[g] = var;
            }
        }
    }

    if (dims == 2)
    {
        const int w = bottom_top_blob.w;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int g = 0; g < group; g++)
        {
            float mean;
            float var;
            if (g_vae_tiled_groupnorm_state == 2)
            {
                mean = means[g];
                var = vars[g];
            }

            ncnn::Mat bottom_top_blob_g = bottom_top_blob.row_range(g * channels_g, channels_g);
            const float* gamma_ptr = affine ? (const float*)gamma_data + g * channels_g : 0;
            const float* beta_ptr = affine ? (const float*)beta_data + g * channels_g : 0;
            groupnorm(bottom_top_blob_g, mean, var, gamma_ptr, beta_ptr, eps, channels_g, w, w);

            if (g_vae_tiled_groupnorm_state == 1)
            {
                means[g] = mean;
                vars[g] = var;
            }
        }
    }

    if (dims == 3 || dims == 4)
    {
        const int size = bottom_top_blob.w * bottom_top_blob.h * bottom_top_blob.d;
        const size_t cstep = bottom_top_blob.cstep;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int g = 0; g < group; g++)
        {
            float mean;
            float var;
            if (g_vae_tiled_groupnorm_state == 2)
            {
                mean = means[g];
                var = vars[g];
            }

            ncnn::Mat bottom_top_blob_g = bottom_top_blob.channel_range(g * channels_g, channels_g);
            const float* gamma_ptr = affine ? (const float*)gamma_data + g * channels_g : 0;
            const float* beta_ptr = affine ? (const float*)beta_data + g * channels_g : 0;
            groupnorm(bottom_top_blob_g, mean, var, gamma_ptr, beta_ptr, eps, channels_g, size, cstep);

            if (g_vae_tiled_groupnorm_state == 1)
            {
                means[g] = mean;
                vars[g] = var;
            }
        }
    }

    if (g_vae_tiled_groupnorm_state == 1)
    {
        g_means[g_meanvar_index] = means;
        g_vars[g_meanvar_index] = vars;
    }

    return 0;
}

int VAE::load(const ncnn::Option& opt)
{
    vae.opt = opt;
    vae.register_custom_layer("GroupNorm", VAETiledGroupNorm_layer_creator);
    vae.load_param("z_image_turbo_vae.ncnn.param");
    vae.load_model("z_image_turbo_vae.ncnn.bin");

    g_means.resize(g_groupnorm_count);
    g_vars.resize(g_groupnorm_count);

    return 0;
}

int VAE::process(const ncnn::Mat& latent, ncnn::Mat& outimage)
{
    g_vae_tiled_groupnorm_state = 0;

    ncnn::Extractor ex = vae.create_extractor();

    ex.input("in0", latent);

    ncnn::Mat vae_out;
    ex.extract("out0", vae_out);

    // -1 ~ 1 to 0 ~ 255
    const float mean_vals[3] = {-1.f, -1.f, -1.f};
    const float norm_vals[3] = {127.5f, 127.5f, 127.5f};
    vae_out.substract_mean_normalize(mean_vals, norm_vals);

    outimage.create(vae_out.w, vae_out.h, (size_t)3u, 3);

#if _WIN32
    vae_out.to_pixels((unsigned char*)outimage.data, ncnn::Mat::PIXEL_RGB2BGR);
#else
    vae_out.to_pixels((unsigned char*)outimage.data, ncnn::Mat::PIXEL_RGB);
#endif

    return 0;
}

int VAE::process_tiled(const ncnn::Mat& latent, int tile_width, int tile_height, ncnn::Mat& outimage)
{
    const int latent_tile_width = tile_width / 8;
    const int latent_tile_height = tile_height / 8;

    if (latent_tile_width >= latent.w && latent_tile_height >= latent.h)
    {
        return process(latent, outimage);
    }

    ncnn::Mat latent_small;
    ncnn::resize_nearest(latent, latent_small, latent_tile_width, latent_tile_height);

    // estimate attention output
    ncnn::Mat attn_small;
    ncnn::Mat attn;
    {
        g_vae_tiled_groupnorm_state = 0;

        ncnn::Extractor ex = vae.create_extractor();

        ex.input("in0", latent_small);

        ex.extract("19", attn_small);

        ncnn::resize_bilinear(attn_small, attn, latent.w, latent.h);
    }

    // collect groupnorm mean and var
    {
        g_vae_tiled_groupnorm_state = 1;

        ncnn::Extractor ex = vae.create_extractor();

        ex.input("in0", latent_small);

        ex.input("19", attn_small);

        ncnn::Mat stub;
        ex.extract("out0", stub);
    }

    // tiled vae with pad 4
    {
        g_vae_tiled_groupnorm_state = 2;

        const int TILE_PAD = 4;

        const int width = latent.w * 8;
        const int height = latent.h * 8;

        outimage.create(width, height, (size_t)3u, 3);

        const int TILES_H = (latent.h + latent_tile_height - 1) / latent_tile_height;
        const int TILES_W = (latent.w + latent_tile_width - 1) / latent_tile_width;

        ncnn::Option opt;
        opt.num_threads = 1;

        for (int ty = 0; ty < TILES_H; ty++)
        {
            for (int tx = 0; tx < TILES_W; tx++)
            {
                // crop latent and attn tile
                ncnn::Mat latent_tile;
                ncnn::Mat attn_tile;
                {
                    int starty = std::max(0, ty * latent_tile_height - TILE_PAD);
                    int endy = std::min(latent.h, (ty + 1) * latent_tile_height + TILE_PAD);
                    int startx = std::max(0, tx * latent_tile_width - TILE_PAD);
                    int endx = std::min(latent.w, (tx + 1) * latent_tile_width + TILE_PAD);

                    // NCNN_LOGE("tile %d %d    %d ~ %d  %d ~ %d", ty, tx, starty, endy, startx, endx);

                    ncnn::copy_cut_border(latent, latent_tile, starty, latent.h - endy, startx, latent.w - endx, opt);
                    ncnn::copy_cut_border(attn, attn_tile, starty, latent.h - endy, startx, latent.w - endx, opt);
                }

                ncnn::Mat vae_out_tile;
                {
                    ncnn::Extractor ex = vae.create_extractor();

                    ex.input("in0", latent_tile);

                    ex.input("19", attn_tile);

                    ex.extract("out0", vae_out_tile);
                }

                // NCNN_LOGE("vae_out_tile %d x %d", vae_out_tile.w, vae_out_tile.h);

                // crop to target roi
                {
                    int pad_top = ty == 0 ? 0 : TILE_PAD * 8;
                    int pad_bottom = ty == TILES_H - 1 ? 0 : TILE_PAD * 8;
                    int pad_left = tx == 0 ? 0 : TILE_PAD * 8;
                    int pad_right = tx == TILES_W - 1 ? 0 : TILE_PAD * 8;

                    ncnn::Mat vae_out_tile_roi;
                    ncnn::copy_cut_border(vae_out_tile, vae_out_tile_roi, pad_top, pad_bottom, pad_left, pad_right, opt);
                    vae_out_tile = vae_out_tile_roi;
                }

                // -1 ~ 1 to 0 ~ 255
                const float mean_vals[3] = {-1.f, -1.f, -1.f};
                const float norm_vals[3] = {127.5f, 127.5f, 127.5f};
                vae_out_tile.substract_mean_normalize(mean_vals, norm_vals);

                // paste to out image roi
                {
                    int out_starty = std::max(0, ty * tile_height);
                    int out_endy = std::min(height, (ty + 1) * tile_height);
                    int out_startx = std::max(0, tx * tile_width);
                    int out_endx = std::min(width, (tx + 1) * tile_width);

                    // NCNN_LOGE("out tile %d %d    %d ~ %d  %d ~ %d", ty, tx, out_starty, out_endy, out_startx, out_endx);

                    unsigned char* data = (unsigned char*)outimage.data + (out_starty * width + out_startx) * 3;
#if _WIN32
                    vae_out_tile.to_pixels(data, ncnn::Mat::PIXEL_RGB2BGR, width * 3);
#else
                    vae_out_tile.to_pixels(data, ncnn::Mat::PIXEL_RGB, width * 3);
#endif
                }
            }
        }

    }

    return 0;
}

} // namespace ZImage
