// z-image implemented with ncnn library

#include <stdio.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <stdint.h>

#include <random>

#if _WIN32
// image encoder with wic
#include "wic_image.h"
#else // _WIN32
// image encoder with libpng
#include "png_image.h"
#endif // _WIN32

#include "bpe_tokenizer.h"

#include "mat.h"
#include "net.h"

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

static void rope_embbedder(const ncnn::Mat& ids, ncnn::Mat& out_cos, ncnn::Mat& out_sin)
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

int main()
{
    // const char* prompt = "風的彷徨.";
    const char* prompt = "風的彷徨.美少女.半身照.";
    const int width = 1024;
    const int height = 1024;
    // const int width = 512;
    // const int height = 512;
    const int steps = 9;
    const int seed = 777;

    const int gpuid = ncnn::get_default_gpu_index();
    const bool use_gpu = true;
    const bool use_bf16 = true;

    // assert width % 16 == 0
    // assert height % 16 == 0
    // assert (width / 16) * (height / 16) >= 32

    // tokenizer
    std::vector<int> input_ids;
    {
        BpeTokenizer bpe = BpeTokenizer::LoadFromFiles("vocab.txt", "merges.txt", SpecialTokensConfig{}, false, true, true);

        bpe.AddAdditionalSpecialToken("<|endoftext|>");
        bpe.AddAdditionalSpecialToken("<|im_start|>");
        bpe.AddAdditionalSpecialToken("<|im_end|>");

        std::string message = std::string("<|im_start|>user\n") + prompt + std::string("<|im_end|>\n<|im_start|>assistant\n");

        input_ids = bpe.encode(message, false, false);
    }

    // text encoder
    ncnn::Mat cap;
    {
        ncnn::Net text_encoder;
        text_encoder.opt.vulkan_device_index = gpuid;
        text_encoder.opt.use_vulkan_compute = use_gpu;
        text_encoder.opt.use_fp16_packed = false;
        text_encoder.opt.use_fp16_storage = false;
        text_encoder.opt.use_fp16_arithmetic = false;
        text_encoder.opt.use_bf16_packed = use_bf16;
        text_encoder.opt.use_bf16_storage = use_bf16;
        text_encoder.load_param("z_image_turbo_text_encoder.ncnn.param");
        text_encoder.load_model("z_image_turbo_text_encoder.ncnn.bin");

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
    }

    // prepare latents
    ncnn::Mat latents;
    {
        const int latents_width = width / 8;
        const int latents_height = height / 8;

        latents.create(latents_width, latents_height, 16);

        std::mt19937 gen(seed);

        float mean = 0.f;
        float stddev = 1.f;
        std::normal_distribution<float> dist(mean, stddev);

        for (int i = 0; i < latents.total(); i++)
        {
            latents[i] = dist(gen);
        }
    }

    ncnn::Mat x_cos;
    ncnn::Mat x_sin;
    {
        const int patch_size = 2;

        const int num_patches_w = latents.w / patch_size;
        const int num_patches_h = latents.h / patch_size;
        const int num_patches = num_patches_w * num_patches_h;

        fprintf(stderr, "num_patches = %d x %d\n", num_patches_w, num_patches_h);

        const int cap_len = cap.h;
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

        rope_embbedder(x_pos_ids, x_cos, x_sin);
    }

    ncnn::Mat cap_cos;
    ncnn::Mat cap_sin;
    {
        const int cap_len = cap.h;

        ncnn::Mat cap_pos_ids(3, cap_len);
        for (int i = 0; i < cap_len; i++)
        {
            int* p = cap_pos_ids.row<int>(i);
            p[0] = 1 + i;
            p[1] = 0;
            p[2] = 0;
        }

        rope_embbedder(cap_pos_ids, cap_cos, cap_sin);
    }

    // concat along seqlen
    ncnn::Mat unified_cos;
    ncnn::Mat unified_sin;
    {
        const int total_half_dim = x_cos.w;
        unified_cos.create(total_half_dim, x_cos.h + cap_cos.h);
        unified_sin.create(total_half_dim, x_sin.h + cap_sin.h);
        memcpy(unified_cos, x_cos, total_half_dim * x_cos.h * sizeof(float));
        memcpy(unified_cos.row(x_cos.h), cap_cos, total_half_dim * cap_cos.h * sizeof(float));
        memcpy(unified_sin, x_sin, total_half_dim * x_sin.h * sizeof(float));
        memcpy(unified_sin.row(x_sin.h), cap_sin, total_half_dim * cap_sin.h * sizeof(float));
    }

    // cap_embedder
    ncnn::Mat cap_embed;
    {
        ncnn::Net cap_embedder;
        cap_embedder.opt.vulkan_device_index = gpuid;
        cap_embedder.opt.use_vulkan_compute = use_gpu;
        cap_embedder.opt.use_fp16_packed = false;
        cap_embedder.opt.use_fp16_storage = false;
        cap_embedder.opt.use_fp16_arithmetic = false;
        cap_embedder.opt.use_bf16_packed = use_bf16;
        cap_embedder.opt.use_bf16_storage = use_bf16;
        cap_embedder.load_param("z_image_turbo_transformer_cap_embedder.ncnn.param");
        cap_embedder.load_model("z_image_turbo_transformer_cap_embedder.ncnn.bin");

        ncnn::Extractor ex = cap_embedder.create_extractor();

        ex.input("in0", cap);

        ex.extract("out0", cap_embed);
    }

    // context_refiner
    ncnn::Mat cap_refine;
    {
        ncnn::Net context_refiner;
        context_refiner.opt.vulkan_device_index = gpuid;
        context_refiner.opt.use_vulkan_compute = use_gpu;
        context_refiner.opt.use_fp16_packed = false;
        context_refiner.opt.use_fp16_storage = false;
        context_refiner.opt.use_fp16_arithmetic = false;
        context_refiner.opt.use_bf16_packed = use_bf16;
        context_refiner.opt.use_bf16_storage = use_bf16;
        context_refiner.load_param("z_image_turbo_transformer_context_refiner.ncnn.param");
        context_refiner.load_model("z_image_turbo_transformer_context_refiner.ncnn.bin");

        ncnn::Extractor ex = context_refiner.create_extractor();

        ex.input("in0", cap_embed);
        ex.input("in1", cap_cos);
        ex.input("in2", cap_sin);

        ex.extract("out0", cap_refine);
    }

    // patchify
    ncnn::Mat x;
    {
        const int patch_size = 2;

        const int num_patches_w = latents.w / patch_size;
        const int num_patches_h = latents.h / patch_size;
        const int num_patches = num_patches_w * num_patches_h;

        const int channels = latents.c;

        x.create(patch_size * patch_size * channels, num_patches);

        for (int py = 0; py < num_patches_h; py++)
        {
            for (int px = 0; px < num_patches_w; px++)
            {
                float* outptr = x.row(py * num_patches_w + px);

                for (int q = 0; q < channels; q++)
                {
                    *outptr++ = latents.channel(q).row(py * 2)[px * 2];
                }
                for (int q = 0; q < channels; q++)
                {
                    *outptr++ = latents.channel(q).row(py * 2)[px * 2 + 1];
                }
                for (int q = 0; q < channels; q++)
                {
                    *outptr++ = latents.channel(q).row(py * 2 + 1)[px * 2];
                }
                for (int q = 0; q < channels; q++)
                {
                    *outptr++ = latents.channel(q).row(py * 2 + 1)[px * 2 + 1];
                }
            }
        }
    }

    // prepare timesteps
    std::vector<float> sigmas;
    std::vector<float> timesteps;
    {
        const float shift = 3.f;

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

    // t_embedder
    ncnn::Mat t_embeds;
    {
        ncnn::Net t_embedder;
        t_embedder.opt.vulkan_device_index = gpuid;
        t_embedder.opt.use_vulkan_compute = use_gpu;
        t_embedder.opt.use_fp16_packed = false;
        t_embedder.opt.use_fp16_storage = false;
        t_embedder.opt.use_fp16_arithmetic = false;
        t_embedder.opt.use_bf16_packed = use_bf16;
        t_embedder.opt.use_bf16_storage = use_bf16;
        t_embedder.load_param("z_image_turbo_transformer_t_embedder.ncnn.param");
        t_embedder.load_model("z_image_turbo_transformer_t_embedder.ncnn.bin");

        const float t_scale = 1000.f;

        ncnn::Mat t_mat(1, steps);
        for (int z = 0; z < steps; z++)
        {
            t_mat.row(z)[0] = timesteps[z] * t_scale;
        }

        ncnn::Extractor ex = t_embedder.create_extractor();

        ex.input("in0", t_mat);

        ex.extract("out0", t_embeds);
    }

    // diffusion transformer loop
    {
        ncnn::Net all_x_embedder;
        all_x_embedder.opt.vulkan_device_index = gpuid;
        all_x_embedder.opt.use_vulkan_compute = use_gpu;
        all_x_embedder.opt.use_fp16_packed = false;
        all_x_embedder.opt.use_fp16_storage = false;
        all_x_embedder.opt.use_fp16_arithmetic = false;
        all_x_embedder.opt.use_bf16_packed = use_bf16;
        all_x_embedder.opt.use_bf16_storage = use_bf16;
        all_x_embedder.load_param("z_image_turbo_transformer_all_x_embedder.ncnn.param");
        all_x_embedder.load_model("z_image_turbo_transformer_all_x_embedder.ncnn.bin");

        ncnn::Net noise_refiner;
        noise_refiner.opt.vulkan_device_index = gpuid;
        noise_refiner.opt.use_vulkan_compute = use_gpu;
        noise_refiner.opt.use_fp16_packed = false;
        noise_refiner.opt.use_fp16_storage = false;
        noise_refiner.opt.use_fp16_arithmetic = false;
        noise_refiner.opt.use_bf16_packed = use_bf16;
        noise_refiner.opt.use_bf16_storage = use_bf16;
        noise_refiner.load_param("z_image_turbo_transformer_noise_refiner.ncnn.param");
        noise_refiner.load_model("z_image_turbo_transformer_noise_refiner.ncnn.bin");

        ncnn::Net unified_refiner;
        unified_refiner.opt.vulkan_device_index = gpuid;
        unified_refiner.opt.use_vulkan_compute = use_gpu;
        unified_refiner.opt.use_fp16_packed = false;
        unified_refiner.opt.use_fp16_storage = false;
        unified_refiner.opt.use_fp16_arithmetic = false;
        unified_refiner.opt.use_bf16_packed = use_bf16;
        unified_refiner.opt.use_bf16_storage = use_bf16;
        unified_refiner.load_param("z_image_turbo_transformer_unified.ncnn.param");
        unified_refiner.load_model("z_image_turbo_transformer_unified.ncnn.bin");

        ncnn::Net all_final_layer;
        all_final_layer.opt.vulkan_device_index = gpuid;
        all_final_layer.opt.use_vulkan_compute = use_gpu;
        all_final_layer.opt.use_fp16_packed = false;
        all_final_layer.opt.use_fp16_storage = false;
        all_final_layer.opt.use_fp16_arithmetic = false;
        all_final_layer.opt.use_bf16_packed = use_bf16;
        all_final_layer.opt.use_bf16_storage = use_bf16;
        all_final_layer.load_param("z_image_turbo_transformer_all_final_layer.ncnn.param");
        all_final_layer.load_model("z_image_turbo_transformer_all_final_layer.ncnn.bin");

        for (int z = 0; z < steps; z++)
        {
            // all_x_embedder
            ncnn::Mat x_embed;
            {
                ncnn::Extractor ex = all_x_embedder.create_extractor();

                ex.input("in0", x);

                ex.extract("out0", x_embed);
            }

            ncnn::Mat t_embed = t_embeds.row_range(z, 1).clone();

            // noise_refiner
            ncnn::Mat x_embed_refine;
            {
                ncnn::Extractor ex = noise_refiner.create_extractor();

                ex.input("in0", x_embed);
                ex.input("in1", x_cos);
                ex.input("in2", x_sin);
                ex.input("in3", t_embed);

                ex.extract("out0", x_embed_refine);
            }

            // concat x_embed_refine and cap_refine
            ncnn::Mat unified_embed;
            {
                const int x_len = x_embed_refine.w;
                unified_embed.create(x_len, x_embed_refine.h + cap_refine.h);
                memcpy(unified_embed, x_embed_refine, x_len * x_embed_refine.h * sizeof(float));
                memcpy(unified_embed.row(x_embed_refine.h), cap_refine, x_len * cap_refine.h * sizeof(float));
            }

            // unified
            ncnn::Mat unified;
            {
                ncnn::Extractor ex = unified_refiner.create_extractor();

                ex.input("in0", unified_embed);
                ex.input("in1", unified_cos);
                ex.input("in2", unified_sin);
                ex.input("in3", t_embed);

                ex.extract("out0", unified);
            }

            // all_final_layer
            ncnn::Mat unified_final;
            {
                ncnn::Extractor ex = all_final_layer.create_extractor();

                ex.input("in0", unified);
                ex.input("in1", t_embed);

                ex.extract("out0", unified_final);
            }

            // euler scheduler step
            {
                const float dt = sigmas[z + 1] - sigmas[z];

                const int total = x.total();
                for (int i = 0; i < total; i++)
                {
                    x[i] = x[i] - dt * unified_final[i];
                }
            }

            fprintf(stderr, "step %d done\n", z);
        }
    }

    // unpatchify
    {
        // 64, 288 (2*2 * 16, pw * ph)
        const int patch_size = 2;

        const int num_patches_w = latents.w / patch_size;
        const int num_patches_h = latents.h / patch_size;
        const int num_patches = num_patches_w * num_patches_h;

        const int channels = latents.c;

        for (int py = 0; py < num_patches_h; py++)
        {
            for (int px = 0; px < num_patches_w; px++)
            {
                const float* ptr = x.row(py * num_patches_w + px);

                for (int q = 0; q < channels; q++)
                {
                    latents.channel(q).row(py * 2)[px * 2] = *ptr++;
                }
                for (int q = 0; q < channels; q++)
                {
                    latents.channel(q).row(py * 2)[px * 2 + 1] = *ptr++;
                }
                for (int q = 0; q < channels; q++)
                {
                    latents.channel(q).row(py * 2 + 1)[px * 2] = *ptr++;
                }
                for (int q = 0; q < channels; q++)
                {
                    latents.channel(q).row(py * 2 + 1)[px * 2 + 1] = *ptr++;
                }
            }
        }
    }

    // vae decode
    ncnn::Mat vae_out;
    {
        const float vae_scaling_factor = 0.3611f;
        const float vae_shift_factor = 0.1159f;

        for (int i = 0; i < latents.total(); i++)
        {
            latents[i] = latents[i] / vae_scaling_factor + vae_shift_factor;
        }

        ncnn::Net vae;
        vae.opt.vulkan_device_index = gpuid;
        vae.opt.use_vulkan_compute = use_gpu;
        vae.opt.use_fp16_packed = false;
        vae.opt.use_fp16_storage = false;
        vae.opt.use_fp16_arithmetic = false;
        vae.opt.use_bf16_packed = use_bf16;
        vae.opt.use_bf16_storage = use_bf16;
        vae.load_param("z_image_turbo_vae.ncnn.param");
        vae.load_model("z_image_turbo_vae.ncnn.bin");

        ncnn::Extractor ex = vae.create_extractor();

        ex.input("in0", latents);

        ex.extract("out0", vae_out);
    }

    // save image
    {
        // -1 ~ 1 to 0 ~ 255
        const float mean_vals[3] = {-1.f, -1.f, -1.f};
        const float norm_vals[3] = {127.5f, 127.5f, 127.5f};
        vae_out.substract_mean_normalize(mean_vals, norm_vals);

        ncnn::Mat rgb(width, height, (size_t)3u, 3);

        vae_out.to_pixels((unsigned char*)rgb.data, ncnn::Mat::PIXEL_RGB);

#if _WIN32
        wic_encode_image(L"out.png", rgb.w, rgb.h, rgb.elempack, rgb.data);
#else
        png_save("out.png", rgb.w, rgb.h, rgb.elempack, (const unsigned char*)rgb.data);
#endif
    }

    return 0;
}
