// z-image implemented with ncnn library

#include "zimage_pipeline.h"

#include <stdint.h>
#include <stdio.h>

#include <vector>

#include "image_io.h"

int ZImagePipeline::load()
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
    opt.use_bf16_packed = loaded_gpuid >= 0;
    opt.use_bf16_storage = loaded_gpuid >= 0;
    // opt.use_mapped_model_loading = true;

    // disable winograd for reducing vae vram usage
    opt.use_winograd_convolution = false;

    heap_budget = loaded_gpuid >= 0 ? ncnn::get_gpu_device(loaded_gpuid)->get_heap_budget() : 0;

    tokenizer.reset(new ZImage::Tokenizer(model));

    loaded = true;
    return 0;
}

int ZImagePipeline::generate() const
{
    if (!loaded)
    {
        fprintf(stderr, "pipeline is not loaded\n");
        return -1;
    }

    int width = this->width;
    int height = this->height;
    int steps = this->steps;

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
    fwprintf(stderr, L"model = %ls\n", model.c_str());
    if (control_enabled)
        fwprintf(stderr, L"control-image = %ls\n", controlpath.c_str());
#else
    fprintf(stderr, "prompt = %s\n", prompt.c_str());
    fprintf(stderr, "negative-prompt = %s\n", negative_prompt.c_str());
    fprintf(stderr, "output-path = %s\n", outpath.c_str());
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
            fprintf(stderr, "control image size must match image-size %d x %d but got %d x %d\n",
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
    for (int b = 0; b < batch; b++)
    {
        ZImage::generate_latent(width, height, seed + b, latents[b]);
    }

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

    // diffusion transformer loop
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
            // patchify
            ncnn::Mat x;
            ZImage::patchify(latents[b], x);

            for (int z = 0; z < steps; z++)
            {
                ncnn::Mat t_embed = t_embeds.row_range(z, 1).clone();

                // all_x_embedder
                ncnn::Mat x_embed;
                all_x_embedder.process(x, x_embed);

                // noise_refiner
                ncnn::Mat x_embed_refine;
                ncnn::Mat control_context;
                if (control_enabled)
                {
                    ncnn::Mat hint0;
                    ncnn::Mat hint1;
                    if (control_refiner.process(control_x, x_embed, x_cos, x_sin, t_embed, hint0, hint1, control_context) != 0)
                        return -1;
                    if (noise_refiner.process_controlled(x_embed, x_cos, x_sin, t_embed, hint0, hint1, control_scale, x_embed_refine) != 0)
                        return -1;
                }
                else
                {
                    noise_refiner.process(x_embed, x_cos, x_sin, t_embed, x_embed_refine);
                }

                ncnn::Mat neg_x_embed_refine;
                ncnn::Mat neg_control_context;
                if (apply_cfg && control_enabled)
                {
                    ncnn::Mat neg_hint0;
                    ncnn::Mat neg_hint1;
                    if (control_refiner.process(control_x, x_embed, neg_x_cos, neg_x_sin, t_embed, neg_hint0, neg_hint1, neg_control_context) != 0)
                        return -1;
                    if (noise_refiner.process_controlled(x_embed, neg_x_cos, neg_x_sin, t_embed, neg_hint0, neg_hint1, control_scale, neg_x_embed_refine) != 0)
                        return -1;
                }

                // concat x_embed_refine and cap_refine
                ncnn::Mat unified_embed;
                ZImage::concat_along_h(x_embed_refine, cap_refine, unified_embed);

                ncnn::Mat neg_unified_embed;
                if (apply_cfg)
                {
                    const ncnn::Mat& neg_refine = control_enabled ? neg_x_embed_refine : x_embed_refine;
                    ZImage::concat_along_h(neg_refine, neg_cap_refine, neg_unified_embed);
                }

                // unified
                ncnn::Mat unified;
                if (control_enabled)
                {
                    ncnn::Mat control_unified_embed;
                    ZImage::concat_along_h(control_context, cap_refine, control_unified_embed);

                    ncnn::Mat hint0;
                    ncnn::Mat hint10;
                    ncnn::Mat hint20;
                    if (control_unified.process(control_unified_embed, unified_embed, unified_cos, unified_sin, t_embed, hint0, hint10, hint20) != 0)
                        return -1;
                    if (unified_refiner.process_controlled(unified_embed, unified_cos, unified_sin, t_embed, hint0, hint10, hint20, control_scale, unified) != 0)
                        return -1;
                }
                else
                {
                    unified_refiner.process(unified_embed, unified_cos, unified_sin, t_embed, unified);
                }

                ncnn::Mat neg_unified;
                if (apply_cfg)
                {
                    if (control_enabled)
                    {
                        ncnn::Mat neg_control_unified_embed;
                        ZImage::concat_along_h(neg_control_context, neg_cap_refine, neg_control_unified_embed);

                        ncnn::Mat neg_hint0;
                        ncnn::Mat neg_hint10;
                        ncnn::Mat neg_hint20;
                        if (control_unified.process(neg_control_unified_embed, neg_unified_embed, neg_unified_cos, neg_unified_sin, t_embed, neg_hint0, neg_hint10, neg_hint20) != 0)
                            return -1;
                        if (unified_refiner.process_controlled(neg_unified_embed, neg_unified_cos, neg_unified_sin, t_embed, neg_hint0, neg_hint10, neg_hint20, control_scale, neg_unified) != 0)
                            return -1;
                    }
                    else
                    {
                        unified_refiner.process(neg_unified_embed, neg_unified_cos, neg_unified_sin, t_embed, neg_unified);
                    }
                }

                // all_final_layer
                ncnn::Mat unified_final;
                all_final_layer.process(unified, t_embed, unified_final);

                ncnn::Mat neg_unified_final;
                if (apply_cfg)
                {
                    all_final_layer.process(neg_unified, t_embed, neg_unified_final);
                }

                if (apply_cfg)
                {
                    // apply cfg
                    const int total = x.total();
                    for (int i = 0; i < total; i++)
                    {
                        float pos = unified_final[i];
                        float neg = neg_unified_final[i];

                        unified_final[i] = pos + guidance_scale * (pos - neg);
                    }
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

                if (batch > 1)
                {
                    fprintf(stderr, "step %d/%d of image %d/%d done\n", z + 1, steps, b + 1, batch);
                }
                else
                {
                    fprintf(stderr, "step %d/%d done\n", z + 1, steps);
                }
            }

            // unpatchify
            ZImage::unpatchify(x, latents[b]);
        }
    }

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
