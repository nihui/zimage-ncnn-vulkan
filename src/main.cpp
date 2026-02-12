// z-image implemented with ncnn library

#include <float.h>
#include <locale.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <vector>

#if _WIN32
// image encoder with wic
#include "wic_image.h"
#else // _WIN32
// image encoder with libjpeg and libpng
#include "jpeg_image.h"
#include "png_image.h"
#endif // _WIN32
#include "webp_image.h"

#if _WIN32
#include <wchar.h>
static wchar_t* optarg = NULL;
static int optind = 1;
static wchar_t getopt(int argc, wchar_t* const argv[], const wchar_t* optstring)
{
    if (optind >= argc || argv[optind][0] != L'-')
        return -1;

    wchar_t opt = argv[optind][1];
    const wchar_t* p = wcschr(optstring, opt);
    if (p == NULL)
        return L'?';

    optarg = NULL;

    if (p[1] == L':')
    {
        optind++;
        if (optind >= argc)
            return L'?';

        optarg = argv[optind];
    }

    optind++;

    return opt;
}

static std::vector<int> parse_optarg_int_array(const wchar_t* optarg)
{
    std::vector<int> array;
    array.push_back(_wtoi(optarg));

    const wchar_t* p = wcschr(optarg, L',');
    while (p)
    {
        p++;
        array.push_back(_wtoi(p));
        p = wcschr(p, L',');
    }

    return array;
}
#else // _WIN32
#include <unistd.h> // getopt()

static std::vector<int> parse_optarg_int_array(const char* optarg)
{
    std::vector<int> array;
    array.push_back(atoi(optarg));

    const char* p = strchr(optarg, ',');
    while (p)
    {
        p++;
        array.push_back(atoi(p));
        p = strchr(p, ',');
    }

    return array;
}
#endif // _WIN32

// ncnn
#include "mat.h"
#include "net.h"

#include "filesystem_utils.h"

#include "zimage.h"

static void print_usage()
{
    fprintf(stdout, "Usage: zimage-ncnn-vulkan -p prompt -o outfile [options]...\n\n");
    fprintf(stdout, "  -h                   show this help\n");
    fprintf(stdout, "  -p prompt            prompt (default=rand)\n");
    fprintf(stdout, "  -n negative-prompt   negative prompt (optional)\n");
    fprintf(stdout, "  -o output-path       output image path (default=out.png)\n");
    fprintf(stdout, "  -s image-size        image resolution (default=1024,1024)\n");
    fprintf(stdout, "  -l steps             denoise steps (default=auto)\n");
    fprintf(stdout, "  -r random-seed       random seed (default=rand)\n");
    fprintf(stdout, "  -m model-path        z-image model path (default=z-image-turbo)\n");
    fprintf(stdout, "  -g gpu-id            gpu device to use (-1=cpu, default=auto)\n");
}

#if _WIN32
int wmain(int argc, wchar_t** argv)
#else
int main(int argc, char** argv)
#endif
{
    srand(time(NULL));

    path_t prompt = PATHSTR("A half-length portrait in the warm light of a convenience store late at night. An East Asian beauty, holding milk, meets your gaze in front of the freezer.");
    path_t negative_prompt;
    path_t outpath = PATHSTR("out.png");
    int width = 1024;
    int height = 1024;
    int steps = -1;
    int seed = rand();
    path_t model = PATHSTR("z-image-turbo");
    int gpuid = 233;

    // parse cli args
    {
#if _WIN32
        setlocale(LC_ALL, "");
        wchar_t opt;
        while ((opt = getopt(argc, argv, L"p:n:o:s:l:r:m:g:h")) != (wchar_t)-1)
        {
            switch (opt)
            {
            case L'p':
                prompt = optarg;
                break;
            case L'n':
                negative_prompt = optarg;
                break;
            case L'o':
                outpath = optarg;
                break;
            case L's':
            {
                std::vector<int> list = parse_optarg_int_array(optarg);
                if (list.size() != 2)
                {
                    print_usage();
                    return -1;
                }
                width = list[0];
                height = list[1];
                break;
            }
            case L'l':
                steps = _wtoi(optarg);
                break;
            case L'r':
                seed = _wtoi(optarg);
                break;
            case L'm':
                model = optarg;
                break;
            case L'g':
                gpuid = _wtoi(optarg);
                break;
            case L'h':
            default:
                print_usage();
                return -1;
            }
        }
#else // _WIN32
        int opt;
        while ((opt = getopt(argc, argv, "p:n:o:s:l:r:m:g:h")) != -1)
        {
            switch (opt)
            {
            case 'p':
                prompt = optarg;
                break;
            case 'n':
                negative_prompt = optarg;
                break;
            case 'o':
                outpath = optarg;
                break;
            case 's':
            {
                std::vector<int> list = parse_optarg_int_array(optarg);
                if (list.size() != 2)
                {
                    print_usage();
                    return -1;
                }
                width = list[0];
                height = list[1];
                break;
            }
            case 'l':
                steps = atoi(optarg);
                break;
            case 'r':
                seed = atoi(optarg);
                break;
            case 'm':
                model = optarg;
                break;
            case 'g':
                gpuid = atoi(optarg);
                break;
            case 'h':
            default:
                print_usage();
                return -1;
            }
        }
#endif // _WIN32
    }

    if (prompt.empty() || outpath.empty())
    {
        print_usage();
        return -1;
    }

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

    float guidance_scale;
    float scheduler_shift;
    if (model.find(PATHSTR("z-image-turbo")) != path_t::npos)
    {
        guidance_scale = 0.f;
        scheduler_shift = 3.f;
        if (steps == -1)
            steps = 9;
    }
    else if (model.find(PATHSTR("z-image")) != path_t::npos)
    {
        guidance_scale = 1.f;
        scheduler_shift = 6.f;
        if (steps == -1)
            steps = 50;
    }
    else
    {
        fprintf(stderr, "unknown model dir type\n");
        return -1;
    }

    if (gpuid == 233)
    {
        gpuid = ncnn::get_default_gpu_index();
    }

#if _WIN32
    CoInitializeEx(NULL, COINIT_MULTITHREADED);
#endif

#if _WIN32
    fwprintf(stderr, L"prompt = %ls\n", prompt.c_str());
    fwprintf(stderr, L"negative-prompt = %ls\n", negative_prompt.c_str());
    fwprintf(stderr, L"output-path = %ls\n", outpath.c_str());
    fwprintf(stderr, L"model = %ls\n", model.c_str());
#else
    fprintf(stderr, "prompt = %s\n", prompt.c_str());
    fprintf(stderr, "negative-prompt = %s\n", negative_prompt.c_str());
    fprintf(stderr, "output-path = %s\n", outpath.c_str());
    fprintf(stderr, "model = %s\n", model.c_str());
#endif
    fprintf(stderr, "image-size = %d x %d\n", width, height);
    fprintf(stderr, "steps = %d\n", steps);
    fprintf(stderr, "seed = %d\n", seed);
    fprintf(stderr, "gpu-id = %d\n", gpuid);

    const bool apply_cfg = guidance_scale > 0.f;

    ncnn::Option opt;
    opt.vulkan_device_index = gpuid;
    opt.use_vulkan_compute = gpuid >= 0;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_packed = gpuid >= 0;
    opt.use_bf16_storage = gpuid >= 0;
    // opt.use_mapped_model_loading = true;

    // estimate transformer and vae memory usage
    const uint32_t heap_budget = gpuid >= 0 ? ncnn::get_gpu_device(gpuid)->get_heap_budget() : 0;

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
        int max_tile_area = (int)((float)heap_budget / 5700 * 1024 * 1024);
        ZImage::get_optimal_tile_size(width, height, max_tile_area, &vae_tile_width, &vae_tile_height);

        NCNN_LOGE("vae_tile_size = %d x %d", vae_tile_width, vae_tile_height);
    }

    // tokenizer
    std::vector<int> input_ids;
    std::vector<int> neg_input_ids;
    {
        ZImage::Tokenizer tokenizer(model);

        tokenizer.encode(prompt, input_ids);

        if (apply_cfg)
        {
            tokenizer.encode(negative_prompt, neg_input_ids);
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

    // prepare latent
    ncnn::Mat latent;
    ZImage::generate_latent(width, height, seed, latent);

    const int patch_size = 2;
    const int num_patches_w = latent.w / patch_size;
    const int num_patches_h = latent.h / patch_size;

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

    // patchify
    ncnn::Mat x;
    ZImage::patchify(latent, x);

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

        for (int z = 0; z < steps; z++)
        {
            ncnn::Mat t_embed = t_embeds.row_range(z, 1).clone();

            // all_x_embedder
            ncnn::Mat x_embed;
            all_x_embedder.process(x, x_embed);

            // noise_refiner
            ncnn::Mat x_embed_refine;
            noise_refiner.process(x_embed, x_cos, x_sin, t_embed, x_embed_refine);

            // concat x_embed_refine and cap_refine
            ncnn::Mat unified_embed;
            ZImage::concat_along_h(x_embed_refine, cap_refine, unified_embed);

            ncnn::Mat neg_unified_embed;
            if (apply_cfg)
            {
                ZImage::concat_along_h(x_embed_refine, neg_cap_refine, neg_unified_embed);
            }

            // unified
            ncnn::Mat unified;
            unified_refiner.process(unified_embed, unified_cos, unified_sin, t_embed, unified);

            ncnn::Mat neg_unified;
            if (apply_cfg)
            {
                unified_refiner.process(neg_unified_embed, neg_unified_cos, neg_unified_sin, t_embed, neg_unified);
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

            fprintf(stderr, "step %d done\n", z);
        }
    }

    // unpatchify
    ZImage::unpatchify(x, latent);

    // vae decode
    ncnn::Mat outimage;
    {
        const float vae_scaling_factor = 0.3611f;
        const float vae_shift_factor = 0.1159f;

        for (int i = 0; i < latent.total(); i++)
        {
            latent[i] = latent[i] / vae_scaling_factor + vae_shift_factor;
        }

        const bool use_vae_tiled = vae_tile_width < width || vae_tile_height < height;

        ZImage::VAE vae;

        vae.load(model, use_vae_tiled, opt);

        if (use_vae_tiled)
        {
            vae.process_tiled(latent, vae_tile_width, vae_tile_height, outimage);
        }
        else
        {
            vae.process(latent, outimage);
        }
    }

    // save image
    {
        int success = 0;

        path_t ext = get_file_extension(outpath);

        if (ext == PATHSTR("webp") || ext == PATHSTR("WEBP"))
        {
            success = webp_save(outpath.c_str(), outimage.w, outimage.h, outimage.elempack, (const unsigned char*)outimage.data);
        }
        else if (ext == PATHSTR("png") || ext == PATHSTR("PNG"))
        {
#if _WIN32
            success = wic_encode_image(outpath.c_str(), outimage.w, outimage.h, outimage.elempack, outimage.data);
#else
            success = png_save(outpath.c_str(), outimage.w, outimage.h, outimage.elempack, (const unsigned char*)outimage.data);
#endif
        }
        else if (ext == PATHSTR("jpg") || ext == PATHSTR("JPG") || ext == PATHSTR("jpeg") || ext == PATHSTR("JPEG"))
        {
#if _WIN32
            success = wic_encode_jpeg_image(outpath.c_str(), outimage.w, outimage.h, outimage.elempack, outimage.data);
#else
            success = jpeg_save(outpath.c_str(), outimage.w, outimage.h, outimage.elempack, (const unsigned char*)outimage.data);
#endif
        }

        if (!success)
        {
#if _WIN32
            fwprintf(stderr, L"encode image %ls failed\n", outpath.c_str());
#else
            fprintf(stderr, "encode image %s failed\n", outpath.c_str());
#endif
        }
    }

    return 0;
}
