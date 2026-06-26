// z-image implemented with ncnn library

#include <locale.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <vector>

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

#include "filesystem_utils.h"

#include "lanpaint.h"
#include "zimage_pipeline.h"

static void print_usage()
{
    fprintf(stdout, "Usage: zimage-ncnn-vulkan -p prompt -o outfile [options]...\n\n");
    fprintf(stdout, "  -h                   show this help\n");
    fprintf(stdout, "  -p prompt            prompt (default=rand)\n");
    fprintf(stdout, "  -n negative-prompt   negative prompt (optional)\n");
    fprintf(stdout, "  -o output-path       output image path (default=out.png)\n");
    fprintf(stdout, "  -i input-image       input image for inpaint (optional)\n");
    fprintf(stdout, "  -k mask-image        inpaint mask, white=paint black=preserve (optional)\n");
    fprintf(stdout, "  -x l,t,r,b           outpaint by expanding input canvas (optional)\n");
    fprintf(stdout, "  -c control-image     control image for ControlNet (optional)\n");
    fprintf(stdout, "  -w control-scale     ControlNet scale (default=1.0)\n");
    fprintf(stdout, "  -t                   Tile ControlNet upscale mode (optional)\n");
    fprintf(stdout, "  -d denoise-strength  denoise strength for -t mode (default=0.5)\n");
    fprintf(stdout, "  -s image-size        image resolution (default=1024,1024)\n");
    fprintf(stdout, "  -l steps             denoise steps (default=auto)\n");
    fprintf(stdout, "  -r random-seed       random seed (default=rand)\n");
    fprintf(stdout, "  -m model-path        z-image model path (default=z-image-turbo)\n");
    fprintf(stdout, "  -g gpu-id            gpu device to use (-1=cpu, default=auto)\n");
    fprintf(stdout, "  -b batch-size        batched generation (default=1)\n");
}

#if _WIN32
int wmain(int argc, wchar_t** argv)
#else
int main(int argc, char** argv)
#endif
{
    srand(time(NULL));

    ZImagePipeline zimage_pipeline;
    LanPaintPipeline lanpaint_pipeline;

    zimage_pipeline.prompt = PATHSTR("A half-length portrait in the warm light of a convenience store late at night. An East Asian beauty, holding milk, meets your gaze in front of the freezer.");
    zimage_pipeline.outpath = PATHSTR("out.png");
    zimage_pipeline.seed = rand();
    zimage_pipeline.model = PATHSTR("z-image-turbo");

    lanpaint_pipeline.prompt = zimage_pipeline.prompt;
    lanpaint_pipeline.outpath = zimage_pipeline.outpath;
    lanpaint_pipeline.seed = zimage_pipeline.seed;
    lanpaint_pipeline.model = zimage_pipeline.model;

    // Keep LanPaint tuning as source-level defaults until the behavior is stable.
    lanpaint_pipeline.lanpaint_steps = 5;
    lanpaint_pipeline.lanpaint_lambda = 16.f;
    lanpaint_pipeline.lanpaint_step_size = 0.2f;
    lanpaint_pipeline.lanpaint_beta = 1.f;
    lanpaint_pipeline.lanpaint_friction = 15.f;
    lanpaint_pipeline.lanpaint_early_stop = 1;
    lanpaint_pipeline.lanpaint_prompt_first = false;
    lanpaint_pipeline.preserve_known = true;
    lanpaint_pipeline.outpaint[0] = 0;
    lanpaint_pipeline.outpaint[1] = 0;
    lanpaint_pipeline.outpaint[2] = 0;
    lanpaint_pipeline.outpaint[3] = 0;
    lanpaint_pipeline.outpaint_set = false;

    // parse cli args
    {
#if _WIN32
        setlocale(LC_ALL, "");
        wchar_t opt;
        while ((opt = getopt(argc, argv, L"p:n:o:i:k:x:c:w:td:s:l:r:m:g:b:h")) != (wchar_t)-1)
        {
            switch (opt)
            {
            case L'p':
                zimage_pipeline.prompt = optarg;
                lanpaint_pipeline.prompt = zimage_pipeline.prompt;
                break;
            case L'n':
                zimage_pipeline.negative_prompt = optarg;
                lanpaint_pipeline.negative_prompt = zimage_pipeline.negative_prompt;
                break;
            case L'o':
                zimage_pipeline.outpath = optarg;
                lanpaint_pipeline.outpath = zimage_pipeline.outpath;
                break;
            case L'i':
                lanpaint_pipeline.inputpath = optarg;
                break;
            case L'k':
                lanpaint_pipeline.maskpath = optarg;
                break;
            case L'x':
            {
                std::vector<int> list = parse_optarg_int_array(optarg);
                if (list.size() != 4)
                {
                    print_usage();
                    return -1;
                }
                for (int j = 0; j < 4; j++)
                    lanpaint_pipeline.outpaint[j] = list[j];
                lanpaint_pipeline.outpaint_set = true;
                break;
            }
            case L'c':
                zimage_pipeline.controlpath = optarg;
                lanpaint_pipeline.controlpath = zimage_pipeline.controlpath;
                break;
            case L'w':
                zimage_pipeline.control_scale = (float)_wtof(optarg);
                lanpaint_pipeline.control_scale = zimage_pipeline.control_scale;
                break;
            case L't':
                zimage_pipeline.control_tile = true;
                break;
            case L'd':
                zimage_pipeline.denoise_strength = (float)_wtof(optarg);
                break;
            case L's':
            {
                std::vector<int> list = parse_optarg_int_array(optarg);
                if (list.size() != 2)
                {
                    print_usage();
                    return -1;
                }
                zimage_pipeline.width = list[0];
                zimage_pipeline.height = list[1];
                lanpaint_pipeline.width = list[0];
                lanpaint_pipeline.height = list[1];
                lanpaint_pipeline.size_set = true;
                break;
            }
            case L'l':
                zimage_pipeline.steps = _wtoi(optarg);
                lanpaint_pipeline.steps = zimage_pipeline.steps;
                break;
            case L'r':
                zimage_pipeline.seed = _wtoi(optarg);
                lanpaint_pipeline.seed = zimage_pipeline.seed;
                break;
            case L'm':
                zimage_pipeline.model = optarg;
                lanpaint_pipeline.model = zimage_pipeline.model;
                break;
            case L'g':
                zimage_pipeline.gpuid = _wtoi(optarg);
                lanpaint_pipeline.gpuid = zimage_pipeline.gpuid;
                break;
            case L'b':
                zimage_pipeline.batch = _wtoi(optarg);
                lanpaint_pipeline.batch = zimage_pipeline.batch;
                break;
            case L'h':
                print_usage();
                return 0;
            default:
                print_usage();
                return -1;
            }
        }
#else // _WIN32
        int opt;
        while ((opt = getopt(argc, argv, "p:n:o:i:k:x:c:w:td:s:l:r:m:g:b:h")) != -1)
        {
            switch (opt)
            {
            case 'p':
                zimage_pipeline.prompt = optarg;
                lanpaint_pipeline.prompt = zimage_pipeline.prompt;
                break;
            case 'n':
                zimage_pipeline.negative_prompt = optarg;
                lanpaint_pipeline.negative_prompt = zimage_pipeline.negative_prompt;
                break;
            case 'o':
                zimage_pipeline.outpath = optarg;
                lanpaint_pipeline.outpath = zimage_pipeline.outpath;
                break;
            case 'i':
                lanpaint_pipeline.inputpath = optarg;
                break;
            case 'k':
                lanpaint_pipeline.maskpath = optarg;
                break;
            case 'x':
            {
                std::vector<int> list = parse_optarg_int_array(optarg);
                if (list.size() != 4)
                {
                    print_usage();
                    return -1;
                }
                for (int j = 0; j < 4; j++)
                    lanpaint_pipeline.outpaint[j] = list[j];
                lanpaint_pipeline.outpaint_set = true;
                break;
            }
            case 'c':
                zimage_pipeline.controlpath = optarg;
                lanpaint_pipeline.controlpath = zimage_pipeline.controlpath;
                break;
            case 'w':
                zimage_pipeline.control_scale = (float)atof(optarg);
                lanpaint_pipeline.control_scale = zimage_pipeline.control_scale;
                break;
            case 't':
                zimage_pipeline.control_tile = true;
                break;
            case 'd':
                zimage_pipeline.denoise_strength = (float)atof(optarg);
                break;
            case 's':
            {
                std::vector<int> list = parse_optarg_int_array(optarg);
                if (list.size() != 2)
                {
                    print_usage();
                    return -1;
                }
                zimage_pipeline.width = list[0];
                zimage_pipeline.height = list[1];
                lanpaint_pipeline.width = list[0];
                lanpaint_pipeline.height = list[1];
                lanpaint_pipeline.size_set = true;
                break;
            }
            case 'l':
                zimage_pipeline.steps = atoi(optarg);
                lanpaint_pipeline.steps = zimage_pipeline.steps;
                break;
            case 'r':
                zimage_pipeline.seed = atoi(optarg);
                lanpaint_pipeline.seed = zimage_pipeline.seed;
                break;
            case 'm':
                zimage_pipeline.model = optarg;
                lanpaint_pipeline.model = zimage_pipeline.model;
                break;
            case 'g':
                zimage_pipeline.gpuid = atoi(optarg);
                lanpaint_pipeline.gpuid = zimage_pipeline.gpuid;
                break;
            case 'b':
                zimage_pipeline.batch = atoi(optarg);
                lanpaint_pipeline.batch = zimage_pipeline.batch;
                break;
            case 'h':
                print_usage();
                return 0;
            default:
                print_usage();
                return -1;
            }
        }
#endif // _WIN32
    }

    if (zimage_pipeline.prompt.empty() || zimage_pipeline.outpath.empty())
    {
        print_usage();
        return -1;
    }

    if (!lanpaint_pipeline.inputpath.empty() || !lanpaint_pipeline.maskpath.empty() || lanpaint_pipeline.outpaint_set)
    {
        if (zimage_pipeline.control_tile)
        {
            fprintf(stderr, "-t tile control upscale cannot be used with LanPaint input, mask, or outpaint\n");
            return -1;
        }
        if (lanpaint_pipeline.inputpath.empty())
        {
            fprintf(stderr, "-k mask-image or -x outpaint requires -i input-image\n");
            return -1;
        }
        if (lanpaint_pipeline.load() != 0)
            return -1;
        return lanpaint_pipeline.generate();
    }

    if (zimage_pipeline.control_tile && zimage_pipeline.controlpath.empty())
    {
        fprintf(stderr, "-t tile control upscale requires -c control-image\n");
        return -1;
    }

    if (zimage_pipeline.load() != 0)
        return -1;
    return zimage_pipeline.generate();
}
