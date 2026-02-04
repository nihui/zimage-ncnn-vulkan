// zimage implemented with ncnn library

#ifndef ZIMAGE_H
#define ZIMAGE_H

#include <memory>
#include <string>

// ncnn
#include "net.h"
#include "gpu.h"
#include "layer.h"

#include "filesystem_utils.h"

#include "bpe_tokenizer.h"

namespace ZImage {

void generate_latent(int width, int height, int seed, ncnn::Mat& latent);

void rope_embedder(const ncnn::Mat& ids, ncnn::Mat& out_cos, ncnn::Mat& out_sin);

void generate_x_freqs(int num_patches_w, int num_patches_h, int cap_len, ncnn::Mat& x_cos, ncnn::Mat& x_sin);

void generate_cap_freqs(int cap_len, ncnn::Mat& cap_cos, ncnn::Mat& cap_sin);

void concat_along_h(const ncnn::Mat& a, const ncnn::Mat& b, ncnn::Mat& out);

void prepare_timestamps(int steps, float shift, std::vector<float>& sigmas, std::vector<float>& timesteps);

void patchify(const ncnn::Mat& latent, ncnn::Mat& x);

void unpatchify(const ncnn::Mat& x, ncnn::Mat& latent);

class Tokenizer
{
public:
    Tokenizer();

    int encode(const path_t& prompt, std::vector<int>& ids) const;

private:
    BpeTokenizer bpe;
};

class TextEncoder
{
public:
    int load(const ncnn::Option& opt);

    int process(const std::vector<int>& input_ids, ncnn::Mat& cap);

private:
    ncnn::Net text_encoder;
};

class CapEmbedder
{
public:
    int load(const ncnn::Option& opt);

    int process(const ncnn::Mat& cap, ncnn::Mat& cap_embed);

private:
    ncnn::Net cap_embedder;
};

class ContextRefiner
{
public:
    int load(const ncnn::Option& opt);

    int process(const ncnn::Mat& cap_embed, const ncnn::Mat& cap_cos, const ncnn::Mat& cap_sin, ncnn::Mat& cap_refine);

private:
    ncnn::Net context_refiner;
};

class TEmbedder
{
public:
    int load(const ncnn::Option& opt);

    int process(const std::vector<float>& timesteps, ncnn::Mat& t_embeds);

private:
    ncnn::Net t_embedder;
};

class AllXEmbedder
{
public:
    int load(const ncnn::Option& opt);

    int process(const ncnn::Mat& x, ncnn::Mat& x_embed);

private:
    ncnn::Net all_x_embedder;
};

class NoiseRefiner
{
public:
    int load(const ncnn::Option& opt);

    int process(const ncnn::Mat& x_embed, const ncnn::Mat& x_cos, const ncnn::Mat& x_sin, const ncnn::Mat& t_embed, ncnn::Mat& x_embed_refine);

private:
    ncnn::Net noise_refiner;
};

class UnifiedRefiner
{
public:
    int load(const ncnn::Option& opt);

    int process(const ncnn::Mat& unified_embed, const ncnn::Mat& unified_cos, const ncnn::Mat& unified_sin, const ncnn::Mat& t_embed, ncnn::Mat& unified);

private:
    ncnn::Net unified_refiner;
};

class AllFinalLayer
{
public:
    int load(const ncnn::Option& opt);

    int process(const ncnn::Mat& unified, const ncnn::Mat& t_embed, ncnn::Mat& unified_final);

private:
    ncnn::Net all_final_layer;
};

class VAE
{
public:
    int load(const ncnn::Option& opt);

    int process(const ncnn::Mat& latent, ncnn::Mat& vae_out);

private:
    ncnn::Net vae;
};

} // namespace ZImage

#endif // ZIMAGE_H
