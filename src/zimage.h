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

void get_optimal_tile_size(int width, int height, int max_tile_area, int* tile_width, int* tile_height);

int prepare_control_x(
    const ncnn::Mat& control_image,
    const path_t& model,
    bool use_vae_tiled,
    int vae_tile_width,
    int vae_tile_height,
    const ncnn::Option& opt,
    ncnn::Mat& control_x);

class Tokenizer
{
public:
    Tokenizer(const path_t& model);

    int encode(const path_t& prompt, std::vector<int>& ids) const;

private:
    BpeTokenizer bpe;
};

class TextEncoder
{
public:
    int load(const path_t& model, const ncnn::Option& opt);

    int process(const std::vector<int>& input_ids, ncnn::Mat& cap) const;

private:
    ncnn::Net text_encoder;
};

class CapEmbedder
{
public:
    int load(const path_t& model, const ncnn::Option& opt);

    int process(const ncnn::Mat& cap, ncnn::Mat& cap_embed) const;

private:
    ncnn::Net cap_embedder;
};

class ContextRefiner
{
public:
    int load(const path_t& model, const ncnn::Option& opt);

    int process(const ncnn::Mat& cap_embed, const ncnn::Mat& cap_cos, const ncnn::Mat& cap_sin, ncnn::Mat& cap_refine) const;

private:
    ncnn::Net context_refiner;
};

class TEmbedder
{
public:
    int load(const path_t& model, const ncnn::Option& opt);

    int process(const std::vector<float>& timesteps, ncnn::Mat& t_embeds) const;

private:
    ncnn::Net t_embedder;
};

class AllXEmbedder
{
public:
    int load(const path_t& model, const ncnn::Option& opt);

    int process(const ncnn::Mat& x, ncnn::Mat& x_embed) const;

private:
    ncnn::Net all_x_embedder;
};

class NoiseRefiner
{
public:
    int load(const path_t& model, const ncnn::Option& opt);

    int process(const ncnn::Mat& x_embed, const ncnn::Mat& x_cos, const ncnn::Mat& x_sin, const ncnn::Mat& t_embed, ncnn::Mat& x_embed_refine) const;

    int process_controlled(
        const ncnn::Mat& x_embed,
        const ncnn::Mat& x_cos,
        const ncnn::Mat& x_sin,
        const ncnn::Mat& t_embed,
        const ncnn::Mat& hint0,
        const ncnn::Mat& hint1,
        float control_scale,
        ncnn::Mat& x_embed_refine) const;

private:
    ncnn::Net noise_refiner;
};

class UnifiedRefiner
{
public:
    int load(const path_t& model, const ncnn::Option& opt);

    int process(const ncnn::Mat& unified_embed, const ncnn::Mat& unified_cos, const ncnn::Mat& unified_sin, const ncnn::Mat& t_embed, ncnn::Mat& unified) const;

    int process_controlled(
        const ncnn::Mat& unified_embed,
        const ncnn::Mat& unified_cos,
        const ncnn::Mat& unified_sin,
        const ncnn::Mat& t_embed,
        const ncnn::Mat& hint0,
        const ncnn::Mat& hint10,
        const ncnn::Mat& hint20,
        float control_scale,
        ncnn::Mat& unified) const;

private:
    ncnn::Net unified_refiner;
};

class ControlRefiner
{
public:
    int load(const path_t& model, const ncnn::Option& opt);

    int process(
        const ncnn::Mat& control_x,
        const ncnn::Mat& x_embed,
        const ncnn::Mat& x_cos,
        const ncnn::Mat& x_sin,
        const ncnn::Mat& t_embed,
        ncnn::Mat& hint0,
        ncnn::Mat& hint1,
        ncnn::Mat& control_context) const;

private:
    ncnn::Net control_refiner;
};

class ControlUnified
{
public:
    int load(const path_t& model, const ncnn::Option& opt);

    int process(
        const ncnn::Mat& control_unified_embed,
        const ncnn::Mat& unified_embed,
        const ncnn::Mat& unified_cos,
        const ncnn::Mat& unified_sin,
        const ncnn::Mat& t_embed,
        ncnn::Mat& hint0,
        ncnn::Mat& hint1,
        ncnn::Mat& hint2) const;

private:
    ncnn::Net control_unified;
};

class AllFinalLayer
{
public:
    int load(const path_t& model, const ncnn::Option& opt);

    int process(const ncnn::Mat& unified, const ncnn::Mat& t_embed, ncnn::Mat& unified_final) const;

private:
    ncnn::Net all_final_layer;
};

class VAEDecoder
{
public:
    int load(const path_t& model, bool use_vae_tiled, const ncnn::Option& opt);

    int process(const ncnn::Mat& latent, ncnn::Mat& outimage) const;

    int process_tiled(const ncnn::Mat& latent, int tile_width, int tile_height, ncnn::Mat& outimage) const;

private:
    ncnn::Net vae_decoder;
};

class VAEEncoder
{
public:
    int load(const path_t& model, bool use_vae_tiled, const ncnn::Option& opt);

    int process(const ncnn::Mat& image, ncnn::Mat& latent) const;

    int process_tiled(const ncnn::Mat& image, int tile_width, int tile_height, ncnn::Mat& latent) const;

private:
    ncnn::Net vae_encoder;
};

} // namespace ZImage

#endif // ZIMAGE_H
