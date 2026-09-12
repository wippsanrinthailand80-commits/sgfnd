#ifndef SGFND_CORE_H
#define SGFND_CORE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdio.h>

#define SGFND_BRIDGES_SMALL 92
#define SGFND_BRIDGES_COMPRESSED 46
#define SGFND_BRIDGES_LARGE 30000

#define SGFND_TILE_SIZE 64
#define SGFND_MAX_TILES 1024
#define SGFND_LATENT_DIM 512
#define SGFND_TIMESTEPS 1000

#ifdef DEBUG
#define SGFND_LOG_DEBUG(fmt, ...) fprintf(stderr, "[DEBUG] %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)
#define SGFND_LOG_INFO(fmt, ...) fprintf(stderr, "[INFO] %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)
#define SGFND_LOG_WARN(fmt, ...) fprintf(stderr, "[WARN] %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)
#define SGFND_LOG_ERROR(fmt, ...) fprintf(stderr, "[ERROR] %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)
#define SGFND_ASSERT(cond) do { if (!(cond)) { fprintf(stderr, "[ASSERT] %s:%d: %s\n", __FILE__, __LINE__, #cond); *(volatile int*)0 = 0; } } while(0)
#else
#define SGFND_LOG_DEBUG(fmt, ...) do {} while(0)
#define SGFND_LOG_INFO(fmt, ...) do {} while(0)
#define SGFND_LOG_WARN(fmt, ...) fprintf(stderr, "[WARN] " fmt "\n", ##__VA_ARGS__)
#define SGFND_LOG_ERROR(fmt, ...) fprintf(stderr, "[ERROR] " fmt "\n", ##__VA_ARGS__)
#define SGFND_ASSERT(cond) do {} while(0)
#endif

typedef enum {
    SGFND_MODE_SMALL = 0,
    SGFND_MODE_LARGE = 1
} sgfnd_mode_t;

typedef enum {
    SGFND_FP32 = 0,
    SGFND_COMPRESSED = 1,
    SGFND_INT8 = 2,
    SGFND_INT4 = 3
} sgfnd_bridge_type_t;

typedef enum {
    SGFND_QUANT_NONE = 0,
    SGFND_QUANT_INT8 = 1,
    SGFND_QUANT_INT4 = 2
} sgfnd_quantization_t;

typedef enum {
    SGFND_LOSS_MSE = 0,
    SGFND_LOSS_L1 = 1,
    SGFND_LOSS_PERCEPTUAL = 2,
    SGFND_LOSS_ADVERSARIAL = 3
} sgfnd_loss_type_t;

typedef struct {
    float *data;
    size_t size;
    size_t capacity;
    sgfnd_bridge_type_t type;
    sgfnd_quantization_t quant;
    int8_t *quant_data;
    float scale;
    float zero_point;
    float *grad;
} sgfnd_bridge_t;

typedef struct {
    sgfnd_bridge_t bridges[SGFND_BRIDGES_SMALL];
    size_t active_count;
    bool compressed;
} sgfnd_small_context_t;

typedef struct {
    sgfnd_bridge_t *bridges;
    size_t count;
    size_t capacity;
    char *weights_path;
    size_t *bridge_offsets;
    bool *bridge_loaded;
    size_t vram_budget;
    size_t vram_used;
} sgfnd_large_context_t;

typedef union {
    sgfnd_small_context_t small;
    sgfnd_large_context_t large;
} sgfnd_context_u;

typedef struct {
    sgfnd_mode_t mode;
    sgfnd_context_u ctx;
    void *vram_pool;
    size_t vram_size;
    sgfnd_quantization_t default_quant;
} sgfnd_engine_t;

typedef struct {
    uint32_t x, y;
    uint32_t width, height;
    uint32_t stride;
    float *pixels;
    bool owns_memory;
} sgfnd_tile_t;

typedef struct {
    uint32_t width;
    uint32_t height;
    uint8_t channels;
    uint8_t bit_depth;
    uint32_t tile_size;
    sgfnd_tile_t *tiles;
    size_t tile_count;
    float *full_pixels;
} sgfnd_image_t;

typedef struct {
    float r, g, b, a;
} sgfnd_color_t;

typedef struct {
    sgfnd_color_t *palette;
    size_t palette_size;
    float *grading_lut;
    size_t lut_size;
} sgfnd_color_grader_t;

typedef struct {
    char **tags;
    float *weights;
    size_t count;
} sgfnd_prompt_t;

typedef struct {
    uint32_t tile_size;
    uint32_t max_tiles_in_vram;
    void (*tile_callback)(const sgfnd_tile_t *tile, void *user_data);
    void *user_data;
} sgfnd_render_config_t;

typedef struct {
    float *latent;
    size_t dim;
    float *mu;
    float *logvar;
    float *eps;
} sgfnd_latent_t;

typedef struct {
    float *weights;
    float *bias;
    float *grad_w;
    float *grad_b;
    size_t in_dim;
    size_t out_dim;
    bool use_bias;
} sgfnd_linear_t;

typedef struct {
    sgfnd_linear_t **layers;
    size_t num_layers;
    size_t *layer_dims;
    float *activations;
    float *pre_activations;
} sgfnd_mlp_t;

typedef struct {
    sgfnd_mlp_t *time_embed;
    sgfnd_mlp_t *cond_embed;
    sgfnd_mlp_t *denoiser;
    float *timesteps;
    float *alphas;
    float *alphas_cumprod;
    float *betas;
    int num_timesteps;
} sgfnd_diffusion_t;

typedef struct {
    sgfnd_diffusion_t *diffusion;
    sgfnd_latent_t *latent;
    sgfnd_mlp_t *encoder;
    sgfnd_mlp_t *decoder;
    sgfnd_mlp_t *discriminator;
    float learning_rate;
    sgfnd_loss_type_t loss_type;
    size_t step_count;
    float *optimizer_state;
} sgfnd_model_t;

typedef struct {
    sgfnd_image_t **images;
    sgfnd_latent_t **latents;
    char **prompts;
    char **metadata;
    size_t count;
    size_t capacity;
    uint32_t img_width;
    uint32_t img_height;
} sgfnd_dataset_t;

typedef struct {
    float loss;
    float recon_loss;
    float kl_loss;
    float adv_loss;
    float grad_norm;
    uint64_t timestamp;
} sgfnd_training_step_t;

sgfnd_engine_t* sgfnd_create(sgfnd_mode_t mode, size_t vram_bytes);
void sgfnd_destroy(sgfnd_engine_t *engine);

int sgfnd_process_input(sgfnd_engine_t *engine, const float *input, size_t input_size);
int sgfnd_compress_bridges(sgfnd_engine_t *engine);
int sgfnd_decompress_bridges(sgfnd_engine_t *engine);

int sgfnd_bridge_quantize(sgfnd_bridge_t *bridge, sgfnd_quantization_t quant);
int sgfnd_bridge_dequantize(const sgfnd_bridge_t *bridge, float *out_data, size_t out_size);

int sgfnd_large_model_load_weights(sgfnd_engine_t *engine, const char *path);
int sgfnd_large_model_ensure_bridge(sgfnd_engine_t *engine, size_t index);
void sgfnd_large_model_evict_bridges(sgfnd_engine_t *engine, size_t keep_count);

sgfnd_image_t* sgfnd_image_create_tiled(uint32_t width, uint32_t height, uint8_t channels, uint32_t tile_size);
void sgfnd_image_free(sgfnd_image_t *img);
sgfnd_tile_t* sgfnd_image_get_tile(sgfnd_image_t *img, uint32_t tile_x, uint32_t tile_y);
int sgfnd_image_assemble_tiles(const sgfnd_image_t *img, float *out_pixels);

int sgfnd_render_tiled(sgfnd_engine_t *engine, const sgfnd_prompt_t *prompt, const sgfnd_render_config_t *config);
int sgfnd_render_tile(sgfnd_engine_t *engine, const sgfnd_prompt_t *prompt, sgfnd_tile_t *tile, uint32_t tile_x, uint32_t tile_y);

sgfnd_image_t* sgfnd_generate_image(sgfnd_engine_t *engine, const sgfnd_prompt_t *prompt);

int sgfnd_image_save_raw(const sgfnd_image_t *img, const char *path);
int sgfnd_image_load_raw(sgfnd_image_t *img, const char *path);
int sgfnd_image_save_tiled_raw(const sgfnd_image_t *img, const char *path);
int sgfnd_image_load_tiled_raw(sgfnd_image_t *img, const char *path);

sgfnd_color_grader_t* sgfnd_color_grader_create(void);
void sgfnd_color_grader_destroy(sgfnd_color_grader_t *grader);
int sgfnd_color_grader_train(sgfnd_color_grader_t *grader, const sgfnd_image_t *img);
sgfnd_color_t sgfnd_color_grader_map(const sgfnd_color_grader_t *grader, const sgfnd_color_t color, const char *component);

typedef struct sgfnd_training_bot sgfnd_training_bot_t;
sgfnd_training_bot_t* sgfnd_training_bot_create(const char *storage_path);
void sgfnd_training_bot_destroy(sgfnd_training_bot_t *bot);
int sgfnd_training_bot_fetch_and_process(sgfnd_training_bot_t *bot, const char *url, sgfnd_color_grader_t *grader);
int sgfnd_training_bot_save_metadata(sgfnd_training_bot_t *bot, const char *category, const char *key, const char *value);
const char* sgfnd_training_bot_get_metadata(sgfnd_training_bot_t *bot, const char *category, const char *key);

sgfnd_latent_t* sgfnd_latent_create(size_t dim);
void sgfnd_latent_destroy(sgfnd_latent_t *latent);
void sgfnd_latent_sample(sgfnd_latent_t *latent);
void sgfnd_latent_encode(sgfnd_latent_t *latent, const float *input, size_t input_dim);
void sgfnd_latent_decode(const sgfnd_latent_t *latent, float *output, size_t output_dim);

sgfnd_linear_t* sgfnd_linear_create(size_t in_dim, size_t out_dim, bool bias);
void sgfnd_linear_destroy(sgfnd_linear_t *layer);
void sgfnd_linear_forward(const sgfnd_linear_t *layer, const float *input, float *output);
void sgfnd_linear_backward(sgfnd_linear_t *layer, const float *input, const float *grad_output, float *grad_input, float lr);

sgfnd_mlp_t* sgfnd_mlp_create(const size_t *layer_dims, size_t num_layers);
void sgfnd_mlp_destroy(sgfnd_mlp_t *mlp);
void sgfnd_mlp_forward(const sgfnd_mlp_t *mlp, const float *input, float *output);
void sgfnd_mlp_backward(sgfnd_mlp_t *mlp, const float *input, const float *target, float lr);

sgfnd_diffusion_t* sgfnd_diffusion_create(int num_timesteps, size_t latent_dim, size_t cond_dim);
void sgfnd_diffusion_destroy(sgfnd_diffusion_t *diff);
void sgfnd_diffusion_forward(const sgfnd_diffusion_t *diff, const float *x_t, int t, const float *cond, float *eps_pred);
void sgfnd_diffusion_sample(sgfnd_diffusion_t *diff, const float *cond, float *out_latent, int steps);
float sgfnd_diffusion_loss(const sgfnd_diffusion_t *diff, const float *x_0, const float *cond, int t);

sgfnd_model_t* sgfnd_model_create(size_t latent_dim, size_t cond_dim, int timesteps, float lr);
void sgfnd_model_destroy(sgfnd_model_t *model);
int sgfnd_model_train_step(sgfnd_model_t *model, const sgfnd_image_t *img, const sgfnd_prompt_t *prompt, sgfnd_training_step_t *step_info);
int sgfnd_model_generate(sgfnd_model_t *model, const sgfnd_prompt_t *prompt, sgfnd_image_t *out_img, int steps);
int sgfnd_model_save_weights(const sgfnd_model_t *model, const char *path);
int sgfnd_model_load_weights(sgfnd_model_t *model, const char *path);
int sgfnd_model_save_full(const sgfnd_model_t *model, const char *path);
int sgfnd_model_load_full(sgfnd_model_t *model, const char *path);

sgfnd_prompt_t* sgfnd_prompt_create_from_text(const char *text, float weight);
sgfnd_prompt_t* sgfnd_prompt_create_from_tags(const char **tags, const float *weights, size_t count);
void sgfnd_prompt_destroy(sgfnd_prompt_t *prompt);

sgfnd_image_t* sgfnd_image_load_from_file(const char *path, uint32_t max_dim);
int sgfnd_image_resize(sgfnd_image_t *img, uint32_t new_width, uint32_t new_height);
void sgfnd_image_normalize(sgfnd_image_t *img, float mean, float std);

sgfnd_dataset_t* sgfnd_dataset_create(uint32_t width, uint32_t height, size_t capacity);
void sgfnd_dataset_destroy(sgfnd_dataset_t *dataset);
int sgfnd_dataset_add(sgfnd_dataset_t *dataset, const sgfnd_image_t *img, const sgfnd_latent_t *latent, const char *prompt, const char *metadata);
int sgfnd_dataset_get_batch(const sgfnd_dataset_t *dataset, size_t batch_size, sgfnd_image_t **out_imgs, sgfnd_latent_t **out_latents, char ***out_prompts);

typedef enum {
    SGFND_NSFW_FILTER_DISABLED = 0,
    SGFND_NSFW_FILTER_ENABLED = 1,
    SGFND_NSFW_FILTER_STRICT = 2
} sgfnd_nsfw_mode_t;

typedef struct {
    sgfnd_nsfw_mode_t mode;
    float threshold;
    size_t blocked_prompts;
    size_t flagged_images;
    size_t sanitized_images;
} sgfnd_nsfw_filter_t;

sgfnd_nsfw_filter_t* sgfnd_nsfw_filter_create(sgfnd_nsfw_mode_t mode, float threshold);
void sgfnd_nsfw_filter_destroy(sgfnd_nsfw_filter_t *filter);

bool sgfnd_nsfw_check_prompt(sgfnd_nsfw_filter_t *filter, const sgfnd_prompt_t *prompt);
float sgfnd_nsfw_check_image(sgfnd_nsfw_filter_t *filter, const sgfnd_image_t *img);
int sgfnd_nsfw_sanitize_image(sgfnd_nsfw_filter_t *filter, sgfnd_image_t *img);

typedef struct sgfnd_training_bot sgfnd_training_bot_t;
sgfnd_training_bot_t* sgfnd_training_bot_create_v2(const char *storage_path, sgfnd_dataset_t *dataset);
void sgfnd_training_bot_destroy_v2(sgfnd_training_bot_t *bot);
int sgfnd_training_bot_fetch_and_train(sgfnd_training_bot_t *bot, const char *url, sgfnd_model_t *model, sgfnd_color_grader_t *grader);
int sgfnd_training_bot_augment_dataset(sgfnd_training_bot_t *bot, sgfnd_dataset_t *dataset);

size_t sgfnd_training_bot_get_fetched(const sgfnd_training_bot_t *bot);
size_t sgfnd_training_bot_get_processed(const sgfnd_training_bot_t *bot);
float sgfnd_training_bot_get_avg_loss(const sgfnd_training_bot_t *bot);
sgfnd_dataset_t* sgfnd_training_bot_get_dataset(const sgfnd_training_bot_t *bot);

#endif