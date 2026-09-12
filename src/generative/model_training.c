#define _GNU_SOURCE
#include "sgfnd_core.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdio.h>

sgfnd_model_t* sgfnd_model_create(size_t latent_dim, size_t cond_dim, int timesteps, float lr) {
    sgfnd_model_t *model = calloc(1, sizeof(sgfnd_model_t));
    if (!model) return NULL;

    model->learning_rate = lr;
    model->loss_type = SGFND_LOSS_MSE;
    model->step_count = 0;

    model->diffusion = sgfnd_diffusion_create(timesteps, latent_dim, cond_dim);
    model->latent = sgfnd_latent_create(latent_dim);

    size_t encoder_dims[] = {latent_dim * 2, 512, 512, latent_dim * 2};
    model->encoder = sgfnd_mlp_create(encoder_dims, 4);

    size_t decoder_dims[] = {latent_dim, 512, 512, latent_dim * 4};
    model->decoder = sgfnd_mlp_create(decoder_dims, 4);

    size_t disc_dims[] = {latent_dim * 4, 512, 256, 1};
    model->discriminator = sgfnd_mlp_create(disc_dims, 4);

    size_t opt_state_size = 0;
    opt_state_size += latent_dim;
    opt_state_size += latent_dim * 2;
    opt_state_size += 512 * 512 * 3;
    model->optimizer_state = calloc(opt_state_size, sizeof(float));

    return model;
}

void sgfnd_model_destroy(sgfnd_model_t *model) {
    if (!model) return;
    sgfnd_diffusion_destroy(model->diffusion);
    sgfnd_latent_destroy(model->latent);
    sgfnd_mlp_destroy(model->encoder);
    sgfnd_mlp_destroy(model->decoder);
    sgfnd_mlp_destroy(model->discriminator);
    free(model->optimizer_state);
    free(model);
}

static void image_to_latent_input(const sgfnd_image_t *img, float *latent_in) {
    if (!img || !latent_in) return;
    size_t pixel_count = img->width * img->height * img->channels;
    size_t latent_in_dim = 512;
    for (size_t i = 0; i < latent_in_dim; i++) {
        latent_in[i] = 0.0f;
    }
    size_t copy = pixel_count < latent_in_dim ? pixel_count : latent_in_dim;
    if (img->full_pixels) {
        for (size_t i = 0; i < copy; i++) {
            latent_in[i] = img->full_pixels[i];
        }
    }
}

static float tag_hash(const char *tag) {
    unsigned int hash = 5381;
    for (const char *p = tag; *p; p++) {
        hash = ((hash << 5) + hash) + (unsigned char)*p;
    }
    return (hash % 10000) / 10000.0f;
}

static void prompt_to_condition(const sgfnd_prompt_t *prompt, float *cond, size_t cond_dim) {
    if (!prompt || !cond) return;
    for (size_t i = 0; i < cond_dim; i++) {
        cond[i] = 0.0f;
    }
    if (!prompt->tags) return;
    
    size_t tags_to_use = prompt->count < cond_dim ? prompt->count : cond_dim;
    for (size_t i = 0; i < tags_to_use; i++) {
        if (prompt->tags[i]) {
            cond[i] = prompt->weights[i] * tag_hash(prompt->tags[i]);
        } else {
            cond[i] = prompt->weights[i] * (1.0f / (i + 1));
        }
    }
    if (prompt->count < cond_dim) {
        for (size_t i = prompt->count; i < cond_dim; i++) {
            cond[i] = 0.0f;
        }
    }
}

int sgfnd_model_train_step(sgfnd_model_t *model, const sgfnd_image_t *img, const sgfnd_prompt_t *prompt, sgfnd_training_step_t *step_info) {
    if (!model || !img) return -1;

    size_t latent_dim = model->latent->dim;
    size_t cond_dim = model->diffusion->cond_embed->layer_dims[0];

    float *latent_input = calloc(latent_dim * 2, sizeof(float));
    float *cond = calloc(cond_dim, sizeof(float));
    float *recon = calloc(latent_dim * 4, sizeof(float));
    float *disc_out = calloc(1, sizeof(float));
    if (!latent_input || !cond || !recon || !disc_out) {
        free(latent_input);
        free(cond);
        free(recon);
        free(disc_out);
        return -1;
    }

    image_to_latent_input(img, latent_input);
    // Initialize logvar part (second half) to -1.0 for stable KL
    for (size_t i = 0; i < latent_dim; i++) {
        latent_input[latent_dim + i] = -1.0f;
    }
    prompt_to_condition(prompt, cond, cond_dim);

    sgfnd_mlp_forward(model->encoder, latent_input, model->latent->mu);
    for (size_t i = 0; i < latent_dim; i++) {
        model->latent->logvar[i] = latent_input[latent_dim + i];
    }
    sgfnd_latent_sample(model->latent);

    int t = rand() % model->diffusion->num_timesteps;
    float diff_loss = sgfnd_diffusion_loss(model->diffusion, model->latent->latent, cond, t);

    sgfnd_mlp_forward(model->decoder, model->latent->latent, recon);

    float recon_loss = 0.0f;
    for (size_t i = 0; i < latent_dim * 4; i++) {
        float target = (i < latent_dim * 2) ? latent_input[i] : 0.0f;
        float diff = recon[i] - target;
        recon_loss += diff * diff;
    }
    recon_loss /= (latent_dim * 4);

    float kl_loss = 0.0f;
    for (size_t i = 0; i < latent_dim; i++) {
        float logvar = fmaxf(-10.0f, fminf(10.0f, model->latent->logvar[i]));
        float mu_sq = model->latent->mu[i] * model->latent->mu[i];
        kl_loss += -0.5f * (1.0f + logvar - mu_sq - expf(logvar));
    }
    kl_loss /= latent_dim;

    sgfnd_mlp_forward(model->discriminator, recon, disc_out);
    float disc_val = fmaxf(1e-8f, fminf(1.0f - 1e-8f, disc_out[0]));
    float adv_loss = -logf(disc_val);

    float total_loss = diff_loss + recon_loss + 0.1f * kl_loss + 0.01f * adv_loss;

    // Clamp total loss to prevent instability
    if (isnan(total_loss) || isinf(total_loss)) {
        fprintf(stderr, "[TRAIN] NaN/Inf detected: diff=%.4f recon=%.4f kl=%.4f adv=%.4f\n", diff_loss, recon_loss, kl_loss, adv_loss);
        total_loss = 1e6f;
    } else if (total_loss > 1e6f) {
        fprintf(stderr, "[TRAIN] High loss clamped: diff=%.4f recon=%.4f kl=%.4f adv=%.4f\n", diff_loss, recon_loss, kl_loss, adv_loss);
        total_loss = 1e6f;
    }

    float *grad_recon = calloc(latent_dim * 4, sizeof(float));
    for (size_t i = 0; i < latent_dim * 4; i++) {
        float target = (i < latent_dim * 2) ? latent_input[i] : 0.0f;
        float diff = recon[i] - target;
        // Clip gradient
        if (diff > 100.0f) diff = 100.0f;
        if (diff < -100.0f) diff = -100.0f;
        grad_recon[i] = 2.0f * diff / (latent_dim * 4);
    }
    sgfnd_mlp_backward(model->decoder, model->latent->latent, grad_recon, model->learning_rate);

    float *grad_disc = calloc(1, sizeof(float));
    // Clamp discriminator gradient to prevent explosion
    float disc_grad = -1.0f / fmaxf(1e-4f, fminf(1.0f - 1e-4f, disc_out[0]));
    if (disc_grad > 100.0f) disc_grad = 100.0f;
    if (disc_grad < -100.0f) disc_grad = -100.0f;
    grad_disc[0] = disc_grad;
    sgfnd_mlp_backward(model->discriminator, recon, grad_disc, model->learning_rate);

    free(latent_input);
    free(cond);
    free(recon);
    free(disc_out);
    free(grad_recon);
    free(grad_disc);

    model->step_count++;

    if (step_info) {
        step_info->loss = total_loss;
        step_info->recon_loss = recon_loss;
        step_info->kl_loss = kl_loss;
        step_info->adv_loss = adv_loss;
        step_info->grad_norm = sqrtf(total_loss);
        step_info->timestamp = (uint64_t)time(NULL);
    }

    return 0;
}

int sgfnd_model_generate(sgfnd_model_t *model, const sgfnd_prompt_t *prompt, sgfnd_image_t *out_img, int steps) {
    if (!model || !prompt || !out_img) return -1;

    size_t latent_dim = model->latent->dim;
    size_t cond_dim = model->diffusion->cond_embed->layer_dims[0];
    size_t decoder_out_dim = model->decoder->layer_dims[model->decoder->num_layers];

    float *cond = calloc(cond_dim, sizeof(float));
    float *latent = calloc(latent_dim, sizeof(float));
    if (!cond || !latent) {
        free(cond);
        free(latent);
        return -1;
    }

    prompt_to_condition(prompt, cond, cond_dim);

    sgfnd_diffusion_sample(model->diffusion, cond, latent, steps);

    float *decoded = calloc(decoder_out_dim, sizeof(float));
    if (!decoded) {
        free(cond);
        free(latent);
        return -1;
    }
    sgfnd_mlp_forward(model->decoder, latent, decoded);

    uint32_t w = out_img->width;
    uint32_t h = out_img->height;
    uint8_t c = out_img->channels;
    size_t pixels_per_channel = w * h;
    size_t total_pixels = w * h * c;

    if (!out_img->full_pixels) {
        out_img->full_pixels = calloc(total_pixels, sizeof(float));
        if (!out_img->full_pixels) {
            free(cond);
            free(latent);
            free(decoded);
            return -1;
        }
    }

    size_t feature_per_pixel = decoder_out_dim / pixels_per_channel;
    if (feature_per_pixel == 0) feature_per_pixel = 1;
    if (feature_per_pixel > 4) feature_per_pixel = 4;

    for (uint32_t y = 0; y < h; y++) {
        for (uint32_t x = 0; x < w; x++) {
            size_t pixel_idx = y * w + x;
            size_t decoded_idx = pixel_idx * feature_per_pixel;
            size_t out_idx = (y * w + x) * c;

            float r = (decoded_idx < decoder_out_dim) ? decoded[decoded_idx] : 0.0f;
            float g = (decoded_idx + 1 < decoder_out_dim) ? decoded[decoded_idx + 1] : r * 0.85f;
            float b = (decoded_idx + 2 < decoder_out_dim) ? decoded[decoded_idx + 2] : r * 0.7f;

            r = fmaxf(0.0f, fminf(1.0f, r));
            g = fmaxf(0.0f, fminf(1.0f, g));
            b = fmaxf(0.0f, fminf(1.0f, b));

            out_img->full_pixels[out_idx + 0] = r;
            out_img->full_pixels[out_idx + 1] = g;
            out_img->full_pixels[out_idx + 2] = b;
            if (c > 3) out_img->full_pixels[out_idx + 3] = 1.0f;
        }
    }

    free(cond);
    free(latent);
    free(decoded);
    return 0;
}

int sgfnd_model_save_weights(const sgfnd_model_t *model, const char *path) {
    if (!model || !path) return -1;

    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    uint32_t magic = 0x5347464E;
    fwrite(&magic, sizeof(uint32_t), 1, f);

    uint32_t version = 1;
    fwrite(&version, sizeof(uint32_t), 1, f);

    size_t latent_dim = model->latent->dim;
    fwrite(&latent_dim, sizeof(size_t), 1, f);

    fwrite(model->latent->mu, sizeof(float), latent_dim, f);
    fwrite(model->latent->logvar, sizeof(float), latent_dim, f);

    fclose(f);
    return 0;
}

int sgfnd_model_load_weights(sgfnd_model_t *model, const char *path) {
    if (!model || !path) return -1;

    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    uint32_t magic, version;
    fread(&magic, sizeof(uint32_t), 1, f);
    fread(&version, sizeof(uint32_t), 1, f);

    if (magic != 0x5347464E) {
        fclose(f);
        return -1;
    }

    size_t latent_dim;
    fread(&latent_dim, sizeof(size_t), 1, f);

    if (latent_dim != model->latent->dim) {
        fclose(f);
        return -1;
    }

    fread(model->latent->mu, sizeof(float), latent_dim, f);
    fread(model->latent->logvar, sizeof(float), latent_dim, f);

    fclose(f);
    return 0;
}

sgfnd_dataset_t* sgfnd_dataset_create(uint32_t width, uint32_t height, size_t capacity) {
    sgfnd_dataset_t *dataset = calloc(1, sizeof(sgfnd_dataset_t));
    if (!dataset) return NULL;

    dataset->capacity = capacity;
    dataset->img_width = width;
    dataset->img_height = height;
    dataset->images = calloc(capacity, sizeof(sgfnd_image_t*));
    dataset->latents = calloc(capacity, sizeof(sgfnd_latent_t*));
    dataset->prompts = calloc(capacity, sizeof(char*));
    dataset->metadata = calloc(capacity, sizeof(char*));

    if (!dataset->images || !dataset->latents || !dataset->prompts || !dataset->metadata) {
        sgfnd_dataset_destroy(dataset);
        return NULL;
    }
    return dataset;
}

void sgfnd_dataset_destroy(sgfnd_dataset_t *dataset) {
    if (!dataset) return;
    for (size_t i = 0; i < dataset->count; i++) {
        sgfnd_image_free(dataset->images[i]);
        sgfnd_latent_destroy(dataset->latents[i]);
        free(dataset->prompts[i]);
        free(dataset->metadata[i]);
    }
    free(dataset->images);
    free(dataset->latents);
    free(dataset->prompts);
    free(dataset->metadata);
    free(dataset);
}

int sgfnd_dataset_add(sgfnd_dataset_t *dataset, const sgfnd_image_t *img, const sgfnd_latent_t *latent, const char *prompt, const char *metadata) {
    if (!dataset || dataset->count >= dataset->capacity) return -1;

    dataset->images[dataset->count] = sgfnd_image_create_tiled(img->width, img->height, img->channels, 64);
    if (!dataset->images[dataset->count]) return -1;
    memcpy(dataset->images[dataset->count]->full_pixels, img->full_pixels, img->width * img->height * img->channels * sizeof(float));

    dataset->latents[dataset->count] = sgfnd_latent_create(latent->dim);
    if (!dataset->latents[dataset->count]) return -1;
    memcpy(dataset->latents[dataset->count]->latent, latent->latent, latent->dim * sizeof(float));
    memcpy(dataset->latents[dataset->count]->mu, latent->mu, latent->dim * sizeof(float));
    memcpy(dataset->latents[dataset->count]->logvar, latent->logvar, latent->dim * sizeof(float));

    dataset->prompts[dataset->count] = prompt ? strdup(prompt) : NULL;
    dataset->metadata[dataset->count] = metadata ? strdup(metadata) : NULL;

    dataset->count++;
    return 0;
}

int sgfnd_dataset_get_batch(const sgfnd_dataset_t *dataset, size_t batch_size, sgfnd_image_t **out_imgs, sgfnd_latent_t **out_latents, char ***out_prompts) {
    if (!dataset || dataset->count == 0) return -1;
    if (batch_size > dataset->count) batch_size = dataset->count;

    for (size_t i = 0; i < batch_size; i++) {
        size_t idx = rand() % dataset->count;
        out_imgs[i] = dataset->images[idx];
        out_latents[i] = dataset->latents[idx];
        out_prompts[i] = &dataset->prompts[idx];
    }
    return batch_size;
}

int sgfnd_model_save_full(const sgfnd_model_t *model, const char *path) {
    if (!model || !path) return -1;

    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    uint32_t magic = 0x5347464E;
    fwrite(&magic, sizeof(uint32_t), 1, f);

    uint32_t version = 2;
    fwrite(&version, sizeof(uint32_t), 1, f);

    size_t latent_dim = model->latent->dim;
    fwrite(&latent_dim, sizeof(size_t), 1, f);

    fwrite(model->latent->mu, sizeof(float), latent_dim, f);
    fwrite(model->latent->logvar, sizeof(float), latent_dim, f);

    size_t step_count = model->step_count;
    fwrite(&step_count, sizeof(size_t), 1, f);
    fwrite(&model->learning_rate, sizeof(float), 1, f);

    size_t opt_state_size = 0;
    opt_state_size += latent_dim;
    opt_state_size += latent_dim * 2;
    opt_state_size += 512 * 512 * 3;
    fwrite(&opt_state_size, sizeof(size_t), 1, f);
    if (model->optimizer_state) {
        fwrite(model->optimizer_state, sizeof(float), opt_state_size, f);
    }

    fclose(f);
    return 0;
}

int sgfnd_model_load_full(sgfnd_model_t *model, const char *path) {
    if (!model || !path) return -1;

    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    uint32_t magic, version;
    fread(&magic, sizeof(uint32_t), 1, f);
    fread(&version, sizeof(uint32_t), 1, f);

    if (magic != 0x5347464E) {
        fclose(f);
        return -1;
    }

    size_t latent_dim;
    fread(&latent_dim, sizeof(size_t), 1, f);

    if (latent_dim != model->latent->dim) {
        fclose(f);
        return -1;
    }

    fread(model->latent->mu, sizeof(float), latent_dim, f);
    fread(model->latent->logvar, sizeof(float), latent_dim, f);

    size_t step_count;
    fread(&step_count, sizeof(size_t), 1, f);
    fread(&model->learning_rate, sizeof(float), 1, f);
    model->step_count = step_count;

    size_t opt_state_size;
    fread(&opt_state_size, sizeof(size_t), 1, f);
    if (model->optimizer_state && opt_state_size > 0) {
        fread(model->optimizer_state, sizeof(float), opt_state_size, f);
    }

    fclose(f);
    return 0;
}

sgfnd_prompt_t* sgfnd_prompt_create_from_text(const char *text, float weight) {
    if (!text) return NULL;

    sgfnd_prompt_t *prompt = calloc(1, sizeof(sgfnd_prompt_t));
    if (!prompt) return NULL;

    char *text_copy = strdup(text);
    if (!text_copy) {
        free(prompt);
        return NULL;
    }

    size_t count = 1;
    for (char *p = text_copy; *p; p++) {
        if (*p == ',' || *p == ';') count++;
    }

    prompt->tags = calloc(count, sizeof(char*));
    prompt->weights = calloc(count, sizeof(float));
    if (!prompt->tags || !prompt->weights) {
        free(text_copy);
        free(prompt->tags);
        free(prompt->weights);
        free(prompt);
        return NULL;
    }

    prompt->count = count;
    char *saveptr = NULL;
    char *token = strtok_r(text_copy, ",;", &saveptr);
    for (size_t i = 0; i < count && token; i++) {
        while (*token == ' ' || *token == '\t') token++;
        char *end = token + strlen(token) - 1;
        while (end > token && (*end == ' ' || *end == '\t')) *end-- = '\0';
        prompt->tags[i] = strdup(token);
        prompt->weights[i] = weight;
        token = strtok_r(NULL, ",;", &saveptr);
    }
    free(text_copy);
    return prompt;
}

sgfnd_prompt_t* sgfnd_prompt_create_from_tags(const char **tags, const float *weights, size_t count) {
    if (!tags || !weights || count == 0) return NULL;

    sgfnd_prompt_t *prompt = calloc(1, sizeof(sgfnd_prompt_t));
    if (!prompt) return NULL;

    prompt->tags = calloc(count, sizeof(char*));
    prompt->weights = calloc(count, sizeof(float));
    if (!prompt->tags || !prompt->weights) {
        free(prompt->tags);
        free(prompt->weights);
        free(prompt);
        return NULL;
    }

    prompt->count = count;
    for (size_t i = 0; i < count; i++) {
        prompt->tags[i] = tags[i] ? strdup(tags[i]) : NULL;
        prompt->weights[i] = weights[i];
    }
    return prompt;
}

void sgfnd_prompt_destroy(sgfnd_prompt_t *prompt) {
    if (!prompt) return;
    for (size_t i = 0; i < prompt->count; i++) {
        free(prompt->tags[i]);
    }
    free(prompt->tags);
    free(prompt->weights);
    free(prompt);
}

#include "stb_image.h"

sgfnd_image_t* sgfnd_image_load_from_file(const char *path, uint32_t max_dim) {
    if (!path) return NULL;

    int w, h, c;
    unsigned char *data = stbi_load(path, &w, &h, &c, 4);
    if (!data) return NULL;

    float scale = 1.0f;
    if (max_dim > 0 && (w > max_dim || h > max_dim)) {
        scale = (float)max_dim / (w > h ? w : h);
    }

    uint32_t new_w = (uint32_t)(w * scale);
    uint32_t new_h = (uint32_t)(h * scale);

    sgfnd_image_t *img = sgfnd_image_create_tiled(new_w, new_h, 4, 64);
    if (!img) {
        stbi_image_free(data);
        return NULL;
    }

    for (uint32_t y = 0; y < new_h; y++) {
        for (uint32_t x = 0; x < new_w; x++) {
            uint32_t src_x = (uint32_t)(x / scale);
            uint32_t src_y = (uint32_t)(y / scale);
            if (src_x >= (uint32_t)w) src_x = w - 1;
            if (src_y >= (uint32_t)h) src_y = h - 1;

            size_t src_idx = (src_y * w + src_x) * 4;
            size_t dst_idx = (y * new_w + x) * 4;

            img->full_pixels[dst_idx + 0] = data[src_idx + 0] / 255.0f;
            img->full_pixels[dst_idx + 1] = data[src_idx + 1] / 255.0f;
            img->full_pixels[dst_idx + 2] = data[src_idx + 2] / 255.0f;
            img->full_pixels[dst_idx + 3] = data[src_idx + 3] / 255.0f;
        }
    }

    stbi_image_free(data);
    return img;
}

int sgfnd_image_resize(sgfnd_image_t *img, uint32_t new_width, uint32_t new_height) {
    if (!img || new_width == 0 || new_height == 0) return -1;
    if (img->width == new_width && img->height == new_height) return 0;

    float *new_pixels = calloc(new_width * new_height * img->channels, sizeof(float));
    if (!new_pixels) return -1;

    float scale_x = (float)img->width / new_width;
    float scale_y = (float)img->height / new_height;

    for (uint32_t y = 0; y < new_height; y++) {
        for (uint32_t x = 0; x < new_width; x++) {
            float src_x = x * scale_x;
            float src_y = y * scale_y;
            uint32_t x0 = (uint32_t)src_x;
            uint32_t y0 = (uint32_t)src_y;
            uint32_t x1 = (x0 + 1 < img->width) ? x0 + 1 : x0;
            uint32_t y1 = (y0 + 1 < img->height) ? y0 + 1 : y0;
            float fx = src_x - x0;
            float fy = src_y - y0;

            for (uint8_t c = 0; c < img->channels; c++) {
                float v00 = img->full_pixels[(y0 * img->width + x0) * img->channels + c];
                float v10 = img->full_pixels[(y0 * img->width + x1) * img->channels + c];
                float v01 = img->full_pixels[(y1 * img->width + x0) * img->channels + c];
                float v11 = img->full_pixels[(y1 * img->width + x1) * img->channels + c];

                float v0 = v00 + fx * (v10 - v00);
                float v1 = v01 + fx * (v11 - v01);
                float v = v0 + fy * (v1 - v0);

                new_pixels[(y * new_width + x) * img->channels + c] = v;
            }
        }
    }

    free(img->full_pixels);
    img->full_pixels = new_pixels;
    img->width = new_width;
    img->height = new_height;
    return 0;
}

void sgfnd_image_normalize(sgfnd_image_t *img, float mean, float std) {
    if (!img || !img->full_pixels) return;
    size_t total = img->width * img->height * img->channels;
    for (size_t i = 0; i < total; i++) {
        img->full_pixels[i] = (img->full_pixels[i] - mean) / std;
    }
}

int sgfnd_image_validate_for_training(const sgfnd_image_t *img) {
    if (!img) {
        fprintf(stderr, "[VALIDATION] Image is NULL\n");
        return 0;
    }

    if (!img->full_pixels) {
        fprintf(stderr, "[VALIDATION] Image has no pixel data\n");
        return 0;
    }

    if (img->width == 0 || img->height == 0) {
        fprintf(stderr, "[VALIDATION] Image has zero dimensions: %ux%u\n", img->width, img->height);
        return 0;
    }

    if (img->channels < 3 || img->channels > 4) {
        fprintf(stderr, "[VALIDATION] Invalid channel count: %u (expected 3 or 4)\n", img->channels);
        return 0;
    }

    size_t total = img->width * img->height * img->channels;
    
    float min_val = 1.0f, max_val = 0.0f, sum = 0.0f;
    float first_val = img->full_pixels[0];
    int all_same = 1;
    
    for (size_t i = 0; i < total; i++) {
        float v = img->full_pixels[i];
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
        sum += v;
        if (i > 0 && fabsf(v - first_val) > 1e-6f) {
            all_same = 0;
        }
    }
    
    float mean = sum / total;
    
    if (all_same) {
        fprintf(stderr, "[VALIDATION] Image is uniform (all pixels = %.4f)\n", first_val);
        return 0;
    }
    
    if (max_val <= 0.0f) {
        fprintf(stderr, "[VALIDATION] Image has no positive values (max = %.4f)\n", max_val);
        return 0;
    }
    
    float range = max_val - min_val;
    if (range < 0.01f) {
        fprintf(stderr, "[VALIDATION] Image has very low dynamic range: %.4f\n", range);
        return 0;
    }
    
    if (min_val < -10.0f || max_val > 10.0f) {
        fprintf(stderr, "[VALIDATION] Image values out of expected range [min=%.4f, max=%.4f]\n", min_val, max_val);
        return 0;
    }
    
    fprintf(stderr, "[VALIDATION] Image OK: %ux%u, channels=%u, range=[%.4f, %.4f], mean=%.4f\n",
            img->width, img->height, img->channels, min_val, max_val, mean);
    
    return 1;
}