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
    for (size_t i = 0; i < pixel_count && i < latent_in_dim; i++) {
        latent_in[i] = img->full_pixels ? img->full_pixels[i] : 0.0f;
    }
}

static void prompt_to_condition(const sgfnd_prompt_t *prompt, float *cond, size_t cond_dim) {
    if (!prompt || !cond) return;
    for (size_t i = 0; i < cond_dim; i++) {
        cond[i] = 0.0f;
    }
    for (size_t i = 0; i < prompt->count && i < cond_dim; i++) {
        cond[i] = prompt->weights[i];
    }
}

int sgfnd_model_train_step(sgfnd_model_t *model, const sgfnd_image_t *img, const sgfnd_prompt_t *prompt, sgfnd_training_step_t *step_info) {
    if (!model || !img) return -1;

    size_t latent_dim = model->latent->dim;
    size_t cond_dim = model->diffusion->cond_embed->layer_dims[0];

    float *latent_input = malloc(latent_dim * 2 * sizeof(float));
    float *cond = malloc(cond_dim * sizeof(float));
    float *recon = malloc(latent_dim * 4 * sizeof(float));
    float *disc_out = malloc(sizeof(float));

    image_to_latent_input(img, latent_input);
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
        kl_loss += -0.5f * (1.0f + model->latent->logvar[i] - model->latent->mu[i] * model->latent->mu[i] - expf(model->latent->logvar[i]));
    }
    kl_loss /= latent_dim;

    sgfnd_mlp_forward(model->discriminator, recon, disc_out);
    float adv_loss = -logf(fmaxf(1e-8f, disc_out[0]));

    float total_loss = diff_loss + recon_loss + 0.1f * kl_loss + 0.01f * adv_loss;

    float *grad_recon = malloc(latent_dim * 4 * sizeof(float));
    for (size_t i = 0; i < latent_dim * 4; i++) {
        float target = (i < latent_dim * 2) ? latent_input[i] : 0.0f;
        grad_recon[i] = 2.0f * (recon[i] - target) / (latent_dim * 4);
    }
    sgfnd_mlp_backward(model->decoder, model->latent->latent, grad_recon, model->learning_rate);

    float *grad_disc = malloc(sizeof(float));
    grad_disc[0] = -1.0f / fmaxf(1e-8f, disc_out[0]);
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

    float *cond = malloc(cond_dim * sizeof(float));
    float *latent = malloc(latent_dim * sizeof(float));

    prompt_to_condition(prompt, cond, cond_dim);

    sgfnd_diffusion_sample(model->diffusion, cond, latent, steps);

    size_t decoder_out_dim = latent_dim * 4;
    float *decoded = malloc(decoder_out_dim * sizeof(float));
    sgfnd_mlp_forward(model->decoder, latent, decoded);

    uint32_t w = out_img->width;
    uint32_t h = out_img->height;
    uint8_t c = out_img->channels;

    if (!out_img->full_pixels) {
        out_img->full_pixels = calloc(w * h * c, sizeof(float));
    }

    for (uint32_t y = 0; y < h; y++) {
        for (uint32_t x = 0; x < w; x++) {
            float u = (float)x / w;
            float v = (float)y / h;

            size_t idx = (y * w + x) * c;
            float base = decoded[(size_t)((u + v * 0.1f) * decoder_out_dim) % decoder_out_dim];
            base = fmaxf(0.0f, fminf(1.0f, base));

            out_img->full_pixels[idx + 0] = base;
            out_img->full_pixels[idx + 1] = base * 0.9f;
            out_img->full_pixels[idx + 2] = base * 0.8f;
            out_img->full_pixels[idx + 3] = 1.0f;
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