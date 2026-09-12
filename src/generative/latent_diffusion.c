#include "sgfnd_core.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define M_PI 3.14159265358979323846f

sgfnd_latent_t* sgfnd_latent_create(size_t dim) {
    sgfnd_latent_t *latent = calloc(1, sizeof(sgfnd_latent_t));
    if (!latent) return NULL;

    latent->dim = dim;
    latent->latent = calloc(dim, sizeof(float));
    latent->mu = calloc(dim, sizeof(float));
    latent->logvar = calloc(dim, sizeof(float));
    latent->eps = calloc(dim, sizeof(float));
    if (!latent->latent || !latent->mu || !latent->logvar || !latent->eps) {
        sgfnd_latent_destroy(latent);
        return NULL;
    }
    return latent;
}

void sgfnd_latent_destroy(sgfnd_latent_t *latent) {
    if (!latent) return;
    free(latent->latent);
    free(latent->mu);
    free(latent->logvar);
    free(latent->eps);
    free(latent);
}

static float randn() {
    static bool has_spare = false;
    static float spare;
    if (has_spare) {
        has_spare = false;
        return spare;
    }
    float u = (rand() + 1.0f) / (RAND_MAX + 2.0f);
    float v = (rand() + 1.0f) / (RAND_MAX + 2.0f);
    float mag = sqrtf(-2.0f * logf(u));
    spare = mag * cosf(2.0f * M_PI * v);
    has_spare = true;
    return mag * sinf(2.0f * M_PI * v);
}

void sgfnd_latent_sample(sgfnd_latent_t *latent) {
    if (!latent) return;
    for (size_t i = 0; i < latent->dim; i++) {
        latent->eps[i] = randn();
        latent->latent[i] = latent->mu[i] + expf(0.5f * latent->logvar[i]) * latent->eps[i];
    }
}

void sgfnd_latent_encode(sgfnd_latent_t *latent, const float *input, size_t input_dim) {
    if (!latent || !input) return;
    size_t copy = latent->dim < input_dim ? latent->dim : input_dim;
    for (size_t i = 0; i < copy; i++) {
        latent->mu[i] = input[i];
        latent->logvar[i] = -1.0f;
    }
    for (size_t i = copy; i < latent->dim; i++) {
        latent->mu[i] = 0.0f;
        latent->logvar[i] = -1.0f;
    }
    sgfnd_latent_sample(latent);
}

void sgfnd_latent_decode(const sgfnd_latent_t *latent, float *output, size_t output_dim) {
    if (!latent || !output) return;
    size_t copy = latent->dim < output_dim ? latent->dim : output_dim;
    memcpy(output, latent->latent, copy * sizeof(float));
    for (size_t i = copy; i < output_dim; i++) {
        output[i] = 0.0f;
    }
}

sgfnd_linear_t* sgfnd_linear_create(size_t in_dim, size_t out_dim, bool bias) {
    sgfnd_linear_t *layer = calloc(1, sizeof(sgfnd_linear_t));
    if (!layer) return NULL;

    layer->in_dim = in_dim;
    layer->out_dim = out_dim;
    layer->use_bias = bias;

    layer->weights = calloc(in_dim * out_dim, sizeof(float));
    layer->grad_w = calloc(in_dim * out_dim, sizeof(float));
    if (!layer->weights || !layer->grad_w) {
        sgfnd_linear_destroy(layer);
        return NULL;
    }

    float scale = sqrtf(2.0f / in_dim);
    for (size_t i = 0; i < in_dim * out_dim; i++) {
        layer->weights[i] = randn() * scale;
    }

    if (bias) {
        layer->bias = calloc(out_dim, sizeof(float));
        layer->grad_b = calloc(out_dim, sizeof(float));
        if (!layer->bias || !layer->grad_b) {
            sgfnd_linear_destroy(layer);
            return NULL;
        }
    }
    return layer;
}

void sgfnd_linear_destroy(sgfnd_linear_t *layer) {
    if (!layer) return;
    free(layer->weights);
    free(layer->grad_w);
    free(layer->bias);
    free(layer->grad_b);
    free(layer);
}

void sgfnd_linear_forward(const sgfnd_linear_t *layer, const float *input, float *output) {
    if (!layer || !input || !output) return;
    if (layer->in_dim == 0 || layer->out_dim == 0) return;
    
    for (size_t j = 0; j < layer->out_dim; j++) {
        float sum = 0.0f;
        for (size_t i = 0; i < layer->in_dim; i++) {
            size_t w_idx = i * layer->out_dim + j;
            if (w_idx >= layer->in_dim * layer->out_dim) {
                fprintf(stderr, "[ERROR] Linear forward: weight index out of bounds\n");
                return;
            }
            sum += input[i] * layer->weights[w_idx];
        }
        if (layer->use_bias) sum += layer->bias[j];
        output[j] = sum;
    }
}

void sgfnd_linear_backward(sgfnd_linear_t *layer, const float *input, const float *grad_output, float *grad_input, float lr) {
    if (!layer || !input || !grad_output) return;

    // Sanity check dimensions
    if (layer->in_dim == 0 || layer->out_dim == 0) {
        fprintf(stderr, "[ERROR] Linear layer has zero dimensions\n");
        return;
    }

    for (size_t i = 0; i < layer->in_dim; i++) {
        float gi = 0.0f;
        for (size_t j = 0; j < layer->out_dim; j++) {
            size_t w_idx = i * layer->out_dim + j;
            // Bounds check
            if (w_idx >= layer->in_dim * layer->out_dim) {
                fprintf(stderr, "[ERROR] Weight index out of bounds: %zu >= %zu\n", w_idx, layer->in_dim * layer->out_dim);
                return;
            }
            gi += grad_output[j] * layer->weights[w_idx];
            layer->grad_w[w_idx] += input[i] * grad_output[j];
        }
        if (grad_input) grad_input[i] = gi;
    }

    if (layer->use_bias) {
        for (size_t j = 0; j < layer->out_dim; j++) {
            if (j >= layer->out_dim) break;
            layer->grad_b[j] += grad_output[j];
        }
    }

    // Clamp gradients to prevent explosion
    for (size_t i = 0; i < layer->in_dim * layer->out_dim; i++) {
        if (layer->grad_w[i] > 100.0f) layer->grad_w[i] = 100.0f;
        if (layer->grad_w[i] < -100.0f) layer->grad_w[i] = -100.0f;
        layer->weights[i] -= lr * layer->grad_w[i];
        layer->grad_w[i] = 0.0f;
    }
    if (layer->use_bias) {
        for (size_t j = 0; j < layer->out_dim; j++) {
            if (layer->grad_b[j] > 100.0f) layer->grad_b[j] = 100.0f;
            if (layer->grad_b[j] < -100.0f) layer->grad_b[j] = -100.0f;
            layer->bias[j] -= lr * layer->grad_b[j];
            layer->grad_b[j] = 0.0f;
        }
    }
}

sgfnd_mlp_t* sgfnd_mlp_create(const size_t *layer_dims, size_t num_layers) {
    if (!layer_dims || num_layers < 2) return NULL;

    sgfnd_mlp_t *mlp = calloc(1, sizeof(sgfnd_mlp_t));
    if (!mlp) return NULL;

    mlp->num_layers = num_layers - 1;
    mlp->layer_dims = malloc(num_layers * sizeof(size_t));
    mlp->layers = calloc(mlp->num_layers, sizeof(sgfnd_linear_t*));
    if (!mlp->layer_dims || !mlp->layers) {
        sgfnd_mlp_destroy(mlp);
        return NULL;
    }

    memcpy(mlp->layer_dims, layer_dims, num_layers * sizeof(size_t));

    for (size_t i = 0; i < mlp->num_layers; i++) {
        mlp->layers[i] = sgfnd_linear_create(layer_dims[i], layer_dims[i + 1], true);
        if (!mlp->layers[i]) {
            sgfnd_mlp_destroy(mlp);
            return NULL;
        }
    }

    size_t max_dim = 0;
    for (size_t i = 0; i < num_layers; i++) {
        if (layer_dims[i] > max_dim) max_dim = layer_dims[i];
    }
    mlp->activations = calloc(max_dim, sizeof(float));
    mlp->pre_activations = calloc(max_dim, sizeof(float));
    return mlp;
}

void sgfnd_mlp_destroy(sgfnd_mlp_t *mlp) {
    if (!mlp) return;
    for (size_t i = 0; i < mlp->num_layers; i++) {
        sgfnd_linear_destroy(mlp->layers[i]);
    }
    free(mlp->layer_dims);
    free(mlp->layers);
    free(mlp->activations);
    free(mlp->pre_activations);
    free(mlp);
}

static inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

static inline float silu_grad(float x) {
    float sig = 1.0f / (1.0f + expf(-x));
    return sig + x * sig * (1.0f - sig);
}

void sgfnd_mlp_forward(const sgfnd_mlp_t *mlp, const float *input, float *output) {
    if (!mlp || !input || !output) return;

    float *current_in = (float*)input;
    float *current_out = mlp->activations;

    for (size_t i = 0; i < mlp->num_layers; i++) {
        sgfnd_linear_forward(mlp->layers[i], current_in, current_out);

        if (i < mlp->num_layers - 1) {
            for (size_t j = 0; j < mlp->layer_dims[i + 1]; j++) {
                mlp->pre_activations[j] = current_out[j];
                current_out[j] = silu(current_out[j]);
            }
            current_in = current_out;
        } else {
            memcpy(output, current_out, mlp->layer_dims[i + 1] * sizeof(float));
        }
    }
}

void sgfnd_mlp_backward(sgfnd_mlp_t *mlp, const float *input, const float *target, float lr) {
    if (!mlp || !input || !target) return;

    size_t max_dim = 0;
    for (size_t i = 0; i <= mlp->num_layers; i++) {
        if (mlp->layer_dims[i] > max_dim) max_dim = mlp->layer_dims[i];
    }

    float *grad_out = calloc(max_dim, sizeof(float));
    float *grad_in = calloc(max_dim, sizeof(float));
    if (!grad_out || !grad_in) {
        free(grad_out);
        free(grad_in);
        return;
    }

    size_t out_dim = mlp->layer_dims[mlp->num_layers];
    for (size_t i = 0; i < out_dim; i++) {
        grad_out[i] = 2.0f * (mlp->activations[i] - target[i]);
    }

    for (int i = mlp->num_layers - 1; i >= 0; i--) {
        float *layer_in = (i == 0) ? (float*)input : mlp->activations;

        if (i < mlp->num_layers - 1) {
            size_t act_dim = mlp->layer_dims[i + 1];
            for (size_t j = 0; j < act_dim; j++) {
                grad_out[j] *= silu_grad(mlp->pre_activations[j]);
            }
        }

        sgfnd_linear_backward(mlp->layers[i], layer_in, grad_out, (i > 0) ? grad_in : NULL, lr);

        if (i > 0) {
            size_t in_dim = mlp->layers[i]->in_dim;
            memcpy(grad_out, grad_in, in_dim * sizeof(float));
            memset(grad_in, 0, in_dim * sizeof(float));
        }
    }

    free(grad_out);
    free(grad_in);
}

sgfnd_diffusion_t* sgfnd_diffusion_create(int num_timesteps, size_t latent_dim, size_t cond_dim) {
    sgfnd_diffusion_t *diff = calloc(1, sizeof(sgfnd_diffusion_t));
    if (!diff) return NULL;

    diff->num_timesteps = num_timesteps;
    diff->timesteps = malloc(num_timesteps * sizeof(float));
    diff->alphas = malloc(num_timesteps * sizeof(float));
    diff->alphas_cumprod = malloc(num_timesteps * sizeof(float));
    diff->betas = malloc(num_timesteps * sizeof(float));

    if (!diff->timesteps || !diff->alphas || !diff->alphas_cumprod || !diff->betas) {
        sgfnd_diffusion_destroy(diff);
        return NULL;
    }

    for (int i = 0; i < num_timesteps; i++) {
        float t = (float)i / num_timesteps;
        diff->timesteps[i] = t;
        diff->betas[i] = 0.0001f + 0.02f * t;
        diff->alphas[i] = 1.0f - diff->betas[i];
        diff->alphas_cumprod[i] = (i == 0) ? diff->alphas[i] : diff->alphas_cumprod[i - 1] * diff->alphas[i];
    }

    size_t time_embed_dims[] = {1, 256, 256};
    diff->time_embed = sgfnd_mlp_create(time_embed_dims, 3);

    size_t cond_embed_dims[] = {cond_dim, 256, 256};
    diff->cond_embed = sgfnd_mlp_create(cond_embed_dims, 3);

    size_t denoiser_dims[] = {latent_dim + 256 + 256, 512, 512, latent_dim};
    diff->denoiser = sgfnd_mlp_create(denoiser_dims, 4);

    return diff;
}

void sgfnd_diffusion_destroy(sgfnd_diffusion_t *diff) {
    if (!diff) return;
    sgfnd_mlp_destroy(diff->time_embed);
    sgfnd_mlp_destroy(diff->cond_embed);
    sgfnd_mlp_destroy(diff->denoiser);
    free(diff->timesteps);
    free(diff->alphas);
    free(diff->alphas_cumprod);
    free(diff->betas);
    free(diff);
}

void sgfnd_diffusion_forward(const sgfnd_diffusion_t *diff, const float *x_t, int t, const float *cond, float *eps_pred) {
    if (!diff || !x_t || !cond || !eps_pred) return;

    float t_norm = (float)t / diff->num_timesteps;
    float t_emb[256], c_emb[256];
    sgfnd_mlp_forward(diff->time_embed, &t_norm, t_emb);
    sgfnd_mlp_forward(diff->cond_embed, cond, c_emb);

    size_t latent_dim = diff->denoiser->layer_dims[0] - 512;
    float *input = malloc((latent_dim + 512) * sizeof(float));
    memcpy(input, x_t, latent_dim * sizeof(float));
    memcpy(input + latent_dim, t_emb, 256 * sizeof(float));
    memcpy(input + latent_dim + 256, c_emb, 256 * sizeof(float));

    sgfnd_mlp_forward(diff->denoiser, input, eps_pred);
    free(input);
}

void sgfnd_diffusion_sample(sgfnd_diffusion_t *diff, const float *cond, float *out_latent, int steps) {
    if (!diff || !cond || !out_latent) return;

    size_t latent_dim = diff->denoiser->layer_dims[0] - 512;
    float *x_t = malloc(latent_dim * sizeof(float));
    float *eps_pred = malloc(latent_dim * sizeof(float));

    for (size_t i = 0; i < latent_dim; i++) {
        x_t[i] = randn();
    }

    int step_size = diff->num_timesteps / steps;
    for (int s = steps - 1; s >= 0; s--) {
        int t = s * step_size;
        float alpha = diff->alphas[t];
        float alpha_cumprod = diff->alphas_cumprod[t];
        float beta = diff->betas[t];

        sgfnd_diffusion_forward(diff, x_t, t, cond, eps_pred);

        for (size_t i = 0; i < latent_dim; i++) {
            float pred_x0 = (x_t[i] - sqrtf(1.0f - alpha_cumprod) * eps_pred[i]) / sqrtf(alpha_cumprod);
            pred_x0 = fmaxf(-1.0f, fminf(1.0f, pred_x0));

            if (s > 0) {
                int t_prev = (s - 1) * step_size;
                float alpha_cumprod_prev = diff->alphas_cumprod[t_prev];
                float c1 = sqrtf(alpha_cumprod_prev) * beta / (1.0f - alpha_cumprod);
                float c2 = sqrtf(alpha) * (1.0f - alpha_cumprod_prev) / (1.0f - alpha_cumprod);
                x_t[i] = c1 * pred_x0 + c2 * x_t[i];
                if (s > 1) x_t[i] += sqrtf(beta) * randn();
            } else {
                x_t[i] = pred_x0;
            }
        }
    }

    memcpy(out_latent, x_t, latent_dim * sizeof(float));
    free(x_t);
    free(eps_pred);
}

float sgfnd_diffusion_loss(const sgfnd_diffusion_t *diff, const float *x_0, const float *cond, int t) {
    if (!diff || !x_0 || !cond) return 0.0f;

    size_t latent_dim = diff->denoiser->layer_dims[0] - 512;
    float *x_t = calloc(latent_dim, sizeof(float));
    float *eps_pred = calloc(latent_dim, sizeof(float));
    float *eps_true = calloc(latent_dim, sizeof(float));
    if (!x_t || !eps_pred || !eps_true) {
        free(x_t);
        free(eps_pred);
        free(eps_true);
        return 1e6f;
    }

    for (size_t i = 0; i < latent_dim; i++) {
        eps_true[i] = randn();
        float alpha_cumprod = diff->alphas_cumprod[t];
        float sqrt_acp = sqrtf(fmaxf(0.0f, alpha_cumprod));
        float sqrt_one_minus_acp = sqrtf(fmaxf(0.0f, 1.0f - alpha_cumprod));
        x_t[i] = sqrt_acp * x_0[i] + sqrt_one_minus_acp * eps_true[i];
    }

    sgfnd_diffusion_forward(diff, x_t, t, cond, eps_pred);

    float loss = 0.0f;
    for (size_t i = 0; i < latent_dim; i++) {
        float diff_val = eps_pred[i] - eps_true[i];
        // Clamp diff to prevent huge values
        if (diff_val > 100.0f) diff_val = 100.0f;
        if (diff_val < -100.0f) diff_val = -100.0f;
        loss += diff_val * diff_val;
    }
    loss /= latent_dim;

    if (isnan(loss) || isinf(loss) || loss > 1e6f) {
        loss = 1e6f;
    }

    free(x_t);
    free(eps_pred);
    free(eps_true);
    return loss;
}