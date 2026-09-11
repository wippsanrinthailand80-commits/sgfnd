#include "sgfnd_core.h"
#include "sgfnd_bridge_internal.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define LARGE_BRIDGE_BATCH 1024
#define LARGE_VRAM_ALIGNMENT 4096

typedef struct {
    sgfnd_bridge_t *bridges;
    size_t count;
    size_t capacity;
    float *vram_buffer;
    size_t vram_offset;
    size_t vram_capacity;
    bool *bridge_active;
} sgfnd_large_model_t;

sgfnd_large_model_t* sgfnd_large_model_create(size_t bridge_count, size_t vram_bytes) {
    sgfnd_large_model_t *model = calloc(1, sizeof(sgfnd_large_model_t));
    if (!model) return NULL;

    model->count = bridge_count;
    model->capacity = bridge_count;
    model->bridges = calloc(bridge_count, sizeof(sgfnd_bridge_t));
    model->bridge_active = calloc(bridge_count, sizeof(bool));
    if (!model->bridges || !model->bridge_active) {
        free(model->bridges);
        free(model->bridge_active);
        free(model);
        return NULL;
    }

    model->vram_capacity = vram_bytes;
    model->vram_buffer = aligned_alloc(LARGE_VRAM_ALIGNMENT, vram_bytes);
    if (!model->vram_buffer) {
        free(model->bridges);
        free(model->bridge_active);
        free(model);
        return NULL;
    }

    for (size_t i = 0; i < bridge_count; i++) {
        model->bridges[i] = bridge_create_internal(SGFND_FP32);
        model->bridge_active[i] = true;
    }

    return model;
}

void sgfnd_large_model_destroy(sgfnd_large_model_t *model) {
    if (!model) return;
    for (size_t i = 0; i < model->count; i++) {
        bridge_destroy_internal(&model->bridges[i]);
    }
    free(model->bridges);
    free(model->bridge_active);
    free(model->vram_buffer);
    free(model);
}

int sgfnd_large_model_process_input(sgfnd_large_model_t *model, const float *input, size_t input_size, int input_index) {
    if (!model || !input || input_index < 0 || input_index >= 3) return -1;

    size_t per_bridge = input_size / model->count;
    size_t start_bridge = (input_index * model->count) / 3;
    size_t end_bridge = ((input_index + 1) * model->count) / 3;

    for (size_t i = start_bridge; i < end_bridge; i++) {
        if (!model->bridge_active[i]) continue;
        sgfnd_bridge_t *b = &model->bridges[i];
        size_t start = (i - start_bridge) * per_bridge;
        size_t end = (i == end_bridge - 1) ? input_size : start + per_bridge;
        for (size_t j = start; j < end && j < input_size; j++) {
            if (bridge_push_internal(b, input[j]) != 0) return -1;
        }
    }
    return 0;
}

extern void rgb_to_hsv(float r, float g, float b, float *h, float *s, float *v);
extern void hsv_to_rgb(float h, float s, float v, float *r, float *g, float *b);
extern sgfnd_color_t sgfnd_color_grader_map(const sgfnd_color_grader_t *grader, const sgfnd_color_t color, const char *component);

int sgfnd_large_model_generate_pixel(sgfnd_large_model_t *model, size_t x, size_t y, size_t width, size_t height,
                                     sgfnd_color_grader_t *grader, const sgfnd_prompt_t *prompt,
                                     sgfnd_color_t *out_color) {
    if (!model || !out_color) return -1;

    float u = (float)x / width;
    float v = (float)y / height;

    size_t bridge_idx = (size_t)(u * model->count) % model->count;
    if (!model->bridge_active[bridge_idx]) {
        out_color->r = out_color->g = out_color->b = 0.0f;
        out_color->a = 1.0f;
        return 0;
    }

    sgfnd_bridge_t *b = &model->bridges[bridge_idx];
    float val = b->size > 0 ? b->data[(size_t)(v * b->size) % b->size] : 0.0f;

    sgfnd_color_t base = {val, val * 0.8f, val * 0.6f, 1.0f};

    if (grader && prompt) {
        const char *components[] = {"eyes", "hair", "skin", "clothing", "lighting", "background", "style"};
        for (size_t i = 0; i < sizeof(components)/sizeof(components[0]); i++) {
            base = sgfnd_color_grader_map(grader, base, components[i]);
        }
    }

    *out_color = base;
    return 0;
}

int sgfnd_large_model_render(sgfnd_large_model_t *model, sgfnd_image_t *img,
                             sgfnd_color_grader_t *grader, const sgfnd_prompt_t *prompt) {
    if (!model || !img || !img->full_pixels) return -1;

    for (size_t y = 0; y < img->height; y++) {
        for (size_t x = 0; x < img->width; x++) {
            sgfnd_color_t color;
            sgfnd_large_model_generate_pixel(model, x, y, img->width, img->height, grader, prompt, &color);
            size_t idx = (y * img->width + x) * img->channels;
            img->full_pixels[idx + 0] = color.r;
            img->full_pixels[idx + 1] = color.g;
            img->full_pixels[idx + 2] = color.b;
            if (img->channels > 3) img->full_pixels[idx + 3] = color.a;
        }
    }
    return 0;
}

int sgfnd_large_model_set_bridge_active(sgfnd_large_model_t *model, size_t index, bool active) {
    if (!model || index >= model->count) return -1;
    model->bridge_active[index] = active;
    return 0;
}

size_t sgfnd_large_model_get_active_count(const sgfnd_large_model_t *model) {
    if (!model) return 0;
    size_t count = 0;
    for (size_t i = 0; i < model->count; i++) {
        if (model->bridge_active[i]) count++;
    }
    return count;
}