#include "sgfnd_core.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#define LUT_SIZE 256
#define MAX_PALETTE 1024

struct ColorComponent {
    char name[64];
    float hue_min, hue_max;
    float sat_min, sat_max;
    float val_min, val_max;
    float weight;
};

static struct ColorComponent default_components[] = {
    {"eyes", 0.5f, 0.7f, 0.3f, 1.0f, 0.2f, 1.0f, 1.0f},
    {"hair", 0.0f, 1.0f, 0.2f, 1.0f, 0.1f, 1.0f, 1.0f},
    {"skin", 0.02f, 0.12f, 0.1f, 0.6f, 0.3f, 1.0f, 1.0f},
    {"clothing", 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f},
    {"lighting", 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f},
    {"background", 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f},
    {"style", 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f},
};

sgfnd_color_grader_t* sgfnd_color_grader_create(void) {
    sgfnd_color_grader_t *grader = calloc(1, sizeof(sgfnd_color_grader_t));
    if (!grader) return NULL;

    grader->palette_size = MAX_PALETTE;
    grader->palette = calloc(MAX_PALETTE, sizeof(sgfnd_color_t));
    grader->lut_size = LUT_SIZE * LUT_SIZE * LUT_SIZE;
    grader->grading_lut = calloc(grader->lut_size, sizeof(float));
    return grader;
}

void sgfnd_color_grader_destroy(sgfnd_color_grader_t *grader) {
    if (!grader) return;
    free(grader->palette);
    free(grader->grading_lut);
    free(grader);
}

void rgb_to_hsv(float r, float g, float b, float *h, float *s, float *v) {
    float max = fmaxf(fmaxf(r, g), b);
    float min = fminf(fminf(r, g), b);
    float delta = max - min;

    *v = max;
    *s = max == 0 ? 0 : delta / max;

    if (delta == 0) {
        *h = 0;
    } else if (max == r) {
        *h = fmodf(60 * (g - b) / delta + 360, 360);
    } else if (max == g) {
        *h = fmodf(60 * (b - r) / delta + 120, 360);
    } else {
        *h = fmodf(60 * (r - g) / delta + 240, 360);
    }
    *h /= 360.0f;
}

void hsv_to_rgb(float h, float s, float v, float *r, float *g, float *b) {
    h *= 360.0f;
    int i = (int)(h / 60) % 6;
    float f = h / 60 - i;
    float p = v * (1 - s);
    float q = v * (1 - f * s);
    float t = v * (1 - (1 - f) * s);

    switch (i) {
        case 0: *r = v; *g = t; *b = p; break;
        case 1: *r = q; *g = v; *b = p; break;
        case 2: *r = p; *g = v; *b = t; break;
        case 3: *r = p; *g = q; *b = v; break;
        case 4: *r = t; *g = p; *b = v; break;
        case 5: *r = v; *g = p; *b = q; break;
    }
}

int sgfnd_color_grader_train(sgfnd_color_grader_t *grader, const sgfnd_image_t *img) {
    if (!grader || !img || !img->full_pixels) return -1;

    size_t pixel_count = img->width * img->height;
    size_t components_per_pixel = img->channels;

    for (size_t i = 0; i < pixel_count; i++) {
        float r = img->full_pixels[i * components_per_pixel + 0];
        float g = img->full_pixels[i * components_per_pixel + 1];
        float b = img->full_pixels[i * components_per_pixel + 2];
        float a = components_per_pixel > 3 ? img->full_pixels[i * components_per_pixel + 3] : 1.0f;

        float h, s, v;
        rgb_to_hsv(r, g, b, &h, &s, &v);

        for (size_t c = 0; c < sizeof(default_components)/sizeof(default_components[0]); c++) {
            if (h >= default_components[c].hue_min && h <= default_components[c].hue_max &&
                s >= default_components[c].sat_min && s <= default_components[c].sat_max &&
                v >= default_components[c].val_min && v <= default_components[c].val_max) {
                size_t idx = (size_t)(h * LUT_SIZE) * LUT_SIZE * LUT_SIZE +
                             (size_t)(s * LUT_SIZE) * LUT_SIZE +
                             (size_t)(v * LUT_SIZE);
                grader->grading_lut[idx] += default_components[c].weight;
            }
        }
    }

    float max_val = 0;
    for (size_t i = 0; i < grader->lut_size; i++) {
        if (grader->grading_lut[i] > max_val) max_val = grader->grading_lut[i];
    }
    if (max_val > 0) {
        for (size_t i = 0; i < grader->lut_size; i++) {
            grader->grading_lut[i] /= max_val;
        }
    }

    return 0;
}

sgfnd_color_t sgfnd_color_grader_map(const sgfnd_color_grader_t *grader, const sgfnd_color_t color, const char *component) {
    sgfnd_color_t result = color;
    if (!grader || !component) return result;

    float h, s, v;
    rgb_to_hsv(color.r, color.g, color.b, &h, &s, &v);

    size_t idx = (size_t)(h * LUT_SIZE) * LUT_SIZE * LUT_SIZE +
                 (size_t)(s * LUT_SIZE) * LUT_SIZE +
                 (size_t)(v * LUT_SIZE);
    idx = idx < grader->lut_size ? idx : grader->lut_size - 1;

    float boost = grader->grading_lut[idx];
    s = fminf(1.0f, s * (1.0f + boost * 0.5f));
    v = fminf(1.0f, v * (1.0f + boost * 0.3f));

    hsv_to_rgb(h, s, v, &result.r, &result.g, &result.b);
    result.a = color.a;

    return result;
}