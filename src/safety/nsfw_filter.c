#define _GNU_SOURCE
#include "sgfnd_core.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <ctype.h>

static const char *nsfw_keywords[] = {
    "nsfw", "explicit", "nude", "naked", "gore", "blood",
    "porn", "adult", "erotic", "sex", "violence", "xxx"
};
#define NUM_NSFW_KEYWORDS (sizeof(nsfw_keywords) / sizeof(nsfw_keywords[0]))

sgfnd_nsfw_filter_t* sgfnd_nsfw_filter_create(sgfnd_nsfw_mode_t mode, float threshold) {
    sgfnd_nsfw_filter_t *filter = calloc(1, sizeof(sgfnd_nsfw_filter_t));
    if (!filter) return NULL;

    filter->mode = mode;
    filter->threshold = (threshold <= 0.0f) ? 0.65f : threshold;
    filter->blocked_prompts = 0;
    filter->flagged_images = 0;
    filter->sanitized_images = 0;

    return filter;
}

void sgfnd_nsfw_filter_destroy(sgfnd_nsfw_filter_t *filter) {
    if (filter) {
        free(filter);
    }
}

static bool str_contains_ic(const char *haystack, const char *needle) {
    if (!haystack || !needle) return false;
    size_t h_len = strlen(haystack);
    size_t n_len = strlen(needle);
    if (n_len > h_len) return false;

    for (size_t i = 0; i <= h_len - n_len; i++) {
        bool match = true;
        for (size_t j = 0; j < n_len; j++) {
            if (tolower((unsigned char)haystack[i + j]) != tolower((unsigned char)needle[j])) {
                match = false;
                break;
            }
        }
        if (match) return true;
    }
    return false;
}

bool sgfnd_nsfw_check_prompt(sgfnd_nsfw_filter_t *filter, const sgfnd_prompt_t *prompt) {
    if (!filter || filter->mode == SGFND_NSFW_FILTER_DISABLED || !prompt) {
        return true;
    }

    for (size_t i = 0; i < prompt->count; i++) {
        if (!prompt->tags || !prompt->tags[i]) continue;
        for (size_t k = 0; k < NUM_NSFW_KEYWORDS; k++) {
            if (str_contains_ic(prompt->tags[i], nsfw_keywords[k])) {
                filter->blocked_prompts++;
                return false;
            }
        }
    }

    return true;
}

float sgfnd_nsfw_check_image(sgfnd_nsfw_filter_t *filter, const sgfnd_image_t *img) {
    if (!filter || !img) return 0.0f;
    if (filter->mode == SGFND_NSFW_FILTER_DISABLED) return 0.0f;

    float *pixels = img->full_pixels;
    if (!pixels && img->tiles && img->tile_count > 0) {
        pixels = img->tiles[0].pixels;
    }
    if (!pixels) return 0.0f;

    size_t pixel_count = img->width * img->height;
    size_t skin_pixels = 0;
    size_t gore_pixels = 0;

    for (size_t i = 0; i < pixel_count; i++) {
        float r = pixels[i * img->channels + 0];
        float g = pixels[i * img->channels + 1];
        float b = pixels[i * img->channels + 2];

        if (r > 0.35f && g > 0.15f && b > 0.05f &&
            r > g && (r - g) > 0.03f && r > b) {
            skin_pixels++;
        }

        if (r > 0.60f && g < 0.20f && b < 0.20f) {
            gore_pixels++;
        }
    }

    float skin_ratio = (float)skin_pixels / (float)pixel_count;
    float gore_ratio = (float)gore_pixels / (float)pixel_count;

    float score = skin_ratio * 0.7f + gore_ratio * 0.9f;

    if (filter->mode == SGFND_NSFW_FILTER_STRICT) {
        score *= 1.3f;
    }

    if (score >= filter->threshold) {
        filter->flagged_images++;
    }

    return score;
}

int sgfnd_nsfw_sanitize_image(sgfnd_nsfw_filter_t *filter, sgfnd_image_t *img) {
    if (!filter || !img || filter->mode == SGFND_NSFW_FILTER_DISABLED) {
        return 0;
    }

    float score = sgfnd_nsfw_check_image(filter, img);
    if (score < filter->threshold) {
        return 0;
    }

    filter->sanitized_images++;

    float *pixels = img->full_pixels;
    if (!pixels) return 0;

    uint32_t w = img->width;
    uint32_t h = img->height;
    uint8_t c = img->channels;
    uint32_t block = 16;

    for (uint32_t by = 0; by < h; by += block) {
        for (uint32_t bx = 0; bx < w; bx += block) {
            float avg_r = 0, avg_g = 0, avg_b = 0;
            uint32_t count = 0;

            for (uint32_t y = by; y < by + block && y < h; y++) {
                for (uint32_t x = bx; x < bx + block && x < w; x++) {
                    size_t idx = (y * w + x) * c;
                    avg_r += pixels[idx + 0];
                    avg_g += pixels[idx + 1];
                    avg_b += pixels[idx + 2];
                    count++;
                }
            }

            if (count > 0) {
                avg_r /= count;
                avg_g /= count;
                avg_b /= count;
            }

            for (uint32_t y = by; y < by + block && y < h; y++) {
                for (uint32_t x = bx; x < bx + block && x < w; x++) {
                    size_t idx = (y * w + x) * c;
                    pixels[idx + 0] = avg_r;
                    pixels[idx + 1] = avg_g;
                    pixels[idx + 2] = avg_b;
                }
            }
        }
    }

    return 1;
}