#include "sgfnd_core.h"
#include "sgfnd_bridge_internal.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

#define TILE_SIZE 64

static inline float lerp(float a, float b, float t) {
    return a + (b - a) * t;
}

static inline float smoothstep(float t) {
    return t * t * (3.0f - 2.0f * t);
}

static float hash2d(float x, float y) {
    float n = sinf(x * 12.9898f + y * 78.233f) * 43758.5453f;
    return n - floorf(n);
}

static float noise2d(float x, float y) {
    int ix = (int)floorf(x);
    int iy = (int)floorf(y);
    float fx = x - ix;
    float fy = y - iy;
    float u = smoothstep(fx);
    float v = smoothstep(fy);
    float a = hash2d(ix, iy);
    float b = hash2d(ix + 1, iy);
    float c = hash2d(ix, iy + 1);
    float d = hash2d(ix + 1, iy + 1);
    return lerp(lerp(a, b, u), lerp(c, d, u), v);
}

static float fbm2d(float x, float y, int octaves) {
    float value = 0.0f;
    float amplitude = 0.5f;
    float frequency = 1.0f;
    for (int i = 0; i < octaves; i++) {
        value += amplitude * noise2d(x * frequency, y * frequency);
        amplitude *= 0.5f;
        frequency *= 2.0f;
    }
    return value;
}

sgfnd_image_t* sgfnd_image_create_tiled(uint32_t width, uint32_t height, uint8_t channels, uint32_t tile_size) {
    sgfnd_image_t *img = calloc(1, sizeof(sgfnd_image_t));
    if (!img) return NULL;

    img->width = width;
    img->height = height;
    img->channels = channels;
    img->bit_depth = 8;
    img->tile_size = tile_size ? tile_size : TILE_SIZE;

    uint32_t tiles_x = (width + img->tile_size - 1) / img->tile_size;
    uint32_t tiles_y = (height + img->tile_size - 1) / img->tile_size;
    img->tile_count = tiles_x * tiles_y;
    img->tiles = calloc(img->tile_count, sizeof(sgfnd_tile_t));
    if (!img->tiles) {
        free(img);
        return NULL;
    }

    for (uint32_t ty = 0; ty < tiles_y; ty++) {
        for (uint32_t tx = 0; tx < tiles_x; tx++) {
            uint32_t idx = ty * tiles_x + tx;
            sgfnd_tile_t *tile = &img->tiles[idx];
            tile->x = tx * img->tile_size;
            tile->y = ty * img->tile_size;
            tile->width = (tx == tiles_x - 1) ? (width - tile->x) : img->tile_size;
            tile->height = (ty == tiles_y - 1) ? (height - tile->y) : img->tile_size;
            tile->stride = tile->width * channels;
            tile->pixels = calloc(tile->width * tile->height * channels, sizeof(float));
            tile->owns_memory = true;
        }
    }

    img->full_pixels = calloc(width * height * channels, sizeof(float));
    if (!img->full_pixels) {
        free(img->tiles);
        free(img);
        return NULL;
    }

    return img;
}

void sgfnd_image_free(sgfnd_image_t *img) {
    if (!img) return;
    if (img->tiles) {
        for (size_t i = 0; i < img->tile_count; i++) {
            if (img->tiles[i].owns_memory) {
                free(img->tiles[i].pixels);
            }
        }
        free(img->tiles);
    }
    free(img->full_pixels);
    free(img);
}

sgfnd_tile_t* sgfnd_image_get_tile(sgfnd_image_t *img, uint32_t tile_x, uint32_t tile_y) {
    if (!img) return NULL;
    uint32_t tiles_x = (img->width + img->tile_size - 1) / img->tile_size;
    if (tile_x >= tiles_x || tile_y >= (img->height + img->tile_size - 1) / img->tile_size) return NULL;
    return &img->tiles[tile_y * tiles_x + tile_x];
}

int sgfnd_image_assemble_tiles(const sgfnd_image_t *img, float *out_pixels) {
    if (!img || !out_pixels) return -1;
    uint32_t tiles_x = (img->width + img->tile_size - 1) / img->tile_size;
    uint32_t tiles_y = (img->height + img->tile_size - 1) / img->tile_size;

    for (uint32_t ty = 0; ty < tiles_y; ty++) {
        for (uint32_t tx = 0; tx < tiles_x; tx++) {
            sgfnd_tile_t *tile = (sgfnd_tile_t*)&img->tiles[ty * tiles_x + tx];
            for (uint32_t y = 0; y < tile->height; y++) {
                for (uint32_t x = 0; x < tile->width; x++) {
                    size_t src_idx = (y * tile->width + x) * img->channels;
                    size_t dst_idx = ((tile->y + y) * img->width + (tile->x + x)) * img->channels;
                    for (uint8_t c = 0; c < img->channels; c++) {
                        out_pixels[dst_idx + c] = tile->pixels[src_idx + c];
                    }
                }
            }
        }
    }
    return 0;
}

extern void rgb_to_hsv(float r, float g, float b, float *h, float *s, float *v);
extern void hsv_to_rgb(float h, float s, float v, float *r, float *g, float *b);
extern sgfnd_color_t sgfnd_color_grader_map(const sgfnd_color_grader_t *grader, const sgfnd_color_t color, const char *component);

/*
static sgfnd_color_t sample_bridge_field(const sgfnd_engine_t *engine, float u, float v, size_t bridge_idx) {
    (void)u; (void)v;
    sgfnd_color_t color = {0};
    if (engine->mode == SGFND_MODE_SMALL) {
        if (bridge_idx >= engine->ctx.small.active_count) return color;
        const sgfnd_bridge_t *b = &engine->ctx.small.bridges[bridge_idx];
        if (b->size == 0) return color;
        size_t idx = (size_t)(v * b->size) % b->size;
        float val = b->quant == SGFND_QUANT_NONE ? b->data[idx] : 0.0f;
        if (b->quant != SGFND_QUANT_NONE) {
            float tmp;
            sgfnd_bridge_dequantize(b, &tmp, 1);
            val = tmp;
        }
        color.r = val;
        color.g = val * 0.8f;
        color.b = val * 0.6f;
        color.a = 1.0f;
    } else {
        if (bridge_idx >= engine->ctx.large.count) return color;
        const sgfnd_bridge_t *b = &engine->ctx.large.bridges[bridge_idx];
        if (b->size == 0) return color;
        size_t idx = (size_t)(v * b->size) % b->size;
        float val = b->quant == SGFND_QUANT_NONE ? b->data[idx] : 0.0f;
        if (b->quant != SGFND_QUANT_NONE) {
            float tmp;
            sgfnd_bridge_dequantize(b, &tmp, 1);
            val = tmp;
        }
        color.r = val;
        color.g = val * 0.8f;
        color.b = val * 0.6f;
        color.a = 1.0f;
    }
    return color;
}
*/

int sgfnd_render_tile(sgfnd_engine_t *engine, const sgfnd_prompt_t *prompt, sgfnd_tile_t *tile, uint32_t tile_x, uint32_t tile_y) {
    (void)tile_x; (void)tile_y;
    if (!engine || !tile || !tile->pixels) return -1;

    const char *components[] = {"eyes", "hair", "skin", "clothing", "lighting", "background", "style"};
    size_t num_components = sizeof(components) / sizeof(components[0]);

    for (uint32_t y = 0; y < tile->height; y++) {
        for (uint32_t x = 0; x < tile->width; x++) {
            float u = (float)(tile->x + x) / (engine->mode == SGFND_MODE_SMALL ? 512.0f : 512.0f);
            float v = (float)(tile->y + y) / (engine->mode == SGFND_MODE_SMALL ? 512.0f : 512.0f);

            float base_noise = fbm2d(u * 8.0f, v * 8.0f, 4);
            float detail_noise = fbm2d(u * 32.0f, v * 32.0f, 2) * 0.3f;

            size_t active_bridges = engine->mode == SGFND_MODE_SMALL ? engine->ctx.small.active_count : engine->ctx.large.count;
            size_t bridge_idx = (size_t)(u * active_bridges) % active_bridges;

            sgfnd_bridge_t *bridge = engine->mode == SGFND_MODE_SMALL ?
                &engine->ctx.small.bridges[bridge_idx] :
                &engine->ctx.large.bridges[bridge_idx];

            float bridge_val = 0.0f;
            if (bridge->size > 0) {
                size_t idx = (size_t)(v * bridge->size) % bridge->size;
                if (bridge->quant == SGFND_QUANT_NONE && bridge->data) {
                    bridge_val = bridge->data[idx];
                } else if (bridge->quant != SGFND_QUANT_NONE && bridge->quant_data) {
                    float tmp = 0.0f;
                    sgfnd_bridge_dequantize(bridge, &tmp, 1);
                    bridge_val = tmp;
                }
            }

            float combined = base_noise * 0.5f + detail_noise + bridge_val * 0.5f;
            combined = fmaxf(0.0f, fminf(1.0f, combined));

            sgfnd_color_t color = {combined, combined * 0.85f, combined * 0.7f, 1.0f};

            if (prompt) {
                for (size_t i = 0; i < num_components; i++) {
                    color = sgfnd_color_grader_map(NULL, color, components[i]);
                }
            }

            float h, s, val;
            rgb_to_hsv(color.r, color.g, color.b, &h, &s, &val);
            s *= 1.0f + base_noise * 0.3f;
            val *= 1.0f + detail_noise * 0.2f;
            s = fminf(1.0f, s);
            val = fminf(1.0f, val);
            hsv_to_rgb(h, s, val, &color.r, &color.g, &color.b);

            size_t idx = (y * tile->width + x) * tile->stride / tile->width;
            tile->pixels[idx + 0] = color.r;
            tile->pixels[idx + 1] = color.g;
            tile->pixels[idx + 2] = color.b;
            tile->pixels[idx + 3] = color.a;
        }
    }
    return 0;
}

int sgfnd_render_tiled(sgfnd_engine_t *engine, const sgfnd_prompt_t *prompt, const sgfnd_render_config_t *config) {
    if (!engine || !config || !config->tile_callback) return -1;

    uint32_t tile_size = config->tile_size ? config->tile_size : TILE_SIZE;
    uint32_t img_width = 512;
    uint32_t img_height = 512;
    uint32_t tiles_x = (img_width + tile_size - 1) / tile_size;
    uint32_t tiles_y = (img_height + tile_size - 1) / tile_size;

    sgfnd_tile_t tile;
    tile.pixels = malloc(tile_size * tile_size * 4 * sizeof(float));
    if (!tile.pixels) return -1;

    for (uint32_t ty = 0; ty < tiles_y; ty++) {
        for (uint32_t tx = 0; tx < tiles_x; tx++) {
            tile.x = tx * tile_size;
            tile.y = ty * tile_size;
            tile.width = (tx == tiles_x - 1) ? (img_width - tile.x) : tile_size;
            tile.height = (ty == tiles_y - 1) ? (img_height - tile.y) : tile_size;
            tile.stride = tile.width * 4;

            sgfnd_render_tile(engine, prompt, &tile, tx, ty);
            config->tile_callback(&tile, config->user_data);
        }
    }

    free(tile.pixels);
    return 0;
}

sgfnd_image_t* sgfnd_generate_image(sgfnd_engine_t *engine, const sgfnd_prompt_t *prompt) {
    if (!engine) return NULL;

    sgfnd_image_t *img = sgfnd_image_create_tiled(512, 512, 4, TILE_SIZE);
    if (!img) return NULL;

    uint32_t tiles_x = (img->width + img->tile_size - 1) / img->tile_size;
    uint32_t tiles_y = (img->height + img->tile_size - 1) / img->tile_size;

    for (uint32_t ty = 0; ty < tiles_y; ty++) {
        for (uint32_t tx = 0; tx < tiles_x; tx++) {
            sgfnd_tile_t *tile = sgfnd_image_get_tile(img, tx, ty);
            sgfnd_render_tile(engine, prompt, tile, tx, ty);
        }
    }

    size_t pixel_count = img->width * img->height * img->channels;
    img->full_pixels = malloc(pixel_count * sizeof(float));
    sgfnd_image_assemble_tiles(img, img->full_pixels);

    return img;
}

int sgfnd_image_save_raw(const sgfnd_image_t *img, const char *path) {
    if (!img || !path) return -1;

    float *pixels = img->full_pixels;
    if (!pixels && img->tiles) {
        size_t pixel_count = img->width * img->height * img->channels;
        pixels = malloc(pixel_count * sizeof(float));
        if (!pixels) return -1;
        sgfnd_image_assemble_tiles(img, pixels);
    }
    if (!pixels) return -1;

    FILE *f = fopen(path, "wb");
    if (!f) {
        if (pixels != img->full_pixels) free(pixels);
        return -1;
    }

    uint32_t header[4] = {img->width, img->height, img->channels, img->bit_depth};
    fwrite(header, sizeof(uint32_t), 4, f);

    size_t pixel_count = img->width * img->height * img->channels;
    uint8_t *buf = malloc(pixel_count);
    for (size_t i = 0; i < pixel_count; i++) {
        buf[i] = (uint8_t)(pixels[i] * 255.0f);
    }
    fwrite(buf, 1, pixel_count, f);
    free(buf);

    if (pixels != img->full_pixels) free(pixels);
    fclose(f);
    return 0;
}

int sgfnd_image_load_raw(sgfnd_image_t *img, const char *path) {
    if (!img || !path) return -1;

    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    uint32_t header[4];
    if (fread(header, sizeof(uint32_t), 4, f) != 4) {
        fclose(f);
        return -1;
    }

    img->width = header[0];
    img->height = header[1];
    img->channels = header[2];
    img->bit_depth = header[3];
    img->tile_size = TILE_SIZE;

    sgfnd_image_t *tiled = sgfnd_image_create_tiled(img->width, img->height, img->channels, TILE_SIZE);
    if (!tiled) {
        fclose(f);
        return -1;
    }

    size_t pixel_count = img->width * img->height * img->channels;
    uint8_t *buf = malloc(pixel_count);
    fread(buf, 1, pixel_count, f);
    fclose(f);

    for (size_t i = 0; i < pixel_count; i++) {
        tiled->full_pixels[i] = buf[i] / 255.0f;
    }
    free(buf);

    sgfnd_image_assemble_tiles(tiled, tiled->full_pixels);

    *img = *tiled;
    free(tiled);
    return 0;
}

int sgfnd_image_save_tiled_raw(const sgfnd_image_t *img, const char *path) {
    return sgfnd_image_save_raw(img, path);
}

int sgfnd_image_load_tiled_raw(sgfnd_image_t *img, const char *path) {
    return sgfnd_image_load_raw(img, path);
}