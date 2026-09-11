#define _GNU_SOURCE
#include "sgfnd_core.h"
#include "sgfnd_bridge_internal.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static inline int clamp(int val, int min, int max) {
    return val < min ? min : (val > max ? max : val);
}

#define COMPRESSION_THRESHOLD 0.01f

sgfnd_engine_t* sgfnd_create(sgfnd_mode_t mode, size_t vram_bytes) {
    sgfnd_engine_t *engine = calloc(1, sizeof(sgfnd_engine_t));
    if (!engine) return NULL;

    engine->mode = mode;
    engine->vram_size = vram_bytes;
    engine->vram_pool = aligned_alloc(4096, vram_bytes);
    engine->default_quant = SGFND_QUANT_NONE;
    if (!engine->vram_pool) {
        free(engine);
        return NULL;
    }

    if (mode == SGFND_MODE_SMALL) {
        for (int i = 0; i < SGFND_BRIDGES_SMALL; i++) {
            engine->ctx.small.bridges[i] = bridge_create_internal(SGFND_FP32);
        }
        engine->ctx.small.active_count = SGFND_BRIDGES_SMALL;
        engine->ctx.small.compressed = false;
    } else {
        engine->ctx.large.capacity = SGFND_BRIDGES_LARGE;
        engine->ctx.large.count = SGFND_BRIDGES_LARGE;
        engine->ctx.large.bridges = calloc(SGFND_BRIDGES_LARGE, sizeof(sgfnd_bridge_t));
        engine->ctx.large.bridge_loaded = calloc(SGFND_BRIDGES_LARGE, sizeof(bool));
        engine->ctx.large.vram_budget = vram_bytes;
        engine->ctx.large.vram_used = 0;
        if (!engine->ctx.large.bridges || !engine->ctx.large.bridge_loaded) {
            free(engine->ctx.large.bridges);
            free(engine->ctx.large.bridge_loaded);
            free(engine->vram_pool);
            free(engine);
            return NULL;
        }
        for (size_t i = 0; i < SGFND_BRIDGES_LARGE; i++) {
            engine->ctx.large.bridges[i] = bridge_create_internal(SGFND_FP32);
        }
    }

    return engine;
}

void sgfnd_destroy(sgfnd_engine_t *engine) {
    if (!engine) return;

    if (engine->mode == SGFND_MODE_SMALL) {
        for (int i = 0; i < SGFND_BRIDGES_SMALL; i++) {
            bridge_destroy_internal(&engine->ctx.small.bridges[i]);
        }
    } else {
        for (size_t i = 0; i < engine->ctx.large.count; i++) {
            bridge_destroy_internal(&engine->ctx.large.bridges[i]);
        }
        free(engine->ctx.large.bridges);
        free(engine->ctx.large.bridge_loaded);
        free(engine->ctx.large.bridge_offsets);
        free(engine->ctx.large.weights_path);
    }

    free(engine->vram_pool);
    free(engine);
}

int sgfnd_process_input(sgfnd_engine_t *engine, const float *input, size_t input_size) {
    if (!engine || !input) return -1;

    if (engine->mode == SGFND_MODE_SMALL) {
        size_t per_bridge = input_size / engine->ctx.small.active_count;
        for (size_t i = 0; i < engine->ctx.small.active_count; i++) {
            sgfnd_bridge_t *b = &engine->ctx.small.bridges[i];
            size_t start = i * per_bridge;
            size_t end = (i == engine->ctx.small.active_count - 1) ? input_size : start + per_bridge;
            for (size_t j = start; j < end; j++) {
                if (bridge_push_internal(b, input[j]) != 0) return -1;
            }
        }
    } else {
        size_t per_bridge = input_size / engine->ctx.large.count;
        for (size_t i = 0; i < engine->ctx.large.count; i++) {
            sgfnd_bridge_t *b = &engine->ctx.large.bridges[i];
            size_t start = i * per_bridge;
            size_t end = (i == engine->ctx.large.count - 1) ? input_size : start + per_bridge;
            for (size_t j = start; j < end; j++) {
                if (bridge_push_internal(b, input[j]) != 0) return -1;
            }
        }
    }
    return 0;
}

static float bridge_importance(const sgfnd_bridge_t *b) {
    if (b->size == 0) return 0.0f;
    float sum = 0.0f;
    for (size_t i = 0; i < b->size; i++) {
        sum += fabsf(b->data[i]);
    }
    return sum / b->size;
}

int sgfnd_compress_bridges(sgfnd_engine_t *engine) {
    if (!engine || engine->mode != SGFND_MODE_SMALL) return -1;
    if (engine->ctx.small.compressed) return 0;

    typedef struct { float importance; int index; } bridge_score_t;
    bridge_score_t scores[SGFND_BRIDGES_SMALL];

    for (int i = 0; i < SGFND_BRIDGES_SMALL; i++) {
        scores[i].importance = bridge_importance(&engine->ctx.small.bridges[i]);
        scores[i].index = i;
    }

    for (int i = 0; i < SGFND_BRIDGES_SMALL - 1; i++) {
        for (int j = i + 1; j < SGFND_BRIDGES_SMALL; j++) {
            if (scores[i].importance < scores[j].importance) {
                bridge_score_t tmp = scores[i];
                scores[i] = scores[j];
                scores[j] = tmp;
            }
        }
    }

    for (int i = SGFND_BRIDGES_COMPRESSED; i < SGFND_BRIDGES_SMALL; i++) {
        bridge_destroy_internal(&engine->ctx.small.bridges[scores[i].index]);
    }

    engine->ctx.small.active_count = SGFND_BRIDGES_COMPRESSED;
    engine->ctx.small.compressed = true;

    return 0;
}

int sgfnd_decompress_bridges(sgfnd_engine_t *engine) {
    if (!engine || engine->mode != SGFND_MODE_SMALL) return -1;
    if (!engine->ctx.small.compressed) return 0;

    for (int i = SGFND_BRIDGES_COMPRESSED; i < SGFND_BRIDGES_SMALL; i++) {
        engine->ctx.small.bridges[i] = bridge_create_internal(SGFND_COMPRESSED);
    }
    engine->ctx.small.active_count = SGFND_BRIDGES_SMALL;
    engine->ctx.small.compressed = false;
    return 0;
}

int sgfnd_bridge_quantize(sgfnd_bridge_t *bridge, sgfnd_quantization_t quant) {
    if (!bridge || bridge->size == 0) return -1;
    if (bridge->quant == quant) return 0;

    float min_val = bridge->data[0];
    float max_val = bridge->data[0];
    for (size_t i = 1; i < bridge->size; i++) {
        if (bridge->data[i] < min_val) min_val = bridge->data[i];
        if (bridge->data[i] > max_val) max_val = bridge->data[i];
    }

    if (quant == SGFND_QUANT_INT8) {
        bridge->scale = (max_val - min_val) / 255.0f;
        bridge->zero_point = -min_val / bridge->scale;
        bridge->quant_data = malloc(bridge->size);
        if (!bridge->quant_data) return -1;
        for (size_t i = 0; i < bridge->size; i++) {
            int val = (int)roundf((bridge->data[i] - min_val) / bridge->scale);
            bridge->quant_data[i] = (int8_t)clamp(val, -128, 127);
        }
        free(bridge->data);
        bridge->data = NULL;
        bridge->type = SGFND_INT8;
        bridge->quant = quant;
    } else if (quant == SGFND_QUANT_INT4) {
        bridge->scale = (max_val - min_val) / 15.0f;
        bridge->zero_point = -min_val / bridge->scale;
        size_t packed_size = (bridge->size + 1) / 2;
        bridge->quant_data = malloc(packed_size);
        if (!bridge->quant_data) return -1;
        for (size_t i = 0; i < bridge->size; i += 2) {
            int val1 = (int)roundf((bridge->data[i] - min_val) / bridge->scale);
            int val2 = (i + 1 < bridge->size) ? (int)roundf((bridge->data[i + 1] - min_val) / bridge->scale) : 0;
            val1 = clamp(val1, 0, 15);
            val2 = clamp(val2, 0, 15);
            bridge->quant_data[i / 2] = (val1 & 0xF) | ((val2 & 0xF) << 4);
        }
        free(bridge->data);
        bridge->data = NULL;
        bridge->type = SGFND_INT4;
        bridge->quant = quant;
    }
    return 0;
}

int sgfnd_bridge_dequantize(const sgfnd_bridge_t *bridge, float *out_data, size_t out_size) {
    if (!bridge || !out_data || bridge->quant == SGFND_QUANT_NONE) return -1;
    size_t copy_size = bridge->size < out_size ? bridge->size : out_size;

    if (bridge->quant == SGFND_QUANT_INT8) {
        for (size_t i = 0; i < copy_size; i++) {
            out_data[i] = (bridge->quant_data[i] - bridge->zero_point) * bridge->scale;
        }
    } else if (bridge->quant == SGFND_QUANT_INT4) {
        for (size_t i = 0; i < copy_size; i++) {
            uint8_t packed = bridge->quant_data[i / 2];
            int val = (i % 2 == 0) ? (packed & 0xF) : (packed >> 4);
            out_data[i] = val * bridge->scale - bridge->zero_point * bridge->scale;
        }
    }
    return 0;
}

int sgfnd_large_model_load_weights(sgfnd_engine_t *engine, const char *path) {
    if (!engine || engine->mode != SGFND_MODE_LARGE || !path) return -1;
    engine->ctx.large.weights_path = strdup(path);
    engine->ctx.large.bridge_offsets = calloc(engine->ctx.large.count, sizeof(size_t));
    return 0;
}

int sgfnd_large_model_ensure_bridge(sgfnd_engine_t *engine, size_t index) {
    if (!engine || engine->mode != SGFND_MODE_LARGE || index >= engine->ctx.large.count) return -1;
    if (engine->ctx.large.bridge_loaded[index]) return 0;

    size_t bridge_size = engine->ctx.large.bridges[index].size * sizeof(float);
    if (engine->ctx.large.vram_used + bridge_size > engine->ctx.large.vram_budget) {
        sgfnd_large_model_evict_bridges(engine, engine->ctx.large.count / 2);
    }

    engine->ctx.large.bridge_loaded[index] = true;
    engine->ctx.large.vram_used += bridge_size;
    return 0;
}

void sgfnd_large_model_evict_bridges(sgfnd_engine_t *engine, size_t keep_count) {
    if (!engine || engine->mode != SGFND_MODE_LARGE) return;
    size_t evicted = 0;
    for (size_t i = 0; i < engine->ctx.large.count && evicted < engine->ctx.large.count - keep_count; i++) {
        if (engine->ctx.large.bridge_loaded[i]) {
            sgfnd_bridge_t *b = &engine->ctx.large.bridges[i];
            if (b->data) {
                engine->ctx.large.vram_used -= b->size * sizeof(float);
                free(b->data);
                b->data = NULL;
                b->size = 0;
            }
            engine->ctx.large.bridge_loaded[i] = false;
            evicted++;
        }
    }
}