#ifndef SGFND_BRIDGE_INTERNAL_H
#define SGFND_BRIDGE_INTERNAL_H

#include "sgfnd_core.h"
#include <stdlib.h>

#define BRIDGE_INIT_CAPACITY 1024

static inline sgfnd_bridge_t bridge_create_internal(sgfnd_bridge_type_t type) {
    sgfnd_bridge_t b = {0};
    b.type = type;
    b.capacity = BRIDGE_INIT_CAPACITY;
    b.data = calloc(b.capacity, sizeof(float));
    return b;
}

static inline void bridge_destroy_internal(sgfnd_bridge_t *b) {
    if (b && b->data) {
        free(b->data);
        b->data = NULL;
    }
    b->size = 0;
    b->capacity = 0;
}

static inline int bridge_push_internal(sgfnd_bridge_t *b, float value) {
    if (!b) return -1;
    if (b->size >= b->capacity) {
        size_t new_cap = b->capacity * 2;
        float *new_data = realloc(b->data, new_cap * sizeof(float));
        if (!new_data) return -1;
        b->data = new_data;
        b->capacity = new_cap;
    }
    b->data[b->size++] = value;
    return 0;
}

#endif