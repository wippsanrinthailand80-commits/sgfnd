#define _GNU_SOURCE
#include "sgfnd_core.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <dirent.h>
#include <sys/stat.h>
#include <curl/curl.h>
#include <cjson/cJSON.h>
#include <time.h>
#include <math.h>

#include "stb_image.h"

#define METADATA_DIR "training_metadata"
#define MAX_URL_LEN 2048

struct sgfnd_training_bot {
    char storage_path[512];
    CURL *curl;
    char metadata_dir[512];
    sgfnd_dataset_t *dataset;
    size_t images_fetched;
    size_t images_processed;
    float avg_loss;
};

struct MemoryBuffer {
    char *data;
    size_t size;
};

static size_t write_callback(void *contents, size_t size, size_t nmemb, void *userp) {
    size_t realsize = size * nmemb;
    struct MemoryBuffer *mem = (struct MemoryBuffer *)userp;
    char *ptr = realloc(mem->data, mem->size + realsize + 1);
    if (!ptr) return 0;
    mem->data = ptr;
    memcpy(&(mem->data[mem->size]), contents, realsize);
    mem->size += realsize;
    mem->data[mem->size] = 0;
    return realsize;
}

static int ensure_dir(const char *path) {
    struct stat st = {0};
    if (stat(path, &st) == -1) {
        return mkdir(path, 0755);
    }
    return 0;
}

static char* get_metadata_path(const char *base, const char *category) {
    static char path[1024];
    snprintf(path, sizeof(path), "%s/%s.json", base, category);
    return path;
}

static int save_image_to_disk(const char *path, const unsigned char *data, size_t size) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    fwrite(data, 1, size, f);
    fclose(f);
    return 0;
}

static sgfnd_image_t* load_image_from_memory(const unsigned char *data, size_t size) __attribute__((unused));

static sgfnd_image_t* load_image_from_memory(const unsigned char *data, size_t size) {
    sgfnd_image_t *img = sgfnd_image_create_tiled(512, 512, 4, 64);
    if (!img) return NULL;

    size_t pixel_count = 512 * 512 * 4;
    for (size_t i = 0; i < pixel_count && i < size; i++) {
        img->full_pixels[i] = data[i] / 255.0f;
    }
    return img;
}

static sgfnd_image_t* load_image_from_memory_stb(const unsigned char *data, size_t size) {
    int w, h, c;
    unsigned char *pixels = stbi_load_from_memory(data, (int)size, &w, &h, &c, 4);
    if (!pixels) return NULL;

    sgfnd_image_t *img = sgfnd_image_create_tiled(512, 512, 4, 64);
    if (!img) {
        stbi_image_free(pixels);
        return NULL;
    }

    if (w != 512 || h != 512) {
        float scale_x = (float)w / 512.0f;
        float scale_y = (float)h / 512.0f;
        for (uint32_t y = 0; y < 512; y++) {
            for (uint32_t x = 0; x < 512; x++) {
                float src_x = x * scale_x;
                float src_y = y * scale_y;
                uint32_t x0 = (uint32_t)src_x;
                uint32_t y0 = (uint32_t)src_y;
                uint32_t x1 = fminf(w - 1, x0 + 1);
                uint32_t y1 = fminf(h - 1, y0 + 1);
                (void)x1; (void)y1;
                float fx = src_x - x0, fy = src_y - y0;

                for (int ch = 0; ch < 4; ch++) {
                    float v00 = pixels[(y0 * w + x0) * 4 + ch] / 255.0f;
                    float v10 = pixels[(y0 * w + x1) * 4 + ch] / 255.0f;
                    float v01 = pixels[(y1 * w + x0) * 4 + ch] / 255.0f;
                    float v11 = pixels[(y1 * w + x1) * 4 + ch] / 255.0f;

                    float v0 = v00 + fx * (v10 - v00);
                    float v1 = v01 + fx * (v11 - v01);
                    float v = v0 + fy * (v1 - v0);

                    img->full_pixels[(y * 512 + x) * 4 + ch] = v;
                }
            }
        }
    } else {
        for (size_t i = 0; i < 512 * 512 * 4; i++) {
            img->full_pixels[i] = pixels[i] / 255.0f;
        }
    }

    stbi_image_free(pixels);
    return img;
}

static void augment_image(sgfnd_image_t *img) {
    if (!img || !img->full_pixels) return;

    int aug_type = rand() % 4;
    float *pixels = img->full_pixels;
    uint32_t w = img->width, h = img->height, c = img->channels;

    if (aug_type == 0) {
        for (size_t i = 0; i < w * h * c; i++) {
            pixels[i] = fminf(1.0f, pixels[i] * (0.8f + 0.4f * ((float)rand() / RAND_MAX)));
        }
    } else if (aug_type == 1) {
        float *flipped = malloc(w * h * c * sizeof(float));
        for (uint32_t y = 0; y < h; y++) {
            for (uint32_t x = 0; x < w; x++) {
                for (uint8_t ch = 0; ch < c; ch++) {
                    flipped[(y * w + x) * c + ch] = pixels[((h - 1 - y) * w + x) * c + ch];
                }
            }
        }
        memcpy(pixels, flipped, w * h * c * sizeof(float));
        free(flipped);
    } else if (aug_type == 2) {
        for (size_t i = 0; i < w * h * c; i++) {
            pixels[i] += 0.02f * (((float)rand() / RAND_MAX) - 0.5f);
            pixels[i] = fmaxf(0.0f, fminf(1.0f, pixels[i]));
        }
    } else if (aug_type == 3) {
        float shift_u = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        float shift_v = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        float *shifted = malloc(w * h * c * sizeof(float));
        for (uint32_t y = 0; y < h; y++) {
            for (uint32_t x = 0; x < w; x++) {
                float src_x = fmaxf(0, fminf(w - 1, x - shift_u * w));
                float src_y = fmaxf(0, fminf(h - 1, y - shift_v * h));
                uint32_t x0 = (uint32_t)src_x, y0 = (uint32_t)src_y;
                uint32_t x1 = fminf(w - 1, x0 + 1), y1 = fminf(h - 1, y0 + 1);
                float fx = src_x - x0, fy = src_y - y0;
                for (uint8_t ch = 0; ch < c; ch++) {
                    float v00 = pixels[(y0 * w + x0) * c + ch];
                    float v10 = pixels[(y0 * w + x1) * c + ch];
                    float v01 = pixels[(y1 * w + x0) * c + ch];
                    float v11 = pixels[(y1 * w + x1) * c + ch];
                    float v0 = v00 * (1 - fx) + v10 * fx;
                    float v1 = v01 * (1 - fx) + v11 * fx;
                    shifted[(y * w + x) * c + ch] = v0 * (1 - fy) + v1 * fy;
                }
            }
        }
        memcpy(pixels, shifted, w * h * c * sizeof(float));
        free(shifted);
    }
}

sgfnd_training_bot_t* sgfnd_training_bot_create(const char *storage_path) {
    sgfnd_training_bot_t *bot = calloc(1, sizeof(sgfnd_training_bot_t));
    if (!bot) return NULL;

    strncpy(bot->storage_path, storage_path, sizeof(bot->storage_path) - 1);
    snprintf(bot->metadata_dir, sizeof(bot->metadata_dir), "%s/%s", storage_path, METADATA_DIR);

    ensure_dir(storage_path);
    ensure_dir(bot->metadata_dir);

    bot->curl = curl_easy_init();
    if (!bot->curl) {
        free(bot);
        return NULL;
    }

    curl_easy_setopt(bot->curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(bot->curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(bot->curl, CURLOPT_TIMEOUT, 30L);
    curl_easy_setopt(bot->curl, CURLOPT_USERAGENT, "SGFND-TrainingBot/2.0");

    bot->images_fetched = 0;
    bot->images_processed = 0;
    bot->avg_loss = 0.0f;

    return bot;
}

sgfnd_training_bot_t* sgfnd_training_bot_create_v2(const char *storage_path, sgfnd_dataset_t *dataset) {
    sgfnd_training_bot_t *bot = sgfnd_training_bot_create(storage_path);
    if (!bot) return NULL;
    bot->dataset = dataset;
    return bot;
}

void sgfnd_training_bot_destroy(sgfnd_training_bot_t *bot) {
    if (!bot) return;
    if (bot->curl) curl_easy_cleanup(bot->curl);
    free(bot);
}

void sgfnd_training_bot_destroy_v2(sgfnd_training_bot_t *bot) {
    sgfnd_training_bot_destroy(bot);
}

int sgfnd_training_bot_fetch_and_process(sgfnd_training_bot_t *bot, const char *url, sgfnd_color_grader_t *grader) {
    if (!bot || !url || !grader) return -1;

    struct MemoryBuffer chunk = {0};
    curl_easy_setopt(bot->curl, CURLOPT_URL, url);
    curl_easy_setopt(bot->curl, CURLOPT_WRITEDATA, &chunk);

    CURLcode res = curl_easy_perform(bot->curl);
    if (res != CURLE_OK) {
        free(chunk.data);
        return -1;
    }

    bot->images_fetched++;

    char filename[512];
    snprintf(filename, sizeof(filename), "%s/training_%zu.png", bot->storage_path, (size_t)time(NULL));
    save_image_to_disk(filename, (unsigned char*)chunk.data, chunk.size);

    sgfnd_image_t *img = load_image_from_memory_stb((unsigned char*)chunk.data, chunk.size);
    free(chunk.data);

    if (!img) return -1;

    sgfnd_image_resize(img, 512, 512);
    sgfnd_image_normalize(img, 0.5f, 0.5f);

    int ret = sgfnd_color_grader_train(grader, img);

    if (bot->dataset) {
        sgfnd_latent_t *latent = sgfnd_latent_create(512);
        sgfnd_latent_encode(latent, img->full_pixels, 512 * 4);
        sgfnd_dataset_add(bot->dataset, img, latent, "auto-fetched", "color_grader_training");
        sgfnd_latent_destroy(latent);
    }

    bot->images_processed++;
    sgfnd_image_free(img);

    return ret;
}

int sgfnd_training_bot_fetch_and_train(sgfnd_training_bot_t *bot, const char *url, sgfnd_model_t *model, sgfnd_color_grader_t *grader) {
    if (!bot || !url || !model) return -1;

    struct MemoryBuffer chunk = {0};
    curl_easy_setopt(bot->curl, CURLOPT_URL, url);
    curl_easy_setopt(bot->curl, CURLOPT_WRITEDATA, &chunk);

    CURLcode res = curl_easy_perform(bot->curl);
    if (res != CURLE_OK) {
        free(chunk.data);
        return -1;
    }

    bot->images_fetched++;

    char filename[512];
    snprintf(filename, sizeof(filename), "%s/training_%zu.png", bot->storage_path, (size_t)time(NULL));
    save_image_to_disk(filename, (unsigned char*)chunk.data, chunk.size);

    sgfnd_image_t *img = load_image_from_memory_stb((unsigned char*)chunk.data, chunk.size);
    free(chunk.data);

    if (!img) return -1;

    sgfnd_image_resize(img, 512, 512);
    sgfnd_image_normalize(img, 0.5f, 0.5f);

    if (grader) {
        sgfnd_color_grader_train(grader, img);
    }

    sgfnd_prompt_t *prompt = sgfnd_prompt_create_from_text("eyes, hair, skin, clothing, lighting, background, style", 1.0f);
    if (!prompt) return -1;

    sgfnd_training_step_t step_info;
    int ret = sgfnd_model_train_step(model, img, prompt, &step_info);

    bot->avg_loss = (bot->avg_loss * bot->images_processed + step_info.loss) / (bot->images_processed + 1);
    bot->images_processed++;

    if (bot->dataset) {
        sgfnd_latent_t *latent = sgfnd_latent_create(model->latent->dim);
        sgfnd_latent_encode(latent, img->full_pixels, 512 * 4);
        sgfnd_dataset_add(bot->dataset, img, latent, "auto-trained", "diffusion_training");
        sgfnd_latent_destroy(latent);
    }

    sgfnd_prompt_destroy(prompt);

    sgfnd_image_free(img);
    return ret;
}

int sgfnd_training_bot_augment_dataset(sgfnd_training_bot_t *bot, sgfnd_dataset_t *dataset) {
    if (!bot || !dataset) return -1;

    size_t original_count = dataset->count;
    for (size_t i = 0; i < original_count && dataset->count < dataset->capacity; i++) {
        sgfnd_image_t *img = sgfnd_image_create_tiled(dataset->images[i]->width, dataset->images[i]->height, dataset->images[i]->channels, 64);
        if (!img) continue;

        memcpy(img->full_pixels, dataset->images[i]->full_pixels, dataset->images[i]->width * dataset->images[i]->height * dataset->images[i]->channels * sizeof(float));

        augment_image(img);

        sgfnd_latent_t *latent = sgfnd_latent_create(dataset->latents[i]->dim);
        memcpy(latent->latent, dataset->latents[i]->latent, latent->dim * sizeof(float));
        memcpy(latent->mu, dataset->latents[i]->mu, latent->dim * sizeof(float));
        memcpy(latent->logvar, dataset->latents[i]->logvar, latent->dim * sizeof(float));

        char aug_prompt[256], aug_meta[256];
        snprintf(aug_prompt, sizeof(aug_prompt), "%s_aug", dataset->prompts[i] ? dataset->prompts[i] : "unknown");
        snprintf(aug_meta, sizeof(aug_meta), "%s_augmented", dataset->metadata[i] ? dataset->metadata[i] : "none");

        sgfnd_dataset_add(dataset, img, latent, aug_prompt, aug_meta);
        sgfnd_latent_destroy(latent);
        sgfnd_image_free(img);
    }
    return 0;
}

int sgfnd_training_bot_save_metadata(sgfnd_training_bot_t *bot, const char *category, const char *key, const char *value) {
    if (!bot || !category || !key || !value) return -1;

    char *path = get_metadata_path(bot->metadata_dir, category);
    cJSON *root = NULL;

    FILE *f = fopen(path, "r");
    if (f) {
        fseek(f, 0, SEEK_END);
        long len = ftell(f);
        fseek(f, 0, SEEK_SET);
        char *buf = malloc(len + 1);
        fread(buf, 1, len, f);
        buf[len] = 0;
        fclose(f);
        root = cJSON_Parse(buf);
        free(buf);
    }
    if (!root) root = cJSON_CreateObject();

    cJSON *obj = cJSON_GetObjectItem(root, key);
    if (obj) cJSON_ReplaceItemInObject(root, key, cJSON_CreateString(value));
    else cJSON_AddItemToObject(root, key, cJSON_CreateString(value));

    cJSON *stats = cJSON_CreateObject();
    cJSON_AddNumberToObject(stats, "images_fetched", bot->images_fetched);
    cJSON_AddNumberToObject(stats, "images_processed", bot->images_processed);
    cJSON_AddNumberToObject(stats, "avg_loss", bot->avg_loss);
    cJSON_AddItemToObject(root, "_bot_stats", stats);

    char *json_str = cJSON_Print(root);
    f = fopen(path, "w");
    if (f) {
        fputs(json_str, f);
        fclose(f);
    }
    free(json_str);
    cJSON_Delete(root);
    return 0;
}

const char* sgfnd_training_bot_get_metadata(sgfnd_training_bot_t *bot, const char *category, const char *key) {
    if (!bot || !category || !key) return NULL;

    char *path = get_metadata_path(bot->metadata_dir, category);
    FILE *f = fopen(path, "r");
    if (!f) return NULL;

    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc(len + 1);
    fread(buf, 1, len, f);
    buf[len] = 0;
    fclose(f);

    cJSON *root = cJSON_Parse(buf);
    free(buf);
    if (!root) return NULL;

    cJSON *val = cJSON_GetObjectItem(root, key);
    const char *result = val ? val->valuestring : NULL;
    cJSON_Delete(root);

    static char static_result[1024];
    if (result) {
        strncpy(static_result, result, sizeof(static_result) - 1);
        return static_result;
    }
    return NULL;
}

size_t sgfnd_training_bot_get_fetched(const sgfnd_training_bot_t *bot) {
    return bot ? bot->images_fetched : 0;
}

size_t sgfnd_training_bot_get_processed(const sgfnd_training_bot_t *bot) {
    return bot ? bot->images_processed : 0;
}

float sgfnd_training_bot_get_avg_loss(const sgfnd_training_bot_t *bot) {
    return bot ? bot->avg_loss : 0.0f;
}

sgfnd_dataset_t* sgfnd_training_bot_get_dataset(const sgfnd_training_bot_t *bot) {
    return bot ? bot->dataset : NULL;
}