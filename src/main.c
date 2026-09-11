#define _GNU_SOURCE
#include "sgfnd_core.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void tile_write_callback(const sgfnd_tile_t *tile, void *user_data) {
    FILE *f = (FILE*)user_data;
    if (!f) return;

    size_t pixel_count = tile->width * tile->height * 4;
    uint8_t *buf = malloc(pixel_count);
    for (size_t i = 0; i < pixel_count; i++) {
        buf[i] = (uint8_t)(tile->pixels[i] * 255.0f);
    }
    fwrite(buf, 1, pixel_count, f);
    free(buf);
}

void print_usage(const char *prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --large           Use large model (30,000 bridges)\n");
    printf("  --small           Use small model (92 bridges -> 46 compressed) [default]\n");
    printf("  --int8            Quantize to INT8\n");
    printf("  --int4            Quantize to INT4\n");
    printf("  --quantize        Apply quantization after input processing\n");
    printf("  --tiled           Use tiled streaming renderer\n");
    printf("  --train           Enable training mode\n");
    printf("  --generate        Generate image after training\n");
    printf("  --steps N         Diffusion sampling steps (default: 50)\n");
    printf("  --epochs N        Training epochs (default: 10)\n");
    printf("  --lr F            Learning rate (default: 1e-4)\n");
    printf("  --batch N         Batch size (default: 4)\n");
    printf("  --url URL         Fetch and train from URL\n");
    printf("  --nsfw MODE       NSFW filter: disabled|enabled|strict (default: disabled)\n");
    printf("  --nsfw-thresh F   NSFW threshold 0.0-1.0 (default: 0.5)\n");
}

int main(int argc, char **argv) {
    SGFND_LOG_INFO("Starting SGFND");
    printf("SGFND - Synthetic Generative Framework Neural Dynamics\n");
    printf("========================================================\n\n");

    sgfnd_mode_t mode = SGFND_MODE_SMALL;
    size_t vram_size = 512 * 1024 * 1024;
    sgfnd_quantization_t quant = SGFND_QUANT_NONE;
    bool use_tiled = false;
    bool quantize_after = false;
    bool train_mode = false;
    bool do_generate = false;
    int diff_steps = 50;
    int epochs = 10;
    float lr = 1e-4f;
    int batch_size = 4;
    const char *train_url = NULL;
    sgfnd_nsfw_mode_t nsfw_mode = SGFND_NSFW_FILTER_DISABLED;
    float nsfw_threshold = 0.5f;

    SGFND_LOG_DEBUG("Parsing %d arguments", argc);
    for (int i = 1; i < argc; i++) {
        SGFND_LOG_DEBUG("Arg %d: %s", i, argv[i]);
        if (strcmp(argv[i], "--large") == 0) mode = SGFND_MODE_LARGE;
        else if (strcmp(argv[i], "--int8") == 0) quant = SGFND_QUANT_INT8;
        else if (strcmp(argv[i], "--int4") == 0) quant = SGFND_QUANT_INT4;
        else if (strcmp(argv[i], "--quantize") == 0) quantize_after = true;
        else if (strcmp(argv[i], "--tiled") == 0) use_tiled = true;
        else if (strcmp(argv[i], "--train") == 0) train_mode = true;
        else if (strcmp(argv[i], "--generate") == 0) do_generate = true;
        else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) diff_steps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) epochs = atoi(argv[++i]);
        else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) lr = atof(argv[++i]);
        else if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) batch_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--url") == 0 && i + 1 < argc) train_url = argv[++i];
        else if (strcmp(argv[i], "--nsfw") == 0 && i + 1 < argc) {
            const char *m = argv[++i];
            if (strcmp(m, "enabled") == 0) nsfw_mode = SGFND_NSFW_FILTER_ENABLED;
            else if (strcmp(m, "strict") == 0) nsfw_mode = SGFND_NSFW_FILTER_STRICT;
            else nsfw_mode = SGFND_NSFW_FILTER_DISABLED;
        }
        else if (strcmp(argv[i], "--nsfw-thresh") == 0 && i + 1 < argc) nsfw_threshold = atof(argv[++i]);
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (mode == SGFND_MODE_LARGE) {
        printf("Mode: LARGE (30,000 bridges, streaming)\n");
    } else {
        printf("Mode: SMALL (92 bridges -> 46 compressed)\n");
    }
    if (quant != SGFND_QUANT_NONE) printf("Quantization: %s\n", quant == SGFND_QUANT_INT8 ? "INT8" : "INT4");
    if (use_tiled) printf("Rendering: Tiled streaming (64x64 tiles)\n");
    if (train_mode) printf("Training: ENABLED (epochs=%d, lr=%.2e, batch=%d, steps=%d)\n", epochs, lr, batch_size, diff_steps);

    sgfnd_engine_t *engine = sgfnd_create(mode, vram_size);
    if (!engine) {
        fprintf(stderr, "Failed to create engine\n");
        return 1;
    }
    engine->default_quant = quant;

    sgfnd_color_grader_t *grader = sgfnd_color_grader_create();
    if (!grader) {
        fprintf(stderr, "Failed to create color grader\n");
        sgfnd_destroy(engine);
        return 1;
    }

    sgfnd_nsfw_filter_t *nsfw_filter = NULL;
    if (nsfw_mode != SGFND_NSFW_FILTER_DISABLED) {
        nsfw_filter = sgfnd_nsfw_filter_create(nsfw_mode, nsfw_threshold);
        if (!nsfw_filter) {
            fprintf(stderr, "Failed to create NSFW filter\n");
            sgfnd_color_grader_destroy(grader);
            sgfnd_destroy(engine);
            return 1;
        }
        printf("NSFW Filter: %s (threshold=%.2f)\n",
               nsfw_mode == SGFND_NSFW_FILTER_STRICT ? "STRICT" : "ENABLED",
               nsfw_threshold);
    }

    sgfnd_dataset_t *dataset = sgfnd_dataset_create(512, 512, 10000);
    if (!dataset) {
        fprintf(stderr, "Failed to create dataset\n");
        sgfnd_color_grader_destroy(grader);
        sgfnd_destroy(engine);
        return 1;
    }

    sgfnd_training_bot_t *bot = sgfnd_training_bot_create_v2("./training_data", dataset);
    if (!bot) {
        fprintf(stderr, "Failed to create training bot\n");
        sgfnd_dataset_destroy(dataset);
        sgfnd_color_grader_destroy(grader);
        sgfnd_destroy(engine);
        return 1;
    }

    sgfnd_model_t *model = NULL;
    if (train_mode || do_generate) {
        model = sgfnd_model_create(512, 7, 1000, lr);
        if (!model) {
            fprintf(stderr, "Failed to create generative model\n");
        } else {
            printf("Generative model created: latent_dim=512, timesteps=1000\n");
        }
    }

    printf("\nEngine initialized.\n");
    printf("VRAM pool: %zu MB\n", vram_size / (1024 * 1024));

    float test_input[1024];
    for (int i = 0; i < 1024; i++) {
        test_input[i] = (float)i / 1024.0f;
    }

    printf("\nProcessing test input (input 1 of 3)...\n");
    sgfnd_process_input(engine, test_input, 1024);

    if (mode == SGFND_MODE_SMALL) {
        printf("Compressing bridges...\n");
        sgfnd_compress_bridges(engine);
        printf("Active bridges: %zu\n", engine->ctx.small.active_count);
    }

    if (quantize_after && mode == SGFND_MODE_SMALL) {
        printf("Quantizing bridges to %s...\n", quant == SGFND_QUANT_INT8 ? "INT8" : "INT4");
        for (size_t i = 0; i < engine->ctx.small.active_count; i++) {
            sgfnd_bridge_quantize(&engine->ctx.small.bridges[i], quant);
        }
        printf("Quantization complete.\n");
    } else if (quantize_after && mode == SGFND_MODE_LARGE) {
        printf("Quantizing loaded bridges to %s...\n", quant == SGFND_QUANT_INT8 ? "INT8" : "INT4");
        for (size_t i = 0; i < engine->ctx.large.count; i++) {
            if (engine->ctx.large.bridge_loaded[i]) {
                sgfnd_bridge_quantize(&engine->ctx.large.bridges[i], quant);
            }
        }
        printf("Quantization complete.\n");
    }

    sgfnd_prompt_t prompt = {0};
    prompt.tags = malloc(7 * sizeof(char*));
    prompt.weights = malloc(7 * sizeof(float));
    prompt.count = 7;
    const char *tags[] = {"eyes", "hair", "skin", "clothing", "lighting", "background", "style"};
    for (int i = 0; i < 7; i++) {
        prompt.tags[i] = strdup(tags[i]);
        prompt.weights[i] = 1.0f;
    }

    if (nsfw_filter && sgfnd_nsfw_check_prompt(nsfw_filter, &prompt)) {
        printf("\nNSFW Filter: Prompt blocked - contains flagged content\n");
        nsfw_filter->blocked_prompts++;
        // Continue anyway for demo purposes
    }

    if (train_mode && train_url) {
        printf("\nFetching and training from URL: %s\n", train_url);
        for (int epoch = 0; epoch < epochs; epoch++) {
            sgfnd_training_step_t step_info = {0};
            int ret = sgfnd_training_bot_fetch_and_train(bot, train_url, model, grader);
            if (ret == 0) {
                printf("  Epoch %d/%d: loss=%.6f (recon=%.6f, kl=%.6f, adv=%.6f)\n",
                       epoch + 1, epochs, step_info.loss, step_info.recon_loss, step_info.kl_loss, step_info.adv_loss);
            } else {
                fprintf(stderr, "  Epoch %d/%d: fetch/train failed\n", epoch + 1, epochs);
            }
        }
        printf("Training complete. Avg loss: %.6f\n", sgfnd_training_bot_get_avg_loss(bot));

        if (sgfnd_training_bot_get_dataset(bot) && sgfnd_training_bot_get_dataset(bot)->count > 0) {
            printf("Augmenting dataset...\n");
            sgfnd_training_bot_augment_dataset(bot, dataset);
            printf("Dataset size: %zu\n", dataset->count);
        }

        sgfnd_model_save_weights(model, "sgfnd_model_weights.bin");
        printf("Model weights saved.\n");
    }

    if (do_generate || !train_mode) {
        printf("\nGenerating image...\n");

        if (model && do_generate) {
            sgfnd_image_t *img = sgfnd_image_create_tiled(512, 512, 4, 64);
            int ret = sgfnd_model_generate(model, &prompt, img, diff_steps);
            if (ret == 0) {
                printf("Diffusion generation complete (%d steps).\n", diff_steps);
                if (nsfw_filter) {
                    float score = sgfnd_nsfw_check_image(nsfw_filter, img);
                    printf("NSFW score: %.3f\n", score);
                    if (score > nsfw_threshold) {
                        printf("NSFW Filter: Image flagged, sanitizing...\n");
                        sgfnd_nsfw_sanitize_image(nsfw_filter, img);
                        nsfw_filter->flagged_images++;
                        nsfw_filter->sanitized_images++;
                    }
                }
                sgfnd_image_save_raw(img, "generated_cuda_image.raw");
                printf("Saved to generated_cuda_image.raw\n");
            }
            sgfnd_image_free(img);
        } else if (use_tiled) {
            printf("Using tiled streaming renderer...\n");
            FILE *f = fopen("generated_tiled.raw", "wb");
            if (f) {
                uint32_t header[4] = {512, 512, 4, 8};
                fwrite(header, sizeof(uint32_t), 4, f);

                sgfnd_render_config_t config = {
                    .tile_size = 64,
                    .max_tiles_in_vram = 16,
                    .tile_callback = tile_write_callback,
                    .user_data = f
                };
                sgfnd_render_tiled(engine, &prompt, &config);
                fclose(f);
                printf("Saved tiled output to generated_tiled.raw\n");
            }
        } else {
            sgfnd_image_t *img = sgfnd_generate_image(engine, &prompt);
            if (img) {
                printf("Image generated: %ux%u, %d channels, %d-bit\n",
                       img->width, img->height, img->channels, img->bit_depth);
                printf("Tiles: %zu (%ux%u each)\n", img->tile_count, img->tile_size, img->tile_size);

                if (nsfw_filter) {
                    float score = sgfnd_nsfw_check_image(nsfw_filter, img);
                    printf("NSFW score: %.3f\n", score);
                    if (score > nsfw_threshold) {
                        printf("NSFW Filter: Image flagged, sanitizing...\n");
                        sgfnd_nsfw_sanitize_image(nsfw_filter, img);
                        nsfw_filter->flagged_images++;
                        nsfw_filter->sanitized_images++;
                    }
                }

                sgfnd_image_save_raw(img, "generated_cuda_image.raw");
                printf("Saved to generated_cuda_image.raw\n");

                sgfnd_image_free(img);
            }
        }
    }

    if (mode == SGFND_MODE_LARGE) {
        printf("\nLarge model streaming stats:\n");
        printf("  VRAM used: %zu MB / %zu MB\n",
               engine->ctx.large.vram_used / (1024 * 1024),
               engine->ctx.large.vram_budget / (1024 * 1024));
        size_t loaded = 0;
        for (size_t i = 0; i < engine->ctx.large.count; i++) {
            if (engine->ctx.large.bridge_loaded[i]) loaded++;
        }
        printf("  Bridges loaded: %zu / %zu\n", loaded, engine->ctx.large.count);
    }

    if (model) {
        printf("\nGenerative model stats:\n");
        printf("  Training steps: %zu\n", model->step_count);
        printf("  Learning rate: %.2e\n", model->learning_rate);
    }

    if (dataset) {
        printf("\nDataset stats:\n");
        printf("  Images: %zu / %zu\n", dataset->count, dataset->capacity);
    }

    printf("\nTraining bot stats:\n");
    printf("  Fetched: %zu\n", sgfnd_training_bot_get_fetched(bot));
    printf("  Processed: %zu\n", sgfnd_training_bot_get_processed(bot));
    printf("  Avg loss: %.6f\n", sgfnd_training_bot_get_avg_loss(bot));

    if (nsfw_filter) {
        printf("\nNSFW Filter stats:\n");
        printf("  Blocked prompts: %zu\n", nsfw_filter->blocked_prompts);
        printf("  Flagged images: %zu\n", nsfw_filter->flagged_images);
        printf("  Sanitized images: %zu\n", nsfw_filter->sanitized_images);
        sgfnd_nsfw_filter_destroy(nsfw_filter);
    }

    for (int i = 0; i < 7; i++) free(prompt.tags[i]);
    free(prompt.tags);
    free(prompt.weights);

    if (model) sgfnd_model_destroy(model);
    sgfnd_training_bot_destroy_v2(bot);
    sgfnd_dataset_destroy(dataset);
    sgfnd_color_grader_destroy(grader);
    sgfnd_destroy(engine);

    printf("\nDone.\n");
    return 0;
}