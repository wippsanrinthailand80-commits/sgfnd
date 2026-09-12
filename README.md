# SGFND - Synthetic Generative Framework Neural Dynamics

**⚠️ EXPERIMENTAL PROJECT** - This is a research prototype, not production software.

## Overview

SGFND is a from-scratch neural image generation framework written in C99. It implements:

- **Bridge Engine**: Compressed feature representation (92 → 46 bridges)
- **Latent Diffusion**: 512-dim latent space, 1000 timesteps, DDPM sampling
- **Tiled Rendering**: 64×64 tiles with streaming to disk (low VRAM)
- **Large Model Streaming**: 30,000 bridges with on-demand VRAM loading
- **Training Bot**: Auto-fetches images from URLs, augments, trains diffusion
- **NSFW Filter**: Keyword-based prompt blocking + pixel-analysis image sanitization
- **Quantization**: INT8/INT4 bridge compression

## Key Strengths

- **Built from scratch**: No PyTorch/TensorFlow dependencies
- **Low memory footprint**: Runs in ~512MB VRAM
- **ARM/mobile compatible**: Pure C99, no SIMD intrinsics, portable math
- **Training from URLs**: Point at image datasets, trains automatically
- **Weight persistence**: Save/load full model state for resuming

## Quick Start

```bash
# Install dependencies
apt-get install libcurl4-openssl-dev libcjson-dev

# Build
make

# Generate images
./sgfnd --generate                    # Small model (92→46 bridges)
./sgfnd --generate --tiled            # Tiled streaming (64×64)
./sgfnd --generate --large            # Large model (30K bridges)
./sgfnd --generate --nsfw enabled     # With content filter
```

## On Kaggle/Colab

```python
!git clone https://github.com/wippsanrinthailand80-commits/sgfnd.git
%cd sgfnd
!apt-get update -qq && apt-get install -y -qq libcurl4-openssl-dev libcjson-dev
!make
!./sgfnd --generate --nsfw enabled
```

See `sgfnd_kaggle.ipynb` and `SGFND_Colab.ipynb` for complete notebooks.

## Architecture

```
src/
├── main.c                      # CLI, argument parsing, NSFW integration
├── bridges/bridge_engine.c     # Bridge processing & compression
├── color/color_grader.c        # HSV-based color grading
├── generative/
│   ├── latent_diffusion.c      # MLP + DDPM diffusion
│   └── model_training.c        # VAE, decoder, training loop
├── io/
│   ├── image_io.c              # Tiled image I/O, rendering
│   └── stb_image_impl.c        # Image loading (stb_image)
├── large_model/large_model.c   # 30K bridge streaming
├── safety/nsfw_filter.c        # NSFW prompt/image detection
└── training/training_bot.c     # Auto-fetch, augment, train
```

## Prompt System

```c
// From text (comma-separated)
sgfnd_prompt_t *prompt = sgfnd_prompt_create_from_text(
    "portrait, blue eyes, blonde hair, soft lighting", 1.0f);

// From arrays
const char *tags[] = {"eyes", "hair", "skin"};
const float weights[] = {1.0f, 0.8f, 0.9f};
sgfnd_prompt_t *prompt = sgfnd_prompt_create_from_tags(tags, weights, 3);

// Cleanup
sgfnd_prompt_destroy(prompt);
```

## Image Loading

```c
// Load and resize from file (PNG, JPG, BMP, etc.)
sgfnd_image_t *img = sgfnd_image_load_from_file("input.png", 512);

// Resize existing image
sgfnd_image_resize(img, 256, 256);

// Normalize to [-1, 1] range
sgfnd_image_normalize(img, 0.5f, 0.5f);
```

## Model Persistence

```c
// Save full state (weights + optimizer + step count)
sgfnd_model_save_full(model, "model.bin");

// Load full state for resume
sgfnd_model_load_full(model, "model.bin");
```

## NSFW Filter

```bash
# Modes: disabled (default), enabled, strict
./sgfnd --generate --nsfw enabled
./sgfnd --generate --nsfw strict --nsfw-thresh 0.3
```

## Memory Safety

- Valgrind clean: 0 errors, 0 leaks
- AddressSanitizer / ThreadSanitizer builds available
- All allocations tracked and freed

```bash
make debug      # ASan + UBSan build
make tsan       # ThreadSanitizer build
make valgrind ARGS="--generate --nsfw enabled"
make gdb ARGS="--generate"
```

## Training

```bash
./sgfnd --train --url https://example.com/dataset.zip \
        --epochs 10 --lr 1e-4 --batch 4 \
        --generate --nsfw enabled
```

Training bot:
1. Fetches images from URL
2. Loads with stb_image (PNG/JPG/BMP/GIF)
3. Resizes to 512×512, normalizes
4. Trains color grader + diffusion
4. Augments dataset (flip, noise, shift)
5. Saves weights periodically

## Requirements

- GCC/Clang (C99)
- libcurl (for training)
- libcjson (for metadata)
- Linux/macOS/Windows (WSL)

## License

MIT License - See LICENSE file

## Disclaimer

This is experimental research code. Generated images may not be coherent.
The diffusion path is under active development. Not suitable for production use.