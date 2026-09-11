CC = gcc
CFLAGS = -std=c11 -Wall -Wextra -O3 -march=native -flto -fPIC
CFLAGS += -Iinclude
LDFLAGS = -lm -lcurl -lcjson

SRC_DIR = src
BUILD_DIR = build

SOURCES = $(SRC_DIR)/main.c \
          $(SRC_DIR)/bridges/bridge_engine.c \
          $(SRC_DIR)/color/color_grader.c \
          $(SRC_DIR)/training/training_bot.c \
          $(SRC_DIR)/io/image_io.c \
          $(SRC_DIR)/large_model/large_model.c \
          $(SRC_DIR)/generative/latent_diffusion.c \
          $(SRC_DIR)/generative/model_training.c \
          $(SRC_DIR)/safety/nsfw_filter.c

OBJECTS = $(SOURCES:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)

TARGET = sgfnd

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

install: $(TARGET)
	cp $(TARGET) /usr/local/bin/

.PHONY: all clean install