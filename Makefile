CC = gcc
CFLAGS = -std=c11 -Wall -Wextra -O3 -march=native -flto -fPIC
CFLAGS += -Iinclude
LDFLAGS = -lm -lcurl -lcjson

DBG_CFLAGS = -std=c11 -Wall -Wextra -g -O0 -fsanitize=address,undefined -fno-omit-frame-pointer -fno-optimize-sibling-calls
DBG_CFLAGS += -Iinclude -DDEBUG -D_FORTIFY_SOURCE=2
DBG_LDFLAGS = -fsanitize=address,undefined -lm -lcurl -lcjson

TSAN_CFLAGS = -std=c11 -Wall -Wextra -g -O1 -fsanitize=thread -fno-omit-frame-pointer
TSAN_CFLAGS += -Iinclude -DDEBUG
TSAN_LDFLAGS = -fsanitize=thread -lm -lcurl -lcjson

SRC_DIR = src
BUILD_DIR = build
DBG_BUILD_DIR = build_dbg
TSAN_BUILD_DIR = build_tsan

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
DBG_OBJECTS = $(SOURCES:$(SRC_DIR)/%.c=$(DBG_BUILD_DIR)/%.o)
TSAN_OBJECTS = $(SOURCES:$(SRC_DIR)/%.c=$(TSAN_BUILD_DIR)/%.o)

TARGET = sgfnd
DBG_TARGET = sgfnd_dbg
TSAN_TARGET = sgfnd_tsan

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

debug: $(DBG_TARGET)

$(DBG_TARGET): $(DBG_OBJECTS)
	$(CC) $(DBG_CFLAGS) -o $@ $^ $(DBG_LDFLAGS)

$(DBG_BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(DBG_CFLAGS) -c $< -o $@

tsan: $(TSAN_TARGET)

$(TSAN_TARGET): $(TSAN_OBJECTS)
	$(CC) $(TSAN_CFLAGS) -o $@ $^ $(TSAN_LDFLAGS)

$(TSAN_BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(TSAN_CFLAGS) -c $< -o $@

valgrind: $(DBG_TARGET)
	valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all --track-origins=yes --error-exitcode=1 ./$(DBG_TARGET) $(ARGS)

gdb: $(DBG_TARGET)
	gdb -ex "set confirm off" -ex "run $(ARGS)" -ex "bt full" -ex "quit" --args ./$(DBG_TARGET) $(ARGS)

clean:
	rm -rf $(BUILD_DIR) $(DBG_BUILD_DIR) $(TSAN_BUILD_DIR) $(TARGET) $(DBG_TARGET) $(TSAN_TARGET)

install: $(TARGET)
	cp $(TARGET) /usr/local/bin/

.PHONY: all debug tsan valgrind gdb clean install