CC ?= cc
AR ?= ar
PYTHON ?= python3
HOLO_USE_ACCELERATE ?= 0
UNAME_S := $(shell uname -s)

BUILD_FLAVOR := scalar
ifeq ($(HOLO_USE_ACCELERATE),1)
BUILD_FLAVOR := accelerate
CPPFLAGS += -DHOLO_USE_ACCELERATE=1
LDLIBS += -framework Accelerate
endif

BUILD_DIR := build/$(BUILD_FLAVOR)
SOKOL_DIR ?= ../vendor/sokol
SOKOL_BUILD_DIR := build/sokol
SOKOL_ASTEROIDS := $(SOKOL_BUILD_DIR)/holo_sokol_asteroids
INCLUDE_DIR := include
SRC_DIR := src
TEST_DIR := tests
BENCH_DIR := benchmarks

CFLAGS ?= -std=c99 -O3 -Wall -Wextra -Wpedantic
PICFLAGS ?= -fPIC
CPPFLAGS += -I$(INCLUDE_DIR)
LDLIBS += -lm

ifeq ($(UNAME_S),Darwin)
SHLIB_EXT := .dylib
SHLIB_FLAGS := -dynamiclib -install_name @rpath/libholoc$(SHLIB_EXT)
SOKOL_SOURCE_MODE := -x objective-c
SOKOL_LDLIBS := -framework Cocoa -framework QuartzCore -framework OpenGL
else
SHLIB_EXT := .so
SHLIB_FLAGS := -shared
SOKOL_SOURCE_MODE :=
SOKOL_LDLIBS := -lX11 -lXi -lXcursor -lGL -ldl -lpthread -lm
endif

LIB := $(BUILD_DIR)/libholoc.a
SHLIB := $(BUILD_DIR)/libholoc$(SHLIB_EXT)
OBJS := $(BUILD_DIR)/holo_core.o $(BUILD_DIR)/holo_trace.o $(BUILD_DIR)/holo_program.o
TESTS := $(BUILD_DIR)/test_core $(BUILD_DIR)/test_trace $(BUILD_DIR)/test_program
BENCHES := $(BUILD_DIR)/bench_trace

.PHONY: all shared test bench bench-fixed bench-program bench-compare sokol-asteroids sokol-run clean

all: $(LIB) $(SHLIB)

shared: $(SHLIB)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(SOKOL_BUILD_DIR):
	mkdir -p $(SOKOL_BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(PICFLAGS) -c $< -o $@

$(LIB): $(OBJS)
	$(AR) rcs $@ $^

$(SHLIB): $(OBJS)
	$(CC) $(SHLIB_FLAGS) $^ $(LDLIBS) -o $@

$(BUILD_DIR)/test_core: $(TEST_DIR)/test_core.c $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LIB) $(LDLIBS) -o $@

$(BUILD_DIR)/test_trace: $(TEST_DIR)/test_trace.c $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LIB) $(LDLIBS) -o $@

$(BUILD_DIR)/test_program: $(TEST_DIR)/test_program.c $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LIB) $(LDLIBS) -o $@

$(BUILD_DIR)/bench_trace: $(BENCH_DIR)/bench_trace.c $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LIB) $(LDLIBS) -o $@

test: $(TESTS)
	./$(BUILD_DIR)/test_core
	./$(BUILD_DIR)/test_trace
	./$(BUILD_DIR)/test_program

bench: $(BENCHES)
	./$(BUILD_DIR)/bench_trace

bench-fixed: $(SHLIB)
	HOLOSTUFF_C_LIB=$(abspath $(SHLIB)) $(PYTHON) benchmarks/bench_bind_fixed.py --summary

bench-program: $(SHLIB)
	HOLOSTUFF_C_LIB=$(abspath $(SHLIB)) $(PYTHON) benchmarks/bench_program.py --summary

bench-compare: $(BENCHES) $(SHLIB)
	$(PYTHON) benchmarks/bench_trace.py --summary --binary ./$(BUILD_DIR)/bench_trace
	HOLOSTUFF_C_LIB=$(abspath $(SHLIB)) $(PYTHON) benchmarks/bench_bind_fixed.py --summary
	HOLOSTUFF_C_LIB=$(abspath $(SHLIB)) $(PYTHON) benchmarks/bench_program.py --summary

sokol-asteroids: $(SOKOL_ASTEROIDS)

sokol-run: $(SOKOL_ASTEROIDS)
	./$(SOKOL_ASTEROIDS)

$(SOKOL_ASTEROIDS): sokol/holo_sokol_asteroids.c $(SOKOL_DIR)/sokol_app.h $(SOKOL_DIR)/sokol_gfx.h $(SOKOL_DIR)/sokol_glue.h $(SOKOL_DIR)/sokol_log.h | $(SOKOL_BUILD_DIR)
	$(CC) $(SOKOL_SOURCE_MODE) $(CPPFLAGS) -I$(SOKOL_DIR) $(CFLAGS) $< $(SOKOL_LDLIBS) -o $@

clean:
	rm -rf build
