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
else
SHLIB_EXT := .so
SHLIB_FLAGS := -shared
endif

LIB := $(BUILD_DIR)/libholoc.a
SHLIB := $(BUILD_DIR)/libholoc$(SHLIB_EXT)
OBJS := $(BUILD_DIR)/holo_core.o $(BUILD_DIR)/holo_trace.o
TESTS := $(BUILD_DIR)/test_core $(BUILD_DIR)/test_trace
BENCHES := $(BUILD_DIR)/bench_trace

.PHONY: all shared test bench bench-compare clean

all: $(LIB) $(SHLIB)

shared: $(SHLIB)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

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

$(BUILD_DIR)/bench_trace: $(BENCH_DIR)/bench_trace.c $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LIB) $(LDLIBS) -o $@

test: $(TESTS)
	./$(BUILD_DIR)/test_core
	./$(BUILD_DIR)/test_trace

bench: $(BENCHES)
	./$(BUILD_DIR)/bench_trace

bench-compare: $(BENCHES)
	$(PYTHON) benchmarks/bench_trace.py --summary --binary ./$(BUILD_DIR)/bench_trace

clean:
	rm -rf build
