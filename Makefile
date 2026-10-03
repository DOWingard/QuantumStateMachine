BUILD_DIR := build
BUILD_TYPE ?= Release
JOBS ?= $(shell nproc)

.PHONY: all configure build test run clean distclean

all: build

configure:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)

build: configure
	cmake --build $(BUILD_DIR) -j$(JOBS)

test: build
	ctest --test-dir $(BUILD_DIR) --output-on-failure

run: build
	$(BUILD_DIR)/qstate

clean:
	cmake --build $(BUILD_DIR) --target clean

distclean:
	rm -rf $(BUILD_DIR)
