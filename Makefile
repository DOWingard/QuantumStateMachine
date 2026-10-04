# Thin wrapper over CMakePresets.json; each build type gets its own tree under build/.
BUILD_TYPE ?= Release
PRESET := $(shell echo $(BUILD_TYPE) | tr '[:upper:]' '[:lower:]')
BUILD_DIR := build/$(PRESET)
JOBS ?= $(shell nproc)

.PHONY: all configure build test run examples docs clean distclean

all: build

docs:
	doxygen Doxyfile

configure:
	cmake --preset $(PRESET)
	ln -sfn $(BUILD_DIR)/compile_commands.json compile_commands.json

build: configure
	cmake --build --preset $(PRESET) -j$(JOBS)

test: build
	ctest --preset $(PRESET) -j$(JOBS)

run: build
	$(BUILD_DIR)/qputer

examples: build
	$(BUILD_DIR)/examples/phase_estimation
	$(BUILD_DIR)/examples/variational
	$(BUILD_DIR)/examples/repetition_code

clean:
	cmake --build $(BUILD_DIR) --target clean

distclean:
	rm -rf build
