# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
# Copyright Ahmad Ali Parr
#
# Thin wrapper around CMake.
#   make                     build everything (sm_80/86/89/90)
#   make ARCHS="86"          build for one architecture
#   make test                run every test (GPU tests skip without a device)
#   make test-cpu            CPU tests only
#   make bench               run both GPU benchmarks
#   make bench-cpu           time the CPU references
#   make figures             regenerate docs/img from docs/img/src

ARCHS     ?= 80;86;89;90
BUILD     ?= build
NVCC      ?= $(shell command -v nvcc 2>/dev/null)
CUDA_ROOT ?= $(patsubst %/bin/nvcc,%,$(NVCC))
JOBS      ?= $(shell nproc 2>/dev/null || echo 4)

CMAKE_ARGS = -S . -B $(BUILD) -DCMAKE_BUILD_TYPE=Release "-DCMAKE_CUDA_ARCHITECTURES=$(ARCHS)"
ifneq ($(NVCC),)
CMAKE_ARGS += -DCMAKE_CUDA_COMPILER=$(NVCC) -DCUDAToolkit_ROOT=$(CUDA_ROOT)
endif

.PHONY: all build configure test test-cpu test-gpu bench bench-cpu figures clean

all: build

configure:
	cmake $(CMAKE_ARGS)

build: configure
	cmake --build $(BUILD) -j $(JOBS)

test: build
	cd $(BUILD) && ctest --output-on-failure

test-cpu: build
	cd $(BUILD) && ctest --output-on-failure -L cpu

test-gpu: build
	cd $(BUILD) && ctest --output-on-failure -L gpu

bench: build
	$(BUILD)/sid_bench
	$(BUILD)/amm_bench

bench-cpu: build
	$(BUILD)/cpu_ref_bench

figures:
	sh docs/img/src/build.sh

clean:
	rm -rf $(BUILD)
