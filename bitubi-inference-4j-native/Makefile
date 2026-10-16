#
# The Jlama Project licenses this file to you under the Apache License,
# version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at:
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

## GNU Makefile designed to build a static library which can be shared across multiple architectures.

## Input environment:
# CC - compiler (gcc or clang)
# AR - archiver (ar)
# JNI_PLATFORM - "linux" for linux and "darwin" for mac.
# LIB_DIR - where the static library will be built in
# OBJ_DIR - where the obj files will be built in (defaults to LIB_DIR)
# LIB_NAME - the name of the native library
# LIB_EXT - the extension of the native library

SRC_DIR = src/main/c
LIB = $(LIB_DIR)/$(LIB_NAME).$(LIB_EXT)

# vector_simd.c fica no piso da arquitetura; cada kernel tem as suas flags.
ARCH := $(shell uname -m)
ifneq ($(filter x86_64 amd64,$(ARCH)),)
ISA_UNITS ?= avx2 avxvnni avx512
BASE_ISA_CFLAGS = -march=x86-64
else ifneq ($(filter aarch64 arm64,$(ARCH)),)
ISA_UNITS ?= neon
BASE_ISA_CFLAGS = -march=armv8-a
else
$(error arquitetura nao suportada: $(ARCH))
endif

SRCS = $(SRC_DIR)/vector_simd.c $(ISA_UNITS:%=$(SRC_DIR)/kernels_%.c)

# flags por ISA (alvos especificos: nao vazam para a unidade base)
$(OBJ_DIR)/kernels_avx2.o:    ISA_CFLAGS = -mavx2 -mfma
$(OBJ_DIR)/kernels_avxvnni.o: ISA_CFLAGS = -mavx2 -mfma -mavxvnni
$(OBJ_DIR)/kernels_avx512.o:  ISA_CFLAGS = -mavx2 -mfma -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vnni
$(OBJ_DIR)/kernels_neon.o:    ISA_CFLAGS = -march=armv8-a
$(OBJ_DIR)/vector_simd.o:     ISA_CFLAGS = $(BASE_ISA_CFLAGS)

OBJS = $(SRCS:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)

all: $(LIB)

$(LIB): $(OBJS)
	mkdir -p $(LIB_DIR)
	$(AR) $(LDFLAGS) -o $(LIB) $^

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c $(wildcard $(SRC_DIR)/*.h $(SRC_DIR)/*.inc)
	mkdir -p $(OBJ_DIR)
	$(CC) -o $@ -c $< $(CFLAGS) $(ISA_CFLAGS)

clean:
	rm -rf $(LIB_DIR) $(OBJ_DIR)

## Debug support
# use make print-VARIABLE name to see the value
print-%  : ; @echo $* = $($*)
