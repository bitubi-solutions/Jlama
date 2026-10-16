#!/bin/bash
set -euo pipefail

MODULE_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$MODULE_DIR"
LIB_DIR="$MODULE_DIR/target/jextract-native/lib"
OBJ_DIR="$MODULE_DIR/target/jextract-native/obj"
mkdir -p "$LIB_DIR"
make -C "$MODULE_DIR" \
  LIB_DIR="$LIB_DIR" OBJ_DIR="$OBJ_DIR" LIB_NAME=libbi4j LIB_EXT=so \
  CC=gcc AR=gcc CFLAGS='-fPIC -O3 -Werror' LDFLAGS=-shared
export LD_LIBRARY_PATH="$LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Generate Java source code
/usr/local/jextract-20/bin/jextract --source \
  --output src/main/java20 \
  -t ao.bitubi.inference4j.tensor.operations.cnative \
  -I . \
  -l bi4j \
  --header-class-name NativeSimd \
  vector_simd.h

/usr/local/jextract-21/bin/jextract --source \
  --output src/main/java21 \
  -t ao.bitubi.inference4j.tensor.operations.cnative \
  -I . \
  -l bi4j \
  --header-class-name NativeSimd \
  vector_simd.h

/usr/local/jextract-22/bin/jextract \
  --output src/main/java22 \
  -t ao.bitubi.inference4j.tensor.operations.cnative \
  -I . \
  -l bi4j \
  --header-class-name NativeSimd \
  vector_simd.h
