#!/bin/bash
# Verifica, com objdump, que nenhuma unidade tem instrucoes acima do seu teto:
#   vector_simd.o        x86-64 generico: nenhuma instrucao VEX/EVEX (v...), nem registos ymm/zmm/k
#   kernels_avx2.o       AVX2+FMA: sem zmm, sem mascaras k, sem ymm/xmm16..31, sem AVX-VNNI (vpdp...)
#   kernels_avxvnni.o    AVX2+FMA+AVX-VNNI: sem zmm, sem mascaras k, sem ymm/xmm16..31
#   kernels_avx512.o     sem teto
# Uso: check-isa.sh <diretorio-dos-objetos>. Sai com erro se alguma unidade passar o teto.
set -u
DIR="${1:?uso: check-isa.sh <diretorio-dos-objetos>}"
OBJDUMP="${OBJDUMP:-objdump}"
status=0

if ! command -v "$OBJDUMP" >/dev/null 2>&1; then
  echo "check-isa: ERRO objdump nao encontrado: $OBJDUMP"
  exit 2
fi
if [ ! -f "$DIR/vector_simd.o" ]; then
  echo "check-isa: ERRO falta $DIR/vector_simd.o"
  exit 1
fi
if ! "$OBJDUMP" -f "$DIR/vector_simd.o" | grep -q 'x86-64'; then
  echo "check-isa: arquitectura nao x86-64; verificacao x86 ignorada"
  exit 0
fi
for obj in kernels_avx2.o kernels_avxvnni.o kernels_avx512.o; do
  if [ ! -f "$DIR/$obj" ]; then
    echo "check-isa: ERRO falta $DIR/$obj"
    status=1
  fi
done
if [ "$status" -ne 0 ]; then exit "$status"; fi

# Instrucoes de uma unidade (so a coluna do mnemonico e dos operandos).
disasm() { "$OBJDUMP" -d --no-show-raw-insn "$1" | grep -E '^\s+[0-9a-f]+:\s'; }

# check <objeto> <regex proibida> <descricao>
check() {
  local obj="$DIR/$1" bad
if [ ! -f "$obj" ]; then echo "check-isa: ERRO falta $obj"; status=1; return; fi
  if ! "$OBJDUMP" -f "$obj" | grep -q 'x86-64'; then echo "check-isa: $1 nao e x86-64, ignorado"; return 0; fi
  bad=$(disasm "$obj" | grep -E "$2" | head -5)
  if [ -n "$bad" ]; then
    echo "check-isa: ERRO $1 tem instrucoes acima do teto ($3):"
    echo "$bad"
    status=1
  else
    echo "check-isa: ok $1 ($3, $(disasm "$obj" | wc -l) instrucoes)"
  fi
}

VEX='^\s+[0-9a-f]+:\s+(notrack |bnd |rep |lock |\{vex\} )?v[a-z0-9]+\s'
REGS='%(ymm|zmm)[0-9]+|%k[0-7]'
HIGH='%(zmm[0-9]+|k[0-7]|[xy]mm(1[6-9]|2[0-9]|3[01]))\b'

check vector_simd.o "$VEX|$REGS" "x86-64 generico, sem VEX/EVEX"
check kernels_avx2.o "$HIGH|\svpdp[a-z]+\s" "AVX2+FMA"
check kernels_avxvnni.o "$HIGH" "AVX2+FMA+AVX-VNNI"
if ! disasm "$DIR/kernels_avxvnni.o" | grep -Eq '\svpdp[a-z]+[[:space:]]'; then
  echo "check-isa: ERRO kernels_avxvnni.o nao contem AVX-VNNI"
  status=1
fi
if ! disasm "$DIR/kernels_avx512.o" | grep -Eq '%zmm' || ! disasm "$DIR/kernels_avx512.o" | grep -Eq '\svpdp[a-z]+[[:space:]]'; then
  echo "check-isa: ERRO kernels_avx512.o nao contem AVX-512 e VNNI"
  status=1
else
  echo "check-isa: ok kernels_avx512.o (AVX-512+VNNI compilado; nao executado aqui)"
fi
exit $status
