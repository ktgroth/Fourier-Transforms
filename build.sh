#!/bin/bash

CC="gcc"
LD="gcc"

CFLAGS="-O2 -Wall -Wextra -g"
LFLAGS=""

SRC=src
OBJ=obj
BUILD=build

SRCS=($(find "$SRC" -name "*.c" -o -path "$SRC" -name "*.c"))
OBJSK=()

for src in "${SRCS[@]}"; do
    obj="${OBJ}/$(basename "$src" .c).o"
    OBJSK+=("$obj")
done

OUTPUT=$BUILD/ft

function build {
    mkdir -p $OBJ
    mkdir -p $BUILD

    for i in "${!SRCS[@]}"; do
        src="${SRCS[$i]}"
        obj="${OBJSK[$i]}"
        # echo $src $obj
        comp $src $obj
    done

    $LD $LFLAGS "${OBJSK[@]}" -o $OUTPUT -lm
}

function run {
    if [ ! -e "./$OUTPUT" ]; then
        build
    fi
    ./$OUTPUT
    gnuplot -p -e "plot 'fft_output.dat' with lines"
}

function comp {
    $CC $CFLAGS -c $1 -o $2
}

function clean {
    clear

    rm -rf *.dat
    rm -rf $OBJ
    rm -rf $BUILD
    rm -rf stats
}

for arg in "$@"; do
    if [ "$arg" = "build" ]; then
        build
    elif [ "$arg" = "run" ]; then
        run
    elif [ "$arg" = "complete" ]; then
        clean
        build
        run
    elif [ "$arg" = "clean" ]; then
        clean
    fi
done
