#!/bin/bash

set -u

fname=${1}
count=${2}

for x in $(seq 1 ${count}); do
    julia ${fname} &
done
wait
