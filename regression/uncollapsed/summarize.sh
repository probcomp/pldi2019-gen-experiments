#!/bin/bash

set -u

fname=${1}
iters=$(cat ${fname}| head -n1 | cut -d, -f1)
median=$(datamash -t, median 2 < ${fname})
iqr=$(datamash -t, iqr 2 < ${fname})
echo $(python -c "print ${median}/${iters}, ${iqr}/${iters}")
