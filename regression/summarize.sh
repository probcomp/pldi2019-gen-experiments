#!/bin/bash

set -u

command -v datamash >/dev/null 2>&1 || {
    echo >&2 "Install datamash using apt-get, please."; exit 1;
}

fnames="${@}"

for fname in ${fnames}; do
    iters=$(cat ${fname}| head -n1 | cut -d, -f1)
    median=$(datamash -t, median 2 < ${fname})
    iqr=$(datamash -t, iqr 2 < ${fname})
    echo ${fname} $(python -c "print 1000*${median}/${iters}, 1000*${iqr}/${iters}")
done
