#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) #from https://stackoverflow.com/questions/59895/how-do-i-get-the-directory-where-a-bash-script-is-located-from-within-the-script
docker build --platform linux/amd64 -t upmem_sdk "${SCRIPT_DIR}"
docker run --platform linux/amd64 --rm -it -v "${PWD}":/mnt/host_cwd --workdir /mnt/host_cwd upmem_sdk
