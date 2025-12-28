podman build --platform linux/amd64 -t upmem_sdk "%~dp0"
podman run --platform linux/amd64 --rm -it -v "%cd%":/mnt/host_cwd --workdir /mnt/host_cwd upmem_sdk
