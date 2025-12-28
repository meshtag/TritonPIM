FROM ubuntu:20.04
RUN apt update && \
DEBIAN_FRONTEND=noninteractive apt install -y software-properties-common && \
add-apt-repository ppa:deadsnakes/ppa && \
DEBIAN_FRONTEND=noninteractive apt install -y git wget libedit-dev libxml2 python3.8-dev libelf-dev libnuma-dev python3-pip pkg-config && \
pip3 install six && \
git clone https://github.com/CMU-SAFARI/UPMEM.git && \
mv UPMEM/upmem-2023.1.0-Linux-x86_64/ /upmem-sdk-2023.1.0
CMD bash -c "source /upmem-sdk-2023.1.0/upmem_env.sh simulator && bash"
