# gnnsample dev image: Intel DPC++ (SYCL) compiler on top of the GridGuard image, which
# already carries the working Intel GPU driver (pinned; the PPA has since dropped those
# builds), PyTorch 2.14.1+xpu and PyG 2.8.1.
# Build `gridguard` first (C:\buidathon\gridguard: docker compose build).
#
# The GPU is reached through WSL2: run with --device /dev/dxg and the host's
# /usr/lib/wsl mounted read-only (docker-compose.yml does this).
FROM gridguard:latest

RUN echo 'Acquire::Retries "5";' > /etc/apt/apt.conf.d/80retries \
    && apt-get update && apt-get install -y --no-install-recommends \
        curl gpg python3-dev build-essential cmake ninja-build git \
    && rm -rf /var/lib/apt/lists/*

# DPC++ compiler 2026.1: must match the SYCL runtime (intel-sycl-rt 2026.1.0) that the
# PyTorch XPU wheel bundles, because our extension and torch share one SYCL runtime in
# one process.
RUN curl -fsSL https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
        | gpg --dearmor -o /usr/share/keyrings/oneapi.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/oneapi.gpg] https://apt.repos.intel.com/oneapi all main" \
        > /etc/apt/sources.list.d/oneapi.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        intel-oneapi-compiler-dpcpp-cpp-2026.1=2026.1.0-235 \
        intel-oneapi-dpcpp-cpp-2026.1=2026.1.0-235 \
        intel-oneapi-compiler-dpcpp-cpp-common-2026.1=2026.1.0-235 \
        intel-oneapi-compiler-dpcpp-cpp-runtime-2026.1=2026.1.0-235 \
        intel-oneapi-compiler-dpcpp-eclipse-cfg-2026.1=2026.1.0-235 \
        intel-oneapi-compiler-shared-2026.1=2026.1.0-235 \
        intel-oneapi-compiler-shared-common-2026.1=2026.1.0-235 \
        intel-oneapi-compiler-shared-runtime-2026.1=2026.1.0-235 \
    && rm -rf /var/lib/apt/lists/*

# Compiler on PATH for builds; at run time torch's bundled libsycl (in /opt/venv/lib)
# comes first so only one SYCL runtime is ever loaded.
ENV ONEAPI_ROOT=/opt/intel/oneapi \
    PATH=/opt/venv/bin:/opt/intel/oneapi/compiler/2026.1/bin:$PATH \
    LD_LIBRARY_PATH=/opt/venv/lib:/usr/lib/wsl/lib

COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
# pyg-lib's C++ neighbor sampler (the PyG baseline). CPU wheel; the sampler is CPU-only.
RUN pip install pyg-lib -f https://data.pyg.org/whl/torch-2.14.0+cpu.html

ENV PYTHONPATH=/work/python
WORKDIR /work
CMD ["bash"]
