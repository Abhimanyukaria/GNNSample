# gnnsample-dev plus Intel VTune Profiler, for CPU and GPU profiling of the sampler.
# Separate image so the dev image stays small. Under WSL2, VTune's API-level GPU timeline
# (Level Zero) and CPU sampling are expected to work; GPU hardware metrics may not.
FROM gnnsample-dev:latest
RUN apt-get update && apt-get install -y --no-install-recommends intel-oneapi-vtune \
    && rm -rf /var/lib/apt/lists/*
ENV PATH=/opt/intel/oneapi/vtune/latest/bin64:$PATH
