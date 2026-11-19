# Converter image: OpenJDK + WebGraph, for decoding LAW BVGraph datasets (Twitter-2010)
# into gnnsample's CSC. Separate from the dev image so the SYCL/PyTorch stack is untouched.
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN echo 'Acquire::Retries "5";' > /etc/apt/apt.conf.d/80retries \
    && apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates ca-certificates-java openjdk-17-jdk-headless maven \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/webgraph
COPY docker/webgraph/pom.xml docker/webgraph/ToCsc.java ./
RUN mvn -q -B dependency:copy-dependencies -DoutputDirectory=lib -DincludeScope=runtime \
    && mvn -q -B dependency:list -DincludeScope=runtime | grep -i webgraph | head -3 \
    && javac -cp "lib/*" -d . ToCsc.java
ENTRYPOINT ["java", "-Xmx2g", "-cp", "/opt/webgraph:/opt/webgraph/lib/*", "ToCsc"]
