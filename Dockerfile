# Use Ubuntu 16.04 LTS
FROM ubuntu:xenial-20161213

# Prepare environment
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                    curl \
                    bzip2 \
                    ca-certificates \
                    xvfb \
                    cython3 \
                    build-essential \
                    autoconf \
                    libtool \
                    gfortran \
                    pkg-config

# Installing and setting up miniconda
RUN curl -sSLO https://repo.continuum.io/miniconda/Miniconda3-4.5.12-Linux-x86_64.sh && \
    bash Miniconda3-4.5.12-Linux-x86_64.sh -b -p /usr/local/miniconda && \
    rm Miniconda3-4.5.12-Linux-x86_64.sh

ENV PATH=/usr/local/miniconda/bin:$PATH \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONNOUSERSITE=1

# Installing precomputed python packages
RUN conda install -y mkl=2019.1 mkl-service;  sync &&\
    conda install -y numpy=1.15.4 \
                     scipy=1.2.0 \
                     scikit-learn=0.20.2 \
                     matplotlib=3.0.2 \
                     pandas=0.24.0 \
                     libxml2=2.9.9 \
                     libxslt=1.1.33 \
                     graphviz=2.40.1 \
                     cython=0.29.2 \
                     jupyter \
                     ipython \
                     traits=4.6.0; sync &&  \
    chmod -R a+rX /usr/local/miniconda; sync && \
    chmod +x /usr/local/miniconda/bin/*; sync && \
    conda clean --all -y; sync && \
    conda clean -tipsy && sync

# Unless otherwise specified each process should only use one thread - nipype
# will handle parallelization
ENV MKL_NUM_THREADS=1 \
    OMP_NUM_THREADS=1

# Installing qsiprep
COPY . /root/src/mittens
RUN apt-get install -y --no-install-recommends cmake \
    && /usr/local/miniconda/bin/pip install networkit
RUN cd /root/src/mittens \
    && /usr/local/miniconda/bin/pip install -e .
