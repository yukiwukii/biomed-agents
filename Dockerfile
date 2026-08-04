# syntax=docker/dockerfile:1.4
# =============================================================================
# Kernel-sandbox image for hypotest's InterpreterEnv  (BixBench-Hypothesis)
# =============================================================================
# This is a PATCHED copy of the upstream Dockerfile shipped in
#   third_party/hypotest @ pinned commit  (src path: <repo>/Dockerfile)
# adapted to build on 1x node of 4x GH200 (aarch64/ARM) on CSCS/Clariden.
#
# WHY A COMMITTED COPY: third_party/ is gitignored and re-checked-out to the
# pinned commit by scripts/00_setup.sh, so edits there do not persist and are
# not shareable. scripts/01_build_kernel_image.sh copies the hypotest tree into
# a build context and then drops THIS file in as the Dockerfile.
#
# Every deviation from upstream is wrapped in a block marked:
#     # >>> PATCH N: <what> — <why>
#     ...changed line(s)...
#     # <<< END PATCH N   (revert: <how to get back to upstream>)
# Search this file for "PATCH" to see all of them. Summary:
#   PATCH 1  Miniconda installer  x86_64 -> aarch64   (GH200 is ARM)
#   PATCH 2  pin r-base=4.3.3 in conda 'pinned'       (stop gratuitous 4.4 upgrade clobber)
#   PATCH 3  blast 2.17.0->2.16.0, mafft 7.526->7.525 (no aarch64 build for pinned ver)
#   PATCH 4  r-coloc: drop from conda, install via CRAN (no R-4.3 aarch64 conda build)
#   PATCH 5  upgrade typing_extensions>=4.13          (jupyter_client 8.9 PEP 728)
#   PATCH 6  chemistry pkgs disabled                  (chempy dep pyodesys has no aarch64 build)
#   PATCH 7  bioinformatics install split into 4 steps (monolithic step OOMs Docker VM on x86_64 emulation)
# On an x86_64 host, reverting PATCH 1 (+ optionally 3/4/6) yields the upstream image.
# =============================================================================
# Standalone scientific computing image (CPU)

FROM ubuntu:22.04

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update -qq && \
    apt-get install -yq --no-install-recommends \
    util-linux \
    git \
    openssh-client \
    wget \
    gpg \
    software-properties-common \
    build-essential \
    zsh && \
    rm -rf /var/lib/apt/lists/*

RUN touch /root/.zshrc

# Download and install Miniconda
# >>> PATCH 1: Miniconda installer x86_64 -> aarch64 — GH200/Grace is ARM; the
#     upstream installer URL is Linux-x86_64 and will not run on this node.
RUN --mount=type=cache,target=/root/.cache/miniconda \
    wget https://repo.anaconda.com/miniconda/Miniconda3-py312_25.3.1-1-Linux-x86_64.sh -O ~/miniconda.sh && \
    chmod +x ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /app/miniconda && \
    rm ~/miniconda.sh && \
    /app/miniconda/bin/conda init bash
# <<< END PATCH 1   (revert: change "Linux-aarch64.sh" back to "Linux-x86_64.sh")

# Override base image python with miniconda python
RUN ln -sf /app/miniconda/bin/python /usr/local/bin/python && \
    ln -sf /app/miniconda/bin/python3 /usr/local/bin/python3 && \
    ln -sf /app/miniconda/bin/pip /usr/local/bin/pip && \
    ln -sf /app/miniconda/bin/pip3 /usr/local/bin/pip3

ENV VIRTUAL_ENV="/app/miniconda"
ENV PATH="/app/miniconda/bin:$PATH"
ENV PYTHONPATH="/app/miniconda/lib/python3.12/site-packages:${PYTHONPATH:-}"

# Install uv and mamba
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --no-cache-dir uv==0.8.19
RUN conda install -c conda-forge mamba==2.3.2 -y

# Create kernel environment with all analysis packages
RUN conda create -p /app/kernel_env python=3.12 -y

# Install R packages
RUN mamba install -p /app/kernel_env -c conda-forge -y \
            r-base=4.3.3 \
            r-r.utils=2.13.0 \
            r-recommended=4.3 \
            r-irkernel=1.3.2 \
            r-tidyverse=2.0.0 \
            r-readxl=1.4.5 \
            r-seurat=5.3.0 \
            rpy2=3.5.11 \
            r-factominer=2.12 \
            r-rcolorbrewer=1.1_3 \
            r-devtools=2.4.5 \
            r-broom=1.0.9 \
            r-data.table=1.17.8 \
            r-enrichr=3.4 \
            r-factoextra=1.0.7 \
            r-ggnewscale=0.5.2 \
            r-ggrepel=0.9.6 \
            r-ggpubr=0.6.1 \
            r-ggvenn=0.1.10 \
            r-janitor=2.2.1 \
            r-multcomp=1.4_28 \
            r-matrix=1.6_5 \
            r-pheatmap=1.0.13 \
            r-reshape=0.8.10 \
            r-rstatix=0.7.2 \
            r-viridis=0.6.5 \
            r-hdf5r=1.3.11

# >>> PATCH 2: pin r-base=4.3.3 — every later unconstrained `mamba install`
#     (core/ML/bio/bioconda steps) is otherwise free to UPGRADE r-base. With current
#     conda-forge aarch64 repodata the solver gratuitously bumps 4.3.3 -> 4.4.3 and
#     libmamba's relink then dies with "cannot copy: File exists .../deactivate-r-base.sh".
#     Upstream targets R 4.3 / Bioconductor 3.18, so freeze r-base via the env's
#     conda 'pinned' file right after the (cached) R layer; this keeps every r-* dep on
#     its r43 build and stops the clobber across all subsequent steps.
RUN mkdir -p /app/kernel_env/conda-meta && echo r-base=4.3.3 > /app/kernel_env/conda-meta/pinned
# <<< END PATCH 2   (revert: delete this RUN; only needed when later steps would bump r-base)

# Install core Python scientific stack
RUN mamba install -p /app/kernel_env -c conda-forge -y \
            numpy=1.26.4 \
            pandas=2.3.2 \
            scipy=1.16.2 \
            scikit-learn=1.7.2 \
            matplotlib=3.10.6 \
            seaborn=0.13.2 \
            plotly=6.3.0 \
            openpyxl=3.1.5 \
            jupyter=1.1.1 \
            ipykernel=6.30.1 \
            nbconvert=7.16.6

# Install ML/optimization packages
RUN mamba install -p /app/kernel_env -c conda-forge -y \
            keras=3.11.2 \
            optuna=4.5.0 \
            imbalanced-learn=0.14.0 \
            lightgbm=4.6.0 \
            statsmodels=0.14.5

# Install bioinformatics Python packages
RUN mamba install -p /app/kernel_env -c conda-forge -y \
            anndata=0.12.2 \
            scanpy=1.11.4 \
            biopython=1.85 \
            muon=0.1.6 \
            umap-learn=0.5.9.post2 \
            leidenalg=0.10.2 \
            python-igraph=0.11.9

# Install visualization and utility packages
RUN mamba install -p /app/kernel_env -c conda-forge -y \
            matplotlib-venn=1.1.2 \
            ete3=3.1.3 \
            fcsparser=0.2.8 \
            datasets=2.2.1 \
            udocker=1.3.17 \
            sqlite=3.50.4

# >>> PATCH 6: chemistry packages DISABLED on aarch64 — chempy=0.10.1 requires
#     pyodesys, which has no linux-aarch64 conda build (solver: "nothing provides
#     pyodesys"); rdkit/pubchempy are not needed for the bioinformatics tasks. Left
#     commented to match the upstream intent that this block is optional on aarch64.
# Install chemistry packages
# RUN mamba install -p /app/kernel_env -c conda-forge -y \
#             rdkit=2025.09.2 \
#             pubchempy=1.0.5 \
#             chempy=0.10.1
# <<< END PATCH 6   (revert on x86_64: uncomment the block; on aarch64 install chempy
#                    via pip instead, e.g. `/app/kernel_env/bin/pip install chempy==0.10.1`)

# Install bioinformatics tools
# >>> PATCH 3: blast 2.17.0->2.16.0 and mafft 7.526->7.525 — the upstream-pinned
#     versions have no linux-aarch64/noarch conda build; the nearest patch version does
#     (verified via the anaconda.org API). Patch-level change, safe for the tasks.
# >>> PATCH 4a: r-coloc removed from this conda step — its ONLY linux-aarch64 conda
#     builds (<=5.1.0.1) were built for R 4.1/4.2; there is no R-4.3 aarch64 build, so it
#     cannot coexist with the R-4.3 Bioconductor 3.18 stack. Installed from CRAN below
#     (PATCH 4b). Upstream line was: `r-coloc=5.2.3 \`.
# >>> PATCH 7: split upstream's single 35-package mamba install into 4 smaller steps —
#     the monolithic step exhausts the Docker VM's RAM (~8GB) under linux/amd64 emulation
#     on Apple Silicon: the libmamba SAT solver loads full repodata for both conda-forge
#     and bioconda into memory simultaneously, gets OOM-killed after several hours with
#     "cannot allocate memory". Splitting reduces per-step solver memory and gives each
#     group its own cache layer so retries don't restart from scratch.
#     Revert: collapse all four RUN blocks back into one.

# Group 1: CLI sequence analysis tools
RUN mamba install -p /app/kernel_env -c conda-forge -c bioconda -y \
            blast=2.16.0 \
            clipkit=2.6.1 \
            clustalo=1.2.4 \
            fastqc=0.12.1 \
            hmmer=3.4 \
            hhsuite=3.3.0 \
            iqtree=3.0.1 \
            mafft=7.525 \
            metaeuk=7.bba0d80 \
            mmseqs2=18.8cc5c \
            samtools=1.22.1 \
            gatk=3.8 \
            spades=4.2.0 \
            trim-galore=0.6.10 \
            perl=5.32.1

# Group 2: Python bioinformatics packages
RUN mamba install -p /app/kernel_env -c conda-forge -c bioconda -y \
            biokit=0.5.0 \
            gseapy=1.1.10 \
            mygene=3.2.2 \
            phykit=2.0.3 \
            pydeseq2=0.5.2 \
            harmonypy=0.0.10

# Group 3: Core Bioconductor genomics stack
RUN mamba install -p /app/kernel_env -c conda-forge -c bioconda -y \
            bioconductor-genomicranges=1.54.1 \
            bioconductor-summarizedexperiment=1.32.0 \
            bioconductor-deseq2=1.42.0 \
            bioconductor-apeglm=1.24.0 \
            bioconductor-limma=3.58.1 \
            bioconductor-org.hs.eg.db=3.18.0 \
            bioconductor-clusterprofiler=4.10.0 \
            bioconductor-geoquery=2.70.0

# Group 4: Additional Bioconductor and R packages (install separately on server)
RUN mamba install -p /app/kernel_env -c conda-forge -c bioconda -y \
            bioconductor-enhancedvolcano=1.20.0 \
            bioconductor-flowcore=2.14.0 \
            bioconductor-flowmeans=1.62.0 \
            r-wgcna=1.73 \
            r-susier=0.14.2 \
            r-mendelianrandomization=0.10.0 \
            r-ldlinkr=1.4.0 \
            r-arrow=13.0.0

# <<< END PATCH 7 (revert: collapse the four RUN blocks above back into one)
# <<< END PATCH 3 (revert: blast=2.17.0, mafft=7.526 — only on x86_64 where they exist)
# <<< END PATCH 4a (revert: add `r-coloc=5.2.3 \` back above and drop PATCH 4b)

# >>> PATCH 4b: install r-coloc from CRAN against the image's R 4.3.3 (compiles in-image).
#     Plain install.packages pulls coloc (5.2.3 on CRAN) + only MISSING deps, leaving the
#     conda-provided deps (susieR, data.table, ggplot2, viridis) untouched. Fails the build
#     if coloc does not import afterwards.
RUN /app/kernel_env/bin/R --vanilla -e 'install.packages("coloc", repos="https://cloud.r-project.org"); if (!requireNamespace("coloc", quietly=TRUE)) quit(status=1, save="no")'
# <<< END PATCH 4b   (revert: delete this RUN and restore r-coloc in the conda step, PATCH 4a)

# Install pytorch (CPU)
RUN /app/kernel_env/bin/python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# >>> PATCH 5: upgrade typing_extensions>=4.13 — mamba pulls jupyter_client 8.9.0, whose
#     connect.py uses PEP 728 `TypedDict(..., extra_items=...)` imported from
#     typing_extensions, which needs >=4.13; conda ships 4.12.2, so the next step
#     (ipykernel install) crashes with `__new__() got an unexpected keyword 'extra_items'`.
#     typing_extensions is a leaf dep, so a pip upgrade here is safe.
RUN /app/kernel_env/bin/pip install --no-cache-dir "typing_extensions>=4.13"
# <<< END PATCH 5   (revert: delete this RUN; only needed while jupyter_client>=8.7 + py3.12)

# Install Jupyter kernels
RUN /app/kernel_env/bin/python -m ipykernel install --name python3 --display-name "Python 3 (ipykernel)" && \
    export PATH="/app/kernel_env/bin:$PATH" && \
    /app/kernel_env/bin/R -e 'IRkernel::installspec(user = FALSE, name = "ir", displayname = "R")'

# Install kernel server dependencies
RUN /app/kernel_env/bin/pip install --no-cache-dir fastapi uvicorn

# Clean up conda caches
RUN mamba clean -all -y && \
    find /app/miniconda \( -type d -name __pycache__ -o -type d -name tests -o -type d -name '*.tests' -o -type d -name 'test' \) -exec rm -rf {} + || true && \
    find /app/miniconda -type f -name '*.a' -delete && \
    find /app/miniconda -type f -name '*.js.map' -delete && \
    find /app/kernel_env \( -type d -name __pycache__ -o -type d -name tests -o -type d -name '*.tests' -o -type d -name 'test' \) -exec rm -rf {} + || true && \
    find /app/kernel_env -type f -name '*.a' -delete && \
    find /app/kernel_env -type f -name '*.js.map' -delete

# Copy kernel server for Docker-based execution
COPY src/hypotest/env/kernel_server.py /envs/kernel_server.py

WORKDIR /workspace
EXPOSE 8000

CMD ["/app/kernel_env/bin/python", "/envs/kernel_server.py", "--work_dir", "/workspace"]