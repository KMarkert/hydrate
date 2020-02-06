FROM continuumio/miniconda3:latest

# USER root

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
     build-essential \
     vim \
     gcc \
     git \
     libssl-dev \
     libffi-dev \
     dpkg-dev \
     fakeroot \
     libfakeroot:amd64 \
  && apt-get autoremove -y \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

RUN conda update -n base -c defaults conda -y \
  && conda install -c conda-forge -y \
    python>=3.7 \
    ipython \
    xarray \
    netcdf4 \
    numpy \
    pandas \
    geopandas \
    requests \
    matplotlib \
    scipy \
    scikit-learn \
    scikit-image \
    earthengine-api \
    affine \
    pyproj \
    rasterstats \
    rasterio \
    pysheds \
    geojson \
    backoff \
    tqdm \
    numba \
    oauth2client \
  && conda update --all -c conda-forge -y \
  && conda install -c conda-forge \
    pyproj=2.1.3 \
    proj4=6.1.0 \
  && conda clean --all -f -y

RUN pip install git+https://github.com/kmarkert/hydrate.git

RUN git clone https://github.com/uw-hydro/vic.git \
  && cd vic \
  && git checkout VIC.4.2.d \
  && cd src\
  && make \
  && ln ./vicNl /usr/local/bin/vicNl \
  && cd ~/

# USER jovyan
