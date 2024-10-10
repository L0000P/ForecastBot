# Use the desired Rapids AI image as base
FROM rapidsai/rapidsai:cuda11.8-runtime-ubuntu22.04-py3.10

# Set up the working directory
WORKDIR /workspace

# Install JupyterLab
RUN pip install jupyterlab

# Copy requirements files from the subdirectories into the container
COPY Arima/requirements.txt /workspace/requirements_arima.txt
COPY Sarimax/requirements.txt /workspace/requirements_sarimax.txt
COPY Informer/requirements.txt /workspace/requirements_informer.txt
COPY PatchTST/requirements.txt /workspace/requirements_patchtst.txt
COPY Pyraformer/requirements.txt /workspace/requirements_pyraformer.txt

# Use Mamba for faster installation and environment setup
# Set up ARIMA and SARIMAX with Python 3.10, cupy >=12.0.0, and cudf=23.6
RUN conda install -n base -c conda-forge mamba && \
    mamba create -n arima python=3.10 && \
    mamba install -n arima cudf=23.6 cupy=12.0.0 cudatoolkit=11.8 -c rapidsai -c conda-forge -c defaults && \
    mamba run -n arima pip install -r /workspace/requirements_arima.txt

RUN conda install -n base -c conda-forge mamba && \
    mamba create -n sarimax python=3.10 && \
    mamba install -n sarimax cudf=23.6 cupy=12.0.0 cudatoolkit=11.8 -c rapidsai -c conda-forge -c defaults && \
    mamba run -n sarimax pip install -r /workspace/requirements_sarimax.txt

# Set up Informer, PatchTST, and Pyraformer with Python 3.9, cupy >=12.0.0, and cudf=23.6
RUN conda install -n base -c conda-forge mamba && \
    mamba create -n informer python=3.9 && \
    mamba install -n informer cudf=23.6 cupy=12.0.0 cudatoolkit=11.8 -c rapidsai -c conda-forge -c defaults && \
    mamba run -n informer pip install -r /workspace/requirements_informer.txt

RUN conda install -n base -c conda-forge mamba && \
    mamba create -n patchtst python=3.9 && \
    mamba install -n patchtst cudf=23.6 cupy=12.0.0 cudatoolkit=11.8 -c rapidsai -c conda-forge -c defaults && \
    mamba run -n patchtst pip install -r /workspace/requirements_patchtst.txt

RUN conda install -n base -c conda-forge mamba && \
    mamba create -n pyraformer python=3.9 && \
    mamba install -n pyraformer cudf=23.6 cupy=12.0.0 cudatoolkit=11.8 -c rapidsai -c conda-forge -c defaults && \
    mamba run -n pyraformer pip install -r /workspace/requirements_pyraformer.txt

# Expose port for JupyterLab
EXPOSE 8888

# Run JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
