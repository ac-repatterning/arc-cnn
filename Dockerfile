# Base
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

# Temporary
ARG GID=3333
ARG UID=$GID


# If the steps of a `Dockerfile` use files that are different from the `context` file, COPY the
# file of each step separately; and RUN the file immediately after COPY
WORKDIR /app
COPY .devcontainer/requirements.txt /app


# Environment
SHELL [ "/bin/bash", "-c" ]


# Virtual Environment
ENV PYTHON_VIRTUAL_ENV=/opt/environment


# Setting-up
# $PYTHON_VIRTUAL_ENV/bin/pip install --upgrade tf-keras --no-cache-dir
RUN groupadd --system automata --gid $GID && \
    useradd --system automaton --uid $UID --gid $GID && \
    apt update && apt -q -y upgrade && apt -y install sudo && \
    sudo apt -y install graphviz && apt -y install vim && \
    sudo apt -y install wget && sudo apt -y install curl && sudo apt -y install unzip && \
    sudo apt -y install software-properties-common && \
    sudo apt -y install build-essential && \
    sudo apt -y install python3 python3-dev python3-pip python3-yaml python3-venv && \
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/tmp/awscliv2.zip" && \
    unzip /tmp/awscliv2.zip -d /tmp/ && cd /tmp && sudo ./aws/install && cd ~ && \
    python3 -m venv $PYTHON_VIRTUAL_ENV && \
    $PYTHON_VIRTUAL_ENV/bin/pip install --upgrade pip && \
    $PYTHON_VIRTUAL_ENV/bin/pip install --requirement /app/requirements.txt --no-cache-dir && \
    mkdir /app/warehouse && \
    chown -R automaton:automata /app/warehouse


# Hence
ENV PATH="${PYTHON_VIRTUAL_ENV}/bin:$PATH"


# Specific COPY
COPY src /app/src
COPY config.py /app/config.py


# Port
EXPOSE 8000 8888


# Create mountpoint
VOLUME /app/warehouse


# automaton
USER automaton


# ENTRYPOINT
ENTRYPOINT ["python"]


# CMD
CMD ["src/main.py"]