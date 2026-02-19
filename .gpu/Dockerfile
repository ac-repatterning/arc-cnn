FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3


# If the steps of a `Dockerfile` use files that are different from the `context` file, COPY the
# file of each step separately; and RUN the file immediately after COPY
WORKDIR /app
COPY /.gpu/requirements.txt /app


# Environment
SHELL [ "/bin/bash", "-c" ]


# Install ...
RUN apt update && apt -q -y upgrade && apt -y install sudo && sudo apt -y install graphviz && \
    sudo apt -y install wget && sudo apt -y install curl && sudo apt -y install unzip && \
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/tmp/awscliv2.zip" && \
    unzip /tmp/awscliv2.zip -d /tmp/ && cd /tmp && sudo ./aws/install && cd ~ && \
    pip install --upgrade tensorflow && \
    pip install --upgrade pip && \
    pip install --requirement /app/requirements.txt --no-cache-dir


# Port
EXPOSE 8000 8888


# ENTRYPOINT
ENTRYPOINT ["bash"]
