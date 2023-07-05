FROM nvidia/cuda:11.4.2-cudnn8-devel-ubuntu20.04

# install ubuntu dependencies
ENV DEBIAN_FRONTEND=noninteractive 
RUN apt-get update && \
    apt-get -y install python3-pip xvfb ffmpeg git build-essential python-opengl
RUN ln -s /usr/bin/python3 /usr/bin/python

# install python dependencies
RUN mkdir cleanrl_utils && touch cleanrl_utils/__init__.py
RUN pip install "poetry==1.4.0" --upgrade
COPY pyproject.toml pyproject.toml
COPY poetry.lock poetry.lock
# RUN poetry install --no-root
# RUN poetry install --no-root --with atari
# RUN poetry install --no-root --with pybullet
RUN poetry install --no-root -E "jax envpool"
RUN poetry run pip install --upgrade "jax[cuda]==0.3.17" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN poetry run pip install gymnax nvidia-ml-py
RUN poetry run python -c "import jax"

# install mujoco_py
COPY entrypoint.sh /usr/local/bin/
RUN chmod 777 /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
