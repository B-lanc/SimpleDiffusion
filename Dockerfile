FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

RUN apt update

RUN pip install \
  lightning==2.0.9