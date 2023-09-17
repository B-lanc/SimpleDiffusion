# SimpleDiffusion
Simple Diffusion model just for training my smol brain

building dockerfile
> docker build . -t b-lanc/simplediffusion

running the container
> docker run -dit --name=SD --runtime=nvidia --gpus=0 --shm-size=2gb -v /mnt/Data/datasets/cifar10:/dataset -v /mnt/Data2/DockerVolumes/SD:/saves -v .:/workspace b-lanc/simplediffusion

going inside container
> docker exec -it SD /bin/bash