# JetPack4(CUDA10.2) PyTorch & TorchVision Build for Python3.8

## Prerequisites

1. Need Ubuntu

1. Install QEMU packages:

```bash
sudo apt-get install qemu-user-static
```

## Usage

To run the build, input the following command:

```bash
docker buildx build --rm -t jetson-pythorch-build -o wheels .
```

If you need to log the build, you can run:

```bash
docker buildx build --no-cache --progress=plain --rm -t jetson-pythorch-build -o wheels . 2>&1 | tee build.log
```

## License

BSD 3-Clause License

## References

1. [qemu-user-static](https://github.com/multiarch/qemu-user-static)
1. [Xavier Geerinck Post](https://xaviergeerinck.com/post/2021/11/25/infrastructure-nvidia-ai-nvidia-building-pytorch)
1. [Building pytorch for arm64](https://github.com/soerensen3/buildx-pytorch-jetson)
1. [Building Jetson Nano libraries on host PC](https://i7y.org/building-jetson-nano-libraries-on-host-pc/)
1. [Install PyTorch on Jetson Nano.](https://qengineering.eu/install-pytorch-on-jetson-nano.html)
