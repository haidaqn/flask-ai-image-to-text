import paddle
print("Paddle version:", paddle.__version__)
print("CUDA version:", paddle.version.cuda())
print("GPU available:", paddle.device.is_compiled_with_cuda())
