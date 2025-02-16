import pycuda.driver as cuda
cuda.init()

print(f"Found {cuda.Device.count()} GPU(s)")
for i in range(cuda.Device.count()):
    device = cuda.Device(i)
    print(f"GPU {i}: {device.name()}, Compute Capability: {device.compute_capability()}")
