print("Imports....")
import cv2
import numpy as np
import torch
from torchvision import transforms
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
from onnx_helper import ONNXClassifierWrapper
import time
import os


transform_compose = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224))
     
    ])

TRT_LOGGER = trt.Logger()


"""

def load_engine(engine_file_path):
    assert os.path.exists(engine_file_path)
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())
"""
print("Loaded model...")
load_start = time.time()


#Creating the engine
BATCH_SIZE = 1
N_CLASSES = 8 # Our ResNet-50 is trained on a 1000 class ImageNet task
ENGINE_NAME = "model.trt"
trt_model = ONNXClassifierWrapper("16workspace_size_64ms_inference.trt", [BATCH_SIZE, N_CLASSES])

#loaded_engine = load_engine("16workspace_size_64ms_inference.trt")
load_end = time.time()
print(f"Loaded in {load_end - load_start} seconds")




img = cv2.imread('input_img.jpg')
converted_pytorch_img = transform_compose(img).unsqueeze(dim=0).numpy()

start = time.time()
def infer(engine, input_image):
    with engine.create_execution_context() as context:
        # Set input shape based on image dimensions for inference
        context.set_binding_shape(engine.get_binding_index("input"), (1, 3, 224, 224))
        # Allocate host and device buffers
        bindings = []
        for binding in engine:
            binding_idx = engine.get_binding_index(binding)
            size = trt.volume(context.get_binding_shape(binding_idx))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            if engine.binding_is_input(binding):
                input_buffer = np.ascontiguousarray(input_image)
                input_memory = cuda.mem_alloc(input_image.nbytes)
                bindings.append(int(input_memory))
            else:
                output_buffer = cuda.pagelocked_empty(size, dtype)
                output_memory = cuda.mem_alloc(output_buffer.nbytes)
                bindings.append(int(output_memory))

        stream = cuda.Stream()
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(input_memory, input_buffer, stream)
        # Run inference
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        # Transfer prediction output from the GPU.
        cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
        # Synchronize the stream
        stream.synchronize()

    return output_buffer
    
    """
    with postprocess(np.reshape(output_buffer, (image_height, image_width))) as img:
        print("Writing output image to file {}".format(output_file))
        img.convert('RGB').save(output_file, "PPM")
    """

#output = infer(loaded_engine, converted_pytorch_img)
output = trt_model.predict(converted_pytorch_img)

end = time.time()
print(f"Time taken: {end - start}")


print(f"Output list: {output}")