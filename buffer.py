import slangpy as spy
import numpy as np

def create_texture_2d(device, width, height, num_channels=4) -> spy.Texture:
    """
    Create a 2D writable texture with num_channels channels and 32 bit floats.
    num_channels can only be 4 or 1
    """
    if num_channels == 4:
        return device.create_texture(
            type = spy.TextureType.texture_2d,
            format = spy.Format.rgba32_float,
            width = width,
            height = height,
            usage = spy.TextureUsage.shader_resource |
                    spy.TextureUsage.unordered_access,
        )
    else:
        assert(num_channels == 1)
        return device.create_texture(
            type = spy.TextureType.texture_2d,
            format = spy.Format.r32_float,
            width = width,
            height = height,
            usage = spy.TextureUsage.shader_resource |
                    spy.TextureUsage.unordered_access,
        )

def structured_buffer_from_numpy(device, arr):
    arr = np.ascontiguousarray(arr)
    # Slang doesn't allow 0 size buffers. Creating a dummy.
    if len(arr) == 0:
        arr = np.zeros((1, *arr.shape[1:]), dtype=arr.dtype)
    return spy.Tensor.from_numpy(device, arr).storage

def structured_typed_buffer_from_numpy(device, arr, dtype):
    arr = np.ascontiguousarray(arr)
    tensor = spy.Tensor.empty(
        device,
        shape=(len(arr),),
        dtype=dtype
    )
    tensor.copy_from_numpy(arr)
    return tensor.storage

def create_structured_buffer(device, dtype, values):
    tensor = spy.Tensor.empty(
        device,
        # Slang doesn't allow 0 size buffers. Guard by at least having a size 1 array.
        shape=(max(len(values), 1),),
        dtype=dtype,
    )

    # The cursor construct can be slow in practice.
    # A production system may need to batch the update
    # using numpy arrays.
    cursor = tensor.cursor()
    for i, value in enumerate(values):
        cursor[i].write(value)
    cursor.apply()

    return tensor.storage
