This showcases the benefits of quantized matrix multiplication over regular matrix mutiplication, both of which use a tiled shared memory structure to reduce memory requests and move the operation
from being memory bound to compute bound.

I approached this by passing float values (fp32) to a driver CUDA function, quantizing the values using a symmetric quantization strategy, performed a tiled matrix multiplication kernel that
executed the dot product operations on int8_t values (accumulated through int32 values), and eventually dequantizing the data once all the values have been calculated in their int32 form.

This approach has two main tradeoffs. First, the benefit: It's far more memory efficient than matmul kernels that directly use floating point values. While fp32 values require 4 bytes of data,
int8_t is only one byte. This reduces the amount of shared memory by a quarter, potentially allowing for larger tiles to be stored in shared memory. This has the main effects of both allowing for
computations (such as machine learning inference) being ran on resource constrained devices that may have much smaller shared memory, without compromising performance. It also allows for larger
tiles to be loaded into shared memory, allowing for even less global memory access overhead. Then, the (initial) downside. While it is more memory efficient, the extra operations required to
dequantize the data and perform static casting for accumulating in int32's adds time overhead, and can significantly increase the execution time. This can be slightly mitigated due to larger
tile sizes being possible, but it still poses a threat. In addition, of course, quantization sacrifices some accuracy as well, meaning that if the data is used for multiple passes it may
become extremely corrupted over time.

This approach of quantized matmul is ideal for edge devices, where milliseconds might not matter so much in favor of saving on size and manufacturing cost, but might not be the best for
high performance computing and the like.

This was also repository to learn about CUDA programming through developing a few basic kernels (like increasingly optimized matrix multiplication and 2D image blurring, most commented out
in src/tiled_matmul.cu)
