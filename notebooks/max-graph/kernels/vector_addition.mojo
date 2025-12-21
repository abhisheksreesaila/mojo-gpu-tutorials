

from math import ceildiv

from gpu import block_dim, block_idx, thread_idx
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor

from layout import Layout, LayoutTensor
from utils.index import IndexList
from gpu import thread_idx, block_idx
from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor
from math import iota
from math import ceildiv
from sys import has_accelerator
from gpu.host import DeviceContext
from gpu import block_dim, block_idx, thread_idx
from layout import Layout, LayoutTensor
import compiler



fn vector_addition_kernel[layout: Layout, float_dtype: DType
](                                                                  #  Tip : "](" <--- This has to be together.  Else you will compile time error
    lhs_tensor: LayoutTensor[float_dtype, layout, MutAnyOrigin],
    rhs_tensor: LayoutTensor[float_dtype, layout, MutAnyOrigin],
    out_tensor: LayoutTensor[float_dtype, layout, MutAnyOrigin],
):
    # Get the global ID
    var tid = block_idx.x * block_dim.x + thread_idx.x

    # let each thread add the elements and store in result
    out_tensor[tid] = lhs_tensor[tid] + rhs_tensor[tid] 

  

@compiler.register("vector_addition")
struct VectorAddition:
    @staticmethod
    fn execute[
        target: StaticString,
        output_size : UInt,
        dtype: DType = DType.float32,

    ](
        output: OutputTensor[rank=1],       ## <----- this has to be first
        lhs: InputTensor[rank = output.rank],      ## <----- any number of inputs, any name is ok.. the order of input musth match your graph input
        rhs: InputTensor[rank = output.rank],      ## <----- any number of inputs, any name is ok.. the order of input musth match your graph input
        ctx: DeviceContextPtr,              ## <----- this has to be laast
    ) raises:


        # Get a reference to GPU
        gpu_ctx = ctx.get_device_context()

        comptime BLOCK_SIZE = 3
        var vector_length = output_size
        var num_blocks = ceildiv(vector_length, BLOCK_SIZE)

        lhs_layout_tensor = lhs.to_layout_tensor()
        rhs_layout_tensor = rhs.to_layout_tensor()
        output_layout_tensor = output.to_layout_tensor()

        comptime lhs_layout = lhs_layout_tensor.layout
        comptime rhs_layout = rhs_layout_tensor.layout
        comptime output_layout = output_layout_tensor.layout


        comptime vector_addition_kernel_ready = vector_addition_kernel[
            output_layout, dtype
        ]

        # call the kernel
        gpu_ctx.enqueue_function_checked[
            vector_addition_kernel_ready, vector_addition_kernel_ready
        ](
            lhs_layout_tensor, 
            rhs_layout_tensor, 
            output_layout_tensor, 
            grid_dim=num_blocks, 
            block_dim=BLOCK_SIZE
        )
    