from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor
from sys import size_of, argv
from testing import assert_equal
import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor
from memory import UnsafePointer
from gpu.host import DeviceBuffer


comptime TPB = 15
comptime BLOCKS_PER_GRID = (2, 1)

####################### KERNEL #########################

fn conv1d_gpu_kernel[
    in_layout: Layout,
    out_layout: Layout,
    conv_layout: Layout,
    input_size: Int,
    conv_size: Int,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[dtype, out_layout, MutAnyOrigin],
    input: LayoutTensor[dtype, in_layout, MutAnyOrigin],
    filter: LayoutTensor[dtype, conv_layout, MutAnyOrigin],
):
    global_i = Int(block_dim.x * block_idx.x + thread_idx.x)
    local_i = Int(thread_idx.x)

    shared_a = LayoutTensor[
        dtype,
        Layout.row_major(TPB + conv_size - 1),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    shared_b = LayoutTensor[
        dtype,
        Layout.row_major(conv_size),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()


    if global_i < input_size:
        shared_a[local_i] = input[global_i]

    # second: load elements needed for convolution at block boundary
    if local_i < conv_size - 1:
        # indices from next block
        next_idx = global_i + TPB
        if next_idx < input_size:
            shared_a[TPB + local_i] = input[next_idx]
        else:
            # Initialize out-of-bounds elements to 0 to avoid reading from uninitialized memory
            # which is an undefined behavior
            shared_a[TPB + local_i] = 0

    if local_i < conv_size:
        shared_b[local_i] = filter[local_i]

    barrier()

    if global_i < input_size:
        var local_sum: output.element_type = 0

        @parameter
        for j in range(conv_size):
            if local_i + j < TPB + conv_size - 1:
                local_sum += shared_a[local_i + j] * shared_b[j]

        output[global_i] = local_sum

####################### KERNEL ENDS #########################



@compiler.register("conv1d")
struct Conv1DCustomOp:
    @staticmethod
    fn execute[
        target: StaticString,
        input_size: Int,
        conv_size: Int,
        dtype: DType = DType.float32,
    ](
        output: OutputTensor[rank=1],  
        input: InputTensor[rank = output.rank],
        filter: InputTensor[rank = output.rank],
        ctx: DeviceContextPtr    
    ) raises:
       
        # first get ref to device
        gpu_ctx = ctx.get_device_context()
       
        # we need extract layout for optimized kernel. By letting the compiler know ahead of time, it is able to plan better
        output_tensor = output.to_layout_tensor()
        input_tensor = input.to_layout_tensor()
        filter_tensor = filter.to_layout_tensor()
        comptime in_layout = input_tensor.layout
        comptime out_layout = output_tensor.layout
        comptime conv_layout = filter_tensor.layout

        
        comptime conv1d_gpu_kernel_ready = conv1d_gpu_kernel[
            in_layout, out_layout, conv_layout, input_size, conv_size
        ]
        
        # call the kernel
        gpu_ctx.enqueue_function_checked[conv1d_gpu_kernel_ready, conv1d_gpu_kernel_ready](
            output_tensor,
            input_tensor,
            filter_tensor,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=(TPB, 1),
        )
