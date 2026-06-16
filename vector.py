import torch 
import triton 
import triton.language as tl

# -- The triton GPU kernel 
@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0,BLOCK_SIZE)
    
    
    mask = offsets < n_elements
    
    # -- Loading the data from the global vram into fast on chip sram/registers 
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask) 
    
    output = x + y
    
    tl.store(output_ptr + offsets, output, mask=mask)
    
    
# The pytorch wrapper launcher 

def triton_add(x: torch.Tensor,y: torch.Tensor):
    x = x.contiguous()
    y = y.contiguous()
    n_elements = x.numel()
    output = torch.empty_like(x) 
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # launch kernel 
    add_kernel[grid] (
        x,y,output,
        n_elements,
        BLOCK_SIZE=1024
    )
    
    return output 



if __name__ == "__main__":
    size = 98432
    print("Initializing random tensors on RTX 3050 GPU...") 
    a= torch.randn(size,device='cuda')
    b= torch.randn(size,device='cuda')
    
    print("Running Triton Custon Kernel....")
    triton_result = triton_add(a,b)
    
    print("Validating with native PyTorch...")
    torch_result = a+b;
    
    if torch.allclose(triton_result,torch_result):
        print("Success ! Your Trition kernal matches Pytorch perfectly..")
        print(f"Sample output Array: {triton_result[:5].tolist()}")
    else:
        print("Error: Triton result does not match PyTorch result.")
    