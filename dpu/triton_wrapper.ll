; Auto-generated Triton wrapper. Do not edit by hand.
source_filename = "triton_wrapper"

%dpu_args = type { i8 addrspace(255)*, i8 addrspace(255)*, i8 addrspace(255)*, i32 }

@DPU_INPUT_ARGUMENTS = external global %dpu_args

declare void @add_kernel(i32 addrspace(255)*, i32 addrspace(255)*, i32 addrspace(255)*, i32, i8 addrspace(1)*, i8 addrspace(1)*)

define i32 @main() {
entry:
  %args = load %dpu_args, %dpu_args* @DPU_INPUT_ARGUMENTS, align 8
  %arg0 = extractvalue %dpu_args %args, 0
  %arg0_cast = bitcast i8 addrspace(255)* %arg0 to i32 addrspace(255)*
  %arg1 = extractvalue %dpu_args %args, 1
  %arg1_cast = bitcast i8 addrspace(255)* %arg1 to i32 addrspace(255)*
  %arg2 = extractvalue %dpu_args %args, 2
  %arg2_cast = bitcast i8 addrspace(255)* %arg2 to i32 addrspace(255)*
  %arg3 = extractvalue %dpu_args %args, 3
  call void @add_kernel(i32 addrspace(255)* %arg0_cast, i32 addrspace(255)* %arg1_cast, i32 addrspace(255)* %arg2_cast, i32 %arg3, i8 addrspace(1)* null, i8 addrspace(1)* null)
  ret i32 0
}
