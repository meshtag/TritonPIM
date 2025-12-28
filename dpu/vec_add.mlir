// DPU input args: {mram_a, mram_b, mram_c, n}
module {
  llvm.mlir.global external @DPU_INPUT_ARGUMENTS()
      : !llvm.struct<"dpu_args", (ptr<i8, 255>, ptr<i8, 255>, ptr<i8, 255>, i32)>

  llvm.func @mem_alloc(i32) -> !llvm.ptr<i8>

  llvm.func @main() -> i32 {
    %c0_i32 = llvm.mlir.constant(0 : i32) : i32
    %c4_i32 = llvm.mlir.constant(4 : i32) : i32
    %c7_i32 = llvm.mlir.constant(7 : i32) : i32
    %cneg8_i32 = llvm.mlir.constant(-8 : i32) : i32
    %c0 = llvm.mlir.constant(0 : index) : i64
    %c1 = llvm.mlir.constant(1 : index) : i64

    %tid = dpu.tid : i32
    %is_zero = llvm.icmp "eq" %tid, %c0_i32 : i32
    llvm.cond_br %is_zero, ^bb0, ^bb_exit

  ^bb0:
    %args_ptr = dpu.input_args
        : !llvm.ptr<struct<"dpu_args", (ptr<i8, 255>, ptr<i8, 255>, ptr<i8, 255>, i32)>>
    %args = llvm.load %args_ptr
        : !llvm.ptr<struct<"dpu_args", (ptr<i8, 255>, ptr<i8, 255>, ptr<i8, 255>, i32)>>
    %a_mram = llvm.extractvalue %args[0]
        : !llvm.struct<"dpu_args", (ptr<i8, 255>, ptr<i8, 255>, ptr<i8, 255>, i32)>
    %b_mram = llvm.extractvalue %args[1]
        : !llvm.struct<"dpu_args", (ptr<i8, 255>, ptr<i8, 255>, ptr<i8, 255>, i32)>
    %c_mram = llvm.extractvalue %args[2]
        : !llvm.struct<"dpu_args", (ptr<i8, 255>, ptr<i8, 255>, ptr<i8, 255>, i32)>
    %n = llvm.extractvalue %args[3]
        : !llvm.struct<"dpu_args", (ptr<i8, 255>, ptr<i8, 255>, ptr<i8, 255>, i32)>

    %n64 = llvm.zext %n : i32 to i64
    %bytes_raw = llvm.mul %n, %c4_i32 : i32
    %bytes_padded = llvm.add %bytes_raw, %c7_i32 : i32
    %bytes_aligned = llvm.and %bytes_padded, %cneg8_i32 : i32

    %a_wram_i8 = llvm.call @mem_alloc(%bytes_aligned) : (i32) -> !llvm.ptr<i8>
    %b_wram_i8 = llvm.call @mem_alloc(%bytes_aligned) : (i32) -> !llvm.ptr<i8>
    %c_wram_i8 = llvm.call @mem_alloc(%bytes_aligned) : (i32) -> !llvm.ptr<i8>
    %a_wram = llvm.bitcast %a_wram_i8 : !llvm.ptr<i8> to !llvm.ptr<i32>
    %b_wram = llvm.bitcast %b_wram_i8 : !llvm.ptr<i8> to !llvm.ptr<i32>
    %c_wram = llvm.bitcast %c_wram_i8 : !llvm.ptr<i8> to !llvm.ptr<i32>

    dpu.ldma %a_wram_i8, %a_mram, %bytes_aligned
        : !llvm.ptr<i8>, !llvm.ptr<i8, 255>, i32
    dpu.ldma %b_wram_i8, %b_mram, %bytes_aligned
        : !llvm.ptr<i8>, !llvm.ptr<i8, 255>, i32

    llvm.br ^bb1(%c0 : i64)

  ^bb1(%i: i64):
    %cmp = llvm.icmp "slt" %i, %n64 : i64
    llvm.cond_br %cmp, ^bb2, ^bb3

  ^bb2:
    %a_ptr = llvm.getelementptr %a_wram[%i]
        : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
    %b_ptr = llvm.getelementptr %b_wram[%i]
        : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
    %c_ptr = llvm.getelementptr %c_wram[%i]
        : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
    %a_val = llvm.load %a_ptr : !llvm.ptr<i32>
    %b_val = llvm.load %b_ptr : !llvm.ptr<i32>
    %sum = llvm.add %a_val, %b_val : i32
    llvm.store %sum, %c_ptr : !llvm.ptr<i32>
    %next = llvm.add %i, %c1 : i64
    llvm.br ^bb1(%next : i64)

  ^bb3:
    dpu.sdma %c_wram_i8, %c_mram, %bytes_aligned
        : !llvm.ptr<i8>, !llvm.ptr<i8, 255>, i32
    llvm.br ^bb_exit

  ^bb_exit:
    llvm.return %c0_i32 : i32
  }
}
