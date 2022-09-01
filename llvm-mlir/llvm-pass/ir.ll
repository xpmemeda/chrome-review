; ModuleID = 'm'
source_filename = "m"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@0 = private unnamed_addr constant [11 x i8] c"At Runtime\00", align 1
@1 = private unnamed_addr constant [15 x i8] c"%s: arg = %ld\0A\00", align 1

define void @fun(i8* %0, float* %1, i64 %2) {
init_block:
  %3 = call i32 (...) @printf(i8* getelementptr inbounds ([15 x i8], [15 x i8]* @1, i32 0, i32 0), i8* getelementptr inbounds ([11 x i8], [11 x i8]* @0, i32 0, i32 0), i64 %2)
  %4 = bitcast i8* %0 to float*
  %counter = alloca i64, align 8
  store i64 0, i64* %counter, align 8
  br label %cmp_block

cmp_block:                                        ; preds = %body_block, %init_block
  %5 = load i64, i64* %counter, align 8
  %6 = icmp ult i64 %5, %2
  br i1 %6, label %body_block, label %end_block

body_block:                                       ; preds = %cmp_block
  %7 = getelementptr float, float* %4, i64 %5
  %8 = getelementptr float, float* %1, i64 %5
  %9 = load float, float* %7, align 4
  %10 = fmul float %9, 2.000000e+00
  store float %10, float* %8, align 4
  %11 = add i64 %5, 1
  store i64 %11, i64* %counter, align 8
  br label %cmp_block

end_block:                                        ; preds = %cmp_block
  store i64 0, i64* %counter, align 8
  ret void
}

declare i32 @printf(...)
