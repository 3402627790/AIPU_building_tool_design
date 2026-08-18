; ModuleID = 'isnan_kernel.cps'
source_filename = "isnan_kernel.cps"
target datalayout = "e-m:e-p:64:64-p3:32:32-p5:32:32-p6:32:32-i1:8-i32:32-i64:64-v32:32-v1024:1024-n32-S32-A5-G1"
target triple = "zhouyi-unknown-compass"

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite)
define dso_local zhouyi_kernel void @isnan_naive_float32_int8(ptr addrspace(1) noalias noundef readonly captures(none) %0, ptr addrspace(1) noalias noundef writeonly captures(none) %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = tail call noundef i32 @llvm.zhouyi.block.id.x()
  %5 = tail call noundef i32 @llvm.zhouyi.num.threads.x()
  %6 = mul i32 %4, %5
  %7 = tail call noundef i32 @llvm.zhouyi.thread.id.x()
  %8 = add i32 %6, %7
  %9 = tail call noundef i32 @llvm.zhouyi.num.blocks.x()
  %10 = mul i32 %9, %5
  %11 = sdiv i32 %2, 8
  %12 = shl nsw i32 %11, 3
  %13 = icmp slt i32 %8, %11
  br i1 %13, label %17, label %14

14:                                               ; preds = %17, %3
  %15 = sub nsw i32 %2, %12
  %16 = icmp slt i32 %8, %15
  br i1 %16, label %77, label %88

17:                                               ; preds = %3, %17
  %18 = phi i32 [ %75, %17 ], [ %8, %3 ]
  %19 = shl nsw i32 %18, 3
  %20 = sext i32 %19 to i64
  %21 = getelementptr inbounds float, ptr addrspace(1) %0, i64 %20
  %22 = load float, ptr addrspace(1) %21, align 4, !tbaa !3
  %23 = fcmp uno float %22, 0.000000e+00
  %24 = zext i1 %23 to i8
  %25 = getelementptr inbounds i8, ptr addrspace(1) %1, i64 %20
  store i8 %24, ptr addrspace(1) %25, align 1, !tbaa !7
  %26 = or disjoint i32 %19, 1
  %27 = sext i32 %26 to i64
  %28 = getelementptr inbounds float, ptr addrspace(1) %0, i64 %27
  %29 = load float, ptr addrspace(1) %28, align 4, !tbaa !3
  %30 = fcmp uno float %29, 0.000000e+00
  %31 = zext i1 %30 to i8
  %32 = getelementptr inbounds i8, ptr addrspace(1) %1, i64 %27
  store i8 %31, ptr addrspace(1) %32, align 1, !tbaa !7
  %33 = or disjoint i32 %19, 2
  %34 = sext i32 %33 to i64
  %35 = getelementptr inbounds float, ptr addrspace(1) %0, i64 %34
  %36 = load float, ptr addrspace(1) %35, align 4, !tbaa !3
  %37 = fcmp uno float %36, 0.000000e+00
  %38 = zext i1 %37 to i8
  %39 = getelementptr inbounds i8, ptr addrspace(1) %1, i64 %34
  store i8 %38, ptr addrspace(1) %39, align 1, !tbaa !7
  %40 = or disjoint i32 %19, 3
  %41 = sext i32 %40 to i64
  %42 = getelementptr inbounds float, ptr addrspace(1) %0, i64 %41
  %43 = load float, ptr addrspace(1) %42, align 4, !tbaa !3
  %44 = fcmp uno float %43, 0.000000e+00
  %45 = zext i1 %44 to i8
  %46 = getelementptr inbounds i8, ptr addrspace(1) %1, i64 %41
  store i8 %45, ptr addrspace(1) %46, align 1, !tbaa !7
  %47 = or disjoint i32 %19, 4
  %48 = sext i32 %47 to i64
  %49 = getelementptr inbounds float, ptr addrspace(1) %0, i64 %48
  %50 = load float, ptr addrspace(1) %49, align 4, !tbaa !3
  %51 = fcmp uno float %50, 0.000000e+00
  %52 = zext i1 %51 to i8
  %53 = getelementptr inbounds i8, ptr addrspace(1) %1, i64 %48
  store i8 %52, ptr addrspace(1) %53, align 1, !tbaa !7
  %54 = or disjoint i32 %19, 5
  %55 = sext i32 %54 to i64
  %56 = getelementptr inbounds float, ptr addrspace(1) %0, i64 %55
  %57 = load float, ptr addrspace(1) %56, align 4, !tbaa !3
  %58 = fcmp uno float %57, 0.000000e+00
  %59 = zext i1 %58 to i8
  %60 = getelementptr inbounds i8, ptr addrspace(1) %1, i64 %55
  store i8 %59, ptr addrspace(1) %60, align 1, !tbaa !7
  %61 = or disjoint i32 %19, 6
  %62 = sext i32 %61 to i64
  %63 = getelementptr inbounds float, ptr addrspace(1) %0, i64 %62
  %64 = load float, ptr addrspace(1) %63, align 4, !tbaa !3
  %65 = fcmp uno float %64, 0.000000e+00
  %66 = zext i1 %65 to i8
  %67 = getelementptr inbounds i8, ptr addrspace(1) %1, i64 %62
  store i8 %66, ptr addrspace(1) %67, align 1, !tbaa !7
  %68 = or disjoint i32 %19, 7
  %69 = sext i32 %68 to i64
  %70 = getelementptr inbounds float, ptr addrspace(1) %0, i64 %69
  %71 = load float, ptr addrspace(1) %70, align 4, !tbaa !3
  %72 = fcmp uno float %71, 0.000000e+00
  %73 = zext i1 %72 to i8
  %74 = getelementptr inbounds i8, ptr addrspace(1) %1, i64 %69
  store i8 %73, ptr addrspace(1) %74, align 1, !tbaa !7
  %75 = add nsw i32 %18, %10
  %76 = icmp slt i32 %75, %11
  br i1 %76, label %17, label %14

77:                                               ; preds = %14, %77
  %78 = phi i32 [ %86, %77 ], [ %8, %14 ]
  %79 = add nsw i32 %78, %12
  %80 = sext i32 %79 to i64
  %81 = getelementptr inbounds float, ptr addrspace(1) %0, i64 %80
  %82 = load float, ptr addrspace(1) %81, align 4, !tbaa !3
  %83 = fcmp uno float %82, 0.000000e+00
  %84 = zext i1 %83 to i8
  %85 = getelementptr inbounds i8, ptr addrspace(1) %1, i64 %80
  store i8 %84, ptr addrspace(1) %85, align 1, !tbaa !7
  %86 = add nsw i32 %78, %10
  %87 = icmp slt i32 %86, %15
  br i1 %87, label %77, label %88

88:                                               ; preds = %77, %14
  ret void
}

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite)
define dso_local zhouyi_kernel void @isnan_naive_float32_int16(ptr addrspace(1) noalias noundef readonly captures(none) %0, ptr addrspace(1) noalias noundef writeonly captures(none) %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = tail call noundef i32 @llvm.zhouyi.block.id.x()
  %5 = tail call noundef i32 @llvm.zhouyi.num.threads.x()
  %6 = mul i32 %4, %5
  %7 = tail call noundef i32 @llvm.zhouyi.thread.id.x()
  %8 = add i32 %6, %7
  %9 = tail call noundef i32 @llvm.zhouyi.num.blocks.x()
  %10 = mul i32 %9, %5
  %11 = sdiv i32 %2, 8
  %12 = shl nsw i32 %11, 3
  %13 = icmp slt i32 %8, %11
  br i1 %13, label %17, label %14

14:                                               ; preds = %17, %3
  %15 = sub nsw i32 %2, %12
  %16 = icmp slt i32 %8, %15
  br i1 %16, label %77, label %88

17:                                               ; preds = %3, %17
  %18 = phi i32 [ %75, %17 ], [ %8, %3 ]
  %19 = shl nsw i32 %18, 3
  %20 = sext i32 %19 to i64
  %21 = getelementptr inbounds float, ptr addrspace(1) %0, i64 %20
  %22 = load float, ptr addrspace(1) %21, align 4, !tbaa !3
  %23 = fcmp uno float %22, 0.000000e+00
  %24 = zext i1 %23 to i16
  %25 = getelementptr inbounds i16, ptr addrspace(1) %1, i64 %20
  store i16 %24, ptr addrspace(1) %25, align 2, !tbaa !8
  %26 = or disjoint i32 %19, 1
  %27 = sext i32 %26 to i64
  %28 = getelementptr inbounds float, ptr addrspace(1) %0, i64 %27
  %29 = load float, ptr addrspace(1) %28, align 4, !tbaa !3
  %30 = fcmp uno float %29, 0.000000e+00
  %31 = zext i1 %30 to i16
  %32 = getelementptr inbounds i16, ptr addrspace(1) %1, i64 %27
  store i16 %31, ptr addrspace(1) %32, align 2, !tbaa !8
  %33 = or disjoint i32 %19, 2
  %34 = sext i32 %33 to i64
  %35 = getelementptr inbounds float, ptr addrspace(1) %0, i64 %34
  %36 = load float, ptr addrspace(1) %35, align 4, !tbaa !3
  %37 = fcmp uno float %36, 0.000000e+00
  %38 = zext i1 %37 to i16
  %39 = getelementptr inbounds i16, ptr addrspace(1) %1, i64 %34
  store i16 %38, ptr addrspace(1) %39, align 2, !tbaa !8
  %40 = or disjoint i32 %19, 3
  %41 = sext i32 %40 to i64
  %42 = getelementptr inbounds float, ptr addrspace(1) %0, i64 %41
  %43 = load float, ptr addrspace(1) %42, align 4, !tbaa !3
  %44 = fcmp uno float %43, 0.000000e+00
  %45 = zext i1 %44 to i16
  %46 = getelementptr inbounds i16, ptr addrspace(1) %1, i64 %41
  store i16 %45, ptr addrspace(1) %46, align 2, !tbaa !8
  %47 = or disjoint i32 %19, 4
  %48 = sext i32 %47 to i64
  %49 = getelementptr inbounds float, ptr addrspace(1) %0, i64 %48
  %50 = load float, ptr addrspace(1) %49, align 4, !tbaa !3
  %51 = fcmp uno float %50, 0.000000e+00
  %52 = zext i1 %51 to i16
  %53 = getelementptr inbounds i16, ptr addrspace(1) %1, i64 %48
  store i16 %52, ptr addrspace(1) %53, align 2, !tbaa !8
  %54 = or disjoint i32 %19, 5
  %55 = sext i32 %54 to i64
  %56 = getelementptr inbounds float, ptr addrspace(1) %0, i64 %55
  %57 = load float, ptr addrspace(1) %56, align 4, !tbaa !3
  %58 = fcmp uno float %57, 0.000000e+00
  %59 = zext i1 %58 to i16
  %60 = getelementptr inbounds i16, ptr addrspace(1) %1, i64 %55
  store i16 %59, ptr addrspace(1) %60, align 2, !tbaa !8
  %61 = or disjoint i32 %19, 6
  %62 = sext i32 %61 to i64
  %63 = getelementptr inbounds float, ptr addrspace(1) %0, i64 %62
  %64 = load float, ptr addrspace(1) %63, align 4, !tbaa !3
  %65 = fcmp uno float %64, 0.000000e+00
  %66 = zext i1 %65 to i16
  %67 = getelementptr inbounds i16, ptr addrspace(1) %1, i64 %62
  store i16 %66, ptr addrspace(1) %67, align 2, !tbaa !8
  %68 = or disjoint i32 %19, 7
  %69 = sext i32 %68 to i64
  %70 = getelementptr inbounds float, ptr addrspace(1) %0, i64 %69
  %71 = load float, ptr addrspace(1) %70, align 4, !tbaa !3
  %72 = fcmp uno float %71, 0.000000e+00
  %73 = zext i1 %72 to i16
  %74 = getelementptr inbounds i16, ptr addrspace(1) %1, i64 %69
  store i16 %73, ptr addrspace(1) %74, align 2, !tbaa !8
  %75 = add nsw i32 %18, %10
  %76 = icmp slt i32 %75, %11
  br i1 %76, label %17, label %14

77:                                               ; preds = %14, %77
  %78 = phi i32 [ %86, %77 ], [ %8, %14 ]
  %79 = add nsw i32 %78, %12
  %80 = sext i32 %79 to i64
  %81 = getelementptr inbounds float, ptr addrspace(1) %0, i64 %80
  %82 = load float, ptr addrspace(1) %81, align 4, !tbaa !3
  %83 = fcmp uno float %82, 0.000000e+00
  %84 = zext i1 %83 to i16
  %85 = getelementptr inbounds i16, ptr addrspace(1) %1, i64 %80
  store i16 %84, ptr addrspace(1) %85, align 2, !tbaa !8
  %86 = add nsw i32 %78, %10
  %87 = icmp slt i32 %86, %15
  br i1 %87, label %77, label %88

88:                                               ; preds = %77, %14
  ret void
}

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite)
define dso_local zhouyi_kernel void @isnan_naive_float32_int32(ptr addrspace(1) noalias noundef readonly captures(none) %0, ptr addrspace(1) noalias noundef writeonly captures(none) %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = tail call noundef i32 @llvm.zhouyi.block.id.x()
  %5 = tail call noundef i32 @llvm.zhouyi.num.threads.x()
  %6 = mul i32 %4, %5
  %7 = tail call noundef i32 @llvm.zhouyi.thread.id.x()
  %8 = add i32 %6, %7
  %9 = tail call noundef i32 @llvm.zhouyi.num.blocks.x()
  %10 = mul i32 %9, %5
  %11 = sdiv i32 %2, 8
  %12 = shl nsw i32 %11, 3
  %13 = icmp slt i32 %8, %11
  br i1 %13, label %17, label %14

14:                                               ; preds = %17, %3
  %15 = sub nsw i32 %2, %12
  %16 = icmp slt i32 %8, %15
  br i1 %16, label %77, label %88

17:                                               ; preds = %3, %17
  %18 = phi i32 [ %75, %17 ], [ %8, %3 ]
  %19 = shl nsw i32 %18, 3
  %20 = sext i32 %19 to i64
  %21 = getelementptr inbounds float, ptr addrspace(1) %0, i64 %20
  %22 = load float, ptr addrspace(1) %21, align 4, !tbaa !3
  %23 = fcmp uno float %22, 0.000000e+00
  %24 = zext i1 %23 to i32
  %25 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %20
  store i32 %24, ptr addrspace(1) %25, align 4, !tbaa !10
  %26 = or disjoint i32 %19, 1
  %27 = sext i32 %26 to i64
  %28 = getelementptr inbounds float, ptr addrspace(1) %0, i64 %27
  %29 = load float, ptr addrspace(1) %28, align 4, !tbaa !3
  %30 = fcmp uno float %29, 0.000000e+00
  %31 = zext i1 %30 to i32
  %32 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %27
  store i32 %31, ptr addrspace(1) %32, align 4, !tbaa !10
  %33 = or disjoint i32 %19, 2
  %34 = sext i32 %33 to i64
  %35 = getelementptr inbounds float, ptr addrspace(1) %0, i64 %34
  %36 = load float, ptr addrspace(1) %35, align 4, !tbaa !3
  %37 = fcmp uno float %36, 0.000000e+00
  %38 = zext i1 %37 to i32
  %39 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %34
  store i32 %38, ptr addrspace(1) %39, align 4, !tbaa !10
  %40 = or disjoint i32 %19, 3
  %41 = sext i32 %40 to i64
  %42 = getelementptr inbounds float, ptr addrspace(1) %0, i64 %41
  %43 = load float, ptr addrspace(1) %42, align 4, !tbaa !3
  %44 = fcmp uno float %43, 0.000000e+00
  %45 = zext i1 %44 to i32
  %46 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %41
  store i32 %45, ptr addrspace(1) %46, align 4, !tbaa !10
  %47 = or disjoint i32 %19, 4
  %48 = sext i32 %47 to i64
  %49 = getelementptr inbounds float, ptr addrspace(1) %0, i64 %48
  %50 = load float, ptr addrspace(1) %49, align 4, !tbaa !3
  %51 = fcmp uno float %50, 0.000000e+00
  %52 = zext i1 %51 to i32
  %53 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %48
  store i32 %52, ptr addrspace(1) %53, align 4, !tbaa !10
  %54 = or disjoint i32 %19, 5
  %55 = sext i32 %54 to i64
  %56 = getelementptr inbounds float, ptr addrspace(1) %0, i64 %55
  %57 = load float, ptr addrspace(1) %56, align 4, !tbaa !3
  %58 = fcmp uno float %57, 0.000000e+00
  %59 = zext i1 %58 to i32
  %60 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %55
  store i32 %59, ptr addrspace(1) %60, align 4, !tbaa !10
  %61 = or disjoint i32 %19, 6
  %62 = sext i32 %61 to i64
  %63 = getelementptr inbounds float, ptr addrspace(1) %0, i64 %62
  %64 = load float, ptr addrspace(1) %63, align 4, !tbaa !3
  %65 = fcmp uno float %64, 0.000000e+00
  %66 = zext i1 %65 to i32
  %67 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %62
  store i32 %66, ptr addrspace(1) %67, align 4, !tbaa !10
  %68 = or disjoint i32 %19, 7
  %69 = sext i32 %68 to i64
  %70 = getelementptr inbounds float, ptr addrspace(1) %0, i64 %69
  %71 = load float, ptr addrspace(1) %70, align 4, !tbaa !3
  %72 = fcmp uno float %71, 0.000000e+00
  %73 = zext i1 %72 to i32
  %74 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %69
  store i32 %73, ptr addrspace(1) %74, align 4, !tbaa !10
  %75 = add nsw i32 %18, %10
  %76 = icmp slt i32 %75, %11
  br i1 %76, label %17, label %14

77:                                               ; preds = %14, %77
  %78 = phi i32 [ %86, %77 ], [ %8, %14 ]
  %79 = add nsw i32 %78, %12
  %80 = sext i32 %79 to i64
  %81 = getelementptr inbounds float, ptr addrspace(1) %0, i64 %80
  %82 = load float, ptr addrspace(1) %81, align 4, !tbaa !3
  %83 = fcmp uno float %82, 0.000000e+00
  %84 = zext i1 %83 to i32
  %85 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %80
  store i32 %84, ptr addrspace(1) %85, align 4, !tbaa !10
  %86 = add nsw i32 %78, %10
  %87 = icmp slt i32 %86, %15
  br i1 %87, label %77, label %88

88:                                               ; preds = %77, %14
  ret void
}

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite)
define dso_local zhouyi_kernel void @isnan_naive_float16_int8(ptr addrspace(1) noalias noundef readonly captures(none) %0, ptr addrspace(1) noalias noundef writeonly captures(none) %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = tail call noundef i32 @llvm.zhouyi.block.id.x()
  %5 = tail call noundef i32 @llvm.zhouyi.num.threads.x()
  %6 = mul i32 %4, %5
  %7 = tail call noundef i32 @llvm.zhouyi.thread.id.x()
  %8 = add i32 %6, %7
  %9 = tail call noundef i32 @llvm.zhouyi.num.blocks.x()
  %10 = mul i32 %9, %5
  %11 = sdiv i32 %2, 8
  %12 = shl nsw i32 %11, 3
  %13 = icmp slt i32 %8, %11
  br i1 %13, label %17, label %14

14:                                               ; preds = %17, %3
  %15 = sub nsw i32 %2, %12
  %16 = icmp slt i32 %8, %15
  br i1 %16, label %77, label %88

17:                                               ; preds = %3, %17
  %18 = phi i32 [ %75, %17 ], [ %8, %3 ]
  %19 = shl nsw i32 %18, 3
  %20 = sext i32 %19 to i64
  %21 = getelementptr inbounds half, ptr addrspace(1) %0, i64 %20
  %22 = load half, ptr addrspace(1) %21, align 2, !tbaa !12
  %23 = fcmp uno half %22, 0xH0000
  %24 = zext i1 %23 to i8
  %25 = getelementptr inbounds i8, ptr addrspace(1) %1, i64 %20
  store i8 %24, ptr addrspace(1) %25, align 1, !tbaa !7
  %26 = or disjoint i32 %19, 1
  %27 = sext i32 %26 to i64
  %28 = getelementptr inbounds half, ptr addrspace(1) %0, i64 %27
  %29 = load half, ptr addrspace(1) %28, align 2, !tbaa !12
  %30 = fcmp uno half %29, 0xH0000
  %31 = zext i1 %30 to i8
  %32 = getelementptr inbounds i8, ptr addrspace(1) %1, i64 %27
  store i8 %31, ptr addrspace(1) %32, align 1, !tbaa !7
  %33 = or disjoint i32 %19, 2
  %34 = sext i32 %33 to i64
  %35 = getelementptr inbounds half, ptr addrspace(1) %0, i64 %34
  %36 = load half, ptr addrspace(1) %35, align 2, !tbaa !12
  %37 = fcmp uno half %36, 0xH0000
  %38 = zext i1 %37 to i8
  %39 = getelementptr inbounds i8, ptr addrspace(1) %1, i64 %34
  store i8 %38, ptr addrspace(1) %39, align 1, !tbaa !7
  %40 = or disjoint i32 %19, 3
  %41 = sext i32 %40 to i64
  %42 = getelementptr inbounds half, ptr addrspace(1) %0, i64 %41
  %43 = load half, ptr addrspace(1) %42, align 2, !tbaa !12
  %44 = fcmp uno half %43, 0xH0000
  %45 = zext i1 %44 to i8
  %46 = getelementptr inbounds i8, ptr addrspace(1) %1, i64 %41
  store i8 %45, ptr addrspace(1) %46, align 1, !tbaa !7
  %47 = or disjoint i32 %19, 4
  %48 = sext i32 %47 to i64
  %49 = getelementptr inbounds half, ptr addrspace(1) %0, i64 %48
  %50 = load half, ptr addrspace(1) %49, align 2, !tbaa !12
  %51 = fcmp uno half %50, 0xH0000
  %52 = zext i1 %51 to i8
  %53 = getelementptr inbounds i8, ptr addrspace(1) %1, i64 %48
  store i8 %52, ptr addrspace(1) %53, align 1, !tbaa !7
  %54 = or disjoint i32 %19, 5
  %55 = sext i32 %54 to i64
  %56 = getelementptr inbounds half, ptr addrspace(1) %0, i64 %55
  %57 = load half, ptr addrspace(1) %56, align 2, !tbaa !12
  %58 = fcmp uno half %57, 0xH0000
  %59 = zext i1 %58 to i8
  %60 = getelementptr inbounds i8, ptr addrspace(1) %1, i64 %55
  store i8 %59, ptr addrspace(1) %60, align 1, !tbaa !7
  %61 = or disjoint i32 %19, 6
  %62 = sext i32 %61 to i64
  %63 = getelementptr inbounds half, ptr addrspace(1) %0, i64 %62
  %64 = load half, ptr addrspace(1) %63, align 2, !tbaa !12
  %65 = fcmp uno half %64, 0xH0000
  %66 = zext i1 %65 to i8
  %67 = getelementptr inbounds i8, ptr addrspace(1) %1, i64 %62
  store i8 %66, ptr addrspace(1) %67, align 1, !tbaa !7
  %68 = or disjoint i32 %19, 7
  %69 = sext i32 %68 to i64
  %70 = getelementptr inbounds half, ptr addrspace(1) %0, i64 %69
  %71 = load half, ptr addrspace(1) %70, align 2, !tbaa !12
  %72 = fcmp uno half %71, 0xH0000
  %73 = zext i1 %72 to i8
  %74 = getelementptr inbounds i8, ptr addrspace(1) %1, i64 %69
  store i8 %73, ptr addrspace(1) %74, align 1, !tbaa !7
  %75 = add nsw i32 %18, %10
  %76 = icmp slt i32 %75, %11
  br i1 %76, label %17, label %14

77:                                               ; preds = %14, %77
  %78 = phi i32 [ %86, %77 ], [ %8, %14 ]
  %79 = add nsw i32 %78, %12
  %80 = sext i32 %79 to i64
  %81 = getelementptr inbounds half, ptr addrspace(1) %0, i64 %80
  %82 = load half, ptr addrspace(1) %81, align 2, !tbaa !12
  %83 = fcmp uno half %82, 0xH0000
  %84 = zext i1 %83 to i8
  %85 = getelementptr inbounds i8, ptr addrspace(1) %1, i64 %80
  store i8 %84, ptr addrspace(1) %85, align 1, !tbaa !7
  %86 = add nsw i32 %78, %10
  %87 = icmp slt i32 %86, %15
  br i1 %87, label %77, label %88

88:                                               ; preds = %77, %14
  ret void
}

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite)
define dso_local zhouyi_kernel void @isnan_naive_float16_int16(ptr addrspace(1) noalias noundef readonly captures(none) %0, ptr addrspace(1) noalias noundef writeonly captures(none) %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = tail call noundef i32 @llvm.zhouyi.block.id.x()
  %5 = tail call noundef i32 @llvm.zhouyi.num.threads.x()
  %6 = mul i32 %4, %5
  %7 = tail call noundef i32 @llvm.zhouyi.thread.id.x()
  %8 = add i32 %6, %7
  %9 = tail call noundef i32 @llvm.zhouyi.num.blocks.x()
  %10 = mul i32 %9, %5
  %11 = sdiv i32 %2, 8
  %12 = shl nsw i32 %11, 3
  %13 = icmp slt i32 %8, %11
  br i1 %13, label %17, label %14

14:                                               ; preds = %17, %3
  %15 = sub nsw i32 %2, %12
  %16 = icmp slt i32 %8, %15
  br i1 %16, label %77, label %88

17:                                               ; preds = %3, %17
  %18 = phi i32 [ %75, %17 ], [ %8, %3 ]
  %19 = shl nsw i32 %18, 3
  %20 = sext i32 %19 to i64
  %21 = getelementptr inbounds half, ptr addrspace(1) %0, i64 %20
  %22 = load half, ptr addrspace(1) %21, align 2, !tbaa !12
  %23 = fcmp uno half %22, 0xH0000
  %24 = zext i1 %23 to i16
  %25 = getelementptr inbounds i16, ptr addrspace(1) %1, i64 %20
  store i16 %24, ptr addrspace(1) %25, align 2, !tbaa !8
  %26 = or disjoint i32 %19, 1
  %27 = sext i32 %26 to i64
  %28 = getelementptr inbounds half, ptr addrspace(1) %0, i64 %27
  %29 = load half, ptr addrspace(1) %28, align 2, !tbaa !12
  %30 = fcmp uno half %29, 0xH0000
  %31 = zext i1 %30 to i16
  %32 = getelementptr inbounds i16, ptr addrspace(1) %1, i64 %27
  store i16 %31, ptr addrspace(1) %32, align 2, !tbaa !8
  %33 = or disjoint i32 %19, 2
  %34 = sext i32 %33 to i64
  %35 = getelementptr inbounds half, ptr addrspace(1) %0, i64 %34
  %36 = load half, ptr addrspace(1) %35, align 2, !tbaa !12
  %37 = fcmp uno half %36, 0xH0000
  %38 = zext i1 %37 to i16
  %39 = getelementptr inbounds i16, ptr addrspace(1) %1, i64 %34
  store i16 %38, ptr addrspace(1) %39, align 2, !tbaa !8
  %40 = or disjoint i32 %19, 3
  %41 = sext i32 %40 to i64
  %42 = getelementptr inbounds half, ptr addrspace(1) %0, i64 %41
  %43 = load half, ptr addrspace(1) %42, align 2, !tbaa !12
  %44 = fcmp uno half %43, 0xH0000
  %45 = zext i1 %44 to i16
  %46 = getelementptr inbounds i16, ptr addrspace(1) %1, i64 %41
  store i16 %45, ptr addrspace(1) %46, align 2, !tbaa !8
  %47 = or disjoint i32 %19, 4
  %48 = sext i32 %47 to i64
  %49 = getelementptr inbounds half, ptr addrspace(1) %0, i64 %48
  %50 = load half, ptr addrspace(1) %49, align 2, !tbaa !12
  %51 = fcmp uno half %50, 0xH0000
  %52 = zext i1 %51 to i16
  %53 = getelementptr inbounds i16, ptr addrspace(1) %1, i64 %48
  store i16 %52, ptr addrspace(1) %53, align 2, !tbaa !8
  %54 = or disjoint i32 %19, 5
  %55 = sext i32 %54 to i64
  %56 = getelementptr inbounds half, ptr addrspace(1) %0, i64 %55
  %57 = load half, ptr addrspace(1) %56, align 2, !tbaa !12
  %58 = fcmp uno half %57, 0xH0000
  %59 = zext i1 %58 to i16
  %60 = getelementptr inbounds i16, ptr addrspace(1) %1, i64 %55
  store i16 %59, ptr addrspace(1) %60, align 2, !tbaa !8
  %61 = or disjoint i32 %19, 6
  %62 = sext i32 %61 to i64
  %63 = getelementptr inbounds half, ptr addrspace(1) %0, i64 %62
  %64 = load half, ptr addrspace(1) %63, align 2, !tbaa !12
  %65 = fcmp uno half %64, 0xH0000
  %66 = zext i1 %65 to i16
  %67 = getelementptr inbounds i16, ptr addrspace(1) %1, i64 %62
  store i16 %66, ptr addrspace(1) %67, align 2, !tbaa !8
  %68 = or disjoint i32 %19, 7
  %69 = sext i32 %68 to i64
  %70 = getelementptr inbounds half, ptr addrspace(1) %0, i64 %69
  %71 = load half, ptr addrspace(1) %70, align 2, !tbaa !12
  %72 = fcmp uno half %71, 0xH0000
  %73 = zext i1 %72 to i16
  %74 = getelementptr inbounds i16, ptr addrspace(1) %1, i64 %69
  store i16 %73, ptr addrspace(1) %74, align 2, !tbaa !8
  %75 = add nsw i32 %18, %10
  %76 = icmp slt i32 %75, %11
  br i1 %76, label %17, label %14

77:                                               ; preds = %14, %77
  %78 = phi i32 [ %86, %77 ], [ %8, %14 ]
  %79 = add nsw i32 %78, %12
  %80 = sext i32 %79 to i64
  %81 = getelementptr inbounds half, ptr addrspace(1) %0, i64 %80
  %82 = load half, ptr addrspace(1) %81, align 2, !tbaa !12
  %83 = fcmp uno half %82, 0xH0000
  %84 = zext i1 %83 to i16
  %85 = getelementptr inbounds i16, ptr addrspace(1) %1, i64 %80
  store i16 %84, ptr addrspace(1) %85, align 2, !tbaa !8
  %86 = add nsw i32 %78, %10
  %87 = icmp slt i32 %86, %15
  br i1 %87, label %77, label %88

88:                                               ; preds = %77, %14
  ret void
}

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite)
define dso_local zhouyi_kernel void @isnan_naive_float16_int32(ptr addrspace(1) noalias noundef readonly captures(none) %0, ptr addrspace(1) noalias noundef writeonly captures(none) %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = tail call noundef i32 @llvm.zhouyi.block.id.x()
  %5 = tail call noundef i32 @llvm.zhouyi.num.threads.x()
  %6 = mul i32 %4, %5
  %7 = tail call noundef i32 @llvm.zhouyi.thread.id.x()
  %8 = add i32 %6, %7
  %9 = tail call noundef i32 @llvm.zhouyi.num.blocks.x()
  %10 = mul i32 %9, %5
  %11 = sdiv i32 %2, 8
  %12 = shl nsw i32 %11, 3
  %13 = icmp slt i32 %8, %11
  br i1 %13, label %17, label %14

14:                                               ; preds = %17, %3
  %15 = sub nsw i32 %2, %12
  %16 = icmp slt i32 %8, %15
  br i1 %16, label %77, label %88

17:                                               ; preds = %3, %17
  %18 = phi i32 [ %75, %17 ], [ %8, %3 ]
  %19 = shl nsw i32 %18, 3
  %20 = sext i32 %19 to i64
  %21 = getelementptr inbounds half, ptr addrspace(1) %0, i64 %20
  %22 = load half, ptr addrspace(1) %21, align 2, !tbaa !12
  %23 = fcmp uno half %22, 0xH0000
  %24 = zext i1 %23 to i32
  %25 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %20
  store i32 %24, ptr addrspace(1) %25, align 4, !tbaa !10
  %26 = or disjoint i32 %19, 1
  %27 = sext i32 %26 to i64
  %28 = getelementptr inbounds half, ptr addrspace(1) %0, i64 %27
  %29 = load half, ptr addrspace(1) %28, align 2, !tbaa !12
  %30 = fcmp uno half %29, 0xH0000
  %31 = zext i1 %30 to i32
  %32 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %27
  store i32 %31, ptr addrspace(1) %32, align 4, !tbaa !10
  %33 = or disjoint i32 %19, 2
  %34 = sext i32 %33 to i64
  %35 = getelementptr inbounds half, ptr addrspace(1) %0, i64 %34
  %36 = load half, ptr addrspace(1) %35, align 2, !tbaa !12
  %37 = fcmp uno half %36, 0xH0000
  %38 = zext i1 %37 to i32
  %39 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %34
  store i32 %38, ptr addrspace(1) %39, align 4, !tbaa !10
  %40 = or disjoint i32 %19, 3
  %41 = sext i32 %40 to i64
  %42 = getelementptr inbounds half, ptr addrspace(1) %0, i64 %41
  %43 = load half, ptr addrspace(1) %42, align 2, !tbaa !12
  %44 = fcmp uno half %43, 0xH0000
  %45 = zext i1 %44 to i32
  %46 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %41
  store i32 %45, ptr addrspace(1) %46, align 4, !tbaa !10
  %47 = or disjoint i32 %19, 4
  %48 = sext i32 %47 to i64
  %49 = getelementptr inbounds half, ptr addrspace(1) %0, i64 %48
  %50 = load half, ptr addrspace(1) %49, align 2, !tbaa !12
  %51 = fcmp uno half %50, 0xH0000
  %52 = zext i1 %51 to i32
  %53 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %48
  store i32 %52, ptr addrspace(1) %53, align 4, !tbaa !10
  %54 = or disjoint i32 %19, 5
  %55 = sext i32 %54 to i64
  %56 = getelementptr inbounds half, ptr addrspace(1) %0, i64 %55
  %57 = load half, ptr addrspace(1) %56, align 2, !tbaa !12
  %58 = fcmp uno half %57, 0xH0000
  %59 = zext i1 %58 to i32
  %60 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %55
  store i32 %59, ptr addrspace(1) %60, align 4, !tbaa !10
  %61 = or disjoint i32 %19, 6
  %62 = sext i32 %61 to i64
  %63 = getelementptr inbounds half, ptr addrspace(1) %0, i64 %62
  %64 = load half, ptr addrspace(1) %63, align 2, !tbaa !12
  %65 = fcmp uno half %64, 0xH0000
  %66 = zext i1 %65 to i32
  %67 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %62
  store i32 %66, ptr addrspace(1) %67, align 4, !tbaa !10
  %68 = or disjoint i32 %19, 7
  %69 = sext i32 %68 to i64
  %70 = getelementptr inbounds half, ptr addrspace(1) %0, i64 %69
  %71 = load half, ptr addrspace(1) %70, align 2, !tbaa !12
  %72 = fcmp uno half %71, 0xH0000
  %73 = zext i1 %72 to i32
  %74 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %69
  store i32 %73, ptr addrspace(1) %74, align 4, !tbaa !10
  %75 = add nsw i32 %18, %10
  %76 = icmp slt i32 %75, %11
  br i1 %76, label %17, label %14

77:                                               ; preds = %14, %77
  %78 = phi i32 [ %86, %77 ], [ %8, %14 ]
  %79 = add nsw i32 %78, %12
  %80 = sext i32 %79 to i64
  %81 = getelementptr inbounds half, ptr addrspace(1) %0, i64 %80
  %82 = load half, ptr addrspace(1) %81, align 2, !tbaa !12
  %83 = fcmp uno half %82, 0xH0000
  %84 = zext i1 %83 to i32
  %85 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %80
  store i32 %84, ptr addrspace(1) %85, align 4, !tbaa !10
  %86 = add nsw i32 %78, %10
  %87 = icmp slt i32 %86, %15
  br i1 %87, label %77, label %88

88:                                               ; preds = %77, %14
  ret void
}

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite)
define dso_local zhouyi_kernel void @isnan_naive_bfloat16_int8(ptr addrspace(1) noalias noundef readonly captures(none) %0, ptr addrspace(1) noalias noundef writeonly captures(none) %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = tail call noundef i32 @llvm.zhouyi.block.id.x()
  %5 = tail call noundef i32 @llvm.zhouyi.num.threads.x()
  %6 = mul i32 %4, %5
  %7 = tail call noundef i32 @llvm.zhouyi.thread.id.x()
  %8 = add i32 %6, %7
  %9 = tail call noundef i32 @llvm.zhouyi.num.blocks.x()
  %10 = mul i32 %9, %5
  %11 = sdiv i32 %2, 8
  %12 = shl nsw i32 %11, 3
  %13 = icmp slt i32 %8, %11
  br i1 %13, label %17, label %14

14:                                               ; preds = %17, %3
  %15 = sub nsw i32 %2, %12
  %16 = icmp slt i32 %8, %15
  br i1 %16, label %77, label %88

17:                                               ; preds = %3, %17
  %18 = phi i32 [ %75, %17 ], [ %8, %3 ]
  %19 = shl nsw i32 %18, 3
  %20 = sext i32 %19 to i64
  %21 = getelementptr inbounds bfloat, ptr addrspace(1) %0, i64 %20
  %22 = load bfloat, ptr addrspace(1) %21, align 2, !tbaa !14
  %23 = fcmp uno bfloat %22, 0xR0000
  %24 = zext i1 %23 to i8
  %25 = getelementptr inbounds i8, ptr addrspace(1) %1, i64 %20
  store i8 %24, ptr addrspace(1) %25, align 1, !tbaa !7
  %26 = or disjoint i32 %19, 1
  %27 = sext i32 %26 to i64
  %28 = getelementptr inbounds bfloat, ptr addrspace(1) %0, i64 %27
  %29 = load bfloat, ptr addrspace(1) %28, align 2, !tbaa !14
  %30 = fcmp uno bfloat %29, 0xR0000
  %31 = zext i1 %30 to i8
  %32 = getelementptr inbounds i8, ptr addrspace(1) %1, i64 %27
  store i8 %31, ptr addrspace(1) %32, align 1, !tbaa !7
  %33 = or disjoint i32 %19, 2
  %34 = sext i32 %33 to i64
  %35 = getelementptr inbounds bfloat, ptr addrspace(1) %0, i64 %34
  %36 = load bfloat, ptr addrspace(1) %35, align 2, !tbaa !14
  %37 = fcmp uno bfloat %36, 0xR0000
  %38 = zext i1 %37 to i8
  %39 = getelementptr inbounds i8, ptr addrspace(1) %1, i64 %34
  store i8 %38, ptr addrspace(1) %39, align 1, !tbaa !7
  %40 = or disjoint i32 %19, 3
  %41 = sext i32 %40 to i64
  %42 = getelementptr inbounds bfloat, ptr addrspace(1) %0, i64 %41
  %43 = load bfloat, ptr addrspace(1) %42, align 2, !tbaa !14
  %44 = fcmp uno bfloat %43, 0xR0000
  %45 = zext i1 %44 to i8
  %46 = getelementptr inbounds i8, ptr addrspace(1) %1, i64 %41
  store i8 %45, ptr addrspace(1) %46, align 1, !tbaa !7
  %47 = or disjoint i32 %19, 4
  %48 = sext i32 %47 to i64
  %49 = getelementptr inbounds bfloat, ptr addrspace(1) %0, i64 %48
  %50 = load bfloat, ptr addrspace(1) %49, align 2, !tbaa !14
  %51 = fcmp uno bfloat %50, 0xR0000
  %52 = zext i1 %51 to i8
  %53 = getelementptr inbounds i8, ptr addrspace(1) %1, i64 %48
  store i8 %52, ptr addrspace(1) %53, align 1, !tbaa !7
  %54 = or disjoint i32 %19, 5
  %55 = sext i32 %54 to i64
  %56 = getelementptr inbounds bfloat, ptr addrspace(1) %0, i64 %55
  %57 = load bfloat, ptr addrspace(1) %56, align 2, !tbaa !14
  %58 = fcmp uno bfloat %57, 0xR0000
  %59 = zext i1 %58 to i8
  %60 = getelementptr inbounds i8, ptr addrspace(1) %1, i64 %55
  store i8 %59, ptr addrspace(1) %60, align 1, !tbaa !7
  %61 = or disjoint i32 %19, 6
  %62 = sext i32 %61 to i64
  %63 = getelementptr inbounds bfloat, ptr addrspace(1) %0, i64 %62
  %64 = load bfloat, ptr addrspace(1) %63, align 2, !tbaa !14
  %65 = fcmp uno bfloat %64, 0xR0000
  %66 = zext i1 %65 to i8
  %67 = getelementptr inbounds i8, ptr addrspace(1) %1, i64 %62
  store i8 %66, ptr addrspace(1) %67, align 1, !tbaa !7
  %68 = or disjoint i32 %19, 7
  %69 = sext i32 %68 to i64
  %70 = getelementptr inbounds bfloat, ptr addrspace(1) %0, i64 %69
  %71 = load bfloat, ptr addrspace(1) %70, align 2, !tbaa !14
  %72 = fcmp uno bfloat %71, 0xR0000
  %73 = zext i1 %72 to i8
  %74 = getelementptr inbounds i8, ptr addrspace(1) %1, i64 %69
  store i8 %73, ptr addrspace(1) %74, align 1, !tbaa !7
  %75 = add nsw i32 %18, %10
  %76 = icmp slt i32 %75, %11
  br i1 %76, label %17, label %14

77:                                               ; preds = %14, %77
  %78 = phi i32 [ %86, %77 ], [ %8, %14 ]
  %79 = add nsw i32 %78, %12
  %80 = sext i32 %79 to i64
  %81 = getelementptr inbounds bfloat, ptr addrspace(1) %0, i64 %80
  %82 = load bfloat, ptr addrspace(1) %81, align 2, !tbaa !14
  %83 = fcmp uno bfloat %82, 0xR0000
  %84 = zext i1 %83 to i8
  %85 = getelementptr inbounds i8, ptr addrspace(1) %1, i64 %80
  store i8 %84, ptr addrspace(1) %85, align 1, !tbaa !7
  %86 = add nsw i32 %78, %10
  %87 = icmp slt i32 %86, %15
  br i1 %87, label %77, label %88

88:                                               ; preds = %77, %14
  ret void
}

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite)
define dso_local zhouyi_kernel void @isnan_naive_bfloat16_int16(ptr addrspace(1) noalias noundef readonly captures(none) %0, ptr addrspace(1) noalias noundef writeonly captures(none) %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = tail call noundef i32 @llvm.zhouyi.block.id.x()
  %5 = tail call noundef i32 @llvm.zhouyi.num.threads.x()
  %6 = mul i32 %4, %5
  %7 = tail call noundef i32 @llvm.zhouyi.thread.id.x()
  %8 = add i32 %6, %7
  %9 = tail call noundef i32 @llvm.zhouyi.num.blocks.x()
  %10 = mul i32 %9, %5
  %11 = sdiv i32 %2, 8
  %12 = shl nsw i32 %11, 3
  %13 = icmp slt i32 %8, %11
  br i1 %13, label %17, label %14

14:                                               ; preds = %17, %3
  %15 = sub nsw i32 %2, %12
  %16 = icmp slt i32 %8, %15
  br i1 %16, label %77, label %88

17:                                               ; preds = %3, %17
  %18 = phi i32 [ %75, %17 ], [ %8, %3 ]
  %19 = shl nsw i32 %18, 3
  %20 = sext i32 %19 to i64
  %21 = getelementptr inbounds bfloat, ptr addrspace(1) %0, i64 %20
  %22 = load bfloat, ptr addrspace(1) %21, align 2, !tbaa !14
  %23 = fcmp uno bfloat %22, 0xR0000
  %24 = zext i1 %23 to i16
  %25 = getelementptr inbounds i16, ptr addrspace(1) %1, i64 %20
  store i16 %24, ptr addrspace(1) %25, align 2, !tbaa !8
  %26 = or disjoint i32 %19, 1
  %27 = sext i32 %26 to i64
  %28 = getelementptr inbounds bfloat, ptr addrspace(1) %0, i64 %27
  %29 = load bfloat, ptr addrspace(1) %28, align 2, !tbaa !14
  %30 = fcmp uno bfloat %29, 0xR0000
  %31 = zext i1 %30 to i16
  %32 = getelementptr inbounds i16, ptr addrspace(1) %1, i64 %27
  store i16 %31, ptr addrspace(1) %32, align 2, !tbaa !8
  %33 = or disjoint i32 %19, 2
  %34 = sext i32 %33 to i64
  %35 = getelementptr inbounds bfloat, ptr addrspace(1) %0, i64 %34
  %36 = load bfloat, ptr addrspace(1) %35, align 2, !tbaa !14
  %37 = fcmp uno bfloat %36, 0xR0000
  %38 = zext i1 %37 to i16
  %39 = getelementptr inbounds i16, ptr addrspace(1) %1, i64 %34
  store i16 %38, ptr addrspace(1) %39, align 2, !tbaa !8
  %40 = or disjoint i32 %19, 3
  %41 = sext i32 %40 to i64
  %42 = getelementptr inbounds bfloat, ptr addrspace(1) %0, i64 %41
  %43 = load bfloat, ptr addrspace(1) %42, align 2, !tbaa !14
  %44 = fcmp uno bfloat %43, 0xR0000
  %45 = zext i1 %44 to i16
  %46 = getelementptr inbounds i16, ptr addrspace(1) %1, i64 %41
  store i16 %45, ptr addrspace(1) %46, align 2, !tbaa !8
  %47 = or disjoint i32 %19, 4
  %48 = sext i32 %47 to i64
  %49 = getelementptr inbounds bfloat, ptr addrspace(1) %0, i64 %48
  %50 = load bfloat, ptr addrspace(1) %49, align 2, !tbaa !14
  %51 = fcmp uno bfloat %50, 0xR0000
  %52 = zext i1 %51 to i16
  %53 = getelementptr inbounds i16, ptr addrspace(1) %1, i64 %48
  store i16 %52, ptr addrspace(1) %53, align 2, !tbaa !8
  %54 = or disjoint i32 %19, 5
  %55 = sext i32 %54 to i64
  %56 = getelementptr inbounds bfloat, ptr addrspace(1) %0, i64 %55
  %57 = load bfloat, ptr addrspace(1) %56, align 2, !tbaa !14
  %58 = fcmp uno bfloat %57, 0xR0000
  %59 = zext i1 %58 to i16
  %60 = getelementptr inbounds i16, ptr addrspace(1) %1, i64 %55
  store i16 %59, ptr addrspace(1) %60, align 2, !tbaa !8
  %61 = or disjoint i32 %19, 6
  %62 = sext i32 %61 to i64
  %63 = getelementptr inbounds bfloat, ptr addrspace(1) %0, i64 %62
  %64 = load bfloat, ptr addrspace(1) %63, align 2, !tbaa !14
  %65 = fcmp uno bfloat %64, 0xR0000
  %66 = zext i1 %65 to i16
  %67 = getelementptr inbounds i16, ptr addrspace(1) %1, i64 %62
  store i16 %66, ptr addrspace(1) %67, align 2, !tbaa !8
  %68 = or disjoint i32 %19, 7
  %69 = sext i32 %68 to i64
  %70 = getelementptr inbounds bfloat, ptr addrspace(1) %0, i64 %69
  %71 = load bfloat, ptr addrspace(1) %70, align 2, !tbaa !14
  %72 = fcmp uno bfloat %71, 0xR0000
  %73 = zext i1 %72 to i16
  %74 = getelementptr inbounds i16, ptr addrspace(1) %1, i64 %69
  store i16 %73, ptr addrspace(1) %74, align 2, !tbaa !8
  %75 = add nsw i32 %18, %10
  %76 = icmp slt i32 %75, %11
  br i1 %76, label %17, label %14

77:                                               ; preds = %14, %77
  %78 = phi i32 [ %86, %77 ], [ %8, %14 ]
  %79 = add nsw i32 %78, %12
  %80 = sext i32 %79 to i64
  %81 = getelementptr inbounds bfloat, ptr addrspace(1) %0, i64 %80
  %82 = load bfloat, ptr addrspace(1) %81, align 2, !tbaa !14
  %83 = fcmp uno bfloat %82, 0xR0000
  %84 = zext i1 %83 to i16
  %85 = getelementptr inbounds i16, ptr addrspace(1) %1, i64 %80
  store i16 %84, ptr addrspace(1) %85, align 2, !tbaa !8
  %86 = add nsw i32 %78, %10
  %87 = icmp slt i32 %86, %15
  br i1 %87, label %77, label %88

88:                                               ; preds = %77, %14
  ret void
}

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite)
define dso_local zhouyi_kernel void @isnan_naive_bfloat16_int32(ptr addrspace(1) noalias noundef readonly captures(none) %0, ptr addrspace(1) noalias noundef writeonly captures(none) %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = tail call noundef i32 @llvm.zhouyi.block.id.x()
  %5 = tail call noundef i32 @llvm.zhouyi.num.threads.x()
  %6 = mul i32 %4, %5
  %7 = tail call noundef i32 @llvm.zhouyi.thread.id.x()
  %8 = add i32 %6, %7
  %9 = tail call noundef i32 @llvm.zhouyi.num.blocks.x()
  %10 = mul i32 %9, %5
  %11 = sdiv i32 %2, 8
  %12 = shl nsw i32 %11, 3
  %13 = icmp slt i32 %8, %11
  br i1 %13, label %17, label %14

14:                                               ; preds = %17, %3
  %15 = sub nsw i32 %2, %12
  %16 = icmp slt i32 %8, %15
  br i1 %16, label %77, label %88

17:                                               ; preds = %3, %17
  %18 = phi i32 [ %75, %17 ], [ %8, %3 ]
  %19 = shl nsw i32 %18, 3
  %20 = sext i32 %19 to i64
  %21 = getelementptr inbounds bfloat, ptr addrspace(1) %0, i64 %20
  %22 = load bfloat, ptr addrspace(1) %21, align 2, !tbaa !14
  %23 = fcmp uno bfloat %22, 0xR0000
  %24 = zext i1 %23 to i32
  %25 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %20
  store i32 %24, ptr addrspace(1) %25, align 4, !tbaa !10
  %26 = or disjoint i32 %19, 1
  %27 = sext i32 %26 to i64
  %28 = getelementptr inbounds bfloat, ptr addrspace(1) %0, i64 %27
  %29 = load bfloat, ptr addrspace(1) %28, align 2, !tbaa !14
  %30 = fcmp uno bfloat %29, 0xR0000
  %31 = zext i1 %30 to i32
  %32 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %27
  store i32 %31, ptr addrspace(1) %32, align 4, !tbaa !10
  %33 = or disjoint i32 %19, 2
  %34 = sext i32 %33 to i64
  %35 = getelementptr inbounds bfloat, ptr addrspace(1) %0, i64 %34
  %36 = load bfloat, ptr addrspace(1) %35, align 2, !tbaa !14
  %37 = fcmp uno bfloat %36, 0xR0000
  %38 = zext i1 %37 to i32
  %39 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %34
  store i32 %38, ptr addrspace(1) %39, align 4, !tbaa !10
  %40 = or disjoint i32 %19, 3
  %41 = sext i32 %40 to i64
  %42 = getelementptr inbounds bfloat, ptr addrspace(1) %0, i64 %41
  %43 = load bfloat, ptr addrspace(1) %42, align 2, !tbaa !14
  %44 = fcmp uno bfloat %43, 0xR0000
  %45 = zext i1 %44 to i32
  %46 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %41
  store i32 %45, ptr addrspace(1) %46, align 4, !tbaa !10
  %47 = or disjoint i32 %19, 4
  %48 = sext i32 %47 to i64
  %49 = getelementptr inbounds bfloat, ptr addrspace(1) %0, i64 %48
  %50 = load bfloat, ptr addrspace(1) %49, align 2, !tbaa !14
  %51 = fcmp uno bfloat %50, 0xR0000
  %52 = zext i1 %51 to i32
  %53 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %48
  store i32 %52, ptr addrspace(1) %53, align 4, !tbaa !10
  %54 = or disjoint i32 %19, 5
  %55 = sext i32 %54 to i64
  %56 = getelementptr inbounds bfloat, ptr addrspace(1) %0, i64 %55
  %57 = load bfloat, ptr addrspace(1) %56, align 2, !tbaa !14
  %58 = fcmp uno bfloat %57, 0xR0000
  %59 = zext i1 %58 to i32
  %60 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %55
  store i32 %59, ptr addrspace(1) %60, align 4, !tbaa !10
  %61 = or disjoint i32 %19, 6
  %62 = sext i32 %61 to i64
  %63 = getelementptr inbounds bfloat, ptr addrspace(1) %0, i64 %62
  %64 = load bfloat, ptr addrspace(1) %63, align 2, !tbaa !14
  %65 = fcmp uno bfloat %64, 0xR0000
  %66 = zext i1 %65 to i32
  %67 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %62
  store i32 %66, ptr addrspace(1) %67, align 4, !tbaa !10
  %68 = or disjoint i32 %19, 7
  %69 = sext i32 %68 to i64
  %70 = getelementptr inbounds bfloat, ptr addrspace(1) %0, i64 %69
  %71 = load bfloat, ptr addrspace(1) %70, align 2, !tbaa !14
  %72 = fcmp uno bfloat %71, 0xR0000
  %73 = zext i1 %72 to i32
  %74 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %69
  store i32 %73, ptr addrspace(1) %74, align 4, !tbaa !10
  %75 = add nsw i32 %18, %10
  %76 = icmp slt i32 %75, %11
  br i1 %76, label %17, label %14

77:                                               ; preds = %14, %77
  %78 = phi i32 [ %86, %77 ], [ %8, %14 ]
  %79 = add nsw i32 %78, %12
  %80 = sext i32 %79 to i64
  %81 = getelementptr inbounds bfloat, ptr addrspace(1) %0, i64 %80
  %82 = load bfloat, ptr addrspace(1) %81, align 2, !tbaa !14
  %83 = fcmp uno bfloat %82, 0xR0000
  %84 = zext i1 %83 to i32
  %85 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %80
  store i32 %84, ptr addrspace(1) %85, align 4, !tbaa !10
  %86 = add nsw i32 %78, %10
  %87 = icmp slt i32 %86, %15
  br i1 %87, label %77, label %88

88:                                               ; preds = %77, %14
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.zhouyi.block.id.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.zhouyi.num.threads.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.zhouyi.thread.id.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.zhouyi.num.blocks.x() #1

attributes #0 = { nofree norecurse nosync nounwind memory(argmem: readwrite) "frame-pointer"="all" "no-jump-tables"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "uniform-work-group-size"="true" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2, !2, !2, !2, !2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{!"clang version 21.1.0 (ssh://regai02@szc-gerritai02.armchina.com:29418/AIPU_llvm_design c611252af12641c6d9cc09e0ed66fb86916f76aa)"}
!3 = !{!4, !4, i64 0}
!4 = !{!"float", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
!7 = !{!5, !5, i64 0}
!8 = !{!9, !9, i64 0}
!9 = !{!"short", !5, i64 0}
!10 = !{!11, !11, i64 0}
!11 = !{!"int", !5, i64 0}
!12 = !{!13, !13, i64 0}
!13 = !{!"_Float16", !5, i64 0}
!14 = !{!15, !15, i64 0}
!15 = !{!"__bf16", !5, i64 0}
