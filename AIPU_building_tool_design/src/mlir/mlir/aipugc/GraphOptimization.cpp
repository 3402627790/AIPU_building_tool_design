//===- ToyCombine.cpp - Toy High Level Optimizer --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a set of simple combiners for optimizing operations in
// the aipugc dialect.
//
//===----------------------------------------------------------------------===//

#include "aipugc/Dialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include <numeric>
using namespace mlir;
using namespace aipugc;

namespace
{
/// Include the patterns defined in the Declarative Rewrite framework.
#include "aipugc/GraphOptimization.cpp.inc"
} // namespace

/// Fold constants.
// OpFoldResult ConstantOp::fold(ArrayRef<Attribute> operands) { return value(); }

/// Fold struct constants.
// OpFoldResult StructConstantOp::fold(ArrayRef<Attribute> operands) {
//   return value();
// }

/// Fold simple struct access operations that access into a constant.
// OpFoldResult StructAccessOp::fold(ArrayRef<Attribute> operands) {
//   auto structAttr = operands.front().dyn_cast_or_null<mlir::ArrayAttr>();
//   if (!structAttr)
//     return nullptr;

//   size_t elementIndex = index();
//   return structAttr[elementIndex];
// }

/// This is an example of a c++ rewrite pattern for the TransposeOp. It
/// optimizes the following scenario: transpose(transpose(x)) -> x
// struct FuseTransposeTranspose : public mlir::OpRewritePattern<TransposeOp>
// {
//   /// We register this pattern to match every toy.transpose in the IR.
//   /// The "benefit" is used by the framework to order the patterns and process
//   /// them in order of profitability.
//   FuseTransposeTranspose(mlir::MLIRContext *context)
//       : OpRewritePattern<TransposeOp>(context, /*benefit=*/1) {}

//   /// This method attempts to match a pattern and rewrite it. The rewriter
//   /// argument is the orchestrator of the sequence of rewrites. The pattern is
//   /// expected to interact with it to perform any changes to the IR from here.
//   mlir::LogicalResult
//   matchAndRewrite(TransposeOp op,
//                   mlir::PatternRewriter &rewriter) const override
//   {
//     // Look through the input of the current transpose.
//     mlir::Value transposeInput = op.getOperand();
//     TransposeOp transposeInputOp = transposeInput.getDefiningOp<TransposeOp>();

//     // Input defined by another transpose? If not, no match.
//     if (!transposeInputOp)
//       return failure();

//     // Otherwise, we have a redundant transpose. Use the rewriter.
//     // Rewrite
//     auto fusedLoc = rewriter.getFusedLoc({op.getLoc(), transposeInputOp.getLoc()});
//     TransposeOp newTrans;
//     {
//       SmallVector<Value, 4> values;
//       SmallVector<NamedAttribute, 4> attrs;
//       values.push_back(transposeInputOp.getOperand());
//       auto perm1 = op.perm();
//       auto perm0 = transposeInputOp.perm();
//       auto perm = perm1;
//       for (unsigned i = 0; i < perm.size(); ++i)
//       {
//         perm[i] = perm0[perm1[i].cast<IntegerAttr>().getInt()];
//       }
//       attrs.emplace_back(rewriter.getStringAttr("perm"), perm);

//       SmallVector<Type, 4> result_types;
//       for (auto v : op.getODSResults(0))
//       {
//         result_types.push_back(v.getType());
//       }
//       newTrans = rewriter.create<TransposeOp>(fusedLoc, result_types, values, attrs);
//     }

//     rewriter.replaceOp(op, {newTrans.getODSResults(0)});
//     return success();

//     // rewriter.replaceOp(op, {transposeInputOp.getOperand()});
//     // return success();
//   }
// };

/// Register our patterns as "canonicalization" patterns on the TransposeOp so
/// that they can be picked up by the Canonicalization framework.
// void TransposeOp::getCanonicalizationPatterns(RewritePatternSet &results,
//                                               MLIRContext *context)
// {
//   results.add<FuseTransposeTranspose>(context);
// }

/// Register our patterns as "canonicalization" patterns on the ReshapeOp so
/// that they can be picked up by the Canonicalization framework.
// void ReshapeOp::getCanonicalizationPatterns(RewritePatternSet &results,
//                                             MLIRContext *context)
// {
//   results.add<ReshapeReshapeOptPattern
//               // RedundantReshapeOptPattern,
//               //             FoldConstantReshapeOptPattern
//               >(context);
// }
