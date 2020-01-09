//===- ConvertStandardToLLVM.cpp - Standard to LLVM dialect conversion-----===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// TODO
//
//===----------------------------------------------------------------------===//

//#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
//#include "mlir/ADT/TypeSwitch.h"
//#include "mlir/Conversion/LoopToStandard/ConvertLoopToStandard.h"
//#include "mlir/Conversion/StandardToLLVM/BarePtrMemRefLowering.h"
#include "mlir/Conversion/StandardToLLVM/CommonStandardToLLVMConversion.h"
//#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
//#include "mlir/Dialect/StandardOps/Ops.h"
//#include "mlir/IR/Builders.h"
//#include "mlir/IR/MLIRContext.h"
//#include "mlir/IR/Module.h"
//#include "mlir/IR/PatternMatch.h"
//#include "mlir/Pass/Pass.h"
//#include "mlir/Support/Functional.h"
//#include "mlir/Transforms/DialectConversion.h"
//#include "mlir/Transforms/Passes.h"
//#include "mlir/Transforms/Utils.h"

//#include "llvm/IR/DerivedTypes.h"
//#include "llvm/IR/IRBuilder.h"
//#include "llvm/IR/Type.h"
//#include "llvm/Support/CommandLine.h"

using namespace mlir;

// Check if the MemRefType `type` is supported by the lowering. We currently
// only support memrefs with identity maps.
bool LLVMLegalizationPattern::isSupportedMemRefType() {
  return type.getAffineMaps().empty() ||
         llvm::all_of(type.getAffineMaps(),
                      [](AffineMap map) { return map.isIdentity(); });
}

PatternMatchResult LoadOpLowering::matchAndRewrite(
    Operation *op, ArrayRef<ValuePtr> operands,
    ConversionPatternRewriter &rewriter) const override {
  auto loadOp = cast<LoadOp>(op);
  OperandAdaptor<LoadOp> transformed(operands);
  auto type = loadOp.getMemRefType();

  ValuePtr dataPtr = getDataPtr(op->getLoc(), type, transformed.memref(),
                                transformed.indices(), rewriter, getModule());
  rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, dataPtr);
  return matchSuccess();
}

PatternMatchResult StoreOpLowering::matchAndRewrite(
    Operation *op, ArrayRef<ValuePtr> operands,
    ConversionPatternRewriter &rewriter) const override {
  auto type = cast<StoreOp>(op).getMemRefType();
  OperandAdaptor<StoreOp> transformed(operands);

  ValuePtr dataPtr = getDataPtr(op->getLoc(), type, transformed.memref(),
                                transformed.indices(), rewriter, getModule());
  rewriter.replaceOpWithNewOp<LLVM::StoreOp>(op, transformed.value(), dataPtr);
  return matchSuccess();
}
