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

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/ADT/TypeSwitch.h"
#include "mlir/Conversion/LoopToStandard/ConvertLoopToStandard.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/Functional.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir;

extern llvm::cl::opt<bool> clUseAlloca;

// Check if the MemRefType `type` is supported by the lowering. We currently
// only support memrefs with identity maps and static shapes.
static bool isSupportedMemRefType(MemRefType type) {
  return (type.getAffineMaps().empty() ||
          llvm::all_of(type.getAffineMaps(),
                       [](AffineMap map) { return map.isIdentity(); })) &&
         type.hasStaticShape();
}

namespace {

// Common base for load and store operations on MemRefs.  Restricts the match
// to supported MemRef types.  Provides functionality to emit code accessing a
// specific element of the underlying data buffer.
template <typename Derived>
struct LoadStoreOpLowering : public LLVMLegalizationPattern<Derived> {
  using LLVMLegalizationPattern<Derived>::LLVMLegalizationPattern;
  using Base = LoadStoreOpLowering<Derived>;

  PatternMatchResult match(Operation *op) const override {
    MemRefType type = cast<Derived>(op).getMemRefType();
    return isSupportedMemRefType(type) ? this->matchSuccess()
                                       : this->matchFailure();
  }

  // TODO: Refactor?
  // Given subscript indices and array sizes in row-major order,
  //   i_n, i_{n-1}, ..., i_1
  //   s_n, s_{n-1}, ..., s_1
  // obtain a value that corresponds to the linearized subscript
  //   \sum_k i_k * \prod_{j=1}^{k-1} s_j
  // by accumulating the running linearized value.
  // Note that `indices` and `allocSizes` are passed in the same order as they
  // appear in load/store operations and memref type declarations.
  ValuePtr linearizeSubscripts(ConversionPatternRewriter &builder, Location loc,
                               ArrayRef<ValuePtr> indices,
                               ArrayRef<ValuePtr> allocSizes) const {
    assert(indices.size() == allocSizes.size() &&
           "mismatching number of indices and allocation sizes");
    assert(!indices.empty() && "cannot linearize a 0-dimensional access");

    ValuePtr linearized = indices.front();
    for (int i = 1, nSizes = allocSizes.size(); i < nSizes; ++i) {
      linearized = builder.create<LLVM::MulOp>(
          loc, this->getIndexType(),
          ArrayRef<ValuePtr>{linearized, allocSizes[i]});
      linearized = builder.create<LLVM::AddOp>(
          loc, this->getIndexType(),
          ArrayRef<ValuePtr>{linearized, indices[i]});
    }
    return linearized;
  }

  // This is a strided getElementPtr variant that linearizes subscripts as:
  //   `base_offset + index_0 * stride_0 + ... + index_n * stride_n`.
  ValuePtr getStridedElementPtr(Location loc, Type elementTypePtr,
                                ValuePtr descriptor, ArrayRef<ValuePtr> indices,
                                ArrayRef<int64_t> strides, int64_t offset,
                                ConversionPatternRewriter &rewriter) const {
    assert(offset != MemRefType::getDynamicStrideOrOffset() &&
           "Dynamic opset is not supported");

    ValuePtr offsetValue = this->createIndexConstant(rewriter, loc, offset);
    for (int i = 0, e = indices.size(); i < e; ++i) {
      assert(strides[i] != MemRefType::getDynamicStrideOrOffset() &&
             "Dynamic strides are not supported");

      ValuePtr stride = this->createIndexConstant(rewriter, loc, strides[i]);
      ValuePtr additionalOffset =
          rewriter.create<LLVM::MulOp>(loc, indices[i], stride);
      offsetValue =
          rewriter.create<LLVM::AddOp>(loc, offsetValue, additionalOffset);
    }
    return rewriter.create<LLVM::GEPOp>(loc, elementTypePtr, descriptor, offsetValue);
  }

  ValuePtr getDataPtr(Location loc, MemRefType type, ValuePtr memRefDesc,
                      ArrayRef<ValuePtr> indices,
                      ConversionPatternRewriter &rewriter,
                      llvm::Module &module) const {
    auto ptrType = memRefDesc->getType().cast<LLVM::LLVMType>();
    int64_t offset;
    SmallVector<int64_t, 4> strides;
    auto successStrides = getStridesAndOffset(type, strides, offset);
    assert(succeeded(successStrides) && "unexpected non-strided memref");
    (void)successStrides;
    return getStridedElementPtr(loc, ptrType, memRefDesc, indices, strides,
                                offset, rewriter);
  }
};

// TODO: Refactor?
// Load operation is lowered to obtaining a pointer to the indexed element
// and loading it.
struct LoadOpLowering : public LoadStoreOpLowering<LoadOp> {
  using Base::Base;

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<ValuePtr> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loadOp = cast<LoadOp>(op);
    OperandAdaptor<LoadOp> transformed(operands);
    auto type = loadOp.getMemRefType();

    ValuePtr dataPtr = getDataPtr(op->getLoc(), type, transformed.memref(),
                                  transformed.indices(), rewriter, getModule());
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, dataPtr);
    return matchSuccess();
  }
};

// TODO: Refactor?
// Store operation is lowered to obtaining a pointer to the indexed element,
// and storing the given value to it.
struct StoreOpLowering : public LoadStoreOpLowering<StoreOp> {
  using Base::Base;

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<ValuePtr> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = cast<StoreOp>(op).getMemRefType();
    OperandAdaptor<StoreOp> transformed(operands);

    ValuePtr dataPtr = getDataPtr(op->getLoc(), type, transformed.memref(),
                                  transformed.indices(), rewriter, getModule());
    rewriter.replaceOpWithNewOp<LLVM::StoreOp>(op, transformed.value(),
                                               dataPtr);
    return matchSuccess();
  }
};

struct FuncOpConversion : public LLVMLegalizationPattern<FuncOp> {
  using LLVMLegalizationPattern<FuncOp>::LLVMLegalizationPattern;

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<ValuePtr> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto funcOp = cast<FuncOp>(op);
    FunctionType type = funcOp.getType();

    // Store the positions of memref-typed arguments so that we can emit loads
    // from them to follow the calling convention.
    SmallVector<unsigned, 4> promotedArgIndices;
    promotedArgIndices.reserve(type.getNumInputs());
    for (auto en : llvm::enumerate(type.getInputs())) {
      if (en.value().isa<MemRefType>() || en.value().isa<UnrankedMemRefType>())
        promotedArgIndices.push_back(en.index());
    }

    // Convert the original function arguments. Struct arguments are promoted to
    // pointer to struct arguments to allow calling external functions with
    // various ABIs (e.g. compiled from C/C++ on platform X).
    auto varargsAttr = funcOp.getAttrOfType<BoolAttr>("std.varargs");
    TypeConverter::SignatureConversion result(funcOp.getNumArguments());
    auto llvmType = lowering.convertFunctionSignature(
        funcOp.getType(), varargsAttr && varargsAttr.getValue(), result);

    // Only retain those attributes that are not constructed by build.
    SmallVector<NamedAttribute, 4> attributes;
    for (const auto &attr : funcOp.getAttrs()) {
      if (attr.first.is(SymbolTable::getSymbolAttrName()) ||
          attr.first.is(impl::getTypeAttrName()) ||
          attr.first.is("std.varargs"))
        continue;
      attributes.push_back(attr);
    }

    // Create an LLVM function, use external linkage by default until MLIR
    // functions have linkage.
    auto newFuncOp = rewriter.create<LLVM::LLVMFuncOp>(
        op->getLoc(), funcOp.getName(), llvmType, LLVM::Linkage::External,
        attributes);
    rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());

    // Tell the rewriter to convert the region signature.
    rewriter.applySignatureConversion(&newFuncOp.getBody(), result);

    // Insert loads from memref descriptor pointers in function bodies.
    if (!newFuncOp.getBody().empty()) {
      Block *firstBlock = &newFuncOp.getBody().front();
      rewriter.setInsertionPoint(firstBlock, firstBlock->begin());
      for (unsigned idx : promotedArgIndices) {
        BlockArgumentPtr arg = firstBlock->getArgument(idx);
        ValuePtr loaded = rewriter.create<LLVM::LoadOp>(funcOp.getLoc(), arg);
        rewriter.replaceUsesOfBlockArgument(arg, loaded);
      }
    }

    rewriter.eraseOp(op);
    return matchSuccess();
  }
};

// An `alloc` is converted into a definition of a memref descriptor value and
// a call to `malloc` to allocate the underlying data buffer.  The memref
// descriptor is of the LLVM structure type where:
//   1. the first element is a pointer to the allocated (typed) data buffer,
//   2. the second element is a pointer to the (typed) payload, aligned to the
//      specified alignment,
//   3. the remaining elements serve to store all the sizes and strides of the
//      memref using LLVM-converted `index` type.
//
// Alignment is obtained by allocating `alignment - 1` more bytes than requested
// and shifting the aligned pointer relative to the allocated memory. If
// alignment is unspecified, the two pointers are equal.
struct AllocOpLowering : public LLVMLegalizationPattern<AllocOp> {
  using LLVMLegalizationPattern<AllocOp>::LLVMLegalizationPattern;

  AllocOpLowering(LLVM::LLVMDialect &dialect_, LLVMTypeConverter &converter,
                  bool useAlloca = false)
      : LLVMLegalizationPattern<AllocOp>(dialect_, converter),
        useAlloca(useAlloca) {}

  PatternMatchResult match(Operation *op) const override {
    MemRefType type = cast<AllocOp>(op).getType();
    if (isSupportedMemRefType(type))
      return matchSuccess();

    return matchFailure();
  }

  void rewrite(Operation *op, ArrayRef<ValuePtr> operands,
               ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto allocOp = cast<AllocOp>(op);
    MemRefType type = allocOp.getType();

    // Get actual sizes of the memref as values: static sizes are constant
    // values and dynamic sizes are passed to 'alloc' as operands.  In case of
    // zero-dimensional memref, assume a scalar (size 1).
    SmallVector<ValuePtr, 4> sizes;
    sizes.reserve(type.getRank());
    for (int64_t s : type.getShape())
      sizes.push_back(createIndexConstant(rewriter, loc, s));
    if (sizes.empty())
      sizes.push_back(createIndexConstant(rewriter, loc, 1));

    // Compute the total number of memref elements.
    ValuePtr cumulativeSize = sizes.front();
    for (unsigned i = 1, e = sizes.size(); i < e; ++i)
      cumulativeSize = rewriter.create<LLVM::MulOp>(
          loc, getIndexType(), ArrayRef<ValuePtr>{cumulativeSize, sizes[i]});

    // Compute the size of an individual element. This emits the MLIR equivalent
    // of the following sizeof(...) implementation in LLVM IR:
    //   %0 = getelementptr %elementType* null, %indexType 1
    //   %1 = ptrtoint %elementType* %0 to %indexType
    // which is a common pattern of getting the size of a type in bytes.
    auto elementType = type.getElementType();
    auto convertedPtrType =
        lowering.convertType(elementType).cast<LLVM::LLVMType>().getPointerTo();
    auto nullPtr = rewriter.create<LLVM::NullOp>(loc, convertedPtrType);
    auto one = createIndexConstant(rewriter, loc, 1);
    auto gep = rewriter.create<LLVM::GEPOp>(loc, convertedPtrType,
                                            ArrayRef<ValuePtr>{nullPtr, one});
    auto elementSize =
        rewriter.create<LLVM::PtrToIntOp>(loc, getIndexType(), gep);
    cumulativeSize = rewriter.create<LLVM::MulOp>(
        loc, getIndexType(), ArrayRef<ValuePtr>{cumulativeSize, elementSize});

    // Allocate the underlying buffer and store a pointer to it in the MemRef
    // descriptor.
    ValuePtr allocated = nullptr;

    if (useAlloca) {
      allocated = rewriter.create<LLVM::AllocaOp>(
          loc, getVoidPtrType(), cumulativeSize, /*alignment=*/0);
    } else {
      // Insert the `malloc` declaration if it is not already present.
      auto module = op->getParentOfType<ModuleOp>();
      auto mallocFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("malloc");
      if (!mallocFunc) {
        OpBuilder moduleBuilder(
            op->getParentOfType<ModuleOp>().getBodyRegion());
        mallocFunc = moduleBuilder.create<LLVM::LLVMFuncOp>(
            rewriter.getUnknownLoc(), "malloc",
            LLVM::LLVMType::getFunctionTy(getVoidPtrType(), getIndexType(),
                                          /*isVarArg=*/false));
      }
      allocated = rewriter
                      .create<LLVM::CallOp>(
                          loc, getVoidPtrType(),
                          rewriter.getSymbolRefAttr(mallocFunc), cumulativeSize)
                      .getResult(0);
    }

    auto structElementType = lowering.convertType(elementType);
    auto elementPtrType = structElementType.cast<LLVM::LLVMType>().getPointerTo(
        type.getMemorySpace());
    ValuePtr bitcastAllocated = rewriter.create<LLVM::BitcastOp>(
        loc, elementPtrType, ArrayRef<ValuePtr>(allocated));

    int64_t offset;
    SmallVector<int64_t, 4> strides;
    auto successStrides = getStridesAndOffset(type, strides, offset);
    assert(succeeded(successStrides) && "unexpected non-strided memref");
    (void)successStrides;
    assert(offset != MemRefType::getDynamicStrideOrOffset() &&
           "unexpected dynamic offset");

    // 0-D memref corner case: they have size 1 ...
    assert(((type.getRank() == 0 && strides.empty() && sizes.size() == 1) ||
            (strides.size() == sizes.size())) &&
           "unexpected number of strides");

    // Return the final value of the descriptor.
    rewriter.replaceOp(op, bitcastAllocated);
  }

  bool useAlloca;
};

// A `dealloc` is converted into a call to `free` on the underlying data buffer.
// The memref descriptor being an SSA value, there is no need to clean it up
// in any way.
struct DeallocOpLowering : public LLVMLegalizationPattern<DeallocOp> {
  using LLVMLegalizationPattern<DeallocOp>::LLVMLegalizationPattern;

  DeallocOpLowering(LLVM::LLVMDialect &dialect_, LLVMTypeConverter &converter,
                    bool useAlloca = false)
      : LLVMLegalizationPattern<DeallocOp>(dialect_, converter),
        useAlloca(useAlloca) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<ValuePtr> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (useAlloca)
      return rewriter.eraseOp(op), matchSuccess();

    assert(operands.size() == 1 && "dealloc takes one operand");
    OperandAdaptor<DeallocOp> transformed(operands);

    // Insert the `free` declaration if it is not already present.
    auto freeFunc =
        op->getParentOfType<ModuleOp>().lookupSymbol<LLVM::LLVMFuncOp>("free");
    if (!freeFunc) {
      OpBuilder moduleBuilder(op->getParentOfType<ModuleOp>().getBodyRegion());
      freeFunc = moduleBuilder.create<LLVM::LLVMFuncOp>(
          rewriter.getUnknownLoc(), "free",
          LLVM::LLVMType::getFunctionTy(getVoidType(), getVoidPtrType(),
                                        /*isVarArg=*/false));
    }

    ValuePtr casted = rewriter.create<LLVM::BitcastOp>(
        op->getLoc(), getVoidPtrType(), transformed.memref());
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, ArrayRef<Type>(), rewriter.getSymbolRefAttr(freeFunc), casted);
    return matchSuccess();
  }

  bool useAlloca;
};

} // namespace

void mlir::populateStdToLLVMMemoryConvPattersBarePtrMemRef(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns) {
  // clang-format off
  patterns.insert<
      // DimOpLowering,
      FuncOpConversion,
      LoadOpLowering,
      //MemRefCastOpLowering,
      StoreOpLowering
      //SubViewOpLowering,
      /*ViewOpLowering*/>(*converter.getDialect(), converter);
  patterns.insert<
      AllocOpLowering,
      DeallocOpLowering>(
        *converter.getDialect(), converter, clUseAlloca.getValue());
  // clang-format on
}

void mlir::populateStdToLLVMConvPatternsBarePtrMemRef(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns) {
  populateStdToLLVMNonMemoryConversionPatterns(converter, patterns);
  populateStdToLLVMMemoryConvPattersBarePtrMemRef(converter, patterns);
}
