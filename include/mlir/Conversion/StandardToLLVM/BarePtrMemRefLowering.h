//===- ConvertStandardToLLVM.h - Convert to the LLVM dialect ----*- C++ -*-===//
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
// Provides a dialect conversion targeting the LLVM IR dialect.  By default, it
// converts Standard ops and types and provides hooks for dialect-specific
// extensions to the conversion.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_STANDARDTOLLVM_BAREPTRMEMREFLOWERING_H
#define MLIR_CONVERSION_STANDARDTOLLVM_BAREPTRMEMREFLOWERING_H

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"

namespace mlir {

// TODO
/// Conversion from types in the Standard dialect to the LLVM IR dialect.
class BarePtrMemRefTypeConverter : public LLVMTypeConverter
{
public:
  BarePtrMemRefTypeConverter(MLIRContext *ctx) : LLVMTypeConverter(ctx) {};
};

} // namespace mlir

#endif // MLIR_CONVERSION_STANDARDTOLLVM_BAREPTRMEMREFLOWERING_H
