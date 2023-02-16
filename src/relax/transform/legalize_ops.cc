/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file legalize.cc
 * \brief Converts an expr to another expr. This pass can be used to transform an op based on its
 * shape, dtype or layout to another op or a sequence of ops.
 */

#include <tvm/relax/analysis.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/relax/transform.h>

#include "utils.h"

namespace tvm {
namespace relax {

using CustomizeMap = Optional<Map<String, PackedFunc>>;

// Call registered FTVMLegalize of an op, returns the legalized expression
class LegalizeMutator : public ExprMutator {
 public:
  explicit LegalizeMutator(const IRModule& mod, const CustomizeMap& cmap)
      : ExprMutator(mod), mod_(std::move(mod)), cmap_(std::move(cmap)) {}

  Expr VisitExpr_(const CallNode* call) final {
    Call visited_call = Downcast<Call>(this->VisitExprPostOrder_(call));
    static const auto& legalize_map = Op::GetAttrMap<FRelaxLegalize>("FRelaxLegalize");
    static const Op& call_tir_op = Op::Get("relax.call_tir");

    if (auto* op_node = visited_call->op.as<OpNode>()) {
      auto op = GetRef<Op>(op_node);
      FRelaxLegalize flegalize;
      bool has_legalize = false, know_all_shape = true;

      // Check if it has legalization registered.
      // Get flegalize from cmap or attribute. Priority: customize > builtin.
      if (cmap_.defined() && cmap_.value().count(op->name)) {
        flegalize = cmap_.value()[op->name];
        has_legalize = true;
      }
      if (!legalize_map.count(op)) {
        if (op != call_tir_op) {
          LOG(WARNING) << "No legalization func for " << op->name << " is found.";
        }
      } else if (!has_legalize) {
        flegalize = legalize_map[op];
        has_legalize = true;
      }

      // Check if all shape values are known
      for (const auto& arg : visited_call->args) {
        know_all_shape |= KnowAllShapeValues(GetStructInfo(arg));
      }
      know_all_shape |= KnowAllShapeValues(GetStructInfo(visited_call));

      if (has_legalize && know_all_shape) {
        return flegalize(this->builder_, visited_call);
      }
    }

    return visited_call;
  }

  IRModule Transform() {
    for (const auto& [gv, func] : mod_->functions) {
      if (func->IsInstance<FunctionNode>()) {
        auto updated_func = RemoveAllUnused(Downcast<Function>(this->VisitExpr(func)));
        builder_->UpdateFunction(gv, Downcast<BaseFunc>(updated_func));
      }
    }
    return builder_->GetContextIRModule();
  }

 private:
  IRModule mod_;
  CustomizeMap cmap_;
};

namespace transform {

Pass LegalizeOps(CustomizeMap cmap) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule mod, PassContext pc) { return LegalizeMutator(mod, cmap).Transform(); };
  return CreateModulePass(/*pass_function=*/pass_func,
                          /*opt_level=*/0,
                          /*pass_name=*/"LegalizeOps",
                          /*required=*/{});
}

TVM_REGISTER_GLOBAL("relax.transform.LegalizeOps").set_body_typed(LegalizeOps);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
