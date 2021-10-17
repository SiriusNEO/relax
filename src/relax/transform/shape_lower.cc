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
 * \file src/relax/transform/shape_lower.cc
 * \brief
 */
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/type.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include "../../printer/text_printer.h"

namespace tvm {
namespace relax {

class ShapeLowerMutator : public ExprMutator {
 public:
  static DataType ShapeDType() { return DataType::Int(32); };

  explicit ShapeLowerMutator(IRModule mod) { mod_ = mod; }

  IRModule Lower() {
    ret_mod_ = IRModule();
    for (auto& p : mod_->functions) {
      if (!p.second->IsInstance<FunctionNode>()) {
        continue;
      }
      // prepare mapping and heap var
      expr2slot_ = PrepareExpr2Slot(Downcast<Function>(p.second));
      // LOG(INFO) << "mapping: " << expr2slot_;
      heap_size_ = IntImm(ShapeDType(), expr2slot_.size());
      DynTensorType heap_type(1, ShapeDType());
      shape_heap_ = Var("shape_heap", ShapeExpr({heap_size_}), heap_type);

      // mutate
      Expr new_func = this->Mutate(p.second);
      ret_mod_->Add(p.first, Downcast<BaseFunc>(new_func));
    }
    return ret_mod_;
  }

  void VisitMatchShape(const MatchShape& binding) override {
    Expr value = binding->value;
    Array<PrimExpr> pattern = binding->pattern;
    Array<PrimExpr> indices;
    for (size_t i = 0; i < pattern.size(); ++i) {
      IntImm idx = expr2slot_.at(pattern[i]);
      indices.push_back(idx);
    }
    builder_->Emit(Call(ExternFunc("decode_shape"), {value, shape_heap_, ShapeExpr(indices)}), "_");
  }

  Expr VisitExpr_(const ShapeExprNode* node) override {
    tir::PrimFunc func = CalculateShape(GetRef<ShapeExpr>(node));
    GlobalVar shape_func_var(name_table_->GetUniqueName("shape_func"));
    // TODO make sure shape_heap doesnt get redefined by local funcs?
    builder_->Emit(Call(shape_func_var, {shape_heap_}), "_");
    ret_mod_->Add(shape_func_var, func);

    // construct shape
    Array<PrimExpr> indices;
    for (PrimExpr e : node->values) {
      indices.push_back(expr2slot_.at(e));
    }
    return builder_->Emit(Call(ExternFunc("construct_shape"), {shape_heap_, ShapeExpr(indices)}),
                          "sh");
  }

  Expr VisitExpr_(const FunctionNode* node) override {
    Array<Var> params;
    for (Var param : node->params) {
      params.push_back(Downcast<Var>(this->Mutate(param)));
    }
    Type ret_type = this->VisitType(node->ret_type);

    builder_->BeginBindingBlock();
    builder_->Emit(VarBinding(
        shape_heap_, Call(ExternFunc("relax.alloc_shape_heap"), {ShapeExpr({heap_size_})})));

    Expr new_body = this->Mutate(node->body);

    Array<BindingBlock> blocks;

    if (const SeqExprNode* seq = new_body.as<SeqExprNode>()) {
      blocks.push_back(builder_->EndBlock());
      blocks.insert(blocks.end(), seq->blocks.begin(), seq->blocks.end());
      builder_->BeginBindingBlock();
      new_body = seq->body;
    }

    builder_->Emit(Call(ExternFunc("relax.free_shape_heap"), {shape_heap_}), "_");
    blocks.push_back(builder_->EndBlock());
    new_body = SeqExpr(blocks, new_body);

    return Function(node->name, params, new_body, ret_type);
  }

  tir::PrimFunc CalculateShape(ShapeExpr s) {
    // TODO(ziheng): avoid generating shape func for known value
    tir::Var heap("heap", DataType::Handle());
    Array<PrimExpr> buffer_shape{heap_size_};
    tir::Buffer buffer = tir::decl_buffer(buffer_shape, ShapeDType(), "H");
    Map<tir::Var, tir::Buffer> buffer_map;
    buffer_map.Set(heap, buffer);

    Array<tir::Stmt> seq;
    for (PrimExpr e : s->values) {
      Map<tir::Var, PrimExpr> var_mapping = BuildVarMapping(e, buffer);
      PrimExpr value = tir::Substitute(e, var_mapping);
      IntImm idx = expr2slot_.at(e);
      seq.push_back(tir::Store(buffer->data, value, idx, tir::const_true()));
    }
    tir::Stmt body = tir::SeqStmt(seq);
    Array<tir::Var> params{heap};
    Type ret_type = VoidType();
    return tir::PrimFunc(params, body, ret_type, buffer_map);
  }

  Map<tir::Var, PrimExpr> BuildVarMapping(PrimExpr expr, tir::Buffer buffer) {
    Map<tir::Var, PrimExpr> ret;
    auto func = [&](const ObjectRef& e) {
      if (e->IsInstance<tir::VarNode>()) {
        PrimExpr prim_e = Downcast<PrimExpr>(e);
        tir::Load load(ShapeDType(), buffer->data, expr2slot_.at(prim_e), tir::const_true());
        ret.Set(Downcast<tir::Var>(e), load);
      }
    };
    tir::PostOrderVisit(expr, func);
    return ret;
  }

  Map<PrimExpr, IntImm> PrepareExpr2Slot(Function expr) const {
    int cnt = 0;
    Map<PrimExpr, IntImm> ret;
    auto func = [&](const Expr& e) {
      if (e->IsInstance<ShapeExprNode>()) {
        ShapeExpr shape = Downcast<ShapeExpr>(e);
        for (auto prim_e : shape->values) {
          if (ret.count(prim_e) == 0) {
            IntImm idx(ShapeDType(), cnt++);
            ret.Set(prim_e, idx);
          }
        }
      }
    };
    PostOrderVisit(expr, func);
    return ret;
  }

 private:
  IRModule mod_;
  IRModule ret_mod_;
  int shape_func_counter_{0};

  // function-wise members
  IntImm heap_size_;
  Var shape_heap_;
  Map<PrimExpr, IntImm> expr2slot_;
};

TVM_REGISTER_GLOBAL("relax.transform.shape_lower").set_body_typed([](IRModule mod) {
  return ShapeLowerMutator(mod).Lower();
});

}  // namespace relax
}  // namespace tvm
