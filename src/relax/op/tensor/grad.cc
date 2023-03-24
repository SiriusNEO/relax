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
 * \file grad.cc
 * \brief Operators to implement operaor gradients.
 */

#include "grad.h"

#include <utility>

namespace tvm {
namespace relax {

Expr nll_loss_backward(Expr output_grad, Expr predictions, Expr targets, Optional<Expr> weights,
                       String reduction, int ignore_index) {
  ObjectPtr<NLLLossAttrs> attrs = make_object<NLLLossAttrs>();

  attrs->reduction = reduction;
  attrs->ignore_index = ignore_index;

  static const Op& op = Op::Get("relax.grad.nll_loss_backward");
  if (weights.defined()) {
    return Call(op,
                {std::move(output_grad), std::move(predictions), std::move(targets),
                 std::move(weights.value())},
                Attrs{attrs}, {});
  } else {
    return Call(op, {std::move(output_grad), std::move(predictions), std::move(targets)},
                Attrs{attrs}, {});
  }
}

TVM_REGISTER_GLOBAL("relax.op.grad.nll_loss_backward").set_body_typed(nll_loss_backward);

StructInfo InferStructInfoNLLLossBackward(const Call& call, const BlockBuilder& ctx) {
  return GetStructInfo(call->args[1]);
}

TVM_REGISTER_OP("relax.grad.nll_loss_backward")
    .set_attrs_type<NLLLossAttrs>()
    .set_num_inputs(4)
    .add_argument("output_grad", "Tensor", "The output gradient.")
    .add_argument("predictions", "Tensor", "The prediction tensor.")
    .add_argument("targets", "Tensor", "The target tensor.")
    .add_argument("weights", "Optional<Tensor>", "The weight of each target values.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoNLLLossBackward);

Expr max_pool2d_backward(Expr output_grad, Expr data, Array<IntImm> pool_size,
                         Array<IntImm> strides, Array<IntImm> padding, Array<IntImm> dilation,
                         bool ceil_mode, String layout, Optional<String> out_layout) {
  auto attrs = make_object<Pool2DAttrs>();
  attrs->pool_size = std::move(pool_size);
  attrs->strides = ConvertIntImmToInt64(strides);
  attrs->padding = ConvertIntImmToInt64(padding);
  attrs->dilation = ConvertIntImmToInt64(dilation);
  attrs->ceil_mode = ceil_mode;
  attrs->layout = layout;
  attrs->out_layout = out_layout.value_or(layout);
  static const Op& op = Op::Get("relax.grad.max_pool2d_backward");
  return Call(op, {std::move(output_grad), std::move(data)}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.grad.max_pool2d_backward").set_body_typed(max_pool2d_backward);

StructInfo InferStructInfoMaxPool2DBackward(const Call& call, const BlockBuilder& ctx) {
  return GetStructInfo(call->args[1]);
}

TVM_REGISTER_OP("relax.grad.max_pool2d_backward")
    .set_num_inputs(2)
    .add_argument("output_grad", "Tensor", "The output gradient.")
    .add_argument("data", "Tensor", "The input tensor")
    .set_attrs_type<Pool2DAttrs>()
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoMaxPool2DBackward);

Expr avg_pool2d_backward(Expr output_grad, Expr data, Array<IntImm> pool_size,
                         Array<IntImm> strides, Array<IntImm> padding, Array<IntImm> dilation,
                         bool ceil_mode, String layout, Optional<String> out_layout) {
  auto attrs = make_object<Pool2DAttrs>();
  attrs->pool_size = std::move(pool_size);
  attrs->strides = ConvertIntImmToInt64(strides);
  attrs->padding = ConvertIntImmToInt64(padding);
  attrs->dilation = ConvertIntImmToInt64(dilation);
  attrs->ceil_mode = ceil_mode;
  attrs->layout = layout;
  attrs->out_layout = out_layout.value_or(layout);
  static const Op& op = Op::Get("relax.grad.avg_pool2d_backward");
  return Call(op, {std::move(output_grad), std::move(data)}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.grad.avg_pool2d_backward").set_body_typed(avg_pool2d_backward);

StructInfo InferStructInfoAvgPool2DBackward(const Call& call, const BlockBuilder& ctx) {
  return GetStructInfo(call->args[1]);
}

TVM_REGISTER_OP("relax.grad.avg_pool2d_backward")
    .set_num_inputs(2)
    .add_argument("output_grad", "Tensor", "The output gradient.")
    .add_argument("data", "Tensor", "The input tensor")
    .set_attrs_type<Pool2DAttrs>()
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoAvgPool2DBackward);

/* relax.take_backward */
TVM_REGISTER_NODE_TYPE(TakeAttrs);

Expr take_backward(Expr output_grad, Expr x, Expr indices, Optional<Integer> axis) {
  ObjectPtr<TakeAttrs> attrs = make_object<TakeAttrs>();
  attrs->axis = std::move(axis);

  static const Op& op = Op::Get("relax.grad.take_backward");
  return Call(op, {std::move(output_grad), std::move(x), std::move(indices)}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.grad.take_backward").set_body_typed(take_backward);

StructInfo InferStructInfoTakeBackward(const Call& call, const BlockBuilder& ctx) {
  return GetStructInfo(call->args[1]);
}

TVM_REGISTER_OP("relax.grad.take_backward")
    .set_attrs_type<TakeAttrs>()
    .set_num_inputs(3)
    .add_argument("output_grad", "Tensor", "The output gradient.")
    .add_argument("x", "Tensor", "The source tensor.")
    .add_argument("indices", "Tensor", "The indices of the values to extract.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoTakeBackward);

}  // namespace relax
}  // namespace tvm
