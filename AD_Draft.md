# Reverse Mode Automatic Differentiation in Relax

*Notice: The syntax of the Relax language used in this document  may be outdate since current parser is not stable.*

Implementation Details of the experimental `Gradient` Pass in Relax.

Updated: 2023/1/4

# 1. Code and API

The pass differentiates the input relax function and return a fresh new differentiated function, with the name `<original_name>_adjoint`. It returns both the original return value and the needed adjoints in the form of Tuple. See the following example:

**E.g. 1.1:**

```python
@I.ir_module
class Before:
    @R.function
    def main(x: R.Tensor((5, 5), "float32"),
            y:  R.Tensor((5, 5), "float32")):
        with R.dataflow():
            lv0 = R.add(x, y)
            gv0 = R.sum(lv0)
            R.output(gv0)
        return gv0

@I.ir_module
class Module:
    @R.function
    def main(x: Tensor((5, 5), "float32"), y: Tensor((5, 5), "float32")) -> 
            Tuple(
                Tensor(None, "float32", ndim = 0), 
                Tuple(
                    Tensor(None, "float32", ndim = 2),
                    Tensor(None, "float32", ndim = 2)
                )
            ):
        with R.dataflow():
            lv0: Tensor((5, 5), "float32")         =  R.add(x, y)
            gv0: Tensor((), "float32")             =  R.sum(lv0)
            gv0_adjoint: Tensor((), "float32")     =  R.ones((), dtype="float32")
            lv0_adjoint: Tensor((5, 5), "float32") =  R.broadcast_to(gv0_adjoint, (5, 5))
            x_adjoint: Tensor((5, 5), "float32")   =  R.collapse_sum_to(lv0_adjoint, (5, 5))
            y_adjoint: Tensor((5, 5), "float32")   =  R.collapse_sum_to(lv0_adjoint, (5, 5))
            R.output(gv0, x_adjoint, y_adjoint)
        return (gv0, (x_adjoint, y_adjoint))
```



## Code Organization

The pass is in `src/relax/transform/gradient.cc` , with the Python API in `relax/transform/transform.py`.

## Requirements

Currently AD can only work in:

- A relax function with a **single dataflow-block**.
- The function's body should be `SeqExpr` 
- The return value should be a `scalar`.

Currently considered AST nodes:

- [x] Primitive Call
- [x] Assignment
- [x] Tuple-aware (Tuple, TupleGetItem)
- [ ] Constant
- [ ] Match Shape
- [ ] Call TIR

AD uses the following operators:

- `relax.zeros`
- `relax.ones`
- `relax.add`

The gradient should be registered as an attribute of the operator, which `attr_key` is `FPrimalGradient`. The type of the gradient is

```c++
FPrimalGradient = runtime::TypedPackedFunc<tvm::Array<Expr>(const Expr& orig_call, const Expr& output_grad)>;
```

It returns an Array which has the same size with the arguments of the call. For each argument of the original call, the Array stores its partial in the corresponding position.

Before the Module is fed into this pass, it should be **normalized**.

## API

```c++
TVM_DLL Pass Gradient(GlobalVar global_var, Optional<Array<Var>> require_grads);
```

- `global_var` is the GlobalVar of the specific function.
- `require_grads` specifies the relax variables whose adjoints are needed. Must be parameters of the given function. If it is not specified, adjoints of all arguments would be computed.

- Returns a new module with the original function and the new differentiated function.



# 2. Simplest Case: AD with Only Primitive Operators

## Goal

Our goal is to calculate the adjoints for inputs (by default or specified by `require_grads`) in respect of **target** (the return value of the original function). For instance, in E.g. 1.1, the adjoint of x can be defined as
$$
\text{x_adjoint} = \frac{\partial\text{gv0}}{\partial \text{x}}
$$


## Starting by Calculating an Example

Let us consider E.g. 1.1 again. In this example, the target gv0 is scalar, which satisfies our requirements. We set the adjoint of target at first preparing for propagation.

```python
gv0_adjoint: Tensor((), "float32")     =  R.ones((), dtype="float32") # gv0_adjoint = 1
```

Because it is a reverse-mode AD, we start from the last binding, which is

```
gv0 = relax.sum(lv0)
```

And we search the gradient corresponding to `relax.sum` by the attribute `FPrimalGradient`. For simplicity, here we can regard the gradient of `sum` is just

```python
@register_gradient("relax.sum")
def sum_grad(orig: Call, grad: Var):
    """Gradient of sum."""
    return [broadcast_to(grad, orig.args[0].shape)]
```

Then we call `sum_grad` by passing orignal call and `gv0_adjoint` to it.

```
lv0_adjoint += sum_grad(call, gv0_adjoint)[0]
```

Since `lv0_adjoint` has not been defined here, `+=` can be replaced with `=` in implementation. Then we get the `lv_adjoint=ones((5, 5))` .

Then we can go to the next binding

```
lv0 = relax.add(x, y)
```

The gradient of `relax.add` is also very clear

```python
@register_gradient("relax.add")
def add_grad(orig: Call, grad: Var):
    """Returns [grad, grad]."""
    return [
        collapse_sum_to(grad, orig.args[0].shape), 
        collapse_sum_to(grad, orig.args[1].shape)
      ]
```

Note that it is necessary to use a `collapse_sum_to` since `relax.add` is a broadcast operator. We can regard `collapse_sum_to` as the opposite of broadcasting. Then we have

```
x_adjoint += add_grad(call, lv0_adjoint)[0]
y_adjoint += add_grad(call, lv0_adjoint)[0]
```

Finally AD ends with the return (Suppose we do not specify `require_grads`)

```
return (gv0, (x_adjoint, y_adjoint))
```

## Expr and Var

In Relax, like many other programming languages, variables and expressions (or values) are separated. The correct way we implement our AD should be: Allocate Vars for every adjoints, calculate the value of adjoints in Expr, using these adjoints in the form of Vars and finally bind & emit every pair of Var and Expr.

To show the point, again consider E.g. 1.1. Recall that we get `x_adjoint` and `y_adjoint` by these calculations.

```python
x_adjoint: Tensor((5, 5), "float32") = R.collapse_sum_to(lv0_adjoint, (5, 5))
y_adjoint: Tensor((5, 5), "float32") = R.collapse_sum_to(lv0_adjoint, (5, 5))
```

If `lv0_adjoint` is not a relax Var but a relax Expr, they can be further expanded:

```python
x_adjoint: Tensor((5, 5), "float32") = R.collapse_sum_to(R.broadcast_to(gv0_adjoint, (5, 5)), (5, 5))
y_adjoint: Tensor((5, 5), "float32") = R.collapse_sum_to(R.broadcast_to(gv0_adjoint, (5, 5)), (5, 5))
```

And also suppose `gv0_adjoint` is a Expr

```python
x_adjoint: Tensor((5, 5), "float32") = R.collapse_sum_to(R.broadcast_to(R.ones((), dtype="float32"),(5, 5)), (5, 5))
y_adjoint: Tensor((5, 5), "float32") = R.collapse_sum_to(R.broadcast_to(R.ones((), dtype="float32"), (5, 5)), (5, 5))
```

After another normalizing, many redundant relax variables will be created and it is catastrophic. Therefore it is important for us to use relax Var to store our intermediate calculation results. The following is the simple and clean version we want.

```python
gv0_adjoint: Tensor((), "float32")     =  R.ones((), dtype="float32")
lv0_adjoint: Tensor((5, 5), "float32") =  R.broadcast_to(gv0_adjoint, (5, 5))
x_adjoint: Tensor((5, 5), "float32")   =  R.collapse_sum_to(lv0_adjoint, (5, 5))
y_adjoint: Tensor((5, 5), "float32")   =  R.collapse_sum_to(lv0_adjoint, (5, 5))
```

Hence in practice, we need two Maps: one for Var and another for Expr.

```c++
// var to its adjoints var
Map<Var, Var> adjoint_var_map_;
// var to its adjoint expr
Map<Var, Expr> adjoint_expr_map_;
```

Indeed there are two styles of implementations in respect of `adjoint_expr_map_`. Another way is using a `Map<Var, Array<Expr>> adjoint_expr_map_;` , with the idea that store the partials first and add them together finally. Here we adopt the first way, `Map<Var, Expr>`. And each time doing update for the adjoint of variable v, we can just use the logic like this (Pseudo Code)

```c++
adjoint_expr_map_.Set(v, adjoint_expr_map_.Get(v) + increment)
```
As stated before, if it is the first time we update to v, `+=` becomes `=` :

```
adjoint_expr_map_.Set(v, increment)
```



## About Irrelevant Parts

**E.g. 1.2:**

```python
@I.ir_module
class Before:
    @R.function
    def main(x: Tensor((5, 5), "float32"),
            y: Tensor((5, 5), "float32")):
        with R.dataflow():
            lv0 = relax.add(x, y)
            lv1 = relax.sub(x, y)
            lv2 = relax.sum(lv1)
            gv0 = relax.sum(lv0)
            R.output(gv0)
        return gv0
```

In this example, `lv1`, `lv2` has no contributions to `gv0`. So mathematically
$$
\frac{\partial\text{gv0}}{\partial \text{lv1}} = \frac{\partial\text{gv0}}{\partial \text{lv2}} = 0
$$
which means  `lv1_adjoint` and `lv2_adjoint` has no contributions to `x`, `y` and any other parts. So for these irrelevant parts, we can safely just ignore them when doing AD.



## Summarization

To sum up, we can write some pseudo code (in Python-like syntax) to describe this simplest case,

**AD-Version 1.**

```python
for binding in reverse_order:
	adjoint_var = Var()
    adjoint_var_map_[binding->var] = adjoint_var
    
    if binding->var not in adjoint_expr_map_: # irrelevant parts or target
        if binding->var == target:
            adjoint_expr_map_[binding->var] = 1 # target
        else:
            continue # irrelevant parts
            
    adjoint_expr = adjoint_expr_map_[binding->var]
    emit_and_bind(adjoint_var, adjoint_expr) # emit and bind
            
    call = cast<Call>(binding->value) # in this case value can only be call
    gradient = gradient_map[call->op]
    partials = gradient(call, adjoint_var) 
    # AD. Note that we pass the adjoint_var to it instead of adjoint_expr !!!
    
    for i in len(call.args):
        arg = call.args[i]
        partial = partials[i]
        if arg not in adjoint_expr_map_:
            adjoint_expr_map_[arg] = partial # frist update
        else:
            adjoint_expr_map_[arg] += partial

# It is clear that after the for, we have adjoint_expr for each input.
# Then we just need to bind them with adjoint_var and emit them.
```



PS: "assignment" is not a call. But it can be viewed as an "identity" operator. The logic to handle this case is similar. Always be care with the "expr and var" issue. How to check whether a binding is a assignment? Just check whether `binding->value` is a Var.

```python
# a = b
adjoint_expr_map_[b] += adjoint_var_map_[a]
```



We can conclude some interesting and reasonable rules for AD. First, in a horizontal view, focus on a single binding

```
d = func(a, b, c)
```

While the direction original (forward) dataflow is `a, b, c -> d`, AD reverses everything and makes the adjoint dataflow be `d_adjoint->a_adjoint, b_adjoint, c_adjoint`, which is from the **user** `d` and to **uses** `a` `b` `c`.

Moreover, in a vertical view, focus on a single variable across bindings

```
def of a
...
use1 of a
use2 of a
use3 of a
```

In the reverse-mode,

```
use3 of a
use2 of a
use1 of a
...
def of a
```

In each use, `adjoint_expr_map_[a]` is updated. And when we meet the def, `adjoint_expr_map_[a]` must be updated completely and it is used to update other adjoints. If Relax is SSA, we can say that after we meet the def, the live time of `a_adjoint` is ended.

If there is a def but not used, it must have no contribution to target so (in "About Irrelevant Parts") it is reasonable to ignore them.  If there is a use but no def is found, it is a error in original program.



# 3. Tuple-Aware AD

## Intro

In this case we need to consider not only Tensor but also Tuple. Two new expressions are introduced:

- Tuple (definition): `c = (a, b)`
- TupleGetItem `b = a[0]`

In Relax specification, after normalization, we can have nested Tuple definition (`c = ((a, b), (e,))`) but not nested Tuple GetItem (`c = b[1][2] `).

An important fact is that the adjoint of a Tuple has **exactly the same shape (structural info)** with the original Tuple. We can conclude the basic idea as:

**E.g. 3.1:**

```
Before:
c = (a, b)

After:
a_adjoint += c_adjoint[0]
b_adjoint += c_adjoint[0]

Before:
b = a[0]

After:
a_adjoint[0] += b_adjoint
```

while we can 100% ensure `c_adjoint` and `a_adjoint` are two Tuples. But there are many accompanying problems.



## Addition

If this AD is tuple-aware, then everything can be Tuple. For instance, in the above example, we can ensure some of them are Tuples but  `b_adjoint` can also be a Tuple. And every variable we meet can also be bound to a Tuple. 

The first and foremost problem is that in Relax, we can not simply use a call `relax.add` to add two Tuples together. 

**Q:** But how can we indicate whether a var is bound to a Tuple or not? 

**A:** We can check the type or shape of the var although it is complicated. But indeed in our AD we do not need to do this By observation, in our AD-Version 1, the only two places we use AD is the updating:

```python
adjoint_expr_map_[arg] += partial # call
adjoint_expr_map_[b] += adjoint_var_map_[a] # assignment
```

Note that partial is generated by the gradient function, which will not be a variable bound to a Tuple. 

And because `adjoint_var_map_[a]` is created in the AD, of course we know whether the corresponding expr (a.k.a `adjoint_expr_map_[a]`) is Tuple.

Once we can know this, the next thing is we should extract a method to do such generalized addition. Because Tuple can be nested, here we need a recursion. This addition takes two arguments but it is not symmetrical.

```python
def Addition(base: Expr, increment: Expr):
    if increment is Tuple:
        ret = []
        assert base is Tuple and len(base) == len(increment)
        for i in len(base):
            ret.append(Addition(base[i], increment[i]))
    else:
        return Call("relax.add", base, increment)
```

Note that `increment` can be `Tuple | Var | Expr` . Here `Var` brings some problems. Consider the following example:

**E.g. 3.2:**

```python
@I.ir_module
class Before:
    @R.function
    def main(x: Tuple(Tensor((5, 5), "float32"), Tensor((5, 5), "float32"))):
        with R.dataflow():
            lv0 = x
            ...
            R.output(gv0)
        return gv0
```

In the first binding `lv0 = x`, we use `lv0_adjoint` to update `x_adjoint`. Since it is an assignment, we will do

```python
adjoint_expr_map_[x] = Addition(adjoint_expr_map_[x], lv0_adjoint)
```

This throws error since `lv0_adjoint` is a Var instead of a Tuple Expr.

There are two ways to solve this. The first is checking the type or shape of the var to see whether it is a Tuple. If it is, then we use `TupleGetItem` and recursively do this until the type/shape is not TupleType/Tuple.

```python
def Addition(base: Expr, increment: Expr):
    if increment is Tuple:
        ret = []
        assert base is Tuple and len(base) == len(increment)
        for i in len(base):
            ret.append(Addition(base[i], increment[i]))
    else if increment is Var and type(increment) is TupleType:
        ret = []
        assert base is Tuple
        for i in range(len(type(increment))):
            ret.append(Addition(base[i], TupleGetItem(increment, i)))
    else if increment is TupleGetItem:
        ... 
        # This case must be created in the above case.
        # Do something to handle...
    else:
        return Call("relax.add", base, increment)
```

It is a bit complicated and we will see there is a simpler way. We agree on that when we call `Addition` we always pass `adjoint_expr` instead of `adjoint_var` to its second argument `increment`. This prevents the problem from happening. 

But what about the "expr and var" issue (When we use an adjoint, we always use the adjoint var)? We can maintain a map from `adjoint_expr` to `adjoint_var` named `adjoint_expr_to_var_` and replace the expr with var in the last step of the addition.

```python
def Addition(base: Expr, increment: Expr):
    if increment is Tuple:
        ret = []
        assert base is Tuple and len(base) == len(increment)
        for i in len(base):
            ret.append(Addition(base[i], increment[i]))
        return Tuple(ret)
    else:
        return Call("relax.add", base, ReplaceExprByVar(increment))

    
def ReplaceExprByVar(expr: Expr):
    if expr in adjoint_expr_to_var_:
        return adjoint_expr_to_var_[expr]
   	return expr # can not replace
```

And when we meet an assignment we just pass the `adjoint_expr`

```python
# a = b
adjoint_expr_map_[b] = Addition(adjoint_expr_map_[b], adjointr_expr_map_[a])
# If adjointr_expr_map_[a] is not a Tuple, it will be replaced with 
# adjointr_var_map_[a] finally by ReplaceExprByVar.
```

Then we finally generalize our addition successfully.



## Deal With Tuple Definition

Recall E.g. 3.1, we start by asking a few questions.

```
Before:
c = (a, b)

After:
a_adjoint += c_adjoint[0]
b_adjoint += c_adjoint[0]
```

**Q1:** Where does `c_adjoint` comes from? It is a tuple so where is this tuple created?

**A1**:As mentioned in Section 2, an adjoint is created when it is firstly updated. As for the tensor case, when it is a first update, we replace `+=` by `=` and this skips the process of creation.

But when it comes to Tuple, we can not just do these. `c_adjoint` is updated only when `c` is used and note that except the `Call` which takes Tuple as inputs, the only place a Tuple is used is `TupleGetItem`. So we can complete the above program

```python
Before:
c = (a, b)
d = c[0]
e = c[1]

After:
c_adjoint[1] += e_adjoint # the first update!
c_adjoint[0] += d_adjoint
a_adjoint += c_adjoint[0]
b_adjoint += c_adjoint[0]
```

Then we can see the problem: the update for a Tuple adjoint is a **partial update**. So we must create an empty Tuple skeleton and then do partial update.

```python
def BuildEmptyNestedTupleExpr(shape: Tuple, type: TupleType): # type is for "relax.zeros"
    ret = []
    for i in range(len(shape)):
        if shape[i] is a Tuple:
            ret.append(BuildEmptyNestedTupleExpr(shape[i], type[i]))
        else:
            assert shape[i] is ShapeExpr
            ret.push_back(Call("relax.zeros", shape[i], type[i]))
    return Tuple(ret)
```

**Q2:** What if the tuple is not updated completely after all partial updates?

**A2:** This means these postions in Tuple are not used. According to the "Irrelevant Parts" section, we can just ignore them (Lettiing them be zeros is right!)

After solving this, finally, we can start to propagate the adjoint. We can suppose the `adjoint_expr_map_[c]` is completely updated expr.

```python
if binding->value is Tuple:
    adjoint_var = adjoint_var_map_[binding->var]
    adjoint_expr = adjoint_expr_map_[binding->var]
    
    assert adjoint_expr is Tuple
    assert len(binding->value) == len(adjoint_var)
    
    for i in range(len(binding->value)):
		v = adjoint_expr[i] # Var
        adjoint_expr_map_[v] = Addition(adjoint_expr_map_[v], 
                                        TupleGetItem(adjoint_var, i))
```

Wait. Something seems wrong. We should be very very careful for each assumption. In this code,  we can not ensure `v = adjoint_expr[i]` is a variable since there exists nested tuple. So we need a recursive logic again here.

```python
def UpdateExprMap(base: Expr, increment: Var):
    if base is Var:
        adjoint_expr_map_[base] = Addition(adjoint_expr_map_[v], increment)
    else if base is Tuple:
        for i in range(len(base)):
            UpdateExprMap(base[i], TupleGetItem(increment, i))
```

Note that here the increment is Var due to the "expr and var" issue. It looks good but a problem hides. We can observe that here we pass a `TupleGetItem(Var, pos)` (or nested TupleGetItem) to Addition as its second argument, which is unacceptable. We can not tell whether a  `TupleGetItem(Var, pos)` is bound to a Tuple or not.

Our temporary solution is ignoring the rule of "expr and var" issue for Tuples:

```python
def UpdateExprMap(base: Expr, increment: Expr):
    if base is Var:
        adjoint_expr_map_[base] = Addition(adjoint_expr_map_[v], increment)
    else if base is Tuple:
        for i in range(len(base)):
            UpdateExprMap(base[i], increment[i])
```

Here the increment is no longer `adjoint_var` but  `adjoint_expr`. 

This violates the principal mentioned in the "Expr and Var" section: We should always use an adjoint in the form of adjoint Var. To memorize and reuse the result of `increment`, we can emit a binding and update the map `adjoint_expr_to_var_` 

```python
def UpdateExprMap(base: Expr, increment: Expr):
    if base is Var:
        if increment in adjoint_expr_to_var_:
            increment_var = adjoint_expr_to_var_[increment]
        else:
            increment_var = Var()
            BindAndEmit(increment_var, increment)
            adjoint_expr_to_var_[increment] = increment_var
        adjoint_expr_map_[base] = Addition(adjoint_expr_map_[v], increment_var)
    else if base is Tuple:
        for i in range(len(base)):
            UpdateExprMap(base[i], increment[i])
```



## Deal With TupleGetItem

Recall E.g. 3.1 and our discussion of the partial update

```
Before:
b = a[0]

After:
a_adjoint[0] += b_adjoint
```

We have explained that when it is a first-time update, we should call `BuildEmptyNestedTupleExpr` to build the whole skeleton of this adjoint Tuple first. 

Next we should consider how to implement this partial update. The difficulty is in Relax `t[0]` can not be a left-value. For a var binding, the left value should always be a variable. So we should do this manually by diving into the tuple. That is to say, we need implement a method `AdditionInTuple` which can do addition in a specific position of Tuple.

Recall that the TupleGetItem can not be nested, which is good news for us because we don't need recursive searching here. We can just scan the Tuple in its first layer:

```python
def AdditionInTuple(tuple: Tuple, pos: int, increment: Expr):
	ret = []
    for i in range(len(tuple)):
        if i == index:
            ret.append(Addition(tuple[i], increment)) # add in this position!
        else:
            ret.append(tuple[i]) # no change
    return Tuple(ret)
```

And it is time to do the propagation

```python
if binding->value is TupleGetItem:
    adjoint_var = adjoint_var_map_[binding->var] # b_adjoint
    adjoint_expr = adjoint_expr_map_[binding->var] # b_adjoint_expr
    
    updated_tuple = binding->value->tuple
    tuple_var = Downcast<Var>(updated_tuple) # a_adjoint
    
    if updated_tuple not in adjont_expr_map_: # first-time update
        BuildEmptyNestedTuple(updated_tuple->shape, updated_tuple->checked_type)
    
    adjoint_expr_map_[tuple_var] = AdditionInTuple(
        adjoint_expr_map_[tuple_var], # a_adjoint_expr: Tuple
        binding->value->index, adjoint_expr)
```



# Q&A of Some Logic

## UpdateExprMap and Addition

There is no duplicate logic in these two methods. (It looks like because they all have a recursively tuple-aware logic.)

- UpdateExprMap is to deal with all leaves

  For a call `d = op(a, b, c)`, we should do

  ```
  UpdateExprMap(a, d_adjoint) 
  UpdateExprMap(b, d_adjoint)
  UpdateExprMap(c, d_adjoint)
  ```

  Here we don't know what is `a/b/c`. Under the assumption of normalization, it can be all relax leaf nodes.

- Addition is just a tuple-aware AD (since `relax.add` does not support Tuple)

## Expr and Var

To show the problem clearly, here is an example:

```c++
@I.ir_module
class Before:
    @R.function
    def main(x: R.Tensor((3, 3), "float32")):
        with R.dataflow():
            lv1 = x
            lv2 = R.add(lv1, x)
            lv3 = R.add(lv2, lv1)
            lv4 = R.sum(lv3)
            R.output(lv4)
        return lv4
```

