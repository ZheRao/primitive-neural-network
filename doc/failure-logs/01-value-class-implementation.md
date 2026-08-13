# [2026-08] Entry

## 1. Tracking the Result as `parent` on the Input Node

**Initial design**

For `a * b = c`, store:

```python
a.parent = c
```

**Why it failed**

Backpropagation naturally needs to move from `c → a, b`. Storing the forward result on the input points the graph representation in the inconvenient direction.

It also becomes ambiguous when an input participates in multiple operations:

```python
c = a * b
d = a * e
```

A single `a.parent` cannot represent both `c` and `d`; the second assignment overwrites the first.

**Realization**

The result should instead remember the nodes that produced it:

```text
c
└── children: (a, b)
```

The important relationship is:

> `c` knows that it was produced from `a` and `b`.

---

## 2. Computing Gradients During the Forward Operation

**Initial design**

Inside `__mul__`:

```python
result = Value(self.data * y.data)
self.grad = y.data * result.grad
```

**Why it failed**

During the forward pass, `result.grad` is still initialized to `0`. The upstream gradient does not exist until the entire forward computation is complete and the final node's gradient is seeded.

Therefore every gradient calculated during `__mul__` becomes `0`.

**Realization**

Separate:

```text
forward time  → calculate values + define backward behavior
backward time → execute that behavior using gradients now available
```

---

## 3. Separating Backpropagation into One Generic Function

**Problem discovered**

Moving gradient computation out of `__mul__` creates another question:

> How does a generic backward function know whether the node came from multiplication, addition, etc.?

A centralized implementation would tend toward:

```python
if operation == "mul":
    ...
elif operation == "add":
    ...
```

**Realization**

The operation itself should define its local backward rule because it already knows the derivative behavior of that operation.

The rule should be **defined during the forward operation but executed later**.

This naturally motivated storing executable behavior as a closure.

---

## 4. Not Understanding Why a Closure Is Useful

**Initial confusion**

Closures previously seemed like unnecessary complexity: why define a function inside another function and preserve its surrounding variables?

**Realization**

A closure solves exactly the requirement:

> Store a rule now without executing it, while preserving access to the objects the rule will need later.

For multiplication, `_back()` can retain references to:

```text
self
y
result
```

even after `__mul__` finishes.

The closure therefore stores **behavior + access to its required environment**, rather than immediately calculating a gradient.

---

## 5. Attaching `_back` to `self` Instead of `result`

**Initial design**

Inside:

```python
result = self * y

def _back():
    parent_grad = result.grad
    self.grad = y_data * parent_grad
    y.grad = self.data * parent_grad
```

attach:

```python
self._back = _back
```

**Why it failed**

The backward rule describes:

> Given `result.grad`, how should that gradient propagate into the inputs that created `result`?

Therefore the rule belongs to `result`, not `self`.

Attaching it to `self` breaks chained computational flow.

**Realization**

```python
result._back = _back
```

Each result node owns the rule for propagating **its gradient one step backward into its children**.

For:

```text
a × b → c
c × d → e
```

the responsibilities become:

```text
e._back() → computes c.grad and d.grad
c._back() → computes a.grad and b.grad
```

---

## 6. Confusing Who Owns `_back` With Whose Gradient `_back` Modifies

**Confusion**

If `c._back()` contains logic that modifies:

```python
a.grad
b.grad
```

it initially seemed asymmetric that the backward code for `a` and `b` lives on `c`.

**Realization**

`_back` is not “the code for calculating this node's gradient.”

It means:

> Given **my already-computed gradient**, propagate its contribution into the nodes that produced me.

Therefore:

```text
node.grad  = gradient state belonging to node

node._back = behavior for sending that state one step backward
```

---

## 7. Confusing `set` Syntax With `tuple` Syntax

**Initial assumption**

```python
(a, b)
```

was assumed to be a `set`.

**Failure**

The constructor expected:

```python
set[Value]
```

but received:

```python
tuple[Value, Value]
```

causing a Pylance type error.

**Realization**

```python
(a, b)   # tuple
{a, b}   # set
[a, b]   # list
```

`set(...)` uses parentheses because `set` is being called; the parentheses are not set syntax.

This also exposed a deeper modeling consideration: tuples preserve duplicate operands such as:

```python
a * a
```

as:

```python
(a, a)
```

whereas a set would collapse duplicate references.

---

## 8. Deciding How to Initialize `_back`

**Initial question**

Should every `Value` initialize `_back` by defining a no-op function?

```python
def _back() -> None:
    pass

self._back = _back
```

**Alternative discovered**

```python
self._back = None
```

is also valid and arguably more explicit during early development.

It means:

> This node has no backward behavior because no operation produced it.

The tradeoff is that traversal must eventually check:

```python
if node._back is not None:
    node._back()
```

instead of universally calling:

```python
node._back()
```

**Realization**

This is not a correctness issue but an API/semantic design decision:

```text
None
→ absence of backward behavior

no-op function
→ backward behavior exists but intentionally does nothing
```

---

## Current Working Primitive

Test:

```python
a = Value(3, "a")
b = Value(4, "b")

c = a * b
c.name = "c"
c.grad = 1

c._back()
```

Result:

```text
c.data = 12
c.grad = 1
c.children = (a, b)

a.grad = 4
b.grad = 3
```

**Current milestone:** a single multiplication node can construct its forward value, retain its graph inputs, store its own delayed backward behavior, and correctly propagate an upstream gradient one step backward.

**Next experiment:** chained multiplication, e.g.

```python
c = a * b
e = c * d
```

to expose how backward rules should be traversed automatically through a multi-level computation graph.
