TorchFFI
========

This package exposes Torch's Tensor and Storage data structures, through
LuaJIT's FFI. This allows extremely fast access to Tensors and Storages, 
all from Lua.

Installation
------------

* Install Torch7 (refer to its own [documentation](www.torch.ch)). It
  is necessary to build it with LuaJIT (the default for new Torch installs).

* Install _torchffi_:

```sh
$ torch-rocks install torchffi
```

Usage
-----

Simply require `torchffi`, and you'll have access to a new function in `torch`,
called `torch.data(obj)`:

```lua
> t = torch.randn(3,2)
> print(t)
 0.8008 -0.6103
 0.6473 -0.1870
-0.0023 -0.4902
[torch.DoubleTensor of dimension 3x2]

> t_data = torch.data(t)
> for i = 0,t:nElement()-1 do t_data[i] = 0 end
> print(t)
0 0
0 0
0 0
[torch.DoubleTensor of dimension 3x2]
```

Accessing the raw data of a Tensor like this is extremely efficient, in fact, it's
almost as fast as C in lots of cases.

WARNING: bear in mind that accessing the raw data like this is dangerous, and should
only be done on contiguous tensors (if a tensor is not contiguous, then you have to
use it size and stride information). Making sure a tensor is contiguous is easy:

```lua
> t = torch.randn(3,2)
> t_noncontiguous = t:transpose(1,2)

-- it would be unsafe to work with torch.data(t_noncontiguous)

> t_transposed_and_contiguous = t:noncontiguous:contiguous()

-- it is now safe to work with the raw pointer

> data = torch.data(t_contiguous)
```

