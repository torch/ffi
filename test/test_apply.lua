
require 'torch'
require 'torchffi'

local mytest = {}
local tester = torch.Tester()

local function eq(tensor1, tensor2, message)
   tester:assertTensorEq(tensor1, tensor2, 1e-300, message)
end

function mytest.test_apply()
   local input = torch.Tensor({1, 2, 3, 4})
   input:apply(function(x)
      return 2 * x
   end)
   eq(input, torch.Tensor({2, 4, 6, 8}), "expecting doubles")
end

function mytest.test_applyWithNilReturn()
   local input = torch.Tensor({1, 2, 3, 4})
   local count = 0
   input:apply(function(x)
      count = count + 1
      return
   end)
   eq(input, torch.Tensor({1, 2, 3, 4}), "expecting no change")
   tester:asserteq(count, 4, "count")
end

function mytest.test_applyWithStorageOffset()
   local origInput = torch.Tensor({1, 2, 3, 4, 5, 6})
   local input = origInput[{{3, 5}}]
   input:apply(function(x)
      return 2 * x
   end)

   eq(origInput, torch.Tensor({1, 2, 6, 8, 10, 6}), "expecting a subset of doubles")
end

function mytest.test_applyWithNonContiguous()
   local origInput = torch.Tensor({{1, 2, 3}, {4, 5, 6}})
   local input = origInput:t()
   assert(not input:isContiguous(), "expecting non-contiguous")
   input:apply(function(x)
      return 2 * x
   end)

   eq(input, torch.Tensor({{2, 4, 6}, {8, 10, 12}}):t(), "expecting transposed doubles")
end

tester:add(mytest)
tester:run()
