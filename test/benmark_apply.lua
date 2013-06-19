
require 'torchffi'

local function measure(nTimes, func)
   -- warming up.
   func()

   local start = torch.tic()
   for i = 1, nTimes do
      func()
   end
   local durationSeconds = torch.toc(start)
   print(string.format("wall time: %.2f seconds", durationSeconds))
end

local origInput = torch.randn(20, 100, 100)

print("\nvectorized")
local input1 = origInput:clone()
measure(1000, function()
   input1:add(1):mul(-1.002)
end)

print("\napply")
local input2 = origInput:clone()
measure(1000, function()
   input2:apply(function(x)
      return (x + 1) * -1.002
   end)
end)

assert(input1:clone():add(-1, input2):abs():max() == 0, "expecting the same output")
