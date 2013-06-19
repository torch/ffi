
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

for i, tensorType in ipairs({'double', 'float', 'int', 'byte'}) do
   local origInput = torch.randn(20, 100, 100)
   origInput = origInput[tensorType](origInput)

   print(string.format("%s vectorized:", tensorType))
   local input1 = origInput:clone()
   measure(1000, function()
      input1:add(1):mul(-1.002)
   end)

   print(string.format("%s apply:", tensorType))
   local input2 = origInput:clone()
   measure(1000, function()
      input2:apply(function(x)
         return (x + 1) * -1.002
      end)
   end)

   local maxSqError = 0
   if tensorType == 'float' then
      maxSqError = 1e-5
   end
   local diff = input1:clone():add(-1, input2)
   assert(diff:cmul(diff):max() <= maxSqError, "expecting the same output")

   print()
end
