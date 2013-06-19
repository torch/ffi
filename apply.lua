
local function enableFastApply()
   local mt = getmetatable(torch.Tensor)
   local origApply = mt.apply
   mt.apply = function(tensor, func)
      if not tensor:isContiguous() then
         return origApply(tensor, func)
      end

      local data = torch.data(tensor)
      local offset = tensor:storageOffset()
      -- A zero-based index is used to access the data.
      -- The end index is (startIndex + nElements - 1).
      for i0 = offset - 1, offset - 1 + tensor:nElement() - 1 do
         data[i0] = func(data[i0]) or data[i0]
      end
      return tensor
   end
end

enableFastApply()
