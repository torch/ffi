
local function getDataArray(tensor, pointerDef)
   return ffi.cast(pointerDef, torch.pointer(tensor)).storage.data
end

local function redefineApply(tensorClass, pointerDef)
   local origApply = tensorClass.apply
   tensorClass.apply = function(tensor, func)
      if not tensor:isContiguous() then
         return origApply(tensor, func)
      end

      local data = getDataArray(tensor, pointerDef)
      local offset = tensor:storageOffset()
      -- A zero-based index is used to access the data.
      -- The end index is (startIndex + nElements - 1).
      for i0 = offset - 1, offset - 1 + tensor:nElement() - 1 do
         data[i0] = func(data[i0]) or data[i0]
      end
      return tensor
   end
end

-- Define the faster apply() for Tensors of all types:
for k, v in pairs(torch) do
   if k:find('(.+)Tensor') then
      if k ~= 'repeatTensor' then
         local pointerDef = k .. '*'
         redefineApply(v, pointerDef)
      end
    end
end
