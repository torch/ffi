
local function getDataArray(tensor, pointerDef)
   return ffi.cast(pointerDef, torch.pointer(tensor)).storage.data
end

local function redefineMap(tensorClass, pointerDef)
   local origMap = tensorClass.map
   tensorClass.map = function(tensor, other, func)
      if (not tensor:isContiguous()) or (not other:isContiguous()) then
         return origMap(tensor, other, func)
      end

      local data = getDataArray(tensor, pointerDef)
   	  local otherdef = torch.typename(other):gfind('torch%.(.*Tensor)')().."*"
      local otherdata = getDataArray(other, otherdef)
      local offset = tensor:storageOffset()
      -- A zero-based index is used to access the data.
      -- The end index is (startIndex + nElements - 1).
      for i0 = offset - 1, offset - 1 + tensor:nElement() - 1 do
         data[i0] = func(data[i0], otherdata[i0]) or data[i0]
      end
      return tensor
   end
end

-- Define the faster map() for Tensors of all types:
for k, v in pairs(torch) do
   if k:find('(.+)Tensor') then
      if k ~= 'repeatTensor' then
         local pointerDef = k .. '*'
         redefineMap(v, pointerDef)
      end
    end
end
