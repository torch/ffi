--[======================================================================[
   TorchFFI: FFI bindings for Torch.

   This package is a pretty straightforward extension to Torch7, which
   exposes its main data structures through FFI.

   Requires LuaJIT.
--]======================================================================]

-- This package is now deprecated
print [[
WARNING: 'torchffi' is now deprecated. All of its functionality has been
merged/integrated into 'torch'.
]]

-- We need FFI, of course :-)
require 'torch'
local ok = pcall(function() ffi = require 'ffi' end)
if not ok then
   error('FFI could not be loaded, please make sure you built Torch with LuaJIT (cmake -DWITH_LUA_JIT=1)')
end

-- Generate Bindings for Storages of all types:
local defs = {}
for k,v in pairs(torch) do
   if k:find('(.+)Storage') then
      if k ~= 'repeatStorage' and k ~= 'CudaStorage' then
         local type_storage = k
         local type_elt = k:gmatch('(.*)Storage')():lower()
         type_elt = type_elt:gsub('byte', 'unsigned char')
         table.insert(defs, [[
            typedef struct
            {
               ]] .. type_elt .. [[ *data;
               long size;
               int refcount;
               char flag;
            } ]] .. type_storage .. [[;
         ]])
      end
   end
end

-- Generate Bindings for Tensors of all types:
for k,v in pairs(torch) do
   if k:find('(.+)Tensor') then
      if k ~= 'repeatTensor' and k ~= 'CudaTensor' then
         local type_tensor = k
         local type_storage = k:gmatch('(.*)Tensor')() .. 'Storage'
         table.insert(defs, [[
            typedef struct 
            {
               long *size;
               long *stride;
               int nDimension;
               ]] .. type_storage .. [[ *storage;
               long storageOffset;
               int refcount;
               char flag;
            } ]] .. type_tensor .. [[;
         ]])
      end
   end
end

-- Load defs
defs = table.concat(defs,'\n')
ffi.cdef(defs)

-- Method to return raw data table
function torch.data(obj, asnumber)
   -- first cast pointer into the right type
   local type_obj = torch.typename(obj)
   local type_tensor = type_obj:gfind('torch%.(.*)Tensor')()
   local type_storage = type_obj:gfind('torch%.(.*)Storage')()
   local ptr
   if type_tensor then
      -- return raw pointer to data
      ptr = ffi.cast(type_tensor .. 'Tensor*', torch.pointer(obj)).storage.data + (obj:storageOffset() - 1)
   elseif type_storage then
      -- return raw pointer to data
      ptr = ffi.cast(type_storage .. 'Storage*', torch.pointer(obj)).data
   else
      print('Unknown data type: ' .. type_obj)
      return
   end
   if asnumber then
      ptr = tonumber(ffi.cast('intptr_t', ptr))
   end
   return ptr
end 

torch.include('torchffi', 'apply.lua')
