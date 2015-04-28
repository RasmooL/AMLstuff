require('torch')
require('nn')
include('WeightedCriterion.lua')

local precision = 1e-8
local module = nn.WeightedCriterion()
local input1 = torch.rand(50)
local input2 = torch.rand(50)

local err = nn.Jacobian.testJacobian(module, {input1, input2})
print('==> error: ' .. err)
if err<precision then
   print('==> module OK')
else
   print('==> error too large, incorrect implementation')
end
