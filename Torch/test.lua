require('torch')
require('nn')
require('optim')

local enc = nn.Sequential()
enc:add(nn.Linear(100, 50))
enc:add(nn.Tanh())

local dec = nn.Sequential()
dec:add(nn.Linear(50, 100))
dec:add(nn.Tanh())

local model = nn.Sequential():add(enc):add(dec)

local w, dw = model:getParameters()

local criterion = nn.MSECriterion()

local input = torch.rand(100)
local feval = function(x)
   dw:zero()

   local hid = enc:forward(input)
   local rec = dec:forward(hid)
   local cost = criterion:forward(input, rec)
   print(cost)

   local cost_grad = criterion:backward(input, rec)
   local dec_grad = dec:backward(hid, cost_grad)
   local enc_grad = enc:backward(input, dec_grad)
   return cost, -dw
end

local optim_state = {}
optim_state.learningRate = 0.1

for i = 1, 500 do
   optim.lbfgs(feval, w, optim_state)
end
