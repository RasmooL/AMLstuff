require('torch')
require('nn')
require('nngraph')
require('optim')

include('Normalize.lua')
include('WeightedCriterion.lua')
include('IdentityLinear.lua')

local BinaryRAE, parent = torch.class('BinaryRAE', 'nn.Module')

function BinaryRAE:__init(emb_dim)
   self.in_dim = 2*emb_dim
   self.hid_dim = emb_dim
   self.rec_dim = 2*emb_dim

   -- encoder
   self.encoder = nn.Sequential()
   self.encoder:add(nn.IdentityLinear(self.in_dim, 50))
   self.encoder:add(nn.Tanh())
   --self.encoder:add(nn.Linear(75, self.hid_dim))
   --self.encoder:add(nn.Tanh())
   self.encoder:add(nn.Normalize()) -- Constrain encoded vector to length 1

   -- decoder
   self.decoder = nn.Sequential()
   self.decoder:add(nn.IdentityLinear(self.hid_dim, 100))
   self.decoder:add(nn.Tanh())
   --self.decoder:add(nn.Linear(75, self.rec_dim))
   --self.decoder:add(nn.Tanh())
   self.decoder:add(nn.Linear(self.rec_dim, self.rec_dim))
   --self.decoder:add(nn.Linear(75, self.rec_dim))
   --self.decoder:add(nn.Tanh())

   self.ae = nn.Sequential()
   self.ae:add(self.encoder):add(self.decoder)

   -- loss
   self.criterion = nn.WeightedCriterion()
   self.criterion.sizeAverage = false -- not mean

   self.params, self.gparams = self.ae:getParameters()

   self.opt = optim.adagrad

   if self.opt == optim.sgd then
      self.optim_state = {learningRate = 1e-5, learningRateDecay = 1e-2, momentum = 0.8, dampening = 0, nesterov = true}
   elseif self.opt == optim.lbfgs then
      self.optim_state = {maxIter = 1, learningRate = 2}--, lineSearch = optim.lswolfe}
   elseif self.opt == optim.nag then
      self.optim_state = {learningRate = 1e-4, momentum = 0.99}
   elseif self.opt == optim.adagrad then
      self.optim_state = {learningRate = 1e-2, learningRateDecay = 0}
   end
end

function BinaryRAE:forward(tree)
   -- last node left = root
   if #tree == 1 then
      return tree[1]
   end
   local best_cost = math.huge
   local best_i = 0
   local best_node = 0

   for i = 1, #tree - 1 do
      local first = tree[i]
      local second = tree[i+1]

      if not first then
         print(tree)
      end

      local input = torch.cat(first.value, second.value)
      local hidden = self.encoder:forward(input)
      local rec = self.decoder:forward(hidden)
      local cost = self.criterion:forward({input, rec}, {first:size(), second:size()})

      if cost < best_cost then
         best_cost = cost
         best_i = i

         -- Construct new node
         local bn = Node()
         bn.value = hidden:clone()
         bn.cost = best_cost + first.cost + second.cost -- sum costs to root
         bn:add_child(first)
         bn:add_child(second)
         best_node = bn
      end
   end

   -- Replace children with parent
   table.remove(tree, best_i)
   table.remove(tree, best_i)
   table.insert(tree, best_i, best_node)

   return self:forward(tree)
end

function BinaryRAE:parameters()
   return self.params, self.gparams
end

function BinaryRAE:calcGrad(node, parentDelta)
   -- todo
   if parentDelta == nil then
      parentDelta = torch.Tensor(self.hid_dim):zero() -- default value
   end

   local first = node.children[1]
   local second = node.children[2]

   local input = torch.cat(first.value, second.value)
   local hidden = self.encoder:forward(input)
   local rec = self.decoder:forward(hidden)
   local cost = self.criterion:forward({input, rec}, {first:size(), second:size()})

   local cost_grad = self.criterion:backward(input, rec)
   local dec_grad = self.decoder:backward(hidden, cost_grad) + parentDelta
   local enc_grad = self.encoder:backward(input, dec_grad)

   -- propagate parent error to children
   local N = enc_grad:size(1)

   if not first:is_leaf() then
      self:calcGrad(first, enc_grad:narrow(1, 1, N/2))
   end
   if not second:is_leaf() then
      self:calcGrad(second, enc_grad:narrow(1, (N/2)+1, (N/2)))
   end
end

function BinaryRAE:resetGrad()
   self.gparams:zero()
end

function BinaryRAE:accGrad(tree)
   local root = self:forward(tree)
   self:calcGrad(root)

   return root.cost
end

function BinaryRAE:train(cost)
   local feval = function(x)
      return cost, -self.gparams
   end

   self.opt(feval, self.params, self.optim_state)
end
