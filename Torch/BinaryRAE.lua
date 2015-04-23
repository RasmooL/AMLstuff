require('torch')
require('nn')
require('nngraph')
require('optim')

include('Normalize.lua')

local BinaryRAE, parent = torch.class('BinaryRAE', 'nn.Module')

function BinaryRAE:__init(config)
   self.in_dim = 100
   self.hid_dim = 50
   self.rec_dim = self.in_dim

   -- encoder
   self.encoder = nn.Sequential()
   self.encoder:add(nn.Linear(self.in_dim, self.hid_dim))
   self.encoder:add(nn.Tanh())
   --self.encoder:add(nn.Normalize()) -- |h|

   -- decoder
   self.decoder = nn.Sequential()
   self.decoder:add(nn.Linear(self.hid_dim, self.rec_dim))
   self.decoder:add(nn.Tanh())

   self.ae = nn.Sequential()
   self.ae:add(self.encoder):add(self.decoder)

   -- loss
   self.criterion = nn.MSECriterion()
   self.criterion.sizeAverage = false -- not mean

   self.params, self.gparams = self.ae:getParameters()

   self.opt = optim.lbfgs

   if self.opt == optim.sgd then
      self.optim_state = {learningRate = 1e-5, momentum = 0.8} -- sgd
   elseif self.opt == optim.lbfgs then
      self.optim_state = {maxIter = 100, lineSearch = optim.lswolfe}
   end
end

function BinaryRAE:forward(tree)
   local function shallowCopy(original)
      local copy = {}
      for key, value in ipairs(original) do
           copy[key] = value
      end
      return copy
   end
   -- last node left = root
   if #tree == 1 then
      self.temp_tree = nil
      return tree[1]
   end
   if self.temp_tree == nil then
      self.temp_tree = shallowCopy(tree)
   end
   local best_cost = math.huge
   local best_i = 0
   local best_node = 0

   for i = 1, #self.temp_tree - 1 do
      local first = self.temp_tree[i]
      local second = self.temp_tree[i+1]

      if not first then
         print(self.temp_tree)
      end

      local input = torch.cat(first.value, second.value)
      local hidden = self.encoder:forward(input)
      local rec = self.decoder:forward(hidden)
      local cost = self.criterion:forward(input, rec)

      if cost < best_cost then
         best_cost = cost
         best_i = i

         -- Construct new node
         local bn = Node()
         bn.value = hidden
         bn.cost = best_cost + first.cost + second.cost -- propagate costs to root?
         bn:add_child(first)
         bn:add_child(second)
         best_node = bn
      end
   end

   -- Replace children with parent
   table.remove(self.temp_tree, best_i)
   table.remove(self.temp_tree, best_i)
   table.insert(self.temp_tree, best_i, best_node)

   return self:forward(self.temp_tree)
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
   local cost = self.criterion:forward(input, rec)

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
