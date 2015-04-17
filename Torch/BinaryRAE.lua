require('torch')
require('nn')
require('nngraph')
require('optim')

local BinaryRAE, parent = torch.class('BinaryRAE', 'nn.Module')

function BinaryRAE:__init(config)
   self.in_dim = 100
   self.hid_dim = 50
   self.rec_dim = self.in_dim

   -- encoder
   local x = nn.Identity()()
   local hAct = nn.Linear(self.in_dim, self.hid_dim)(x)
   local h = nn.Tanh()(hAct)
   self.encoder = nn.gModule({x}, {hAct, h}) -- return activation and hidden

   -- decoder
   local hi = nn.Identity()()
   local rAct = nn.Linear(self.hid_dim, self.rec_dim)(hi)
   local r = nn.Tanh()(rAct)
   self.decoder = nn.gModule({hi}, {rAct, r})

   --self.decoder = nn.Sequential()
   --self.decoder:add(nn.Linear(self.hid_dim, self.rec_dim))
   --self.decoder:add(nn.Tanh())

   -- loss
   self.criterion = nn.MSECriterion()
   self.criterion.sizeAverage = false -- not mean

   grad = {}
   grad[1] = torch.Tensor(self.hid_dim, self.in_dim):zero() -- W1
   grad[2] = torch.Tensor(self.hid_dim):zero() -- b1
   grad[3] = torch.Tensor(self.rec_dim, self.hid_dim):zero() -- W2
   grad[4] = torch.Tensor(self.rec_dim):zero() -- b2
   self.params, self.gparams = self:getParameters()

   self.optim_state = {learningRate = 6e-4, momentum=0.8}
end

function BinaryRAE:forwardProp(input)
   local enc = self.encoder:forward(input)
   local hiddenAct = enc[1]
   local hidden = enc[2]
   local dec = self.decoder:forward(hidden)
   local recAct = dec[1]
   local rec = dec[2]

   return hiddenAct, hidden, recAct, rec
end

function BinaryRAE:forward(tree)
   local function shallowCopy(original)
       local copy = {}
       for key, value in ipairs(original) do
           copy[key] = value
       end
       return copy
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

      local input = torch.cat(first.value, second.value)
      local hiddenAct, hidden, recAct, rec = self:forwardProp(input)
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

   -- last node left = root
   if #self.temp_tree == 1 then
      local root = self.temp_tree[1]
      self.temp_tree = nil
      return root
   end

   return self:forward(self.temp_tree)
end

function BinaryRAE:parameters()
  local params = {}
  local ep, _ = self.encoder:parameters()
  tablex.insertvalues(params, ep)
  local dp, _ = self.decoder:parameters()
  tablex.insertvalues(params, dp)
  return params, grad
end

function BinaryRAE:calcGrad2(node, parentDelta)
   -- todo
   if parentDelta == nil then
      parentDelta = torch.Tensor(self.hid_dim):zero() -- default value
   end

   local first = node.children[1]
   local second = node.children[2]

   local input = torch.cat(first.value, second.value)
   local hiddenAct, hidden, recAct, rec = self:forwardProp(input)
   local cost = self.criterion:forward(input, rec)

   local error = input - rec
   local delta_r = torch.cmul(-error, torch.ones(self.rec_dim) - torch.pow(torch.tanh(recAct), 2)) -- elementwise multiplication with tanh'(recAct)

   -- gradient for output layer
   local W2grad = torch.ger(delta_r, hidden) -- outer product: delta_r * hidden^T
   local b2grad = delta_r

   -- error to hidden layer
   local W2 = self.decoder:parameters()[1] -- decoder weights
   local delta_e = torch.cmul(W2:t() * delta_r, torch.ones(self.hid_dim) - torch.pow(torch.tanh(hiddenAct), 2)) + parentDelta

   -- gradient for hidden layer
   local W1grad = torch.ger(delta_e, input)
   local b1grad = delta_e

   --
   -- From here is BPTS
   --

   -- error due to parent node in tree
   local W1 = self.encoder:parameters()[1]
   local delta_p = torch.cmul(W1:t() * delta_e + error, torch.ones(self.in_dim) - torch.pow(torch.tanh(input), 2))

   grad[1]:add(W1grad)-- = grad[1] + W1grad
   grad[2]:add(b1grad)-- = grad[2] + b1grad
   grad[3]:add(W2grad)-- = grad[3] + W2grad
   grad[4]:add(b2grad)-- = grad[4] + b2grad

   local N = delta_p:size(1)
   local lC = node.children[1]
   local rC = node.children[2]

   if not lC:is_leaf() then
      self:calcGrad2(lC, delta_p:narrow(1, 1, N/2))
   end

   if not rC:is_leaf() then
      self:calcGrad2(rC, delta_p:narrow(1, (N/2)+1, (N/2)))
   end
end

function BinaryRAE:calcGrad(node, parentDelta)
   -- todo
   if parentDelta == nil then
      parentDelta = torch.Tensor(self.hid_dim):zero() -- default value
   end

   local first = node.children[1]
   local second = node.children[2]

   local input = torch.cat(first.value, second.value)
   local hiddenAct, hidden, recAct, rec = self:forwardProp(input)
   local cost = self.criterion:forward(input, rec)

   local error = input - rec
   local delta_r = torch.cmul(-error, torch.ones(self.rec_dim) - torch.pow(torch.tanh(recAct), 2)) -- elementwise multiplication with tanh'(recAct)

   -- gradients for decoder weights
   local W2grad = torch.ger(delta_r, hidden) -- outer product: delta_r * hidden^T
   local b2grad = delta_r

   -- error to hidden layer
   local W2 = self.decoder:parameters()[1] -- decoder weights
   local delta_e = torch.cmul(W2:t() * delta_r, torch.ones(self.hid_dim) - torch.pow(torch.tanh(hiddenAct), 2)) + parentDelta

   -- gradients for encoder weights
   local W1grad = torch.ger(delta_e, input)
   local b1grad = delta_e

   --
   -- From here is BPTS
   --

   -- error due to parent node in tree
   local W1 = self.encoder:parameters()[1]
   local delta_p = torch.cmul(W1:t() * delta_e + error, torch.ones(self.in_dim) - torch.pow(torch.tanh(input), 2))

   grad[1]:add(W1grad)-- = grad[1] + W1grad
   grad[2]:add(b1grad)-- = grad[2] + b1grad
   grad[3]:add(W2grad)-- = grad[3] + W2grad
   grad[4]:add(b2grad)-- = grad[4] + b2grad

   local N = delta_p:size(1)
   local lC = node.children[1]
   local rC = node.children[2]

   if not lC:is_leaf() then
      self:calcGrad2(lC, delta_p:narrow(1, 1, N/2))
   end

   if not rC:is_leaf() then
      self:calcGrad2(rC, delta_p:narrow(1, (N/2)+1, (N/2)))
   end
end

function BinaryRAE:resetGrad()
   --grad = {}
   grad[1]:zero()-- = torch.Tensor(self.hid_dim, self.in_dim):zero() -- W1
   grad[2]:zero()-- = torch.Tensor(self.hid_dim):zero() -- b1
   grad[3]:zero()-- = torch.Tensor(self.rec_dim, self.hid_dim):zero() -- W2
   grad[4]:zero()-- = torch.Tensor(self.rec_dim):zero() -- b2
end

function BinaryRAE:train(data)
   self:resetGrad()
   local root = nil
   local feval = function(x)
      root = self:forward(data)
      self:calcGrad2(root)
      return root.cost, self.gparams
   end

   -- accumulate gradients for tree
   --local root = self:forward(data)
   --self:calcGrad2(root)

   -- gradient descent
   --local lr = 3e-4
   --params[1]:add(-torch.mul(grad["W1"], lr))
   --params[2]:add(-torch.mul(grad["b1"], lr))
   --params[3]:add(-torch.mul(grad["W2"], lr))
   --params[4]:add(-torch.mul(grad["b2"], lr))

   optim.adagrad(feval, self.params, self.optim_state)

   return root.cost
end
