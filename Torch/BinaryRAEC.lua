require('torch')
require('nn')
require('nngraph')
require('optim')

include('Normalize.lua')
include('WeightedCriterion.lua')
include('IdentityLinear.lua')

local BinaryRAE, parent = torch.class('BinaryRAE', 'nn.Module')

function BinaryRAE:__init(emb_dim, num_classes)
   self.in_dim = 2*emb_dim
   self.hid_dim = emb_dim
   self.rec_dim = 2*emb_dim

   self.num_classes = num_classes

   -- encoder
   self.encoder = nn.Sequential()
   --self.encoder:add(nn.IdentityLinear(self.in_dim, self.hid_dim))
   self.encoder:add(nn.Linear(self.in_dim, self.hid_dim))
   self.encoder:add(nn.Tanh())
   --self.encoder:add(nn.Linear(self.hid_dim, self.hid_dim))
   --self.encoder:add(nn.Tanh())
   self.encoder:add(nn.Normalize()) -- Constrain encoded vector to length 1

   -- decoder
   self.decoder = nn.Sequential()
   --self.decoder:add(nn.Dropout(0.2))
   --self.decoder:add(nn.IdentityLinear(self.hid_dim, self.rec_dim))
   self.decoder:add(nn.Linear(self.hid_dim, self.rec_dim))
   --self.decoder:add(nn.Tanh())
   --self.decoder:add(nn.Linear(75, self.rec_dim))
   --self.decoder:add(nn.Tanh())
   --self.decoder:add(nn.Linear(self.rec_dim, self.rec_dim))
   --self.decoder:add(nn.Linear(75, self.rec_dim))
   --self.decoder:add(nn.Tanh())

   -- loss
   self.msecriterion = nn.WeightedCriterion()
   self.criterion = nn.ClassNLLCriterion()

   --classifier
   self.classifier = nn.Sequential()
   --self.classifier:add(nn.Linear(self.hid_dim, self.hid_dim))
   --self.classifier:add(nn.Tanh())
   self.classifier:add(nn.Linear(self.hid_dim, self.num_classes))
   self.classifier:add(nn.LogSoftMax()) -- outputs log probabilities to NLL criterion
   self.class_weight = 0.999 -- this weighs the classifier gradient versus the autoencoder gradient

   -- this is ONLY for getting parameter and grad matrices!
   local seq = nn.Sequential()
   seq:add(self.encoder):add(self.decoder):add(self.classifier)
   self.params, self.gparams = seq:getParameters()

   self.opt = optim.sgd

   if self.opt == optim.sgd then
      self.optim_state = {learningRate = 1e-2, learningRateDecay = 5e-3, momentum = 0.7, dampening = 0, nesterov = true}
   elseif self.opt == optim.nag then
      self.optim_state = {learningRate = 1e-4, momentum = 0.9}
   elseif self.opt == optim.adagrad then
      self.optim_state = {learningRate = 2e-2, learningRateDecay = 0}
   end
end

function BinaryRAE:forward(tree, label)
   -- last node left = root
   if #tree == 1 then
      --return tree[1]

      -- return cost also
      local logprobs = self.classifier:forward(tree[1].value)
      return tree[1], self.criterion:forward(logprobs, label)
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
      local cost = self.msecriterion:forward({input, rec}, {first:size(), second:size()})
      --local cost = self.criterion:forward(input, rec)
      --local logprobs = self.classifier:forward(hidden)
      --local cost = self.criterion:forward(logprobs, label)

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

   return self:forward(tree, label)
end

function BinaryRAE:parameters()
   return self.params, self.gparams
end

function BinaryRAE:calcGrad_(node, label, parentDelta)
   parentDelta = parentDelta or torch.Tensor(self.hid_dim):zero()

   -- calculate and backprop as autoencoder
   local first = node.children[1]
   local second = node.children[2]

   local input = torch.cat(first.value, second.value)
   local hidden = self.encoder:forward(input)
   local rec = self.decoder:forward(hidden)
   local cost = self.msecriterion:forward({input, rec}, {first:size(), second:size()})
   --local cost = self.criterion:forward(input, rec)

   local rec_grad = self.msecriterion:backward(input, rec):mul(1 - self.class_weight) -- weighted
   local dec_grad = self.decoder:backward(hidden, rec_grad)

   -- calculate and backprop cross entropy
   local logprobs = self.classifier:forward(node.value)
   local crossentropy = self.criterion:forward(logprobs, label)
   if node.parent == nil then
      node.ccost = crossentropy -- book-keeping for printing
   end

   local crit_grad = self.criterion:backward(logprobs, label):mul(-self.class_weight) -- weighted
   local class_grad = self.classifier:backward(node.value, crit_grad)
   local enc_grad = self.encoder:backward(input, dec_grad + parentDelta + class_grad)

   -- propagate parent error to children
   local N = enc_grad:size(1)

   if not first:is_leaf() then
      self:calcGrad(first, label, enc_grad:narrow(1, 1, N/2))
   end
   if not second:is_leaf() then
      self:calcGrad(second, label, enc_grad:narrow(1, (N/2)+1, (N/2)))
   end
end

-- alternative, only classifier on root node (not on internals)
function BinaryRAE:calcGrad(node, label, parentDelta)
   parentDelta = parentDelta or torch.Tensor(self.hid_dim):zero()

   if node.parent == nil then
      -- calculate and backprop cross entropy
      local logprobs = self.classifier:forward(node.value)
      local crossentropy = self.criterion:forward(logprobs, label)
      node.ccost = crossentropy -- book-keeping for printing

      local crit_grad = self.criterion:backward(logprobs, label):mul(-self.class_weight) -- weighted (seriously why is this negative? math pls)
      local class_grad = self.classifier:backward(node.value, crit_grad)

      local first = node.children[1]
      local second = node.children[2]
      local input = torch.cat(first.value, second.value)

      local enc_grad = self.encoder:backward(input, class_grad)

      -- propagate parent error to children
      local N = enc_grad:size(1)

      if not first:is_leaf() then
         self:calcGrad(first, nil, enc_grad:narrow(1, 1, N/2))
      end
      if not second:is_leaf() then
         self:calcGrad(second, nil, enc_grad:narrow(1, (N/2)+1, (N/2)))
      end
   else
      -- calculate and backprop as autoencoder
      local first = node.children[1]
      local second = node.children[2]

      local input = torch.cat(first.value, second.value)
      local hidden = self.encoder:forward(input)
      local rec = self.decoder:forward(hidden)
      local cost = self.msecriterion:forward({input, rec}, {first:size(), second:size()})
      --local cost = self.criterion:forward(input, rec)

      local rec_grad = self.msecriterion:backward(input, rec):mul(1 - self.class_weight) -- weighted
      local dec_grad = self.decoder:backward(hidden, rec_grad)
      local enc_grad = self.encoder:backward(input, dec_grad + parentDelta)

      -- propagate parent error to children
      local N = enc_grad:size(1)

      if not first:is_leaf() then
         self:calcGrad(first, label, enc_grad:narrow(1, 1, N/2))
      end
      if not second:is_leaf() then
         self:calcGrad(second, label, enc_grad:narrow(1, (N/2)+1, (N/2)))
      end
   end
end

function BinaryRAE:resetGrad()
   self.gparams:zero()
end

function BinaryRAE:accGrad(tree, label)
   local root = self:forward(tree, label)
   self:calcGrad(root, label)

   return root
end

function BinaryRAE:train(cost, batchsize)
   local feval = function(x)
      return cost, -self.gparams
   end
   --local _, cgp = self.classifier:parameters()
   --print(cgp[1])

   self.opt(feval, self.params, self.optim_state)
end
