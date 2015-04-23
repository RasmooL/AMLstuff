require('torch')
require('nn')
require('nngraph')
require('optim')

local RAE, parent = torch.class('RAE', 'nn.Module')

function RAE:__init(config)
   self.in_dim = 100
   self.hid_dim = 50
   self.rec_dim = self.in_dim

   -- encoder
   self.encoder = nn.Sequential()
   self.encoder:add(nn.Linear(self.in_dim, self.hid_dim))
   self.encoder:add(nn.Tanh())

   -- decoder
   self.decoder = nn.Sequential()
   self.decoder:add(nn.Linear(self.hid_dim, self.rec_dim))
   self.decoder:add(nn.Tanh())

   -- loss
   self.criterion = nn.MSECriterion()
   self.criterion.sizeAverage = false -- not mean

   --self.params, self.gparams = self.ae:getParameters()

   self.optim_state = {learningRate = 1e-4, momentum=0.8}
end

function RAE:createNode(children)
   local seq = nn.Sequential()

   local parallel = nn.ParallelTable()
   for _, child in ipairs(children) do
      parallel:add(child)
   end
   seq:add(parallel)

   seq:add(nn.JoinTable(1))

   local sharedEncoder = self.encoder:clone('weight', 'bias')
   seq:add(sharedEncoder)

   return seq
end

function RAE:createTree(tree)
   if not self.model then
      self.model = nn.Sequential()
      self.model:add(nn.JoinTable(1))
   end

   for i = 1, #tree - 1 do
      local first = tree[i]
      local second = tree[i+1]

      if first:is_leaf() then first = nn.Identity() end
      if second:is_leaf() then second = nn.Identity() end
      local node = self:createNode({first, second})
   end
end
