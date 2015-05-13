require('torch')
require('nn')

local IdentityLinear, parent = torch.class('nn.IdentityLinear', 'nn.Linear')

-- override constructor
function IdentityLinear:__init(inSize, outSize)
   parent.__init(self, inSize, outSize)
end

-- initial weights are identity plus noise
function IdentityLinear:reset()
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(2))
   end
   if nn.oldSeed then
      for i=1,self.weight:size(1) do
         self.weight:select(1, i):apply(function()
            return torch.uniform(-stdv, stdv)
         end)
         self.bias[i] = torch.uniform(-stdv, stdv)
      end
   else
      self.weight:uniform(-stdv, stdv)
      self.bias:uniform(-stdv, stdv)
   end

   -- add identity
   if 2*self.weight:size(1) == self.weight:size(2) then -- 2*inSize = outSize
      local id = torch.eye(self.weight:size(1)):div(2) -- can adjust division
      self.weight:narrow(2, 1, self.weight:size(2)/2):add(id)
      self.weight:narrow(2, self.weight:size(2)/2+1, self.weight:size(2)/2):add(id)
   elseif self.weight:size(1) == 2*self.weight:size(2) then -- inSize = 2*outSize
      local id = torch.eye(self.weight:size(2)):div(2) -- can adjust division
      self.weight:narrow(1, 1, self.weight:size(1)/2):add(id)
      self.weight:narrow(1, self.weight:size(1)/2+1, self.weight:size(1)/2):add(id)
   end

   return self
end
