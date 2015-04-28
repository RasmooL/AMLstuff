--[[
   Criterion that weights the quadratic reconstruction cost between two nodes.
--]]
require('torch')
require('nn')

local WeightedCriterion, parent = torch.class('nn.WeightedCriterion', 'nn.Criterion')

function WeightedCriterion:__init()
   parent.__init(self)
end

function WeightedCriterion:forward(vecs, weights)
   local N = vecs[1]:size(1)
   self.v1 = torch.add(vecs[1]:narrow(1, 1, N/2),     -vecs[2]:narrow(1, 1, N/2))
   self.v2 = torch.add(vecs[1]:narrow(1, N/2+1, N/2), -vecs[2]:narrow(1, N/2+1, N/2))
   self.output =  torch.pow(torch.norm(self.v1, 2), 2) * (weights[1] / (weights[1] + weights[2])) +
                  torch.pow(torch.norm(self.v2, 2), 2) * (weights[2] / (weights[1] + weights[2]))
   return self.output
end

function WeightedCriterion:backward(input, target)
   self.gradInput = torch.cat(torch.mul(self.v1, 2), torch.mul(self.v2, 2), 1)
   return self.gradInput
end
