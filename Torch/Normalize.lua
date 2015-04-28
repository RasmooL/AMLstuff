--[[
 Divides each element of a Tensor by their 2-norm.
--]]

local Normalize, parent = torch.class('nn.Normalize', 'nn.Module')

function Normalize:__init()
   parent.__init(self)
end

function Normalize:updateOutput(input)
   self.output:resizeAs(input):copy(input)
   self.norm = torch.norm(input, 2)
   self.output:div(self.norm)
   return self.output
end

function Normalize:updateGradInput(input, gradOutput)
   local size = input:size(1)
   self.gradInput:resizeAs(input)
   for i = 1, size do
      local output = torch.Tensor(size):copy(self.output)
      output:div(-torch.pow(self.norm, 2))
      output:mul(input[i])
      output[i] = output[i] + (1 / self.norm)

      self.gradInput[i] = gradOutput:dot(output)
   end

   return self.gradInput
end
