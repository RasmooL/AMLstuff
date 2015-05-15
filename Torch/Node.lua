require('torch')

local Node = torch.class('Node')

function Node:__init()
  self.parent = nil
  self.num_children = 0
  self.children = {}
  self.value = 0
  self.cost = 0
end

function Node:add_child(c)
  c.parent = self
  self.num_children = self.num_children + 1
  self.children[self.num_children] = c
end

function Node:is_leaf()
   return self.num_children == 0
end


function Node:size_()
  if self._size ~= nil then return self._size end
  local size = 1
  for i = 1, self.num_children do
    size = size + self.children[i]:size()
  end
  self._size = size
  return size
end

-- alternative: counts number of leaf nodes under node
function Node:size()
  if self._size ~= nil then return self._size end
  local size = 0
  if self:is_leaf() then
     size = 1
  end
  for i = 1, self.num_children do
    size = size + self.children[i]:size()
  end
  self._size = size
  return size
end

function Node:depth()
  local depth = 0
  if self.num_children > 0 then
    for i = 1, self.num_children do
      local child_depth = self.children[i]:depth()
      if child_depth > depth then
        depth = child_depth
      end
    end
    depth = depth + 1
  end
  return depth
end
