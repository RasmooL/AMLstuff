require('torch')
require('xlua')

local path = '../model.word2vec'
local vocabpath = 'vocab_word2vec.th'
local vecpath = 'vectors_word2vec.100d.th'
local prefix_toks = stringx.split(path, '.')
print('Converting ' .. path .. ' to Torch serialized format')

-- get dimension and number of lines
local file = io.open(path, 'r')
local line = file:read()
local toks = stringx.split(line)
local count = tonumber(toks[1])
local dim = tonumber(toks[2])

print('count = ' .. count)
print('dim = ' .. dim)

-- convert to torch-friendly format
-- file:seek('set')
local vocab = io.open(vocabpath, 'w')
local vecs = torch.FloatTensor(count, dim)
for i = 1, count do
  xlua.progress(i, count)
  local tokens = stringx.split(file:read())
  local word = tokens[1]
  vocab:write(word .. '\n')
  for j = 1, dim do
    vecs[{i, j}] = tonumber(tokens[j + 1])
  end
end
file:close()
vocab:close()
torch.save(vecpath, vecs)
