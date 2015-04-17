require('pl')
require('torch')
require('nn')
require('gnuplot')

include('BinaryRAE.lua')
include('util.lua')
include('Vocab.lua')

model = BinaryRAE()

-- load vocab & word embeddings
local vocab = Vocab('vocab.th')
local emb_vecs = torch.load('vectors.50d.th')
local emb = nn.LookupTable(emb_vecs:size(1), emb_vecs:size(2))
emb.weight:copy(emb_vecs)

-- load corpus - TODO: Handle UNK???
local train_dir = 'train/'
--load_corpus(train_dir, vocab, emb)

--quit()

-- test BPTS
local num_epochs = 2000
local costs = torch.Tensor(num_epochs)
for i = 1, num_epochs do
   local tree = leaf_tree("the cat sat on the mat", vocab, emb)
   costs[i] = model:train(tree)
   --print("Cost at iteration " .. i .. " is " .. costs[i])
end

-- training curve
gnuplot.figure(1)
gnuplot.plot(costs)

-- encoder weights
--gnuplot.figure(2)
--gnuplot.imagesc(model:parameters()[1])

-- decoder weights
--gnuplot.figure(3)
--gnuplot.imagesc(model:parameters()[3])
