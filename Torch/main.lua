require('pl')
require('torch')
require('nn')
require('gnuplot')

include('BinaryRAE.lua')
include('util.lua')
include('Vocab.lua')

cmd = torch.CmdLine()
cmd:text()
cmd:option('-train', false, 'train the model (true/false)')
cmd:option('-q', 'computer science', 'query')
params = cmd:parse(arg)

local model = BinaryRAE()
if path.isfile('model.th') then
   model = torch.load('model.th')
end

-- load vocab & word embeddings
local vocab = Vocab('vocab.th')
vocab:add_unk_token()
local emb_vecs = torch.load('vectors.50d.th')
local emb = nn.LookupTable(emb_vecs:size(1), emb_vecs:size(2))
emb.weight:copy(emb_vecs)

-- load corpus - TODO: Handle UNK???
local train_dir = 'train/'
local corpus, files = load_corpus(train_dir, vocab, emb)

-- train
if params.train then
   local num_epochs = 100
   local costs = torch.Tensor(num_epochs)
   for i = 1, num_epochs do
      local cost = 0
      local count = 0
      model:resetGrad()
      for d = 1, #corpus do
         local doc = corpus[d]
         for s = 1, #doc do
            if doc[s]:size(1) ~= 1 then -- 1 word in sentence
               count = count + 1
               local tree = leaf_tree(doc[s])
               cost = cost + model:accGrad(tree)
            end
         end
      end
      costs[i] = cost / count
      model:train(cost)
      --local tree = leaf_tree("the cat sat", vocab, emb)
      --costs[i] = model:train(tree)
      print("Cost at iteration " .. i .. " is " .. costs[i])
   end

   torch.save('model.th', model)

   -- training curve
   gnuplot.figure(1)
   gnuplot.plot(costs)

   -- encoder weights
   gnuplot.figure(2)
   gnuplot.imagesc(model.encoder:parameters()[1], 'color')

   -- decoder weights
   gnuplot.figure(3)
   gnuplot.imagesc(model.decoder:parameters()[1], 'color')
else
   -- build corpus vectors
   local vecs = {}
   for d = 1, #corpus do
      local doc = corpus[d]
      vecs[d] = {}
      for s = 1, #doc do
         local root = model:forward(leaf_tree(doc[s]))
         vecs[d][s] = root.value:clone()
      end
   end

   local q = leaf_tree_str(params.q, vocab, emb)
   local vec = model:forward(q).value

   local best_cost = math.huge
   local best_d = 0
   local best_s = 0
   for d = 1, #corpus do
      local doc = corpus[d]
      for s = 1, #doc do
         local cost = torch.norm(vecs[d][s] - vec, 2)
         if cost < best_cost then
            best_cost = cost
            best_d = d
            best_s = s
         end
      end
   end

   print(files[best_d], best_s, best_cost)
end
