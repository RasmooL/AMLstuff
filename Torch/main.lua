require('pl')
require('torch')
require('nn')
require('gnuplot')

include('BinaryRAE.lua')
include('util.lua')
include('Vocab.lua')

cmd = torch.CmdLine()
cmd:text()
cmd:option('-train', false,               'train the model (true/false)')
cmd:option('-draw', false, 'draw matrices')
cmd:option('-q',     'computer science',  'query')
cmd:option('-tsne',  false,               'run t-sne on corpus')
params = cmd:parse(arg)

local emb_dim = 50
local model = BinaryRAE(emb_dim)
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
   gnuplot.plot(costs)
elseif params.draw then
   -- encoder weights
   gnuplot.figure(2)
   gnuplot.imagesc(model.encoder:parameters()[1], 'color')

   -- decoder weights
   gnuplot.figure(3)
   gnuplot.imagesc(model.decoder:parameters()[1], 'color')
elseif params.tsne then
   -- build corpus vectors
   local num_doc = #corpus
   local vecs = {}
   local labels = {}
   local count = 1
   for d = 1, num_doc do
      local doc = corpus[d]
      vecs[d] = torch.Tensor(#doc, 50)
      for s = 1, #doc do
         local root = model:forward(leaf_tree(doc[s]))
         local vec = root.value:clone()

         vecs[d][s] = vec

         labels[count] = d
         count = count + 1
      end
   end

   -- run t-sne on corpus
   local function show_scatter_plot(mapped_x, labels, opts)

     -- count label sizes:
     local K = num_doc
     local cnts = torch.zeros(K)
     for n = 1,labels:nElement() do
        cnts[labels[n]] = cnts[labels[n]] + 1
     end

     -- separate mapped data per label:
     mapped_data = {}
     for k = 1,K do
       mapped_data[k] = { key = 'Document ' .. k, values = torch.Tensor(cnts[k], opts.ndims) }
     end
     local offset = torch.Tensor(K):fill(1)
     for n = 1,labels:nElement() do
       mapped_data[labels[n]].values[offset[labels[n]]]:copy(mapped_x[n])
       offset[labels[n]] = offset[labels[n]] + 1
     end

     -- show results in scatter plot:
     local gfx = require 'gfx.js'
     gfx.chart(mapped_data, {
        chart = 'scatter',
        width = 800,
        height = 800,
     })
   end
   local manifold = require('manifold')
   local opts = {ndims = 2, perplexity = 15, pca = 14, use_bh = false}
   --local cor = torch.Tensor(vecs[1])
   --print(cor)
   --vecs:resize(vecs:size(1) * vecs:size(2), vecs:size(3))
   vecs = torch.concat(vecs)
   local mapped = manifold.embedding.tsne(vecs, opts)
   print(mapped)
   show_scatter_plot(mapped, torch.Tensor(labels), opts)

else
   -- build corpus vectors
   local num_doc = #corpus
   local vecs = {}
   local labels = {}
   local count = 1
   for d = 1, num_doc do
      local doc = corpus[d]
      vecs[d] = {}
      for s = 1, #doc do
         local root = model:forward(leaf_tree(doc[s]))
         local vec = root.value:clone()

         vecs[d][s] = vec

         labels[count] = d
         count = count + 1
      end
   end

   local q = leaf_tree_str(params.q, vocab, emb)
   local vec = model:forward(q).value

   local sorted = {}
   local count = 1
   while #sorted < 5 do
      local best_cost = math.huge
      local best_d = 0
      local best_s = 0
      for d, doc in ipairs(vecs) do
         for s, svec in ipairs(doc) do
            --local cost = torch.dot(svec, vec) / (torch.norm(svec, 2) * torch.norm(vec, 2))
            cost = torch.norm(svec - vec, 2)
            if cost < best_cost then
               best_cost = cost
               best_d = d
               best_s = s
            end
         end
      end

      sorted[count] = {best_d, best_s, best_cost}
      vecs[best_d][best_s] = nil
      if #vecs[best_d] == 0 then
         vecs[best_d] = nil
      end
      count = count + 1
   end

   for i,v in ipairs(sorted) do
      print(i, v[3], v[2], files[v[1]])
   end
end
