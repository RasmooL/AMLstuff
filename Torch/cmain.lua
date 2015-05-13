require('pl')
require('torch')
require('nn')
require('gnuplot')
require('xlua')
require('torchx')

include('BinaryRAEC.lua')
include('util.lua')
include('Vocab.lua')

torch.setnumthreads(4)
cmd = torch.CmdLine()
cmd:text()
cmd:option('-train', false,   'train the model (true/false)')
cmd:option('-draw',  false,   'draw matrices')
cmd:option('-q',     '',   'query for nearest sentence in corpus')
cmd:option('-tsne',  false,   'run t-sne on corpus')
cmd:option('-test',  '',   'test sentence reconstruction')
params = cmd:parse(arg)

local emb_dim = 100
local num_cats = 5
local model = BinaryRAE(emb_dim, num_cats)
if path.isfile('model.th') then
   model = torch.load('model.th')
end
model.decoder:evaluate() -- evaluate mode for dropout (if enabled)

-- load vocab & word embeddings
local vocab = Vocab('vocab_glove.th')
vocab:add_unk_token()
local emb_vecs = torch.load('vectors_glove_norm.100d.th')
local emb = nn.LookupTable(emb_vecs:size(1), emb_vecs:size(2))
emb.weight:copy(emb_vecs)

-- load corpus
local uris, questions, corpus, labels = load_corpus_file('../yahoo_train.txt', vocab, emb)
local uris_test, questions_test, corpus_test, labels_test = load_corpus_file('../yahoo_test.txt', vocab, emb)

-- train
if params.train then
   local num_epochs = 10
   local save_epochs = 1
   local num_minibatches = 100
   local minibatch_size = math.floor(#corpus / num_minibatches)
   local last_add = #corpus % num_minibatches
   local costs = torch.Tensor(num_epochs*num_minibatches)
   local ccosts = torch.Tensor(num_epochs*num_minibatches)
   local costs_test = torch.Tensor(num_epochs*num_minibatches)
   model.decoder:training() -- training mode for dropout
   for i = 1, num_epochs do
      -- should you shuffle the data in each epoch?
      for n = 1, num_minibatches do
         -- test set cost
         --local cost_test = 0
         --for t = 1, #corpus_test do
         --   local tree = leaf_tree(corpus_test[t])
         --   local _, tcost = model:forward(tree, labels_test[t])
         --   cost_test = cost_test + tcost
         --   xlua.progress(t, #corpus_test)
         --end
         --costs_test[(i-1)*num_minibatches + n] = cost_test / #corpus_test
         --print("Test cost at iteration " .. n .. " is " .. costs_test[(i-1)*num_minibatches + n])

         -- forward - backward pass on training set
         local cost = 0 -- accumulate reconstruction cost
         local ccost = 0 -- accumulate classification cost
         model:resetGrad() -- reset accumulated gradients
         local from_s = (n-1)*minibatch_size + 1
         local to_s = n*minibatch_size
         if n == num_minibatches then to_s = to_s + last_add end
         for s = from_s, to_s do
            local tree = leaf_tree(corpus[s])
            local root = model:accGrad(tree, labels[s])
            cost = cost + root.cost
            ccost = ccost + root.ccost
            xlua.progress(s, to_s)
         end
         costs[(i-1)*num_minibatches + n] = cost / minibatch_size
         ccosts[(i-1)*num_minibatches + n] = ccost / minibatch_size
         print("Training rec cost at iteration " .. n .. " is " .. costs[(i-1)*num_minibatches + n])
         print("Training class cost at iteration " .. n .. " is " .. ccosts[(i-1)*num_minibatches + n])

         -- one optimization step
         model:train(ccost / minibatch_size, minibatch_size)
      end

      if i % save_epochs == 0 then
         torch.save('model_' .. i .. '.th', model)
         torch.save('costs.th', costs)
         torch.save('ccosts.th', ccosts)
         torch.save('costs_test.th', costs_test)
      end
   end

   torch.save('model.th', model)
   torch.save('costs.th', costs)
   torch.save('ccosts.th', ccosts)
   torch.save('costs_test.th', costs_test)
elseif params.draw then
   -- encoder weights
   gnuplot.figure(1)
   gnuplot.imagesc(model.encoder:parameters()[1], 'color')

   -- decoder weights
   gnuplot.figure(2)
   gnuplot.imagesc(model.decoder:parameters()[1], 'color')

   gnuplot.figure(3)
   gnuplot.imagesc(model.classifier:parameters()[1], 'color')

   gnuplot.figure(4)
   local costs = torch.load('costs.th')
   local ccosts = torch.load('ccosts.th')
   local costs_test = torch.load('costs_test.th')
   gnuplot.plot({'train rec', costs}, {'train class', ccosts}, {'test', costs_test})
elseif params.tsne then
   -- load or build corpus vectors
   local vecs = torch.Tensor(#corpus, emb_dim)
   if path.isfile('corpus_vecs.th') then
      vecs = torch.load('corpus_vecs.th')
   else
      for s = 1, #corpus do
         local root = model:forward(leaf_tree(corpus[s]))
         local vec = root.value:clone()

         vecs[s] = vec
         xlua.progress(s, #corpus)
      end
   end

   -- run t-sne on corpus
   local function show_scatter_plot(mapped_x, tsne_labels, opts)

     -- count label sizes:
     local K = 0
     local cnts = torch.zeros(25)
     local keys_id = {}
     for _,l in ipairs(tsne_labels) do
        if not keys_id[l] then
           K = K + 1
           keys_id[l] = K
           cnts[K] = 1
        else
           local key = keys_id[l]
           cnts[key] = cnts[key] + 1
        end
     end
     print(K)

     -- Get inverse keys_id table
     local id_keys = {}
     for k,v in pairs(keys_id) do
        id_keys[v] = k
     end

     -- separate mapped data per label:
     mapped_data = {}
     for k = 1, K do
       mapped_data[k] = { key = id_keys[k], values = torch.Tensor(cnts[k], opts.ndims) }
     end
     local offset = torch.Tensor(K):fill(1)
     for n,l in ipairs(tsne_labels) do
        local key = keys_id[l]
        mapped_data[key].values[offset[key]]:copy(mapped_x[n])
        offset[key] = offset[key] + 1
     end

     -- show results in scatter plot:
     local gfx = require 'gfx.js'
     gfx.chart(mapped_data, {
        chart = 'scatter',
        width = 1600,
        height = 900,
     })
   end
   local manifold = require('manifold')
   local opts = {ndims = 2, perplexity = 30, pca = 100, use_bh = true}
   --vecs = torch.concat(vecs)
   local mapped = manifold.embedding.tsne(vecs, opts)
   show_scatter_plot(mapped, labels, opts)

   -- save mappings to matlab file
   local mattorch = require('fb.mattorch')
   mattorch.save('tsne.mat', mapped)

elseif params.q ~= '' then
   -- load or build corpus vectors
   local vecs = torch.Tensor(#corpus, emb_dim)
   local vecs_test = torch.Tensor(#corpus_test, emb_dim)
   if path.isfile('corpus_vecs.th') then
      vecs = torch.load('corpus_vecs.th')
   else
      for s = 1, #corpus do
         local root = model:forward(leaf_tree(corpus[s]))
         local vec = root.value:clone()

         -- Variation: average all nodes in tree as representation (instead of top node)
         local function sum_nodes(node, parent_sum)
            parent_sum:add(node.value)

            if not node:is_leaf() then
               sum_nodes(node.children[1], parent_sum)
               sum_nodes(node.children[2], parent_sum)
            end
         end
         local vec = torch.Tensor():resizeAs(root.value):fill(0)
         sum_nodes(root, vec)
         vec:div(root:size())

         -- Variation 2: average leaf nodes
         local vec = torch.Tensor():resizeAs(root.value):fill(0)
         local leaves = leaf_tree(corpus[s])
         for _,l in ipairs(leaves) do
            vec:add(l.value)
         end
         vec:div(#leaves)

         vecs[s] = vec
         xlua.progress(s, #corpus)
      end

      for s = 1, #corpus_test do
         local root = model:forward(leaf_tree(corpus_test[s]))
         local vec = root.value:clone()

         -- Variation: average all nodes in tree as representation (instead of top node)
         local function sum_nodes(node, parent_sum)
            parent_sum:add(node.value)

            if not node:is_leaf() then
               sum_nodes(node.children[1], parent_sum)
               sum_nodes(node.children[2], parent_sum)
            end
         end
         local vec = torch.Tensor():resizeAs(root.value):fill(0)
         sum_nodes(root, vec)
         vec:div(root:size())

         -- Variation 2: average leaf nodes
         local vec = torch.Tensor():resizeAs(root.value):fill(0)
         local leaves = leaf_tree(corpus_test[s])
         for _,l in ipairs(leaves) do
            vec:add(l.value)
         end
         vec:div(#leaves)

         vecs_test[s] = vec
         xlua.progress(s, #corpus_test)
      end

      -- Save in Torch serialized format for fast querying
      torch.save('corpus_vecs.th', vecs)

      -- Save in numpy format for Python classifier tests
      local py = require('fb.python')
      py.exec([=[
import numpy as np
f = open('corpus_vecs.npy', 'w+')
np.save(f, vecs)
f = open('corpus_vecs_test.npy', 'w+')
np.save(f, vecs_test)
f = open('corpus_labels.npy', 'w+')
np.save(f, labels)
f = open('corpus_labels_test.npy', 'w+')
np.save(f, labels_test)
      ]=], {vecs = vecs, vecs_test = vecs_test, labels = labels, labels_test = labels_test})
   end

   local q = leaf_tree_str(params.q, vocab, emb)
   local vec = model:forward(q).value

   local count = 0
   local sorted = {}
   local last_best = -math.huge
   while #sorted < 5 do
      local best_cost = math.huge
      local best_d = 0
      local best_s = 0
      for d, doc in ipairs(vecs) do
         for s, svec in ipairs(doc) do
            --local cost = torch.dot(svec, vec) / (torch.norm(svec, 2) * torch.norm(vec, 2))
            local cost = torch.norm(svec - vec, 2)
            if cost < best_cost and cost > last_best then
               best_cost = cost
               best_d = d
               best_s = s
            end
         end
      end

      sorted[count] = {best_d, best_s, best_cost}
      last_best = best_cost
      count = count + 1
   end

   for i,v in ipairs(sorted) do
      print(i, v[3], v[2], labels[v[1]])
   end
end
