require('pl')
require('torch')
require('nn')

include('Node.lua')

function string_embed(str, vocab, emb)
   local tokens = stringx.split(str)
   local len = #tokens
   local sent = torch.IntTensor(len)
   for i = 1, len do
      local token = tokens[i]
      sent[i] = vocab:index(token)
   end
   return emb:forward(sent)
end

function leaf_tree_str(str, vocab, emb)
   local embs = string_embed(str, vocab, emb)
   local leaves = {}
   for i = 1, embs:size(1) do
      local node = Node()
      node.value = embs[i]
      node.leafid = i
      leaves[i] = node
   end
   return leaves
end

function leaf_tree(embs)
   local leaves = {}
   for i = 1, embs:size(1) do
      local node = Node()
      node.value = embs[i]
      leaves[i] = node
   end
   return leaves
end

function read_sentences(fname, vocab, emb)
   local sentences = {}
   local file = assert(io.open(fname, 'r'))
   while true do
      local line = file:read()
      if line == nil then break end
      local tokens = stringx.split(line)
      local len = #tokens
      if len > 2 then -- Discard sentences under length 3
         local sent = torch.IntTensor(len)
         for i = 1, len do
            local token = tokens[i]
            sent[i] = vocab:index(token)
         end
         sentences[#sentences + 1] = sent
      end
   end
   file:close()
   return sentences
end

function load_corpus(dir, vocab, emb)
   require('torchx')
   local corpus = {} -- dim 1: file, dim 2: sentence, dim 3: vector
   local labels = {} -- dim 1: {folder, file name}
   local files = paths.indexdir(dir, 'txt')
   for i = 1, files:size() do
      local fpath = files:filename(i)
      local fdir, fname = string.match(fpath, "data/(.+)/(%d+).txt")
      labels[i] = {fdir, fname}
      corpus[i] = {}
      local sentences = read_sentences(fpath, vocab, emb)
      for s = 1, #sentences do
         corpus[i][s] = emb:forward(sentences[s]):clone() -- maybe not best way...
      end
   end
   return corpus, labels
end
