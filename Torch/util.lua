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
   local nSent = 0
   for i = 1, files:size() do
      local fpath = files:filename(i)
      local fdir, fname = string.match(fpath, "data/(.+)/(%d+).txt")
      corpus[i] = {}
      local sentences = read_sentences(fpath, vocab, emb)
      for s = 1, #sentences do
         nSent = nSent + 1
         labels[nSent] = {fdir, fname}
         corpus[i][s] = emb:forward(sentences[s]):clone() -- maybe not best way...
      end
   end
   return corpus, labels
end

function load_corpus_file(fname, vocab, emb)
   require('json')
   local uris = {}
   local questions = {}
   local answers = {}
   local maincats = {}

   local file = assert(io.open(fname, 'r'))
   for line in file:lines() do
      local data = json.decode(line)
      table.insert(uris, data[1])

      -- Question and answer must be encoded with embeddings
      table.insert(questions, string_embed(data[2], vocab, emb):clone())
      table.insert(answers, string_embed(data[3], vocab, emb):clone())


      table.insert(maincats, data[4])
   end
   return uris, questions, answers, maincats
end

-- most similar vector using distance measure
function most_similar(vec, emb)
   local function dist(v1, v2)
      -- euclidean
      return torch.norm(v1 - v2, 2)

      -- cosine
      --return torch.dot(v1, v2) / (torch.norm(v1, 2) * torch.norm(v2, 2))
   end

   local best_vec = nil
   local best_idx = nil
   local best_dist = math.huge
   for i = 1, emb.weight:size(1) do
      local v = emb:forward(torch.IntTensor(1):fill(i))
      local d = dist(vec, v)
      print(i, d)
      if d < best_dist then
         best_vec = v
         best_idx = i
         best_dist = d
      end
   end

   return best_vec, best_idx
end
