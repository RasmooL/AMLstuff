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

function leaf_tree(str, vocab, emb)
   local embs = string_embed(str, vocab, emb)
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
   local file = io.open(fname, 'r')
   while true do
      local line = file:read()
      if line == nil then break end
      local tokens = stringx.split(line)
      local len = #tokens
      local sent = torch.IntTensor(len)
      for i = 1, len do
         local token = tokens[i]
         sent[i] = vocab:index(token)
      end
      sentences[#sentences + 1] = sent
   end
   return sentences
end

function load_corpus(dir, vocab, emb)
   require('torchx')
   local corpus = {}
   local files = paths.indexdir(dir, 'txt')
   for i = 1, files:size() do
      local sentences = read_sentences(files:filename(i), vocab, emb)
      for s = 1, #sentences do
         print(files:filename(i))
         print(sentences[s])
         corpus[i] = emb:forward(sentences[s])
      end
   end
end
