include('RAE.lua')
include('util.lua')
include('Vocab.lua')

model = RAE()

-- load vocab & word embeddings
local vocab = Vocab('vocab.th')
vocab:add_unk_token()
local emb_vecs = torch.load('vectors.50d.th')
local emb = nn.LookupTable(emb_vecs:size(1), emb_vecs:size(2))
emb.weight:copy(emb_vecs)

local tree = model:createTree(leaf_tree_str("the cat sat", vocab, emb))
