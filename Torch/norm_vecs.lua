require('torch')

local emb_vecs = torch.load('vectors_glove.100d.th')
for i = 1, emb_vecs:size(1) do
   norm = torch.norm(emb_vecs[i], 2)
   emb_vecs[i]:div(norm)
end

torch.save('vectors_glove_norm.100d.th', emb_vecs)
