require('torch')

local emb_vecs = torch.load('small_glove.th')
for i = 1, emb_vecs:size(1) do
   norm = torch.norm(emb_vecs[i], 2)
   emb_vecs[i]:div(norm)
end

torch.save('small_glove_norm.th', emb_vecs)
