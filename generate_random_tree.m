function S=generate_random_tree(n,p)

if p==1
    S=ones(n);
    return;
end

S=zeros(n);

while sum(S(:))<numel(S)*p
    r=randperm(n);
    S(sub2ind(size(S),(1:size(S,1))', r'))=1;
end



end