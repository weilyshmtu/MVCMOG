function [S] = initialize_S(X, n_samples, n_views, k)
sumS = 0;
for v= 1 : n_views
        W{v} = constructW_PKN(X{v}',k, 1);
        sumS = sumS + W{v};
end
S = sumS/n_views;
S= S./repmat(sum(S, 2)+eps,1, n_samples);