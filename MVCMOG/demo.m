clear
% clc
close all
addpath('.\algs\')
addpath('.\metric\')
addpath('..\..\..\DataSets\Multi-view datasets\')


dataset = ["3Sources_1", "MSRC-v1","BBCsport", "WebKB" ,"COIL20","100Leaves","MNIST-10k","NUS-WIDE_11280"];% 
rng(7)


alpha_set = [1, 2, 5, 10, 50, 100, 200, 500];
beta_set =  [1, 2, 5, 10, 50, 100, 200, 500];

for d_ind = 1:length(dataset)
    data = dataset(d_ind);
    load(data)
    fprintf("Experiments on %s dataset.\n", data)

    if exist('y', 'var')
        n_clusters = max(unique(y));
        labels = y;
    else
        n_clusters = max(unique(Y));
        labels = Y;
    end

    [n, m] = size(X);
    if n > m
        n_views = n;
    else
        n_views = m;
    end
    
    n_samples = size(X{1}, 1);

    % normalize data if required
    flag = 1;
    X = data_normalize(X, flag);
    
    for v= 1 : n_views
        D_X{v} = EuDist2(X{v}, X{v}, 2);
        XXt{v} = X{v} * X{v}';
    end

    opts = [];
    opts.n_samples = n_samples;
    opts.n_views = n_views;
    opts.n_clusters = n_clusters;
    opts.neighbor_size =  20; 
    opts.cutflag = true;
    [S] = initialize_S(X, n_samples, n_views, opts.neighbor_size);
    S_init = 0.5*(S + S');
    D_S = diag(sum(S, 2));
    L_S = D_S - S;
    [F_init, ~] = eigs(L_S, n_clusters, "smallestreal");
   
    for alpha_ind = 1: length(alpha_set)
        for beta_ind = 1:length(beta_set)
            alpha = 1 * alpha_set(9);
            beta =  beta_set(5);
            [S, C, p, q, obj] = mvcmog2(XXt, D_X, S_init, F_init, alpha, beta, opts);
            predy = SpectralClustering(S, n_clusters);
            curr_result = Clustering8Measure(labels, predy);

            acc1(alpha_ind,beta_ind) = curr_result(1)
            nmi1(alpha_ind,beta_ind) = curr_result(2)
            ari1(alpha_ind,beta_ind) = curr_result(3)
            fscore1(alpha_ind,beta_ind) = curr_result(4)
            purity1(alpha_ind,beta_ind) = curr_result(5);
            precision1(alpha_ind,beta_ind) = curr_result(6);
            recall1(alpha_ind,beta_ind) = curr_result(7);
            obj_array{alpha_ind,beta_ind} = obj;

          

        end
        % figure
        % for b_i = 1 : length(beta_set)
        %     subplot(1, length(beta_set), b_i)
        %     plot( 1:length(obj_array{alpha_ind, b_i}), obj_array{alpha_ind, b_i})
        % end
    end
    cd("results")
    if opts.cutflag
        result_name = data + "_results_knn_newp_" + num2str(opts.neighbor_size);
    else  
        result_name = data + "_results_uncut";
    end
    save(result_name, "acc1" , "nmi1", "ari1", "fscore1", "purity1", ...
        "precision1", "recall1", "obj_array" );
    cd ..
    clear acc1 nmi1 ari1 fscore1 purity1 precision1 recall1 obj_array
end






function normX = data_normalize(X, flag)
n = size(X{1},1) ;
if flag == 1
    for i=1:length(X)
        tX = X{i};
        for j=1:n
            tX(j,:) = tX(j,:)/(eps + norm(tX(j,:),2));
        end
        normX{i} = tX;
    end
end 
end