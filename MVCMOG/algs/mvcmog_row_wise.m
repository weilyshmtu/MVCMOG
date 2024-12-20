function [S, C, obj, curr_result, residual_S] = mvcmog_ori(XXt, D_X, alpha, beta, opts)

n_views = opts.n_views;
n_samples = opts.n_samples;
n_clusters = opts.n_clusters;
obj_flag = opts.obj_flag;

cut_flag = opts.cut_flag;
result_required = opts.results_required;
labels = opts.labels;

S = opts.S_init;
C = opts.C_init;
F = opts.F_init;

I = eye(n_samples);
p = 0.5*ones(1, n_views);
q = 0.5;
w = ones(1, n_views)/n_views;

L_S = diag(sum(S, 2)) - S;
SS = S*S;
L_SS = diag(sum(SS, 2)) - SS;

D_F = EuDist2(F, F, 2);



tol = 1e-5;
iter = 0;
maxIter = 50;


while iter < maxIter
    iter = iter + 1;
    G_S = 0;
    N_S = 0; 
    % update Cv    
    for v = 1 : n_views
        C_v_pre = C{v};
        G_C_v =  alpha*w(v)*(q^2*L_S + (1-q)^2*L_SS) - (1-p(v))^2*XXt{v} ;
        N_C_v = p(v)^2*D_X{v};
        [C{v}] = fun_alm_row_wise(G_C_v, N_C_v, C_v_pre, cut_flag);
        % C{v} = fun_alm_matrix(G_C_v, N_C_v, C_v_pre, cut_flag);
        CCt{v} = C{v}*C{v}';
        D_C{v} = EuDist2(C{v}, C{v},2);

        G_S = G_S + w(v)*CCt{v};
        N_S = N_S + w(v)*D_C{v};
    end
    
    % update S

    S_pre = S;
    G_S = -alpha*(1-q)^2*G_S;
    N_S = alpha*q^2*N_S + beta*D_F;
    S = fun_alm_row_wise(G_S, N_S, S_pre, cut_flag);
    % S = fun_alm_matrix(G_S, N_S, S_pre, cut_flag);

    if result_required
        predy = SpectralClustering(S, n_clusters);
        % predy = conncomp(graph(S));
        curr_result(iter,:) = Clustering8Measure(labels, predy);
        residual_S(iter) = norm(S - S_pre,"fro");
    end
    SS = S*S;

    % update F
    L_S = diag(sum(S, 2)) - S;
    [F, ev] = eigs(L_S, n_clusters, "smallestreal");
    F = real(F);
    D_F = EuDist2(F, F, 2);

    % update p
    for v =1:n_views
        J11(v) = sum(sum(D_X{v}.*C{v}));
        J12(v) = sum(sum(D_X{v}.*CCt{v}));

        J21(v) = sum(sum(D_C{v}.*S));
        J22(v) = sum(sum(D_C{v}.*SS));

    end
    p = J12./(J11+J12);
    q = sum(w.^2.*J22)/(sum(w.^2.*J21) + sum(w.^2.*J22));

    % update w
    J = 1./(J21*q^2 + J22*(1-q)^2);
    w = J./sum(J);



    % update OBJ
    if obj_flag
        obj1 = sum((p.^2).*(J11) + ((1-p).^2).*(J12));
        obj2 = sum(alpha*w.*(q^2*(J21) + (1-q)^2*(J22)));
        obj3 = beta*sum(sum(D_F.*S));
        obj_current = obj1 + obj2  + obj3;
       
        obj(iter) = obj_current;
        if iter > 1
            if abs(obj_current - obj(iter - 1)) <  tol && iter > 30
                break;
            end
        end
        if mod(iter, 10) == 0 
            fprintf("The obj is %d.\n", obj(iter))
        end
    
        
       
    end
end

end


function [S] = fun_alm_row_wise(G, N, S_pre, cutflag)
    % min Tr(S'*G*S) + Tr(S'*N)
    % s.t. Se = e, S>=0, S = S'
    % in column-wise
    n_samples = size(S_pre,1);
    S = zeros(n_samples, n_samples);
    n = size(G, 1);
     for i = 1: n
        index = 1:n;
        if cutflag
            index = find(S_pre(i,:)>0);
        end
        A = G(index,index);
        b = N(i,index);
        S(i,index) = fun_alm(A,b);
     end
    S = 0.5*(S + S');
end


function [s, obj] = fun_alm(A,b)
    if size(b,1) == 1
        b = b';
    end
    
    % initialize
    rho = 1.5;
    mu = 30;
    n = size(A,1);
    q = ones(n,1);
    t = ones(n,1)/n;
    tol = 1e-5;

    % obj = [v'*A*v-v'*b];
    iter = 0;
    while iter < 30
        
    
        % update s
        mm = t - q/mu - (A*t + b)/mu;
        s = EProjSimplex_new(mm);

        % update t
        t = s + q/mu - A'*s/mu;
    
        % update alpha and mu
        q = q + mu*(s - t);
        mu = rho * mu;
        iter = iter + 1;

        % obj = [obj;v'*A*v-v'*b];
        % if abs(obj(end) - obj(end-1)) < tol
        %     break;
        % end
    end
    s = s';
end
