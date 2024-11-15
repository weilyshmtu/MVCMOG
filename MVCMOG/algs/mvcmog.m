function [S, C, p, q, obj] = mvcmog(XXt, D_X, S_init, F_init, alpha, beta, opts)

n_views = opts.n_views;
n_samples = opts.n_samples;
n_clusters = opts.n_clusters;
neighbor_size = opts.neighbor_size;
cutflag = opts.cutflag;

I = eye(n_samples);
p = ones(1, n_views)/n;
q = 0.5;
gamma = ones(1, n_views)/n_views;

S = S_init;
for v = 1: n_views
    C{v} = S;
end
L_S = diag(sum(S, 2)) - S;
SS = S*S;
L_SS = diag(sum(SS, 2)) - SS;

F = F_init;
D_F = EuDist2(F, F, 2);

tol = 1e-5;
iter = 0;
maxIter = 30;

while iter < maxIter
    iter = iter + 1;

    % update Cv
    for v = 1 : n_views
        A = alpha*gamma(v)*(q^2*L_S + (1-q)^2*L_SS) - (1-p(v))^2*XXt{v} ;
        B = p(v)^2*D_X{v};
        initC_v = C{v};
        [C{v}, obj_t] = fun_alm(A, B, initC_v, cut_flag);
        CC{v} = C{v}*C{v}';
        D_C{v} = EuDist2(C{v}, C{v},2);
    end
    
    % update S
    A = 0;
    D_C_total = 0;
    for v = 1:n_views
        A = gamma(v)*CC{v} + A;
        D_C_total = D_C_total + gamma(v)*D_C{v};
    end
    A = -alpha*(1-q)^2*A;
    B = alpha*q^2*D_C_total + beta*D_F;
    init_S = S;
    [S, obj_t] = fun_alm(A, B, init_S, cut_flag);
    SS = S*S;
    L_SS = diag(sum(SS, 2)) - SS;
    
    % update F
    L_S = diag(sum(S, 2)) - S;
    [F, ~] = eigs(L_S, n_clusters, "smallestreal");
    D_F = EuDist2(F, F, 2);
    
    % update p
    for v =1:n_views
        J1(v) = sum(sum(D_X{v}.*C{v}));
        J2(v) = sum(sum(D_X{v}.*CC{v}));

        J11(v) = sum(sum(D_C{v}.*S));
        J22(v) = sum(sum(D_C{v}.*SS));
    end
    p = J2./(J1+J2);
    q = sum(J22)/(sum(J11) + sum(J22));

    % update gamma
    J = 1./(J11 + J22);
    gamma = J./sum(J);

    % update OBJ
    obj1 = sum((p.^2).*(J1) + ((1-p).^2).*(J2));
    obj2 = sum(alpha*gamma.*(q^2*(J11) + (1-q)^2*(J22)));
    obj3 = beta*sum(sum(D_F.*S));
    obj_current = obj1 + obj2 + obj3;
       obj(iter) = obj_current;
    
    if iter > 1
        if abs(obj_a - obj(iter - 1)) <  1e-5 && iter > 20
            break;
        else
            obj(iter) = obj_current;
        end
    end
    if mod(iter, 10) == 0 
        fprintf("The obj is %d.\n", obj(iter))
    end
end

end


function [S, obj] = fun_alm(G, N, S_pre, cutflag)

    rho = 1.5;
    mu = 30;
    n = size(G, 1);
    T = ones(n,n)/n;
    Q = ones(n, n);

    iter = 0;
    mIter = 30;
    obj = 0;
    while iter < mIter 
        for i = 1:n
            if cutflag
                index = find(S_pre(:,i)>0);
            else
                index = 1:n;
            end

            q = Q(index,i);
            t_i = T(index,i);
            n_i = N(index,i);
            
            mm_i = (G(index, index)*t_i + n_i)/mu + q/mu - t_i;
            S(index, i) = EProjSimplex_new(-mm_i);
        end

        T = S + Q/mu - G'*S/mu;
        T = (T + T')/2;
        
        Q = Q + mu*(S - T);
        mu = min(rho*mu, 1e30);
        iter = iter + 1;

        obj = [obj;trace(S'*G*S) + sum(sum(N.*S))];
    end
end
