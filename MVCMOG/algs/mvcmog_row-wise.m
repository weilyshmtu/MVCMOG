function [S, C, p, q, obj] = mvcmog(XXt, D_X, S_init, F_init, alpha, beta, opts)

n_views = opts.n_views;
n_samples = opts.n_samples;
n_clusters = opts.n_clusters;
neighbor_size = opts.neighbor_size;
cutflag = opts.cutflag;

I = eye(n_samples);
p = ones(1, n_views)/n_views;
q = 0.5;
gamma = ones(1, n_views)/n_views;

S = S_init;
for v = 1: n_views
    C{v} = S;
end
L_S = diag(sum(S, 2)) - S;
SS = S*S;

F = F_init;
D_F = EuDist2(F, F, 2);

tol = 1e-5;
iter = 0;
maxIter = 30;

while iter < maxIter
    iter = iter + 1;

    % update Cv
    L_SS = diag(sum(SS, 2)) - SS;
    
    for v = 1 : n_views
        A = alpha*gamma(v)*(q^2*L_S + (1-q)^2*L_SS) - (1-p(v))^2*XXt{v} ;
        initC_v = C{v};
        for i = 1: n_samples
            index = 1:n_samples;
            if cutflag
                index = find(initC_v(i,:)>0);
            end
            b = -p(v)^2*D_X{v}(i,index);
            [C_v(i, index)] = fun_alm(A(index, index), b,n_samples);
        end
        C{v} = 0.5*(C_v + C_v');
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
    init_S = S;
    for i = 1: n_samples
        index = 1:n_samples;
        if cutflag
            index = find(init_S(i,:)>0);
        end
        b = -alpha*q^2*D_C_total(i,index) - beta*D_F(i,index);
        [S(i, index)] = fun_alm(A(index, index), b,n_samples);
    end
    S = 0.5*(S + S');
    SS = S*S;

    % update F
    L_S = diag(sum(S, 2)) - S;
    [F, ~] = eigs(L_S, n_clusters, "smallestreal");
    F = real(F);
    D_F = EuDist2(F, F, 2);
    % update p
    for v =1:n_views
        J1(v) = sum(sum(D_X{v}.*C{v}));
        J2(v) = sum(sum(D_X{v}.*CC{v}));

        J11(v) = sum(sum(D_C{v}.*S));
        J22(v) = sum(sum(D_C{v}.*SS));
    end
     p = J2./(J1+J2);
    q = sum(gamma.^2.*J22)/(sum(gamma.^2.*J11) + sum(gamma.^2.*J22));

    % update gamma
    J = 1./(q^2*J11 + (1-q)^2*J22);
    gamma = J./sum(J);



    % update OBJ
    obj1 = sum((p.^2).*(J1) + ((1-p).^2).*(J2));
    obj2 = sum(alpha*gamma.*(q^2*(J11) + (1-q)^2*(J22)));
    obj3 = beta*sum(sum(D_F.*S));
    obj_current = obj1 + obj2 + obj3;
   
    
    % if iter > 1
    %     if abs(obj_current - obj(end)) < tol
    %         break;
    %     end
    % end

    obj(iter) = obj_current;
    if mod(iter, 5) == 0 
        fprintf("The obj is %d.\n", obj_current)
    end
end

end


function [v, obj] = fun_alm(A,b, n_samples)
    if size(b,1) == 1
        b = b';
    end
    
    % initialize
    rho = 1.5;
    mu = max(90);
    n = size(A,1);
    alpha = ones(n,1);
    v = ones(n,1)/n;
    tol = 1e-5;

    % obj = [v'*A*v-v'*b];
    iter = 0;
    while iter < 30
        % update z
        z = v - A'*v/mu + alpha/mu;
    
        % update v
        c = A*z-b;
        d = alpha/mu-z;
        mm = d + c/mu;
        v = EProjSimplex_new(-mm);
    
        % update alpha and mu
        alpha = alpha + mu*(v-z);
        mu = rho * mu;
        iter = iter + 1;

        % obj = [obj;v'*A*v-v'*b];
        % if abs(obj(end) - obj(end-1)) < tol
        %     break;
        % end
    end
    v = v';
end