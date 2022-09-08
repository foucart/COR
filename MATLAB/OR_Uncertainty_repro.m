%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reproducible MATLAB file accompanying the paper
%   INSTANCES OF COMPUTATIONAL OPTIMAL RECOVERY:
%         DEALING WITH OBSERVATION ERRORS      
% by M. Ettehad and S. Foucart.
% Written by M. Ettehad and S. Foucart in March 2020
% Updated in January 2021
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% CVX [2] is required to execute this reproducible
% Chebfun [3] is required for parts of this reproducible

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Part 1: Locally Optimal Recovery Maps
%  for a linear QoI mapping into \ell_{\infty}^K 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; clc;
cvx_quiet true

%% Chebyshev center relative to a polytope

% model set: n-dim cube described in inequality form as
% { f in R^n: Af<= b }
n = 20; 
N = 2*n;
A = [eye(n);-eye(n)];
b = ones(N,1);
% an object from this model set to be observed
f = (2*rand(n,1)-1);          
% create observational data
m = 10;                       % the number of data
L = randn(m,n);               % the observation operator
eta = 1e-1;                   % an uncertainty threshold
y = L*f+eta*(2*rand(m,1)-1);  % the observed data
% K-dim QoI to be recovered (here K = 2):
% averages over odd-indexed and even-indexed entries
Q = zeros(2,n);
Q(1,1:2:n) = ones(1,floor((n+1)/2))/floor((n+1)/2);
Q(2,2:2:n) = ones(1,floor(n/2))/floor(n/2);
% define auxiliary quantities
A_tilde = [A; +L; -L];
b_tilde = [b; +y+eta; -y+eta];

% perform the linear minimization
cvx_solver gurobi  % remove if no gurobi license
cvx_begin
variable z(2)
variable r
variable xp1(N+2*m)
variable xm1(N+2*m)
variable xp2(N+2*m)
variable xm2(N+2*m)
minimize r
subject to
A_tilde'*xp1 == +Q(1,:)';
A_tilde'*xm1 == -Q(1,:)';
A_tilde'*xp2 == +Q(2,:)';
A_tilde'*xm2 == -Q(2,:)';
xp1 >= 0;
xm1 >= 0;
xp2 >= 0;
xm2 >= 0;
b_tilde'*xp1 <= r+z(1);
b_tilde'*xm1 <= r-z(1);
b_tilde'*xp2 <= r+z(2);
b_tilde'*xm2 <= r-z(2);
cvx_end
Chebyshev_Radius = r;
Chebyshev_Center = z

%% Chebyshev center relative to the polynomial unit ball

% model set: unit ball of the n-dim polynomial space
n = 9;
% an object from this model set to be observed
f = chebfun('x^3');            
% create observational data
m = 12;
t = sort(2*rand(m,1)-1);       % the evaluation points
eta = 1e-1;                    % an uncertainty threshold 
y = f(t)+eta*(2*rand(m,1)-1);  % the observed data
% K-dim QoI to be recovered (here K = 2): Q(f)=[f(-1); f(1)];
% define auxiliary matrices C_k and B (B replaces the A_i's)
C1 = toeplitz((-1).^(0:n-1));
C2 = toeplitz(ones(1,n));
B = zeros(n,m);
for j=1:n
    Tj = chebpoly(j-1);
    B(j,:) = Tj(t); 
end

% perform the semidefinite minimization
cvx_solver mosek   % remove if no mosek license
cvx_begin
variable z(2)
variable r
variable xp1(n)
variable xm1(n)
variable xp2(n)
variable xm2(n)
variable up1(m)
variable um1(m)
variable up2(m)
variable um2(m)
variable vp1(m)
variable vm1(m)
variable vp2(m)
variable vm2(m)
minimize r
subject to
xp1(1) + (y+eta)'*up1 - (y-eta)'*vp1 <= r+z(1);
xm1(1) + (y+eta)'*um1 - (y-eta)'*vm1 <= r-z(1);
xp2(1) + (y+eta)'*up2 - (y-eta)'*vp2 <= r+z(2);
xm2(1) + (y+eta)'*um2 - (y-eta)'*vm2 <= r-z(2); 
up1 >= 0; vp1 >= 0;
um1 >= 0; vm1 >= 0;
up2 >= 0; vp2 >= 0;
um2 >= 0; vm2 >= 0;
toeplitz(xp1) - C1 - toeplitz(B*(vp1-up1)) == semidefinite(n);
toeplitz(xp1) + C1 + toeplitz(B*(vp1-up1)) == semidefinite(n);
toeplitz(xm1) + C1 - toeplitz(B*(vm1-um1)) == semidefinite(n);
toeplitz(xm1) - C1 + toeplitz(B*(vm1-um1)) == semidefinite(n);
toeplitz(xp2) - C2 - toeplitz(B*(vp2-up2)) == semidefinite(n);
toeplitz(xp2) + C2 + toeplitz(B*(vp2-up2)) == semidefinite(n);
toeplitz(xm2) + C2 - toeplitz(B*(vm2-um2)) == semidefinite(n);
toeplitz(xm2) - C2 + toeplitz(B*(vm2-um2)) == semidefinite(n);
cvx_end
Chebyshev_Radius = r;
Chebyshev_Center = z

%% An aside: verification of the SDP duality lemma

clear all;
cvx_quiet true
l = 4;  % dimension of the matrices
m = 6;  % number of matrices A_i
n = 8;  % number of matrices B_j
% create the symmetric matrices A_i, B_j, and C
A = zeros(l,l,m);
for i=1:m
    aux = randn(l,l);
    A(:,:,i) = aux+aux';
end
B = zeros(l,l,n);
for j=1:n
    aux = rand(l,l);
    B(:,:,j) = aux+aux';
end
C = rand(l,l);
C = C+C';
% choose the alpha_i and beta_j to ensure feasibility
aux = randn(l,l);
M_feas = aux'*aux;
aux = randn(l,l);
P_feas = aux'*aux;
alpha = zeros(m,1);
for i=1:m
    alpha(i) = trace(A(:,:,i)*(P_feas-M_feas)) + 1;
end
beta = zeros(n,1);
for j=1:n
   beta(j) = trace(B(:,:,j)*(P_feas+M_feas)); 
end
% the primal problem
cvx_begin
variable M(l,l) semidefinite
variable P(l,l) semidefinite
maximize trace(C*(P-M))
subject to
for i=1:m
    trace(A(:,:,i)*(P-M)) <= alpha(i);
end
for j=1:n
    trace(B(:,:,j)*(P+M)) == beta(j);
end
cvx_end
primal = cvx_optval;
% the dual program
cvx_begin
variable x(n)
variable u(m)
expression LHS(l,l)
LHS = zeros(l,l);
for j=1:n
    LHS = LHS + x(j)*B(:,:,j);
end
expression RHS(l,l)
RHS = C;
for i=1:m
    RHS = RHS - u(i)*A(:,:,i);
end
minimize sum(beta.*x)+sum(alpha.*u)
subject to 
LHS - RHS == semidefinite(l);
LHS + RHS == semidefinite(l);
u >= 0;
cvx_end
dual = cvx_optval;
% verify that the primal value and the dual values coincide
[primal dual]


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Part 2: Globally (Near)-Optimal Recovery Maps
%  relative to approximability models in C[-1,1]
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; clc;
cvx_quiet true

%% Estimation of a linear functional --- optimality
% guiding example: polynomials and Fourier observations
% the real-valued linear QoI is chosen as the evaluation at 1

% approximability set with V made of odd polynonials of degree <2n
n = 5;
epsilon = 0.1;
V = cell(n,1);
for j=1:n
    V{j} = chebpoly(2*j-1);
end
% uncertainty set
p = 2;
p_conj = p/(p-1);
eta = 0.05;
% observation functionals via the fist m sine coefficients
m = 10;
G = cell(m,1);
for i=1:m
    x = chebfun('x');
    G{i} = sin(i*pi*x); 
end
% define auxiliary quantities
M = zeros(n,m);
b = zeros(n,1);
for j=1:n
    b(j) = V{j}(1);
    for i=1:m
        M(j,i) = sum(V{j}*G{i});
    end
end
N = 350;  % size of the truncated SDP
Moments_Q = zeros(N,1);
Moments_G = zeros(N,m);
for k=1:N
    Tk = chebpoly(k-1);
    Moments_Q(k) = Tk(1);
    for i=1:m
        Moments_G(k,i) = sum(Tk*G{i});
    end
end

% perform the semidefinite minimization
cvx_solver mosek   % remove if no mosek license
cvx_begin
variable a(m)
variable zp(N)
variable zm(N)
minimize zp(1) + zm(1) + (eta/epsilon)*norm(a,p_conj)
subject to
M*a == b;
zp - zm == Moments_Q - Moments_G*a;
toeplitz(zp) == semidefinite(N);
toeplitz(zm) == semidefinite(N);
cvx_end
alpha_N = cvx_optval;
a_N = a;
% Compute the accuracy estimator delta
aux = chebfun('0');
for i=1:m
    aux = aux + a_N(i)*G{i};
end
delta_N = 1 + sum(abs(aux)) + ...
    (eta/epsilon)*norm(a_N,p_conj) - alpha_N;
% Bound on the relative error between alpha_opt and alpha_N
delta_N/alpha_N

%% Full approximation --- discontinuity of optimal weights

% use the same n, m, epsilon, eta as above
% create m evaluation points
ev_pts = 2*rand(1,m)-1;
% single out one of these evaluation points
k = round(m/3);
% define auxiliary quantities
M = zeros(n,m);
b = zeros(n,1);
for j=1:n
    Tj = chebpoly(j-1);
    M(j,:) = Tj(ev_pts);
    b(j) = Tj(ev_pts(k));
end
% perform the minimization for p=1 (p'=Inf)
p_conj = Inf;
cvx_begin
variable a(m)
minimize norm(a,1)+(eta/epsilon)*norm(a,p_conj)
subject to
M*a == b
cvx_end
min_p1 = cvx_optval;
% perform the minimization for p=2 (p'=2)
p_conj = 2;
cvx_begin
variable a(m)
minimize norm(a,1)+(eta/epsilon)*norm(a,p_conj)
subject to
M*a == b
cvx_end
min_p2 = cvx_optval;
% perform the minimization for p=Inf (p'=1)
p_conj = 1;
cvx_begin
variable a(m)
minimize norm(a,1)+(eta/epsilon)*norm(a,p_conj)
subject to
M*a == b
cvx_end
min_pInf = cvx_optval;
% verify that, in general, 
% min_p1 < min_p2 < min_pInf = 1+eta/epsilon
[min_p1 min_p2 min_pInf 1+eta/epsilon]

%% Full approximation --- construction of a near-optimal recovery map

% dimensions of the space V and of the superspace U
n = 5;
N = 25;
% define a basis for U
basisU = cell(1,N);
for j = 1:N
    basisU{j} = chebpoly(j-1);
end
% define the point zeta_1,...,zeta_K 
p_conj = 2;
K = 2*p_conj*N;
zeta = chebpts(K)';
% choose the evaluation points to be equispaced
m = 12;
eqpts = linspace(-1,1,m);
% define the matrix B
B = zeros(n,m);
for j=1:n
   B(j,:) = basisU{j}(eqpts); 
end
% define the matrix Z
Z = zeros(N,K);
for j=1:N
    Z(j,:) = basisU{j}(zeta);
end
% perform the optimization program
cvx_quiet true
cvx_begin
variable A(m,N)
variable D(m,K)
variable d
variable dd
minimize d + (eta/epsilon)*dd
subject to
B*A == [eye(n) zeros(n,N-n)];
A*Z <= D;
A*Z >= -D;
for k=1:K
   sum(D(:,k)) <= d;
   norm(D(:,k),p_conj) <= dd;
end
cvx_end
% return the functions phi_1,...,phi_m and visualize them
Phi = cell(1,m);
for i=1:m
    phi_i = chebfun(0);
    for j=1:N
        phi_i = phi_i + A(i,j)*basisU{j};
    end
    Phi{i} = phi_i;
end
figure(1);
plot(eqpts,zeros(1,m),'ko');
pbaspect([2 1 1])
title(strcat(sprintf('m=%d',m),sprintf(', n=%d',n),sprintf(' , N=%d',N),...
    sprintf(' , K=%d',K)),'FontSize',16)
hold on;
cmap = colormap(jet(m));
linestyles = {'-', '--', ':', '-.'};
for i=1:m
    plot(Phi{i},'LineStyle', linestyles{rem(i-1,numel(linestyles))+1},...
    'Color',cmap(i,:),'LineWidth',2); 
end


%% References

% 1. M. Ettehad and S. Foucart,
% "Instances of computational optimal recovery:
%  dealing with observation errors",
% Preprint.

% 2. CVX Research, Inc., 
% "CVX: MATLAB software for disciplined convex programming"
% version 2.1, 2014, http://cvxr.com/cvx.

% 3. L. N. Trefethen et al., 
% "Chebfun Version 5, The Chebfun Development Team", 2014,
% http://www.chebfun.org.