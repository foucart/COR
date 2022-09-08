%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reproducible MATLAB file accompanying the paper
%   INSTANCES OF COMPUTATIONAL OPTIMAL RECOVERY:
%          REFINED APPROXIMABILITY MODELS        
% by S. Foucart.
% Written by S. Foucart in March 2020
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% CVX [2], Chebfun [3] are required to run this reproducible

%% Case 1: Q is the normalized integral,
% while the observation functionals are evaluations
% at points in [-1,-0.25] U [0.25, 1].
% The space V consists of polynomials of degree < n.

clear all; clc;
cvx_quiet true

%% Case 1a: bounded approximability model of the second type

m_half = 25;
m = 2*m_half;    % # of observational data
ev_pts = [-1+0.75*rand(1,m_half), 1-0.75*rand(1,m_half)];
n = 70;          % # of model parameters (n>m: overparametrization)
epsilon = 0.1;
kappa = 100;
% define auxiliary quantities
b = zeros(n,1);
C = zeros(n,m);
for j=1:n
    Tj = chebpoly(j-1);
    b(j) = sum(Tj)/2;
    C(j,:) = Tj(ev_pts);
end

% perform the semidefinite minimization
cvx_solver mosek   % remove if no mosek license
cvx_begin
variable a(m)
variable s(m)
variable u(n)
minimize epsilon*sum(s) + kappa*u(1)
subject to
toeplitz(u+C*a-b) == semidefinite(n);
toeplitz(u-C*a+b) == semidefinite(n);
s+a >= 0;
s-a >= 0;
cvx_end
min_typeII = cvx_optval;
a_typeII = a;

%% Case 1b: bounded approximability model of the first type
% we keep the same parameters (m,n,epsilon,kappa) as above

N = 300;     % size of the truncated SDP
% define auxiliary quantities
b_ext = zeros(N,1);
C_ext = zeros(N,m);
for k=1:N
    Tk = chebpoly(k-1);
    b_ext(k) = sum(Tk)/2;
    C_ext(k,:) = Tk(ev_pts);
end

% perform the semidefinite minimization
cvx_solver mosek   % remove if no mosek license
cvx_begin
variable a(m)
variable wp(N)
variable wm(N)
variable zp(N)
variable zm(N)
minimize epsilon*(wp(1)+wm(1)) + kappa*(zp(1)+zm(1))
subject to
wp - wm + zp - zm == b_ext - C_ext*a;
wp(1:n) - wm(1:n) == 0;
toeplitz(wp) == semidefinite(N);
toeplitz(wm) == semidefinite(N);
toeplitz(zp) == semidefinite(N);
toeplitz(zm) == semidefinite(N);
cvx_end
min_typeI = cvx_optval;
a_typeI = a;

%% note: the optimal weights are not that close to one another
norm(a_typeII - a_typeI)

%% Case 2: Q is now the evaluation at 0,
% while the observation functionals are still evaluations
% at points in [-1,-0.25] U [0.25, 1].
% The space V still consists of polynomials of degree < n.
% Note: overparametrization seems insignificant in this case. 

clear all; clc;
cvx_quiet true

%% Case 2a: bounded approximability model of the second type

m_half = 25;
m = 2*m_half;    % # of observational data
ev_pts = [-1+0.75*rand(1,m_half), 1-0.75*rand(1,m_half)];
n = 15;          % # of model parameters (n<m: underparametrization)
epsilon = 0.1;
kappa = 10;
% define auxiliary quantities
b = zeros(n,1);
C = zeros(n,m);
for j=1:n
    Tj = chebpoly(j-1);
    b(j) = Tj(0);
    C(j,:) = Tj(ev_pts);
end

% perform the semidefinite minimization
cvx_solver mosek   % remove if no mosek license
cvx_begin
variable a(m)
variable s(m)
variable u(n)
minimize epsilon*sum(s) + kappa*u(1)
subject to
toeplitz(u+C*a-b) == semidefinite(n);
toeplitz(u-C*a+b) == semidefinite(n);
s+a >= 0;
s-a >= 0;
cvx_end
min_typeII = cvx_optval;
a_typeII = a;
% For kappa = Inf, the minimizer is n-sparse; what about here? 
length(find(abs(a_typeII)>1e-4))

%% Case 2b: bounded approximability model of the first type
% we keep the same parameters (m,n,epsilon,kappa) as above

% first, a lower bound by semidefinite programming
N = 300;     % size of the truncated SDP
% define auxiliary quantities
b_ext = zeros(N,1);
C_ext = zeros(N,m);
for k=1:N
    Tk = chebpoly(k-1);
    b_ext(k) = Tk(0);
    C_ext(k,:) = Tk(ev_pts);
end

% perform the semidefinite minimization
cvx_solver mosek   % remove if no mosek license
cvx_begin
variable a(m)
variable wp(N)
variable wm(N)
variable zp(N)
variable zm(N)
minimize epsilon*(wp(1)+wm(1)) + kappa*(zp(1)+zm(1))
subject to
wp - wm + zp - zm == b_ext - C_ext*a;
wp(1:n)-wm(1:n) == 0;
toeplitz(wp) == semidefinite(N);
toeplitz(wm) == semidefinite(N);
toeplitz(zp) == semidefinite(N);
toeplitz(zm) == semidefinite(N);
cvx_end
min_typeI_LB = cvx_optval;
a_typeI_LB = a;

% second, an upper bound by linear programming
K = 300;                    % size of the grid
grid = linspace(-1,1,K);    % equispaced grid
% define auxiliary quantities
D = zeros(n,K);
for j=1:n
    Tj = chebpoly(j-1);
    D(j,:) = Tj(grid);
end

% perform the linear minimization
cvx_solver gurobi   % remove if no gurobi license
cvx_begin
variable a(m)
variable u(1+m+K)
variable v(1+m+K)
variable r(1+m+K)
variable s(1+m+K)
minimize epsilon*sum(r) + kappa*sum(s)
subject to
u + v == [1;-a;zeros(K,1)];
[b,C,D]*u == 0;
r + u >= 0;
r - u >= 0;
s + v >= 0;
s - v >= 0;
cvx_end
min_typeI_UB = cvx_optval;
a_typeI_UB = a;

% note: the minimal values of the two programs are close,
% the optimal coefficient vectors are often close (not always).
norm(a_typeI_LB - a_typeI_UB)


%% References

% 1. S. Foucart,
% "Instances of computational optimal recovery:
%  refined approximability models",
% Preprint.

% 2. CVX Research, Inc., 
% "CVX: MATLAB software for disciplined convex programming"
% version 2.1, 2014, http://cvxr.com/cvx.

% 3. L. N. Trefethen et al., 
% "Chebfun Version 5, The Chebfun Development Team", 2014,
% http://www.chebfun.org.