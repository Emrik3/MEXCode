%COEFFSPOLM15plus  MATLAB code fragments 4.1 and 4.2 from [1]  
%   for computing the coefficients of the evaluation formulas
%   (improvements from [2])
%   y0=A^2*(sqrt(c8)*A^2+c7/(2*sqrt(c8))*A);
%   y1 = sum(c.*A.^([2:8]'));
%   y2 = (y1+d2*A^2+d1*A)*(y1+e0*y0+e1*A)+f0*y1+g0*y0+h2*A^2+h1*A+h0I;
%   where y2 is a polynomial of degree 16 and 
%   y2 = sum_{i=0}^{15}=(bi*A^i) + c8^2 A^16
%
%   References: 
%
%   [1] J. Sastre, J. Ibanez, Efficient Evaluation of Matrix Polynomials
%       beyond the Paterson–Stockmeyer Method. Mathematics 2021, 9, 1600.
%       https://doi.org/10.3390/math9141600.
%   [2] J. Sastre, Efficient evaluation of matrix polynomials, Linear 
%       Algebra Appl., 539, 2018, 229-250.
%   [3] J. Sastre, J. Ibanez, E. Defez, Boosting the computation of
%       the matrix exponential, Appl. Math. Comput. 340 (2019), 206-220.

%
%   Author: Jorge Sastre
%   Revised version: 2020/10/31.
%
%   Group of High Performance Scientific Computing (HiPerSC)
%   Universitat Politecnica de Valencia (Spain)
%   http://hipersc.blogs.upv.es 

% MATLAB code fragment 4.1 solves coefficient c8 of the system of 
% equations (29) from [1] for general coefficients bi

syms A c2 c3 c4 c5 c6 c7 c8 d0 d1 d2  e0 e1 f0 g0 h2 h1 h0I

syms b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16

c=[c2; c3; c4; c5; c6; c7; c8];
%General Solution
b=[b16; b15; b14; b13; b12; b11; b10; b9; b8; b7; b6; b5; b4; b3; b2; b1; b0]; 

%b=1./factorial(sym(16:-1:0).'); %For obtaining the coefficients for
%evaluating the matrix exponential Taylor polynomial of order m=15+ from [Sec. 3.2, 3]

%b=1./factorial(2*sym(18:-1:2).').*(-1).^((18:-1:2).') %For obtaining 
%the coefficients for evaluating the matrix cosine Taylor polynomial of order m=17+ 
%from [Ex. 4.1, 3]

y0=A^2*(sqrt(c8)*A^2+c7/(2*sqrt(c8))*A); %solution with positive sign in (28) of [3]

y1=sum(c.*A.^([2:8]'));

y2=(y1+d2*A^2+d1*A)*(y1+e0*y0+e1*A)+f0*y1+g0*y0+h2*A^2+h1*A+h0I;

[cy2,a1]=coeffs(y2,A);

cy2=cy2.';

v=[cy2 b a1.'] %Shows the coefficients of each power of A

cy2=cy2(2:end)-b(2:end); %System of equations

c7s=solve(cy2(1),c7,'ReturnConditions',true); %c7s=f(c8,bi)

c7s.conditions %c8 ~= 0

c7s=c7s.c7;

cy2=subs(cy2,c7,c7s);

c6s=solve(cy2(2), c6); %c6s depends on c8 bi

cy2=subs(cy2,c6,c6s);

c5s=solve(cy2(3), c5); %c5s depends on c8 bi

cy2=simplify(subs(cy2,c5,c5s));

symvar(cy2(4))  %cy2(4) depends on c8, c4, e0 bi

e0s=solve(cy2(4), e0); 

cy2=simplify(subs(cy2,e0,e0s));

symvar(cy2(5))  %cy2(5) depends on c8, c3, c4, bi

c3s=solve(cy2(5), c3); 

cy2=simplify(subs(cy2,c3,c3s));

symvar(cy2(6)) %depends only on c8, c2, d2, bi

d2s=solve(cy2(6), d2);

cy2=simplify(subs(cy2,d2,d2s));

symvar(cy2(7)) %cy2(7) depends only on c8, d1, e1, bi

d1s=solve(cy2(7), d1);

cy2=simplify(subs(cy2,d1,d1s));

symvar(cy2(8)) %cy2(8) depends only on c8, c4, f0, bi

f0s=solve(cy2(8), f0);

cy2=simplify(subs(cy2,f0,f0s));

symvar(cy2(9)) %cy2(9) depends only on c8, b7, b8,...,b15

c8s=solve(cy2(9), c8)

%digits(32);c8s=vpasolve(cy2(9), c8) %For solving c8 if bi are numbers setting 
%before the desired decimal digits

% MATLAB code fragment 4.2: solves coefficient c2 of the
% system of equations (29) for general coefficients bi by
% using the solutions for coefficient c8 obtained using the
% MATLAB piece of code 4.1

symvar(cy2(10)) %cy2(10) depends on c8, c2, c4, bi 

c4s=solve(cy2(10), c4) %It gives two solutions depending on c8, c2, bi

cy2=simplify(subs(cy2,c4,c4s(1))) %change c4s(1) for c4s(2) for more solutions

symvar(cy2(11)) %cy2(11) depends on c8, c2, e1, bi

e1s=solve(cy2(11), e1)

cy2=simplify(subs(cy2,e1,e1s))

symvar(cy2(12)) %cy2(12) depends on c8, c2, g0, bi

g0s=solve(cy2(12), g0,'ReturnConditions',true)

g0s.conditions %conditions for the existence of solutions:
%3*b15^2 ~= 8*b14*c8^2 &
%27*b15^6*c8^(45/2) + 576*b14^2*b15^2*c8^(53/2) ~=
%512*b14^3*c8^(57/2) + 216*b14*b15^4*c8^(49/2) &
%c8 ~= 0

g0s=g0s.g0;

cy2=simplify(subs(cy2,g0,g0s))

symvar(cy2(13)) %cy2(13) depends on c8, c2, bi