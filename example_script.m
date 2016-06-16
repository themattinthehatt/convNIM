% example script showing how to set up and fit convNIM models
%
% Note: to make an m x n array of uninitialized convNIM objects, use
%	fits(m, n) = convNIM([],[],[],[],[]);


%% set up empirical convolution kernel to initialize models
dt = 0.01;
x = dt:dt:(400*dt);
y = exp(-0.5*x);
y1 = exp(-4*x);
y2 = exp(-1*x);
y3 = y2-y1;


%% Initialize model

% first set of convolution kernel params
tent_spacing1 = 10;
num_lags1 = 20;
conv_params = convNIM.create_conv_params([num_lags1 1 1],... % 20 lags @ 10 bins per lag @ 10 ms per bin = 2 second conv kernel
							'boundary_conds',[0 0 0],'tent_spacing',tent_spacing1);

% second set of convolution kernel params (just as an example)
tent_spacing2 = 10;
num_lags2 = 40;
conv_params(2) = convNIM.create_conv_params([num_lags2 1 1],... % 40 lags @ 10 bins per lag @ 10 ms per bin = 4 second conv kernel
							'boundary_conds',[0 0 0],'tent_spacing',tent_spacing2);

% use 3 linear subunits with 2 different sets of kernel params
conv_targs = [1 1 2];			% stim subunits will target first entry in conv_params struture
NL_types = {'lin','lin','lin'};	% define subunits as linear
mod_signs = [1 1 1];			% define subunits as excitatory

% call constructor
fit0 = convNIM(conv_params,conv_targs,stim_params,NL_types,mod_signs,'spkNL','lin','noise_dist','gaussian','Xtargets',[1 2 3]);

% set initial conv_kernels
for i = 1:2
	fit0.subunits(i).convK = y3(1:tent_spacing1:num_lags1*tent_spacing1)';
	fit0.subunits(i).convK = fit0.subunits(i).convK/max(fit0.subunits(i).convK);
	fit0.subunits(i).conv_reg_lambdas.d2t = 100;
	fit0.subunits(i).reg_lambdas.d2x = 100;
end
fit0.subunits(end).convK = y3(1:tent_spacing2:num_lags2*tent_spacing2)';
fit0.subunits(end).convK = fit0.subunits(end).convK/max(fit0.subunits(end).convK);
fit0.subunits(i).conv_reg_lambdas.d2t = 100;
fit0.subunits(end).reg_lambdas.l2 = 100;


%% Fit model

% Robs, Xs, indx_tr and optim_params already defined

% fit stim filters
fit1 = fit0.fit_stim_filters(Robs, Xs, indx_tr, 'silent', 1, ...
							'optim_params', optim_params, 'subs', [1 2 3]);

% fit convolution kernels
fit1 = fit1.fit_conv_filters(Robs, Xs, indx_tr, 'silent', 1, ...
							'optim_params', optim_params, 'subs', [1 2 3]);

% renormalize convolution kernels to have max abs val of 1
for i = 1:3
	fit1.subunits(i).convK = fit1.subunits(i).convK/max(abs(fit1.subunits(i).convK));
end

% refit fit stim filters
fit1 = fit1.fit_stim_filters(Robs, Xs, indx_tr, 'silent', 1, ...
							'optim_params', optim_params, 'subs', [1 2 3]);


