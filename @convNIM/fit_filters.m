function cnim = fit_filters(cnim, Robs, Xstims, varargin)
% Usage: cnim = cnim.fit_filters( Robs, Xstims, <train_inds>, varargin )
%
% Overloaded method for convNIM; includes convolution step in calculation
% of function value and gradients
% INPUTS:
%	conv_kernels: array whose columns are convolution kernels that match up
%		with convSUBUNITS array in cnim
%   Robs: vector of response observations (e.g. spike counts)
%   Xstims: cell array of stimuli
%   <train_inds>: index values of data on which to fit the model [default to all indices in provided data]
%   optional flags:
%       ('subs',fit_subs): set of subunits whos filters we want to optimize [default is all]
%       ('gain_funs',gain_funs): matrix of multiplicative factors, one column for each subunit
%       ('fit_offsets',fit_offsets): vector of bools, (or single bool) specifying whether
%           to fit the additive offset terms associated with each subunit
%       ('optim_params',optim_params): struct of desired optimization parameters, can also
%           be used to override any of the default values for other optional inputs
%       ('silent',silent): boolean variable indicating whether to suppress the iterative optimization display
%       ('fit_spk_hist',fit_spk_hist): boolean indicating whether to hold the spk NL filter constant
%
% OUTPUTS:
%   cnim: new cnim object with optimized subunit filters
%
% TODO:
% handle spike history term and nontarg_g's
%
% NOTE:
% as it stands, gain_funs are applied AFTER the convolution

Nsubs = length(cnim.subunits); %number of subunits

% Set defaults for optional inputs
defaults.subs = 1:Nsubs; % defualt to fitting all subunits (plus -1 for spkHist filter)
defaults.gain_funs = []; % default has no gain_funs
defaults.fit_spk_hist = cnim.spk_hist.spkhstlen > 0; % default is to fit the spkNL filter if it exists
defaults.fit_offsets = false(1,Nsubs); % default is NOT to fit the offset terms
defaults.silent = false; % default is to display optimization output
option_list = {'subs','gain_funs','silent','fit_spk_hist','fit_offsets'}; % list of possible option strings

% Over-ride any defaults with user-specified values
OP_loc = find(strcmp(varargin,'optim_params')); % find if optim_params is provided as input
if ~isempty(OP_loc)
	optim_params = varargin{OP_loc+1};
	varargin(OP_loc:(OP_loc+1)) = [];
	OP_fields = lower(fieldnames(optim_params));
	for ii = 1:length(OP_fields) % loop over fields of optim_params
		if ismember(OP_fields{ii},option_list); %if the field is a valid input option, over-ride the default
			eval(sprintf('%s = optim_params.(''%s'');',OP_fields{ii},OP_fields{ii}));
			optim_params = rmfield(optim_params,OP_fields{ii}); %and remove the field from optim_params, so that the remaining fields are options for the optimizer
		end
	end
else
	optim_params = [];
end

% Now parse explicit optional input args
[train_inds,parsed_options] = NIM.parse_varargin( varargin, [], defaults );
NIM.validate_parsed_options( parsed_options, option_list );

gain_funs = parsed_options.gain_funs;
fit_subs = parsed_options.subs;
assert(all(ismember(fit_subs,1:Nsubs)),'invalid target subunits specified');
silent = parsed_options.silent;
assert(ismember(silent,[0 1]),'silent must be 0 or 1');
fit_spk_hist = parsed_options.fit_spk_hist;
assert(ismember(fit_spk_hist,[0 1]),'fit_spk_hist must be 0 or 1');

fit_offsets = parsed_options.fit_offsets;
if length(fit_offsets) == 1 % if only one value specified, assume we want to do the same for all subunits
	fit_offsets = repmat(fit_offsets,1,Nsubs); 
end
assert(~logical(sum(~ismember(fit_offsets,[0 1]))),'fit_offsets must be 0 or 1');
if length(fit_subs) < Nsubs && length(fit_offsets) == Nsubs
	fit_offsets = fit_offsets(fit_subs); % if only fitting subset of filters, set fit_offsets accordingly
end

mod_NL_types = {cnim.subunits.NLtype}; % NL types for each targeted subunit
fit_offsets(strcmp(mod_NL_types,'lin')) = 0;
fit_offsets(strcmp(mod_NL_types,'nonpar')) = 0;

% Validate inputs
if ~iscell(Xstims)
	tmp = Xstims;
	clear Xstims
	Xstims{1} = tmp;
end
if size(Robs,2) > size(Robs,1); Robs = Robs'; end; % make Robs a column vector
cnim.check_inputs( Robs, Xstims, train_inds, gain_funs ); % make sure input format is correct

Nfit_subs = length(fit_subs); %number of targeted subunits
non_fit_subs = setdiff( 1:Nsubs, fit_subs ); %elements of the model held constant
spkhstlen = cnim.spk_hist.spkhstlen; %length of spike history filter
if fit_spk_hist; assert(spkhstlen > 0,'no spike history term initialized!'); end;
if spkhstlen > 0 % create spike history Xmat IF NEEDED
	Xspkhst = cnim.create_spkhist_Xmat( Robs );
else
	Xspkhst = [];
end
if ~isnan(train_inds) %if specifying a subset of indices to train model params
	for nn = 1:length(Xstims)
		Xstims{nn} = Xstims{nn}(train_inds,:); %grab the subset of indices for each stimulus element
	end
	Robs = Robs(train_inds);
	if ~isempty(Xspkhst); Xspkhst = Xspkhst(train_inds,:); end;
	if ~isempty(gain_funs); gain_funs = gain_funs(train_inds,:); end;
end

% pull out convolution kernels and filter with tent-basis functions
filtered_kernels = cell(Nsubs+1,1); % store filtered conv kernels; store tbspaces in last cell for offsets
tbspaces = zeros(Nsubs,1);
for i = 1:Nsubs
	if ~isempty(cnim.conv_params(cnim.subunits(i).conv_targ).tent_spacing)
		tbspace = cnim.conv_params(cnim.subunits(i).conv_targ).tent_spacing;
		% create a tent-basis (triangle) filter
		tent_filter = [(1:tbspace)/tbspace 1-(1:tbspace-1)/tbspace]/tbspace;

		num_lags = cnim.conv_params(cnim.subunits(i).conv_targ).dims(1);
		filtered_kernel = zeros(tbspace*(num_lags+1),1); % take care of overlap
		for nn = 1:length(tent_filter)
			filtered_kernel((0:num_lags-1)*tbspace+nn) = filtered_kernel((0:num_lags-1)*tbspace+nn) + tent_filter(nn) * cnim.subunits(i).convK;
		end
		filtered_kernels{i} = filtered_kernel;
		tbspaces(i) = tbspace;
	else
		filtered_kernels{i} = cnim.subunits(i).convK;
		tbspaces(i) = 1;
	end
end
filtered_kernels{end} = tbspaces;

% pre-compute convolutions in the case of linear subunits to speed up
% gradient calculations
Xstims_conv = cell(Nfit_subs,1);
for i = 1:Nfit_subs
	indx = fit_subs(i);
	% use Xstim_convs if linear subunit
	if strcmp(cnim.subunits(indx).NLtype,'lin')
		temp = zeros(length(Robs)+length(filtered_kernels{indx})-1,size(Xstims{cnim.subunits(indx).Xtarg},2));	
		% look for gain_funs
		if ~isempty(gain_funs)
			for j = 1:size(Xstims{cnim.subunits(indx).Xtarg},2)
				temp(:,j) = conv(Xstims{cnim.subunits(indx).Xtarg}(:,j).*gain_funs(:,indx),filtered_kernels{indx});
			end
		else
			for j = 1:size(Xstims{cnim.subunits(indx).Xtarg},2)
				temp(:,j) = conv(Xstims{cnim.subunits(indx).Xtarg}(:,j),filtered_kernels{indx});
			end
		end
		Xstims_conv{i} = temp(filtered_kernels{end}(indx):(filtered_kernels{end}(indx)+length(Robs)-1),:);
	else
		Xstims_conv{i} = {};
	end
end

%% PARSE INITIAL PARAMETERS
[init_params,lambda_L1,sign_con] = deal([]);
for imod = fit_subs
	cur_kern = cnim.subunits(imod).filtK;
	if (cnim.subunits(imod).Ksign_con ~= 0) % add sign constraints on the filters of this subunit if needed
		sign_con(length(init_params)+(1:length(cur_kern))) = cnim.subunits(imod).Ksign_con;
	end
	lambda_L1(length(init_params) + (1:length(cur_kern))) = cnim.subunits(imod).reg_lambdas.l1;
	init_params = [init_params; cur_kern]; % add coefs to initial param vector
	% Verify non-parametric derivative is up-to-date
	cnim.subunits(imod).TBy_deriv = cnim.subunits(imod).get_TB_derivative();
end
lambda_L1 = lambda_L1'/sum(Robs); % since we are dealing with LL/spk

% Add in filter offsets if needed
for ii = 1:length(fit_subs)
	if fit_offsets(ii)
		init_params = [init_params; cnim.subunits(fit_subs(ii)).NLoffset]; 
		lambda_L1 = [lambda_L1; 0];
	end
end
Nfit_filt_params = length(init_params); % number of filter coefficients in param vector
% Add in spike history coefs
if fit_spk_hist
	init_params = [init_params; cnim.spk_hist.coefs];
	lambda_L1 = [lambda_L1; zeros(size(cnim.spk_hist.coefs))];
end

init_params = [init_params; cnim.spkNL.theta]; % add constant offset
lambda_L1 = [lambda_L1; 0];
[nontarg_g] = cnim.conv_process_stimulus(Xstims,non_fit_subs,gain_funs);
if ~fit_spk_hist && spkhstlen > 0 % add in spike history filter output, if we're not fitting it
	nontarg_g = nontarg_g + Xspkhst*cnim.spk_hist.coefs(:);
end

%% IDENTIFY ANY CONSTRAINTS 
use_con = 0;
LB = -Inf*ones(size(init_params));
UB = Inf*ones(size(init_params));
% Constrain any of the filters to be positive or negative
if any(sign_con ~= 0)
	LB(sign_con == 1) = 0;
	UB(sign_con == -1) = 0;
	use_con = 1;
end
if fit_spk_hist % if optimizing spk history term
	% negative constraint on spk history coefs
	if cnim.spk_hist.negCon
		spkhist_inds = Nfit_filt_params + (1:spkhstlen);
		UB(spkhist_inds) = 0;
		use_con = 1;
	end
end

%% GENERATE REGULARIZATION MATRICES
Tmats = cnim.make_Tikhonov_matrices();

fit_opts = struct( 'fit_spk_hist',fit_spk_hist, 'fit_subs',fit_subs, 'fit_offsets',fit_offsets ); % put any additional fitting options into this struct
% The function we want to optimize
opt_fun = @(K) internal_LL_filters(cnim, K, filtered_kernels, Xstims_conv, Robs, Xstims, Xspkhst, nontarg_g, gain_funs, Tmats, fit_opts);

% Determine which optimizer were going to use
if max(lambda_L1) > 0
	assert(~use_con,'Can use L1 penalty with constraints');
	assert(exist('L1General2_PSSas','file') == 2,'Need Mark Schmidts optimization tools installed to use L1');
	optimizer = 'L1General_PSSas';
else
	if ~use_con %if there are no constraints
		if exist('minFunc','file') == 2
			optimizer = 'minFunc';
		else
			optimizer = 'fminunc';
		end
	else
		if exist('minConf_TMP','file')==2
			optimizer = 'minConf_TMP';
		else
			optimizer = 'fmincon';
		end
	end
end 
optim_params = cnim.set_optim_params(optimizer,optim_params,silent);
if ~silent; fprintf('Running optimization using %s\n\n',optimizer); end;
switch optimizer %run optimization
	case 'L1General_PSSas'
		[params] = L1General2_PSSas(opt_fun,init_params,lambda_L1,optim_params);
	case 'minFunc'
		[params] = minFunc(opt_fun, init_params, optim_params);
	case 'fminunc'
		[params] = fminunc(opt_fun, init_params, optim_params);
	case 'minConf_TMP'
		[params] = minConf_TMP(opt_fun, init_params, LB, UB, optim_params);
	case 'fmincon'
		[params] = fmincon(opt_fun, init_params, [], [], [], [], LB, UB, [], optim_params);
end
[~,penGrad] = opt_fun(params);
first_order_optim = max(abs(penGrad));
if first_order_optim > cnim.opt_check_FO
	warning(sprintf('First-order optimality: %.3f, fit might not be converged!',first_order_optim));
end

%% PARSE MODEL FIT
cnim.spkNL.theta = params(end); % set new offset parameter
if fit_spk_hist
	cnim.spk_hist.coefs = params(Nfit_filt_params + (1:spkhstlen));
end
kOffset = 0; %position counter for indexing param vector
for ii = 1:Nfit_subs
	filtLen = length(cnim.subunits(fit_subs(ii)).filtK);
	cur_kern = params((1:filtLen) + kOffset); % grab parameters corresponding to this subunit's filters
	cnim.subunits(fit_subs(ii)).filtK = cur_kern(:); % assign new filter values
	kOffset = kOffset + filtLen;
end
for ii = 1:Nfit_subs % parse any fit offset parameters
	if fit_offsets(ii) 
		cnim.subunits(fit_subs(ii)).NLoffset = params(kOffset + 1);
		kOffset = kOffset + 1;
	end
end
[LL,~,mod_internals,LL_data] = cnim.eval_model(Robs,Xstims,'gain_funs',gain_funs);
cnim = cnim.set_subunit_scales(mod_internals.fgint); %update filter scales
cur_fit_details = struct('fit_type','filter','LL',LL,'filt_pen',LL_data.filt_pen,...
    'NL_pen',LL_data.NL_pen,'FO_optim',first_order_optim,'fit_subs',fit_subs);
cnim.fit_props = cur_fit_details; %store details of this fit
cnim.fit_history = cat(1,cnim.fit_history,cur_fit_details);

end



%% ************************** INTERNAL FUNCTION ***************************

function [penLL, penLLgrad] = internal_LL_filters( cnim, params, conv_kernels, Xstims_conv, Robs, Xstims, Xspkhst, nontarg_g, gain_funs, Tmats, fit_opts )
% computes the penalized LL and its gradient wrt the filters for the given nim with parameter vector params

fit_subs = fit_opts.fit_subs;
Nfit_subs = length(fit_subs);					% number of targeted subs
fit_offsets = fit_opts.fit_offsets;				% which filters are we fitting offset parameters for

% USEFUL VALUES
theta = params(end);							% overall model offset
gint = nan(length(Robs),Nfit_subs);				% initialize matrix for storing filter outputs
filtLen = zeros(Nfit_subs,1);					% store the length of each (target) sub's filter
filtKs = cell(Nfit_subs,1);						% store the filter coefs for all (target) subs)
param_inds = cell(Nfit_subs,1);					% this will store the index values of each subunit's filter coefs within the parameter vector
Xtarg_set = [cnim.subunits(fit_subs).Xtarg];	% vector of Xfit_subs for set of subunits being optimized
un_Xtargs = unique(Xtarg_set);					% set of unique Xfit_subs
mod_NL_types = {cnim.subunits(fit_subs).NLtype};% NL types for each targeted subunit
unique_NL_types = unique(mod_NL_types);			% unique set of NL types being used
mod_weights = [cnim.subunits(fit_subs).weight]';% signs of targeted subunits

G = theta + nontarg_g; % initialize overall generating function G with the offset term and the contribution from nontarget subs

NKtot = 0;										% init filter coef counter
for ii = 1:Nfit_subs							% loop over subunits, get filter coefs and their indices within the parameter vector
	filtLen(ii) = length(cnim.subunits(fit_subs(ii)).filtK); % length of filter
	param_inds{ii} = NKtot + (1:filtLen(ii));	% set of param indices associated with this subunit's filters
	filtKs{ii} = params(param_inds{ii});		% store filter coefs
	NKtot = NKtot + filtLen(ii);				% inc counter
end
sub_offsets = [cnim.subunits(fit_subs).NLoffset]; % default offsets to whatever they're set at
offset_inds = NKtot + (1:sum(fit_offsets));		% indices within parameter vector of offset terms we're fitting
sub_offsets(fit_offsets == 1) = params(offset_inds); % if were fitting, overwrite these values with current params

if ~isempty(un_Xtargs)
	for ii = 1:length(un_Xtargs)				% loop over the unique Xtargs and compute the generating signals for all relevant filters
		cur_subs = find(Xtarg_set == un_Xtargs(ii)); % set of targeted subunits that act on this Xtarg
		gint(:,cur_subs) = Xstims{un_Xtargs(ii)} * cat(2,filtKs{cur_subs}); % apply filters to stimulus
	end
	gint = bsxfun(@plus,gint,sub_offsets);		% add in filter offsets
end

use_batch_calc = false;
 fgint = gint; %init subunit outputs by filter outputs
for ii = 1:length(unique_NL_types) %loop over unique subunit NL types and apply NLs to gint in batch
	cur_subs = find(strcmp(mod_NL_types,unique_NL_types{ii})); %set of subs with this NL type
	if ~strcmp(unique_NL_types{ii},'lin') %if its a linear subunit we dont have to do anything
		NLparam_mat = cat(1,cnim.subunits(fit_subs(cur_subs)).NLparams); %matrix of upstream NL parameters
		if isempty(NLparam_mat) || (length(cur_subs) > 1 && max(max(abs(diff(NLparam_mat)))) == 0)     
			use_batch_calc = true; %if there are no NLparams, or if all subunits have the same NLparams, use batch calc  
		end
		if strcmp(unique_NL_types{ii},'nonpar') || ~use_batch_calc % if were using nonpar NLs or parametric NLs with unique parameters, need to apply NLs individually
			for jj = 1:length(cur_subs) % for TB NLs need to apply each subunit's NL individually
				fgint(:,cur_subs(jj)) = cnim.subunits(fit_subs(cur_subs(jj))).apply_NL(gint(:,cur_subs(jj)));
			end
		else % apply upstream NL in batch to all subunits in current set
			fgint(:,cur_subs) = cnim.subunits(fit_subs(cur_subs(1))).apply_NL(gint(:,cur_subs)); % apply upstream NL to all subunits of this type
		end	
	end
end

% Multiply by weight (and multiplier, if appl)
if ~isempty(fit_subs)
	if isempty(gain_funs)
		fgint = bsxfun(@times,fgint,mod_weights');
	else
		fgint = bsxfun(@times,(fgint.*gain_funs(:,fit_subs)),mod_weights');  
	end
end

% convolve
if ~isempty(fit_subs)
	for n = 1:length(fit_subs)
		% as convolution kernels get larger the time-domain computations in
		% 'conv' take too long (N^2); fftfilt shape implements an Nlog(N)
		% algorithm
		if length(conv_kernels{fit_subs(n)}) < 1000
			temp = conv(fgint(:,n),conv_kernels{fit_subs(n)});
		else
			temp = fftfiltshape(conv_kernels{fit_subs(n)},fgint(:,n));
		end
		temp = temp(conv_kernels{end}(fit_subs(n)):(conv_kernels{end}(fit_subs(n))+length(G)-1));
		G = G + temp;
	end	
end

% Add contribution from spike history filter
if fit_opts.fit_spk_hist
	G = G + Xspkhst*params(NKtot + length(offset_inds) + (1:cnim.spk_hist.spkhstlen));
end

pred_rate = cnim.apply_spkNL(G);
[penLL,LL_norm] = cnim.internal_LL(pred_rate,Robs); %compute LL and its normalization

%residual = LL'[r].*F'[g]
residual = cnim.internal_LL_deriv(pred_rate,Robs) .* cnim.apply_spkNL_deriv(G,pred_rate <= cnim.min_pred_rate);

penLLgrad = zeros(length(params),1); % initialize LL gradient
penLLgrad(end) = sum(residual);      % calculate derivatives with respect to constant term (theta)

% Calculate derivative with respect to spk history filter
if fit_opts.fit_spk_hist
	penLLgrad(NKtot + length(offset_inds) + (1:cnim.spk_hist.spkhstlen)) = residual'*Xspkhst;
end

% loop through subunits and calculate gradients
for i = 1:Nfit_subs
	indx = fit_subs(i);
	% use Xstim_convs if linear subunit
	if ~strcmp(cnim.subunits(indx).NLtype,'lin')
		temp = zeros(length(G)+length(conv_kernels{indx})-1,size(Xstims{cnim.subunits(indx).Xtarg},2));	
		% look for gain_funs
		if ~isempty(gain_funs)
			for j = 1:size(Xstims{cnim.subunits(indx).Xtarg},2)
				temp(:,j) = conv(Xstims{cnim.subunits(indx).Xtarg}(:,j).*cnim.subunits(indx).apply_NL_deriv(gint(:,i)).*gain_funs(:,indx),conv_kernels{indx});
			end
		else
			for j = 1:size(Xstims{cnim.subunits(indx).Xtarg},2)
				temp(:,j) = conv(Xstims{cnim.subunits(indx).Xtarg}(:,j).*cnim.subunits(indx).apply_NL_deriv(gint(:,i)),conv_kernels{indx});
			end
		end
		temp = temp(conv_kernels{end}(indx):(conv_kernels{end}(indx)+length(G)-1),:);
		penLLgrad(param_inds{i}) = residual'*temp*mod_weights(i);
	else
		penLLgrad(param_inds{i}) = residual'*Xstims_conv{i}*mod_weights(i);
	end
	if fit_offsets(i)
		if ~isempty(gain_funs)
			penLLgrad(offset_inds(i)) = (residual'*(gain_funs(:,indx).*cnim.subunits(indx).apply_NL_deriv(gint(:,i))))*mod_weights(i);
		else
			penLLgrad(offset_inds(i)) = (residual'*cnim.subunits(indx).apply_NL_deriv(gint(:,i)))*mod_weights(i);
		end
	end
end

net_penalties = zeros(size(fit_subs));
net_pen_grads = zeros(length(params),1);
for ii = 1:length(Tmats) % loop over the derivative regularization matrices
	cur_subs = find([cnim.subunits(fit_subs).Xtarg] == Tmats(ii).Xtarg); % set of subunits acting on the stimulus given by this Tmat
	if ~isempty(cur_subs)
		penalties = sum((Tmats(ii).Tmat * cat(2,filtKs{cur_subs})).^2);
		pen_grads = 2*(Tmats(ii).Tmat' * Tmats(ii).Tmat * cat(2,filtKs{cur_subs}));
		cur_lambdas = cnim.get_reg_lambdas(Tmats(ii).type,'subs',fit_subs(cur_subs)); % current lambdas
		net_penalties(cur_subs) = net_penalties(cur_subs) + penalties.*cur_lambdas;
		net_pen_grads(cat(2,param_inds{cur_subs})) = net_pen_grads(cat(2,param_inds{cur_subs})) + reshape(bsxfun(@times,pen_grads,cur_lambdas),[],1);
	end
end

l2_lambdas = cnim.get_reg_lambdas('subs',fit_subs,'l2');
if any(l2_lambdas > 0)
	net_penalties = net_penalties + l2_lambdas.*cellfun(@(x) sum(x.^2),filtKs)';
	for ii = 1:length(un_Xtargs)
		cur_subs = find(Xtarg_set == un_Xtargs(ii)); % set of targeted subunits that act on this Xtarg
		net_pen_grads(cat(2,param_inds{cur_subs})) = net_pen_grads(cat(2,param_inds{cur_subs})) + reshape(2*bsxfun(@times,l2_lambdas(cur_subs),cat(2,filtKs{cur_subs})),[],1);
	end
end

penLL = penLL - sum(net_penalties);
penLLgrad = penLLgrad - net_pen_grads;

% CONVERT TO NEGATIVE LLS AND NORMALIZE 
penLL = -penLL/LL_norm;
penLLgrad = -penLLgrad/LL_norm;

end
