classdef convNIM < NIM
    
% Class implementation of a convolutional NIM based on the NIM class. A 
% convolution kernel for each subunit is specified that acts on the output 
% of that subunit before all convolved subunit outputs are summed and 
% passed through the spiking nonlinearity.
%
% The main difference between this class and the NIM class is the use of
% the CONVSUBUNIT class in place of the SUBUNIT class, which forces the set
% of fitting methods in the base NIM to be overridden. 

% TODO
% take care of spike history term wrt its own convolution kernel
% only works with fit_filters and upstream_NLs; other fitting methods will not work
%	(spkNL, NL_params)


%% PROPERTIES
properties
	
	% Inherited from NIM
	% spkNL;			% struct defining the spiking NL function
	% subunits;			% array of subunit objects
	% stim_params;		% struct array of parameters characterizing the stimuli that the model acts on, must have a .dims field
	% noise_dist;		% noise distribution class specifying the noise model
	% spk_hist;			% class defining the spike-history filter properties
	% fit_props;		% struct containing information about model fit evaluations
	% fit_history;		% struct containing info about history of fitting
	
	% convolution properties
	conv_params;		% struct array of parameters characterizing the convolution kernels used in the model

end

properties (Hidden)
	
	% Inherited from the NIM
	% init_props;         % struct containing details about model initialization
	% allowed_reg_types = {'nld2','d2xt','d2x','d2t','l2','l1'}; % set of allowed regularization types
	% allowed_spkNLs = {'lin','rectpow','exp','softplus','logistic'}; % set of NL functions currently implemented
	% allowed_noise_dists = {'poisson','bernoulli','gaussian'}; % allowed noise distributions
	% version = '1.0';    % source code version used to generate the model
	% create_on = date;   % date model was generated
	% min_pred_rate = 1e-50; % minimum predicted rate (for non-negative data) to avoid NAN LL values
	% opt_check_FO = 1e-2;   % threshold on first-order optimality for fit-checking

end

% methods defined in other files
methods
	cnim = fit_filters( cnim, Robs, Xstims, varargin );
	cnim = fit_upstreamNLs( cnim, Robs, Xstims, varargin );
	cnim = reg_path_sim( cnim, Robs, Xstims, Uindx, XVindx, varargin );
end
	
%% ********************** Constructor *************************************
methods 
	
	function cnim = convNIM( conv_params, conv_targs, stim_params, NLtypes, mod_signs, varargin )
	% cnim = convNIM( conv_params, conv_targs, stim_params, NLtypes, mod_signs, varargin )
	%
	% Constructor method for convNIM. Mostly just passes needed properties
	% on to the NIM constructor. Appends kernel params to the base NIM and
	% initializes subunits with delta function convolution kernels.
	% INPUTS:
	%	conv_params:	same params as stim_params that provides
	%					information on the convolution kernel dimensions
	%	conv_targs:		array of integers that specify which entry in
	%					conv_params structure each subunit will reference
	
	if nargin == 0 || isempty(conv_params)
		stim_params = [];
		NLtypes = [];
		mod_signs = [];
	end
	
	cnim@NIM( stim_params, NLtypes, mod_signs, varargin{:} );
	
	if nargin == 0 || isempty(conv_params)
		return
	end
	
	
	% make sure conv_targs and conv_params match
	assert(max(conv_targs) <= length(conv_params), 'Invalid conv_params targets')
	
	% check length of conv_targs; if 1, assume this set of params for
	% convolution kernels of all subunits
	if length(conv_targs) ~= length(NLtypes)
		if length(conv_params) == 1
			conv_targs = ones(length(NLtypes),1);
		else
			error('Dimension mismatch between conv_params and conv_targs')
		end
	end
	
	% change subunits to CONVSUBUNIT types
	num_subunits = length(cnim.subunits);
	subunits = cnim.subunits;
	cnim.subunits = [];
	nim_subunit_info = cell(7,1);
	for n = 1:num_subunits
		
		% get NIM subunit info to reinitialize subunit
		nim_subunit_info{1} = subunits(n).filtK;
		nim_subunit_info{2} = subunits(n).weight;
		nim_subunit_info{3} = subunits(n).NLtype; 
		nim_subunit_info{4} = subunits(n).Xtarg;
		nim_subunit_info{5} = subunits(n).NLoffset;
		nim_subunit_info{6} = subunits(n).NLparams;
		nim_subunit_info{7} = subunits(n).Ksign_con;
		
		% set Ksign_con (make an option later)
		convKsign_con = 0;

		% set initial convolution kernel to ones
		init_conv = ones(conv_params(conv_targs(n)).dims(1),1);
		
		% construct new subunit
		cnim.subunits = cat(1,cnim.subunits,CONVSUBUNIT(nim_subunit_info{:}, init_conv, conv_targs(n), convKsign_con));
	end
			
	% set conv_params
	cnim.conv_params = conv_params;
	
	end
	
end

%% ********************** Fitting Methods *********************************
methods 
		
	function cnim = fit_stim_filters(cnim, Robs, Xstims, varargin)
	% Usage: cnim = cnim.fit_stim_filters( Robs, Xstims, <train_inds>, varargin )
	%
	% Estimates stimulus filters of convNIM model.
	% INPUTS:
	%   Robs: vector of response observations (e.g. spike counts)
	%   Xstims: cell array of stimuli
	%   <train_inds>: index values of data on which to fit the model [default to
	%		all indices in provided data]
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
	%   cnim: new cnim object with optimized stimulus filters
	

		cnim = cnim.fit_filters(Robs, Xstims, varargin{:});
	
	end
	
	function cnim = fit_conv_filters(cnim, Robs, Xstims, varargin)
	% Usage: cnim = cnim.fit_conv_filters( Robs, Xstims, <train_inds>, varargin )
	%
	% Estimates convolutional kernels of convNIM model.
	% INPUTS:
	%   Robs: vector of response observations (e.g. spike counts)
	%   Xstims: cell array of stimuli
	%   <train_inds>: index values of data on which to fit the model [default to
	%		all indices in provided data]
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
	%   cnim: new cnim object with optimized convolution kernels
	
		num_subunits = length(cnim.subunits);

		% parse input arguments to find gain_funs
		[~,parsed_options] = NIM.parse_varargin(varargin);
		if isfield(parsed_options, 'gain_funs')
			gain_funs = parsed_options.gain_funs;
		else
			gain_funs = [];	% default is empty
		end
		
		% call normal NIM fit_filters method to get kernel values
		[temp_nim, Xsubout] = cnim.convert2NIM(Xstims, gain_funs);
		temp_nim = temp_nim.fit_filters(Robs, Xsubout, varargin{:});

		% update cnim
		% assign temp_nim subunit info to cnim subunit info
		for n = 1:num_subunits
			cnim.subunits(n).convK = temp_nim.subunits(n).filtK;
		end
		% copy over spiking nonlinearity fits and fit history
		cnim.spkNL = temp_nim.spkNL;
		cnim.fit_props = temp_nim.fit_props;
		cnim.fit_props.fit_type = 'conv_kernel';
		cnim.fit_history = temp_nim.fit_history;
		cnim.fit_history(end).fit_type = 'conv_kernel';
	
	end
	
	function cnim = reg_path_stim(cnim, Robs, Xstims, varargin)
	% Usage: cnim = cnim.reg_path_stim( Robs, Xstims, <train_inds>, varargin )
	%
	% Estimates stimulus filters of convNIM model.
	% INPUTS:
	%   Robs: vector of response observations (e.g. spike counts)
	%   Xstims: cell array of stimuli
	%   <train_inds>: index values of data on which to fit the model [default to
	%		all indices in provided data]
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
	%   cnim: new cnim object with optimized stimulus filters
	

		cnim = cnim.reg_path_sim(Robs, Xstims, varargin{:});
	
	end
	
	function cnim = reg_path_conv(cnim, Robs, Xstims, varargin)
	% Usage: cnim = cnim.reg_path_conv( Robs, Xstims, <train_inds>, varargin )
	%
	% Estimates convolutional kernels of convNIM model.
	% INPUTS:
	%   Robs: vector of response observations (e.g. spike counts)
	%   Xstims: cell array of stimuli
	%   <train_inds>: index values of data on which to fit the model [default to
	%		all indices in provided data]
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
	%   cnim: new cnim object with optimized convolution kernels
	
		num_subunits = length(cnim.subunits);

		% 1) process stimulus with each subunit
		% parse input arguments to find gain_funs
		[~,parsed_options] = NIM.parse_varargin(varargin);
		if isfield(parsed_options, 'gain_funs')
			gain_funs = parsed_options.gain_funs;
		else
			gain_funs = [];	% default is empty
		end
		
		% call normal NIM fit_filters method to get kernel values
		[temp_nim, Xsubout] = cnim.convert2NIM(Xstims, gain_funs);
		temp_nim = temp_nim.reg_path(Robs, Xsubout, varargin{:});

		% 4) update cnim
		% assign temp_nim subunit info to cnim subunit info
		for n = 1:num_subunits
			cnim.subunits(n).convK = temp_nim.subunits(n).filtK;
			cnim.subunits(n).conv_reg_lambdas = temp_nim.subunits(n).reg_lambdas;
		end
		% copy over spiking nonlinearity fits and fit history
		cnim.spkNL = temp_nim.spkNL;
		cnim.fit_props = temp_nim.fit_props;
		cnim.fit_props.fit_type = 'conv_kernel';
		cnim.fit_history = temp_nim.fit_history;
		cnim.fit_history(end).fit_type = 'conv_kernel';
	
	end
	
	function cnim = fit_SCalt_filters(cnim, Robs, Xstims, varargin)
	% Usage: cnim = cnim.fit_SCalt_filters(Robs, Xstims, varargin)

		% Check to see if silent (for alt function)
		silent = 0;
		if ~isempty(varargin)
			for j = 1:length(varargin)
				if strcmp( varargin{j}, 'silent' );
					silent = varargin{j+1};
				end
			end		
		end
					
		LLtol = 0.0002; max_iter = 12;
		varargin{end+1} = 'silent';
		varargin{end+1} = 1;

		LL = cnim.eval_model( Robs, Xstims, varargin{:} );

		LLpast = -1e10;
		if ~silent
			fprintf( 'Beginning LL = %f\n', LL )
		end
		
		iter = 1;	
		while (((LL-LLpast) > LLtol) && (iter < max_iter))
	
			cnim = cnim.fit_stim_filters( Robs, Xstims, varargin{:} );
			cnim = cnim.fit_conv_filters( Robs, Xstims, varargin{:} );

			LLpast = LL;
			LL = cnim.fit_props.LL;
			iter = iter + 1;

			if ~silent
				fprintf( '  Iter %2d: LL = %f\n', iter, LL )
			end	
		end
		
		% finish with fitting stim filters since conv kernels have been
		% normalized
		cnim = cnim.fit_stim_filters( Robs, Xstims, varargin{:} );
		
	end
	
	function cnim = fit_CSalt_filters(cnim, Robs, Xstims, varargin)
	% Usage: cnim = cnim.fit_alt_filtersCS(Robs, Xstims, varargin)

		% Check to see if silent (for alt function)
		silent = 0;
		if ~isempty(varargin)
			for j = 1:length(varargin)
				if strcmp( varargin{j}, 'silent' );
					silent = varargin{j+1};
				end
			end		
		end
					
		LLtol = 0.0002; max_iter = 12;
		varargin{end+1} = 'silent';
		varargin{end+1} = 1;

		LL = cnim.eval_model( Robs, Xstims, varargin{:} );

		LLpast = -1e10;
		if ~silent
			fprintf( 'Beginning LL = %f\n', LL )
		end
		
		iter = 1;	
		while (((LL-LLpast) > LLtol) && (iter < max_iter))
	
			cnim = cnim.fit_conv_filters( Robs, Xstims, varargin{:} );
			cnim = cnim.fit_stim_filters( Robs, Xstims, varargin{:} );
			
			LLpast = LL;
			LL = cnim.fit_props.LL;
			iter = iter + 1;

			if ~silent
				fprintf( '  Iter %2d: LL = %f\n', iter, LL )
			end	
		end
		
	end
	
	function cnim = fit_NLparams(cnim, Robs, Xstims, varargin)
		warning('fit_NLparams not currently supported for convNIM class')
	end
	
	function cnim = fit_spkNL(cnim, Robs, Xstims, varargin)
		warning('fit_spkNL not currently supported for convNIM class')
	end
	
	function cnim = fit_weights(cnim, Robs, Xstims, varargin)
		warning('fit_weights not currently supported for convNIM class')
	end
	
end

%% ********************** Getting Methods *********************************
methods 
	
	function [filt_penalties, NL_penalties, conv_penalties] = get_reg_pen( cnim, Tmats, convTmats )
	% Usage: [filt_penalties,NL_penalties,conv_penalties] = cnim.get_reg_pen( <Tmats>, <convTmats> )
	%
	% Calculates the regularization penalties on each subunit, separately for filter and NL regularization
	% INPUTS: 
	%   <Tmats>:		struct array of 'Tikhonov' regularization matrices for
	%					stimulus filters
	%   <convTmats>:	struct array of 'Tikhonov' regularization matrices for
	%					convolution kernels
	%
	% OUTPUTS: 
	%   filt_penalties: Kx1 vector of regularization penalties for each filter
	%   NL_penalties:	Kx1 vector of regularization penalties for each upstream NL
	%   conv_penalties: Kx1 vector of regularization penalties for each conv kernel
	
		if nargin < 2 || isempty(Tmats) %if the Tmats are not precomputed and supplied, compute them here        
			[Tmats, convTmats] = make_Tikhonov_matrices(cnim);
		elseif nargin < 3 || isempty(convTmats)
			[~, convTmats] = make_Tikhonov_matrices(cnim);
		end
		
		% set constants
		Nsubs = length(cnim.subunits);
		Xtargs = [cnim.subunits(:).Xtarg];
		filtKs = cnim.get_filtKs();
		filt_penalties = zeros(1,Nsubs);
		convXtargs = [cnim.subunits(:).conv_targ];
		convKs = cnim.get_convKs();
		conv_penalties = zeros(1,Nsubs);
		
		% loop over the derivative regularization matrices for stimulus
		% filters
		for ii = 1:length(Tmats) 
			cur_subs = find(Xtargs == Tmats(ii).Xtarg); % set of subunits acting on the stimulus given by this Tmat
			cur_penalties = sum((Tmats(ii).Tmat * cat(2,filtKs{cur_subs})).^2);
			cur_lambdas = cnim.get_reg_lambdas(Tmats(ii).type,'subs',cur_subs); % current lambdas
			filt_penalties(cur_subs) = filt_penalties(cur_subs) + cur_penalties.*cur_lambdas; % reg penalties for filters
		end
		
		% compute L2 penalties on the filter coefficients
		l2_lambdas = cnim.get_reg_lambdas('l2');
		if any(l2_lambdas > 0) 
			filt_penalties = filt_penalties + l2_lambdas.*cellfun(@(x) sum(x.^2),filtKs)';
		end
		
		
		% loop over the derivative regularization matrices for convolution
		% kernels
		for ii = 1:length(convTmats) % loop over the derivative regularization matrices
			cur_subs = find(convXtargs == convTmats(ii).Xtarg); % set of subunits acting on the stimulus given by this Tmat
			cur_penalties = sum((convTmats(ii).Tmat * cat(2,convKs{cur_subs})).^2);
			cur_lambdas = cnim.get_conv_reg_lambdas(convTmats(ii).type,'subs',cur_subs); % current lambdas
			conv_penalties(cur_subs) = conv_penalties(cur_subs) + cur_penalties.*cur_lambdas; % reg penalties for filters
		end
		
		% compute L2 penalties on the convolution kernel
		l2_lambdas = cnim.get_conv_reg_lambdas('l2');
		if any(l2_lambdas > 0) % compute L2 penalties on the filter coefficients
			conv_penalties = conv_penalties + l2_lambdas.*cellfun(@(x) sum(x.^2),convKs)';
		end
		
		
		% compute the reg penalty on each subunit's NL
		nl_lambdas = cnim.get_reg_lambdas('nld2');  % reg lambdas on the NL TB coefficients
		NL_penalties = zeros(1,Nsubs);
		if any(nl_lambdas > 0)
			Tmat = cnim.make_NL_Tmat();
			nonpar_subs = find(strcmp(cnim.get_NLtypes,'nonpar'))';
			for imod = nonpar_subs 
				NL_penalties(imod) = nl_lambdas(imod)*sum((Tmat*cnim.subunits(imod).NLnonpar.TBy').^2);
			end
		end
		
	end

	function [Tmats, convTmats] = make_Tikhonov_matrices( cnim )
	% Usage: Tmats = cnim.make_Tikhonov_matrices()
	% Creates a struct containing the Tikhonov regularization matrices, given the stimulus and regularization 
	% parameters specified in the cnim
    
		Nstims = length(cnim.stim_params); % number of unique stimuli 
		Xtargs = [cnim.subunits(:).Xtarg];

		deriv_reg_types = cnim.allowed_reg_types(strncmp(cnim.allowed_reg_types,'d',1)); % set of regularization types where we need a Tikhonov matrix
		
		% get stimulus Tmats
		cnt = 1;
		Tmats = [];
		for ii = 1:Nstims							% for each stimulus
			cur_subs = find(Xtargs == ii);			% get set of subunits acting on this stimuls
			for jj = 1:length(deriv_reg_types)		% check each possible derivative regularization type
				cur_lambdas = cnim.get_reg_lambdas(deriv_reg_types{jj},'subs',cur_subs);
				if any(cur_lambdas > 0)
					cur_Tmat = NIM.create_Tikhonov_matrix(cnim.stim_params(ii),deriv_reg_types{jj});
					Tmats(cnt).Tmat = cur_Tmat;
					Tmats(cnt).Xtarg = ii;
					Tmats(cnt).type = deriv_reg_types{jj};
					cnt = cnt + 1;
				end
			end
		end
		
		% get convolution Tmats
		Nsubs = length(cnim.subunits);
		cnt = 1;
		convTmats = [];
		for ii = 1:Nsubs
			for jj = 1:length(deriv_reg_types)		% check each possible derivative regularization type
				cur_lambda = cnim.get_conv_reg_lambdas(deriv_reg_types{jj},'subs',ii);
				if cur_lambda > 0
					cur_Tmat = NIM.create_Tikhonov_matrix(cnim.conv_params(cnim.subunits(ii).conv_targ),deriv_reg_types{jj});
					convTmats(cnt).Tmat = cur_Tmat;
					convTmats(cnt).Xtarg = cnim.subunits(ii).conv_targ;
					convTmats(cnt).type = deriv_reg_types{jj};
					cnt = cnt + 1;
				end
			end
		end
		
	end
	
	function lambdas = get_conv_reg_lambdas( cnim, varargin )
	% Usage: lambdas = cnim.get_reg_lambdas( varargin )
	% Gets regularizatoin lambda values for convolution kernel of specified type
	% from a set of nim subunits
	%
	% INPUTS:
	%   optional flags:
	%      'subs': vector specifying which subunits to extract lambda values from
	%      'lambda_type': string specifying the regularization type
	%
	% OUTPUTS:
	%   lambdas: [K,N] matrix of lambda values, K is the number of specified 
	%		lambda_types and N is the number of subunits

		sub_inds = 1:length(cnim.subunits); % default is to grab reg values from all subunits
		
		% INPUT PARSING    
		jj = 1;
		reg_types = {};
		while jj <= length(varargin)
			switch lower(varargin{jj})
				case 'subs'
					sub_inds = varargin{jj+1};
					assert(all(ismember(sub_inds,1:length(cnim.subunits))),'invalid target subunits specified');
					jj = jj + 2;
				case cnim.allowed_reg_types
					reg_types = cat(1,reg_types,lower(varargin{jj}));
					jj = jj + 1;
				otherwise
					error('Invalid input flag'); 
			end		
		end
		
		lambdas = nan(length(reg_types),length(sub_inds));
		if isempty(reg_types)
			warning( 'No regularization type specified, returning nothing' );
		end
		for ii = 1:length(reg_types)
			for jj = 1:length(sub_inds)
				lambdas(ii,jj) = cnim.subunits(sub_inds(jj)).conv_reg_lambdas.(reg_types{ii});  
			end
		end	
	end
	
	function convKs = get_convKs( cnim, sub_inds )
	% Usage: convKs = nim.get_convKs( sub_inds )
	% Gets convolution kernels for specified set of subunits 
	%
	% INPUTS:
	% 	<sub_inds>: vector specifying which subunits to get filters from (default is all subs)
	%
	% OUTPUTS:
	%   convKs: Cell array of convolution kernels
      
		Nsubs = length(cnim.subunits);
		if nargin < 2
			sub_inds = 1:Nsubs; %default is to grab filters for all subunits
		end
		convKs = cell(length(sub_inds),1);
		for ii = 1:length(sub_inds)
			convKs{ii} = cnim.subunits(sub_inds(ii)).get_convK;
		end	
	end
	
end

%% ********************** Hidden Methods **********************************
methods (Hidden) 
	
	function [nim, Xsubout] = convert2NIM(cnim, Xstims, gain_funs)
	% takes a convNIM object and turns it into an NIM whos subunits
	% represent the convolution filters; for fitting
	
	num_subunits = length(cnim.subunits);
	
	% process stimulus with each subunit
	[~, fgint] = cnim.process_stimulus(Xstims, 1:num_subunits, gain_funs);

	% embed each stimulus into its own Xmat using however many lags are
	% specified in the convolution kernel
	Xsubout = cell(num_subunits,1);
	for n = 1:num_subunits
		stim_params(n) = cnim.conv_params(cnim.subunits(n).conv_targ);
		Xsubout{n} = NIM.create_time_embedding(fgint(:,n), stim_params(n));
	end
		
	% set up initial NIM model
	NLtypes = repmat({'lin'}, 1, num_subunits);
	mod_signs = ones(1, num_subunits);
	nim = NIM(stim_params, NLtypes, mod_signs, 'xtargets', 1:num_subunits);
	
	% pass cnim properties to nim
	nim.spkNL = cnim.spkNL;
	nim.noise_dist = cnim.noise_dist;
	nim.spk_hist = cnim.spk_hist;
	nim.fit_history = cnim.fit_history;
	
	% convert cnim subunits to nim subunits
	for n = 1:num_subunits
		nim.subunits(n).filtK = cnim.subunits(n).convK;
		nim.subunits(n).Xtarg = n;	% to match fgint
		nim.subunits(n).reg_lambdas = cnim.subunits(n).conv_reg_lambdas;
		nim.subunits(n).Ksign_con = cnim.subunits(n).convKsign_con;
	end
	
	end
	
	function [G, fgint, gint] = conv_process_stimulus( cnim, Xstims, sub_inds, gain_funs )
	% Usage: [G, fgint, gint] = conv_process_stimulus( cnim, Xstims, sub_inds, gain_funs )
	%
	% Processes the stimulus with the subunits specified in sub_inds. G and
	% fgint will include convolution with kernels
    % INPUTS:
    %	Xstims:		stimulus as cell array
    %   sub_inds:	set of subunits to process
    %   gain_funs:	temporally modulated gain of each subunit
	%
    % OUTPUTS:
    %   G:			summed generating signal (sum of fgints)
    %   fgint:		output of each subunit convolved with appropriate kernel
    %   gint:		output of each subunit filter

        NT = size(Xstims{1},1);
		if isempty(sub_inds);
			[G,fgint,gint] = deal(zeros(NT,1));
			return
		end
		
		Nsubs = length(sub_inds);
        filter_offsets = [cnim.subunits(sub_inds).NLoffset]; %set of filter offsets

		% pull out convolution kernels and filter with tent-basis functions
		conv_kernels = cell(Nsubs+1,1); % store filtered conv kernels; store tbspaces in last cell for offsets
		tbspaces = zeros(Nsubs,1);
		for n = 1:Nsubs
			indx = sub_inds(n);
			if ~isempty(cnim.conv_params(cnim.subunits(indx).conv_targ).tent_spacing)
				tbspace = cnim.conv_params(cnim.subunits(indx).conv_targ).tent_spacing;
				% create a tent-basis (triangle) filter
				tent_filter = [(1:tbspace)/tbspace 1-(1:tbspace-1)/tbspace]/tbspace;

				num_lags = cnim.conv_params(cnim.subunits(indx).conv_targ).dims(1);
				filtered_kernel = zeros(tbspace*(num_lags+1),1); % take care of overlap
				for nn = 1:length(tent_filter)
					filtered_kernel((0:num_lags-1)*tbspace+nn) = filtered_kernel((0:num_lags-1)*tbspace+nn) + tent_filter(nn) * cnim.subunits(n).convK;
				end
				conv_kernels{n} = filtered_kernel;
				tbspaces(n) = tbspace;
			else
				conv_kernels{n} = cnim.subunits(indx).convK;
				tbspaces(n) = 1;
			end
		end
		conv_kernels{end} = tbspaces;

        gint = zeros(size(Xstims{1},1),Nsubs);
		fgint = gint;
		% loop over all subunits to compute the generating signals
        for n = 1:Nsubs
			indx = sub_inds(n);
            gint(:,n) = Xstims{cnim.subunits(indx).Xtarg} * cnim.subunits(indx).filtK + filter_offsets(n); % apply filter to stimulus
			fgint(:,n) = cnim.subunits(indx).apply_NL(gint(:,n));	% apply upstream NL
			if ~isempty(gain_funs)
				fgint = fgint.*gain_funs(:,sub_inds);				% apply gain modulation if needed
			end
			temp = conv(fgint(:,n),conv_kernels{n});
			fgint(:,n) = temp(conv_kernels{end}(n):(conv_kernels{end}(n)+size(fgint,1)-1));
        end
		
        G = fgint*[cnim.subunits(sub_inds).weight]';

	end
	
	function [LL, pred_rate, mod_internals, LL_data] = eval_model( cnim, Robs, Xstims, varargin )
	% Usage: [LL, pred_rate, mod_internals, LL_data] = cnim.eval_model( Robs, Xstims, <eval_inds>, varargin )
	% Evaluates the model on the supplied data
	%
	% INPUTS:
	%   Robs: vector of observed data (leave empty [] if not interested in LL)
	%   Xstims: cell array of stimuli
	%   <eval_inds>: optional vector of indices on which to evaluate the model
	%   optional flags:
	%     'gain_funs': [TxK] matrix specifying gain at each timepoint for each subunit
	%
	% OUTPUTS:
	%   LL: log-likelihood per spike
	%   pred_rate: predicted firing rates (in counts/bin)
	%   mod_internals: struct containing the internal components of the model prediction
	%     G: is the total generating signal (not including the constant offset theta). 
	%        This is the sum of subunit outputs (weighted by their subunit weights w)
	%     fgint: is the output of each subunit after convolution
	%     gint: is the output of each subunit's linear filter
	%   LL_data: struct containing more detailed info about model performance:
	%     filt_pen: total regularization penalty on filter coefs 
	%     NL_pen total regularization penalty on filter upstream NLs
	%     nullLL: LL of constant-rate model

		Nsubs = length(cnim.subunits); % number of subunits
		NT = length(Robs); % number of time points
      
		% PROCESS INPUTS
		[eval_inds,parsed_options] = NIM.parse_varargin( varargin );
% 		NIM.validate_parsed_options( parsed_options, {'gain_funs'} );
		gain_funs = []; % default has no gain_funs
		if isfield( parsed_options, 'gain_funs' )
			gain_funs = parsed_options.gain_funs;
		end

		if ~iscell(Xstims)
			tmp = Xstims; clear Xstims
			Xstims{1} = tmp;
		end
		
		if isempty(Robs); Robs = zeros(size(Xstims{1},1),1); end	% if empty, make null list
		if size(Robs,2) > size(Robs,1); Robs = Robs'; end;			% make Robs a column vector
		cnim.check_inputs(Robs,Xstims,eval_inds,gain_funs);			% make sure input format is correct
		if cnim.spk_hist.spkhstlen > 0								% add in spike history term if needed
			Xspkhst = cnim.create_spkhist_Xmat( Robs );
		else			
			Xspkhst = [];  
		end

		% if specifying a subset of indices to train model params
		if ~isempty(eval_inds) 
			for nn = 1:length(Xstims)
				Xstims{nn} = Xstims{nn}(eval_inds,:); % grab the subset of indices for each stimulus element
			end
			Robs = Robs(eval_inds);
			if ~isempty(Xspkhst); Xspkhst = Xspkhst(eval_inds,:); end;
			if ~isempty(gain_funs); gain_funs = gain_funs(eval_inds,:); end;
		end
		
		[G, fgint, gint] = cnim.conv_process_stimulus(Xstims,1:Nsubs,gain_funs);
		if cnim.spk_hist.spkhstlen > 0 % add in spike history term if needed
			G = G + Xspkhst*cnim.spk_hist.coefs(:);  
		end
		
		pred_rate = cnim.apply_spkNL(G + cnim.spkNL.theta); % apply spiking NL
		[LL,norm_fact] = cnim.internal_LL(pred_rate,Robs);	% compute LL
		LL = LL/norm_fact; % normalize by spikes
    
		if nargout > 2 % if outputting model internals
			mod_internals.G = G;
			mod_internals.fgint = fgint;
			mod_internals.gint = gint;
		end
		
		if nargout > 3  % if we want more detailed model evaluation info, create an LL_data struct
			LL_data.LL = LL;
			[filt_penalties,NL_penalties] = cnim.get_reg_pen(); % get regularization penalty for each subunit
			LL_data.filt_pen = sum(filt_penalties)/norm_fact; % normalize by number of spikes
			LL_data.NL_pen = sum(NL_penalties)/norm_fact;
			avg_rate = mean(Robs);
			null_prate = ones(length(Robs),1)*avg_rate;
			nullLL = cnim.internal_LL(null_prate,Robs)/norm_fact;
			LL_data.nullLL = nullLL;
		end		
	end
	
end

%% ********************** Static Methods **********************************
methods (Static)
	
	function conv_params = create_conv_params(dims,varargin)
	% Usage: kernel_params = create_kernel_params(stim_dims,<varargin>)
	% Creates a struct containing kernel parameters; same struct as stim
	% parameters, but only a subset of the full functionality is necessary
	%
	% INPUTS:
	%     dims: dimensionality of the (time-embedded) stimulus, in the
	%         form: [nLags nXPix nYPix]. For convolution kernels just use
	%         nLags.
	%     optional_flags:
	%       ('upsampling',up_samp_fac): optional up-sampling of the stimulus from its raw form
	%       ('tent_spacing',tent_spacing): optional spacing of tent-basis functions when using a tent-basis
	%         representaiton of the stimulus (allows for the stimulus filters to be
	%         represented at a lower time resolution than other model components). 
	%         Default = []: no tent_bases
	%       ('boundary_conds',boundary_conds): vector of boundary conditions on each
	%           dimension (Inf is free, 0 is tied to 0, and -1 is periodi)
	%       ('split_pts',split_pts): specifies an internal boundary as a 3-element vector: [direction boundary_ind boundary_cond]
	% OUTPUTS:
	%     stim_params: struct of stimulus parameters
    
		% Set defaults
		stim_dt = 1; %cefault to unitless time
		stim_dx = 1; %default unitless spatial resolution
		up_samp_fac = 1; %default no temporal up-sampling
		tent_spacing = []; %default no tent-bases
		boundary_conds = [0 0 0]; %tied to 0 in all dims
		split_pts = []; %no internal boundaries
		
		% Parse inputs
		j = 1;
		while j <= length(varargin)
			switch lower(varargin{j})
				case 'stim_dt'
					stim_dt = varargin{j+1};
					j = j + 2;
				case 'stim_dx'
					stim_dx = varargin{j+1};
					j = j + 2;
				case 'upsampling'
					up_samp_fac = varargin{j+1};
					j = j + 2;
				case 'tent_spacing'
					tent_spacing = varargin{j+1};
					j = j + 2;
				case 'boundary_conds'
					boundary_conds = varargin{j+1};
					j = j + 2;
				case 'split_pts'
					split_pts = varargin{j+1};
					j = j + 2;
				otherwise
					error('Invalid input flag'); 
			end	
		end
		
		% Make sure stim_dims input has form [nLags nXPix nYPix] and concatenate 1's if necessary    
		while length(dims) < 3 %pad dims with 1s for book-keeping
			dims = cat(2,dims,1);
		end
		while length(boundary_conds) < 3
			boundary_conds = cat(2,boundary_conds,0); %assume free boundaries on spatial dims if not specified
		end
		
		dt = stim_dt/up_samp_fac; %model fitting dt
		conv_params = struct('dims',dims,'dt',dt,'dx',stim_dx,'up_fac',up_samp_fac,...
			'tent_spacing',tent_spacing,'boundary_conds',boundary_conds,'split_pts',split_pts);
	end 
	
end

end









