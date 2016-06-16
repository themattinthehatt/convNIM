classdef CONVSUBUNIT < SUBUNIT
	
% Class implementing the subunits comprising a convNIM model

% TOADD
% display_stim_filter needs work

properties
	
	% Inherited from SUBUNIT
	% filtK;			% filter coefficients, [dx1] array where d is the dimensionality of the target stimulus
	% NLtype;			% upstream nonlinearity type (string)
	% NLparams;			% vector of 'shape' parameters associated with the upstream NL function (for parametric functions)
	% NLnonpar;			% struct of settings and values for non-parametric fit (TBx TBy TBparams)
	% NLoffset;			% scalar offset value added to filter output
	% weight;			% subunit weight (typically +/- 1)
	% Xtarg;			% index of stimulus the subunit filter acts on
	% reg_lambdas;		% struct of regularization hyperparameters
	% Ksign_con;		% scalar defining any constraints on the filter coefs [-1 is negative con; +1 is positive con; 0 is no con]
	
	% convolution properties
	convK;				% convolution kernel coefficients
	conv_targ;			% index into convolution params struct; specifies number of lags, tent_spacing, etc.
	conv_reg_lambdas;	% struct of regularization hyperparameters
	convKsign_con;		% scalar defining any constraints on the conv coefs [-1 is negative con; +1 is positive con; 0 is no con]
	
end
	
properties (Hidden)
	
	% Inherited from SUBUNIT
	% allowed_subunitNLs = {'lin','quad','rectlin','rectpow','softplus','nonpar'}; % set of subunit NL functions currently implemented
	% TBy_deriv;   % internally stored derivative of tent-basis NL
	% scale;       % SD of the subunit output derived from most-recent fit
	
end

%% ********************** Constructor *************************************
methods 
	
	function csubunit = CONVSUBUNIT(init_filt, weight, NLtype, Xtarg, NLoffset, NLparams, Ksign_con, ...
									init_conv, conv_targ, convKsign_con)
	%
	% Constructor for CONVSUBUNIT. Mostly just passes needed properties
	% on to the SUBUNIT constructor. Appends kernel properties to the base 
	% SUBUNIT object and initializes the convolution kernel.
		
		if nargin == 0
			return % handle the no-input-argument case by returning a null model. This is important when initializing arrays of objects  
		end
		
		csubunit@SUBUNIT(init_filt, weight, NLtype, Xtarg, NLoffset, NLparams, Ksign_con);
		
		% error checking
		assert(conv_targ > 0, 'Invalid target for convolution kernel');
		assert(ismember(convKsign_con,[-1,0,1]), 'Invalid value for convKsign_con');
		
		% set properties
		csubunit.convK = init_conv;
		csubunit.conv_targ = conv_targ;
		csubunit.conv_reg_lambdas = csubunit.reg_lambdas; % initializes all to 0
		csubunit.convKsign_con = convKsign_con;
		
	end
	
end

%% ********************** Getting Methods *********************************
methods
	
	function convK = get_convK(csubunit)
	% Usage: convK = convsubunit.get_convK()
	%
	% Gets vector of convolution kernel coefs from the subunit
	
		convK = csubunit.convK;      
	end
	
end

%% ********************** Display Methods *********************************
methods
	
	function [] = display_filter( csubunit, dims, varargin )
	% Usage: [] = convsubunit.display_filter( dims, <plot_location>, varargin )
	% Plots subunit filter in a 1-row, 2-column subplot
	%
	% INPUTS:
	% plot_location: 3-integer list = [Fig_rows Fig_col Loc] arguments to subplot. Default = [1 2 1]
	% optional arguments (varargin)
	%	'color': enter to specify color of non-image-plots (default is blue). This could also have dashes etc
	%	'colormap': choose colormap for 2-D plots. Default is 'gray'
	%	'dt': enter if you want time axis scaled by dt
	%	'time_rev': plot temporal filter reversed in time (zero lag on right)
	%	'xt_rev': plot 2-D plots with time on x-axis
	%	'single': 1-D plots have only best latency/spatial position instead of all
	%	'notitle': suppress title labeling subunit type
	%	'xt-separable': do not plot x-t plot, but rather separable x and seperable t
	%	'xt-spatial': for xt-plots (1-D space), plot spatial instead of temporal as second subplot
	
		assert((nargin > 1) && ~isempty(dims), 'Must enter filter dimensions.' )

		[plotloc,parsed_options,modvarargin] = NIM.parse_varargin( varargin );
		if isempty(plotloc)
			plotloc = [1 2 1];
		end
		assert( plotloc(3) < prod(plotloc(1:2)), 'Invalid plot location.' )
		titleloc = plotloc(3);
		
		% get conv kernel dims
		cdims = length(csubunit.convK);
		cdims = [cdims 1 1];
		
		if prod(dims([2 3])) == 1 || prod(dims([1 3])) == 1
			% then 1-dimensional filter
	
			% plot stimulus filter
			subplot( plotloc(1), plotloc(2), plotloc(3) ); hold on
			csubunit.display_stim_filter( dims, modvarargin{:} );
			titleloc = plotloc(3);
			
			% plot convolution kernel
			subplot( plotloc(1), plotloc(2), plotloc(3) + 1 ); hold on
			csubunit.display_conv_kernel( cdims, modvarargin{:} );
			
		elseif dims(3) == 1
			
			% then space-time plot in first subplot
			subplot( plotloc(1), plotloc(2), plotloc(3) )
			k = reshape( csubunit.get_filtK(), dims(1:2) );
			imagesc( 1:dims(1),1:dims(2), k, max(abs(k(:)))*[-1 1] )
			if isfield(parsed_options,'colormap')
				colormap(parsed_options.colormap);
			else
				colormap('jet');
			end
				
			% Plot convolution kernel in second subplot
			subplot( plotloc(1), plotloc(2), plotloc(3)+1 ); hold on
			csubunit.display_conv_kernel( cdims, modvarargin{:} );
			
		else
			
			% 3-d filter
			subplot( plotloc(1), plotloc(2), plotloc(3) )
			csubunit.display_temporal_filter( dims, modvarargin{:} );
			subplot( plotloc(1), plotloc(2), plotloc(3)+1 )
			csubunit.display_spatial_filter( dims, modvarargin{:} );
			
		end
		
		if ~isfield( parsed_options, 'notitle' )
			subplot( plotloc(1), plotloc(2), titleloc )		% to put title back on the first
			title('Stim Filter','FontSize',10)
			subplot( plotloc(1), plotloc(2), titleloc + 1 ) % to put title back on the first
			title('Conv Kernel','FontSize',10)
		end
		
	end
	
	function [] = display_stim_filter( subunit, dims, varargin )
	% Usage: [] = convsubunit.display_stim_filter( dims, varargin )
	%
	% Plots subunit filter in a 1-row, 1-column subplot
	% INPUTS:
	%	  plot_location: 3-integer list = [Fig_rows Fig_col Loc] arguments to subplot. Default = [1 1 1]
	%	  optional arguments (varargin)
	%	    'single': plot single temporal function at best spatial position
	%	    'color': enter to specify color of non-image-plots (default is blue). This could also have dashes etc
	%			'colormap': choose colormap for 2-D plots. Default is 'gray'

		assert( (nargin > 1) && ~isempty(dims), 'Must enter filter dimensions.' )

		[~,parsed_options] = NIM.parse_varargin( varargin );
		if isfield(parsed_options,'color')
			clr = parsed_options.color;
		else
			clr = 'b';
		end
		if isfield(parsed_options,'colormap')
			clrmap = parsed_options.colormap;
		else
			clrmap = 'jet';
		end
		
		k = reshape( subunit.get_filtK(), [dims(1) prod(dims(2:3))] );
		if dims(2) == 1
			% just temporal filter
			% Time axis details
			NT = dims(1);
			if isfield(parsed_options,'dt')
				dt = parsed_options.dt;
			else
				dt = 1;
			end
			if isfield(parsed_options,'time_rev')
				ts = -dt*(0:NT-1);
			else
				ts = dt*(1:NT);
				if isfield(parsed_options,'dt')
					ts = ts-dt;  % if entering dt into time axis, shift to zero lag
				end
			end
		
			plot( k, clr, 'LineWidth',0.8 );
			hold on
			plot([ts(1) ts(end)],[0 0],'k--')
			L = max(k(:))-min(k(:));
			axis([1 dims(1) min(k(:))+L*[-0.1 1.1]])
		else
			% filter with single spatial dim
			plot( k, clr, 'LineWidth',0.8 );
			hold on
			L = max(k(:))-min(k(:));
			axis([1 dims(2) min(k(:))+L*[-0.1 1.1]])
		end
				
% 		if dims(3) == 1
% 			% then 1-dimensional spatial filter
% 			if isfield(parsed_options,'single')
% 				% then find best spatial
% 				[~,bestT] = max(std(k,1,2));
% 				k = k(bestT,:);
% 			end			
% 		else
% 			% then 2-dimensional spatial filter
% 			[~,bestlat] = max(max(abs(k')));
% 			Kmax = max(abs(k(:)));
% 
% 			imagesc( reshape(k(bestlat,:)/Kmax,dims(2:3)), [-1 1] )								
% 			colormap(clrmap)
% 		end
		
	end
	
	function [] = display_conv_kernel( subunit, dims, varargin )
	% Usage: [] = convsubunit.display_conv_kernel( dims, varargin )
	%
	% Plots temporal elements of convolution kernel in a 2-row, 1-column subplot
	% INPUTS:
	%	  plot_location: 3-integer list = [Fig_rows Fig_col Loc] arguments to subplot. Default = [1 2 1]
	%	  optional arguments (varargin)
	%	    'color': enter to specify color of non-image-plots (default is blue). This could also have dashes etc
	%	    'dt': enter if you want time axis scaled by dt
	%	    'time_rev': plot temporal filter reversed in time (zero lag on right)
	%	    'single': plot single temporal function at best spatial position
	
		assert((nargin > 1) && ~isempty(dims), 'Must enter filter dimensions.' )
		if dims(1) == 1
			% warning( 'No temporal dimensions to plot.' )
			return
		end

		[~,parsed_options] = NIM.parse_varargin( varargin );
		if isfield(parsed_options,'color')
			clr = parsed_options.color;
		else
			clr = 'b';
		end
		
		% Time axis details
		NT = dims(1);
		if isfield(parsed_options,'dt')
			dt = parsed_options.dt;
		else
			dt = 1;
		end
		if isfield(parsed_options,'time_rev')
			ts = -dt*(0:NT-1);
		else
			ts = dt*(1:NT);
			if isfield(parsed_options,'dt')
				ts = ts-dt;  % if entering dt into time axis, shift to zero lag
			end
		end
		
		% get kernel
		k = subunit.get_convK();
		L = max(k)-min(k);
				
		plot( ts, k, clr, 'LineWidth',0.8 );
		hold on
		plot([ts(1) ts(end)],[0 0],'k--')
		
		axis([min(ts) max(ts) min(subunit.convK)+L*[-0.1 1.1]])
		if isfield(parsed_options,'time_rev')
			box on
		else
			box off
		end
	end			
	
end

end



