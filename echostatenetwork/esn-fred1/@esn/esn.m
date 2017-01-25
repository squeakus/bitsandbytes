function net = esn(IUC, HUC, OUC)
% ESN.ESN create ESN object and initialize weights to 0

% set up number of units
net.numUnitsInput     = IUC;
net.numUnitsHidden    = HUC;
net.numUnitsOutput    = OUC;

% set up weights
net.weightsRecurrent  = zeros(HUC, HUC);
net.weightsBackward   = zeros(HUC, OUC);
net.weightsForward    = zeros(OUC, HUC+IUC+1);
net.weightsInput      = zeros(HUC, IUC+1);

% set up initial activation
net.actInital         = zeros(HUC, 1);

% create net object
net = class(net, 'esn');

    function disp(this);
        disp('HELLO');
    end
end