%% Useful links

% 1. https://www.mathworks.com/help/deeplearning/ref/dlarray.dlode45.html#mw_4924e83e-975d-4636-a89b-d9bcbfe63959

% 2. https://www.mathworks.com/help/deeplearning/ug/dynamical-system-modeling-using-neural-ode.html

% A note about all the "CB"'s...
% These are the "formats" for the matlab dlarray.
% C: channel
% B: batch

% A note on how the autodiff through ODE is working:
% -by default, dlode45 uses "Direct" Gradient Mode, which simply computes
% derivatives of the discreteized forward RK45 solution.
% - alternatively, you can set GradientMode to "adjoint", and MATLAB will
% solve the adjoint ODE system to compute gradients.

%% Begin basic demo code
% Define the Lorenz-63 parameters
sigma = 10;
rho = 28;
beta = 8/3;

% In this example, we have 3 channels (i.e. components of our state/parameter)
% and 1 batch (i.e., just 1 initial condition y0 and just 1 system
% parameterization theta)

% Set the initial conditions and time span
y0 = dlarray([10; 3; 20], "CB"); % Initial condition

tspan = 0:0.01:1;

% Set the parameters
theta = dlarray([sigma; rho; beta], "CB");

% Solve the ODE at true IC and true params
Y_data = dlode45(@dl_l63, tspan, y0, theta);

% Plot the time series data
figure;
plot(squeeze(extractdata(Y_data(1,:,:))));hold on
plot(squeeze(extractdata(Y_data(2,:,:))));hold on;
plot(squeeze(extractdata(Y_data(3,:,:))));
xlabel('Time');
ylabel('State Variables');
title('Lorenz-63 Time Series Data');
grid on;
legend('y_1', 'y_2', 'y_3');

%% Verify that Loss and Grads are 0 when expected...and non-zero when expected
% Evaluate Loss and Gradients at true IC and params (should be all 0's)
[loss, dloss_dx0, dloss_dparams, dloss_dtarget] = dlfeval(@modelLoss, y0, theta, Y_data, tspan)

% Now, evaluate Loss and Gradients at perturbed IC and perturbed params
% (should be non-zero's since there is now mismatch)
[loss, dloss_dx0, dloss_dparams, dloss_dtarget] = dlfeval(@modelLoss, y0+0.1, theta+0.1, Y_data, tspan)

%% Sweep over different values of sigma to validate gradient computations w.r.t. ODE parameters
sigma_vals = 5:0.5:15;
loss_vals = zeros(size(sigma_vals));
grad_vals = zeros(size(sigma_vals));

for i = 1:length(sigma_vals)
    % Update sigma value in parameters
    theta_sweep = dlarray([sigma_vals(i); rho; beta], "CB");
    
    % Compute loss and gradients
    [loss, ~, dloss_dparams, ~] = dlfeval(@modelLoss, y0, theta_sweep, Y_data, tspan);
    
    % Store loss and gradient values
    loss_vals(i) = extractdata(loss);
    grad_vals(i) = extractdata(dloss_dparams(1)); % Gradient w.r.t sigma
end

% Plot loss and gradient as a function of sigma on the same plot
figure;
yyaxis left;
plot(sigma_vals, loss_vals, '-o');
ylabel('Loss');

yyaxis right;
plot(sigma_vals, grad_vals, '-o');
ylabel('Gradient w.r.t \sigma');

xlabel('\sigma');
title('Loss and its gradient as a Function of \sigma');
grid on;
hold on;

% Add vertical line for true sigma value
xline(sigma, '--k', 'LineWidth', 1.5);

% Add horizontal line for zero gradient
yline(0, '--k', 'LineWidth', 1.5);

hold off;

%% Sweep over different values of y0(1) to validate gradient computations w.r.t. ODE initial conditions
% Sweep over different values of the first component of initial condition
y0_vals = 3:0.5:18;
loss_vals_ic = zeros(size(y0_vals));
grad_vals_ic = zeros(size(y0_vals));

for i = 1:length(y0_vals)
    % Update the first component of the initial condition
    y0_sweep = dlarray([y0_vals(i); 3; 20], "CB");
    
    % Compute loss and gradients
    [loss, dloss_dx0, ~, ~] = dlfeval(@modelLoss, y0_sweep, theta, Y_data, tspan);
    
    % Store loss and gradient values
    loss_vals_ic(i) = extractdata(loss);
    grad_vals_ic(i) = extractdata(dloss_dx0(1)); % Gradient w.r.t first component of initial condition
end

% Plot loss and gradient as a function of the first component of initial condition on the same plot
figure;
yyaxis left;
plot(y0_vals, loss_vals_ic, '-o');
ylabel('Loss');

yyaxis right;
plot(y0_vals, grad_vals_ic, '-o');
ylabel('Gradient w.r.t y0_1');

xlabel('y0_1');
title('Loss and its gradient as a Function of Initial Condition y0_1');
grid on;
hold on;

% Add vertical line for true initial condition value
xline(extractdata(y0(1)), '--k', 'LineWidth', 1.5);

% Add horizontal line for zero gradient
yline(0, '--k', 'LineWidth', 1.5);

hold off;

%% Functions declarations
% define loss function for discrepancy between ODE solution vs "target"
% data
function [loss, dloss_dx0, dloss_dparams, dloss_dtarget] = modelLoss(x0, params, target, times)
    y_pred = dlode45(@dl_l63, times, x0, params);
    loss = l2loss(y_pred, target); % log-probability_sigma
    dloss_dx0 = dlgradient(loss, x0);
    dloss_dparams = dlgradient(loss, params); % dlgradient needs to be INSIDE of modelLoss (or more specifically, inside of dfeval)
    dloss_dtarget = dlgradient(loss, target);
end

% define ode RHS
function dy_dt = dl_l63(t, y, p)
% y must be a dlarray
    dy_dt = dlarray([
        p(1) * (y(2) - y(1));
        y(1) * (p(2) - y(3)) - y(2);
        y(1) * y(2) - p(3) * y(3)
    ], "CB");
end
