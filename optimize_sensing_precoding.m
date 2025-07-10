function [W, X_m, total_power] = optimize_sensing_precoding(H_sense, P, SNR_dB, numRx, N, varargin)
% OPTIMIZE_SENSING_PRECODING - Optimization for sensing channel and precoding matrix design
%
% Inputs:
%   H_sense - Sensing channel matrix (1 x numTx)
%   P       - Power budget (scalar)
%   SNR_dB  - Signal-to-noise ratio in dB (scalar)
%   numRx   - Number of receive antennas (scalar)
%   N       - Number of training sequences
%   varargin - Optional parameters (name-value pairs):
%              'L' - Parameter L in the formula (default: N)
%              'tol' - Tolerance for water-filling algorithm (default: 1e-6)
%              'maxIter' - Maximum iterations for water-filling (default: 1000)
%              'verbose' - Display detailed information (default: true)
%
% Outputs:
%   W           - Optimized precoding matrix (numTx x numTx)
%   X_m         - Sensing matrix (numTx x N)
%   total_power - Total power consumption
%
% Formula implemented:
%   W_WF = sqrt(sigma_s^2 * N_R / L) * Q * [(mu0*I - Lambda^(-1))^+]^(1/2)
%   where ||W_WF||_F^2 = P

% Parse input parameters
p = inputParser;
addRequired(p, 'H_sense', @(x) ismatrix(x) && size(x,1) == 1);
addRequired(p, 'P', @(x) isscalar(x) && x > 0);
addRequired(p, 'SNR_dB', @(x) isscalar(x));
addRequired(p, 'numRx', @(x) isscalar(x) && x > 0 && mod(x,1) == 0);
addRequired(p, 'N', @(x) isscalar(x) && x > 0);
addParameter(p, 'L', [], @(x) isempty(x) || (isscalar(x) && x > 0));
addParameter(p, 'tol', 1e-6, @(x) isscalar(x) && x > 0);
addParameter(p, 'maxIter', 1000, @(x) isscalar(x) && x > 0);
addParameter(p, 'verbose', true, @islogical);

parse(p, H_sense, P, SNR_dB, numRx, N, varargin{:});

% Extract parameters
L = p.Results.L;
if isempty(L)
    L = N; % Default: L = N
end
tol = p.Results.tol;
maxIter = p.Results.maxIter;
verbose = p.Results.verbose;

% Calculate noise variance from SNR
SNR_linear = 10^(SNR_dB / 10);
sigma_s2 = 1 / SNR_linear; % Normalized noise variance

% Get number of transmit antennas
numTx = size(H_sense, 2);

% Display system parameters
if verbose
    fprintf('=== System Configuration ===\n');
    fprintf('  Number of Tx antennas (numTx): %d\n', numTx);
    fprintf('  Number of Rx antennas (numRx): %d\n', numRx);
    fprintf('  SNR: %.2f dB (linear: %.4f)\n', SNR_dB, SNR_linear);
    fprintf('  Noise variance (sigma_s2): %.6f\n', sigma_s2);
    fprintf('  Power budget (P): %.4f\n', P);
    fprintf('  Training sequences (N): %d\n', N);
    fprintf('  Parameter L: %d\n', L);
    fprintf('  Scaling factor: %.6f\n', sqrt(sigma_s2 * numRx / L));
    fprintf('\n');
end

% Validate channel matrix
if any(isnan(H_sense(:))) || any(isinf(H_sense(:)))
    error('Channel matrix H_sense contains NaN or Inf values');
end

% Compute channel covariance matrix R_H
R_H = H_sense' * H_sense; % Hermitian product, numTx x numTx matrix

if verbose
    fprintf('=== Channel Analysis ===\n');
    fprintf('Channel H_sense: ');
    for i = 1:numTx
        fprintf('[%.4f%+.4fi] ', real(H_sense(i)), imag(H_sense(i)));
    end
    fprintf('\n');
    fprintf('Channel covariance matrix R_H:\n');
    disp(R_H);
end

% Eigenvalue decomposition of R_H
[Q, Lambda] = eig(R_H);
Lambda = diag(Lambda); % Extract eigenvalues as a vector

% Sort eigenvalues and eigenvectors in descending order
[Lambda, idx] = sort(Lambda, 'descend');
Q = Q(:, idx);

if verbose
    fprintf('Eigenvalues of R_H (sorted): [');
    fprintf('%.6f ', Lambda);
    fprintf(']\n');
end

% Compute Lambda^{-1}, handle zero eigenvalues
Lambda_inv = zeros(numTx, numTx);
for i = 1:numTx
    if Lambda(i) > 1e-12 % Avoid division by zero
        Lambda_inv(i,i) = 1 / Lambda(i);
    else
        Lambda_inv(i,i) = 1e12; % For zero eigenvalues, set a very large value
        if verbose
            fprintf('Warning: Zero eigenvalue detected at position %d\n', i);
        end
    end
end

if verbose
    fprintf('Lambda_inv diagonal elements: [');
    fprintf('%.6f ', diag(Lambda_inv));
    fprintf(']\n\n');
end

% Water-filling algorithm implementation
% Formula: W_WF = sqrt(sigma_s^2 * N_R / L) * Q * [(mu0*I - Lambda^(-1))^+]^(1/2)
% Constraint: ||W_WF||_F^2 = P

scaling_factor_sq = sigma_s2 * numRx / L;

% Find valid eigenvalues for water-filling
valid_idx = diag(Lambda_inv) < 1e11;
if ~any(valid_idx)
    error('All eigenvalues are zero or too small for water-filling');
end

min_lambda_inv = min(diag(Lambda_inv(valid_idx, valid_idx)));
max_lambda_inv = max(diag(Lambda_inv(valid_idx, valid_idx)));

% Initial water level estimation
mu0_init = min_lambda_inv + P / (scaling_factor_sq * sum(valid_idx));

if verbose
    fprintf('=== Water-filling Algorithm ===\n');
    fprintf('Initial water level estimate: %.6f\n', mu0_init);
    fprintf('Valid eigenvalue range: [%.6f, %.6f]\n', min_lambda_inv, max_lambda_inv);
end

% Binary search for optimal water level
mu_low = min_lambda_inv;
mu_high = max(mu0_init * 2, min_lambda_inv + 10 * P / scaling_factor_sq);
iter = 0;
mu0 = mu0_init;
Sigma = [];

while iter < maxIter
    mu_mid = (mu_low + mu_high) / 2;
    
    % Compute Sigma matrix: [(mu0*I - Lambda^(-1))^+]^(1/2)
    Sigma_diag_temp = max(mu_mid - diag(Lambda_inv), 0);
    Sigma_temp = diag(sqrt(Sigma_diag_temp));
    
    % Compute corresponding W matrix
    W_temp = sqrt(scaling_factor_sq) * Q * Sigma_temp;
    
    % Compute Frobenius norm squared
    power_temp = norm(W_temp, 'fro')^2;
    
    if verbose && (iter < 10 || mod(iter, 100) == 0)
        fprintf('Iter %d: mu = %.8f, ||W||_F^2 = %.8f, target = %.8f, error = %.2e\n', ...
            iter+1, mu_mid, power_temp, P, abs(power_temp - P));
    end
    
    if abs(power_temp - P) < tol
        mu0 = mu_mid;
        Sigma = Sigma_temp;
        break;
    elseif power_temp > P
        mu_high = mu_mid;
    else
        mu_low = mu_mid;
    end
    
    iter = iter + 1;
    
    % Prevent infinite loop due to numerical precision
    if (mu_high - mu_low) < tol * 1e-6
        mu0 = mu_mid;
        Sigma_diag = max(mu0 - diag(Lambda_inv), 0);
        Sigma = diag(sqrt(Sigma_diag));
        break;
    end
end

% Check convergence
if iter >= maxIter
    warning('Water-filling algorithm did not converge within %d iterations', maxIter);
    if isempty(Sigma)
        Sigma_diag = max(mu0 - diag(Lambda_inv), 0);
        Sigma = diag(sqrt(Sigma_diag));
    end
end

if verbose
    fprintf('Final water level mu0: %.8f after %d iterations\n', mu0, iter);
    fprintf('Sigma diagonal elements: [');
    fprintf('%.6f ', diag(Sigma));
    fprintf(']\n\n');
end

% Construct final W matrix according to the formula
% W_WF = sqrt(sigma_s^2 * N_R / L) * Q * [(mu0*I - Lambda^(-1))^+]^(1/2)
scaling_factor = sqrt(scaling_factor_sq);
W = scaling_factor * Q * Sigma;

% Generate orthogonal training matrix S_D (DFT matrix)
S_D = zeros(numTx, N);
for k = 0:N-1
    for n = 0:numTx-1
        S_D(n+1, k+1) = exp(-1j * 2 * pi * k * n / numTx) / sqrt(numTx);
    end
end

% Normalize S_D to satisfy orthogonality condition
S_D = S_D * sqrt(numTx / N); % Adjust normalization

% Generate sensing matrix X_m
X_m = W * S_D;

% Calculate actual total power
total_power = norm(W, 'fro')^2;

% Display results
if verbose
    fprintf('=== Results ===\n');
    fprintf('Actual power consumption: %.8f (target: %.8f)\n', total_power, P);
    fprintf('Power constraint error: %.2e\n', abs(total_power - P));
    fprintf('W matrix statistics:\n');
    fprintf('  Frobenius norm: %.6f\n', norm(W, 'fro'));
    fprintf('  Max element magnitude: %.6f\n', max(abs(W(:))));
    fprintf('  Min element magnitude: %.6f\n', min(abs(W(:))));
    if rcond(W) > eps
        fprintf('  Condition number: %.6f\n', cond(W));
    else
        fprintf('  Matrix is singular\n');
    end
    fprintf('  Rank: %d (full rank: %d)\n', rank(W), numTx);
    
    fprintf('\nSensing matrix X_m statistics:\n');
    fprintf('  Size: %d x %d\n', size(X_m, 1), size(X_m, 2));
    fprintf('  Frobenius norm: %.6f\n', norm(X_m, 'fro'));
    fprintf('  Max element magnitude: %.6f\n', max(abs(X_m(:))));
    
    fprintf('\nTraining matrix S_D statistics:\n');
    fprintf('  Orthogonality check ||S_D*S_D^H - (N/numTx)*I||_F: %.2e\n', ...
        norm(S_D*S_D' - (N/numTx)*eye(numTx), 'fro'));
    fprintf('\n');
end

% Validation checks
if any(isnan(W(:))) || any(isinf(W(:)))
    error('Resulting W matrix contains NaN or Inf values');
end

if norm(W, 'fro') < 1e-10
    warning('Resulting W matrix is nearly zero - check input parameters');
end

end

% Example usage and test function
function example_usage()
    fprintf('=== Example Usage of optimize_sensing_precoding ===\n\n');
    
    % System parameters
    numTx = 4;          % Number of transmit antennas
    numRx = 2;          % Number of receive antennas
    N = 8;              % Number of training sequences
    P = 1;              % Power budget
    SNR_dB = 10;        % Signal-to-noise ratio in dB
    
    % Generate example sensing channel
    pathLossFactor = 2;
    distances = [10, 15, 20, 25]; % distances in meters
    
    H_sense = zeros(1, numTx);
    for tx = 1:numTx
        pathLoss = distances(tx)^(-pathLossFactor);
        H_sense(tx) = sqrt(pathLoss / 2) * (randn + 1j * randn); % Rayleigh fading
    end
    
    fprintf('Test Case 1: Basic usage\n');
    fprintf('------------------------\n');
    
    % Call the optimization function
    [W1, X_m1, power1] = optimize_sensing_precoding(H_sense, P, SNR_dB, numRx, N);
    
    fprintf('Test Case 2: Custom parameters\n');
    fprintf('-------------------------------\n');
    
    % Call with custom parameters
    [W2, X_m2, power2] = optimize_sensing_precoding(H_sense, P, SNR_dB, numRx, N, ...
        'L', 16, 'tol', 1e-8, 'maxIter', 2000, 'verbose', false);
    
    fprintf('Custom parameters result:\n');
    fprintf('Power consumption: %.8f\n', power2);
    fprintf('W matrix max element: %.6f\n', max(abs(W2(:))));
    
    fprintf('\nTest Case 3: Different SNR values\n');
    fprintf('----------------------------------\n');
    
    SNR_values = [0, 5, 10, 15, 20]; % dB
    powers = zeros(size(SNR_values));
    
    for i = 1:length(SNR_values)
        [W_temp, ~, power_temp] = optimize_sensing_precoding(H_sense, P, SNR_values(i), numRx, N, 'verbose', false);
        powers(i) = power_temp;
        fprintf('SNR = %2d dB: Power = %.8f, W_max = %.6f\n', ...
            SNR_values(i), power_temp, max(abs(W_temp(:))));
    end
    
    fprintf('\n=== Example completed successfully ===\n');
end

% Uncomment the line below to run the example
% example_usage();