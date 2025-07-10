clc;
clear;

%% 系统参数设置 
% rng(223,'twister');
numTx = 3; % 发射天线数
numRx = 3; % 接收天线数
snrRange = -15:5:35; % 信噪比范围（dB）
numSymbols = 1000; % 发送符号数量
txPower_dBm = 30; % 发射功率（dBm）
txPower = 10^((txPower_dBm-30)/10); % 发射功率（线性单位，瓦特）
pathLossFactor = 3.5; % 路径损耗指数
numMonteCarlo = 10; % 蒙特卡罗仿真次数

%% 新增通信系统参数
fc = 2.4e9; % 载波频率 2.4GHz
fs = 20e6; % 采样频率 20MHz
symbolRate = 1e6; % 符号速率 1Msps
samplesPerSymbol = fs / symbolRate; % 每符号采样点数
M = 4; % QPSK调制阶数

%% 滤波器设计参数
rolloff = 0.25; % 滚降因子
filterSpan = 6; % 滤波器跨度（符号周期数）

%% 距离和信道参数
distances = 50 + (100-50)*rand(numRx, numTx); % 通信用户距离 50-100米
distances_X = 50 + (100-50)*rand(1, numTx); % 感知目标距离 50-100米

% 生成多径衰落信道 H (numRx x numTx)
H = zeros(numRx, numTx);
for tx = 1:numTx
    % 路径损耗（线性尺度）：d^{-α}
    pathLoss = distances(tx)^(-pathLossFactor);

    % 瑞利衰落信道 + 路径损耗
    H(:, tx) = sqrt(pathLoss / 2) * (randn(numRx,1) + 1j*randn(numRx,1));
end


%% 设计发射和接收滤波器
txFilter = rcosdesign(rolloff, filterSpan, samplesPerSymbol, 'sqrt');
rxFilter = txFilter; % 匹配滤波器

%% 生成通信比特流和调制
txBits = randi([0 1], numTx* numSymbols * log2(M),1); % 生成比特流
% 将比特流重新整形为符号
% txBitsReshaped = reshape(txBits', log2(M), [])';
% txBitsDecimal = bi2de(txBitsReshaped, 'left-msb');

% QPSK调制
txSymbols_raw = qammod(txBits, M, 'InputType', 'bit', 'UnitAveragePower', true);
txSymbolsComm = reshape(txSymbols_raw, numTx, []);
% 逐列归一化（每列平均能量归一化为1）
for col = 1:size(txSymbolsComm,2)
    txSymbolsComm(:,col) = txSymbolsComm(:,col) / sqrt(mean(abs(txSymbolsComm(:,col)).^2));
end

% 功率归一化
% txSymbolsComm = sqrt(txPower) * txSymbolsComm;

%% 结果存储
mseErrors_proposed = zeros(length(snrRange), 1);
mseErrors_benchmark = zeros(length(snrRange), 1);
mseErrors_decode_benchmark = zeros(length(snrRange), 1);
berErrors = zeros(length(snrRange), 1);

%% 主仿真循环
for mc = 1:numMonteCarlo
    fprintf('Monte Carlo iteration: %d/%d\n', mc, numMonteCarlo);
    benchmarkErrors = 0;
    for idx = 1:length(snrRange)
        snr = snrRange(idx);
        noisePower = 10^(-snr/10);
        
        %% 感知信道估计 - 改进方案
        sensingErrors = 0;
        commErrors = 0;
        bitErrorCount = 0;
        benchmark_decode_Errors = 0;
        for symbolIdx = 1:2:numSymbols-1
            % 生成感知正交矩阵
            H_sense = zeros(1, numTx);
            for tx = 1:numTx
                pathLoss = distances_X(tx)^(-pathLossFactor);
                H_sense(tx) = sqrt(pathLoss / 2) * (randn + 1j*randn);
            end
            W = optimize_sensing_precoding(H_sense, 1/2, snr-3, 1, numTx-1);
            
            % 自定义参数
            % [W, X_m, total_power] = optimize_sensing_precoding(H_sense, P, ...
            %     'N', 16, 'sigma_s2', 0.005);
            X_sen_combined = [];
            Y_received_combined = [];
            
            % 处理连续两个符号
            for j = symbolIdx:symbolIdx+1
                % 生成感知信号
                sen_X = get_sen_X_improved(numTx);
                sen_X = W * sen_X;
                % 感知信号矩阵归一化（按整列向量归一化）
                % sen_X = sen_X / sqrt(mean(abs(sen_X).^2));
                % 逐列归一化（每列平均能量归一化为1）
                for col = 1:size(sen_X,2)
                    sen_X(:,col) = sen_X(:,col) / sqrt(mean(abs(sen_X(:,col)).^2));
                end

                X_sen_combined = [X_sen_combined, sen_X];
                
                % 生成ISAC信号
                ISAC_signal = generate_isac_signal(txSymbolsComm(:,j), sen_X, numTx, numRx);
                
                % 上变频和脉冲成形
                [txSignal_upsampled, t] = pulse_shaping_and_upconvert(ISAC_signal, txFilter, fc, fs, samplesPerSymbol);
                
                % 信道传输
                rxSignal = channel_transmission(txSignal_upsampled, H_sense, noisePower, pathLossFactor);
                
                % 接收端处理：下变频和匹配滤波
                [rxSignal_processed, rxSymbols] = downconvert_and_filter(rxSignal, rxFilter, fc, fs, samplesPerSymbol,numTx);
                
                % 感知信号恢复
                Y_rec = recover_sensing_signal(rxSymbols, numTx);
                Y_received_combined = [Y_received_combined, Y_rec];


                % TIN 的方法
                X_tin = ISAC_signal - txSymbolsComm(:,j);
                
                X_tin = X_tin / sqrt(mean(abs(X_tin(:)).^2));
                H_radar_est = rxSymbols * pinv(X_tin);
                benchmarkErrors = benchmarkErrors + mean(abs(H_radar_est - H_sense).^2);

                %非相干检测
                % decode_symblos = noncoherent_detection(rxSignal, fc, fs, samplesPerSymbol, M, noisePower);
                decoded_symbols = noncoherent_detection(rxSymbols, M);
                decoded_symbols_map = (qammod(decoded_symbols-1,M)/sqrt(2))';
                ISAC_signal_recover = generate_isac_signal(decoded_symbols_map,sen_X,numTx,numRx);
                H_code_est = rxSymbols * pinv(ISAC_signal_recover);
                benchmark_decode_Errors = benchmark_decode_Errors + + mean(abs(H_code_est - H_sense).^2);
            end
            
            % 感知信道估计
            H_recovered = estimate_sensing_channel(Y_received_combined, X_sen_combined, pathLossFactor);
            
            % 计算感知MSE
            sensingErrors = sensingErrors + mean(abs(H_recovered - H_sense).^2);
        end
        
        %% 基准ISAC感知方案
        
        % for symbolIdx = 1:numSymbols
        %     % 简单的雷达感知信号
        %     radarSymbol = (randn + 1j*randn) / sqrt(2);
        %     H_radar = (randn(numRx, 1) + 1j*randn(numRx, 1));
        % 
        %     % 添加噪声
        %     noise = sqrt(noisePower/2) * (randn + 1j*randn);
        %     Y_radar = H_radar * radarSymbol + H * txSymbolsComm(:,symbolIdx) + noise;
        %     fprintf('ISAC_signal size: [%d, %d]\n', size(ISAC_signal,1), size(ISAC_signal,2));
        %     fprintf('H_sense size: [%d, %d]\n', size(H_sense,1), size(H_sense,2));
        %     % 信道估计
        %     H_radar_est = Y_radar / radarSymbol;
        %     benchmarkErrors = benchmarkErrors + mean(abs(H_radar_est - H_radar).^2);
        % end
        
        %% 通信性能评估
        % 解调和比特错误率计算
        % for symbolIdx = 1:numSymbols
        %     % 接收信号处理（简化版本用于BER计算）
        %     noise_comm = sqrt(noisePower/2) * (randn(numRx, numTx) + 1j*randn(numRx, numTx));
        %     rxSymbols_comm = H * txSymbolsComm(:,symbolIdx) + noise_comm;
        % 
        %     % 简单的ZF接收
        %     rxSymbols_detected = pinv(H) * rxSymbols_comm;
        % 
        %     % 解调
        %     rxBitsDecimal = qamdemod(rxSymbols_detected, M, 'UnitAveragePower', true);
        %     rxBits = de2bi(rxBitsDecimal, log2(M), 'left-msb');
        %     rxBits = reshape(rxBits', [], numTx)';
        % 
        %     % 计算比特错误
        %     originalBits = txBits(:, (symbolIdx-1)*log2(M)+1:symbolIdx*log2(M));
        %     bitErrorCount = bitErrorCount + sum(sum(originalBits ~= rxBits));
        % end
        % 
        % 存储结果
        mseErrors_proposed(idx) = mseErrors_proposed(idx) + sensingErrors / numSymbols;
        mseErrors_benchmark(idx) = mseErrors_benchmark(idx) + benchmarkErrors / numSymbols;
        mseErrors_decode_benchmark(idx) = mseErrors_decode_benchmark(idx) + benchmark_decode_Errors / numSymbols;
        % berErrors(idx) = berErrors(idx) + bitErrorCount / (numSymbols * numTx * log2(M));
    end
end

%% 平均结果
mseErrors_proposed = mseErrors_proposed / numMonteCarlo;
mseErrors_benchmark = mseErrors_benchmark / numMonteCarlo;
mseErrors_decode_benchmark = mseErrors_decode_benchmark / numMonteCarlo;
berErrors = berErrors / numMonteCarlo;

% 转换为dB
mseErrors_proposed_dB = 10 * log10(mseErrors_proposed);
mseErrors_benchmark_dB = 10 * log10(mseErrors_benchmark);
mseErrors_decode_benchmark_dB = 10*log10(mseErrors_decode_benchmark);
%% 结果可视化
figure;
plot(snrRange(3:end), mseErrors_proposed_dB(3:end), '-o', 'LineWidth', 2);
hold on;
plot(snrRange(3:end), mseErrors_benchmark_dB(3:end), '--s', 'LineWidth', 2);
hold on;
plot(snrRange(3:end), mseErrors_decode_benchmark_dB(3:end), ':r', 'LineWidth', 2);
xlabel('SNR (dB)');
ylabel('Channel Estimation Error (dB)');
legend('Proposed ISAC Sensing', 'Benchmark ISAC Sensing', 'decode ISAC Sensing', 'Location', 'best');
% title('感知性能比较');
grid on;

% subplot(2,1,2);
% semilogy(snrRange, berErrors, '-^', 'LineWidth', 2);
% xlabel('SNR (dB)');
% ylabel('Bit Error Rate');
% title('通信性能 - 比特错误率');
% grid on;

%% 显示性能增益
fprintf('\n=== 性能分析 ===\n');
fprintf('感知性能增益 (dB): \n');
disp(mseErrors_benchmark_dB - mseErrors_proposed_dB);
fprintf('在SNR=30dB时的BER: %.2e\n', berErrors(end-1));

%% 辅助函数定义

function ISAC_signal = generate_isac_signal(commSymbols, sensingMatrix, numTx, numRx)
    % 通信符号归一化
    % commSymbols = commSymbols / sqrt(mean(abs(commSymbols).^2));

    % 感知信号矩阵归一化（按整列向量归一化）
    % sensingMatrix = sensingMatrix / sqrt(mean(abs(sensingMatrix(:)).^2));

    % 初始化ISAC联合信号
    ISAC_signal = zeros(numTx, numRx);

    for i = 1:numTx
        if i == 1
            ISAC_signal(:,i) = commSymbols + sum(sensingMatrix, 2);
        else
            ISAC_signal(:,i) = commSymbols + sum(sensingMatrix, 2) - sensingMatrix(:,i-1);
        end
    end
end


function [txSignal_upsampled, t] = pulse_shaping_and_upconvert(baseband_signal, txFilter, fc, fs, samplesPerSymbol)
    % 脉冲成形和上变频 - 修正版本
    
    [numTx, numSymbols] = size(baseband_signal);
    
    % 为每个符号分配采样点
    total_samples = numSymbols * samplesPerSymbol + length(txFilter) - 1;
    txSignal_shaped = zeros(numTx, total_samples);
    
    for tx_idx = 1:numTx
        % 上采样：在符号位置插入符号值
        upsampled = zeros(1, numSymbols * samplesPerSymbol);
        for sym_idx = 1:numSymbols
            sample_idx = (sym_idx - 1) * samplesPerSymbol + 1;
            upsampled(sample_idx) = baseband_signal(tx_idx, sym_idx);
        end
        
        % 脉冲成形
        txSignal_shaped(tx_idx, :) = conv(upsampled, txFilter);
    end
    
    % 生成时间向量
    t = (0:size(txSignal_shaped,2)-1) / fs;
    
    % 上变频
    txSignal_upsampled = zeros(size(txSignal_shaped));
    for i = 1:numTx
        % txSignal_upsampled(i,:) = real(txSignal_shaped(i,:) .* exp(1j*2*pi*fc*t));
        txSignal_upsampled(i,:) = txSignal_shaped(i,:) .* exp(1j*2*pi*fc*t);
    end
end

function rxSignal = channel_transmission(txSignal, H_channel, noisePower, pathLossFactor)
    % 信道传输模拟 - 修正版本
    [numTx, numSamples] = size(txSignal);
    [numRx, ~] = size(H_channel);
    
    % 初始化接收信号
    rxSignal = zeros(numRx, numSamples);
    
    % 对每个时间采样点进行信道卷积
    for sample_idx = 1:numSamples
        % 提取当前时刻所有发射天线的信号
        tx_sample = txSignal(:, sample_idx); % numTx × 1
        
        % 通过信道矩阵
        rx_sample = (H_channel * tx_sample) / sqrt(pathLossFactor); % numRx × 1
        
        % 存储到接收信号中
        rxSignal(:, sample_idx) = rx_sample;
    end
    
    % 添加噪声
    noise = sqrt(noisePower/2) * (randn(size(rxSignal)) + 1j*randn(size(rxSignal)));
    rxSignal = rxSignal + real(noise);
end

function [rxSignal_processed, rxSymbols] = downconvert_and_filter(rxSignal, rxFilter, fc, fs, samplesPerSymbol,numTX)
    % 下变频和匹配滤波
    [numRx, numSamples] = size(rxSignal);
    
    % 生成时间向量
    t = (0:numSamples-1) / fs;
    
    % 下变频
    rxSignal_downconv = zeros(numRx, numSamples);
    for i = 1:numRx
        rxSignal_complex = rxSignal(i,:) .* exp(-1j*2*pi*fc*t);
        rxSignal_downconv(i,:) = rxSignal_complex;
    end
    
    % 匹配滤波
    rxSignal_filtered = zeros(numRx, numSamples + length(rxFilter) - 1);
    for i = 1:numRx
        rxSignal_filtered(i,:) = conv(rxSignal_downconv(i,:), rxFilter);
    end
    
    % 计算滤波器延迟
    % 抽取中心3个符号
    delay = floor((length(rxFilter) - 1) / 2);
    symbol_pos = delay + (0:numTX-1)*samplesPerSymbol + 1;
    
    rxSymbols = rxSignal_filtered(:, symbol_pos);

    delay = floor((length(rxFilter) - 1) / 2);
    startIdx = delay + 1;
    
    % 下采样 - 提取符号
    available_samples = size(rxSignal_filtered, 2) - startIdx + 1;
    num_symbols = floor(available_samples / samplesPerSymbol);
    
    % rxSymbols = zeros(numRx, num_symbols);
    % for sym_idx = 1:num_symbols
    %     sample_idx = startIdx + (sym_idx - 1) * samplesPerSymbol;
    %     rxSymbols(:, sym_idx) = rxSignal_filtered(:, sample_idx);
    % end
    
    rxSignal_processed = rxSignal_filtered;
    
    % 调试信息
    fprintf('Debug: Filter delay: %d samples\n', delay);
    fprintf('Debug: Available samples for symbol extraction: %d\n', available_samples);
    fprintf('Debug: Calculated num_symbols: %d\n', num_symbols);
    fprintf('Debug: Final rxSymbols size: [%d, %d]\n', size(rxSymbols,1), size(rxSymbols,2));
end

function Y_recovered = recover_sensing_signal(rxSymbols, numTx)
    % 恢复感知信号
    Y_recovered = zeros(size(rxSymbols,1), size(rxSymbols,2)-1);
    
    for i = 1:size(rxSymbols,2)-1
        Y_recovered(:,i) = rxSymbols(:,1) - rxSymbols(:,i+1);
    end
end

function H_estimated = estimate_sensing_channel(Y_combined, X_combined, pathLossFactor)
    % 感知信道估计
    H_estimated = Y_combined * sqrt(pathLossFactor) * pinv(X_combined);
end


% function decoded_symbols = noncoherent_detection(rxSymbols, M)
%     % 非相干检测解码通信信号
%     % 输入：
%     %   rxSymbols - 匹配滤波和下采样后的符号（numRx x numSymbols）
%     %   M - 调制阶数（QPSK 为 4）
% 
%     numSymbols = size(rxSymbols, 2);
%     decoded_symbols = zeros(1, numSymbols);
% 
%     for sym_idx = 1:numSymbols
%         if sym_idx == 1
%             % 第一个符号跳过或使用参考（这里假设为 1）
%             decoded_symbols(sym_idx) = 1;
%         else
%             % 差分检测
%             phase_diff = angle(rxSymbols(1, sym_idx)) - angle(rxSymbols(1, sym_idx-1));
%             phase_diff = mod(phase_diff + pi, 2*pi) - pi; % 归一化到 [-π, π]
% 
%             % QPSK 相位映射
%             if phase_diff >= -pi/4 && phase_diff < pi/4
%                 decoded_symbols(sym_idx) = 1; % 0°
%             elseif phase_diff >= pi/4 && phase_diff < 3*pi/4
%                 decoded_symbols(sym_idx) = 2; % 90°
%             elseif phase_diff >= 3*pi/4 || phase_diff < -3*pi/4
%                 decoded_symbols(sym_idx) = 3; % 180°
%             else
%                 decoded_symbols(sym_idx) = 4; % 270°
%             end
%         end
%     end
% end
% function decoded_symbols = noncoherent_detection(rxSymbols, M)
%     % 非相干差分QPSK解调（单天线接收）
%     % 输入:
%     %   rxSymbols - 接收符号（1 × N）
%     %   M         - 调制阶数（例如 QPSK 为 4）
% 
%     numSymbols = size(rxSymbols, 2);
%     decoded_symbols = zeros(1, numSymbols);
% 
%     % QPSK 标准相位角（单位圆均匀分布）
%     ref_phases = angle(qammod(0:M-1, M));  % [0, pi/2, pi, -pi/2] for QPSK
% 
%     % 第一个符号没有参考，默认映射为 0
%     decoded_symbols(1) = 0;
% 
%     for sym_idx = 2:numSymbols
%         phase_diff = angle(rxSymbols(1, sym_idx)) - angle(rxSymbols(1, sym_idx-1));
%         % 归一化到 [-pi, pi]
%         phase_diff = mod(phase_diff + pi, 2*pi) - pi;
% 
%         % 找最接近的相位
%         [~, symbol_idx] = min(abs(mod(phase_diff - ref_phases + pi, 2*pi) - pi));
%         decoded_symbols(sym_idx) = symbol_idx - 1;  % 0-based
%     end
% end
function decoded_symbols = noncoherent_detection(rxSymbols, M)
    % 非相干检测解码通信信号
    % 输入：
    %   rxSymbols - 匹配滤波和下采样后的符号（numRx x numSymbols）
    %   M - 调制阶数（QPSK 为 4）
    
    numSymbols = size(rxSymbols, 2);
    decoded_symbols = zeros(1, numSymbols);
    
    for sym_idx = 1:numSymbols
        if sym_idx == 1
            % 第一个符号跳过或使用参考（这里假设为 1）
            decoded_symbols(sym_idx) = 1;
        else
            % 差分检测
            phase_diff = angle(rxSymbols(1, sym_idx)) - angle(rxSymbols(1, sym_idx-1));
            phase_diff = mod(phase_diff + pi, 2*pi) - pi; % 归一化到 [-π, π]
            
            % QPSK 相位映射
            if phase_diff >= -pi/4 && phase_diff < pi/4
                decoded_symbols(sym_idx) = 1; % 0°
            elseif phase_diff >= pi/4 && phase_diff < 3*pi/4
                decoded_symbols(sym_idx) = 2; % 90°
            elseif phase_diff >= 3*pi/4 || phase_diff < -3*pi/4
                decoded_symbols(sym_idx) = 3; % 180°
            else
                decoded_symbols(sym_idx) = 4; % 270°
            end
        end
    end
end
function X_filtered = apply_same_processing(X_signal, numTx, numRx)
    % 对参考信号应用相同的滤波处理
    % 这里简化处理，实际应该包括完整的脉冲成形和匹配滤波
    
    % 滤波器参数（应该与主代码一致）
    rolloff = 0.25;
    filterSpan = 6;
    samplesPerSymbol = 20; % fs/symbolRate = 20MHz/1MHz
    
    % 设计相同的滤波器
    txFilter = rcosdesign(rolloff, filterSpan, samplesPerSymbol, 'sqrt');
    
    % 对每个发射天线的信号进行相同处理
    X_filtered = zeros(size(X_signal));
    for tx_idx = 1:numTx
        for rx_idx = 1:numRx
            % 上采样
            upsampled = zeros(1, samplesPerSymbol);
            upsampled(1) = X_signal(tx_idx, rx_idx);
            
            % 脉冲成形
            shaped = conv(upsampled, txFilter);
            
            % 匹配滤波
            filtered = conv(shaped, txFilter);
            
            % 符号定时恢复（简化版本）
            delay = length(txFilter) - 1;
            symbol_pos = delay + samplesPerSymbol;
            if symbol_pos <= length(filtered)
                X_filtered(tx_idx, rx_idx) = filtered(symbol_pos);
            else
                X_filtered(tx_idx, rx_idx) = X_signal(tx_idx, rx_idx); % 回退
            end
        end
    end
end
