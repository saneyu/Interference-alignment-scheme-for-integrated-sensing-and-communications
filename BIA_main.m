clear all;
K = 3; % 用户数量
t0 = 2*K; % 时间槽
EbN0_dB = 5:5:30; % 目标每比特信噪比 (dB)
a = 3.5; % 衰落因子
f = 2.4e9; % 载波频率 (Hz)
c = 3e8; % 光速 (m/s)
lambda = c / f;
d0 = 1; % 参考距离 (m)
PL_d0 = (4 * pi * d0 / lambda)^2;
bits_length = 10;  % 数据比特的长度
data_bits = randi([0, 1], K, bits_length);  % 生成随机数据比特（K行，1000列）
H_L= generate_ldpc_matrix(2*bits_length,bits_length);
ldpcencode_data = zeros(K,2*bits_length);
for i = 1:K
    ldpcencode_data(i,:) = ldpc_encode(data_bits(i,:),2*bits_length,bits_length,H_L);
end
modulated_symbols = 1 - 2 * ldpcencode_data; % 0 → +1，1 → -1
%得到感知信号
sen_signal_all = [];
for t_idx = 1:(bits_length)
    sen_signal = sengene(K);  % 生成感知信号
    coded_sen = zeros(K, 2*K);
    for part_idx = 1:2
    % 提取当前部分的列
        start_col = (part_idx-1)*(K-1) + 1;
        end_col = part_idx*(K-1);
        current_part = sen_signal(:, start_col:end_col);
        
        % 编码当前部分
        coded_part = zeros(K, K);
        sum_current = sum(current_part, 2); % 第一列是总和
        coded_part(:, 1) = sum_current;
    
    % 后续列是总和减去原各列
        for j = 2:K
            coded_part(:, j) = sum_current - current_part(:, j-1);
        end
        
    % 将编码后的部分存入最终矩阵
        coded_sen(:, (part_idx-1)*K + 1 : part_idx*K) = coded_part;
    end
    sen_signal_all = [sen_signal_all coded_sen];
end

%得到ISAC信号
for i = 1:bits_length*2
    statr_col = 1 + (i-1)*K;
    end_col = i*K;
    isac_signal(:,statr_col:end_col) = modulated_symbols(:,i) + sen_signal_all(:,statr_col:end_col);
end
% 定义发射机和接收机的位置坐标
tx_positions = [0, 0; 100, 0; 200, 0]; % 三个发射机沿 x 轴分布
rx_positions = [50, 50; 150, 50; 250, 50]; % 三个接收机沿 x 轴分布，偏离发射机

K = size(ldpcencode_data, 1);  % 行数，表示比特流数量（K=3）
n = size(ldpcencode_data, 2);  % 每行的比特流长度（n=2000）

% % 创建一个空矩阵，用于存储调制后的符号
% modulated_data = zeros(K, n/2);  % 每行的调制符号数量是原来的1/2
% 
% % 对每行进行 QAM 调制
% for i = 1:K
%     % 将每行比特流重构为每 2 个比特一组
%     reshaped_bits = reshape(ldpcencode_data(i,:), [], 2);  % 每2个比特一组
%     origin_data = bi2de(reshaped_bits);
%     % 使用 4-QAM (QPSK) 调制
%     modulated_symbols = qammod(origin_data, 4);  % 调制后的符号，注意这是复数
% 
%     % 将调制后的符号存储到 modulated_data
%     modulated_data(i,:) = modulated_symbols;  % 每行存储调制后的符号
% end

Y_recover = zeros(K,2*bits_length);
for time_idx = 1:bits_length*2
    H = [];
    Y = zeros(K,K);
    Y_recv = zeros(K,K);
    Y_temp = zeros(K,1);
    
    % data_per_slot = data_bits(:,time_idx);
    for t = 1:K
        H_temp = zeros(K,K);
        for i = 1:K
            for j = 1:K
                % 计算 Tx j 到 Rx i 的距离
                dx = rx_positions(i, 1) - tx_positions(j, 1);
                dy = rx_positions(i, 2) - tx_positions(j, 2);
                d = sqrt(dx^2 + dy^2); % 欧氏距离
                
                % 计算路径损耗（线性值）
                PL = PL_d0 * (d / d0)^(2*n); % 对数距离模型
                PL = 1;
                % 小尺度衰落（瑞利衰落）
                h = (randn + 1j * randn) / sqrt(2); % 复高斯信道
                
                % 组合路径损耗和衰落
                H_temp(i, j) = h / sqrt(PL); % 信道增益（归一化功率）
            end
        end
        H = [H H_temp];
        % 按照H[11](1) H[12](1) H[13](1) H[11](2) H[12](2) H[13](2)...
        %     H[21](1) H[22](1) H[23](1) H[21](2) H[22](2) H[23](2)...这样得到信道矩阵
    end

    for t = 1:K
        Y(:,t) =  H(:,1+K*(t-1):K*t) * (isac_signal(:,t+K*(time_idx-1))-sen_signal_all(:,t+K*(time_idx-1)));
    end
    
    for k = 1:K
        Y_k = Y(k,:);
        Y_k = Y_k.';
        H_k = H(k,:);
        H_k_matrix = reshape(H_k,K,K).';
        W_zf = H_k_matrix' / (H_k_matrix * H_k_matrix' + 1e-6*eye(K));
        Y_k_recover = W_zf * Y_k;
        Y_recover(:,time_idx) = Y_k_recover;
    end

end
% 解调：计算 LLR
EbN0 = 10^(EbN0_dB(1)/10);
noise_power = 1 / EbN0;
received_llr = 2 * real(Y_recover) / noise_power;
max_iter = 50; % 最大迭代次数
decoded_bits = zeros(K, bits_length);
for k = 1:K
    decoded_bits(k, :) = ldpc_decode(H_L, received_llr(k, :), max_iter);
end

% 计算误码率
bernum = sum(data_bits(:) ~= decoded_bits(:));
ber = bernum / (K * bits_length);
disp(['误码率: ', num2str(ber)]);
