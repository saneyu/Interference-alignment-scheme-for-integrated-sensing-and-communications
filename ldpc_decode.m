function [decoded_bits] = ldpc_decode(H, received_llr, max_iter)
% LDPC_DECODE 基于 Belief Propagation 算法的 LDPC 解码
% 输入：
%   H: 奇偶校验矩阵 (m x n)
%   received_llr: 接收到的软信息（LLR，n x 1）
%   max_iter: 最大迭代次数
% 输出：
%   decoded_bits: 解码后的信息比特（k x 1）

    [m, n] = size(H);
    k = n - m; % 信息位长度

    % 初始化消息传递变量
    variable_to_check = zeros(m, n); % 变量节点到校验节点的消息
    check_to_variable = zeros(m, n); % 校验节点到变量节点的消息
    posterior = received_llr; % 后验信息初始化为接收 LLR

    % 迭代解码
    for iter = 1:max_iter
        % Step 1: 校验节点到变量节点的消息更新
        for i = 1:m
            for j = 1:n
                if H(i, j) == 1
                    % 计算乘积项（排除当前边）
                    product = 1;
                    for jj = 1:n
                        if H(i, jj) == 1 && jj ~= j
                            product = product * tanh(check_to_variable(i, jj) / 2);
                        end
                    end
                    variable_to_check(i, j) = log((1 + product)/(1 - product));
                end
            end
        end

        % Step 2: 变量节点到校验节点的消息更新
        for j = 1:n
            for i = 1:m
                if H(i, j) == 1
                    % 计算和项（排除当前边）
                    sum_messages = posterior(j);
                    for ii = 1:m
                        if H(ii, j) == 1 && ii ~= i
                            sum_messages = sum_messages + variable_to_check(ii, j);
                        end
                    end
                    check_to_variable(i, j) = sum_messages;
                end
            end
        end

        % Step 3: 更新后验信息
        for j = 1:n
            posterior(j) = received_llr(j);
            for i = 1:m
                if H(i, j) == 1
                    posterior(j) = posterior(j) + variable_to_check(i, j);
                end
            end
        end

        % Step 4: 硬判决并检查校验方程
        decoded_bits = double(posterior < 0); % LLR < 0 → 1，否则 → 0
        syndrome = mod(H * decoded_bits', 2); % 计算校验子
        
        % 如果校验子全为0，提前终止迭代
        if all(syndrome == 0)
            break;
        end
    end

    % 提取信息比特（假设系统码形式，前k位是信息位）
    decoded_bits = decoded_bits(1:k)';
end