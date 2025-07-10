function [sen_X] = get_sen_X_improved(nt)
% 改进的正交感知矩阵生成函数
% 输入: nt - 发射天线数量
% 输出: sen_X - nt×(nt-1)的正交感知矩阵

    % 初始化感知矩阵
    sen_X = zeros(nt, nt-1);
    
    % 改进的正交化过程 - 使用Gram-Schmidt正交化
    % 但增加了功率约束和优化的初始向量选择
    
    for i = 1:nt-1
        maxIterations = 100; % 最大迭代次数
        iteration = 0;
        
        while iteration < maxIterations
            iteration = iteration + 1;
            
            % 生成优化的初始向量 - 使用更好的随机分布
            if i == 1
                % 第一个向量：使用均匀相位分布
                phases = 2*pi*rand(nt, 1);
                v = exp(1j * phases) / sqrt(nt);
            else
                % 后续向量：在前面向量的正交空间中生成
                v = (randn(nt, 1) + 1j * randn(nt, 1)) / sqrt(2*nt);
            end
            
            % Gram-Schmidt正交化过程
            for j = 1:i-1
                % 投影到已有向量上
                projection = (sen_X(:, j)' * v) * sen_X(:, j);
                v = v - projection;
            end
            
            % 归一化
            v_norm = norm(v);
            if v_norm > 1e-10  % 避免数值问题
                v = v / v_norm;
            else
                continue; % 重新生成向量
            end
            
            % 检验正交性 - 更严格的条件
            orthogonal = true;
            if i > 1
                for j = 1:i-1
                    inner_product = abs(sen_X(:, j)' * v);
                    if inner_product > 1e-12  % 更严格的正交性要求
                        orthogonal = false;
                        break;
                    end
                end
            end
            
            % 额外的优化：确保感知矩阵具有良好的条件数
            if orthogonal
                % 临时矩阵用于条件数检查
                temp_matrix = [sen_X(:, 1:i-1), v];
                if i == 1 || cond(temp_matrix' * temp_matrix) < 1e10
                    sen_X(:, i) = v;
                    break;
                end
            end
            
            % 如果达到最大迭代次数，使用当前最好的结果
            if iteration == maxIterations && v_norm > 1e-10
                warning('最大迭代次数达到，使用当前结果');
                sen_X(:, i) = v / v_norm;
                break;
            end
        end
    end
    
    % 最终验证和优化
    sen_X = optimize_sensing_matrix(sen_X, nt);
end

function [optimized_X] = optimize_sensing_matrix(sen_X, nt)
% 对生成的感知矩阵进行最终优化
% 确保矩阵具有良好的感知性能

    optimized_X = sen_X;
    
    % 检查矩阵的条件数
    condition_number = cond(sen_X' * sen_X);
    
    if condition_number > 1e8
        warning('感知矩阵条件数较大: %.2e，进行重新正交化', condition_number);
        
        % 使用QR分解进行重新正交化
        [Q, R] = qr(sen_X, 0);
        
        % 确保对角元素为正
        signs = sign(diag(R));
        signs(signs == 0) = 1;
        optimized_X = Q * diag(signs);
    end
    
    % 功率归一化 - 确保每列的功率为1
    for i = 1:size(optimized_X, 2)
        col_power = norm(optimized_X(:, i))^2;
        if col_power > 1e-10
            optimized_X(:, i) = optimized_X(:, i) / sqrt(col_power);
        end
    end
    
    % 验证最终结果
    verify_sensing_matrix(optimized_X);
end

function verify_sensing_matrix(sen_X)
% 验证感知矩阵的质量
    
    [nt, num_cols] = size(sen_X);
    
    % 检查正交性
    gram_matrix = sen_X' * sen_X;
    off_diagonal_max = max(max(abs(gram_matrix - eye(num_cols))));
    
    if off_diagonal_max > 1e-10
        warning('感知矩阵正交性检查失败，最大非对角元素: %.2e', off_diagonal_max);
    end
    
    % 检查功率归一化
    power_vector = diag(gram_matrix);
    power_deviation = max(abs(power_vector - 1));
    
    if power_deviation > 1e-10
        warning('感知矩阵功率归一化检查失败，最大功率偏差: %.2e', power_deviation);
    end
    
    % 检查条件数
    condition_number = cond(gram_matrix);
    if condition_number > 1e6
        warning('感知矩阵条件数较大: %.2e', condition_number);
    end
    
    % 输出矩阵质量信息
    % fprintf('感知矩阵质量报告:\n');
    % fprintf('  - 矩阵维度: %d × %d\n', nt, num_cols);
    % fprintf('  - 正交性偏差: %.2e\n', off_diagonal_max);
    % fprintf('  - 功率偏差: %.2e\n', power_deviation);
    % fprintf('  - 条件数: %.2e\n', condition_number);
end

% 额外的实用函数：生成具有特定性质的感知矩阵

function [sen_X] = generate_dft_sensing_matrix(nt)
% 基于DFT的感知矩阵生成（备选方案）
% 适用于需要特定频域特性的场景

    if nt < 2
        error('发射天线数量必须至少为2');
    end
    
    % 生成DFT矩阵
    dft_matrix = dftmtx(nt) / sqrt(nt);
    
    % 取前nt-1列作为感知矩阵
    sen_X = dft_matrix(:, 1:nt-1);
    
    % 验证矩阵质量
    verify_sensing_matrix(sen_X);
end

function [sen_X] = generate_hadamard_sensing_matrix(nt)
% 基于Hadamard矩阵的感知矩阵生成（备选方案）
% 适用于实数信号处理场景

    % Hadamard矩阵只对特定维度存在
    valid_sizes = [2, 4, 8, 12, 16, 20, 24, 28, 32];
    
    if ~ismember(nt, valid_sizes)
        error('Hadamard矩阵不支持维度 %d', nt);
    end
    
    % 生成Hadamard矩阵
    H = hadamard(nt) / sqrt(nt);
    
    % 转换为复数形式并取前nt-1列
    sen_X = complex(H(:, 1:nt-1));
    
    % 验证矩阵质量
    verify_sensing_matrix(sen_X);
end