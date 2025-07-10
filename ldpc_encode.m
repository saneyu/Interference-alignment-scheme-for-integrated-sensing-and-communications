function encoded_bits = ldpc_encode(data_bits, n, k, H)
    % 输入：
    % data_bits - 信息比特流 (1 x k)
    % n - 编码后比特流的长度
    % k - 信息比特流的长度
    % H - 校验矩阵 (m x n)，其中 m = n - k
    
    % Step 1: 生成矩阵 G
    % 校验矩阵 H 需要是标准形式 [P | I]，其中 P 是 m x (n-k) 的矩阵，I 是 m x m 单位矩阵
    [m, n] = size(H);
    P = H(:, 1:n-k);  % P 是校验矩阵的前 n-k 列
    G = [eye(k), P'];  % 生成矩阵 G

    % Step 2: 编码，使用生成矩阵 G 将数据比特编码成代码字
    % 输入比特流为 data_bits, 编码后的比特流为 encoded_bits
    encoded_bits = mod(data_bits * G, 2);  % 对生成矩阵 G 进行矩阵乘法并对 2 取模
    
end
