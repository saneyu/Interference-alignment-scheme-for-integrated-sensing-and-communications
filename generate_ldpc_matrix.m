function H = generate_ldpc_matrix(n, k)
    % 生成LDPC校验矩阵H（系统码形式H = [P | I]）
    % 输入：
    %   n - 码字长度（总比特数）
    %   k - 信息比特数
    % 输出：
    %   H - 稀疏校验矩阵(m x n)，m = n-k

    m = n - k;  % 校验位数量
    P = randi([0, 1], k, m);
    H = [P' , eye(m)];
end