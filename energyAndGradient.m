function [E, grad] = energyAndGradient(D, IL, IR, lambda_matrix)
    % 输入:
    %   D - 当前视差图
    %   IL, IR - 左右图像
    %   lambda_matrix - 区域感知的平滑权重矩阵
    %
    % 输出:
    %   E - 总能量（数据项 + 平滑项）
    %   grad - 关于 D 的梯度（列向量）

    % 鲁棒性惩罚函数
    rho       = @(x) sqrt(x.^2 + 1e-6);
    rho_prime = @(x) x ./ sqrt(x.^2 + 1e-6);

    [rows, cols] = size(D);
    num_pixels   = rows * cols;
    grad         = zeros(num_pixels, 1);
    direction_weight = 1;

    E_data   = 0;
    E_smooth = 0;

    neighbors = [0, 1; 0, -1; 1, 0; -1, 0];
    num_neighbors = size(neighbors, 1);

    % 数据项 
    for i = 1:rows
        for j = 1:cols
            d = D(i, j);
            j_right = j - d;  % 从左图向右图投影

            if j_right >= 1 && j_right <= cols
                Il = double(IL(i, j));
                Ir = double(IR(i, round(j_right)));  
                diff = Il - Ir;

                E_data = E_data + rho(diff^2);

                idx = sub2ind([rows, cols], i, j);
                grad(idx) = grad(idx) - 2 * rho_prime(diff) * sign(diff);
            end
        end
    end

    % 平滑项
    for i = 2:rows-1
        for j = 2:cols-1
            idx = sub2ind([rows, cols], i, j);
            lambda_ij = lambda_matrix(i, j);

            smooth_energy_sum = 0;
            smooth_grad_sum   = 0;

            for n = 1:num_neighbors
                ni = i + neighbors(n, 1);
                nj = j + neighbors(n, 2);

                if ni < 1 || ni > rows || nj < 1 || nj > cols
                    continue;
                end

                neighbor_val = D(ni, nj);
                current_val  = D(i, j);
                delta = neighbor_val - current_val;

                local_rho  = rho(delta^2);
                local_grad = rho_prime(delta);
                local_weight = direction_weights(n);
                
                smooth_energy_sum = smooth_energy_sum + lambda_ij * local_weight * local_rho;
                smooth_grad_sum   = smooth_grad_sum   + lambda_ij * local_weight * local_grad;
            end

            E_smooth     = E_smooth + smooth_energy_sum;
            grad(idx)    = grad(idx) + smooth_grad_sum;
        end
    end

    %总能量
    E = E_data + E_smooth;
end

