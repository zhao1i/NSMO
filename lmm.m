function [x_opt, val_opt, iter] = lmm(Fk, JFk, x0, IL, IR, edge_mask)
    % LMM 迭代优化主函数
    % 输入：
    %   Fk, JFk     - 能量函数与梯度函数
    %   x0          - 初始视差图
    %   IL, IR      - 左右图像
    %   edge_mask   - 超像素边缘掩码
    % 输出：
    %   x_opt       - 优化后的视差图
    %   val_opt     - 最终能量值
    %   iter        - 实际迭代次数

    %% 参数设置
    max_iter       = 10000;
    epsilon        = 1e-5;
    alpha_init     = 0.1;
    alpha_decay    = 0.8;  
    rho            = 0.5;
    sigma          = 0.4;

    % 双边滤波参数
    sigma_space     = 0.7;
    sigma_intensity = 2.2;

    % 区域感知平滑权重
    lambda_min = 0.4;
    lambda_max = 0.6;
    lambda_matrix = lambda_min + (lambda_max - lambda_min) .* (1 - edge_mask);

    % 初始化变量
    x     = x0;
    alpha = alpha_init;
    iter  = 0;

    %% 主迭代过程
    while iter < max_iter
        val  = Fk(x, IL, IR, lambda_matrix);
        grad = JFk(x, IL, IR, lambda_matrix);

        % 收敛判断
        grad_norm = norm(grad);
        if grad_norm < epsilon
            fprintf('[Iter %d] Gradient below threshold. Converged.\n', iter);
            break;
        end

        m = 0;
        success = false;

        while m < 5
            delta_x = -alpha * reshape(grad, size(x));
            x_new   = x + delta_x;

            val_new = Fk(x_new, IL, IR, lambda_matrix);

            if val_new < val + sigma * rho^m
                % 应用双边滤波
                x_filtered = bilateral_filter(x_new, sigma_space, sigma_intensity);
                x = x_filtered;
                success = true;
                break;
            else
                alpha = alpha * alpha_decay; 
                m = m + 1;
            end
        end

        if ~success
            fprintf('[Iter %d] Line search failed to improve energy.\n', iter);
            break;
        end

        
        iter = iter + 1;
        fprintf('[Iter %d] Energy: %.6f | Grad norm: %.6f\n', iter, val_new, grad_norm);
    end

    % 更新视差图
    x_opt   = x;
    val_opt = val;
end

