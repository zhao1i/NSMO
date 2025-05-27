function [x_opt, val_opt, iter] = lmm(Fk, JFk, x0, IL, IR, edge_mask)
    % LMM �����Ż�������
    % ���룺
    %   Fk, JFk     - �����������ݶȺ���
    %   x0          - ��ʼ�Ӳ�ͼ
    %   IL, IR      - ����ͼ��
    %   edge_mask   - �����ر�Ե����
    % �����
    %   x_opt       - �Ż�����Ӳ�ͼ
    %   val_opt     - ��������ֵ
    %   iter        - ʵ�ʵ�������

    %% ��������
    max_iter       = 10000;
    epsilon        = 1e-5;
    alpha_init     = 0.1;
    alpha_decay    = 0.8;  
    rho            = 0.5;
    sigma          = 0.4;

    % ˫���˲�����
    sigma_space     = 0.7;
    sigma_intensity = 2.2;

    % �����֪ƽ��Ȩ��
    lambda_min = 0.4;
    lambda_max = 0.6;
    lambda_matrix = lambda_min + (lambda_max - lambda_min) .* (1 - edge_mask);

    % ��ʼ������
    x     = x0;
    alpha = alpha_init;
    iter  = 0;

    %% ����������
    while iter < max_iter
        val  = Fk(x, IL, IR, lambda_matrix);
        grad = JFk(x, IL, IR, lambda_matrix);

        % �����ж�
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
                % Ӧ��˫���˲�
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

    % �����Ӳ�ͼ
    x_opt   = x;
    val_opt = val;
end

