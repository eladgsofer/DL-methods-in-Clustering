% Created by:
%   Yehonatan Dahan - 313441131
%   Elad Sofer      - 312124662
% As part of Course "Clustering and Unsupervised learning"
% Ben-Gurion University

%% Excersize 02: Unsupervised Optimal Fuzzy Clustering (UOFC)
close all
clear
clc

[x,y] = Create_dataset('Random'); % Random | Hash 

num_of_clusters = max(unique(y));
fuzziness=2;
max_iters = 100;
tolerance = 1e-6;

[centers, ~, ~] = fuzzy_kmeans(x, num_of_clusters, fuzziness, max_iters, tolerance);
figure(312)
hold on
scatter(centers(1,:),centers(2,:),150,'m','diamond','filled','MarkerEdgeColor','k')

[centers, ~, ~] = fuzzy_kmeans_exponential(x, num_of_clusters, fuzziness, max_iters, tolerance);
figure(312)
hold on
scatter(centers(1,:),centers(2,:),250,'c','pentagram','filled','MarkerEdgeColor','k')


%% fuzzy_kmeans Function
function [centers, U, num_iters] = fuzzy_kmeans(data, num_clusters, fuzziness, max_iters, tolerance)
    [feature_num, datapoints_num] = size(data);
    U = rand(num_clusters, datapoints_num);
    U = U ./ sum(U, 1);                             % Equation(25)
    centers = zeros(feature_num, num_clusters);
    for iter = 1:max_iters
        centers_prev = centers;
        
        % Step 1: Update cluster centers
        U_b = U .^ fuzziness;
        centers = (data * U_b') ./ sum(U_b, 2)';
        
        % Step 2: Update membership values
%         dist_mat = zeros(num_clusters, datapoints_num);
%         for c = 1:num_clusters
%             dist_mat(c, :) = power(vecnorm(data - centers(:, c), 2, 1), 1 / (1 - fuzziness));
%         end
        dist_mat = zeros(num_clusters, datapoints_num);
        for c = 1:num_clusters
            for i = 1:datapoints_num
                d = (data(:,i) - centers(:, c))' * (data(:,i) - centers(:, c));
                dist_mat(c, i) = d .^ (1 / (1 - fuzziness));
            end
        end
        U = dist_mat ./ sum(dist_mat, 1);
        
        % Step 3: Check for convergence
        center_difference = norm(centers - centers_prev, 'fro');
        if center_difference < tolerance
            break;
        end

        figure(312)
        hold on
        if iter == 1
            sctr = scatter(centers(1,:),centers(2,:),150,'m','diamond','filled','MarkerEdgeColor','k');
        else
            set(sctr, 'XData',centers(1,:),'YData',centers(2,:))
            hold on
            pause(0.001)
        end
    end
    
    num_iters = iter;
end

%% fuzzy_kmeans_exponential Function
function [centers, U, num_iters] = fuzzy_kmeans_exponential(data, num_clusters, fuzziness, max_iters, tolerance)
    [feature_num, datapoints_num] = size(data);
    U = rand(num_clusters, datapoints_num);
    U = U ./ sum(U, 1);
    centers = zeros(feature_num, num_clusters);
    
    for iter = 1:max_iters
        centers_prev = centers;
        
        % Step 1: Update cluster centers
        U_b = U .^ fuzziness;
        centers = (data * U_b') ./ sum(U_b, 2)';
        
        % Step 2: Update membership values with exponential distance using fuzzy covariance matrix F
%         dist_mat = zeros(num_clusters, datapoints_num);
%         for c = 1:num_clusters
%             F = compute_fuzzy_covariance_matrix(data, centers, U, c);
%             dist_mat(c, :) = det(F)^0.5 * exp((data - centers(:, c)) * inv(F) * (data - centers(:, c))' / 2);
%         end
        dist_mat = zeros(num_clusters, datapoints_num);
        for c = 1:num_clusters
            Fk = zeros(feature_num);
            for i = 1:datapoints_num
                X_c = data(:,i) - centers(:, c);
                Fk = Fk + U(c, i) .* (X_c * X_c') ;
            end
            Fk = Fk / sum(U(c, :), 2);
            for i = 1:datapoints_num
                X_c = data(:,i) - centers(:, c);
                d = det(Fk)^0.5 * exp(X_c' * inv(Fk) * X_c / 2);
                dist_mat(c, i) = d .^ (1 / (1 - fuzziness));
            end
        end
        U = dist_mat ./ sum(dist_mat, 1);
        
        % Step 3: Check for convergence
        center_difference = norm(centers - centers_prev, 'fro');
        if center_difference < tolerance
            break;
        end
        figure(312)
        hold on
        if iter == 1
            sctr = scatter(centers(1,:),centers(2,:),250,'c','pentagram','filled','MarkerEdgeColor','k');
        else
            set(sctr, 'XData',centers(1,:),'YData',centers(2,:))
            hold on
            pause(0.001)
        end
    end
    
    num_iters = iter;
end