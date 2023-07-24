% Created by:
%   Yehonatan Dahan - 313441131
%   Elad Sofer      - 312124662
% As part of Course "Clustering and Unsupervised learning"
% Ben-Gurion University

%% Excersize 03: Agglomerative hierarchical clustring
close all
clear
clc

[x,y] = Create_dataset('Random'); % Random | Hash |

num_of_clusters = max(unique(y));

pause(0.001)
centers = agglomerative_hierarchical_clustering(x, num_of_clusters, 'Dmean');

figure(312)
hold on
scatter(centers(1,:),centers(2,:),40,'red','diamond','filled','MarkerEdgeColor','k')

%% Agglomerative_hierarchical_clustering Function

function cluster_centers = agglomerative_hierarchical_clustering(data, num_clusters, DistanceType)
    % Initialize each data point as its own cluster
    num_datapoints = size(data, 2);
    for ii = 1:num_datapoints
        clusters{ii} = {[data(:,ii)]};
    end

    % Main loop for merging clusters
    for k = 1:(num_datapoints - num_clusters)
        if(mod(k,10)==0)
            disp(strcat(num2str(k)," / ",num2str(num_datapoints - num_clusters)))
        end
        min_dist = Inf;
        for c1 = 1:length(clusters)
            c1_data = cell2mat(clusters{c1});
            for c2 = (c1+1):length(clusters)
                c2_data = cell2mat(clusters{c2});
                switch DistanceType
                    case "Dmin"
                        c1c2_dist = min_dist_function(c1_data,c2_data);
                    case "Dmax"
                        c1c2_dist = max_dist_function(c1_data,c2_data);
                    case 'Davg'
                        c1c2_dist = avg_dist_function(c1_data,c2_data);
                    case 'Dmean'
                        c1c2_dist = mean_dist_function(c1_data,c2_data);
                    otherwise
                        disp("Wrong DistanceType given")
                end

                if c1c2_dist < min_dist
                    min_dist = c1c2_dist;
                    pair = [c1,c2];
                end
            end
        end

        % Merge the clusters
        clusters{pair(1)} = [clusters{pair(1)}, clusters{pair(2)}];
        clusters(pair(2)) = []; % Empty the cluster that has been merged

    end

    % Calculate cluster centers
    cluster_centers = zeros(size(data, 1),num_clusters);
    for i = 1:num_clusters
        cluster_data = cell2mat(clusters{i});
        cluster_centers(:, i) = mean(cluster_data, 2);
    end
end

function final_dist = min_dist_function(c1_data,c2_data)
final_dist = Inf;
for ii = 1:size(c1_data,2)
    for jj = 1:1:size(c2_data,2)
        final_dist = min(final_dist,norm(c1_data(:,ii)-c2_data(:,jj)));
    end
end
end

function final_dist = max_dist_function(c1_data,c2_data)
final_dist = 0;
for ii = 1:size(c1_data,2)
    for jj = 1:1:size(c2_data,2)
        final_dist = max(final_dist,norm(c1_data(:,ii)-c2_data(:,jj)));
    end
end
end

function avg_distance = avg_dist_function(c1_data,c2_data)
avg_distance = 0;
for ii = 1:size(c1_data,2)
    for jj = 1:1:size(c2_data,2)
        avg_distance = avg_distance + norm(c1_data(:,ii)-c2_data(:,jj));
    end
end
avg_distance = avg_distance / (ii+jj);
end

function mean_distance = mean_dist_function(c1_data,c2_data)
mean_distance = norm(mean(c1_data,2)-mean(c2_data,2));
end
