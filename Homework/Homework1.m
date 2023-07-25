% Created by:
%   Yehonatan Dahan - 313441131
%   Elad Sofer      - 312124662
% As part of Course "Clustering and Unsupervised learning"
% Ben-Gurion University

%% Excersize 00 - Create Dataset
% https://www.mathworks.com/help/stats/gmdistribution.html#mw_5518832d-e041-46fd-a8b6-4615e85d498a
close all
clear
clc

Scenerio = 5

NumSamples = 1000;

switch Scenerio
    case 1
        mus = [1 2;-3 -5];
        sigmas = cat(3,[2 .5],[1 1]); % 1-by-2-by-2 array
    case 2
        mus = [2 2;-5 -5];
        sigmas = cat(3,[1 10],[10 1]); % 1-by-2-by-2 array
    case 3
        mu1 = [2 2];
        sigma1 = [10 1 ; 1 1];
        mu2 = [-5 -5];
        sigma2 = [10 7 ; 7 8];
        
        mus     = cat(1, mu1, mu2);          clear mu1 mu2
        sigmas  = cat(3, sigma1, sigma2);    clear sigma1 sigma2
    case 4
        mu1 = [0 2];
        sigma1 = [10 0; 0 .10];
        mu2 = [-2 0];
        sigma2 = [.10 0; 0 10];
        mu3 = [0 -2];
        sigma3 = [10 0; 0 .10];
        mu4 = [2 0];
        sigma4 = [.10 0; 0 10];
        
        mus     = cat(1, mu1, mu2, mu3, mu4);               clear mu1 mu2 mu3 mu4
        sigmas  = cat(3, sigma1, sigma2, sigma3, sigma4);   clear sigma1 sigma2 sigma3 sigma4
    case 5
        mu1 = [3 5];
        sigma1 = [10 0; 0 .10];
        mu2 = [3 -5];
        sigma2 = [.10 0; 0 10];
        mu3 = [-5 3];
        sigma3 = [10 0; 0 .10];
        mu4 = [-3 5];
        sigma4 = [.10 0; 0 10];
        
        mus     = cat(1, mu1, mu2, mu3, mu4);               clear mu1 mu2 mu3 mu4
        sigmas  = cat(3, sigma1, sigma2, sigma3, sigma4);   clear sigma1 sigma2 sigma3 sigma4

    otherwise
        disp("This Scenerio does not exist")
        return
end

c = length(mus);
P = ones(1,c) .* 1/c; % P(w)

gm = gmdistribution(mus,sigmas,P); clear c P mus sigmas
gmPDF = @(x,y) arrayfun(@(x0,y0) pdf(gm,[x0 y0]),x,y);
X = random(gm,NumSamples);


%%%%%%%%%%%%%%%%%%%
%%%%%% Plots %%%%%%
%%%%%%%%%%%%%%%%%%%
fignum = 1;

% 2D Contour Plot
figure(fignum); fignum = fignum+1;
hold on
title("2D Contour Plot")
fcontour(gmPDF,[-10 10])
grid on; axis equal; box on; hold off

% 2D Contour Plot - With Samples
figure(fignum); fignum = fignum+1;
hold on;
title("2D Contour Plot - With Samples")
fcontour(gmPDF,[-10 10])
scatter(X(:,1),X(:,2),10,'.') % Scatter plot with points of size 10
grid on; axis equal; box on; hold off

% 2D Contour Plot - Samples only
figure(fignum); fignum = fignum+1;
title("Samples only")
hold on;
scatter(X(:,1),X(:,2),10,'.') % Scatter plot with points of size 10
grid on; axis equal; box on; hold off

% 3D Surface Plot
figure(fignum); fignum = fignum+1;
% hold on
title("3D Surface Plot")
fsurf(gmPDF,[-10 10])
hold off


clearvars -except gm gmPDF X

%% Excersize 01 - MLE Implementation
clearvars -except gm gmPDF X
max_iter = 20;
log.InputData.GM = gm;
log.InputData.Samples = X;

% Variables
c = gm.NumComponents;
n = length(X);
dim = gm.NumVariables;

% Initial Conditions
P = ones(1,c) .* 1/c; % Start all P(wi) to be equal
switch length(gm.mu)
    case 2
        mu = [1 1; -1,-1]*5;
    case 4
        mu = [1 1; -1,1; 1,-1; -1,-1]*5;
end
% mu = rand(size(gm.mu))*2 - 1;

sigma = repmat(eye(2), [1, 1, length(gm.mu)]);

log.P       = nan([size(P),max_iter]);
log.mu      = nan([size(mu),max_iter]);
log.sigma   = nan([size(sigma),max_iter]);
log         = input_log(log, 1, P, mu, sigma);

for iter = 2:max_iter
    for ii = 1:c
        P_hat_ii = P_hat(X, P(ii), mu(ii,:), sigma(:,:,ii));
        P(ii) = (1/n) * sum(P_hat_ii);
        mu(ii,:) = sum(P_hat_ii .* X) / (n * P(ii));
        temp_sigma = nan(dim,dim,n);
        for k = 1 : n
            Xk_minus_mui = (X(k,:)-mu(ii,:))';
            temp_sigma(:,:,k) = P_hat_ii(k) * (Xk_minus_mui * Xk_minus_mui');
        end
        sigma(:,:,ii) = sum(temp_sigma,3);

    end

    log = input_log(log, iter, P, mu, sigma);

    % 2D Contour Plot
    figure(10)
    hold off
    cur_gmPDF = @(x,y) arrayfun(@(x0,y0) pdf(gmdistribution(mu,sigma,P),[x0 y0]),x,y);
    fcontour(cur_gmPDF,[-10 10])
    hold on
    fcontour(gmPDF,'--',[-10 10])
    title(["2D Contour Plot",strcat("Iter = ",num2str(iter))])
    grid on; axis equal; box on; hold off
    pause(0.0001)
end

%%%%%%%%%%%%%%%%%%%
%%%%%% Plots %%%%%%
%%%%%%%%%%%%%%%%%%%
fignum = 100;

% Mus
figure(fignum); fignum = fignum+1;
hold on
title("\mu s")
sbplot_num = 1;
for ii = 1:c
    for jj = 1:dim
        subplot(c,dim,sbplot_num); sbplot_num = sbplot_num + 1;
        plot(squeeze(log.mu(ii,jj,:)))
        grid on; axis equal; box on;
    end 
end


%% Functions
function log = input_log(log, iter, P, mu, sigma)
    log.P(:,:,iter) = P;
    log.mu(:,:,iter) = mu;
    log.sigma(:,:,:,iter) = sigma;
end

function p_hat = P_hat(x, p, mu, sigma)
    p_hat = p .* mvnpdf(x, mu .* ones(size(x)), sqrtm(sigma));
    p_hat = p_hat ./ sum(p_hat);
end




















