function [x,y] = Create_dataset(type)

% num_of_datapoints = 20 + randi(500); % 21 : 520
num_of_datapoints = 20 + randi(50); % 21 : 70

x = [];
y = [];
switch type
    case 'Random'
%         num_of_gaussians = 5 + randi(5); % 6 : 10
        num_of_gaussians = 1 + randi(5); % 2 : 6
        for ii = 1:num_of_gaussians
            mu_x = (rand(1)*2 - 1) * 350;
            mu_y = (rand(1)*2 - 1) * 350;
            sigma_x = rand(1) * 50;
            sigma_y = rand(1) * 50;
            x_new = [normrnd(mu_x,sigma_x,1,num_of_datapoints);normrnd(mu_y,sigma_y,1,num_of_datapoints)];
            y_new = ii*ones(1,size(x_new,2));
            x = [x,x_new];
            y = [y,y_new];

            figure(312)
            hold on
            scatter(x_new(1,:),x_new(2,:),'filled','MarkerFaceColor',[rand,rand,rand],'MarkerEdgeColor','k')
            hold on
        end
        title(['Num of Clusters: ',num2str(num_of_gaussians)])
        box on
        grid on
    case 'Hash'
        mu1 = [0 25];
        sigma1 = [25 0; 0 1];
        mu2 = [-25 0];
        sigma2 = [1 0; 0 25];
        mu3 = [0 -25];
        sigma3 = [25 0; 0 1];
        mu4 = [25 0];
        sigma4 = [1 0; 0 25];
        mus     = cat(1, mu1, mu2, mu3, mu4);               clear mu1 mu2 mu3 mu4
        sigmas  = cat(3, sigma1, sigma2, sigma3, sigma4);   clear sigma1 sigma2 sigma3 sigma4
        for ii = 1:4
            x_new = [normrnd(mus(ii,1),sigmas(1,1,ii),1,num_of_datapoints);normrnd(mus(ii,2),sigmas(2,2,ii),1,num_of_datapoints)];
            y_new = ii*ones(1,size(x_new,2));
            x = [x,x_new];
            y = [y,y_new];

            figure(312)
            hold on
            scatter(x_new(1,:),x_new(2,:),'filled','MarkerEdgeColor','k')
            hold on
            
        end
        title('Num of Clusters: 4')
        box on
        grid on
    otherwise
        disp("Type is not defined Properly")
end

end