%/*******************************************************
% * Copyright (C) 2019-2020 Ruixu Liu <liur05@udayton.edu>
% * 
% * This file is part of Artificial Neural Network.
% * 
% * MIT License
% *******************************************************/

%==========================================================================
% HW4
%==========================================================================

%% Loading

% Load the train and test images
training_images = loadMNISTImages('train-images.idx3-ubyte');
% train_images(:,i) is a double matrix of size 784xi(where i = 1 to 60000)
% Intensity rescale to [0,1]

training_labels = loadMNISTLabels('train-labels.idx1-ubyte');
% train_labels(i) - 60000x1 vector

testing_images = loadMNISTImages('t10k-images.idx3-ubyte');
% testing_images(:,i) is a double matrix of size 784xi(where i = 1 to 10000)
% Intensity rescale to [0,1]

testing_labels = loadMNISTLabels('t10k-labels.idx1-ubyte');
% test_labels(i) - 10000x1 vector

% Prepare experinment data
number_of_training_images = 200;
number_of_testing_images = 100;
[balanced_train_image, balanced_train_labels] = balance_MNIST_selection(...
    training_images,training_labels,number_of_training_images);
[balanced_test_image, balanced_test_labels] = balance_MNIST_selection(...
    testing_images,testing_labels,number_of_testing_images);

train_images = zeros(784, number_of_training_images);
train_labels = zeros(number_of_training_images,1);
testing_images = zeros(784, number_of_testing_images);
testing_labels = zeros(number_of_testing_images,1);

for i = 1: number_of_training_images
    train_images(:,i) = balanced_train_image(:,i);
    train_labels(i) = balanced_train_labels(i);
end

for i = 1: number_of_testing_images
    testing_images(:,i) = balanced_test_image(:,i);
    testing_labels(i) = balanced_test_labels(i);
end

%%   Training
% 200 train images
x = train_images(:,1:number_of_training_images);
% x(784,200)

% No complement input
X = [x];
% X(784,200)

% Set choice parameter
aleph = 0.001;
% Set vigilance parameter
p_value = [0.7 0.8 0.9 0.95];
% Set learning rate parameter
beta_value = [0.6 0.8];


for b = 1 : length(beta_value)
    figure;
    
    for p = 1:length(p_value)
        
        % Initial weights
        w = zeros(784,1);
        % Initial labels
        class_label = zeros(1,1);
        output_label = zeros(1,10);
        
        New_output_label = 0;
        
        flag = 0;
        iteration = 0;
        while (flag == 0)
            
            for i = 1 : number_of_training_images
                
                [row, col_w] = size(w);
                
                X_choice = repmat(X(:,i),[1,col_w]);
                
                
                % compute the choice function
                numerator_CF_train = sum(min(w,X_choice));
                
                denominator_CF_train = aleph + sum(w);
                
                CF = numerator_CF_train ./ denominator_CF_train;
                
                [CF_J, J] = max(CF);
                
                % compute the match function or the category choice
                numerator_MF_train = sum(min(w(:,J),X(:,i)));
                
                denominator_MF_train = sum(X(:,i));
                
                MF =  numerator_MF_train / denominator_MF_train;
                
                j = 0; % No resets initially
                
                while( MF < p_value(p) )

                    % Create a new node if all previous nodes fail
                    % (All MFs are zeros)

                    if j == col_w
                        w = horzcat(w,X(:,i));
                        
                        [row, col] = size(w);
                        
                        numerator_CF_train = sum(min(w(:,col),X(:,i)));
                        
                        denominator_CF_train = aleph + sum(w(:,col));
                        
                        CF(col) = numerator_CF_train / denominator_CF_train;
                    end

                    % Reset the winner node (Some steps are additional but don't affect the output)
                    CF(J) = 0;
                    
                    [CF_J, J] = max(CF);
                    
                    numerator_MF_train = sum(min(w(:,J),X(:,i)));
                    
                    denominator_MF_train = sum(X(:,i));
                    
                    MF =  numerator_MF_train / denominator_MF_train;
                    
                    j = j+1; % Number of Resets
                    
                end
                
                % Update weights
                w(:,J) = beta_value(b) * min( w(:,J) , X(:,i) ) + (1-beta_value(b)) * w(:,J);
                
                % Update labels
                [cl_row,cl_col] = size(class_label);
                if j == col
                    class_label(1,cl_col+1) = train_labels(i)+1;
                else
                    class_label(cl_row+1,J) = train_labels(i)+1;
                end
            end
            
            % Compute output label
            output_label = zeros(10,size(class_label,2));
            for k = 2 : cl_col
                group = class_label(:,k);
                label = unique(group);
                output_label(1:size(label,1),k) = label;
                
                % One node can only have one output class
%                 group(group==0) = [];
%                 tbl= tabulate(group);
%                 [~,idx] = max(tbl(:,3));
%                 output_label(k) = idx;
                
            end
            
            % Repeat some times until no change
            if size(New_output_label) == size(output_label) % No change in the number of categories
                if New_output_label == output_label % No change in the patterns associated with their categories
                    flag = 1;
                end
            end
            
            New_output_label = output_label;
            
            iteration = iteration+1;
        end
        
%% Testing

% Error in the training data
        % Matrix to record the recognition values
        True = zeros(1,10);
        False = zeros(1,10);
        number = zeros(number_of_training_images,1);
        for i = 1:number_of_training_images
            %     i = 1;
            [row, col] = size(w);
            
            % Compute choice function
            X_choice = repmat(X(:,i),[1,col]);
            CF_train_as_test = sum(min(w,X_choice))./(aleph + sum(w));
            
            % Find the max to get the winner node J
            [~, J] = max(CF_train_as_test );
            
%             number(i) = output_label(J);
            
            % Compute the numbers of right and wrong values
            if find(output_label(:,J)==(train_labels(i)+1))
                True(train_labels(i)+1)=True(train_labels(i)+1)+1;
            else
                False(train_labels(i)+1)=False(train_labels(i)+1)+1;
            end
            
            
        end
        
        % Compute the percentage error with training data
        error_train = 1- True./(True + False);
        % Calculate the average percentage error
        e_train(p,b) = (sum(error_train))/10;
        
        % Plot
        subplot(length(p_value),2,2*p-1);
        bar(0:9,error_train);
        axis([-1 11 0 1])
        str = sprintf('Learning rate %g, Vigilance %g, Train avg error %.2g'...
            ,beta_value(b),p_value(p),e_train(p,b));
        title(str);
        xlabel('Patteren 0 to 9');
        ylabel('Error Percentage');
        
        
% Error in the testing data
        True = zeros(1,10);
        False = zeros(1,10);
        number2 = zeros(200,1);
        
        x_test = testing_images(:,1:number_of_testing_images);
        % x(784,100)
        
        % No complement input
        X_test = [x_test];
        % X(784,100)
        
        for i = 1:number_of_testing_images
            [row, col] = size(w);
            
            % Compute choice function
            X_choice = repmat(X_test(:,i),[1,col]);
            CF_test = sum(min(w,X_choice))./(aleph + sum(w));
            
            % Find the max to get the winner node J
            [~, J] = max(CF_test );
            
%             number = output_label(:,J);
            
            % Compute the numbers of right and wrong values
            if find(output_label(:,J)==(testing_labels(i)+1))
                True(testing_labels(i)+1)=True(testing_labels(i)+1)+1;
            else
                False(testing_labels(i)+1)=False(testing_labels(i)+1)+1;
            end
            
            
        end
        
        % Compute the testing data error error
        error_test = 1- True./(True + False);
        % Calculate the average percentage error
        e_test(p,b) = (sum(error_test))/10;
        
        % Plot
        subplot(length(p_value),2,2*p);
        bar(0:9,error_test);
        axis([-1 11 0 1])
        str = sprintf('Learning rate %g, Vigilance %g, Testing avg error %.2g'...
            ,beta_value(b),p_value(p),e_test(p,b));
        title(str);
        xlabel('Patteren 0 to 9');
        ylabel('Error Percentage');
        

    end
end
%==========================================================================