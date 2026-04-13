
clear 

filename = 'data_all';

data = load([filename, '.mat']);

traning_table = array2table([data.trainlab,data.trainv]);
traning_table.Properties.VariableNames{1} = 'label';

testing_table = array2table([data.testlab,data.testv]);
testing_table.Properties.VariableNames{1} = 'label';

vec_size_e= data.vec_size;
num_train_e= data.num_train;
num_test_e = data.num_test;

other_variables = struct('vec_size',vec_size_e,'num_train', num_train_e,'num_test', num_test_e);


variable_table = struct2table(other_variables);


writetable(traning_table, 'NMIST_training_set.csv')

writetable(testing_table, 'NMIST_test_set.csv')

writetable(variable_table, 'NMIST_meta_data.csv')
% 
disp('Success')

