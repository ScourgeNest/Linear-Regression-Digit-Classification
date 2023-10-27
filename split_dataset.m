function [x_train, y_train, X_test, y_test] = split_dataset(X, y, percent)
  
  % Punem in n_train numarul de exemple de antrenament
  n_train = round(percent * size(X, 1));
  
  % amestecam indicii
  index = randperm(size(X, 1));
  
  % primele n_train indici vor fi pentru setul de antrenament
  train_index = index(1:n_train);
  
  % restul de indici vor fi pentru setul de test
  test_index = index(n_train+1:end);
  
  % setul de antrenament
  x_train = X(train_index, :);
  y_train = y(train_index);
  
  % setul de test
  X_test = X(test_index, :);
  y_test = y(test_index);
  
endfunction