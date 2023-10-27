function [J, grad] = cost_function(params, X, y, lambda, ...
  input_layer_size, hidden_layer_size, ...
  output_layer_size)

  % Folosim exact formula din enunt temei.

  % Folosim parametrii primiti ca input pentru a reconstrui matricele de
  % ponderi Theta1 si Theta2
  Theta1 = reshape(params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
  Theta2 = reshape(params((1 + (hidden_layer_size * (input_layer_size + 1))):end), output_layer_size, (hidden_layer_size + 1));
  m = size(X, 1);
  J = 0;
  % Initializam matricele de gradienti cu 0
  Theta1_grad = zeros(size(Theta1));
  Theta2_grad = zeros(size(Theta2));
  % Adaugam coloana de 1 la X
  X = [ones(m, 1) X];
  a1 = X;
  % Calculam a2 si a3
  z2 = a1 * Theta1';
  a2 = sigmoid(z2);
  a2 = [ones(m, 1) a2];
  z3 = a2 * Theta2';
  a3 = sigmoid(z3);
  % Salvam a3 intr-o matrice pentru a putea calcula costul
  h = a3;
  % Calculam costul
  y_matrix = eye(output_layer_size)(y,:);
  % Calculam costul regularizat
  J = (1/m) * sum(sum((-y_matrix .* log(h)) - ((1 - y_matrix) .* log(1 - h))));
  reg = (lambda/(2*m)) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));
  % Adaugam regularizarea la cost
  J = J + reg;
  % Calculam gradientii
  delta3 = a3 - y_matrix;
  delta2 = (delta3 * Theta2(:,2:end)) .* sigmoid(z2) .* (1 - sigmoid(z2));
  Delta1 = delta2' * a1;
  Delta2 = delta3' * a2;
  % Calculam gradientii regularizati
  Theta1_grad = (1/m) * Delta1;
  Theta2_grad = (1/m) * Delta2;
  % Adaugam regularizarea la gradienti
  Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + ((lambda/m) * Theta1(:,2:end));
  Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + ((lambda/m) * Theta2(:,2:end));
  % Unim gradientii intr-un vector
  grad = [Theta1_grad(:) ; Theta2_grad(:)];
  
endfunction
