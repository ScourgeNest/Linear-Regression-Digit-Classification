function [classes] = predict_classes(X, weights, ...
                     input_layer_size, hidden_layer_size, ...
                     output_layer_size)

  
  % Transformam vectorul de parametrii intr-o matrice de parametrii
  Theta1 = reshape(weights(1:hidden_layer_size * (input_layer_size + 1)), ...
    hidden_layer_size, (input_layer_size + 1));
    % Transformam vectorul de parametrii intr-o matrice de parametrii
  Theta2 = reshape(weights((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
      output_layer_size, (hidden_layer_size + 1));

  % Punem in m numarul de exemple de antrenare  
  m = size(X, 1);

  % Respectam in exactitate pasii scrisi in enuntul temei.

  % Calculam clasele pentru fiecare exemplu de antrenare
  a1 = [ones(m, 1) X];
  z2 = a1 * Theta1';
  a2 = sigmoid(z2);
  a2 = [ones(m, 1) a2];
  z3 = a2 * Theta2';
  a3 = sigmoid(z3);
  % Determinam clasa cu probabilitatea maxima pentru fiecare exemplu de antrenare
  [max_values, classes] = max(a3, [], 2);

endfunction