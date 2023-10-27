function [matrix] = initialize_weights(L_prev, L_next)

  % Calculam valoarea lui e
  e = sqrt(6) / sqrt(L_prev + L_next);
  % Generam matricea Folosind formula din enuntul temei
  matrix = rand(L_next, L_prev + 1) * 2 * e - e;
  
endfunction
