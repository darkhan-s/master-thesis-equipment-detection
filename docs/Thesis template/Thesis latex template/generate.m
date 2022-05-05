
% Aaltologo char generator

function y = generate(fileID, x, dx = 0)
d = size(x);
xmin = 1000;
for i = 1:d(1) 
 if (x(i,2) < xmin) xmin = x(i,2); end
end

ddx = 0;
% If negative, absolute x position; if positive, offset
if (dx < 0) ddx = xmin - abs(dx); else ddx = -dx; end

for i = 1:d(1) 
  if     (x(i,1) == 0) fprintf(fileID, '(%3.4f,%3.4f)..', x(i,2) - ddx, x(i,3));
  elseif (x(i,1) == 1) fprintf(fileID, '(%3.4f,%3.4f)--cycle\n%%---\n', x(i,2) - ddx, x(i,3));
  elseif (x(i,1) == 2) fprintf(fileID, '(%3.4f,%3.4f)--', x(i,2) - ddx, x(i,3));
  elseif (x(i,1) == 3) fprintf(fileID, 'controls(%3.4f,%3.4f)', x(i,2) - ddx, x(i,3));
  elseif (x(i,1) == 4) fprintf(fileID, 'and(%3.4f,%3.4f)..\n', x(i,2) - ddx, x(i,3));
  elseif (x(i,1) == 5) fprintf(fileID, '(%3.4f,%3.4f)\n', x(i,2) - ddx, x(i,3));
  else disp('ERROR')
end

end
