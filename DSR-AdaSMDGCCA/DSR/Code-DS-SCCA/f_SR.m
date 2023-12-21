function W = f_SR(X, options)


    c = unique(options.label);
        c_n = length(c);
        for i = 1:c_n 
            index = find(options.label == c(i));
            index_n = length(index);
            for j = 1:index_n
                y = [X(:, index(j)); 1];
                A = [X(:, setdiff(index, index(j))); ones(1, index_n - 1)];
                wi = nnLeastR(A, y, options.lambda, options.opts);
                W(setdiff(index, index(j)), index(j)) = wi;
                fprintf('Sample %d is done \n', index(j));
            end
        end
end