using Distributions

P = [0.5 0.5; 0.5 0.5]
y1 = 5
y2 = 6

T = 1000000
y_vec = zeros(T)
y_vec[1] = y2 

for t in 2:T
    rn = rand()
    if y_vec[t-1] == y1
        y_vec[t] = (rn <= 0.5) ? y1 : y2
    else  # y_vec[t-1] == y2
        y_vec[t] = (rn <= 0.5) ? y2 : y1
    end
end

println("mean(y_vec) = ", mean(y_vec))
println("|mean(y_vec) - (π1 y1 + π2 y2)| = ", abs(mean(y_vec) - 5.5))