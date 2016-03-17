using PyPlot

function cartesian_product(product_set)
        N = length(product_set)
        M = 1
        for i = 1 : N
                M *= length(product_set[i])
        end

        product_returned = [product_set[1]...]

        for class_index = 2 : N
                class_number = length(product_set[class_index])
                current_length = length(product_returned[:,1])

                enlarged_matrix = Array(Int64,current_length*class_number,class_index)

                if class_number == 1
                        enlarged_matrix[:,1:class_index-1] = product_returned
                        for i = 1 : current_length
                                enlarged_matrix[i,class_index] = product_set[class_index][1]
                        end
                else

                        for enlarge_times = 1 : class_number
                                enlarged_matrix[(enlarge_times-1)*current_length+1:enlarge_times*current_length,1:class_index-1] =
                                product_returned
                        end

                        for i = 1 : class_number * current_length
                                item_index = div(i-1,current_length) + 1
                                enlarged_matrix[i,class_index] = product_set[class_index][item_index]
                        end
                end
                product_returned = enlarged_matrix

        end

        rearrange_matrix = Array(Int64,size(product_returned))
        for i = 1 : N
            rearrange_matrix[:,i] = product_returned[:,N+1-i]
        end

        product_returned = 0

        return rearrange_matrix
end


function belief_update(b,a,o,T,O)
    n_s = length(b)
    b_p = Array(Float64,n_s)
    for s_p = 1 : n_s
        sum_s = 0
        for i = 1 : n_s
            #print((T[i,a,s_p],b[i]))
            sum_s += T[i,a,s_p] * b[i]
        end
        b_p[s_p] = O[s_p,a,o] * sum_s
        #println(b_p[s_p])
    end


    if sum(b_p) == 0.0
        b_p = (1/n_s) * ones(Float64,n_s)
        return b_p
    else
        b_p = b_p / sum(b_p)
        return b_p
    end

end



function Q_value_iteration(Q_0,T,R,delta,gamma)
    sigma = Inf
    (n_s,n_a) = size(T)[1:2]

    Q = zeros(Float64,n_s,n_a)

    while sigma > delta
        for s = 1 : n_s
            for a = 1 : n_a
                sum_s_p = 0
                for s_p = 1 : n_s
                    V = maximum(Q_0[s_p,:])
                    sum_s_p += T[s,a,s_p] * ( R[s,a,s_p] + gamma * V)
                end
            Q[s,a] = sum_s_p
            end
        end
        sigma = norm(Q-Q_0,Inf)
        Q_0 = copy(Q)
    end
    return Q
end

function Q_value_iteration_count(Q_0,T,R,count,gamma)
    sigma = Inf
    (n_s,n_a) = size(T)[1:2]

    Q = zeros(Float64,n_s,n_a)
    t = 1

    while t < count + 1
        t += 1
        Q = zeros(Float64,n_s,n_a)
        for s = 1 : n_s
            for a = 1 : n_a
                sum_s_p = 0

                if (s==7 && a == 3)
                    sum_check = 0
                    for sp = 1 : n_s
                        sum_check += T[s,a,sp] * R[s,a,sp]
                        #println(T[s,a,sp] * R[s,a,sp])
                    end
                    println(sum_check)
                end

                for s_p = 1 : n_s
                    V = maximum(Q_0[s_p,:])
                    #if ((t,s,a)==(1,7,1)); println(V); end;
                    sum_s_p += T[s,a,s_p] * ( R[s,a,s_p] + gamma * V)
                end
            Q[s,a] = sum_s_p
            end
        end
        Q_0 = copy(Q)
    end
    return Q
end


function Q_open(T,R,d,gamma)
    (n_s,n_a) = size(T)[1:2]

    action_one_set = tuple(collect(1:n_a)...)

    # Compute probability tables
    p_table_time = Dict()
    p_table_time[1] = T

    for i = 2 : d # i is the number of actions acted between two states

        # T_i stores the information P(s,a_1,a_2,...,a_i,s')
        T_i = zeros(Float64,n_s,n_a^i,n_s)

        # action list stores all possible action string with depth i
        action_set = Array(Any,i)
        for j = 1 : i
            action_set[j] = action_one_set
        end
        action_list = cartesian_product(action_set)

        # Do iteration on p from previous depth
        for a = 1 : length(action_list[:,1])
            a_previous = div(a-1,n_a)+1
            a_now = (a-1)%n_a + 1

            for s_initial = 1 : n_s
                for s_end = 1 : n_s
                    sum_p = 0
                    for s_mid = 1 : n_s
                        sum_p += p_table_time[i-1][s_initial,a_previous,s_mid] *
                                 T[s_mid,a_now,s_end]
                    end
                    T_i[s_initial,a,s_end] = sum_p
                end
            end
        end

        p_table_time[i] = T_i
    end


    # Compute expected rewards after taking action string
    r_table_time = Dict()

    # depth = 1
    R_1 = zeros(Float64,n_s,n_a)
    for s_initial = 1 : n_s
        for a = 1 : n_a
            r_sum = 0
            for s_end = 1 : n_s
                r_sum += T[s_initial,a,s_end] * R[s_initial,a,s_end]
            end

            R_1[s_initial,a] = r_sum
        end
    end
    r_table_time[1] = R_1

    for i = 2 : d

        #R_i stores the information r(s,a_1,a_2,...,a_i,s')
        R_i = zeros(Float64,n_s,n_a^i)

        # action list stores all possible action string with depth i
        action_set = Array(Any,i)
        for j = 1 : i
            action_set[j] = action_one_set
        end
        action_list = cartesian_product(action_set)

        # Do iteration on r by previous p of depth i-1
        for a = 1 : length(action_list[:,1])
            a_previous = div(a-1,n_a)+1
            a_now = (a-1)%n_a + 1

            for s_initial = 1 : n_s

                r_sum_1 = 0

                for s_end = 1 : n_s

                    r_sum_2 = 0

                    for s_addition = 1 : n_s

                        r_sum_2 += T[s_end,a_now,s_addition] *
                                   R[s_end,a_now,s_addition]
                    end

                    r_sum_1 += p_table_time[i-1][s_initial,a_previous,s_end]*
                               r_sum_2
                end

                R_i[s_initial,a] = r_sum_1

            end
        end

        r_table_time[i] = R_i

    end

    #return (p_table_time,r_table_time)

    g_table = zeros(Float64,n_s,n_a)

    action_set = Array(Any,d)
    for j = 1 : d
        action_set[j] = action_one_set
    end

    action_set_big = cartesian_product(action_set)

    for s= 1 : n_s

        for first_action = 1 : n_a

            best_string = 0
            best_string_value = - Inf

            # Compute the best action sequence after the first action

            for following_action = 1 : n_a^(d-1)

                big_index = (first_action - 1) * (n_a ^ (d-1)) + following_action

                current_action = [first_action action_set_big[following_action,2:end]]

                cumulative_reward = 0

                for j = 1 : d

                    sub_action = current_action[1:j]
                    action_index = 0

                    for k = 1 : j
                        if k == 1
                            action_index += sub_action[end]
                        else
                            action_index += (n_a^(k-1)) * (sub_action[end+1-k] - 1)
                        end
                    end

                    cumulative_reward += r_table_time[j][s,action_index]

                end

                if cumulative_reward > best_string_value

                    best_string_value = cumulative_reward
                    best_string = following_action

                end

            end

            println(s,first_action,best_string)

            g_table[s,first_action] = (best_string_value / d ) * (1/(1-gamma))

        end
    end

    return (p_table_time,r_table_time,g_table)

end


function FIB(alpha_0,T,R,O,delta,gamma)
    sigma = Inf
    (n_s,n_a) = size(T)[1:2]
    n_o = size(O)[3]

    alpha = zeros(Float64,n_s,n_a)

    while sigma > delta
        alpha = zeros(Float64,n_s,n_a)
        for s = 1 : n_s
            for a = 1 : n_a

                # Imediate reward
                for sp = 1 : n_s
                    alpha[s,a] += R[s,a,sp] * T[s,a,sp]
                end


                # Next step
                for o = 1 : n_o

                    best_ap = 0
                    best_ap_value = -Inf

                    for ap = 1 : n_a

                        ap_value = 0

                        for sp = 1 : n_s

                            ap_value += O[sp,a,o] * T[s,a,sp] * alpha_0[sp,ap]

                        end

                        if ap_value > best_ap_value

                            best_ap_value = ap_value
                            best_ap = ap

                        end
                    end

                    alpha[s,a] += gamma * best_ap_value

                end

            end
        end

        sigma = norm(alpha-alpha_0,Inf)
        alpha_0 = copy(alpha)

    end

    return alpha
end

function QUMDP(Q_0,T,R,delta,gamma)

    sigma = Inf
    (n_s,n_a) = size(T)[1:2]

    Q = zeros(Float64,n_s,n_a)

    while sigma > delta
        for s = 1 : n_s
            for a = 1 : n_a

                value_ap_index = 0
                value_ap = -Inf


                for ap = 1 : n_a
                    value_sp = 0
                    for sp = 1 : n_s
                        value_sp += T[s,a,sp] * Q_0[sp,ap]
                    end

                    if value_sp > value_ap
                        value_ap_index = ap
                        value_ap = value_sp
                    end

                end

                im_reward = 0
                for sp = 1 : n_s
                    im_reward += T[s,a,sp] * R[s,a,sp]
                end

                Q[s,a] = im_reward + gamma * value_ap

            end
        end

        sigma = norm(Q-Q_0,Inf)
        Q_0 = copy(Q)
    end

    return Q
end

function weight(b)
    n_s = length(b)

    entrop = 0

    for i = 1 : n_s

        if b[i] != 0
            entrop -= b[i] * log(b[i])
        end

    end

    w_1 = (n_s - exp(entrop))/(n_s - 1)

    w_2 = (log(n_s) - entrop)/(log(n_s))

    return (w_1,w_2)

end

function entropy_function(b)

    n_s = length(b)

    entrop = 0

    for i = 1 : n_s

        if b[i] != 0
            entrop -= b[i] * log(b[i])
        end

    end

    return entrop
end

function relative_entropy(a,b)

    n_s = length(b)

    en = 0

    for i = 1 : n_s
        if (a[i] != 0)&&(b[i] != 0)
            en += a[i] * ( log(a[i]) - log(b[i]) )
        end
    end

    return en

end

function scaling_entropy(a,b)

    n_a = length(a)
    n_b = length(b)

    entrop1 = entropy_function(a)
    entrop2 = entropy_function(b)

    entrop = (entrop1 + entrop2)

    w = (log(n_a * n_b) - entrop)/(log(n_a * n_b))
    return w
end

function direct_update(b,T,a)

    (n_s,n_a) = size(T)[1:2]

    b_update = Array(Float64,n_s)

    for i = 1 : n_s

        b_update[i] = 0

        for j = 1 : n_s

            b_update[i] += T[j,a,i] * b[j]

        end

    end

    return b_update

end


function value_approx_QMDP(Q_0_MDP,T,R,delta,gamma,all_b)

    Q_MDP = Q_value_iteration(Q_0_MDP,T,R,delta,gamma)

    (n_s,n_a) = size(T)[1:2]

    value_action = Array(Any,length(all_b[:,1]),2)

    for nb = 1 : length(all_b[:,1])
        b = all_b[nb,:]


        best_action_index = 0
        best_action_value = -Inf

        for a = 1 : n_a

            Q_sa = 0

            for s = 1 : n_s

                Q_sa += b[s] * (Q_MDP[s,a] )

            end

            if best_action_value < Q_sa

                best_action_index = a
                best_action_value = Q_sa

            end

        end

        value_action[nb,:] = [best_action_value,best_action_index]

    end

    return value_action

end

function weight_propogation_f(T,O,delta,gamma)

    n_s = size(T)[1]
    n_a = size(T)[2]
    n_o = size(O)[3]

    f = zeros(Float64,n_s,n_a)

    for s = 1 : n_s
        for a = 1 : n_a
            f[s,a] = weight(O[s,a,:])[1]
        end
    end

    return f
end

function weight_propogation(T,O,delta,gamma)

    n_s = size(T)[1]
    n_a = size(T)[2]
    n_o = size(O)[3]

    f = zeros(Float64,n_s,n_a)

    for s = 1 : n_s
        for a = 1 : n_a
            f[s,a] = weight(O[s,a,:])[2]
        end
    end

    W_0 = zeros(Float64,n_s,n_a)
    W = zeros(Float64,n_s,n_a)

    sigma = Inf

    while sigma > delta
        for s = 1 : n_s
            for a = 1 : n_a
                sum_s_p = f[s,a]
                for s_p = 1 : n_s
                    V = maximum(W_0[s_p,:])
                    sum_s_p += gamma * T[s,a,s_p] * V
                end
                W[s,a] = (sum_s_p)/(1+gamma)
            end
        end
        sigma = norm(W-W_0,Inf)
        W_0 = copy(W)
    end
    return W
end

function weight_propogation_2(T,O,delta,gamma)

    n_s = size(T)[1]
    n_a = size(T)[2]
    n_o = size(O)[3]

    f = zeros(Float64,n_s,n_a)

    for s = 1 : n_s
        for a = 1 : n_a
            f[s,a] = weight(O[s,a,:])[2]
        end
    end

    W_0 = zeros(Float64,n_s,n_a)
    W = zeros(Float64,n_s,n_a)

    sigma = Inf

    while sigma > delta
        for s = 1 : n_s
            for a = 1 : n_a
                sum_s_p = f[s,a]
                for s_p = 1 : n_s
                    V = maximum(W_0[s_p,:])
                    sum_s_p += gamma * T[s,a,s_p] * V
                end
                W[s,a] = (sum_s_p)
            end
        end
        sigma = norm(W-W_0,Inf)
        W_0 = copy(W)
    end

    W = W * (1-gamma)
    return W
end

function weight_propogation_3(T,O,delta,gamma)

    n_s = size(T)[1]
    n_a = size(T)[2]
    n_o = size(O)[3]

    f = zeros(Float64,n_s,n_a)

    for s = 1 : n_s
        for a = 1 : n_a
            f[s,a] = weight(O[s,a,:])[2]
        end
    end

    W_0 = zeros(Float64,n_s,n_a)
    W = zeros(Float64,n_s,n_a)

    sigma = Inf

    while sigma > delta
        for s = 1 : n_s
            for a = 1 : n_a
                im_w = f[s,a]

                # Choose a action
                best_ap_v = - Inf

                for ap = 1 : n_a

                    ap_v = 0

                    for sp = 1 : n_s

                        ap_v += T[s,a,sp] * W_0[sp,ap]

                    end

                    if ap_v > best_ap_v

                        best_ap_v = ap_v

                    end

                end

                W[s,a] = im_w + gamma * best_ap_v
            end
        end

        sigma = norm(W-W_0,Inf)
        W_0 = copy(W)
    end

    W = W * (1-gamma)
    return W
end





function value_approx_v1(Q_0_MDP,Q_0_UMDP,T,R,delta,gamma,all_b)

    ##### Only distribution of belief state is considered for weighting #####

    Q_UMDP = QUMDP(Q_0_UMDP,T,R,delta,gamma)
    Q_MDP = Q_value_iteration(Q_0_MDP,T,R,delta,gamma)

    (n_s,n_a) = size(T)[1:2]

    value_action = Array(Any,length(all_b[:,1]),2)

    for nb = 1 : length(all_b[:,1])
        b = all_b[nb,:]
        weight_b = weight(b)[2]

        best_action_index = 0
        best_action_value = -Inf

        for a = 1 : n_a

            Q_sa = 0

            for s = 1 : n_s

                Q_sa += b[s] * (Q_MDP[s,a] * weight_b + Q_UMDP[s,a] * (1 - weight_b))

            end

            if best_action_value < Q_sa

                best_action_index = a
                best_action_value = Q_sa

            end

        end

        value_action[nb,:] = [best_action_value,best_action_index]

    end

    return value_action

end

function value_approx_v2(Q_0_MDP,Q_0_UMDP,T,R,O,delta,gamma,all_b)

    ##### Only distribution of O[s,a,:] is considered for weighting #####

    Q_UMDP = QUMDP(Q_0_UMDP,T,R,delta,gamma)
    Q_MDP = Q_value_iteration(Q_0_MDP,T,R,delta,gamma)

    (n_s,n_a) = size(T)[1:2]
    n_o = size(T)[3]

    value_action = Array(Any,length(all_b[:,1]),2)

    for nb = 1 : length(all_b[:,1])
        b = all_b[nb,:]

        best_action_index = 0
        best_action_value = -Inf

        for a = 1 : n_a

            Q_sa = 0

            for s = 1 : n_s

                O_distr = O[s,a,:]
                weight_b = weight(O_distr)[1]

                Q_sa += b[s] * (Q_MDP[s,a] * weight_b + Q_UMDP[s,a] * (1 - weight_b))

            end

            if best_action_value < Q_sa

                best_action_index = a
                best_action_value = Q_sa

            end

        end

        value_action[nb,:] = [best_action_value,best_action_index]

    end

    return value_action

end


function value_approx_v3(Q_0_MDP,Q_0_UMDP,T,R,O,delta,gamma,all_b)

    ##### both distribution of O[s,a,:] and b[s] are considered for weighting #####
    ##### They are considered as average of entropies #####

    Q_UMDP = QUMDP(Q_0_UMDP,T,R,delta,gamma)
    Q_MDP = Q_value_iteration(Q_0_MDP,T,R,delta,gamma)

    (n_s,n_a) = size(T)[1:2]
    n_o = size(T)[3]

    value_action = Array(Any,length(all_b[:,1]),2)

    for nb = 1 : length(all_b[:,1])
        b = all_b[nb,:]

        best_action_index = 0
        best_action_value = -Inf

        for a = 1 : n_a

            Q_sa = 0

            for s = 1 : n_s

                O_distr = O[s,a,:]

                weight_b = scaling_entropy(O_distr,b)

                Q_sa += b[s] * (Q_MDP[s,a] * weight_b + Q_UMDP[s,a] * (1 - weight_b))

            end

            if best_action_value < Q_sa

                best_action_index = a
                best_action_value = Q_sa

            end

        end

        value_action[nb,:] = [best_action_value,best_action_index]

    end

    return value_action

end



function value_approx_v4(Q_0_MDP,Q_0_UMDP,T,R,O,delta,gamma,all_b)

    ##### UMDP is weighted by (1-w(b))(1-w(o)) #####

    Q_UMDP = QUMDP(Q_0_UMDP,T,R,delta,gamma) #Q_open(T,R,5,gamma)[3] #
    Q_MDP = Q_value_iteration(Q_0_MDP,T,R,delta,gamma)

    (n_s,n_a) = size(T)[1:2]
    n_o = size(T)[3]

    value_action = Array(Any,length(all_b[:,1]),2)

    for nb = 1 : length(all_b[:,1])
        b = all_b[nb,:]
        w_b = weight(b)[1]

        best_action_index = 0
        best_action_value = -Inf

        for a = 1 : n_a

            Q_sa = 0

            for s = 1 : n_s

                O_distr = O[s,a,:]

                w_o = weight(O_distr)[1]

                #www = (1-w_b) * (1-w_o)
                #Q_sa += b[s] * (Q_MDP[s,a] * (1-www) + Q_UMDP[s,a] * www)

                www = maximum([w_b w_o])

                Q_sa += b[s] * (Q_MDP[s,a] * (www) + Q_UMDP[s,a] * (1-www))

            end


            if best_action_value < Q_sa

                best_action_index = a
                best_action_value = Q_sa

            end

        end

        value_action[nb,:] = [best_action_value,best_action_index]

    end

    return value_action

end

function value_approx_v5(Q_0_MDP,Q_0_UMDP,T,R,O,delta,gamma,all_b)

    ##### UMDP is weighted by (1-w(bp))(1-w(o)), where bp is for the next state belief #####

    Q_UMDP = QUMDP(Q_0_UMDP,T,R,delta,gamma)
    Q_MDP = Q_value_iteration(Q_0_MDP,T,R,delta,gamma)

    (n_s,n_a) = size(T)[1:2]
    n_o = size(T)[3]

    value_action = Array(Any,length(all_b[:,1]),2)

    for nb = 1 : length(all_b[:,1])
        b = all_b[nb,:]


        best_action_index = 0
        best_action_value = -Inf

        for a = 1 : n_a

            Q_sa = 0

            for s = 1 : n_s

                bp = direct_update(b,T,a)

                w_b = weight(bp)[2]


                O_distr = O[s,a,:]

                w_o = weight(O_distr)[2]

                www = (1-w_b) * (1-w_o)

                Q_sa += b[s] * (Q_MDP[s,a] * (1-www) + Q_UMDP[s,a] * www)

            end

            if best_action_value < Q_sa

                best_action_index = a
                best_action_value = Q_sa

            end

        end

        value_action[nb,:] = [best_action_value,best_action_index]

    end

    return value_action

end

function value_approx_v6(Q_0_MDP,Q_0_UMDP,T,R,O,delta,gamma,all_b)

    ##### Taking the expectation over obsevation #####

    Q_UMDP = QUMDP(Q_0_UMDP,T,R,delta,gamma)
    Q_MDP = Q_value_iteration(Q_0_MDP,T,R,delta,gamma)

    (n_s,n_a) = size(T)[1:2]
    n_o = size(T)[3]

    value_action = Array(Any,length(all_b[:,1]),2)

    for nb = 1 : length(all_b[:,1])
        b = all_b[nb,:]
        #w_b = weight(b)[1]


        best_action_index = 0
        best_action_value = -Inf

        for a = 1 : n_a

            prob_o = zeros(Float64,n_o)

            bp1 = direct_update(b,T,a)

            for o = 1 : n_o
                for s = 1 : n_s
                    prob_o[o] += O[s,a,o] * bp1[s]

                end
            end

            Q_sa = 0

            for o = 1 : n_o
                for s = 1 : n_s
                    bp = belief_update(b,a,o,T,O)
                    w_b = exp((-1) * relative_entropy(bp,b))
                    Q_sa += b[s] * (Q_MDP[s,a] * w_b + Q_UMDP[s,a] * (1-w_b)) * prob_o[o]
                end
            end


            if best_action_value < Q_sa

                best_action_index = a
                best_action_value = Q_sa

            end

        end

        value_action[nb,:] = [best_action_value,best_action_index]

    end

    return value_action

end

function purely_iteration(Q_0,T,R,O,delta,gamma)

    (n_s,n_a) = size(T)[1:2]
    n_0 = size(O)[3]

    Q = zeros(Float64,n_s,n_a)

    sigma = Inf

    ##### Weight_matrix ######
    W = zeros(Float64,n_s,n_a)
    for s = 1 : n_s
        for a = 1 : n_a
            W[s,a] =  weight(O[s,a,:])[2]
        end
    end

    ##### Value Iteration #####

    while sigma > delta

        Q = zeros(Float64,n_s,n_a)

        for s = 1 : n_s
            for a = 1 : n_a
                ##### Imediate reward #####
                imediate_r = 0
                for sp = 1 : n_s
                    imediate_r += T[s,a,sp] * R[s,a,sp]
                end

                ##### next step : MDP piece #####

                MDP_r = 0
                for sp = 1 : n_s
                    V_MDP = maximum(Q_0[sp,:])
                    MDP_r += T[s,a,sp] * W[sp,a] * V_MDP
                end

                ##### next step : UMDP piece ####

                UMDP_r = -Inf

                for ap = 1 : n_a

                    umdp_ap = 0

                    for sp = 1 : n_s

                        umdp_ap += T[s,a,sp] * ( 1 - W[sp,a]) * Q_0[sp,ap]
                    end

                    if umdp_ap > UMDP_r
                        UMDP_r = umdp_ap
                    end
                end

                ##### Sum with discount factor #####

                Q[s,a] = imediate_r + 0.5 * gamma * (MDP_r + UMDP_r)

            end
        end

        sigma = norm( Q - Q_0, Inf)
        Q_0 = copy(Q)

    end

    return Q

end

function value_approx_purely(Q_0_MDP,T,R,O,delta,gamma,all_b)

    Q_MDP = purely_iteration(Q_0_MDP,T,R,O,delta,gamma)


    (n_s,n_a) = size(T)[1:2]

    value_action = Array(Any,length(all_b[:,1]),2)

    for nb = 1 : length(all_b[:,1])
        b = all_b[nb,:]


        best_action_index = 0
        best_action_value = -Inf

        for a = 1 : n_a

            Q_sa = 0

            for s = 1 : n_s

                Q_sa += b[s] * (Q_MDP[s,a] )

            end

            if best_action_value < Q_sa

                best_action_index = a
                best_action_value = Q_sa

            end

        end

        value_action[nb,:] = [best_action_value,best_action_index]

    end

    return value_action

end

function purely_iteration_v2(Q_0,T,R,O,delta,gamma)

    (n_s,n_a) = size(T)[1:2]
    n_0 = size(O)[3]

    Q = zeros(Float64,n_s,n_a)

    sigma = Inf

    ##### Weight_matrix ######
    W = weight_propogation(T,O,delta,gamma)

    ##### Value Iteration #####

    while sigma > delta

        Q = zeros(Float64,n_s,n_a)

        for s = 1 : n_s
            for a = 1 : n_a
                ##### Imediate reward #####
                imediate_r = 0
                for sp = 1 : n_s
                    imediate_r += T[s,a,sp] * R[s,a,sp]
                end

                ##### next step : MDP piece #####

                MDP_r = 0
                for sp = 1 : n_s
                    V_MDP = maximum(Q_0[sp,:])
                    MDP_r += T[s,a,sp] * V_MDP
                end

                ##### next step : UMDP piece ####

                UMDP_r = -Inf

                for ap = 1 : n_a

                    umdp_ap = 0

                    for sp = 1 : n_s

                        umdp_ap += T[s,a,sp] * Q_0[sp,ap]
                    end

                    if umdp_ap > UMDP_r
                        UMDP_r = umdp_ap
                    end
                end

                ##### Sum with discount factor #####
                W[s,a] = 0.05

                Q[s,a] = imediate_r + gamma * ( W[s,a] * MDP_r + ( 1 - W[s,a]) *UMDP_r)

            end
        end

        sigma = norm( Q - Q_0, Inf)
        Q_0 = copy(Q)

    end

    return Q

end

function value_approx_purely_v2(Q_0_MDP,T,R,O,delta,gamma,all_b)

    Q_MDP = purely_iteration_v2(Q_0_MDP,T,R,O,delta,gamma)


    (n_s,n_a) = size(T)[1:2]

    value_action = Array(Any,length(all_b[:,1]),2)

    for nb = 1 : length(all_b[:,1])
        b = all_b[nb,:]


        best_action_index = 0
        best_action_value = -Inf

        for a = 1 : n_a

            Q_sa = 0

            for s = 1 : n_s

                Q_sa += b[s] * (Q_MDP[s,a] )

            end

            if best_action_value < Q_sa

                best_action_index = a
                best_action_value = Q_sa

            end

        end

        value_action[nb,:] = [best_action_value,best_action_index]

    end

    return value_action

end

######################################
######### Sample from model ##########
##### Observation Model #####

function observe_sampling(O,s,a)
    n_o = (size(O)[3])
    distr = O[s,a,:]

    cdf_distr = zeros(n_o)

    cdf_distr[1] = distr[1]

    for i = 2 : n_o
        cdf_distr[i] = cdf_distr[i-1] + distr[i]
    end

    x = rand()

    for i = 1 : n_o
        if cdf_distr[i] >= x
            return i
        end
    end
end

##### Transition and Reward Model #####

function tran_reward_sampling(T,R,s,a)
    n_s = (size(T)[1])
    distr = T[s,a,:]

    cdf_distr = zeros(Float64,n_s+1)

    cdf_distr[1] = 0

    for i = 2 : n_s + 1
        cdf_distr[i] = cdf_distr[i-1] + distr[i-1]
    end

    cdf_distr[n_s + 1] = 1

    x = rand()
    sp = 0

    for i = 1 : n_s
        if (cdf_distr[i+1] > x)&(cdf_distr[i] <= x)
            sp = i
        end
    end

    r = R[s,a,sp]


    return (sp,r)

end

##### Choose action #####

function action_to_take(b,Q)

    (n_s,n_a) = size(Q)

    best_action = 0
    best_action_value = -Inf

    for a = 1 : n_a

        current_action_value = 0

        for s = 1 : n_s

            current_action_value += b[s] * Q[s,a]

        end

        if current_action_value > best_action_value

            best_action = a
            best_action_value = current_action_value

        end

    end

    return best_action

end

######################################
##############  Test 1 ###############
######### Crying Baby Problem ########
######################################

b_set = Array(Float64,101,2)

for i = 1 : 101

    b_set[i,1] = 0.01 * (i-1)
    b_set[i,2] = 1 - b_set[i,1]

end


T = Array(Float64,2,2,2)
T[1,2,1] = 1;   T[1,2,2] = 0;   T[1,1,2] = 1; T[1,1,1] = 0;
T[2,2,1] = 0.1; T[2,2,2] = 0.9; T[2,1,2] = 1; T[2,1,1] = 0;

O = Array(Float64,2,2,2)
o1 = 0.8; o2 = 0.9

O[2,2,1] = 1-o2; O[1,2,1] = o1; O[2,2,2] = o2; O[1,2,2] = 1-o1;
O[2,1,1] = 1-o2; O[1,1,1] = o1; O[2,1,2] = o2; O[1,1,2] = 1-o1;

R = Array(Float64,2,2,2)
R[2,2,2] = 0; R[2,1,2] = -5; R[1,2,2] = -10; R[1,1,2] = -15;
R[2,2,1] = 0; R[2,1,1] = -5; R[1,2,1] = -10; R[1,1,1] = -15;

b_0 = [0.5,0.5];

V1 = FIB(zeros(Float64,2,2),T,R,O,0.01,0.9)
#Z4 = value_approx_v4(zeros(Float64,2,2),zeros(Float64,2,2),T,R,O,0.01,0.9,b_set);
#MDP_c = Q_value_iteration(zeros(Float64,2,2),T,R,0.01,0.9)
#QMDP_v = value_approx_QMDP(zeros(Float64,2,2),T,R,0.01,0.9,b_set);

#XXX = FIB(zeros(Float64,2,2),T,R,O,0.01,0.9)

#plot(collect(1:1:51),QMDP_v[:,2])#,collect(1:1:51),Z4[:,2])



gamma = 0.9
delta = 0.01

Q_0 = zeros(Float64,2,2)

T_1 = Array(Float64,2,3,2)
T_1[1,1,1] = 0.5; T_1[1,1,2] = 0.5;
T_1[1,2,1] = 0.5; T_1[1,2,2] = 0.5;
T_1[1,3,1] = 0.5; T_1[1,3,2] = 0.5;

T_1[2,1,1] = 0.5; T_1[2,1,2] = 0.5;
T_1[2,2,1] = 0.5; T_1[2,2,1] = 0.5;
T_1[2,3,1] = 0.5; T_1[2,3,2] = 0.5;


R_1 = zeros(Float64,2,3,2)

R_1[1,1,1] = 1; R_1[1,1,2] = 1;
R_1[1,2,1] = 20; R_1[1,2,2] = 20;
R_1[1,3,1] = 1; R_1[1,3,2] = 1;

R_1[2,1,1] = 1; R_1[2,1,2] = 1;
R_1[2,2,1] = -20; R_1[2,2,1] = -20;
R_1[2,3,1] = 1; R_1[2,3,2] = 1;


#################################################
################### Test 2 ######################
###############  Tiger Problem ##################
#################################################

# S1 = Tiger at left
# S2 = Tiger at right
# A1 = Open left door
# A2 = Open right door
# A3 = Listen
# O1 = Tiger at left
# O2 = Tiger at right

T_T = Array(Float64,2,3,2)
T_T[1,1,1] = 0.5; T_T[1,1,2] = 0.5;
T_T[1,2,1] = 0.5; T_T[1,2,2] = 0.5;
T_T[1,3,1] = 1.0; T_T[1,3,2] = 0.0;
T_T[2,1,1] = 0.5; T_T[2,1,2] = 0.5;
T_T[2,2,1] = 0.5; T_T[2,2,2] = 0.5;
T_T[2,3,1] = 0.0; T_T[2,3,2] = 1.0;

R_T = Array(Float64,2,3,2)
cl = -10
R_T[1,1,1] = -100; R_T[1,1,2] = -100;
R_T[1,2,1] = 10; R_T[1,2,2] = 10;
R_T[1,3,1] = cl; R_T[1,3,2] = cl;
R_T[2,1,1] = 10; R_T[2,1,2] = 10;
R_T[2,2,1] = -100; R_T[2,2,2] = -100;
R_T[2,3,1] = cl; R_T[2,3,2] = cl;

O_T = Array(Float64,2,3,2)
best_observe = 0.5
O_T[1,1,1] = 0.5; O_T[1,1,2] = 0.5;
O_T[1,2,1] = 0.5; O_T[1,2,2] = 0.5;
O_T[1,3,1] = best_observe; O_T[1,3,2] = 1- best_observe;
O_T[2,1,1] = 0.5; O_T[2,1,2] = 0.5;
O_T[2,2,1] = 0.5; O_T[2,2,2] = 0.5;
O_T[2,3,1] = 1-best_observe; O_T[2,3,2] = best_observe;



#WWW = Q_open(T_T,R_T,5,gamma)[3];
#W = Q_value_iteration(zeros(Float64,2,3),T_T,R_T,delta,gamma)
#WW = QUMDP(zeros(Float64,2,3),T_T,R_T,delta,gamma)
#V = FIB(zeros(Float64,2,3),T_T,R_T,O_T,delta,gamma)



#b_set_1 = [0.96 0.4]

#Z1 = value_approx_v1(zeros(Float64,2,3),zeros(Float64,2,3),T_T,R_T,delta,gamma,b_set)
#Z4 = value_approx_v4(zeros(Float64,2,3),zeros(Float64,2,3),T_T,R_T,O_T,delta,gamma,b_set)

#X1 = value_approx_QMDP(zeros(Float64,2,3),T_T,R_T,delta,gamma,b_set);
#X2 = value_approx_purely(zeros(Float64,2,3),T_T,R_T,O_T,delta,gamma,b_set)
#X3 = value_approx_purely_v3(zeros(Float64,2,3),T_T,R_T,O_T,delta,gamma,b_set)
#WW1 = purely_iteration(zeros(Float64,2,3),T_T,R_T,O_T,delta,gamma)
#WW2 = purely_iteration_v2(zeros(Float64,2,3),T_T,R_T,O_T,delta,gamma)
#1
#plot(collect(1:1:51),Z6[:,1])
#New = new_approach_v1(zeros(Float64,2,3),zeros(Float64,2,3),T_T,R_T,O_T,delta,gamma,b_set)
