# This file implements QMDP, FIB, QUMDP, and the sampling of environment model

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

#########################################
########## Existing Methods #############
#########################################

function Q_value_iteration(Q_0,T,R,delta,gamma)

    ##################
    # MDP Q-function #
    ##################

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


function Q_open(T,R,d,gamma)

    ###########################
    # My new open loop method #
    ###########################
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

    #########################################
    # Alpha vectors for Fast Informed Bound #
    #########################################

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

    ##########################
    # Alpha vectors for UMDP #
    ##########################

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

#####################################
# Some functions related to entropy #
#####################################

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

    ###########################################
    # Update belief without observation model #
    ###########################################

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


######################################
######### Sample from model ##########
######################################

##### Sample from initial belief #####

function sample_from_belief(b)

    n_s = length(b)
    b_cum = zeros(Float64,n_s)
    b_cum[1] = b[1]

    for i = 2 : n_s
        b_cum[i] = b_cum[i-1] + b[i]
    end

    x = rand()

    for i = 1 : n_s
        if b_cum[i] >= x
            return i
        end
    end
end


##### Observation Model #####

function observe_sampling(O,s,a)

    # Given next state s and current action a, sampling observation o

    n_o = (size(O)[3])
    distr = O[s,a,:]

    # Build cumulative probability distribution

    cdf_distr = zeros(n_o)

    cdf_distr[1] = distr[1]

    for i = 2 : n_o
        cdf_distr[i] = cdf_distr[i-1] + distr[i]
    end

    # Sampling

    x = rand()

    for i = 1 : n_o
        if cdf_distr[i] >= x
            return i
        end
    end

end

##### Transition and Reward Model #####

function tran_reward_sampling(T,R,s,a)

    # Given current state s and action a, sample next state sp and obtain immediate reward r

    n_s = (size(T)[1])
    distr = T[s,a,:]

    # Build cumulative probability distribution

    cdf_distr = zeros(Float64,n_s+1)

    cdf_distr[1] = 0

    for i = 2 : n_s + 1
        cdf_distr[i] = cdf_distr[i-1] + distr[i-1]
    end

    cdf_distr[n_s + 1] = 1

    # Sample next state sp
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

##### Choose action from the state-action pair alpha vectors #####

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
cl = -1 # Cost for listening
R_T[1,1,1] = -100; R_T[1,1,2] = -100;
R_T[1,2,1] = 10; R_T[1,2,2] = 10;
R_T[1,3,1] = cl; R_T[1,3,2] = cl;
R_T[2,1,1] = 10; R_T[2,1,2] = 10;
R_T[2,2,1] = -100; R_T[2,2,2] = -100;
R_T[2,3,1] = cl; R_T[2,3,2] = cl;

O_T = Array(Float64,2,3,2)
best_observe = 0.85
O_T[1,1,1] = 0.5; O_T[1,1,2] = 0.5;
O_T[1,2,1] = 0.5; O_T[1,2,2] = 0.5;
O_T[1,3,1] = best_observe; O_T[1,3,2] = 1- best_observe;
O_T[2,1,1] = 0.5; O_T[2,1,2] = 0.5;
O_T[2,2,1] = 0.5; O_T[2,2,2] = 0.5;
O_T[2,3,1] = 1-best_observe; O_T[2,3,2] = best_observe;

