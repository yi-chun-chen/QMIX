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