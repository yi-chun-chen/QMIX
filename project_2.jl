include("project_1.jl")

function sample_in_belief_space(n_s)

    sample = zeros(Float64,n_s)
    total_left = 1

    for i = 1 : n_s-1

        sample[i] = rand() * total_left
        total_left = total_left - sample[i]

    end

    sample[end] = total_left

    return sample

end

function new_weight(T,O)

    (n_s,n_a,n_o) = (size(T)[1],size(T)[2],size(O)[3])

    W = zeros(Float64,n_s,n_a,n_s)

    for s = 1 : n_s

        for a = 1 : n_a

            o_distr = O[s,a,:]

            b = zeros(Float64,n_s); b[s] = 1

            for o = 1 : n_o

                if o_distr[o] != 0

                update_at_sp = belief_update(b,a,o,T,O)

                for sp = 1 : n_s

                    adding = update_at_sp[sp] * o_distr[o]
                    #println((s,a,o,sp),(update_at_sp[sp],o_distr[o]),adding)
                    W[s,a,sp] += adding

                end

                end

            end

        end

    end

    return W

end

function new_weight_2(T,O)

    (n_s,n_a,n_o) = (size(T)[1],size(T)[2],size(O)[3])

    W = zeros(Float64,n_s,n_a,n_s)

    for s = 1 : n_s

        for a = 1 : n_a

            b = zeros(Float64,n_s); b[s] = 1

            for sp = 1 : n_s

                o_distr = O[sp,a,:]

                for o = 1 : n_o

                    if o_distr[o] != 0

                        update_at_sp = belief_update(b,a,o,T,O)
                        adding = update_at_sp[sp] * o_distr[o]
                        W[s,a,sp] += adding
                        #if update_at_sp[sp] != 0.0; println((o,o_distr[o],update_at_sp[sp],adding)); end

                    end

                end
                #if (s,a,sp) == (1,1,1); return W[s,a,sp]; end
            end


        end



    end

    return W

end

function trust_score(T,O)

    (n_s,n_a,n_o) = (size(T)[1],size(T)[2],size(O)[3])

    W = new_weight_2(T,O)

    ts = zeros(Float64,n_s,n_a)

    for s = 1 : n_s

        for a = 1 : n_a

            prob_vec = vec(W[s,a,:])
            prob_vec = prob_vec / sum(prob_vec)
            ts[s,a] = weight(prob_vec)[2]

        end

    end

    return ts

end

function trust_score_2(T,O)

    (n_s,n_a,n_o) = (size(T)[1],size(T)[2],size(O)[3])

    W = new_weight_2(T,O)

    ts = zeros(Float64,n_s,n_a)

    for s = 1 : n_s

        for a = 1 : n_a

            prob_vec = vec(W[s,a,:])
            max_p = maximum(prob_vec)
            ts[s,a] = max_p / sum(prob_vec)

        end

    end

    return ts

end


function purely_iteration_v3(Q_0,T,R,O,delta,gamma)

    ############################
    ##### the 1/2 version ######
    ############################

    (n_s,n_a) = size(T)[1:2]
    n_0 = size(O)[3]

    Q = zeros(Float64,n_s,n_a)

    sigma = Inf

    ##### Weight_matrix ######
    W = new_weight_2(T,O)

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
                    MDP_r += T[s,a,sp] * W[s,a,sp] * V_MDP
                end

                ##### next step : UMDP piece ####

                UMDP_r = -Inf

                for ap = 1 : n_a

                    umdp_ap = 0

                    for sp = 1 : n_s

                        umdp_ap += T[s,a,sp] * ( 1 - W[s,a,sp]) * Q_0[sp,ap]
                    end

                    if umdp_ap > UMDP_r
                        UMDP_r = umdp_ap
                    end
                end

                ##### Sum with discount factor #####

                Q[s,a] = imediate_r + 0.5 * gamma * (MDP_r + UMDP_r)

            end
        end
        #println(norm(Q,Inf))
        sigma = norm( Q - Q_0, Inf)
        Q_0 = copy(Q)

    end

    return Q

end

function value_approx_purely_v3(Q_0_MDP,T,R,O,delta,gamma,all_b)

    Q_MDP = purely_iteration_v3(Q_0_MDP,T,R,O,delta,gamma)


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

#value_approx_purely_v3(zeros(Float64,2,3),T_T,R_T,O_T,delta,gamma,b_set)

function purely_iteration_v4(Q_0,T,R,O,delta,gamma)

    ##### The normalized version #####

    (n_s,n_a) = size(T)[1:2]
    n_0 = size(O)[3]

    Q = zeros(Float64,n_s,n_a)

    sigma = Inf

    ##### Weight_matrix ######
    W = new_weight(T,O)

    ##### Value Iteration #####

    TP1 = zeros(Float64,n_s,n_a,n_s)

    for s = 1 : n_s
        for a = 1 : n_a
            for sp = 1 : n_s
                TP1[s,a,sp] = T[s,a,sp] * W[s,a,sp]
            end
            TP1[s,a,:] = TP1[s,a,:]/sum(TP1[s,a,:])
        end
    end

    TP2 = zeros(Float64,n_s,n_a,n_s)

    for s = 1 : n_s
        for a = 1 : n_a
            for sp = 1 : n_s
                TP2[s,a,sp] = T[s,a,sp] * (1 - W[s,a,sp])
            end
            TP2[s,a,:] = TP2[s,a,:]/sum(TP2[s,a,:])
        end
    end


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
                    MDP_r += TP1[s,a,sp] * V_MDP
                end

                ##### next step : UMDP piece ####

                UMDP_r = -Inf

                for ap = 1 : n_a

                    umdp_ap = 0

                    for sp = 1 : n_s

                        umdp_ap += TP2[s,a,sp] * Q_0[sp,ap]
                    end

                    if umdp_ap > UMDP_r
                        UMDP_r = umdp_ap
                    end
                end

                ##### Sum with discount factor #####

                Q[s,a] = imediate_r + 0.5 * gamma * (MDP_r + UMDP_r)

            end
        end
        #println(norm(Q,Inf))
        sigma = norm( Q - Q_0, Inf)
        Q_0 = copy(Q)

    end

    return Q

end

function value_approx_purely_v4(Q_0_MDP,T,R,O,delta,gamma,all_b)

    Q_MDP = purely_iteration_v4(Q_0_MDP,T,R,O,delta,gamma)


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

#X3 = value_approx_purely_v3(zeros(Float64,2,3),T_T,R_T,O_T,delta,gamma,b_set)
#X4 = value_approx_purely_v4(zeros(Float64,2,3),T_T,R_T,O_T,delta,gamma,b_set)

function purely_iteration_v5(Q_0,T,R,O,delta,gamma)

    ##### The ratio version #####

    (n_s,n_a) = size(T)[1:2]
    n_0 = size(O)[3]

    Q = zeros(Float64,n_s,n_a)

    sigma = Inf

    ##### Weight_matrix ######
    W = new_weight_2(T,O)
    TS = trust_score_2(T,O)

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
                    MDP_r += T[s,a,sp] * W[s,a,sp] * V_MDP
                end

                ##### next step : UMDP piece ####

                UMDP_r = -Inf

                for ap = 1 : n_a

                    umdp_ap = 0

                    for sp = 1 : n_s

                        umdp_ap += T[s,a,sp] * ( 1 - W[s,a,sp]) * Q_0[sp,ap]

                    end

                    if umdp_ap > UMDP_r
                        UMDP_r = umdp_ap
                    end
                end

                ##### Sum with discount factor #####
                #println((www_1,www_2,www_1+www_2))
                www = (n_o - 1)/(n_s - 1)

                Q[s,a] = imediate_r + gamma * (www * MDP_r + (1 - www) * UMDP_r)

            end
        end
        #println(norm(Q,Inf))
        sigma = norm( Q - Q_0, Inf)
        Q_0 = copy(Q)

    end

    return Q

end

function purely_iteration_v6(Q_0,T,R,O,delta,gamma)

    ##### The 0 version #####

    (n_s,n_a) = size(T)[1:2]
    n_0 = size(O)[3]

    Q = zeros(Float64,n_s,n_a)

    sigma = Inf

    ##### Weight_matrix ######
    W = new_weight_2(T,O)
    TS = trust_score_2(T,O)

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
                    MDP_r += T[s,a,sp] * W[s,a,sp] * V_MDP
                end

                ##### next step : UMDP piece ####

                UMDP_r = -Inf

                for ap = 1 : n_a

                    umdp_ap = 0

                    for sp = 1 : n_s

                        umdp_ap += T[s,a,sp] * ( 1 - W[s,a,sp]) * Q_0[sp,ap]

                    end

                    if umdp_ap > UMDP_r
                        UMDP_r = umdp_ap
                    end
                end

                ##### Sum with discount factor #####
                #println((www_1,www_2,www_1+www_2))
                www = 0

                Q[s,a] = imediate_r + gamma * (www * MDP_r + (1 - www) * UMDP_r)

            end
        end
        #println(norm(Q,Inf))
        sigma = norm( Q - Q_0, Inf)
        Q_0 = copy(Q)

    end

    return Q

end

function purely_iteration_v7(Q_0,T,R,O,delta,gamma)

    ##### The 1 version #####

    (n_s,n_a) = size(T)[1:2]
    n_0 = size(O)[3]

    Q = zeros(Float64,n_s,n_a)

    sigma = Inf

    ##### Weight_matrix ######
    W = new_weight_2(T,O)
    TS = trust_score_2(T,O)

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
                    MDP_r += T[s,a,sp] * W[s,a,sp] * V_MDP
                end

                ##### next step : UMDP piece ####

                UMDP_r = -Inf

                for ap = 1 : n_a

                    umdp_ap = 0

                    for sp = 1 : n_s

                        umdp_ap += T[s,a,sp] * ( 1 - W[s,a,sp]) * Q_0[sp,ap]

                    end

                    if umdp_ap > UMDP_r
                        UMDP_r = umdp_ap
                    end
                end

                ##### Sum with discount factor #####
                #println((www_1,www_2,www_1+www_2))
                www = 1

                Q[s,a] = imediate_r + gamma * (www * MDP_r + (1 - www) * UMDP_r)

            end
        end
        #println(norm(Q,Inf))
        sigma = norm( Q - Q_0, Inf)
        Q_0 = copy(Q)

    end

    return Q

end

