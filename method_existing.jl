# This file implements QMDP, FIB, QUMDP, and the sampling of environment model

using PyPlot

function belief_update!(
    b_p::Vector{Float64},
    b::Vector{Float64},
    a::Int64,
    o::Int64,
    T::Array{Float64,3}, # Transition model T(s,a,sp)
    O::Array{Float64,3}  # Observation model O(s,a,o)
    )

    n_s = length(b)
    for s_p in 1 : n_s
        sum_s = 0.0
        for i in 1 : n_s
            sum_s += T[i,a,s_p] * b[i]
        end
        b_p[s_p] = O[s_p,a,o] * sum_s
    end


    #if sum(b_p) == 0.0
    #    fill!(b_p, 1.0)
    #end

    b_p ./= sum(b_p)

    return b_p
end

function Q_value_iteration(
    Q_0::Matrix{Float64}, # Initial Q matrix
    T::Array{Float64,3},  # Transition model (s,a,s')
    R::Array{Float64,3},  # Reward model (s,a,s')
    delta::Float64,       # value iteration Q-value Inf-norm tolerance
    gamma::Float64,       # discount factor
    )

    ##################
    # MDP Q-function #
    ##################

    sigma = Inf
    n_s = size(T,1)
    n_a = size(T,2)

    Q = zeros(Float64,n_s,n_a)

    while sigma > delta
        for s in 1 : n_s
            for a in 1 : n_a
                sum_s_p = 0.0
                for s_p in 1 : n_s
                    V = maximum(Q_0[s_p,:])
                    sum_s_p += T[s,a,s_p] * ( R[s,a,s_p] + gamma * V)
                end
            Q[s,a] = sum_s_p
            end
        end
        sigma = norm(Q-Q_0,Inf)
        copy!(Q_0, Q)
    end

    return Q_0
end



function FIB(
    alpha_0 :: Matrix{Float64},
    T :: Array{Float64,3},
    R :: Array{Float64,3},
    O :: Array{Float64,3},
    delta :: Float64,
    gamma :: Float64,
    )

    #########################################
    # Alpha vectors for Fast Informed Bound #
    #########################################

    sigma = Inf
    n_s = size(T,1)
    n_a = size(T,2)
    n_o = size(O,3)

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

                        ap_value = 0.0

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
        copy!(alpha_0,alpha)

    end

    return alpha
end

function QUMDP(
    Q_0::Matrix{Float64}, # Initial Q matrix
    T::Array{Float64,3},  # Transition model (s,a,s')
    R::Array{Float64,3},  # Reward model (s,a,s')
    delta::Float64,       # value iteration Q-value Inf-norm tolerance
    gamma::Float64,       # discount factor
    )

    ##########################
    # Alpha vectors for UMDP #
    ##########################

    sigma = Inf
    n_s = size(T,1)
    n_a = size(T,2)

    Q = zeros(Float64,n_s,n_a)

    while sigma > delta
        for s in 1 : n_s
            for a in 1 : n_a

                value_ap_index = 0
                value_ap = -Inf


                for ap in 1 : n_a
                    value_sp = 0.0
                    for sp in 1 : n_s
                        value_sp += T[s,a,sp] * Q_0[sp,ap]
                    end

                    if value_sp > value_ap
                        value_ap_index = ap
                        value_ap = value_sp
                    end

                end

                im_reward = 0.0
                for sp = 1 : n_s
                    im_reward += T[s,a,sp] * R[s,a,sp]
                end

                Q[s,a] = im_reward + gamma * value_ap

            end
        end

        sigma = norm(Q-Q_0,Inf)
        copy!(Q_0,Q)
    end

    return Q
end


######################################
######### Sample from model ##########
######################################

######################################
##### Sample from initial belief #####
######################################

function sample_from_vector(b::Vector{Float64})

    x = rand()
    i = 1
    b_cum = 0.0

    while b_cum + b[i] < x && i < length(b)
        i += 1
        b_cum += b[i]
    end

    return i
end
sample_from_belief(b::Vector{Float64}) = sample_from_vector(b)

#############################
##### Observation Model #####
#############################

function observe_sampling(O,s,a)

    # Given next state s and current action a, sampling observation o

    n_o = size(O,3)
    distr = vec(O[s,a,:])
    return sample_from_vector(distr)
end

############################
##### Transition Model #####
############################

function tran_sampling(T,s,a)

    # Given current state s and action a, sample next state sp
    n_s = size(T,1)
    distr = vec(T[s,a,:])
    sp = sample_from_vector(distr)

    return sp

end

##### Choose action from the state-action pair alpha vectors #####

function action_to_take(b::Vector{Float64},Q::Matrix{Float64})

    (n_s,n_a) = size(Q)

    best_action = 0
    best_action_value = -Inf

    for a in 1 : n_a

        current_action_value = 0.0

        for s in 1 : n_s
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
R[2,2,2] = 0; R[2,1,2] = -5; R[1,2,2] = -10; R[1,1,2] = -15
R[2,2,1] = 0; R[2,1,1] = -5; R[1,2,1] = -10; R[1,1,1] = -15

b_0 = [0.5,0.5]

