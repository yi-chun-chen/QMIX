include("project_1.jl")
include("project_2.jl")

n_s = 4; n_a = 2; n_o = 2;

T = zeros(Float64,n_s,n_a,n_s)

Ta1 = [
    [1.0 0.0 0.0 0.0 ];
    [1.0 0.0 0.0 0.0 ];
    [0.0 0.0 0.0 1.0 ];
    [0.333333 0.333333 0.333333 0.0]];

Ta2 = [
    [0.0 1.0 0.0 0.0 ];
    [0.0 0.0 0.0 1.0 ];
    [0.0 0.0 1.0 0.0 ];
    [0.333333 0.333333 0.333333 0.0]];

for i = 1 : 4
    for j = 1 : 4
        T[i,1,j] = Ta1[i,j]
        T[i,2,j] = Ta2[i,j]
    end
end

O = zeros(Float64,n_s,n_a,n_o)

OO = [
[1.0 0.0];
[1.0 0.0];
[1.0 0.0];
[0.0 1.0]]

for i = 1 : 4
    for j = 1 : 2
        O[i,1,j] = OO[i,j]
        O[i,2,j] = OO[i,j]
    end
end

R = zeros(Float64,n_s,n_a,n_s)

for i = 1 : 4
    R[i,1,4] = 1
    R[i,2,4] = 1
end

Q_MDP = Q_value_iteration(zeros(Float64,4,2),T,R,0.01,0.9)
Q_UMDP = QUMDP(zeros(Float64,4,2),T,R,0.01,0.9)
Q_FIB = FIB(zeros(Float64,4,2),T,R,O,0.01,0.9)
Q_M1 = purely_iteration(zeros(Float64,4,2),T,R,O,0.01,0.9)
Q_M2 = purely_iteration_v2(zeros(Float64,4,2),T,R,O,0.01,0.9)
Q_M3 = purely_iteration_v3(zeros(Float64,4,2),T,R,O,0.01,0.9)

function one_1D_maze_trial(T,R,O,t_step,alpha)
    delta = 0.1; gamma = 0.9;

    # initial belief
    b = zeros(Float64,4)
    b[1] = 0.3333
    b[2] = 0.3333
    b[3] = 0.3333

    # Initialize state
    x = round(Int64,3*rand()) + 1

    # intialize total reward
    total_r = 0

    for t = 1 : t_step

        # Choose the action
        action_to_do = action_to_take(b,alpha)

        # Get reward and the next state
        (xp,r) = tran_reward_sampling(T,R,x,action_to_do)
        total_r += r

        # Get observation
        o = observe_sampling(O,x,action_to_do)

        # update the belief
        bp = belief_update(b,action_to_do,o,T,O)

        # printing
        #println("Time Step = ",t)
        #println("S,A,O,R,SP",(x,action_to_do,o,r,xp))

        # update
        b = bp
        x = xp

    end

    return (total_r / t_step)

end

function one_1D_maze_trial_terminal(T,R,O,t_step,alpha)
    delta = 0.1; gamma = 0.9;

    # initial belief
    b = zeros(Float64,4)
    b[1] = 0.3333
    b[2] = 0.3333
    b[3] = 0.3333

    # Initialize state
    x = round(Int64,3*rand()) + 1

    # intialize total reward
    total_r = 0

    for t = 1 : t_step

        # Choose the action
        action_to_do = action_to_take(b,alpha)

        # Get reward and the next state
        (xp,r) = tran_reward_sampling(T,R,x,action_to_do)
        total_r += r

        # Get observation
        o = observe_sampling(O,x,action_to_do)

        # update the belief
        bp = belief_update(b,action_to_do,o,T,O)

        # printing
        #println("Time Step = ",t)
        #println("S,A,O,R,SP",(x,action_to_do,o,r,xp))

        # update
        b = bp
        x = xp

    end

    return (total_r / t_step)

end


QMDP_r_sum = 0
QUMDP_r_sum = 0
FIB_r_sum = 0
MY_1_r_sum = 0
MY_2_r_sum = 0
MY_3_r_sum = 0
t_trial = 10000

for i = 1 : t_trial
    if (i%100 == 0); println("trial = ",i); end
    QMDP_r_sum += one_1D_maze_trial(T,R,O,60,Q_MDP)
    QUMDP_r_sum += one_1D_maze_trial(T,R,O,60,Q_UMDP)
    FIB_r_sum += one_1D_maze_trial(T,R,O,60,Q_FIB)
    MY_1_r_sum += one_1D_maze_trial(T,R,O,60,Q_M1)
    MY_2_r_sum += one_1D_maze_trial(T,R,O,60,Q_M2)
    MY_3_r_sum += one_1D_maze_trial(T,R,O,60,Q_M3)
end

println((QMDP_r_sum/t_trial,QUMDP_r_sum/t_trial,FIB_r_sum/t_trial))
println((MY_1_r_sum/t_trial,MY_2_r_sum/t_trial,MY_3_r_sum/t_trial))
