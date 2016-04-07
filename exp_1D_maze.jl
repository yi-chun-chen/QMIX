include("method_existing.jl")
include("method_QMIX.jl")

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

#for i = 1 : 4
#    R[i,1,4] = 1
#    R[i,2,4] = 1
#end

R[4,1,4] = 1.0
R[4,2,4] = 1.0


function one_1D_maze_trial(T,R,O,t_step,alpha,gamma)
    delta = 0.1;

    # initial belief
    b = zeros(Float64,4)
    b[1] = 1
    #b[1] = 0.3333
    #b[2] = 0.3333
    #b[3] = 0.3333

    # Initialize state
    x = 1
    #x = round(Int64,3*rand()) + 1

    # intialize total reward
    total_r = 0.0

    for t = 1 : t_step

        # Choose the action
        action_to_do = action_to_take(b,alpha)

        # Get reward and the next state
        (xp,r) = tran_reward_sampling(T,R,x,action_to_do)
        total_r += r

        # Get observation
        o = observe_sampling(O,xp,action_to_do)

        # update the belief
        bp = belief_update(b,action_to_do,o,T,O)

        # printing
        #println("Time Step = ",t)
        #println("S,A,O,R,SP",(x,action_to_do,o,r,xp))

        # update
        b = bp
        x = xp

    end

    return (total_r)

end

function one_1D_maze_trial_terminal(T,R,O,t_step,alpha,gamma)
    delta = 0.1;

    # initial belief
    b = zeros(Float64,4)
    b[1] = 0.3333
    b[2] = 0.3333
    b[3] = 0.3333

    # Initialize state
    x = round(Int64,3*rand()) + 1

    # intialize total reward
    total_r = 0.0

    for t = 1 : t_step

        # Choose the action
        action_to_do = action_to_take(b,alpha)

        # Get reward and the next state
        (xp,r) = tran_reward_sampling(T,R,x,action_to_do)
        total_r += r

        # Get observation
        o = observe_sampling(O,xp,action_to_do)

        # update the belief
        bp = belief_update(b,action_to_do,o,T,O)

        # printing
        #println("Time Step = ",t)
        #println("S,A,O,R,SP",(x,action_to_do,o,r,xp))

        # update
        b = bp
        x = xp

    end

    return (total_r)

end


gamma_simulation = 0.75
grid = 1

gamma_p = collect(linspace(0.75, 0.75, grid))
QMDP_r = zeros(Float64,grid)
UMDP_r = zeros(Float64,grid)
FIB_r = zeros(Float64,grid)


for (reduced_time, gamma) in enumerate(gamma_p)

    println(reduced_time)

    #redu_f = 1 + 0.1 * (reduced_time - 1)

    QMDP_alpha = Q_value_iteration(zeros(Float64,n_s,n_a),T,R,0.01,gamma)
    QUMDP_alpha = QUMDP(zeros(Float64,n_s,n_a),T,R,0.01,gamma)
    FIB_alpha = FIB(zeros(Float64,n_s,n_a),T,R,O,0.01,gamma)

    QMDP_r_sum = 0.0
    QUMDP_r_sum = 0.0
    FIB_r_sum = 0.0

    t_trial = 1000
    t_step = 200

    for i = 1 : t_trial

        QMDP_r_sum += one_1D_maze_trial(T,R,O,t_step,QMDP_alpha,gamma_simulation)
        QUMDP_r_sum += one_1D_maze_trial(T,R,O,t_step,QUMDP_alpha,gamma_simulation)
        FIB_r_sum += one_1D_maze_trial(T,R,O,t_step,FIB_alpha,gamma_simulation)

    end

    QMDP_r[reduced_time] = QMDP_r_sum/t_trial
    UMDP_r[reduced_time] = QUMDP_r_sum/t_trial
    FIB_r[reduced_time] = FIB_r_sum/t_trial

end

plot(gamma_p,QMDP_r,label="QMDP")
plot(gamma_p,UMDP_r,label="UMDP")
plot(gamma_p,FIB_r,label="FIB")
xlabel("discount factor")
ylabel("Reward")
title("Mini-Hallway with gamma 0.9 and uniform initial belief")
legend(loc="upper right",fancybox="true")
