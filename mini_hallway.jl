include("project_1.jl")
include("project_2.jl")

n_s = 13; n_a = 3; n_o = 9;

T = zeros(Float64,n_s,n_a,n_s)
R = zeros(Float64,n_s,n_a,n_s)
O = zeros(Float64,n_s,n_a,n_o)

T[1,1,1]=1.0
T[1,2,2]=1.0
T[1,3,4]=1.0
T[2,1,2]=1.0
T[2,2,3]=1.0
T[2,3,1]=1.0
T[3,1,7]=1.0
T[3,2,4]=1.0
T[3,3,2]=1.0
T[4,1,4]=1.0
T[4,2,1]=1.0
T[4,3,3]=1.0
T[5,1,1]=1.0
T[5,2,6]=1.0
T[5,3,8]=1.0
T[6,1,10]=1.0
T[6,2,7]=1.0
T[6,3,5]=1.0
T[7,1,1]=1.0
T[7,2,8]=1.0
T[7,3,6]=1.0
T[8,1,8]=1.0
T[8,2,5]=1.0
T[8,3,7]=1.0
T[9,1,9]=1.0
T[9,2,10]=1.0
T[9,3,12]=1.0
T[10,1,10]=1.0
T[10,2,11]=1.0
T[10,3,9]=1.0
T[11,1,13]=1.0
T[11,2,12]=1.0
T[11,3,10]=1.0
T[12,1,8]=1.0
T[12,2,9]=1.0
T[12,3,11]=1.0

for a = 1 : n_a
    T[13,a,:] = [0.083337 0.083333 0.083333 0.083333 0.083333 0.083333 0.083333 0.083333 0.083333 0.083333 0.083333 0.083333 0.0]
end

for a = 1 : n_a
    O[1,a,1] = 1
    O[2,a,2] = 1
    O[3,a,3] = 1
    O[4,a,4] = 1
    O[5,a,5] = 1
    O[6,a,6] = 1
    O[7,a,7] = 1
    O[8,a,8] = 1
    O[9,a,7] = 1
    O[10,a,8] = 1
    O[11,a,5] = 1
    O[12,a,6] = 1
    O[13,a,9] = 1
end

for s = 1 : n_s
    for a = 1 : n_a
        R[s,a,13] = 1
    end
end

#################################
gamma = 0.95

function one_mini_hallway_trial(T,R,O,t_step,alpha,gamma)
    (n_s,n_a,n_o) = (size(T)[1],size(T)[2],size(O)[3])

    # initial belief
    b = zeros(Float64,n_s)
    b[1:n_s-1] = 0.083333 * ones(Float64,n_s-1)

    # Initialize state
    x = round(Int64,div(rand()*(n_s-1),1)) + 1

    # intialize total reward
    total_r = 0

    for t = 1 : t_step

        # Choose the action
        action_to_do = action_to_take(b,alpha)

        # Get reward and the next state
        (xp,r) = tran_reward_sampling(T,R,x,action_to_do)
        total_r += r *  ((gamma)^t)

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

Q_MDP = Q_value_iteration(zeros(Float64,n_s,n_a),T,R,0.01,gamma/1.5)
Q_UMDP = QUMDP(zeros(Float64,n_s,n_a),T,R,0.01,gamma/1.5)
Q_FIB = FIB(zeros(Float64,n_s,n_a),T,R,O,0.01,gamma/1.5)
Q_M3 = purely_iteration_v3(zeros(Float64,n_s,n_a),T,R,O,0.01,gamma)
Q_M5 = purely_iteration_v5(zeros(Float64,n_s,n_a),T,R,O,0.01,gamma)
Q_M6 = purely_iteration_v6(zeros(Float64,n_s,n_a),T,R,O,0.01,gamma)
Q_M7 = purely_iteration_v7(zeros(Float64,n_s,n_a),T,R,O,0.01,gamma)

QMDP_r_sum = 0
QUMDP_r_sum = 0
FIB_r_sum = 0
MY_3_r_sum = 0
MY_5_r_sum = 0
MY_6_r_sum = 0
MY_7_r_sum = 0
t_trial = 2000
t_step = 200

for i = 1 : t_trial
    if (i%500 == 0); println("trial = ",i); end
    QMDP_r_sum += one_mini_hallway_trial(T,R,O,t_step,Q_MDP,gamma)
    QUMDP_r_sum += one_mini_hallway_trial(T,R,O,t_step,Q_UMDP,gamma)
    FIB_r_sum += one_mini_hallway_trial(T,R,O,t_step,Q_FIB,gamma)
    MY_3_r_sum += one_mini_hallway_trial(T,R,O,t_step,Q_M3,gamma)
    MY_5_r_sum += one_mini_hallway_trial(T,R,O,t_step,Q_M5,gamma)
    MY_6_r_sum += one_mini_hallway_trial(T,R,O,t_step,Q_M6,gamma)
    MY_7_r_sum += one_mini_hallway_trial(T,R,O,t_step,Q_M7,gamma)
end

println(QMDP_r_sum/t_trial)
println(QUMDP_r_sum/t_trial)
println(FIB_r_sum/t_trial)
println(MY_3_r_sum/t_trial)
println(MY_5_r_sum/t_trial)
println(MY_7_r_sum/t_trial)
println(MY_6_r_sum/t_trial)
