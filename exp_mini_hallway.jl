include("method_existing.jl")
include("method_QMIX.jl")

#############################
###### The model data #######
#############################

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
    O[1,a,1] = 1.0
    O[2,a,2] = 1.0
    O[3,a,3] = 1.0
    O[4,a,4] = 1.0
    O[5,a,5] = 1.0
    O[6,a,6] = 1.0
    O[7,a,7] = 1.0
    O[8,a,8] = 1.0
    O[9,a,7] = 1.0
    O[10,a,8] = 1.0
    O[11,a,5] = 1.0
    O[12,a,6] = 1.0
    O[13,a,9] = 1.0
end

for s = 1 : n_s
    for a = 1 : n_a
        R[s,a,13] = 1.0
    end
end

###############################
### Function for simulation ###
###############################


function one_mini_hallway_trial(
    T      :: Array{Float64,3}, # Transition model (s,a,s')
    R      :: Array{Float64,3}, # Reward model (s,a,s')
    O      :: Array{Float64,3}, # Observation model (s,a,o)
    t_step :: Int64,            # horizon for simulation / simulation length
    alpha  :: Matrix{Float64},  # alpha vector for deciding action
    gamma  :: Float64,          # disount factor
    )

    n_s = size(T,1)
    n_a = size(T,2)
    n_o = size(O,3)

    # initial belief
    b = zeros(Float64,n_s)
    b[1:n_s-1] = 0.083333 * ones(Float64,n_s-1)

    # Initialize state
    x = round(Int64,div(rand()*(n_s-1),1)) + 1

    # intialize total reward
    total_r = 0.0

    for t in 1 : t_step

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

#gamma = 0.95

#Q_MDP = Q_value_iteration(zeros(Float64,n_s,n_a),T,R,0.01,gamma/1.5)
#Q_UMDP = QUMDP(zeros(Float64,n_s,n_a),T,R,0.01,gamma/1.5)
#Q_FIB = FIB(zeros(Float64,n_s,n_a),T,R,O,0.01,gamma/1.5)
#Q_M3 = purely_iteration_v3(zeros(Float64,n_s,n_a),T,R,O,0.01,gamma)
#Q_M5 = purely_iteration_v5(zeros(Float64,n_s,n_a),T,R,O,0.01,gamma)
#Q_M6 = purely_iteration_v6(zeros(Float64,n_s,n_a),T,R,O,0.01,gamma)
#Q_M7 = purely_iteration_v7(zeros(Float64,n_s,n_a),T,R,O,0.01,gamma)

#QMDP_r_sum = 0
#QUMDP_r_sum = 0
#FIB_r_sum = 0
#MY_3_r_sum = 0
#MY_5_r_sum = 0
#MY_6_r_sum = 0
#MY_7_r_sum = 0
#t_trial = 2000
#t_step = 200

#for i = 1 : t_trial
#    if (i%500 == 0); println("trial = ",i); end
#    QMDP_r_sum += one_mini_hallway_trial(T,R,O,t_step,Q_MDP,gamma)
#    QUMDP_r_sum += one_mini_hallway_trial(T,R,O,t_step,Q_UMDP,gamma)
#    FIB_r_sum += one_mini_hallway_trial(T,R,O,t_step,Q_FIB,gamma)
#    MY_3_r_sum += one_mini_hallway_trial(T,R,O,t_step,Q_M3,gamma)
#    MY_5_r_sum += one_mini_hallway_trial(T,R,O,t_step,Q_M5,gamma)
#    MY_6_r_sum += one_mini_hallway_trial(T,R,O,t_step,Q_M6,gamma)
#    MY_7_r_sum += one_mini_hallway_trial(T,R,O,t_step,Q_M7,gamma)
#end

#println(QMDP_r_sum/t_trial)
#println(QUMDP_r_sum/t_trial)
#println(FIB_r_sum/t_trial)
#println(MY_3_r_sum/t_trial)
#println(MY_5_r_sum/t_trial)
#println(MY_7_r_sum/t_trial)
#println(MY_6_r_sum/t_trial)




gamma_simulation = 0.95
grid = 100

gamma_p = collect(linspace(0.95, 0.5, grid))
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

    t_trial = 2000
    t_step = 200

    for i = 1 : t_trial

        QMDP_r_sum += one_mini_hallway_trial(T,R,O,t_step,QMDP_alpha,gamma_simulation)
        QUMDP_r_sum += one_mini_hallway_trial(T,R,O,t_step,QUMDP_alpha,gamma_simulation)
        FIB_r_sum += one_mini_hallway_trial(T,R,O,t_step,FIB_alpha,gamma_simulation)

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
#annotate("SARSOP = 1.217",xy=[1;0],xycoords="axes fraction",xytext=[-10,10],textcoords="offset points",fontsize=12.0,ha="right",va="bottom")
