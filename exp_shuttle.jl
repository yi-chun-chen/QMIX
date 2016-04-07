include("method_existing.jl")
include("method_QMIX.jl")

n_s = 8; n_a = 3; n_o = 5;

T = zeros(Float64,n_s,n_a,n_s)

Ta1 = [
[0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0  ];
[0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0  ];
[0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0  ];
[0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0  ];
[0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0  ];
[0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0  ];
[0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0  ];
[0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 ]];

Ta2 = [
    [0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0  ];
    [0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0  ];
    [0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0  ];
    [0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0  ];
    [0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0  ];
    [0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0  ];
    [0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0  ];
    [0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0]
]

Ta3 = [
    [0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 ];
    [0.0 0.4 0.3 0.0 0.3 0.0 0.0 0.0 ];
    [0.0 0.0 0.1 0.8 0.0 0.0 0.1 0.0 ];
    [0.7 0.0 0.0 0.3 0.0 0.0 0.0 0.0 ];
    [0.0 0.0 0.0 0.0 0.3 0.0 0.0 0.7 ];
    [0.0 0.1 0.0 0.0 0.8 0.1 0.0 0.0 ];
    [0.0 0.0 0.0 0.3 0.0 0.3 0.4 0.0 ];
    [0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0]]

for i = 1 : n_s
    for j = 1 : n_s
        T[i,1,j] = Ta1[i,j]
        T[i,2,j] = Ta2[i,j]
        T[i,3,j] = Ta3[i,j]
    end
end

O = zeros(Float64,n_s,n_a,n_o)
Oa = [
    [0.0 0.0 0.0 0.0 1.0 ];
    [0.0 1.0 0.0 0.0 0.0 ];
    [0.0 0.7 0.0 0.3 0.0 ];
    [0.0 0.0 0.0 1.0 0.0 ];
    [0.0 0.0 0.0 1.0 0.0 ];
    [0.7 0.0 0.0 0.3 0.0 ];
    [1.0 0.0 0.0 0.0 0.0 ];
    [0.0 0.0 1.0 0.0 0.0 ]];

for i = 1 : n_s
    for j = 1 : n_o
        O[i,1,j] = Oa[i,j]
        O[i,2,j] = Oa[i,j]
        O[i,3,j] = Oa[i,j]
    end
end

R = zeros(Float64,n_s,n_a,n_s)
R[2,2,2] = -3
R[7,2,7] = -3
R[4,3,1] = 10

#Q_MDP = Q_value_iteration(zeros(Float64,8,3),T,R,0.01,0.95)
#Q_UMDP = QUMDP(zeros(Float64,8,3),T,R,0.01,0.95)
#Q_FIB = FIB(zeros(Float64,8,3),T,R,O,0.01,0.95)
#Q_M3 = purely_iteration_v3(zeros(Float64,8,3),T,R,O,0.01,0.95)
#Q_M5 = purely_iteration_v5(zeros(Float64,8,3),T,R,O,0.01,0.95)
#Q_M6 = purely_iteration_v6(zeros(Float64,8,3),T,R,O,0.01,0.95)
#Q_M7 = purely_iteration_v7(zeros(Float64,8,3),T,R,O,0.01,0.95)

function one_shuttle_trial(T,R,O,t_step,alpha,gamma)
    delta = 0.1

    # initial belief
    #b = zeros(Float64,8)
    b = ones(Float64,8) * (1/8)

    # Initialize state
    x = round(Int64,div(rand()*(8),1)) + 1

    # intialize total reward
    total_r = 0.0
    for t in 1 : t_step

        # Choose the action
        action_to_do = action_to_take(b,alpha)

        # Get reward and the next state
        (xp,r) = tran_reward_sampling(T,R,x,action_to_do)
        xp = tran_sampling(T, x, action_to_do)
        r = R[x,action_to_do,xp]
        total_r += r *  ((gamma)^(t))

        # Get observation
        o = observe_sampling(O,xp,action_to_do)

        # update the belief
        belief_update!(b_p, b,action_to_do,o,T,O)

        # printing
        #println("Time Step = ",t)
        #println("S,A,O,R,SP",(x,action_to_do,o,r,xp))

        # update
        copy!(b, b_p)
        x = xp
    end

    return total_r

end

#QMDP_r_sum = 0
#QUMDP_r_sum = 0
#FIB_r_sum = 0
#MY_3_r_sum = 0
#MY_5_r_sum = 0
#MY_6_r_sum = 0
#MY_7_r_sum = 0
#t_trial = 2000
#t_step = 150

#for i = 1 : t_trial
#    if (i%100 == 0); println("trial = ",i); end
#    QMDP_r_sum += one_shuttle_trial(T,R,O,t_step,Q_MDP)
#    QUMDP_r_sum += one_shuttle_trial(T,R,O,t_step,Q_UMDP)
#    FIB_r_sum += one_shuttle_trial(T,R,O,t_step,Q_FIB)
#    MY_3_r_sum += one_shuttle_trial(T,R,O,t_step,Q_M3)
#    MY_5_r_sum += one_shuttle_trial(T,R,O,t_step,Q_M5)
#    MY_6_r_sum += one_shuttle_trial(T,R,O,t_step,Q_M6)
#    MY_7_r_sum += one_shuttle_trial(T,R,O,t_step,Q_M7)
#end

#println(QMDP_r_sum/t_trial)
#println(QUMDP_r_sum/t_trial)
#println(FIB_r_sum/t_trial)
#println(MY_3_r_sum/t_trial)
#println(MY_5_r_sum/t_trial)
#println(MY_7_r_sum/t_trial)
#println(MY_6_r_sum/t_trial)


gamma = 0.95
grid = 50

red = zeros(Float64,grid+1)
QMDP_r = zeros(Float64,grid+1)
UMDP_r = zeros(Float64,grid+1)
FIB_r = zeros(Float64,grid+1)


for reduced_time = 1 : grid+1

    println(reduced_time)

    redu_f = 1 + 0.1 * (reduced_time - 1)


    QMDP_alpha = Q_value_iteration(zeros(Float64,n_s,n_a),T,R,0.01,gamma/redu_f)
    QUMDP_alpha = QUMDP(zeros(Float64,n_s,n_a),T,R,0.01,gamma/redu_f)
    FIB_alpha = FIB(zeros(Float64,n_s,n_a),T,R,O,0.01,gamma/redu_f)

    #println(QMDP_alpha)

    QMDP_r_sum = 0
    QUMDP_r_sum = 0
    FIB_r_sum = 0

    t_trial = 2000
    t_step = 200

    for i = 1 : t_trial

        QMDP_r_sum += one_shuttle_trial(T,R,O,t_step,QMDP_alpha,gamma)
        QUMDP_r_sum += one_shuttle_trial(T,R,O,t_step,QUMDP_alpha,gamma)
        FIB_r_sum += one_shuttle_trial(T,R,O,t_step,FIB_alpha,gamma)

    end

    red[reduced_time] = redu_f
    QMDP_r[reduced_time] = QMDP_r_sum/t_trial
    UMDP_r[reduced_time] = QUMDP_r_sum/t_trial
    FIB_r[reduced_time] = FIB_r_sum/t_trial

end

plot(red,QMDP_r,label="QMDP")
plot(red,UMDP_r,label="UMDP")
plot(red,FIB_r,label="FIB")
xlabel("reduced factor")
ylabel("Discounted Reward")
title("Shuttle with gamma 0.95 and uniform initial belief")
legend(loc="upper right",fancybox="true")
annotate("SARSOP = 32.89",
	xy=[1;0],
	xycoords="axes fraction",
	xytext=[-10,10],
	textcoords="offset points",
	fontsize=12.0,
	ha="right",
	va="bottom")
