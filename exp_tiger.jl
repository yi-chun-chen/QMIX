include("method_existing.jl")
include("method_QMIX.jl")

n_s = 2
n_a = 3
n_o = 2

# s1 = left, s2 = right
# o1 = left, o2 = right
# a1 = listen, a2 = open left, a3 = open right
T = Array(Float64,2,3,2)
T[1,1,1] = 1.0; T[1,1,2] = 0.0;
T[1,2,1] = 0.5; T[1,2,2] = 0.5;
T[1,3,1] = 0.5; T[1,3,2] = 0.5;
T[2,1,1] = 0.0; T[2,1,2] = 1.0;
T[2,2,1] = 0.5; T[2,2,2] = 0.5;
T[2,3,1] = 0.5; T[2,3,2] = 0.5;

R = Array(Float64,2,3,2)
R[1,1,1] = -1; R[1,1,2] = -1;
R[1,2,1] = -100; R[1,2,2] = -100;
R[1,3,1] = 10; R[1,3,2] = 10;
R[2,1,1] = -1; R[2,1,2] = -1;
R[2,2,1] = 10; R[2,2,2] = 10;
R[2,3,1] = -100; R[2,3,2] = -100;

O = Array(Float64,2,3,2)
O[1,1,1] = 0.85; O[1,1,2] = 0.15;
O[1,2,1] = 0.5 ; O[1,2,2] = 0.5;
O[1,3,1] = 0.5 ; O[1,3,2] = 0.5;
O[2,1,1] = 0.15; O[2,1,2] = 0.85;
O[2,2,1] = 0.5 ; O[2,2,2] = 0.5;
O[2,3,1] = 0.5 ; O[2,3,2] = 0.5;


function one_tiger_trial(
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
    b[1] = 0.5
    b[2] = 0.5
    b_p = deepcopy(b)

    # Initialize state
    x = sample_from_vector(b)

    # intialize total reward
    total_r = 0.0
    for t in 1 : t_step

        # Choose the action
        action_to_do = action_to_take(b,alpha)
        #println("action_to_do")

        # Get reward and the next state
        (xp,r) = tran_reward_sampling(T,R,x,action_to_do)
        xp = tran_sampling(T, x, action_to_do)
        r = R[x,action_to_do,xp]
        total_r += r #*  ((gamma)^(t))

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

gamma_simulation = 1.0
grid = 2

gamma_p = collect(linspace(gamma_simulation, 0.5, grid))
QMDP_r = zeros(Float64,grid)
UMDP_r = zeros(Float64,grid)
FIB_r = zeros(Float64,grid)
#FIB_alpha = FIB(zeros(Float64,n_s,n_a),T,R,O,0.01,0.95)

for (reduced_time, gamma) in enumerate(gamma_p)

    println(reduced_time)

    #redu_f = 1 + 0.1 * (reduced_time - 1)

    #QMDP_alpha = Q_value_iteration(zeros(Float64,n_s,n_a),T,R,0.01,gamma)
    #QUMDP_alpha = QUMDP(zeros(Float64,n_s,n_a),T,R,0.01,gamma)
    #FIB_alpha = FIB(zeros(Float64,n_s,n_a),T,R,O,0.01,gamma)
    FIB_alpha = [[300.0  -100.0  400.0];[100.0 300.0 -10.0]]

    QMDP_r_sum = 0.0
    QUMDP_r_sum = 0.0
    FIB_r_sum = 0.0

    t_trial = 100
    t_step = 1000

    for i = 1 : t_trial

        #QMDP_r_sum += one_tiger_trial(T,R,O,t_step,QMDP_alpha,gamma_simulation)
        #QUMDP_r_sum += one_tiger_trial(T,R,O,t_step,QUMDP_alpha,gamma_simulation)
        FIB_r_sum += one_tiger_trial(T,R,O,t_step,FIB_alpha,1.0)

    end

    QMDP_r[reduced_time] = QMDP_r_sum/t_trial
    UMDP_r[reduced_time] = QUMDP_r_sum/t_trial
    FIB_r[reduced_time] = FIB_r_sum/t_trial

end
println(FIB_r[1])

#plot(gamma_p,QMDP_r,label="QMDP")
#plot(gamma_p,UMDP_r,label="UMDP")
#plot(gamma_p,FIB_r,label="FIB")
#xlabel("discount factor")
#ylabel("Reward")
#title("Mini-Hallway with gamma 0.9 and uniform initial belief")
#legend(loc="upper right",fancybox="true")