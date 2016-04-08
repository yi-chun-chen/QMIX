using POMDPs


type TigerPOMDP <: POMDP
    r_listen::Float64 # reward for listening (default -1)
    r_findtiger::Float64 # reward for finding the tiger (default -100)
    r_escapetiger::Float64 # reward for escaping (default 10)
    p_listen_correctly::Float64 # prob of correctly listening (default 0.85)
    discount_factor::Float64 # discount
end
# default constructor
function TigerPOMDP()
    return TigerPOMDP(-1.0, -100.0, 10.0, 0.85, 0.95)
end

pomdp = TigerPOMDP()

type TigerState <: State
    tigerleft::Bool
end
# initialization function
POMDPs.create_state(::TigerPOMDP) = TigerState(true);

type TigerAction <: Action
    act::Symbol
end
# initialization function
POMDPs.create_action(::TigerPOMDP) = TigerAction(:listen);

a = TigerAction(:listen)

type TigerObservation <: Observation
    obsleft::Bool
end
# initialization function
POMDPs.create_observation(::TigerPOMDP) = TigerObservation(true);

type TigerStateSpace <: AbstractSpace
    states::Vector{TigerState}
end

POMDPs.states(::TigerPOMDP) = TigerStateSpace([TigerState(true), TigerState(false)])
POMDPs.iterator(space::TigerStateSpace) = space.states;
POMDPs.index(::TigerPOMDP, s::TigerState) = (Int64(s.tigerleft) + 1)

type TigerActionSpace <: AbstractSpace
    actions::Vector{TigerAction}
end
# define actions function
POMDPs.actions(::TigerPOMDP) = TigerActionSpace([TigerAction(:openl), TigerAction(:openr), TigerAction(:listen)]); # default
POMDPs.actions(::TigerPOMDP, ::TigerState, acts::TigerActionSpace) = acts; # convenience (actions do not change in different states)
# define iterator function
POMDPs.iterator(space::TigerActionSpace) = space.actions;

type TigerObservationSpace <: AbstractSpace
    obs::Vector{TigerObservation}
end
# function returning observation space
POMDPs.observations(::TigerPOMDP) = TigerObservationSpace([TigerObservation(true), TigerObservation(false)]);
POMDPs.observations(::TigerPOMDP, s::TigerState, obs::TigerObservationSpace) = obs;
# function returning an iterator over that space
POMDPs.iterator(space::TigerObservationSpace) = space.obs;

# transition distribution type
type TigerTransitionDistribution <: AbstractDistribution
    probs::Vector{Float64}
end
# transition distribution initializer
POMDPs.create_transition_distribution(::TigerPOMDP) = TigerTransitionDistribution([0.5, 0.5])

# observation distribution type
type TigerObservationDistribution <: AbstractDistribution
    probs::Vector{Float64}
end
# observation distribution initializer
POMDPs.create_observation_distribution(::TigerPOMDP) = TigerObservationDistribution([0.5, 0.5]);

# transition pdf
function POMDPs.pdf(d::TigerTransitionDistribution, s::TigerState)
    s.tigerleft ? (return d.probs[1]) : (return d.probs[2])
end;
# obsevation pdf
function POMDPs.pdf(d::TigerObservationDistribution, o::TigerObservation)
    o.obsleft ? (return d.probs[1]) : (return d.probs[2])
end;

using POMDPDistributions

# sample from transition distribution
function POMDPs.rand(rng::AbstractRNG, d::TigerTransitionDistribution, s::TigerState)
    # we use a categorical distribution, and this will usually be enough for a discrete problem
    c = Categorical(d.probs) # this comes from POMDPDistributions
    # sample an integer from c
    sp = rand(rng, c) # this function is also from POMDPDistributions
    # if sp is 1 then tiger is on the left
    sp == 1 ? (s.tigerleft=true) : (s.tigerleft=false)
    return s
end

# similar for smapling from the observation distribution
function POMDPs.rand(rng::AbstractRNG, d::TigerObservationDistribution, o::TigerObservation)
    c = Categorical(d.probs)
    op = rand(rng, c)
    # if op is 1 then we hear tiger on the left
    op == 1 ? (o.obsleft=true) : (o.obsleft=false)
    return o
end;

# the transition mode
function POMDPs.transition(pomdp::TigerPOMDP, s::TigerState, a::TigerAction, d::TigerTransitionDistribution=create_transition_distribution(pomdp))
    probs = d.probs
    # if open a door reset the tiger probs
    if a.act == :openl || a.act == :openr
        fill!(probs, 0.5)
    # if tiger is on the left, distribution = 1.0 over the first state
    elseif s.tigerleft
        probs[1] = 1.0
        probs[2] = 0.0
    # otherwise distribution = 1.0 over the second state
    else
        probs[1] = 0.0
        probs[2] = 1.0
    end
    d
end;

function POMDPs.reward(pomdp::TigerPOMDP, s::TigerState, a::TigerAction)
    r = 0.0
    # small penalty for listening
    if a.act == :listen
        r += pomdp.r_listen
    end
    if a.act == :openl
        # find tiger behind left door
        if s.tigerleft
            r += pomdp.r_findtiger
        # escape through left door
        else
            r += pomdp.r_escapetiger
        end
    end
    if a.act == :openr
        # escape through the right door
        if s.tigerleft
            r += pomdp.r_escapetiger
            # find tiger behind the right door
        else
            r += pomdp.r_findtiger
        end
    end
    return r
end;
# to match the interface
POMDPs.reward(pomdp::TigerPOMDP, s::TigerState, a::TigerAction, sp::TigerState) = reward(pomdp, s, a)

function POMDPs.observation(pomdp::TigerPOMDP, s::TigerState, a::TigerAction, d::TigerObservationDistribution=create_observation_distribution(pomdp))
    probs = d.probs
    p = pomdp.p_listen_correctly # probability of listening correctly
    if a.act == :listen
        # if tiger is behind left door
        if s.tigerleft
            probs[1] = p # correct prob
            probs[2] = (1.0-p) # wring prob
        # if tiger is behind right door
        else
            probs[1] = (1.0-p) # wrong prob
            probs[2] = p # correct prob
        end
    # if don't listen uniform
    else
        fill!(probs, 0.5)
    end
    d
end;

POMDPs.discount(pomdp::TigerPOMDP) = pomdp.discount_factor
POMDPs.n_states(::TigerPOMDP) = 2
POMDPs.n_actions(::TigerPOMDP) = 3
POMDPs.n_observations(::TigerPOMDP) = 2;

# we will use the POMDPToolbox module
using POMDPToolbox

# define a initialization function
POMDPs.create_belief(::TigerPOMDP) = DiscreteBelief(2) # the belief is over our two states
# initial belief is same as create
POMDPs.initial_belief(::TigerPOMDP) = DiscreteBelief(2);


##########################################
############## SARSOP ####################
##########################################

using QMDP

# initialize the solver
# key-word args are the maximum number of iterations the solver will run for, and the Bellman tolerance
solver = QMDPSolver(max_iterations=1, tolerance=1e-3)

# initialize the QMDP policy
qmdp_policy = create_policy(solver, pomdp)

# run the solver
solve(solver, pomdp, qmdp_policy, verbose=true)
qmdp_policy.alphas = [[-100 400 300];[300 -10 100]]

s = create_state(pomdp)
o = create_observation(pomdp)

b = initial_belief(pomdp)

updater = DiscreteUpdater(pomdp) # this comes from POMDPToolbox

#rng = MersenneTwister(9) # initialize a random number generator

# lets run the simulation for ten time steps
r_tot = 0.0
for j = 1 : 1000
    rtot = 0.0
for i = 1:1000
    # get the action from our SARSOP policy
    a = action(qmdp_policy, b) # the QMDP action function returns the POMDP action not its index like the SARSOP action function
    # compute the reward
    r = reward(pomdp, s, a)
    rtot += r

    #println("Time step $i")
    #println("Have belief: $(b.b), taking action: $(a), got reward: $(r)")

    # transition the system state
    trans_dist = transition(pomdp, s, a)
    rand(rng, trans_dist, s)

    # sample a new observation
    obs_dist = observation(pomdp, s, a)
    rand(rng, obs_dist, o)

    # update the belief
    b = update(updater, b, a, o)

    #println("Saw observation: $(o), new belief: $(b.b)\n")

end
r_tot += rtot
end
println(r_tot/1000)
