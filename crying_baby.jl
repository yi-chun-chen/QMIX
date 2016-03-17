using POMDPs

type BabyPOMDP <: POMDP
    r_feed::Float64
    r_hungry::Float64
    p_become_hungry::Float64
    p_cry_when_hungry::Float64
    p_cry_when_not_hungry::Float64
    discount::Float64
end

function BabyPOMDP()
    return BabyPOMDP(-5, -10, 0.1, 0.8, 0.1, 0.9)
end

pomdp = BabyPOMDP()

type BabyState <: State
    hungry::Bool
end

POMDPs.create_state(::BabyPOMDP) = BabyState(true);

type BabyAction <: Action
    feed::Bool
end
# initialization function
POMDPs.create_action(::BabyPOMDP) = BabyAction(true);

####a = TigerAction(:listen)

type BabyObservation <: Observation
    crying::Bool
end
# initialization function
POMDPs.create_observation(::BabyPOMDP) = BabyObservation(true);

type BabyStateSpace <: AbstractSpace
    states::Vector{BabyState}
end

POMDPs.states(::BabyPOMDP) = BabyStateSpace([BabyState(true), BabyState(false)])
POMDPs.iterator(space::BabyStateSpace) = space.states;
POMDPs.index(::BabyPOMDP, s::BabyState) = (Int64(s.hungry) + 1)

type BabyActionSpace <: AbstractSpace
    actions::Vector{BabyAction}
end
# define actions function
POMDPs.actions(::BabyPOMDP) = BabyActionSpace([BabyAction(true), BabyAction(false)]);
POMDPs.actions(::BabyPOMDP, ::BabyState, acts::BabyActionSpace) = acts; # convenience (actions do not change in different states)
# define iterator function
POMDPs.iterator(space::BabyActionSpace) = space.actions;

type BabyObservationSpace <: AbstractSpace
    obs::Vector{BabyObservation}
end
# function returning observation space
POMDPs.observations(::BabyPOMDP) = BabyObservationSpace([BabyObservation(true), BabyObservation(false)]);
POMDPs.observations(::BabyPOMDP, s::BabyState, obs::BabyObservationSpace) = obs;
# function returning an iterator over that space
POMDPs.iterator(space::BabyObservationSpace) = space.obs;

# transition distribution type
type BabyTransitionDistribution <: AbstractDistribution
    probs::Vector{Float64}
end
# transition distribution initializer
POMDPs.create_transition_distribution(::BabyPOMDP) = BabyTransitionDistribution([0.5, 0.5])

# observation distribution type
type BabyObservationDistribution <: AbstractDistribution
    probs::Vector{Float64}
end
# observation distribution initializer
POMDPs.create_observation_distribution(::BabyPOMDP) = BabyObservationDistribution([0.5, 0.5]);

# transition pdf
function POMDPs.pdf(d::BabyTransitionDistribution, s::BabyState)
    s.hungry ? (return d.probs[1]) : (return d.probs[2])
end;
# obsevation pdf
function POMDPs.pdf(d::BabyObservationDistribution, o::BabyObservation)
    o.crying ? (return d.probs[1]) : (return d.probs[2])
end;

using POMDPDistributions

# sample from transition distribution
function POMDPs.rand(rng::AbstractRNG, d::BabyTransitionDistribution, s::BabyState)
    # we use a categorical distribution, and this will usually be enough for a discrete problem
    c = Categorical(d.probs) # this comes from POMDPDistributions
    # sample an integer from c
    sp = rand(rng, c) # this function is also from POMDPDistributions
    # if sp is 1 then tiger is on the left
    sp == 1 ? (s.hungry=true) : (s.hungry=false)
    return s
end

# similar for smapling from the observation distribution
function POMDPs.rand(rng::AbstractRNG, d::BabyObservationDistribution, o::BabyObservation)
    c = Categorical(d.probs)
    op = rand(rng, c)
    # if op is 1 then we hear tiger on the left
    op == 1 ? (o.crying=true) : (o.crying=false)
    return o
end;

# the transition mode
function POMDPs.transition(pomdp::BabyPOMDP, s::BabyState, a::BabyAction, d::BabyTransitionDistribution=create_transition_distribution(pomdp))
    probs = d.probs
    if s.hungry & a.feed
        probs[1] = 0.0
        probs[2] = 1.0
    elseif s.hungry & !a.feed
        probs[1] = 1.0
        probs[2] = 0.0
    elseif !s.hungry & !a.feed
        probs[1] = 0.1
        probs[2] = 0.9
    else
        probs[1] = 0.0
        probs[2] = 1.0
    end
    d
end;


function POMDPs.reward(pomdp::BabyPOMDP, s::BabyState, a::BabyAction)
    r = 0.0

    if a.feed
        r += -5
    end

    if s.hungry
        r += -10
    end
    return r
end;

# to match the interface
POMDPs.reward(pomdp::BabyPOMDP, s::BabyState, a::BabyAction, sp::BabyState) = reward(pomdp, s, a)

function POMDPs.observation(pomdp::BabyPOMDP, s::BabyState, a::BabyAction, d::BabyObservationDistribution=create_observation_distribution(pomdp))
    probs = d.probs

    if s.hungry
        probs[1] = 0.8
        probs[2] = 0.2
    else
        probs[1] = 0.1
        probs[2] = 0.9
    end
    d
end;


POMDPs.discount(pomdp::BabyPOMDP) = pomdp.discount
POMDPs.n_states(::BabyPOMDP) = 2
POMDPs.n_actions(::BabyPOMDP) = 2
POMDPs.n_observations(::BabyPOMDP) = 2;

# we will use the POMDPToolbox module
using POMDPToolbox

# define a initialization function
POMDPs.create_belief(::BabyPOMDP) = DiscreteBelief(2) # the belief is over our two states
# initial belief is same as create
POMDPs.initial_belief(::BabyPOMDP) = DiscreteBelief(2);

using QMDP

solver = QMDPSolver(max_iterations=100, tolerance=1e-3)
qmdp_policy = create_policy(solver, pomdp)
solve(solver, pomdp, qmdp_policy, verbose=false)



#s = create_state(pomdp)
#o = create_observation(pomdp)
#b = initial_belief(pomdp)
#ppp = 0.8
#b = POMDPToolbox.DiscreteBelief([ppp,1-ppp],[0.5,0.5],2,true)
#updater = DiscreteUpdater(pomdp) # this comes from POMDPToolbox
#rng = MersenneTwister(9) # initialize a random number generator

#rtot = 0.0
# lets run the simulation for ten time steps
#for i = 1:5
#    # get the action from our SARSOP policy
#    a = action(qmdp_policy, b) # the QMDP action function returns the POMDP action not its index like the SARSOP action function
#    # compute the reward
#    r = reward(pomdp, s, a)
#    rtot += r
#
#    println("Time step $i")
#    println("Have belief: $(b.b), taking action: $(a), got reward: $(r)")
#
#    # transition the system state
#    trans_dist = transition(pomdp, s, a)
#    rand(rng, trans_dist, s)
#
#    # sample a new observation
#    obs_dist = observation(pomdp, s, a)
#    rand(rng, obs_dist, o)
#
#    # update the belief
#    b = update(updater, b, a, o)
#
#    println("Saw observation: $(o), new belief: $(b.b)\n")
#
#end
#println("Total reward: $rtot")
