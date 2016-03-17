include("project_1.jl")
include("project_2.jl")

longti = 10; latit = 11;
#longti = 4; latit = 5;


T = zeros(Float64,longti * latit , 4 , longti * latit)
R = zeros(Float64,longti * latit , 4 , longti * latit)
O = zeros(Float64,longti * latit , 4 , longti * latit)

pf = 0.7

##### Define transition Matrix #####

#### corner case ####

#### up and left ####

T[1,1,2] = 0.7; T[1,1,1+longti] = 0.1; T[1,1,1] = 0.2;
T[1,2,1] = 0.8; T[1,2,2] = 0.1; T[1,2,1+longti] = 0.1;
T[1,4,1] = 0.8; T[1,4,2] = 0.1; T[1,4,1+longti] = 0.1;
T[1,3,1+longti] = 0.7; T[1,3,1] = 0.2; T[1,3,2] = 0.1;
#T[1,5,1] = 0.8; T[1,5,1+longti] = 0.1; T[1,5,2] = 0.1;

#### down and left ####
dl = longti * latit - longti + 1;

T[dl,1,dl+1] = 0.7; T[dl,1,dl - longti] = 0.1; T[dl,1,dl] = 0.2;
T[dl,2,dl - longti] = 0.7; T[dl,2,dl+1] = 0.1; T[dl,2,dl] = 0.2;
T[dl,4,dl] = 0.8; T[dl,4,dl-longti] = 0.1; T[dl,4,dl+1] = 0.1;
T[dl,3,dl] = 0.8; T[dl,3,dl+1] = 0.1; T[dl,3,dl-longti] = 0.1;
#T[1,5,1] = 0.8; T[1,5,1+longti] = 0.1; T[1,5,2] = 0.1;

#### up and right
T[longti,1,longti] = 1.0;
T[longti,2,longti] = 1.0;
T[longti,3,longti] = 1.0;
T[longti,4,longti] = 1.0;

#### up and right
T[longti*latit,1,longti*latit] = 1.0;
T[longti*latit,2,longti*latit] = 1.0;
T[longti*latit,3,longti*latit] = 1.0;
T[longti*latit,4,longti*latit] = 1.0;

##### left column #####
for i = 2 : latit-1
    pos_ind = (i - 1) * longti + 1;
    T[pos_ind,1,pos_ind + 1] = 0.7; T[pos_ind,1,pos_ind + longti] = 0.1; T[pos_ind,1,pos_ind - longti] = 0.1; T[pos_ind,1,pos_ind] = 0.1;
    T[pos_ind,2,pos_ind + 1] = 0.1; T[pos_ind,2,pos_ind + longti] = 0.1; T[pos_ind,2,pos_ind - longti] = 0.7; T[pos_ind,2,pos_ind] = 0.1;
    T[pos_ind,3,pos_ind + 1] = 0.1; T[pos_ind,3,pos_ind + longti] = 0.7; T[pos_ind,3,pos_ind - longti] = 0.1; T[pos_ind,3,pos_ind] = 0.1;
    T[pos_ind,4,pos_ind + 1] = 0.1; T[pos_ind,4,pos_ind + longti] = 0.1; T[pos_ind,4,pos_ind - longti] = 0.1; T[pos_ind,4,pos_ind] = 0.7;
end

#### right column #####
for i = 2 : latit - 1
    pos_ind = i * longti
    T[pos_ind,1,pos_ind] = 1.0; T[pos_ind,2,pos_ind] = 1.0; T[pos_ind,3,pos_ind] = 1.0; T[pos_ind,4,pos_ind] = 1.0;
end

#### top row #####
for i = 2 : longti - 1
    T[i,1,i+1] = 0.7; T[i,1,i+longti] = 0.1; T[i,1,i] = 0.1; T[i,1,i-1] = 0.1;
    T[i,2,i+1] = 0.1; T[i,2,i+longti] = 0.1; T[i,2,i] = 0.7; T[i,2,i-1] = 0.1;
    T[i,3,i+1] = 0.1; T[i,3,i+longti] = 0.7; T[i,3,i] = 0.1; T[i,3,i-1] = 0.1;
    T[i,4,i+1] = 0.1; T[i,4,i+longti] = 0.1; T[i,4,i] = 0.1; T[i,4,i-1] = 0.7;
end

#### bottom row ####
for i = (longti * (latit - 1) + 2) : ( longti*latit - 1 )
    T[i,1,i+1] = 0.7; T[i,1,i-longti] = 0.1; T[i,1,i] = 0.1; T[i,1,i-1] = 0.1;
    T[i,2,i+1] = 0.1; T[i,2,i-longti] = 0.7; T[i,2,i] = 0.1; T[i,2,i-1] = 0.1;
    T[i,3,i+1] = 0.1; T[i,3,i-longti] = 0.1; T[i,3,i] = 0.7; T[i,3,i-1] = 0.1;
    T[i,4,i+1] = 0.1; T[i,4,i-longti] = 0.1; T[i,4,i] = 0.1; T[i,4,i-1] = 0.7;
end

#### center area ####
for i = 1 : latit - 2
    for j = 1 : longti - 2
        pos_ind = i * longti + j + 1
        T[pos_ind,1,pos_ind+1] = 0.7; T[pos_ind,1,pos_ind-1] = 0.1; T[pos_ind,1,pos_ind-longti] = 0.1; T[pos_ind,1,pos_ind+longti] = 0.1;
        T[pos_ind,2,pos_ind+1] = 0.1; T[pos_ind,2,pos_ind-1] = 0.1; T[pos_ind,2,pos_ind-longti] = 0.7; T[pos_ind,2,pos_ind+longti] = 0.1;
        T[pos_ind,3,pos_ind+1] = 0.1; T[pos_ind,3,pos_ind-1] = 0.1; T[pos_ind,3,pos_ind-longti] = 0.1; T[pos_ind,3,pos_ind+longti] = 0.7;
        T[pos_ind,4,pos_ind+1] = 0.1; T[pos_ind,4,pos_ind-1] = 0.7; T[pos_ind,4,pos_ind-longti] = 0.1; T[pos_ind,4,pos_ind+longti] = 0.1;
    end
end


##### Define Observation Matrix#####


half_levels = Int((latit - 1) / 2)
for level = 1 : Int((latit - 1) / 2)
    for a = 1 : 4
        current_level = collect( (level-1)*longti + 1 : level*longti )
        prob_stay = 1 - (level - 1)/(half_levels)

        for i in current_level
            for j = 1 : longti * latit
                if j == i
                    O[i,a,j] = prob_stay
                else
                    O[i,a,j] = ( 1 - prob_stay )/(longti*latit-1)
                end
            end
        end

        current_level = collect((latit - level)*longti + 1 : (latit - level + 1)*longti)

        for i in current_level
            for j = 1 : longti * latit
                if j == i
                    O[i,a,j] = prob_stay
                else
                    O[i,a,j] = (1 - prob_stay)/(longti*latit-1)
                end
            end
        end
    end
end

center_level = collect( half_levels *longti + 1 : (half_levels+1) * longti )

for i in center_level
    for a = 1 : 4
        O[i,a,:] = ones(Float64,longti*latit) * 1 / (longti*latit)
    end
end

O = zeros(Float64,longti * latit , 4 , longti * latit)

for i = 1 : longti * latit
    for a = 1 : 4
        if i in collect(1:1:longti)
            O[i,a,i] = 1
        elseif i in collect(longti*(latit-1)+1:1:longti*latit)
            O[i,a,i] = 1
        else
            for j = 1 : longti * latit
                O[i,a,j] = 1/(longti * latit)
            end
        end
    end
end

##### Reward ######

hit_rock = -10
hit_destination = 2

#for i = 1 : longti * latit
#    for a = 1 : 4
#        if i % longti == 0
#            for j = 1 : longti * latit
#                R[i,a,j] = hit_destination
#            end
#        end
#
#        if i % longti == longti - 1
#            if ((div(i,longti)) % 2) == 1
#                for j = 1 : longti * latit
#                    R[i,a,j] = hit_rock
#                end
#            end
#        end
#    end
#end

for i = 1 : latit
    pos_ind = i* longti - 1
    for a = 1 : 4
        R[pos_ind,a,pos_ind+1] = hit_destination
    end
end

for i = 1 : longti * latit
    for j = 2 : 2 : latit - 1
        rock_pos = j * longti - 1
        if (i != rock_pos) & (i % longti != 0)
        for a = 1 : 4
            R[i,a,rock_pos] = hit_rock
        end
        end
    end
end

#for i = 1 : longti * latit
#    for a = 1 : 4
#        for j = 1 : longti * latit
#            R[i,a,j] -= 1
#        end
#    end
#end








##### Initial Distribution #####

b_initial = zeros(Float64,1,longti * latit)

center = longti * ( Int((latit + 1 )/ 2) - 1 ) + 1

b_initial[1,center] = 1/9
b_initial[1,center - longti] = 1/9
b_initial[1,center - 2 * longti] = 1/9
b_initial[1,center + longti] = 1/9
b_initial[1,center + 2 * longti] = 1/9
b_initial[1,center + 1] = 1/9
b_initial[1,center - longti + 1] = 1/9
b_initial[1,center + longti + 1] = 1/9
b_initial[1,center + 2] = 1/9



##### Try some updates #####

##### Do sequence ######
## well for -10 and 2 and no other reward
b_test_1 = zeros(Float64,1,longti * latit)
b_test_1[1,1 * longti+1 : 1 * longti + 4] = (1/16) * ones(Float64,1,4)
b_test_1[1,2 * longti+1 : 2 * longti + 4] = (1/16) * ones(Float64,1,4)
b_test_1[1,3 * longti+1 : 3 * longti + 4] = (1/16) * ones(Float64,1,4)
b_test_1[1,4 * longti+1 : 4 * longti + 4] = (1/16) * ones(Float64,1,4)

b_test_6 = zeros(Float64,1,longti * latit)
b_test_1[1,1 * longti+4 : 1 * longti + 6] = (1/9) * ones(Float64,1,3)
b_test_1[1,2 * longti+4 : 2 * longti + 6] = (1/9) * ones(Float64,1,3)
b_test_1[1,3 * longti+4 : 3 * longti + 6] = (1/9) * ones(Float64,1,3)


b_test_2 = zeros(Float64,1,longti * latit)
b_test_2[1,1 * longti+5] = 1/4
b_test_2[1,2 * longti+5] = 1/4
b_test_2[1,3 * longti+5] = 1/4
b_test_2[1,4 * longti+5] = 1/4

b_test_3 = zeros(Float64,1,longti * latit)
b_test_3[1,4 * longti+6] = 1

b_test_4 = zeros(Float64,1,longti * latit)
st = 1; en = st + 4;
b_test_4[1,1 * longti+st : 1 * longti + en] = (1/25) * ones(Float64,1,5)
b_test_4[1,2 * longti+st : 2 * longti + en] = (1/25) * ones(Float64,1,5)
b_test_4[1,3 * longti+st : 3 * longti + en] = (1/25) * ones(Float64,1,5)
b_test_4[1,4 * longti+st : 4 * longti + en] = (1/25) * ones(Float64,1,5)
b_test_4[1,5 * longti+st : 5 * longti + en] = (1/25) * ones(Float64,1,5)

b_test_5 = zeros(Float64,1,longti * latit)
width = 3; stp = (2,2)
b_test_5[1,(stp[1]-1) * longti + stp[2] : (stp[1]-1) * longti + stp[2] + width] = (1/(4*(width+1))) * ones(Float64,1,width+1)
b_test_5[1,(stp[1]+0) * longti + stp[2] : (stp[1]+0) * longti + stp[2] + width] = (1/(4*(width+1))) * ones(Float64,1,width+1)
b_test_5[1,(stp[1]+1) * longti + stp[2] : (stp[1]+1) * longti + stp[2] + width] = (1/(4*(width+1))) * ones(Float64,1,width+1)
b_test_5[1,(stp[1]+2) * longti + stp[2] : (stp[1]+2) * longti + stp[2] + width] = (1/(4*(width+1))) * ones(Float64,1,width+1)

N_sample = 5; dd = 3 * longti #Int(longti*latit/2);
b_test_ramdom_set = zeros(Float64,N_sample,dd)
for t = 1 : N_sample
    total = 1
    for d = 1 : dd - 1
        b_test_ramdom_set[t,d] = total * rand()
        total =  total - b_test_ramdom_set[t,d]
    end
    b_test_ramdom_set[t,dd] = total
end

b_test_ramdom_set_full = zeros(Float64,N_sample,longti*latit)
b_test_ramdom_set_full[:,1:dd] = b_test_ramdom_set

#######################################################################

Q1 = Q_value_iteration(zeros(Float64,longti*latit,4),T,R,0.01,0.9)
Q2 = FIB(zeros(Float64,longti*latit,4),T,R,O,0.01,0.9)
QM1 = purely_iteration(zeros(Float64,longti*latit,4),T,R,O,0.01,0.9)
QM2 = purely_iteration_v2(zeros(Float64,longti*latit,4),T,R,O,0.01,0.9)
QM3 = purely_iteration_v3(zeros(Float64,longti*latit,4),T,R,O,0.01,0.9)
