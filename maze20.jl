include("project_1.jl")
include("project_2.jl")

# The following matrix and data for maze20 probelm are from Tony's POMDP website:
# www.pomdp.org

n_s = 20; n_a = 6; n_o = 8;

T = zeros(Float64,n_s,n_a,n_s)
O = zeros(Float64,n_s,n_a,n_o)
R = zeros(Float64,n_s,n_a,n_s)

Ta0 = [
    [0.15 0.15 0 0 0 0.7 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ]
    ;[0.15 0.7 0.15 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ]
    ;[0 0.15 0.7 0.15 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ]
    ;[0 0 0.15 0 0.15 0 0 0 0.7 0 0 0 0 0 0 0 0 0 0 0 ]
    ;[0 0 0 0.15 0.85 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ]
    ;[0 0 0 0 0 0.15 0.15 0 0 0 0.7 0 0 0 0 0 0 0 0 0 ]
    ;[0 0 0 0 0 0.15 0.15 0 0 0 0 0.7 0 0 0 0 0 0 0 0 ]
    ;[0.3 0 0 0 0.3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.4 ]
    ;[0 0 0 0 0 0 0 0 0.85 0.15 0 0 0 0 0 0 0 0 0 0 ]
    ;[0 0 0 0 0 0 0 0 0.15 0.15 0 0 0 0 0.7 0 0 0 0 0 ]
    ;[0 0 0 0 0 0 0 0 0 0 0.15 0.15 0 0 0 0.7 0 0 0 0 ]
    ;[0 0 0 0 0 0 0 0 0 0 0.15 0.85 0 0 0 0 0 0 0 0 ]
    ;[0 0 0 0 0 0 0 0 0 0 0 0 0.3 0 0 0 0 0.7 0 0 ]
    ;[0 0 0 0 0 0 0 0 0 0 0 0 0 0.15 0.15 0 0 0 0.7 0 ]
    ;[0 0 0 0 0 0 0 0 0 0 0 0 0 0.15 0.85 0 0 0 0 0 ]
    ;[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.85 0.15 0 0 0 ]
    ;[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.15 0.7 0.15 0 0 ]
    ;[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.15 0.7 0.15 0 ]
    ;[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.15 0.7 0.15 ]
    ;[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.15 0.85 ]
];

Ta1 = [
    [0.85 0.15 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ]
    ;[0.15 0.7 0.15 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ]
    ;[0 0.15 0.7 0.15 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ]
    ;[0 0 0.15 0.7 0.15 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ]
    ;[0 0 0 0.15 0.85 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ]
    ;[0.7 0 0 0 0 0.15 0.15 0 0 0 0 0 0 0 0 0 0 0 0 0 ]
    ;[0 0 0 0 0 0.15 0.85 0 0 0 0 0 0 0 0 0 0 0 0 0 ]
    ;[0.3 0 0 0 0.3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.4 ]
    ;[0 0 0 0.7 0 0 0 0 0.15 0.15 0 0 0 0 0 0 0 0 0 0 ]
    ;[0 0 0 0 0 0 0 0 0.15 0.85 0 0 0 0 0 0 0 0 0 0 ]
    ;[0 0 0 0 0 0.7 0 0 0 0 0.15 0.15 0 0 0 0 0 0 0 0 ]
    ;[0 0 0 0 0 0 0.7 0 0 0 0.15 0.15 0 0 0 0 0 0 0 0 ]
    ;[0 0 0 0 0 0 0 0.7 0 0 0 0 0.3 0 0 0 0 0 0 0 ]
    ;[0 0 0 0 0 0 0 0 0 0 0 0 0 0.85 0.15 0 0 0 0 0 ]
    ;[0 0 0 0 0 0 0 0 0 0.7 0 0 0 0.15 0.15 0 0 0 0 0 ]
    ;[0 0 0 0 0 0 0 0 0 0 0.7 0 0 0 0 0.15 0.15 0 0 0 ]
    ;[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.15 0.7 0.15 0 0 ]
    ;[0 0 0 0 0 0 0 0 0 0 0 0 0.7 0 0 0 0.15 0 0.15 0 ]
    ;[0 0 0 0 0 0 0 0 0 0 0 0 0 0.7 0 0 0 0.15 0 0.15 ]
    ;[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.15 0.85 ]
];


Ta2 = [
    [0.15 0.7 0 0 0 0.15 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ]
    ;[0 0.3 0.7 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ]
    ;[0 0 0.3 0.7 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ]
    ;[0 0 0 0.15 0.7 0 0 0 0.15 0 0 0 0 0 0 0 0 0 0 0 ]
    ;[0 0 0 0 1.0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ]
    ;[0.15 0 0 0 0 0 0.7 0 0 0 0.15 0 0 0 0 0 0 0 0 0]
    ;[0 0 0 0 0 0 0.85 0 0 0 0 0.15 0 0 0 0 0 0 0 0 ]
    ;[0.3 0 0 0 0.3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.4 ]
    ;[0 0 0 0.15 0 0 0 0 0.15 0.7 0 0 0 0 0 0 0 0 0 0 ]
    ;[0 0 0 0 0 0 0 0 0 0.85 0 0 0 0 0.15 0 0 0 0 0 ]
    ;[0 0 0 0 0 0.15 0 0 0 0 0 0.7 0 0 0 0.15 0 0 0 0 ]
    ;[0 0 0 0 0 0 0.15 0 0 0 0 0.85 0 0 0 0 0 0 0 0 ]
    ;[0 0 0 0 0 0 0 0.15 0 0 0 0 0.7 0 0 0 0 0.15 0 0 ]
    ;[0 0 0 0 0 0 0 0 0 0 0 0 0 0.15 0.7 0 0 0 0.15 0 ]
    ;[0 0 0 0 0 0 0 0 0 0.15 0 0 0 0 0.85 0 0 0 0 0 ]
    ;[0 0 0 0 0 0 0 0 0 0 0.15 0 0 0 0 0.15 0.7 0 0 0 ]
    ;[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.3 0.7 0 0 ]
    ;[0 0 0 0 0 0 0 0 0 0 0 0 0.15 0 0 0 0 0.15 0.7 0 ]
    ;[0 0 0 0 0 0 0 0 0 0 0 0 0 0.15 0 0 0 0 0.15 0.7 ]
    ;[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1.0 ]
];

Ta3 = [
    [0.85 0 0 0 0 0.15 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ]
    ;[0.7 0.3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ]
    ;[0 0.7 0.3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ]
    ;[0 0 0.7 0.15 0 0 0 0 0.15 0 0 0 0 0 0 0 0 0 0 0 ]
    ;[0 0 0 0.7 0.3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ]
    ;[0.15 0 0 0 0 0.7 0 0 0 0 0.15 0 0 0 0 0 0 0 0 0 ]
    ;[0 0 0 0 0 0.7 0.15 0 0 0 0 0.15 0 0 0 0 0 0 0 0 ]
    ;[0.3 0 0 0 0.3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.4 ]
    ;[0 0 0 0.15 0 0 0 0 0.85 0 0 0 0 0 0 0 0 0 0 0 ]
    ;[0 0 0 0 0 0 0 0 0.7 0.15 0 0 0 0 0.15 0 0 0 0 0 ]
    ;[0 0 0 0 0 0.15 0 0 0 0 0.7 0 0 0 0 0.15 0 0 0 0 ]
    ;[0 0 0 0 0 0 0.15 0 0 0 0.7 0.15 0 0 0 0 0 0 0 0 ]
    ;[0 0 0 0 0 0 0 0.15 0 0 0 0 0.7 0 0 0 0 0.15 0 0 ]
    ;[0 0 0 0 0 0 0 0 0 0 0 0 0 0.85 0 0 0 0 0.15 0 ]
    ;[0 0 0 0 0 0 0 0 0 0.15 0 0 0 0.7 0.15 0 0 0 0 0 ]
    ;[0 0 0 0 0 0 0 0 0 0 0.15 0 0 0 0 0.85 0 0 0 0 ]
    ;[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.7 0.3 0 0 0 ]
    ;[0 0 0 0 0 0 0 0 0 0 0 0 0.15 0 0 0 0.7 0.15 0 0 ]
    ;[0 0 0 0 0 0 0 0 0 0 0 0 0 0.15 0 0 0 0.7 0.15 0 ]
    ;[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.7 0.3   ]
];

Ta4 = [
    [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ]
    ;[0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ]
    ;[0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ]
    ;[0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ]
    ;[0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ]
    ;[0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ]
    ;[0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 ]
    ;[0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 ]
    ;[0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 ]
    ;[0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 ]
    ;[0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 ]
    ;[0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 ]
    ;[0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 ]
    ;[0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 ]
    ;[0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 ]
    ;[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 ]
    ;[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 ]
    ;[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 ]
    ;[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 ]
    ;[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 ]
];

Ta5 =[
     [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ]
    ;[0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ]
    ;[0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ]
    ;[0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ]
    ;[0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ]
    ;[0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ]
    ;[0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 ]
    ;[0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 ]
    ;[0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 ]
    ;[0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 ]
    ;[0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 ]
    ;[0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 ]
    ;[0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 ]
    ;[0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 ]
    ;[0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 ]
    ;[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 ]
    ;[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 ]
    ;[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 ]
    ;[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 ]
    ;[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]];

for i = 1 : n_s
    for j = 1 : n_s
        T[i,1,j] = Ta0[i,j]
        T[i,2,j] = Ta1[i,j]
        T[i,3,j] = Ta2[i,j]
        T[i,4,j] = Ta3[i,j]
        T[i,5,j] = Ta4[i,j]
        T[i,6,j] = Ta5[i,j]
    end
end

Oa0 = [
    [1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
];

Oa1 = [
    [1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]];

Oa2 = [
    [1 0 0 0 0 0 0 0]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
];

Oa3 = [
    [1 0 0 0 0 0 0 0]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0 ]
    ;[1 0 0 0 0 0 0 0]
];

Oa4 = [
    [0 0.14 0.01 0.8 0.05 0 0 0 ]
    ;[0 0.05 0.1 0.1 0.75 0 0 0 ]
    ;[0 0.05 0.1 0.1 0.75 0 0 0 ]
    ;[0 0.14 0.01 0.8 0.05 0 0 0 ]
    ;[0 0.05 0.1 0.1 0.75 0 0 0 ]
    ;[0 0.89 0.05 0.05 0.01 0 0 0 ]
    ;[0 0.14 0.01 0.8 0.05 0 0 0 ]
    ;[0 0.14 0.01 0.8 0.05 0 0 0 ]
    ;[0 0.14 0.8 0.01 0.05 0 0 0 ]
    ;[0 0.14 0.01 0.8 0.05 0 0 0 ]
    ;[0 0.89 0.05 0.05 0.01 0 0 0 ]
    ;[0 0.14 0.8 0.01 0.05 0 0 0 ]
    ;[0 0.89 0.05 0.05 0.01 0 0 0 ]
    ;[0 0.14 0.01 0.8 0.05 0 0 0 ]
    ;[0 0.14 0.8 0.01 0.05 0 0 0 ]
    ;[0 0.14 0.8 0.01 0.05 0 0 0 ]
    ;[0 0.05 0.1 0.1 0.75 0 0 0 ]
    ;[0 0.14 0.8 0.01 0.05 0 0 0 ]
    ;[0 0.14 0.8 0.01 0.05 0 0 0 ]
    ;[0 0.05 0.1 0.1 0.75 0 0 0]
];

Oa5 = [
    [0 0.14 0 0 0 0.01 0.8 0.05 ]
    ;[0 0.89 0 0 0 0.05 0.05 0.01 ]
    ;[0 0.89 0 0 0 0.05 0.05 0.01 ]
    ;[0 0.89 0 0 0 0.05 0.05 0.01 ]
    ;[0 0.14 0 0 0 0.8 0.01 0.05 ]
    ;[0 0.14 0 0 0 0.01 0.8 0.05 ]
    ;[0 0.14 0 0 0 0.8 0.01 0.05 ]
    ;[0 0.05 0 0 0 0.1 0.1 0.75 ]
    ;[0 0.14 0 0 0 0.01 0.8 0.05 ]
    ;[0 0.14 0 0 0 0.8 0.01 0.05 ]
    ;[0 0.14 0 0 0 0.01 0.8 0.05 ]
    ;[0 0.14 0 0 0 0.8 0.01 0.05 ]
    ;[0 0.05 0 0 0 0.1 0.1 0.75 ]
    ;[0 0.14 0 0 0 0.01 0.8 0.05 ]
    ;[0 0.14 0 0 0 0.8 0.01 0.05 ]
    ;[0 0.14 0 0 0 0.01 0.8 0.05 ]
    ;[0 0.89 0 0 0 0.05 0.05 0.01 ]
    ;[0 0.89 0 0 0 0.05 0.05 0.01 ]
    ;[0 0.89 0 0 0 0.05 0.05 0.01 ]
    ;[0 0.14 0 0 0 0.8 0.01 0.05 ]];

for i = 1 : n_s
    for j = 1 : n_o
        O[i,1,j] = Oa0[i,j]
        O[i,2,j] = Oa1[i,j]
        O[i,3,j] = Oa2[i,j]
        O[i,4,j] = Oa3[i,j]
        O[i,5,j] = Oa4[i,j]
        O[i,6,j] = Oa5[i,j]
    end
end

a0_list = [3.4 1.2 1.2 4.0 0.6 3.4 3.4 150.0 0.6 3.4 3.4 0.6 2.8 3.4 0.6 0.6 1.2 1.2 1.2 0.6]
a1_list = [0.6 1.2 1.2 1.2 0.6 3.4 0.6 150.0 3.4 0.6 3.4 3.4 2.8 0.6 3.4 3.4 1.2 4.0 4.0 0.6]
a2_list = [3.4 2.8 2.8 3.4 0.0 4.0 0.6 150.0 3.4 0.6 4.0 0.6 1.2 3.4 0.6 3.4 2.8 3.4 3.4 0.0]
a3_list = [0.6 2.8 2.8 3.4 2.8 1.2 3.4 150.0 0.6 3.4 1.2 3.4 1.2 0.6 3.4 0.6 2.8 3.4 3.4 2.8]

for i = 1 : n_s
    for j = 1 : n_s
        R[i,1,j] = a0_list[i]
        R[i,2,j] = a1_list[i]
        R[i,3,j] = a2_list[i]
        R[i,4,j] = a3_list[i]
        R[i,5,j] = 2
        R[i,6,j] = 2
    end
end


################################################
############ Simulation  Function ##############
################################################

# One trial

function one_maze20_trial(T,R,O,t_step,alpha,gamma)
    # T,R,O are the environment models
    # t_step is the simulation length
    # alpha are alpha vectors, obtained from value-function approximation method
    # gamma is the discounted factor

    (n_s,n_a,n_o) = (size(T)[1],size(T)[2],size(O)[3])

    # initial belief (given by the file)
    b = zeros(Float64,n_s)
    b[1] = 0.3; b[5] = 0.3; b[20] = 0.4;

    # Initialize state
    x = 0
    initi = rand()
    if initi < 0.3
        x = 1
    elseif initi < 0.6
        x = 5
    else
        x = 20
    end

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

function one_maze20_trial_2(T,R,O,t_step,alpha,gamma)
    (n_s,n_a,n_o) = (size(T)[1],size(T)[2],size(O)[3])

    # initial belief
    b = ones(Float64,n_s) * (1/n_s)

    # Initialize state (uniformly over all states)
    x = round(Int64,div(rand()*20,1)) + 1

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

#############################################
############ Run Simulation #################
#############################################

f = open("run_collect","w")


gamma = 0.9
#redu_f = 1.5
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
#@time MY_3_alpha = purely_iteration_v3(zeros(Float64,n_s,n_a),T,R,O,0.01,gamma)
#@time MY_5_alpha = purely_iteration_v5(zeros(Float64,n_s,n_a),T,R,O,0.01,gamma)
#@time MY_6_alpha = purely_iteration_v6(zeros(Float64,n_s,n_a),T,R,O,0.01,gamma)
#@time MY_7_alpha = purely_iteration_v7(zeros(Float64,n_s,n_a),T,R,O,0.01,gamma)

QMDP_r_sum = 0
QUMDP_r_sum = 0
FIB_r_sum = 0
#MY_3_r_sum = 0
#MY_5_r_sum = 0
#MY_6_r_sum = 0
#MY_7_r_sum = 0
t_trial = 2000
t_step = 60

for i = 1 : t_trial
    #if (i%100 == 0); println("trial = ",i); end
    QMDP_r_sum += one_maze20_trial(T,R,O,t_step,QMDP_alpha,gamma)
    QUMDP_r_sum += one_maze20_trial(T,R,O,t_step,QUMDP_alpha,gamma)
    FIB_r_sum += one_maze20_trial(T,R,O,t_step,FIB_alpha,gamma)
    #MY_3_r_sum += one_maze20_trial(T,R,O,t_step,MY_3_alpha,gamma)
    #MY_5_r_sum += one_maze20_trial(T,R,O,t_step,MY_5_alpha,gamma)
    #MY_6_r_sum += one_maze20_trial(T,R,O,t_step,MY_6_alpha,gamma)
    #MY_7_r_sum += one_maze20_trial(T,R,O,t_step,MY_7_alpha,gamma)
end

    red[reduced_time] = redu_f
    QMDP_r[reduced_time] = QMDP_r_sum/t_trial
    UMDP_r[reduced_time] = QUMDP_r_sum/t_trial
    FIB_r[reduced_time] = FIB_r_sum/t_trial


#println(f,redu_f)
#println(f,QMDP_r_sum/t_trial)
#println(f,QUMDP_r_sum/t_trial)
#println(f,FIB_r_sum/t_trial)
#println(f,MY_3_r_sum/t_trial)
#println(MY_5_r_sum/t_trial)
#println(MY_7_r_sum/t_trial)
#println(MY_6_r_sum/t_trial)


end

close(f)

plot(red,QMDP_r,label="QMDP")
plot(red,UMDP_r,label="UMDP")
plot(red,FIB_r,label="FIB")
xlabel("reduced factor")
ylabel("Discounted Reward")
title("Maze20 with gamma 0.95 and concentrated initial belief")
legend(loc="upper right",fancybox="true")
annotate("SARSOP = 116.84",
	xy=[1;0],
	xycoords="axes fraction",
	xytext=[-10,10],
	textcoords="offset points",
	fontsize=12.0,
	ha="right",
	va="bottom")
