f = open("collect.txt","w")

string = "T: 0 : 12 : 12 1.000000
T: 1 : 12 : 17 0.050000
T: 1 : 12 : 28 0.025000
T: 1 : 12 : 30 0.025000
T: 1 : 12 : 11 0.050000
T: 1 : 12 : 12 0.850000
T: 2 : 12 : 12 0.100000
T: 2 : 12 : 13 0.700000
T: 2 : 12 : 14 0.100000
T: 2 : 12 : 15 0.100000
T: 3 : 12 : 12 0.100000
T: 3 : 12 : 13 0.150000
T: 3 : 12 : 14 0.600000
T: 3 : 12 : 15 0.150000
T: 4 : 12 : 12 0.100000
T: 4 : 12 : 13 0.100000
T: 4 : 12 : 14 0.100000
T: 4 : 12 : 15 0.700000
T: 0 : 13 : 13 1.000000
T: 1 : 13 : 17 0.800000
T: 1 : 13 : 30 0.050000
T: 1 : 13 : 9 0.025000
T: 1 : 13 : 11 0.025000
T: 1 : 13 : 13 0.100000
T: 2 : 13 : 12 0.100000
T: 2 : 13 : 13 0.100000
T: 2 : 13 : 14 0.700000
T: 2 : 13 : 15 0.100000
T: 3 : 13 : 12 0.150000
T: 3 : 13 : 13 0.100000
T: 3 : 13 : 14 0.150000
T: 3 : 13 : 15 0.600000
T: 4 : 13 : 12 0.700000
T: 4 : 13 : 13 0.100000
T: 4 : 13 : 14 0.100000
T: 4 : 13 : 15 0.100000
T: 0 : 14 : 14 1.000000
T: 1 : 14 : 17 0.050000
T: 1 : 14 : 30 0.800000
T: 1 : 14 : 11 0.050000
T: 1 : 14 : 14 0.100000
T: 2 : 14 : 12 0.100000
T: 2 : 14 : 13 0.100000
T: 2 : 14 : 14 0.100000
T: 2 : 14 : 15 0.700000
T: 3 : 14 : 12 0.600000
T: 3 : 14 : 13 0.150000
T: 3 : 14 : 14 0.100000
T: 3 : 14 : 15 0.150000
T: 4 : 14 : 12 0.100000
T: 4 : 14 : 13 0.700000
T: 4 : 14 : 14 0.100000
T: 4 : 14 : 15 0.100000
T: 0 : 15 : 15 1.000000
T: 1 : 15 : 17 0.025000
T: 1 : 15 : 19 0.025000
T: 1 : 15 : 30 0.050000
T: 1 : 15 : 11 0.800000
T: 1 : 15 : 15 0.100000
T: 2 : 15 : 12 0.700000
T: 2 : 15 : 13 0.100000
T: 2 : 15 : 14 0.100000
T: 2 : 15 : 15 0.100000
T: 3 : 15 : 12 0.150000
T: 3 : 15 : 13 0.600000
T: 3 : 15 : 14 0.150000
T: 3 : 15 : 15 0.100000
T: 4 : 15 : 12 0.100000
T: 4 : 15 : 13 0.100000
T: 4 : 15 : 14 0.700000
T: 4 : 15 : 15 0.100000
T: 0 : 16 : 16 1.000000
T: 1 : 16 : 32 0.025000
T: 1 : 16 : 34 0.025000
T: 1 : 16 : 15 0.050000
T: 1 : 16 : 16 0.900000
T: 2 : 16 : 16 0.100000
T: 2 : 16 : 17 0.700000
T: 2 : 16 : 18 0.100000
T: 2 : 16 : 19 0.100000
T: 3 : 16 : 16 0.100000
T: 3 : 16 : 17 0.150000
T: 3 : 16 : 18 0.600000
T: 3 : 16 : 19 0.150000
T: 4 : 16 : 16 0.100000
T: 4 : 16 : 17 0.100000
T: 4 : 16 : 18 0.100000
T: 4 : 16 : 19 0.700000
T: 0 : 17 : 17 1.000000
T: 1 : 17 : 34 0.050000
T: 1 : 17 : 13 0.025000
T: 1 : 17 : 15 0.025000
T: 1 : 17 : 17 0.900000
T: 2 : 17 : 16 0.100000
T: 2 : 17 : 17 0.100000
T: 2 : 17 : 18 0.700000
T: 2 : 17 : 19 0.100000
T: 3 : 17 : 16 0.150000
T: 3 : 17 : 17 0.100000
T: 3 : 17 : 18 0.150000
T: 3 : 17 : 19 0.600000
T: 4 : 17 : 16 0.700000
T: 4 : 17 : 17 0.100000
T: 4 : 17 : 18 0.100000
T: 4 : 17 : 19 0.100000
T: 0 : 18 : 18 1.000000
T: 1 : 18 : 34 0.800000
T: 1 : 18 : 15 0.050000
T: 1 : 18 : 18 0.150000
T: 2 : 18 : 16 0.100000
T: 2 : 18 : 17 0.100000
T: 2 : 18 : 18 0.100000
T: 2 : 18 : 19 0.700000
T: 3 : 18 : 16 0.600000
T: 3 : 18 : 17 0.150000
T: 3 : 18 : 18 0.100000
T: 3 : 18 : 19 0.150000
T: 4 : 18 : 16 0.100000
T: 4 : 18 : 17 0.700000
T: 4 : 18 : 18 0.100000
T: 4 : 18 : 19 0.100000
T: 0 : 19 : 19 1.000000
T: 1 : 19 : 34 0.050000
T: 1 : 19 : 15 0.800000
T: 1 : 19 : 19 0.150000
T: 2 : 19 : 16 0.700000
T: 2 : 19 : 17 0.100000
T: 2 : 19 : 18 0.100000
T: 2 : 19 : 19 0.100000
T: 3 : 19 : 16 0.150000
T: 3 : 19 : 17 0.600000
T: 3 : 19 : 18 0.150000
T: 3 : 19 : 19 0.100000
T: 4 : 19 : 16 0.100000
T: 4 : 19 : 17 0.100000
T: 4 : 19 : 18 0.700000
T: 4 : 19 : 19 0.100000
T: 0 : 20 : 20 1.000000
T: 1 : 20 : 0 0.800000
T: 1 : 20 : 25 0.050000
T: 1 : 20 : 20 0.150000
T: 2 : 20 : 20 0.100000
T: 2 : 20 : 21 0.700000
T: 2 : 20 : 22 0.100000
T: 2 : 20 : 23 0.100000
T: 3 : 20 : 20 0.100000
T: 3 : 20 : 21 0.150000
T: 3 : 20 : 22 0.600000
T: 3 : 20 : 23 0.150000
T: 4 : 20 : 20 0.100000
T: 4 : 20 : 21 0.100000
T: 4 : 20 : 22 0.100000
T: 4 : 20 : 23 0.700000
T: 0 : 21 : 21 1.000000
T: 1 : 21 : 0 0.050000
T: 1 : 21 : 25 0.800000
T: 1 : 21 : 21 0.150000
T: 2 : 21 : 20 0.100000
T: 2 : 21 : 21 0.100000
T: 2 : 21 : 22 0.700000
T: 2 : 21 : 23 0.100000
T: 3 : 21 : 20 0.150000
T: 3 : 21 : 21 0.100000
T: 3 : 21 : 22 0.150000
T: 3 : 21 : 23 0.600000
T: 4 : 21 : 20 0.700000
T: 4 : 21 : 21 0.100000
T: 4 : 21 : 22 0.100000
T: 4 : 21 : 23 0.100000
T: 0 : 22 : 22 1.000000
T: 1 : 22 : 0 0.025000
T: 1 : 22 : 2 0.025000
T: 1 : 22 : 25 0.050000
T: 1 : 22 : 22 0.900000
T: 2 : 22 : 20 0.100000
T: 2 : 22 : 21 0.100000
T: 2 : 22 : 22 0.100000
T: 2 : 22 : 23 0.700000
T: 3 : 22 : 20 0.600000
T: 3 : 22 : 21 0.150000
T: 3 : 22 : 22 0.100000
T: 3 : 22 : 23 0.150000
T: 4 : 22 : 20 0.100000
T: 4 : 22 : 21 0.700000
T: 4 : 22 : 22 0.100000
T: 4 : 22 : 23 0.100000
T: 0 : 23 : 23 1.000000
T: 1 : 23 : 0 0.050000
T: 1 : 23 : 25 0.025000
T: 1 : 23 : 27 0.025000
T: 1 : 23 : 23 0.900000
T: 2 : 23 : 20 0.700000
T: 2 : 23 : 21 0.100000
T: 2 : 23 : 22 0.100000
T: 2 : 23 : 23 0.100000
T: 3 : 23 : 20 0.150000
T: 3 : 23 : 21 0.600000
T: 3 : 23 : 22 0.150000
T: 3 : 23 : 23 0.100000
T: 4 : 23 : 20 0.100000
T: 4 : 23 : 21 0.100000
T: 4 : 23 : 22 0.700000
T: 4 : 23 : 23 0.100000
T: 0 : 24 : 24 1.000000
T: 1 : 24 : 4 0.800000
T: 1 : 24 : 23 0.050000
T: 1 : 24 : 24 0.150000
T: 2 : 24 : 24 0.100000
T: 2 : 24 : 25 0.700000
T: 2 : 24 : 26 0.100000
T: 2 : 24 : 27 0.100000
T: 3 : 24 : 24 0.100000
T: 3 : 24 : 25 0.150000
T: 3 : 24 : 26 0.600000
T: 3 : 24 : 27 0.150000
T: 4 : 24 : 24 0.100000
T: 4 : 24 : 25 0.100000
T: 4 : 24 : 26 0.100000
T: 4 : 24 : 27 0.700000
T: 0 : 25 : 25 1.000000
T: 1 : 25 : 4 0.050000
T: 1 : 25 : 21 0.025000
T: 1 : 25 : 23 0.025000
T: 1 : 25 : 25 0.900000
T: 2 : 25 : 24 0.100000
T: 2 : 25 : 25 0.100000
T: 2 : 25 : 26 0.700000
T: 2 : 25 : 27 0.100000
T: 3 : 25 : 24 0.150000
T: 3 : 25 : 25 0.100000
T: 3 : 25 : 26 0.150000
T: 3 : 25 : 27 0.600000
T: 4 : 25 : 24 0.700000
T: 4 : 25 : 25 0.100000
T: 4 : 25 : 26 0.100000
T: 4 : 25 : 27 0.100000
T: 0 : 26 : 26 1.000000
T: 1 : 26 : 4 0.025000
T: 1 : 26 : 6 0.025000
T: 1 : 26 : 23 0.050000
T: 1 : 26 : 26 0.900000
T: 2 : 26 : 24 0.100000
T: 2 : 26 : 25 0.100000
T: 2 : 26 : 26 0.100000
T: 2 : 26 : 27 0.700000
T: 3 : 26 : 24 0.600000
T: 3 : 26 : 25 0.150000
T: 3 : 26 : 26 0.100000
T: 3 : 26 : 27 0.150000
T: 4 : 26 : 24 0.100000
T: 4 : 26 : 25 0.700000
T: 4 : 26 : 26 0.100000
T: 4 : 26 : 27 0.100000
T: 0 : 27 : 27 1.000000
T: 1 : 27 : 4 0.050000
T: 1 : 27 : 23 0.800000
T: 1 : 27 : 27 0.150000
T: 2 : 27 : 24 0.700000
T: 2 : 27 : 25 0.100000
T: 2 : 27 : 26 0.100000
T: 2 : 27 : 27 0.100000
T: 3 : 27 : 24 0.150000
T: 3 : 27 : 25 0.600000
T: 3 : 27 : 26 0.150000
T: 3 : 27 : 27 0.100000
T: 4 : 27 : 24 0.100000
T: 4 : 27 : 25 0.100000
T: 4 : 27 : 26 0.700000
T: 4 : 27 : 27 0.100000
T: 0 : 28 : 28 1.000000
T: 1 : 28 : 12 0.800000
T: 1 : 28 : 33 0.050000
T: 1 : 28 : 28 0.150000
T: 2 : 28 : 28 0.100000
T: 2 : 28 : 29 0.700000
T: 2 : 28 : 30 0.100000
T: 2 : 28 : 31 0.100000
T: 3 : 28 : 28 0.100000
T: 3 : 28 : 29 0.150000
T: 3 : 28 : 30 0.600000
T: 3 : 28 : 31 0.150000
T: 4 : 28 : 28 0.100000
T: 4 : 28 : 29 0.100000
T: 4 : 28 : 30 0.100000
T: 4 : 28 : 31 0.700000
T: 0 : 29 : 29 1.000000
T: 1 : 29 : 12 0.050000
T: 1 : 29 : 33 0.800000
T: 1 : 29 : 29 0.150000
T: 2 : 29 : 28 0.100000
T: 2 : 29 : 29 0.100000
T: 2 : 29 : 30 0.700000
T: 2 : 29 : 31 0.100000
T: 3 : 29 : 28 0.150000
T: 3 : 29 : 29 0.100000
T: 3 : 29 : 30 0.150000
T: 3 : 29 : 31 0.600000
T: 4 : 29 : 28 0.700000
T: 4 : 29 : 29 0.100000
T: 4 : 29 : 30 0.100000
T: 4 : 29 : 31 0.100000
T: 0 : 30 : 30 1.000000
T: 1 : 30 : 12 0.025000
T: 1 : 30 : 14 0.025000
T: 1 : 30 : 33 0.050000
T: 1 : 30 : 30 0.900000
T: 2 : 30 : 28 0.100000
T: 2 : 30 : 29 0.100000
T: 2 : 30 : 30 0.100000
T: 2 : 30 : 31 0.700000
T: 3 : 30 : 28 0.600000
T: 3 : 30 : 29 0.150000
T: 3 : 30 : 30 0.100000
T: 3 : 30 : 31 0.150000
T: 4 : 30 : 28 0.100000
T: 4 : 30 : 29 0.700000
T: 4 : 30 : 30 0.100000
T: 4 : 30 : 31 0.100000
T: 0 : 31 : 31 1.000000
T: 1 : 31 : 12 0.050000
T: 1 : 31 : 33 0.025000
T: 1 : 31 : 35 0.025000
T: 1 : 31 : 31 0.900000
T: 2 : 31 : 28 0.700000
T: 2 : 31 : 29 0.100000
T: 2 : 31 : 30 0.100000
T: 2 : 31 : 31 0.100000
T: 3 : 31 : 28 0.150000
T: 3 : 31 : 29 0.600000
T: 3 : 31 : 30 0.150000
T: 3 : 31 : 31 0.100000
T: 4 : 31 : 28 0.100000
T: 4 : 31 : 29 0.100000
T: 4 : 31 : 30 0.700000
T: 4 : 31 : 31 0.100000
T: 0 : 32 : 32 1.000000
T: 1 : 32 : 16 0.800000
T: 1 : 32 : 31 0.050000
T: 1 : 32 : 32 0.150000
T: 2 : 32 : 32 0.100000
T: 2 : 32 : 33 0.700000
T: 2 : 32 : 34 0.100000
T: 2 : 32 : 35 0.100000
T: 3 : 32 : 32 0.100000
T: 3 : 32 : 33 0.150000
T: 3 : 32 : 34 0.600000
T: 3 : 32 : 35 0.150000
T: 4 : 32 : 32 0.100000
T: 4 : 32 : 33 0.100000
T: 4 : 32 : 34 0.100000
T: 4 : 32 : 35 0.700000
T: 0 : 33 : 33 1.000000
T: 1 : 33 : 16 0.050000
T: 1 : 33 : 29 0.025000
T: 1 : 33 : 31 0.025000
T: 1 : 33 : 33 0.900000
T: 2 : 33 : 32 0.100000
T: 2 : 33 : 33 0.100000
T: 2 : 33 : 34 0.700000
T: 2 : 33 : 35 0.100000
T: 3 : 33 : 32 0.150000
T: 3 : 33 : 33 0.100000
T: 3 : 33 : 34 0.150000
T: 3 : 33 : 35 0.600000
T: 4 : 33 : 32 0.700000
T: 4 : 33 : 33 0.100000
T: 4 : 33 : 34 0.100000
T: 4 : 33 : 35 0.100000
T: 0 : 34 : 34 1.000000
T: 1 : 34 : 16 0.025000
T: 1 : 34 : 18 0.025000
T: 1 : 34 : 31 0.050000
T: 1 : 34 : 34 0.900000
T: 2 : 34 : 32 0.100000
T: 2 : 34 : 33 0.100000
T: 2 : 34 : 34 0.100000
T: 2 : 34 : 35 0.700000
T: 3 : 34 : 32 0.600000
T: 3 : 34 : 33 0.150000
T: 3 : 34 : 34 0.100000
T: 3 : 34 : 35 0.150000
T: 4 : 34 : 32 0.100000
T: 4 : 34 : 33 0.700000
T: 4 : 34 : 34 0.100000
T: 4 : 34 : 35 0.100000
T: 0 : 35 : 35 1.000000
T: 1 : 35 : 16 0.050000
T: 1 : 35 : 31 0.800000
T: 1 : 35 : 35 0.150000
T: 2 : 35 : 32 0.700000
T: 2 : 35 : 33 0.100000
T: 2 : 35 : 34 0.100000
T: 2 : 35 : 35 0.100000
T: 3 : 35 : 32 0.150000
T: 3 : 35 : 33 0.600000
T: 3 : 35 : 34 0.150000
T: 3 : 35 : 35 0.100000
T: 4 : 35 : 32 0.100000
T: 4 : 35 : 33 0.100000
T: 4 : 35 : 34 0.700000
T: 4 : 35 : 35 0.100000"

X = split(string,"\n")
for i = 1 : length(X)
    row = X[i]
    words = split(row," ")
    action = parse(words[2]) + 1
    init_s = parse(words[4]) + 1
    second_s = parse(words[6]) + 1
    prob = float(words[7])
    #prob = float(words[9])
    println(f,"T[",init_s,",",action,",",second_s,"]=",prob)
    #println(f,"RR[",init_s,",",action,"]=",prob)
end

#X = split(string," ")
#for i = 1 : length(X)
#    println(f,"T: * : 57 : ",i-1," ",X[i])
#end
#for i = 1 : length(X)
#    println(f,"T: * : 58 : ",i-1," ",X[i])
#end
#for i = 1 : length(X)
#    println(f,"T: * : 59 : ",i-1," ",X[i])
#end

close(f)
