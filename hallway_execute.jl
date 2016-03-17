include("project_1.jl")
include("project_2.jl")

T = zeros(Float64,60,5,60)

T[1,1,1]=1.0
T[1,2,6]=0.05
T[1,2,1]=0.95
T[1,3,1]=0.1
T[1,3,2]=0.7
T[1,3,3]=0.1
T[1,3,4]=0.1
T[1,4,1]=0.1
T[1,4,2]=0.15
T[1,4,3]=0.6
T[1,4,4]=0.15
T[1,5,1]=0.1
T[1,5,2]=0.1
T[1,5,3]=0.1
T[1,5,4]=0.7
T[2,1,2]=1.0
T[2,2,6]=0.8
T[2,2,2]=0.2
T[2,3,1]=0.1
T[2,3,2]=0.1
T[2,3,3]=0.7
T[2,3,4]=0.1
T[2,4,1]=0.15
T[2,4,2]=0.1
T[2,4,3]=0.15
T[2,4,4]=0.6
T[2,5,1]=0.7
T[2,5,2]=0.1
T[2,5,3]=0.1
T[2,5,4]=0.1
T[3,1,3]=1.0
T[3,2,6]=0.05
T[3,2,3]=0.95
T[3,3,1]=0.1
T[3,3,2]=0.1
T[3,3,3]=0.1
T[3,3,4]=0.7
T[3,4,1]=0.6
T[3,4,2]=0.15
T[3,4,3]=0.1
T[3,4,4]=0.15
T[3,5,1]=0.1
T[3,5,2]=0.7
T[3,5,3]=0.1
T[3,5,4]=0.1
T[4,1,4]=1.0
T[4,2,6]=0.025
T[4,2,8]=0.025
T[4,2,4]=0.95
T[4,3,1]=0.7
T[4,3,2]=0.1
T[4,3,3]=0.1
T[4,3,4]=0.1
T[4,4,1]=0.15
T[4,4,2]=0.6
T[4,4,3]=0.15
T[4,4,4]=0.1
T[4,5,1]=0.1
T[4,5,2]=0.1
T[4,5,3]=0.7
T[4,5,4]=0.1
T[5,1,5]=1.0
T[5,2,10]=0.05
T[5,2,4]=0.05
T[5,2,5]=0.9
T[5,3,5]=0.1
T[5,3,6]=0.7
T[5,3,7]=0.1
T[5,3,8]=0.1
T[5,4,5]=0.1
T[5,4,6]=0.15
T[5,4,7]=0.6
T[5,4,8]=0.15
T[5,5,5]=0.1
T[5,5,6]=0.1
T[5,5,7]=0.1
T[5,5,8]=0.7
T[6,1,6]=1.0
T[6,2,10]=0.8
T[6,2,2]=0.025
T[6,2,4]=0.025
T[6,2,6]=0.15
T[6,3,5]=0.1
T[6,3,6]=0.1
T[6,3,7]=0.7
T[6,3,8]=0.1
T[6,4,5]=0.15
T[6,4,6]=0.1
T[6,4,7]=0.15
T[6,4,8]=0.6
T[6,5,5]=0.7
T[6,5,6]=0.1
T[6,5,7]=0.1
T[6,5,8]=0.1
T[7,1,7]=1.0
T[7,2,10]=0.05
T[7,2,4]=0.05
T[7,2,7]=0.9
T[7,3,5]=0.1
T[7,3,6]=0.1
T[7,3,7]=0.1
T[7,3,8]=0.7
T[7,4,5]=0.6
T[7,4,6]=0.15
T[7,4,7]=0.1
T[7,4,8]=0.15
T[7,5,5]=0.1
T[7,5,6]=0.7
T[7,5,7]=0.1
T[7,5,8]=0.1
T[8,1,8]=1.0
T[8,2,10]=0.025
T[8,2,12]=0.025
T[8,2,4]=0.8
T[8,2,8]=0.15
T[8,3,5]=0.7
T[8,3,6]=0.1
T[8,3,7]=0.1
T[8,3,8]=0.1
T[8,4,5]=0.15
T[8,4,6]=0.6
T[8,4,7]=0.15
T[8,4,8]=0.1
T[8,5,5]=0.1
T[8,5,6]=0.1
T[8,5,7]=0.7
T[8,5,8]=0.1
T[9,1,9]=1.0
T[9,2,14]=0.05
T[9,2,45]=0.025
T[9,2,47]=0.025
T[9,2,8]=0.05
T[9,2,9]=0.85
T[9,3,9]=0.1
T[9,3,10]=0.7
T[9,3,11]=0.1
T[9,3,12]=0.1
T[9,4,9]=0.1
T[9,4,10]=0.15
T[9,4,11]=0.6
T[9,4,12]=0.15
T[9,5,9]=0.1
T[9,5,10]=0.1
T[9,5,11]=0.1
T[9,5,12]=0.7
T[10,1,10]=1.0
T[10,2,14]=0.8
T[10,2,47]=0.05
T[10,2,6]=0.025
T[10,2,8]=0.025
T[10,2,10]=0.1
T[10,3,9]=0.1
T[10,3,10]=0.1
T[10,3,11]=0.7
T[10,3,12]=0.1
T[10,4,9]=0.15
T[10,4,10]=0.1
T[10,4,11]=0.15
T[10,4,12]=0.6
T[10,5,9]=0.7
T[10,5,10]=0.1
T[10,5,11]=0.1
T[10,5,12]=0.1
T[11,1,11]=1.0
T[11,2,14]=0.05
T[11,2,47]=0.8
T[11,2,8]=0.05
T[11,2,11]=0.1
T[11,3,9]=0.1
T[11,3,10]=0.1
T[11,3,11]=0.1
T[11,3,12]=0.7
T[11,4,9]=0.6
T[11,4,10]=0.15
T[11,4,11]=0.1
T[11,4,12]=0.15
T[11,5,9]=0.1
T[11,5,10]=0.7
T[11,5,11]=0.1
T[11,5,12]=0.1
T[12,1,12]=1.0
T[12,2,14]=0.025
T[12,2,16]=0.025
T[12,2,47]=0.05
T[12,2,8]=0.8
T[12,2,12]=0.1
T[12,3,9]=0.7
T[12,3,10]=0.1
T[12,3,11]=0.1
T[12,3,12]=0.1
T[12,4,9]=0.15
T[12,4,10]=0.6
T[12,4,11]=0.15
T[12,4,12]=0.1
T[12,5,9]=0.1
T[12,5,10]=0.1
T[12,5,11]=0.7
T[12,5,12]=0.1
T[13,1,13]=1.0
T[13,2,18]=0.05
T[13,2,12]=0.05
T[13,2,13]=0.9
T[13,3,13]=0.1
T[13,3,14]=0.7
T[13,3,15]=0.1
T[13,3,16]=0.1
T[13,4,13]=0.1
T[13,4,14]=0.15
T[13,4,15]=0.6
T[13,4,16]=0.15
T[13,5,13]=0.1
T[13,5,14]=0.1
T[13,5,15]=0.1
T[13,5,16]=0.7
T[14,1,14]=1.0
T[14,2,18]=0.8
T[14,2,10]=0.025
T[14,2,12]=0.025
T[14,2,14]=0.15
T[14,3,13]=0.1
T[14,3,14]=0.1
T[14,3,15]=0.7
T[14,3,16]=0.1
T[14,4,13]=0.15
T[14,4,14]=0.1
T[14,4,15]=0.15
T[14,4,16]=0.6
T[14,5,13]=0.7
T[14,5,14]=0.1
T[14,5,15]=0.1
T[14,5,16]=0.1
T[15,1,15]=1.0
T[15,2,18]=0.05
T[15,2,12]=0.05
T[15,2,15]=0.9
T[15,3,13]=0.1
T[15,3,14]=0.1
T[15,3,15]=0.1
T[15,3,16]=0.7
T[15,4,13]=0.6
T[15,4,14]=0.15
T[15,4,15]=0.1
T[15,4,16]=0.15
T[15,5,13]=0.1
T[15,5,14]=0.7
T[15,5,15]=0.1
T[15,5,16]=0.1
T[16,1,16]=1.0
T[16,2,18]=0.025
T[16,2,20]=0.025
T[16,2,12]=0.8
T[16,2,16]=0.15
T[16,3,13]=0.7
T[16,3,14]=0.1
T[16,3,15]=0.1
T[16,3,16]=0.1
T[16,4,13]=0.15
T[16,4,14]=0.6
T[16,4,15]=0.15
T[16,4,16]=0.1
T[16,5,13]=0.1
T[16,5,14]=0.1
T[16,5,15]=0.7
T[16,5,16]=0.1
T[17,1,17]=1.0
T[17,2,22]=0.05
T[17,2,49]=0.025
T[17,2,51]=0.025
T[17,2,16]=0.05
T[17,2,17]=0.85
T[17,3,17]=0.1
T[17,3,18]=0.7
T[17,3,19]=0.1
T[17,3,20]=0.1
T[17,4,17]=0.1
T[17,4,18]=0.15
T[17,4,19]=0.6
T[17,4,20]=0.15
T[17,5,17]=0.1
T[17,5,18]=0.1
T[17,5,19]=0.1
T[17,5,20]=0.7
T[18,1,18]=1.0
T[18,2,22]=0.8
T[18,2,51]=0.05
T[18,2,14]=0.025
T[18,2,16]=0.025
T[18,2,18]=0.1
T[18,3,17]=0.1
T[18,3,18]=0.1
T[18,3,19]=0.7
T[18,3,20]=0.1
T[18,4,17]=0.15
T[18,4,18]=0.1
T[18,4,19]=0.15
T[18,4,20]=0.6
T[18,5,17]=0.7
T[18,5,18]=0.1
T[18,5,19]=0.1
T[18,5,20]=0.1
T[19,1,19]=1.0
T[19,2,22]=0.05
T[19,2,51]=0.8
T[19,2,16]=0.05
T[19,2,19]=0.1
T[19,3,17]=0.1
T[19,3,18]=0.1
T[19,3,19]=0.1
T[19,3,20]=0.7
T[19,4,17]=0.6
T[19,4,18]=0.15
T[19,4,19]=0.1
T[19,4,20]=0.15
T[19,5,17]=0.1
T[19,5,18]=0.7
T[19,5,19]=0.1
T[19,5,20]=0.1
T[20,1,20]=1.0
T[20,2,22]=0.025
T[20,2,24]=0.025
T[20,2,51]=0.05
T[20,2,16]=0.8
T[20,2,20]=0.1
T[20,3,17]=0.7
T[20,3,18]=0.1
T[20,3,19]=0.1
T[20,3,20]=0.1
T[20,4,17]=0.15
T[20,4,18]=0.6
T[20,4,19]=0.15
T[20,4,20]=0.1
T[20,5,17]=0.1
T[20,5,18]=0.1
T[20,5,19]=0.7
T[20,5,20]=0.1
T[21,1,21]=1.0
T[21,2,26]=0.05
T[21,2,20]=0.05
T[21,2,21]=0.9
T[21,3,21]=0.1
T[21,3,22]=0.7
T[21,3,23]=0.1
T[21,3,24]=0.1
T[21,4,21]=0.1
T[21,4,22]=0.15
T[21,4,23]=0.6
T[21,4,24]=0.15
T[21,5,21]=0.1
T[21,5,22]=0.1
T[21,5,23]=0.1
T[21,5,24]=0.7
T[22,1,22]=1.0
T[22,2,26]=0.8
T[22,2,18]=0.025
T[22,2,20]=0.025
T[22,2,22]=0.15
T[22,3,21]=0.1
T[22,3,22]=0.1
T[22,3,23]=0.7
T[22,3,24]=0.1
T[22,4,21]=0.15
T[22,4,22]=0.1
T[22,4,23]=0.15
T[22,4,24]=0.6
T[22,5,21]=0.7
T[22,5,22]=0.1
T[22,5,23]=0.1
T[22,5,24]=0.1
T[23,1,23]=1.0
T[23,2,26]=0.05
T[23,2,20]=0.05
T[23,2,23]=0.9
T[23,3,21]=0.1
T[23,3,22]=0.1
T[23,3,23]=0.1
T[23,3,24]=0.7
T[23,4,21]=0.6
T[23,4,22]=0.15
T[23,4,23]=0.1
T[23,4,24]=0.15
T[23,5,21]=0.1
T[23,5,22]=0.7
T[23,5,23]=0.1
T[23,5,24]=0.1
T[24,1,24]=1.0
T[24,2,26]=0.025
T[24,2,28]=0.025
T[24,2,20]=0.8
T[24,2,24]=0.15
T[24,3,21]=0.7
T[24,3,22]=0.1
T[24,3,23]=0.1
T[24,3,24]=0.1
T[24,4,21]=0.15
T[24,4,22]=0.6
T[24,4,23]=0.15
T[24,4,24]=0.1
T[24,5,21]=0.1
T[24,5,22]=0.1
T[24,5,23]=0.7
T[24,5,24]=0.1
T[25,1,25]=1.0
T[25,2,30]=0.05
T[25,2,53]=0.025
T[25,2,55]=0.025
T[25,2,24]=0.05
T[25,2,25]=0.85
T[25,3,25]=0.1
T[25,3,26]=0.7
T[25,3,27]=0.1
T[25,3,28]=0.1
T[25,4,25]=0.1
T[25,4,26]=0.15
T[25,4,27]=0.6
T[25,4,28]=0.15
T[25,5,25]=0.1
T[25,5,26]=0.1
T[25,5,27]=0.1
T[25,5,28]=0.7
T[26,1,26]=1.0
T[26,2,30]=0.8
T[26,2,55]=0.05
T[26,2,22]=0.025
T[26,2,24]=0.025
T[26,2,26]=0.1
T[26,3,25]=0.1
T[26,3,26]=0.1
T[26,3,27]=0.7
T[26,3,28]=0.1
T[26,4,25]=0.15
T[26,4,26]=0.1
T[26,4,27]=0.15
T[26,4,28]=0.6
T[26,5,25]=0.7
T[26,5,26]=0.1
T[26,5,27]=0.1
T[26,5,28]=0.1
T[27,1,27]=1.0
T[27,2,30]=0.05
T[27,2,55]=0.8
T[27,2,24]=0.05
T[27,2,27]=0.1
T[27,3,25]=0.1
T[27,3,26]=0.1
T[27,3,27]=0.1
T[27,3,28]=0.7
T[27,4,25]=0.6
T[27,4,26]=0.15
T[27,4,27]=0.1
T[27,4,28]=0.15
T[27,5,25]=0.1
T[27,5,26]=0.7
T[27,5,27]=0.1
T[27,5,28]=0.1
T[28,1,28]=1.0
T[28,2,30]=0.025
T[28,2,32]=0.025
T[28,2,55]=0.05
T[28,2,24]=0.8
T[28,2,28]=0.1
T[28,3,25]=0.7
T[28,3,26]=0.1
T[28,3,27]=0.1
T[28,3,28]=0.1
T[28,4,25]=0.15
T[28,4,26]=0.6
T[28,4,27]=0.15
T[28,4,28]=0.1
T[28,5,25]=0.1
T[28,5,26]=0.1
T[28,5,27]=0.7
T[28,5,28]=0.1
T[29,1,29]=1.0
T[29,2,34]=0.05
T[29,2,28]=0.05
T[29,2,29]=0.9
T[29,3,29]=0.1
T[29,3,30]=0.7
T[29,3,31]=0.1
T[29,3,32]=0.1
T[29,4,29]=0.1
T[29,4,30]=0.15
T[29,4,31]=0.6
T[29,4,32]=0.15
T[29,5,29]=0.1
T[29,5,30]=0.1
T[29,5,31]=0.1
T[29,5,32]=0.7
T[30,1,30]=1.0
T[30,2,34]=0.8
T[30,2,26]=0.025
T[30,2,28]=0.025
T[30,2,30]=0.15
T[30,3,29]=0.1
T[30,3,30]=0.1
T[30,3,31]=0.7
T[30,3,32]=0.1
T[30,4,29]=0.15
T[30,4,30]=0.1
T[30,4,31]=0.15
T[30,4,32]=0.6
T[30,5,29]=0.7
T[30,5,30]=0.1
T[30,5,31]=0.1
T[30,5,32]=0.1
T[31,1,31]=1.0
T[31,2,34]=0.05
T[31,2,28]=0.05
T[31,2,31]=0.9
T[31,3,29]=0.1
T[31,3,30]=0.1
T[31,3,31]=0.1
T[31,3,32]=0.7
T[31,4,29]=0.6
T[31,4,30]=0.15
T[31,4,31]=0.1
T[31,4,32]=0.15
T[31,5,29]=0.1
T[31,5,30]=0.7
T[31,5,31]=0.1
T[31,5,32]=0.1
T[32,1,32]=1.0
T[32,2,34]=0.025
T[32,2,36]=0.025
T[32,2,28]=0.8
T[32,2,32]=0.15
T[32,3,29]=0.7
T[32,3,30]=0.1
T[32,3,31]=0.1
T[32,3,32]=0.1
T[32,4,29]=0.15
T[32,4,30]=0.6
T[32,4,31]=0.15
T[32,4,32]=0.1
T[32,5,29]=0.1
T[32,5,30]=0.1
T[32,5,31]=0.7
T[32,5,32]=0.1
T[33,1,33]=1.0
T[33,2,38]=0.05
T[33,2,57]=0.025
T[33,2,59]=0.025
T[33,2,32]=0.05
T[33,2,33]=0.85
T[33,3,33]=0.1
T[33,3,34]=0.7
T[33,3,35]=0.1
T[33,3,36]=0.1
T[33,4,33]=0.1
T[33,4,34]=0.15
T[33,4,35]=0.6
T[33,4,36]=0.15
T[33,5,33]=0.1
T[33,5,34]=0.1
T[33,5,35]=0.1
T[33,5,36]=0.7
T[34,1,34]=1.0
T[34,2,38]=0.8
T[34,2,59]=0.05
T[34,2,30]=0.025
T[34,2,32]=0.025
T[34,2,34]=0.1
T[34,3,33]=0.1
T[34,3,34]=0.1
T[34,3,35]=0.7
T[34,3,36]=0.1
T[34,4,33]=0.15
T[34,4,34]=0.1
T[34,4,35]=0.15
T[34,4,36]=0.6
T[34,5,33]=0.7
T[34,5,34]=0.1
T[34,5,35]=0.1
T[34,5,36]=0.1
T[35,1,35]=1.0
T[35,2,38]=0.05
T[35,2,59]=0.8
T[35,2,32]=0.05
T[35,2,35]=0.1
T[35,3,33]=0.1
T[35,3,34]=0.1
T[35,3,35]=0.1
T[35,3,36]=0.7
T[35,4,33]=0.6
T[35,4,34]=0.15
T[35,4,35]=0.1
T[35,4,36]=0.15
T[35,5,33]=0.1
T[35,5,34]=0.7
T[35,5,35]=0.1
T[35,5,36]=0.1
T[36,1,36]=1.0
T[36,2,38]=0.025
T[36,2,40]=0.025
T[36,2,59]=0.05
T[36,2,32]=0.8
T[36,2,36]=0.1
T[36,3,33]=0.7
T[36,3,34]=0.1
T[36,3,35]=0.1
T[36,3,36]=0.1
T[36,4,33]=0.15
T[36,4,34]=0.6
T[36,4,35]=0.15
T[36,4,36]=0.1
T[36,5,33]=0.1
T[36,5,34]=0.1
T[36,5,35]=0.7
T[36,5,36]=0.1
T[37,1,37]=1.0
T[37,2,42]=0.05
T[37,2,36]=0.05
T[37,2,37]=0.9
T[37,3,37]=0.1
T[37,3,38]=0.7
T[37,3,39]=0.1
T[37,3,40]=0.1
T[37,4,37]=0.1
T[37,4,38]=0.15
T[37,4,39]=0.6
T[37,4,40]=0.15
T[37,5,37]=0.1
T[37,5,38]=0.1
T[37,5,39]=0.1
T[37,5,40]=0.7
T[38,1,38]=1.0
T[38,2,42]=0.8
T[38,2,34]=0.025
T[38,2,36]=0.025
T[38,2,38]=0.15
T[38,3,37]=0.1
T[38,3,38]=0.1
T[38,3,39]=0.7
T[38,3,40]=0.1
T[38,4,37]=0.15
T[38,4,38]=0.1
T[38,4,39]=0.15
T[38,4,40]=0.6
T[38,5,37]=0.7
T[38,5,38]=0.1
T[38,5,39]=0.1
T[38,5,40]=0.1
T[39,1,39]=1.0
T[39,2,42]=0.05
T[39,2,36]=0.05
T[39,2,39]=0.9
T[39,3,37]=0.1
T[39,3,38]=0.1
T[39,3,39]=0.1
T[39,3,40]=0.7
T[39,4,37]=0.6
T[39,4,38]=0.15
T[39,4,39]=0.1
T[39,4,40]=0.15
T[39,5,37]=0.1
T[39,5,38]=0.7
T[39,5,39]=0.1
T[39,5,40]=0.1
T[40,1,40]=1.0
T[40,2,42]=0.025
T[40,2,44]=0.025
T[40,2,36]=0.8
T[40,2,40]=0.15
T[40,3,37]=0.7
T[40,3,38]=0.1
T[40,3,39]=0.1
T[40,3,40]=0.1
T[40,4,37]=0.15
T[40,4,38]=0.6
T[40,4,39]=0.15
T[40,4,40]=0.1
T[40,5,37]=0.1
T[40,5,38]=0.1
T[40,5,39]=0.7
T[40,5,40]=0.1
T[41,1,41]=1.0
T[41,2,40]=0.05
T[41,2,41]=0.95
T[41,3,41]=0.1
T[41,3,42]=0.7
T[41,3,43]=0.1
T[41,3,44]=0.1
T[41,4,41]=0.1
T[41,4,42]=0.15
T[41,4,43]=0.6
T[41,4,44]=0.15
T[41,5,41]=0.1
T[41,5,42]=0.1
T[41,5,43]=0.1
T[41,5,44]=0.7
T[42,1,42]=1.0
T[42,2,38]=0.025
T[42,2,40]=0.025
T[42,2,42]=0.95
T[42,3,41]=0.1
T[42,3,42]=0.1
T[42,3,43]=0.7
T[42,3,44]=0.1
T[42,4,41]=0.15
T[42,4,42]=0.1
T[42,4,43]=0.15
T[42,4,44]=0.6
T[42,5,41]=0.7
T[42,5,42]=0.1
T[42,5,43]=0.1
T[42,5,44]=0.1
T[43,1,43]=1.0
T[43,2,40]=0.05
T[43,2,43]=0.95
T[43,3,41]=0.1
T[43,3,42]=0.1
T[43,3,43]=0.1
T[43,3,44]=0.7
T[43,4,41]=0.6
T[43,4,42]=0.15
T[43,4,43]=0.1
T[43,4,44]=0.15
T[43,5,41]=0.1
T[43,5,42]=0.7
T[43,5,43]=0.1
T[43,5,44]=0.1
T[44,1,44]=1.0
T[44,2,40]=0.8
T[44,2,44]=0.2
T[44,3,41]=0.7
T[44,3,42]=0.1
T[44,3,43]=0.1
T[44,3,44]=0.1
T[44,4,41]=0.15
T[44,4,42]=0.6
T[44,4,43]=0.15
T[44,4,44]=0.1
T[44,5,41]=0.1
T[44,5,42]=0.1
T[44,5,43]=0.7
T[44,5,44]=0.1
T[45,1,45]=1.0
T[45,2,9]=0.8
T[45,2,45]=0.2
T[45,3,45]=0.1
T[45,3,46]=0.7
T[45,3,47]=0.1
T[45,3,48]=0.1
T[45,4,45]=0.1
T[45,4,46]=0.15
T[45,4,47]=0.6
T[45,4,48]=0.15
T[45,5,45]=0.1
T[45,5,46]=0.1
T[45,5,47]=0.1
T[45,5,48]=0.7
T[46,1,46]=1.0
T[46,2,9]=0.05
T[46,2,46]=0.95
T[46,3,45]=0.1
T[46,3,46]=0.1
T[46,3,47]=0.7
T[46,3,48]=0.1
T[46,4,45]=0.15
T[46,4,46]=0.1
T[46,4,47]=0.15
T[46,4,48]=0.6
T[46,5,45]=0.7
T[46,5,46]=0.1
T[46,5,47]=0.1
T[46,5,48]=0.1
T[47,1,47]=1.0
T[47,2,9]=0.025
T[47,2,11]=0.025
T[47,2,47]=0.95
T[47,3,45]=0.1
T[47,3,46]=0.1
T[47,3,47]=0.1
T[47,3,48]=0.7
T[47,4,45]=0.6
T[47,4,46]=0.15
T[47,4,47]=0.1
T[47,4,48]=0.15
T[47,5,45]=0.1
T[47,5,46]=0.7
T[47,5,47]=0.1
T[47,5,48]=0.1
T[48,1,48]=1.0
T[48,2,9]=0.05
T[48,2,48]=0.95
T[48,3,45]=0.7
T[48,3,46]=0.1
T[48,3,47]=0.1
T[48,3,48]=0.1
T[48,4,45]=0.15
T[48,4,46]=0.6
T[48,4,47]=0.15
T[48,4,48]=0.1
T[48,5,45]=0.1
T[48,5,46]=0.1
T[48,5,47]=0.7
T[48,5,48]=0.1
T[49,1,49]=1.0
T[49,2,17]=0.8
T[49,2,49]=0.2
T[49,3,49]=0.1
T[49,3,50]=0.7
T[49,3,51]=0.1
T[49,3,52]=0.1
T[49,4,49]=0.1
T[49,4,50]=0.15
T[49,4,51]=0.6
T[49,4,52]=0.15
T[49,5,49]=0.1
T[49,5,50]=0.1
T[49,5,51]=0.1
T[49,5,52]=0.7
T[50,1,50]=1.0
T[50,2,17]=0.05
T[50,2,50]=0.95
T[50,3,49]=0.1
T[50,3,50]=0.1
T[50,3,51]=0.7
T[50,3,52]=0.1
T[50,4,49]=0.15
T[50,4,50]=0.1
T[50,4,51]=0.15
T[50,4,52]=0.6
T[50,5,49]=0.7
T[50,5,50]=0.1
T[50,5,51]=0.1
T[50,5,52]=0.1
T[51,1,51]=1.0
T[51,2,17]=0.025
T[51,2,19]=0.025
T[51,2,51]=0.95
T[51,3,49]=0.1
T[51,3,50]=0.1
T[51,3,51]=0.1
T[51,3,52]=0.7
T[51,4,49]=0.6
T[51,4,50]=0.15
T[51,4,51]=0.1
T[51,4,52]=0.15
T[51,5,49]=0.1
T[51,5,50]=0.7
T[51,5,51]=0.1
T[51,5,52]=0.1
T[52,1,52]=1.0
T[52,2,17]=0.05
T[52,2,52]=0.95
T[52,3,49]=0.7
T[52,3,50]=0.1
T[52,3,51]=0.1
T[52,3,52]=0.1
T[52,4,49]=0.15
T[52,4,50]=0.6
T[52,4,51]=0.15
T[52,4,52]=0.1
T[52,5,49]=0.1
T[52,5,50]=0.1
T[52,5,51]=0.7
T[52,5,52]=0.1
T[53,1,53]=1.0
T[53,2,25]=0.8
T[53,2,53]=0.2
T[53,3,53]=0.1
T[53,3,54]=0.7
T[53,3,55]=0.1
T[53,3,56]=0.1
T[53,4,53]=0.1
T[53,4,54]=0.15
T[53,4,55]=0.6
T[53,4,56]=0.15
T[53,5,53]=0.1
T[53,5,54]=0.1
T[53,5,55]=0.1
T[53,5,56]=0.7
T[54,1,54]=1.0
T[54,2,25]=0.05
T[54,2,54]=0.95
T[54,3,53]=0.1
T[54,3,54]=0.1
T[54,3,55]=0.7
T[54,3,56]=0.1
T[54,4,53]=0.15
T[54,4,54]=0.1
T[54,4,55]=0.15
T[54,4,56]=0.6
T[54,5,53]=0.7
T[54,5,54]=0.1
T[54,5,55]=0.1
T[54,5,56]=0.1
T[55,1,55]=1.0
T[55,2,25]=0.025
T[55,2,27]=0.025
T[55,2,55]=0.95
T[55,3,53]=0.1
T[55,3,54]=0.1
T[55,3,55]=0.1
T[55,3,56]=0.7
T[55,4,53]=0.6
T[55,4,54]=0.15
T[55,4,55]=0.1
T[55,4,56]=0.15
T[55,5,53]=0.1
T[55,5,54]=0.7
T[55,5,55]=0.1
T[55,5,56]=0.1
T[56,1,56]=1.0
T[56,2,25]=0.05
T[56,2,56]=0.95
T[56,3,53]=0.7
T[56,3,54]=0.1
T[56,3,55]=0.1
T[56,3,56]=0.1
T[56,4,53]=0.15
T[56,4,54]=0.6
T[56,4,55]=0.15
T[56,4,56]=0.1
T[56,5,53]=0.1
T[56,5,54]=0.1
T[56,5,55]=0.7
T[56,5,56]=0.1

state_60_tr = [0.017865 0.017857 0.017857 0.017857 0.017857 0.017857 0.017857 0.017857 0.017857 0.017857 0.017857 0.017857 0.017857 0.017857 0.017857 0.017857 0.017857 0.017857 0.017857 0.017857 0.017857 0.017857 0.017857 0.017857 0.017857 0.017857 0.017857 0.017857 0.017857 0.017857 0.017857 0.017857 0.017857 0.017857 0.017857 0.017857 0.017857 0.017857 0.017857 0.017857 0.017857 0.017857 0.017857 0.017857 0.017857 0.017857 0.017857 0.017857 0.017857 0.017857 0.017857 0.017857 0.017857 0.017857 0.017857 0.017857 0.0 0.0 0.0 0.0]

for a = 1 : 5
    T[57,a,:] = state_60_tr
    T[58,a,:] = state_60_tr
    T[59,a,:] = state_60_tr
    T[60,a,:] = state_60_tr
end


O_data = [
[0.000949 0.008549 0.008549 0.076949 0.000049 0.000449 0.000449 0.004049 0.008549 0.076949 0.076949 0.692550 0.000449 0.004049 0.004049 0.036464 0.0 0.0 0.0 0.0 0.0];
[0.000949 0.008549 0.008549 0.076949 0.008549 0.076949 0.076949 0.692550 0.000049 0.000449 0.000449 0.004049 0.000449 0.004049 0.004049 0.036464 0.0 0.0 0.0 0.0 0.0];
[0.000949 0.000049 0.008549 0.000449 0.008549 0.000449 0.076949 0.004049 0.008549 0.000449 0.076949 0.004049 0.076949 0.004049 0.692550 0.036464 0.0 0.0 0.0 0.0 0.0];
[0.000949 0.008549 0.000049 0.000449 0.008549 0.076949 0.000449 0.004049 0.008549 0.076949 0.000449 0.004049 0.076949 0.692550 0.004049 0.036464 0.0 0.0 0.0 0.0 0.0];
[0.009024 0.000474 0.081225 0.004275 0.000474 0.000024 0.004275 0.000225 0.081225 0.004275 0.731024 0.038475 0.004275 0.000225 0.038475 0.002030 0.0 0.0 0.0 0.0 0.0];
[0.009024 0.081225 0.000474 0.004275 0.081225 0.731024 0.004275 0.038475 0.000474 0.004275 0.000024 0.000225 0.004275 0.038475 0.000225 0.002030 0.0 0.0 0.0 0.0 0.0];
[0.009024 0.000474 0.081225 0.004275 0.000474 0.000024 0.004275 0.000225 0.081225 0.004275 0.731024 0.038475 0.004275 0.000225 0.038475 0.002030 0.0 0.0 0.0 0.0 0.0];
[0.009024 0.081225 0.000474 0.004275 0.081225 0.731024 0.004275 0.038475 0.000474 0.004275 0.000024 0.000225 0.004275 0.038475 0.000225 0.002030 0.0 0.0 0.0 0.0 0.0];
[0.085737 0.004512 0.004512 0.000237 0.004512 0.000237 0.000237 0.000012 0.771637 0.040612 0.040612 0.002137 0.040612 0.002137 0.002137 0.000120 0.0 0.0 0.0 0.0 0.0];
[0.085737 0.771637 0.004512 0.040612 0.004512 0.040612 0.000237 0.002137 0.004512 0.040612 0.000237 0.002137 0.000237 0.002137 0.000012 0.000120 0.0 0.0 0.0 0.0 0.0];
    [0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0];
[0.085737 0.004512 0.004512 0.000237 0.771637 0.040612 0.040612 0.002137 0.004512 0.000237 0.000237 0.000012 0.040612 0.002137 0.002137 0.000120 0.0 0.0 0.0 0.0 0.0];
[0.009024 0.000474 0.081225 0.004275 0.000474 0.000024 0.004275 0.000225 0.081225 0.004275 0.731024 0.038475 0.004275 0.000225 0.038475 0.002030 0.0 0.0 0.0 0.0 0.0];
[0.009024 0.081225 0.000474 0.004275 0.081225 0.731024 0.004275 0.038475 0.000474 0.004275 0.000024 0.000225 0.004275 0.038475 0.000225 0.002030 0.0 0.0 0.0 0.0 0.0];
[0.009024 0.000474 0.081225 0.004275 0.000474 0.000024 0.004275 0.000225 0.081225 0.004275 0.731024 0.038475 0.004275 0.000225 0.038475 0.002030 0.0 0.0 0.0 0.0 0.0];
[0.009024 0.081225 0.000474 0.004275 0.081225 0.731024 0.004275 0.038475 0.000474 0.004275 0.000024 0.000225 0.004275 0.038475 0.000225 0.002030 0.0 0.0 0.0 0.0 0.0];
[0.085737 0.004512 0.004512 0.000237 0.004512 0.000237 0.000237 0.000012 0.771637 0.040612 0.040612 0.002137 0.040612 0.002137 0.002137 0.000120 0.0 0.0 0.0 0.0 0.0];
[0.085737 0.771637 0.004512 0.040612 0.004512 0.040612 0.000237 0.002137 0.004512 0.040612 0.000237 0.002137 0.000237 0.002137 0.000012 0.000120 0.0 0.0 0.0 0.0 0.0];
    [0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0];
[0.085737 0.004512 0.004512 0.000237 0.771637 0.040612 0.040612 0.002137 0.004512 0.000237 0.000237 0.000012 0.040612 0.002137 0.002137 0.000120 0.0 0.0 0.0 0.0 0.0];
[0.009024 0.000474 0.081225 0.004275 0.000474 0.000024 0.004275 0.000225 0.081225 0.004275 0.731024 0.038475 0.004275 0.000225 0.038475 0.002030 0.0 0.0 0.0 0.0 0.0];
[0.009024 0.081225 0.000474 0.004275 0.081225 0.731024 0.004275 0.038475 0.000474 0.004275 0.000024 0.000225 0.004275 0.038475 0.000225 0.002030 0.0 0.0 0.0 0.0 0.0];
[0.009024 0.000474 0.081225 0.004275 0.000474 0.000024 0.004275 0.000225 0.081225 0.004275 0.731024 0.038475 0.004275 0.000225 0.038475 0.002030 0.0 0.0 0.0 0.0 0.0];
[0.009024 0.081225 0.000474 0.004275 0.081225 0.731024 0.004275 0.038475 0.000474 0.004275 0.000024 0.000225 0.004275 0.038475 0.000225 0.002030 0.0 0.0 0.0 0.0 0.0];
[0.085737 0.004512 0.004512 0.000237 0.004512 0.000237 0.000237 0.000012 0.771637 0.040612 0.040612 0.002137 0.040612 0.002137 0.002137 0.000120 0.0 0.0 0.0 0.0 0.0];
[0.085737 0.771637 0.004512 0.040612 0.004512 0.040612 0.000237 0.002137 0.004512 0.040612 0.000237 0.002137 0.000237 0.002137 0.000012 0.000120 0.0 0.0 0.0 0.0 0.0];
    [0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0];
[0.085737 0.004512 0.004512 0.000237 0.771637 0.040612 0.040612 0.002137 0.004512 0.000237 0.000237 0.000012 0.040612 0.002137 0.002137 0.000120 0.0 0.0 0.0 0.0 0.0];
[0.009024 0.000474 0.081225 0.004275 0.000474 0.000024 0.004275 0.000225 0.081225 0.004275 0.731024 0.038475 0.004275 0.000225 0.038475 0.002030 0.0 0.0 0.0 0.0 0.0];
[0.009024 0.081225 0.000474 0.004275 0.081225 0.731024 0.004275 0.038475 0.000474 0.004275 0.000024 0.000225 0.004275 0.038475 0.000225 0.002030 0.0 0.0 0.0 0.0 0.0];
[0.009024 0.000474 0.081225 0.004275 0.000474 0.000024 0.004275 0.000225 0.081225 0.004275 0.731024 0.038475 0.004275 0.000225 0.038475 0.002030 0.0 0.0 0.0 0.0 0.0];
[0.009024 0.081225 0.000474 0.004275 0.081225 0.731024 0.004275 0.038475 0.000474 0.004275 0.000024 0.000225 0.004275 0.038475 0.000225 0.002030 0.0 0.0 0.0 0.0 0.0];
[0.085737 0.004512 0.004512 0.000237 0.004512 0.000237 0.000237 0.000012 0.771637 0.040612 0.040612 0.002137 0.040612 0.002137 0.002137 0.000120 0.0 0.0 0.0 0.0 0.0];
[0.085737 0.771637 0.004512 0.040612 0.004512 0.040612 0.000237 0.002137 0.004512 0.040612 0.000237 0.002137 0.000237 0.002137 0.000012 0.000120 0.0 0.0 0.0 0.0 0.0];
    [0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0];
[0.085737 0.004512 0.004512 0.000237 0.771637 0.040612 0.040612 0.002137 0.004512 0.000237 0.000237 0.000012 0.040612 0.002137 0.002137 0.000120 0.0 0.0 0.0 0.0 0.0];
[0.009024 0.000474 0.081225 0.004275 0.000474 0.000024 0.004275 0.000225 0.081225 0.004275 0.731024 0.038475 0.004275 0.000225 0.038475 0.002030 0.0 0.0 0.0 0.0 0.0];
[0.009024 0.081225 0.000474 0.004275 0.081225 0.731024 0.004275 0.038475 0.000474 0.004275 0.000024 0.000225 0.004275 0.038475 0.000225 0.002030 0.0 0.0 0.0 0.0 0.0];
[0.009024 0.000474 0.081225 0.004275 0.000474 0.000024 0.004275 0.000225 0.081225 0.004275 0.731024 0.038475 0.004275 0.000225 0.038475 0.002030 0.0 0.0 0.0 0.0 0.0];
[0.009024 0.081225 0.000474 0.004275 0.081225 0.731024 0.004275 0.038475 0.000474 0.004275 0.000024 0.000225 0.004275 0.038475 0.000225 0.002030 0.0 0.0 0.0 0.0 0.0];
[0.000949 0.000049 0.008549 0.000449 0.008549 0.000449 0.076949 0.004049 0.008549 0.000449 0.076949 0.004049 0.076949 0.004049 0.692550 0.036464 0.0 0.0 0.0 0.0 0.0];
[0.000949 0.008549 0.000049 0.000449 0.008549 0.076949 0.000449 0.004049 0.008549 0.076949 0.000449 0.004049 0.076949 0.692550 0.004049 0.036464 0.0 0.0 0.0 0.0 0.0];
[0.000949 0.008549 0.008549 0.076949 0.000049 0.000449 0.000449 0.004049 0.008549 0.076949 0.076949 0.692550 0.000449 0.004049 0.004049 0.036464 0.0 0.0 0.0 0.0 0.0];
[0.000949 0.008549 0.008549 0.076949 0.008549 0.076949 0.076949 0.692550 0.000049 0.000449 0.000449 0.004049 0.000449 0.004049 0.004049 0.036464 0.0 0.0 0.0 0.0 0.0];
[0.000949 0.008549 0.008549 0.076949 0.008549 0.076949 0.076949 0.692550 0.000049 0.000449 0.000449 0.004049 0.000449 0.004049 0.004049 0.036464 0.0 0.0 0.0 0.0 0.0];
[0.000949 0.000049 0.008549 0.000449 0.008549 0.000449 0.076949 0.004049 0.008549 0.000449 0.076949 0.004049 0.076949 0.004049 0.692550 0.036464 0.0 0.0 0.0 0.0 0.0];
[0.000949 0.008549 0.000049 0.000449 0.008549 0.076949 0.000449 0.004049 0.008549 0.076949 0.000449 0.004049 0.076949 0.692550 0.004049 0.036464 0.0 0.0 0.0 0.0 0.0];
[0.000949 0.008549 0.008549 0.076949 0.000049 0.000449 0.000449 0.004049 0.008549 0.076949 0.076949 0.692550 0.000449 0.004049 0.004049 0.036464 0.0 0.0 0.0 0.0 0.0];
[0.000949 0.008549 0.008549 0.076949 0.008549 0.076949 0.076949 0.692550 0.000049 0.000449 0.000449 0.004049 0.000449 0.004049 0.004049 0.036464 0.0 0.0 0.0 0.0 0.0];
[0.000949 0.000049 0.008549 0.000449 0.008549 0.000449 0.076949 0.004049 0.008549 0.000449 0.076949 0.004049 0.076949 0.004049 0.692550 0.036464 0.0 0.0 0.0 0.0 0.0];
[0.000949 0.008549 0.000049 0.000449 0.008549 0.076949 0.000449 0.004049 0.008549 0.076949 0.000449 0.004049 0.076949 0.692550 0.004049 0.036464 0.0 0.0 0.0 0.0 0.0];
[0.000949 0.008549 0.008549 0.076949 0.000049 0.000449 0.000449 0.004049 0.008549 0.076949 0.076949 0.692550 0.000449 0.004049 0.004049 0.036464 0.0 0.0 0.0 0.0 0.0];
[0.000949 0.008549 0.008549 0.076949 0.008549 0.076949 0.076949 0.692550 0.000049 0.000449 0.000449 0.004049 0.000449 0.004049 0.004049 0.036464 0.0 0.0 0.0 0.0 0.0];
[0.000949 0.000049 0.008549 0.000449 0.008549 0.000449 0.076949 0.004049 0.008549 0.000449 0.076949 0.004049 0.076949 0.004049 0.692550 0.036464 0.0 0.0 0.0 0.0 0.0];
[0.000949 0.008549 0.000049 0.000449 0.008549 0.076949 0.000449 0.004049 0.008549 0.076949 0.000449 0.004049 0.076949 0.692550 0.004049 0.036464 0.0 0.0 0.0 0.0 0.0];
[0.000949 0.008549 0.008549 0.076949 0.000049 0.000449 0.000449 0.004049 0.008549 0.076949 0.076949 0.692550 0.000449 0.004049 0.004049 0.036464 0.0 0.0 0.0 0.0 0.0];
    [0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0];
    [0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0];
    [0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0];
    [0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0]
]

O = zeros(Float64,60,5,21)
for i = 1 : 60
    for a = 1 : 5
        O[i,a,:] = O_data[i,:]
    end
end

R= zeros(Float64,60,5,60)
for s = 1 : 60
    for a = 1 : 5
        R[s,a,57] = 1
        R[s,a,58] = 1
        R[s,a,59] = 1
        R[s,a,60] = 1
    end
end

#############################

gamma = 0.95

@time Q_MDP = Q_value_iteration(zeros(Float64,60,5),T,R,0.01,gamma)
@time Q_UMDP = QUMDP(zeros(Float64,60,5),T,R,0.01,gamma)
@time Q_FIB = FIB(zeros(Float64,60,5),T,R,O,0.01,gamma)
@time Q_M3 = purely_iteration_v3(zeros(Float64,60,5),T,R,O,0.01,gamma)
@time Q_M5 = purely_iteration_v5(zeros(Float64,60,5),T,R,O,0.01,gamma)
@time Q_M6 = purely_iteration_v6(zeros(Float64,60,5),T,R,O,0.01,gamma)
@time Q_M7 = purely_iteration_v7(zeros(Float64,60,5),T,R,O,0.01,gamma)

function one_hallway60_trial(T,R,O,t_step,alpha,gamma)

    # initial belief
    b = zeros(Float64,60)
    for i = 1 : 56
        b[i] = 1/56
    end

    # Initialize state
    x = round(Int64,rand()*56) + 1

    # intialize total reward
    total_r = 0

    for t = 1 : t_step

        # Choose the action
        action_to_do = action_to_take(b,alpha)

        # Get reward and the next state
        (xp,r) = tran_reward_sampling(T,R,x,action_to_do)
        total_r += r * ((gamma)^t)

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

    return (total_r)# / t_step)

end

function one_hallway60_trial_2(T,R,O,t_step,alpha,gamma)

    # initial belief
    b = zeros(Float64,60)
    b[1] = 1.0

    # Initialize state
    x = 1

    # intialize total reward
    total_r = 0

    for t = 1 : t_step

        # Choose the action
        action_to_do = action_to_take(b,alpha)

        # Get reward and the next state
        (xp,r) = tran_reward_sampling(T,R,x,action_to_do)
        total_r += r * ((gamma)^t)

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

    return (total_r)# / t_step)

end

QMDP_r_sum = 0
QUMDP_r_sum = 0
FIB_r_sum = 0
MY_3_r_sum = 0
MY_5_r_sum = 0
MY_6_r_sum = 0
MY_7_r_sum = 0


t_trial = 1000
t_step = 300
for i = 1 : t_trial
    if (i%100 == 0); println("trial = ",i); end
    QMDP_r_sum += one_hallway60_trial(T,R,O,t_step,Q_MDP,1)#gamma)
    QUMDP_r_sum += one_hallway60_trial(T,R,O,t_step,Q_UMDP,1)#gamma)
    FIB_r_sum += one_hallway60_trial(T,R,O,t_step,Q_FIB,1)#gamma)
    MY_3_r_sum += one_hallway60_trial(T,R,O,t_step,Q_M3,1)#gamma)
    MY_5_r_sum += one_hallway60_trial(T,R,O,t_step,Q_M5,1)#gamma)
    MY_6_r_sum += one_hallway60_trial(T,R,O,t_step,Q_M6,1)#gamma)
    MY_7_r_sum += one_hallway60_trial(T,R,O,t_step,Q_M7,1)#gamma)

end

#for i = 1 : t_trial
#    if (i%100 == 0); println("trial = ",i); end
#    QMDP_r_sum += one_hallway60_trial_2(T,R,O,t_step,Q_MDP,gamma)
#    QUMDP_r_sum += one_hallway60_trial_2(T,R,O,t_step,Q_UMDP,gamma)
#    FIB_r_sum += one_hallway60_trial_2(T,R,O,t_step,Q_FIB,gamma)
#    MY_1_r_sum += one_hallway60_trial_2(T,R,O,t_step,Q_M1,gamma)
#    MY_2_r_sum += one_hallway60_trial_2(T,R,O,t_step,Q_M2,gamma)
#    MY_3_r_sum += one_hallway60_trial_2(T,R,O,t_step,Q_M3,gamma)
#    MY_4_r_sum += one_hallway60_trial_2(T,R,O,t_step,Q_M4,gamma)
#    MY_5_r_sum += one_hallway60_trial_2(T,R,O,t_step,Q_M5,gamma)
#end


println(QMDP_r_sum/t_trial)
println(QUMDP_r_sum/t_trial)
println(FIB_r_sum/t_trial)
println(MY_3_r_sum/t_trial)
println(MY_5_r_sum/t_trial)
println(MY_7_r_sum/t_trial)
println(MY_6_r_sum/t_trial)

