??(
??
B
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??!
?
conv2d_136/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_136/kernel

%conv2d_136/kernel/Read/ReadVariableOpReadVariableOpconv2d_136/kernel*&
_output_shapes
:*
dtype0
v
conv2d_136/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_136/bias
o
#conv2d_136/bias/Read/ReadVariableOpReadVariableOpconv2d_136/bias*
_output_shapes
:*
dtype0
?
batch_normalization_46/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_46/gamma
?
0batch_normalization_46/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_46/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_46/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_46/beta
?
/batch_normalization_46/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_46/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_46/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_46/moving_mean
?
6batch_normalization_46/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_46/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_46/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_46/moving_variance
?
:batch_normalization_46/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_46/moving_variance*
_output_shapes
:*
dtype0
?
conv2d_137/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_137/kernel

%conv2d_137/kernel/Read/ReadVariableOpReadVariableOpconv2d_137/kernel*&
_output_shapes
:*
dtype0
v
conv2d_137/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_137/bias
o
#conv2d_137/bias/Read/ReadVariableOpReadVariableOpconv2d_137/bias*
_output_shapes
:*
dtype0
?
batch_normalization_47/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_47/gamma
?
0batch_normalization_47/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_47/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_47/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_47/beta
?
/batch_normalization_47/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_47/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_47/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_47/moving_mean
?
6batch_normalization_47/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_47/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_47/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_47/moving_variance
?
:batch_normalization_47/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_47/moving_variance*
_output_shapes
:*
dtype0
?
conv2d_138/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_138/kernel

%conv2d_138/kernel/Read/ReadVariableOpReadVariableOpconv2d_138/kernel*&
_output_shapes
:*
dtype0
v
conv2d_138/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_138/bias
o
#conv2d_138/bias/Read/ReadVariableOpReadVariableOpconv2d_138/bias*
_output_shapes
:*
dtype0
?
conv2d_139/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_139/kernel

%conv2d_139/kernel/Read/ReadVariableOpReadVariableOpconv2d_139/kernel*&
_output_shapes
:*
dtype0
v
conv2d_139/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_139/bias
o
#conv2d_139/bias/Read/ReadVariableOpReadVariableOpconv2d_139/bias*
_output_shapes
:*
dtype0
?
batch_normalization_48/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_48/gamma
?
0batch_normalization_48/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_48/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_48/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_48/beta
?
/batch_normalization_48/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_48/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_48/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_48/moving_mean
?
6batch_normalization_48/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_48/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_48/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_48/moving_variance
?
:batch_normalization_48/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_48/moving_variance*
_output_shapes
:*
dtype0
?
conv2d_140/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_140/kernel

%conv2d_140/kernel/Read/ReadVariableOpReadVariableOpconv2d_140/kernel*&
_output_shapes
:*
dtype0
v
conv2d_140/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_140/bias
o
#conv2d_140/bias/Read/ReadVariableOpReadVariableOpconv2d_140/bias*
_output_shapes
:*
dtype0
?
batch_normalization_49/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_49/gamma
?
0batch_normalization_49/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_49/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_49/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_49/beta
?
/batch_normalization_49/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_49/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_49/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_49/moving_mean
?
6batch_normalization_49/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_49/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_49/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_49/moving_variance
?
:batch_normalization_49/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_49/moving_variance*
_output_shapes
:*
dtype0
?
conv2d_141/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_141/kernel

%conv2d_141/kernel/Read/ReadVariableOpReadVariableOpconv2d_141/kernel*&
_output_shapes
:*
dtype0
v
conv2d_141/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_141/bias
o
#conv2d_141/bias/Read/ReadVariableOpReadVariableOpconv2d_141/bias*
_output_shapes
:*
dtype0
?
conv2d_142/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_142/kernel

%conv2d_142/kernel/Read/ReadVariableOpReadVariableOpconv2d_142/kernel*&
_output_shapes
:*
dtype0
v
conv2d_142/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_142/bias
o
#conv2d_142/bias/Read/ReadVariableOpReadVariableOpconv2d_142/bias*
_output_shapes
:*
dtype0
?
batch_normalization_50/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_50/gamma
?
0batch_normalization_50/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_50/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_50/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_50/beta
?
/batch_normalization_50/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_50/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_50/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_50/moving_mean
?
6batch_normalization_50/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_50/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_50/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_50/moving_variance
?
:batch_normalization_50/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_50/moving_variance*
_output_shapes
:*
dtype0
?
conv2d_143/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_143/kernel

%conv2d_143/kernel/Read/ReadVariableOpReadVariableOpconv2d_143/kernel*&
_output_shapes
:*
dtype0
v
conv2d_143/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_143/bias
o
#conv2d_143/bias/Read/ReadVariableOpReadVariableOpconv2d_143/bias*
_output_shapes
:*
dtype0
?
batch_normalization_51/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_51/gamma
?
0batch_normalization_51/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_51/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_51/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_51/beta
?
/batch_normalization_51/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_51/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_51/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_51/moving_mean
?
6batch_normalization_51/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_51/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_51/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_51/moving_variance
?
:batch_normalization_51/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_51/moving_variance*
_output_shapes
:*
dtype0
?
conv2d_144/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_144/kernel

%conv2d_144/kernel/Read/ReadVariableOpReadVariableOpconv2d_144/kernel*&
_output_shapes
:*
dtype0
v
conv2d_144/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_144/bias
o
#conv2d_144/bias/Read/ReadVariableOpReadVariableOpconv2d_144/bias*
_output_shapes
:*
dtype0
z
dense_58/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_58/kernel
s
#dense_58/kernel/Read/ReadVariableOpReadVariableOpdense_58/kernel*
_output_shapes

: *
dtype0
r
dense_58/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_58/bias
k
!dense_58/bias/Read/ReadVariableOpReadVariableOpdense_58/bias*
_output_shapes
: *
dtype0
z
dense_59/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_59/kernel
s
#dense_59/kernel/Read/ReadVariableOpReadVariableOpdense_59/kernel*
_output_shapes

: *
dtype0
r
dense_59/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_59/bias
k
!dense_59/bias/Read/ReadVariableOpReadVariableOpdense_59/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/conv2d_136/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_136/kernel/m
?
,Adam/conv2d_136/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_136/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_136/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_136/bias/m
}
*Adam/conv2d_136/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_136/bias/m*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_46/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_46/gamma/m
?
7Adam/batch_normalization_46/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_46/gamma/m*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_46/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_46/beta/m
?
6Adam/batch_normalization_46/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_46/beta/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_137/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_137/kernel/m
?
,Adam/conv2d_137/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_137/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_137/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_137/bias/m
}
*Adam/conv2d_137/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_137/bias/m*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_47/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_47/gamma/m
?
7Adam/batch_normalization_47/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_47/gamma/m*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_47/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_47/beta/m
?
6Adam/batch_normalization_47/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_47/beta/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_138/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_138/kernel/m
?
,Adam/conv2d_138/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_138/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_138/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_138/bias/m
}
*Adam/conv2d_138/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_138/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_139/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_139/kernel/m
?
,Adam/conv2d_139/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_139/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_139/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_139/bias/m
}
*Adam/conv2d_139/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_139/bias/m*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_48/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_48/gamma/m
?
7Adam/batch_normalization_48/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_48/gamma/m*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_48/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_48/beta/m
?
6Adam/batch_normalization_48/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_48/beta/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_140/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_140/kernel/m
?
,Adam/conv2d_140/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_140/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_140/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_140/bias/m
}
*Adam/conv2d_140/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_140/bias/m*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_49/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_49/gamma/m
?
7Adam/batch_normalization_49/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_49/gamma/m*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_49/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_49/beta/m
?
6Adam/batch_normalization_49/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_49/beta/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_141/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_141/kernel/m
?
,Adam/conv2d_141/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_141/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_141/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_141/bias/m
}
*Adam/conv2d_141/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_141/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_142/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_142/kernel/m
?
,Adam/conv2d_142/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_142/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_142/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_142/bias/m
}
*Adam/conv2d_142/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_142/bias/m*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_50/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_50/gamma/m
?
7Adam/batch_normalization_50/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_50/gamma/m*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_50/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_50/beta/m
?
6Adam/batch_normalization_50/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_50/beta/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_143/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_143/kernel/m
?
,Adam/conv2d_143/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_143/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_143/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_143/bias/m
}
*Adam/conv2d_143/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_143/bias/m*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_51/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_51/gamma/m
?
7Adam/batch_normalization_51/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_51/gamma/m*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_51/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_51/beta/m
?
6Adam/batch_normalization_51/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_51/beta/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_144/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_144/kernel/m
?
,Adam/conv2d_144/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_144/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_144/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_144/bias/m
}
*Adam/conv2d_144/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_144/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_58/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_58/kernel/m
?
*Adam/dense_58/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_58/kernel/m*
_output_shapes

: *
dtype0
?
Adam/dense_58/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_58/bias/m
y
(Adam/dense_58/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_58/bias/m*
_output_shapes
: *
dtype0
?
Adam/dense_59/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_59/kernel/m
?
*Adam/dense_59/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_59/kernel/m*
_output_shapes

: *
dtype0
?
Adam/dense_59/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_59/bias/m
y
(Adam/dense_59/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_59/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_136/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_136/kernel/v
?
,Adam/conv2d_136/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_136/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_136/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_136/bias/v
}
*Adam/conv2d_136/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_136/bias/v*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_46/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_46/gamma/v
?
7Adam/batch_normalization_46/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_46/gamma/v*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_46/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_46/beta/v
?
6Adam/batch_normalization_46/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_46/beta/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_137/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_137/kernel/v
?
,Adam/conv2d_137/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_137/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_137/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_137/bias/v
}
*Adam/conv2d_137/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_137/bias/v*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_47/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_47/gamma/v
?
7Adam/batch_normalization_47/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_47/gamma/v*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_47/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_47/beta/v
?
6Adam/batch_normalization_47/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_47/beta/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_138/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_138/kernel/v
?
,Adam/conv2d_138/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_138/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_138/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_138/bias/v
}
*Adam/conv2d_138/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_138/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_139/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_139/kernel/v
?
,Adam/conv2d_139/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_139/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_139/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_139/bias/v
}
*Adam/conv2d_139/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_139/bias/v*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_48/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_48/gamma/v
?
7Adam/batch_normalization_48/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_48/gamma/v*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_48/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_48/beta/v
?
6Adam/batch_normalization_48/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_48/beta/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_140/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_140/kernel/v
?
,Adam/conv2d_140/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_140/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_140/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_140/bias/v
}
*Adam/conv2d_140/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_140/bias/v*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_49/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_49/gamma/v
?
7Adam/batch_normalization_49/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_49/gamma/v*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_49/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_49/beta/v
?
6Adam/batch_normalization_49/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_49/beta/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_141/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_141/kernel/v
?
,Adam/conv2d_141/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_141/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_141/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_141/bias/v
}
*Adam/conv2d_141/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_141/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_142/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_142/kernel/v
?
,Adam/conv2d_142/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_142/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_142/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_142/bias/v
}
*Adam/conv2d_142/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_142/bias/v*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_50/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_50/gamma/v
?
7Adam/batch_normalization_50/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_50/gamma/v*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_50/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_50/beta/v
?
6Adam/batch_normalization_50/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_50/beta/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_143/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_143/kernel/v
?
,Adam/conv2d_143/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_143/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_143/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_143/bias/v
}
*Adam/conv2d_143/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_143/bias/v*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_51/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_51/gamma/v
?
7Adam/batch_normalization_51/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_51/gamma/v*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_51/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_51/beta/v
?
6Adam/batch_normalization_51/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_51/beta/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_144/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_144/kernel/v
?
,Adam/conv2d_144/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_144/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_144/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_144/bias/v
}
*Adam/conv2d_144/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_144/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_58/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_58/kernel/v
?
*Adam/dense_58/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_58/kernel/v*
_output_shapes

: *
dtype0
?
Adam/dense_58/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_58/bias/v
y
(Adam/dense_58/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_58/bias/v*
_output_shapes
: *
dtype0
?
Adam/dense_59/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_59/kernel/v
?
*Adam/dense_59/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_59/kernel/v*
_output_shapes

: *
dtype0
?
Adam/dense_59/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_59/bias/v
y
(Adam/dense_59/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_59/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer-6
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
layer_with_weights-8
layer-11
layer_with_weights-9
layer-12
layer-13
layer-14
layer_with_weights-10
layer-15
layer_with_weights-11
layer-16
layer_with_weights-12
layer-17
layer_with_weights-13
layer-18
layer_with_weights-14
layer-19
layer-20
layer-21
layer-22
layer_with_weights-15
layer-23
layer_with_weights-16
layer-24
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
 
h

 kernel
!bias
"regularization_losses
#trainable_variables
$	variables
%	keras_api
?
&axis
	'gamma
(beta
)moving_mean
*moving_variance
+regularization_losses
,trainable_variables
-	variables
.	keras_api
h

/kernel
0bias
1regularization_losses
2trainable_variables
3	variables
4	keras_api
?
5axis
	6gamma
7beta
8moving_mean
9moving_variance
:regularization_losses
;trainable_variables
<	variables
=	keras_api
h

>kernel
?bias
@regularization_losses
Atrainable_variables
B	variables
C	keras_api
R
Dregularization_losses
Etrainable_variables
F	variables
G	keras_api
R
Hregularization_losses
Itrainable_variables
J	variables
K	keras_api
h

Lkernel
Mbias
Nregularization_losses
Otrainable_variables
P	variables
Q	keras_api
?
Raxis
	Sgamma
Tbeta
Umoving_mean
Vmoving_variance
Wregularization_losses
Xtrainable_variables
Y	variables
Z	keras_api
h

[kernel
\bias
]regularization_losses
^trainable_variables
_	variables
`	keras_api
?
aaxis
	bgamma
cbeta
dmoving_mean
emoving_variance
fregularization_losses
gtrainable_variables
h	variables
i	keras_api
h

jkernel
kbias
lregularization_losses
mtrainable_variables
n	variables
o	keras_api
R
pregularization_losses
qtrainable_variables
r	variables
s	keras_api
R
tregularization_losses
utrainable_variables
v	variables
w	keras_api
h

xkernel
ybias
zregularization_losses
{trainable_variables
|	variables
}	keras_api
?
~axis
	gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?trainable_variables
?	variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?trainable_variables
?	variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
	?iter
?beta_1
?beta_2

?decay
?learning_rate m?!m?'m?(m?/m?0m?6m?7m?>m??m?Lm?Mm?Sm?Tm?[m?\m?bm?cm?jm?km?xm?ym?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m? v?!v?'v?(v?/v?0v?6v?7v?>v??v?Lv?Mv?Sv?Tv?[v?\v?bv?cv?jv?kv?xv?yv?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?
 
?
 0
!1
'2
(3
/4
05
66
77
>8
?9
L10
M11
S12
T13
[14
\15
b16
c17
j18
k19
x20
y21
22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?
 0
!1
'2
(3
)4
*5
/6
07
68
79
810
911
>12
?13
L14
M15
S16
T17
U18
V19
[20
\21
b22
c23
d24
e25
j26
k27
x28
y29
30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43
?44
?45
?
regularization_losses
?layers
?layer_metrics
?metrics
trainable_variables
	variables
 ?layer_regularization_losses
?non_trainable_variables
 
][
VARIABLE_VALUEconv2d_136/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_136/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

 0
!1

 0
!1
?
"regularization_losses
?layers
?layer_metrics
?metrics
#trainable_variables
$	variables
 ?layer_regularization_losses
?non_trainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_46/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_46/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_46/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_46/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

'0
(1

'0
(1
)2
*3
?
+regularization_losses
?layers
?layer_metrics
?metrics
,trainable_variables
-	variables
 ?layer_regularization_losses
?non_trainable_variables
][
VARIABLE_VALUEconv2d_137/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_137/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

/0
01

/0
01
?
1regularization_losses
?layers
?layer_metrics
?metrics
2trainable_variables
3	variables
 ?layer_regularization_losses
?non_trainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_47/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_47/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_47/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_47/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

60
71

60
71
82
93
?
:regularization_losses
?layers
?layer_metrics
?metrics
;trainable_variables
<	variables
 ?layer_regularization_losses
?non_trainable_variables
][
VARIABLE_VALUEconv2d_138/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_138/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

>0
?1

>0
?1
?
@regularization_losses
?layers
?layer_metrics
?metrics
Atrainable_variables
B	variables
 ?layer_regularization_losses
?non_trainable_variables
 
 
 
?
Dregularization_losses
?layers
?layer_metrics
?metrics
Etrainable_variables
F	variables
 ?layer_regularization_losses
?non_trainable_variables
 
 
 
?
Hregularization_losses
?layers
?layer_metrics
?metrics
Itrainable_variables
J	variables
 ?layer_regularization_losses
?non_trainable_variables
][
VARIABLE_VALUEconv2d_139/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_139/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

L0
M1

L0
M1
?
Nregularization_losses
?layers
?layer_metrics
?metrics
Otrainable_variables
P	variables
 ?layer_regularization_losses
?non_trainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_48/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_48/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_48/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_48/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

S0
T1

S0
T1
U2
V3
?
Wregularization_losses
?layers
?layer_metrics
?metrics
Xtrainable_variables
Y	variables
 ?layer_regularization_losses
?non_trainable_variables
][
VARIABLE_VALUEconv2d_140/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_140/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 

[0
\1

[0
\1
?
]regularization_losses
?layers
?layer_metrics
?metrics
^trainable_variables
_	variables
 ?layer_regularization_losses
?non_trainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_49/gamma5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_49/beta4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_49/moving_mean;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_49/moving_variance?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

b0
c1

b0
c1
d2
e3
?
fregularization_losses
?layers
?layer_metrics
?metrics
gtrainable_variables
h	variables
 ?layer_regularization_losses
?non_trainable_variables
][
VARIABLE_VALUEconv2d_141/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_141/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE
 

j0
k1

j0
k1
?
lregularization_losses
?layers
?layer_metrics
?metrics
mtrainable_variables
n	variables
 ?layer_regularization_losses
?non_trainable_variables
 
 
 
?
pregularization_losses
?layers
?layer_metrics
?metrics
qtrainable_variables
r	variables
 ?layer_regularization_losses
?non_trainable_variables
 
 
 
?
tregularization_losses
?layers
?layer_metrics
?metrics
utrainable_variables
v	variables
 ?layer_regularization_losses
?non_trainable_variables
^\
VARIABLE_VALUEconv2d_142/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEconv2d_142/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE
 

x0
y1

x0
y1
?
zregularization_losses
?layers
?layer_metrics
?metrics
{trainable_variables
|	variables
 ?layer_regularization_losses
?non_trainable_variables
 
hf
VARIABLE_VALUEbatch_normalization_50/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_50/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_50/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_50/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

0
?1

0
?1
?2
?3
?
?regularization_losses
?layers
?layer_metrics
?metrics
?trainable_variables
?	variables
 ?layer_regularization_losses
?non_trainable_variables
^\
VARIABLE_VALUEconv2d_143/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEconv2d_143/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?regularization_losses
?layers
?layer_metrics
?metrics
?trainable_variables
?	variables
 ?layer_regularization_losses
?non_trainable_variables
 
hf
VARIABLE_VALUEbatch_normalization_51/gamma6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_51/beta5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_51/moving_mean<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_51/moving_variance@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1
 
?0
?1
?2
?3
?
?regularization_losses
?layers
?layer_metrics
?metrics
?trainable_variables
?	variables
 ?layer_regularization_losses
?non_trainable_variables
^\
VARIABLE_VALUEconv2d_144/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEconv2d_144/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?regularization_losses
?layers
?layer_metrics
?metrics
?trainable_variables
?	variables
 ?layer_regularization_losses
?non_trainable_variables
 
 
 
?
?regularization_losses
?layers
?layer_metrics
?metrics
?trainable_variables
?	variables
 ?layer_regularization_losses
?non_trainable_variables
 
 
 
?
?regularization_losses
?layers
?layer_metrics
?metrics
?trainable_variables
?	variables
 ?layer_regularization_losses
?non_trainable_variables
 
 
 
?
?regularization_losses
?layers
?layer_metrics
?metrics
?trainable_variables
?	variables
 ?layer_regularization_losses
?non_trainable_variables
\Z
VARIABLE_VALUEdense_58/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_58/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?regularization_losses
?layers
?layer_metrics
?metrics
?trainable_variables
?	variables
 ?layer_regularization_losses
?non_trainable_variables
\Z
VARIABLE_VALUEdense_59/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_59/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?regularization_losses
?layers
?layer_metrics
?metrics
?trainable_variables
?	variables
 ?layer_regularization_losses
?non_trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
 

?0
?1
 
Z
)0
*1
82
93
U4
V5
d6
e7
?8
?9
?10
?11
 
 
 
 
 
 
 
 
 

)0
*1
 
 
 
 
 
 
 
 
 

80
91
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

U0
V1
 
 
 
 
 
 
 
 
 

d0
e1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

?0
?1
 
 
 
 
 
 
 
 
 

?0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
?~
VARIABLE_VALUEAdam/conv2d_136/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_136/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_46/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_46/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_137/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_137/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_47/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_47/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_138/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_138/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_139/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_139/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_48/gamma/mQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_48/beta/mPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_140/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_140/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_49/gamma/mQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_49/beta/mPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_141/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_141/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/conv2d_142/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/conv2d_142/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_50/gamma/mRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_50/beta/mQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/conv2d_143/kernel/mSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/conv2d_143/bias/mQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_51/gamma/mRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_51/beta/mQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/conv2d_144/kernel/mSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/conv2d_144/bias/mQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_58/kernel/mSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_58/bias/mQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_59/kernel/mSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_59/bias/mQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_136/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_136/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_46/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_46/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_137/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_137/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_47/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_47/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_138/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_138/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_139/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_139/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_48/gamma/vQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_48/beta/vPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_140/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_140/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_49/gamma/vQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_49/beta/vPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_141/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_141/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/conv2d_142/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/conv2d_142/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_50/gamma/vRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_50/beta/vQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/conv2d_143/kernel/vSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/conv2d_143/bias/vQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_51/gamma/vRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_51/beta/vQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/conv2d_144/kernel/vSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/conv2d_144/bias/vQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_58/kernel/vSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_58/bias/vQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_59/kernel/vSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_59/bias/vQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_30Placeholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_30conv2d_136/kernelconv2d_136/biasbatch_normalization_46/gammabatch_normalization_46/beta"batch_normalization_46/moving_mean&batch_normalization_46/moving_varianceconv2d_137/kernelconv2d_137/biasbatch_normalization_47/gammabatch_normalization_47/beta"batch_normalization_47/moving_mean&batch_normalization_47/moving_varianceconv2d_138/kernelconv2d_138/biasconv2d_139/kernelconv2d_139/biasbatch_normalization_48/gammabatch_normalization_48/beta"batch_normalization_48/moving_mean&batch_normalization_48/moving_varianceconv2d_140/kernelconv2d_140/biasbatch_normalization_49/gammabatch_normalization_49/beta"batch_normalization_49/moving_mean&batch_normalization_49/moving_varianceconv2d_141/kernelconv2d_141/biasconv2d_142/kernelconv2d_142/biasbatch_normalization_50/gammabatch_normalization_50/beta"batch_normalization_50/moving_mean&batch_normalization_50/moving_varianceconv2d_143/kernelconv2d_143/biasbatch_normalization_51/gammabatch_normalization_51/beta"batch_normalization_51/moving_mean&batch_normalization_51/moving_varianceconv2d_144/kernelconv2d_144/biasdense_58/kerneldense_58/biasdense_59/kerneldense_59/bias*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_2459541
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?/
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%conv2d_136/kernel/Read/ReadVariableOp#conv2d_136/bias/Read/ReadVariableOp0batch_normalization_46/gamma/Read/ReadVariableOp/batch_normalization_46/beta/Read/ReadVariableOp6batch_normalization_46/moving_mean/Read/ReadVariableOp:batch_normalization_46/moving_variance/Read/ReadVariableOp%conv2d_137/kernel/Read/ReadVariableOp#conv2d_137/bias/Read/ReadVariableOp0batch_normalization_47/gamma/Read/ReadVariableOp/batch_normalization_47/beta/Read/ReadVariableOp6batch_normalization_47/moving_mean/Read/ReadVariableOp:batch_normalization_47/moving_variance/Read/ReadVariableOp%conv2d_138/kernel/Read/ReadVariableOp#conv2d_138/bias/Read/ReadVariableOp%conv2d_139/kernel/Read/ReadVariableOp#conv2d_139/bias/Read/ReadVariableOp0batch_normalization_48/gamma/Read/ReadVariableOp/batch_normalization_48/beta/Read/ReadVariableOp6batch_normalization_48/moving_mean/Read/ReadVariableOp:batch_normalization_48/moving_variance/Read/ReadVariableOp%conv2d_140/kernel/Read/ReadVariableOp#conv2d_140/bias/Read/ReadVariableOp0batch_normalization_49/gamma/Read/ReadVariableOp/batch_normalization_49/beta/Read/ReadVariableOp6batch_normalization_49/moving_mean/Read/ReadVariableOp:batch_normalization_49/moving_variance/Read/ReadVariableOp%conv2d_141/kernel/Read/ReadVariableOp#conv2d_141/bias/Read/ReadVariableOp%conv2d_142/kernel/Read/ReadVariableOp#conv2d_142/bias/Read/ReadVariableOp0batch_normalization_50/gamma/Read/ReadVariableOp/batch_normalization_50/beta/Read/ReadVariableOp6batch_normalization_50/moving_mean/Read/ReadVariableOp:batch_normalization_50/moving_variance/Read/ReadVariableOp%conv2d_143/kernel/Read/ReadVariableOp#conv2d_143/bias/Read/ReadVariableOp0batch_normalization_51/gamma/Read/ReadVariableOp/batch_normalization_51/beta/Read/ReadVariableOp6batch_normalization_51/moving_mean/Read/ReadVariableOp:batch_normalization_51/moving_variance/Read/ReadVariableOp%conv2d_144/kernel/Read/ReadVariableOp#conv2d_144/bias/Read/ReadVariableOp#dense_58/kernel/Read/ReadVariableOp!dense_58/bias/Read/ReadVariableOp#dense_59/kernel/Read/ReadVariableOp!dense_59/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp,Adam/conv2d_136/kernel/m/Read/ReadVariableOp*Adam/conv2d_136/bias/m/Read/ReadVariableOp7Adam/batch_normalization_46/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_46/beta/m/Read/ReadVariableOp,Adam/conv2d_137/kernel/m/Read/ReadVariableOp*Adam/conv2d_137/bias/m/Read/ReadVariableOp7Adam/batch_normalization_47/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_47/beta/m/Read/ReadVariableOp,Adam/conv2d_138/kernel/m/Read/ReadVariableOp*Adam/conv2d_138/bias/m/Read/ReadVariableOp,Adam/conv2d_139/kernel/m/Read/ReadVariableOp*Adam/conv2d_139/bias/m/Read/ReadVariableOp7Adam/batch_normalization_48/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_48/beta/m/Read/ReadVariableOp,Adam/conv2d_140/kernel/m/Read/ReadVariableOp*Adam/conv2d_140/bias/m/Read/ReadVariableOp7Adam/batch_normalization_49/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_49/beta/m/Read/ReadVariableOp,Adam/conv2d_141/kernel/m/Read/ReadVariableOp*Adam/conv2d_141/bias/m/Read/ReadVariableOp,Adam/conv2d_142/kernel/m/Read/ReadVariableOp*Adam/conv2d_142/bias/m/Read/ReadVariableOp7Adam/batch_normalization_50/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_50/beta/m/Read/ReadVariableOp,Adam/conv2d_143/kernel/m/Read/ReadVariableOp*Adam/conv2d_143/bias/m/Read/ReadVariableOp7Adam/batch_normalization_51/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_51/beta/m/Read/ReadVariableOp,Adam/conv2d_144/kernel/m/Read/ReadVariableOp*Adam/conv2d_144/bias/m/Read/ReadVariableOp*Adam/dense_58/kernel/m/Read/ReadVariableOp(Adam/dense_58/bias/m/Read/ReadVariableOp*Adam/dense_59/kernel/m/Read/ReadVariableOp(Adam/dense_59/bias/m/Read/ReadVariableOp,Adam/conv2d_136/kernel/v/Read/ReadVariableOp*Adam/conv2d_136/bias/v/Read/ReadVariableOp7Adam/batch_normalization_46/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_46/beta/v/Read/ReadVariableOp,Adam/conv2d_137/kernel/v/Read/ReadVariableOp*Adam/conv2d_137/bias/v/Read/ReadVariableOp7Adam/batch_normalization_47/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_47/beta/v/Read/ReadVariableOp,Adam/conv2d_138/kernel/v/Read/ReadVariableOp*Adam/conv2d_138/bias/v/Read/ReadVariableOp,Adam/conv2d_139/kernel/v/Read/ReadVariableOp*Adam/conv2d_139/bias/v/Read/ReadVariableOp7Adam/batch_normalization_48/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_48/beta/v/Read/ReadVariableOp,Adam/conv2d_140/kernel/v/Read/ReadVariableOp*Adam/conv2d_140/bias/v/Read/ReadVariableOp7Adam/batch_normalization_49/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_49/beta/v/Read/ReadVariableOp,Adam/conv2d_141/kernel/v/Read/ReadVariableOp*Adam/conv2d_141/bias/v/Read/ReadVariableOp,Adam/conv2d_142/kernel/v/Read/ReadVariableOp*Adam/conv2d_142/bias/v/Read/ReadVariableOp7Adam/batch_normalization_50/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_50/beta/v/Read/ReadVariableOp,Adam/conv2d_143/kernel/v/Read/ReadVariableOp*Adam/conv2d_143/bias/v/Read/ReadVariableOp7Adam/batch_normalization_51/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_51/beta/v/Read/ReadVariableOp,Adam/conv2d_144/kernel/v/Read/ReadVariableOp*Adam/conv2d_144/bias/v/Read/ReadVariableOp*Adam/dense_58/kernel/v/Read/ReadVariableOp(Adam/dense_58/bias/v/Read/ReadVariableOp*Adam/dense_59/kernel/v/Read/ReadVariableOp(Adam/dense_59/bias/v/Read/ReadVariableOpConst*?
Tin?
2}	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__traced_save_2461520
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_136/kernelconv2d_136/biasbatch_normalization_46/gammabatch_normalization_46/beta"batch_normalization_46/moving_mean&batch_normalization_46/moving_varianceconv2d_137/kernelconv2d_137/biasbatch_normalization_47/gammabatch_normalization_47/beta"batch_normalization_47/moving_mean&batch_normalization_47/moving_varianceconv2d_138/kernelconv2d_138/biasconv2d_139/kernelconv2d_139/biasbatch_normalization_48/gammabatch_normalization_48/beta"batch_normalization_48/moving_mean&batch_normalization_48/moving_varianceconv2d_140/kernelconv2d_140/biasbatch_normalization_49/gammabatch_normalization_49/beta"batch_normalization_49/moving_mean&batch_normalization_49/moving_varianceconv2d_141/kernelconv2d_141/biasconv2d_142/kernelconv2d_142/biasbatch_normalization_50/gammabatch_normalization_50/beta"batch_normalization_50/moving_mean&batch_normalization_50/moving_varianceconv2d_143/kernelconv2d_143/biasbatch_normalization_51/gammabatch_normalization_51/beta"batch_normalization_51/moving_mean&batch_normalization_51/moving_varianceconv2d_144/kernelconv2d_144/biasdense_58/kerneldense_58/biasdense_59/kerneldense_59/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv2d_136/kernel/mAdam/conv2d_136/bias/m#Adam/batch_normalization_46/gamma/m"Adam/batch_normalization_46/beta/mAdam/conv2d_137/kernel/mAdam/conv2d_137/bias/m#Adam/batch_normalization_47/gamma/m"Adam/batch_normalization_47/beta/mAdam/conv2d_138/kernel/mAdam/conv2d_138/bias/mAdam/conv2d_139/kernel/mAdam/conv2d_139/bias/m#Adam/batch_normalization_48/gamma/m"Adam/batch_normalization_48/beta/mAdam/conv2d_140/kernel/mAdam/conv2d_140/bias/m#Adam/batch_normalization_49/gamma/m"Adam/batch_normalization_49/beta/mAdam/conv2d_141/kernel/mAdam/conv2d_141/bias/mAdam/conv2d_142/kernel/mAdam/conv2d_142/bias/m#Adam/batch_normalization_50/gamma/m"Adam/batch_normalization_50/beta/mAdam/conv2d_143/kernel/mAdam/conv2d_143/bias/m#Adam/batch_normalization_51/gamma/m"Adam/batch_normalization_51/beta/mAdam/conv2d_144/kernel/mAdam/conv2d_144/bias/mAdam/dense_58/kernel/mAdam/dense_58/bias/mAdam/dense_59/kernel/mAdam/dense_59/bias/mAdam/conv2d_136/kernel/vAdam/conv2d_136/bias/v#Adam/batch_normalization_46/gamma/v"Adam/batch_normalization_46/beta/vAdam/conv2d_137/kernel/vAdam/conv2d_137/bias/v#Adam/batch_normalization_47/gamma/v"Adam/batch_normalization_47/beta/vAdam/conv2d_138/kernel/vAdam/conv2d_138/bias/vAdam/conv2d_139/kernel/vAdam/conv2d_139/bias/v#Adam/batch_normalization_48/gamma/v"Adam/batch_normalization_48/beta/vAdam/conv2d_140/kernel/vAdam/conv2d_140/bias/v#Adam/batch_normalization_49/gamma/v"Adam/batch_normalization_49/beta/vAdam/conv2d_141/kernel/vAdam/conv2d_141/bias/vAdam/conv2d_142/kernel/vAdam/conv2d_142/bias/v#Adam/batch_normalization_50/gamma/v"Adam/batch_normalization_50/beta/vAdam/conv2d_143/kernel/vAdam/conv2d_143/bias/v#Adam/batch_normalization_51/gamma/v"Adam/batch_normalization_51/beta/vAdam/conv2d_144/kernel/vAdam/conv2d_144/bias/vAdam/dense_58/kernel/vAdam/dense_58/bias/vAdam/dense_59/kernel/vAdam/dense_59/bias/v*?
Tin?
~2|*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__traced_restore_2461899΃
?	
?
E__inference_dense_59_layer_call_and_return_conditional_losses_2461119

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
G__inference_conv2d_142_layer_call_and_return_conditional_losses_2458578

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_50_layer_call_and_return_conditional_losses_2460789

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_49_layer_call_and_return_conditional_losses_2460673

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_47_layer_call_and_return_conditional_losses_2457614

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
??
?8
 __inference__traced_save_2461520
file_prefix0
,savev2_conv2d_136_kernel_read_readvariableop.
*savev2_conv2d_136_bias_read_readvariableop;
7savev2_batch_normalization_46_gamma_read_readvariableop:
6savev2_batch_normalization_46_beta_read_readvariableopA
=savev2_batch_normalization_46_moving_mean_read_readvariableopE
Asavev2_batch_normalization_46_moving_variance_read_readvariableop0
,savev2_conv2d_137_kernel_read_readvariableop.
*savev2_conv2d_137_bias_read_readvariableop;
7savev2_batch_normalization_47_gamma_read_readvariableop:
6savev2_batch_normalization_47_beta_read_readvariableopA
=savev2_batch_normalization_47_moving_mean_read_readvariableopE
Asavev2_batch_normalization_47_moving_variance_read_readvariableop0
,savev2_conv2d_138_kernel_read_readvariableop.
*savev2_conv2d_138_bias_read_readvariableop0
,savev2_conv2d_139_kernel_read_readvariableop.
*savev2_conv2d_139_bias_read_readvariableop;
7savev2_batch_normalization_48_gamma_read_readvariableop:
6savev2_batch_normalization_48_beta_read_readvariableopA
=savev2_batch_normalization_48_moving_mean_read_readvariableopE
Asavev2_batch_normalization_48_moving_variance_read_readvariableop0
,savev2_conv2d_140_kernel_read_readvariableop.
*savev2_conv2d_140_bias_read_readvariableop;
7savev2_batch_normalization_49_gamma_read_readvariableop:
6savev2_batch_normalization_49_beta_read_readvariableopA
=savev2_batch_normalization_49_moving_mean_read_readvariableopE
Asavev2_batch_normalization_49_moving_variance_read_readvariableop0
,savev2_conv2d_141_kernel_read_readvariableop.
*savev2_conv2d_141_bias_read_readvariableop0
,savev2_conv2d_142_kernel_read_readvariableop.
*savev2_conv2d_142_bias_read_readvariableop;
7savev2_batch_normalization_50_gamma_read_readvariableop:
6savev2_batch_normalization_50_beta_read_readvariableopA
=savev2_batch_normalization_50_moving_mean_read_readvariableopE
Asavev2_batch_normalization_50_moving_variance_read_readvariableop0
,savev2_conv2d_143_kernel_read_readvariableop.
*savev2_conv2d_143_bias_read_readvariableop;
7savev2_batch_normalization_51_gamma_read_readvariableop:
6savev2_batch_normalization_51_beta_read_readvariableopA
=savev2_batch_normalization_51_moving_mean_read_readvariableopE
Asavev2_batch_normalization_51_moving_variance_read_readvariableop0
,savev2_conv2d_144_kernel_read_readvariableop.
*savev2_conv2d_144_bias_read_readvariableop.
*savev2_dense_58_kernel_read_readvariableop,
(savev2_dense_58_bias_read_readvariableop.
*savev2_dense_59_kernel_read_readvariableop,
(savev2_dense_59_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop7
3savev2_adam_conv2d_136_kernel_m_read_readvariableop5
1savev2_adam_conv2d_136_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_46_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_46_beta_m_read_readvariableop7
3savev2_adam_conv2d_137_kernel_m_read_readvariableop5
1savev2_adam_conv2d_137_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_47_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_47_beta_m_read_readvariableop7
3savev2_adam_conv2d_138_kernel_m_read_readvariableop5
1savev2_adam_conv2d_138_bias_m_read_readvariableop7
3savev2_adam_conv2d_139_kernel_m_read_readvariableop5
1savev2_adam_conv2d_139_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_48_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_48_beta_m_read_readvariableop7
3savev2_adam_conv2d_140_kernel_m_read_readvariableop5
1savev2_adam_conv2d_140_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_49_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_49_beta_m_read_readvariableop7
3savev2_adam_conv2d_141_kernel_m_read_readvariableop5
1savev2_adam_conv2d_141_bias_m_read_readvariableop7
3savev2_adam_conv2d_142_kernel_m_read_readvariableop5
1savev2_adam_conv2d_142_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_50_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_50_beta_m_read_readvariableop7
3savev2_adam_conv2d_143_kernel_m_read_readvariableop5
1savev2_adam_conv2d_143_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_51_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_51_beta_m_read_readvariableop7
3savev2_adam_conv2d_144_kernel_m_read_readvariableop5
1savev2_adam_conv2d_144_bias_m_read_readvariableop5
1savev2_adam_dense_58_kernel_m_read_readvariableop3
/savev2_adam_dense_58_bias_m_read_readvariableop5
1savev2_adam_dense_59_kernel_m_read_readvariableop3
/savev2_adam_dense_59_bias_m_read_readvariableop7
3savev2_adam_conv2d_136_kernel_v_read_readvariableop5
1savev2_adam_conv2d_136_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_46_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_46_beta_v_read_readvariableop7
3savev2_adam_conv2d_137_kernel_v_read_readvariableop5
1savev2_adam_conv2d_137_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_47_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_47_beta_v_read_readvariableop7
3savev2_adam_conv2d_138_kernel_v_read_readvariableop5
1savev2_adam_conv2d_138_bias_v_read_readvariableop7
3savev2_adam_conv2d_139_kernel_v_read_readvariableop5
1savev2_adam_conv2d_139_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_48_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_48_beta_v_read_readvariableop7
3savev2_adam_conv2d_140_kernel_v_read_readvariableop5
1savev2_adam_conv2d_140_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_49_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_49_beta_v_read_readvariableop7
3savev2_adam_conv2d_141_kernel_v_read_readvariableop5
1savev2_adam_conv2d_141_bias_v_read_readvariableop7
3savev2_adam_conv2d_142_kernel_v_read_readvariableop5
1savev2_adam_conv2d_142_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_50_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_50_beta_v_read_readvariableop7
3savev2_adam_conv2d_143_kernel_v_read_readvariableop5
1savev2_adam_conv2d_143_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_51_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_51_beta_v_read_readvariableop7
3savev2_adam_conv2d_144_kernel_v_read_readvariableop5
1savev2_adam_conv2d_144_bias_v_read_readvariableop5
1savev2_adam_dense_58_kernel_v_read_readvariableop3
/savev2_adam_dense_58_bias_v_read_readvariableop5
1savev2_adam_dense_59_kernel_v_read_readvariableop3
/savev2_adam_dense_59_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?E
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:|*
dtype0*?D
value?DB?D|B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:|*
dtype0*?
value?B?|B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?5
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_conv2d_136_kernel_read_readvariableop*savev2_conv2d_136_bias_read_readvariableop7savev2_batch_normalization_46_gamma_read_readvariableop6savev2_batch_normalization_46_beta_read_readvariableop=savev2_batch_normalization_46_moving_mean_read_readvariableopAsavev2_batch_normalization_46_moving_variance_read_readvariableop,savev2_conv2d_137_kernel_read_readvariableop*savev2_conv2d_137_bias_read_readvariableop7savev2_batch_normalization_47_gamma_read_readvariableop6savev2_batch_normalization_47_beta_read_readvariableop=savev2_batch_normalization_47_moving_mean_read_readvariableopAsavev2_batch_normalization_47_moving_variance_read_readvariableop,savev2_conv2d_138_kernel_read_readvariableop*savev2_conv2d_138_bias_read_readvariableop,savev2_conv2d_139_kernel_read_readvariableop*savev2_conv2d_139_bias_read_readvariableop7savev2_batch_normalization_48_gamma_read_readvariableop6savev2_batch_normalization_48_beta_read_readvariableop=savev2_batch_normalization_48_moving_mean_read_readvariableopAsavev2_batch_normalization_48_moving_variance_read_readvariableop,savev2_conv2d_140_kernel_read_readvariableop*savev2_conv2d_140_bias_read_readvariableop7savev2_batch_normalization_49_gamma_read_readvariableop6savev2_batch_normalization_49_beta_read_readvariableop=savev2_batch_normalization_49_moving_mean_read_readvariableopAsavev2_batch_normalization_49_moving_variance_read_readvariableop,savev2_conv2d_141_kernel_read_readvariableop*savev2_conv2d_141_bias_read_readvariableop,savev2_conv2d_142_kernel_read_readvariableop*savev2_conv2d_142_bias_read_readvariableop7savev2_batch_normalization_50_gamma_read_readvariableop6savev2_batch_normalization_50_beta_read_readvariableop=savev2_batch_normalization_50_moving_mean_read_readvariableopAsavev2_batch_normalization_50_moving_variance_read_readvariableop,savev2_conv2d_143_kernel_read_readvariableop*savev2_conv2d_143_bias_read_readvariableop7savev2_batch_normalization_51_gamma_read_readvariableop6savev2_batch_normalization_51_beta_read_readvariableop=savev2_batch_normalization_51_moving_mean_read_readvariableopAsavev2_batch_normalization_51_moving_variance_read_readvariableop,savev2_conv2d_144_kernel_read_readvariableop*savev2_conv2d_144_bias_read_readvariableop*savev2_dense_58_kernel_read_readvariableop(savev2_dense_58_bias_read_readvariableop*savev2_dense_59_kernel_read_readvariableop(savev2_dense_59_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop3savev2_adam_conv2d_136_kernel_m_read_readvariableop1savev2_adam_conv2d_136_bias_m_read_readvariableop>savev2_adam_batch_normalization_46_gamma_m_read_readvariableop=savev2_adam_batch_normalization_46_beta_m_read_readvariableop3savev2_adam_conv2d_137_kernel_m_read_readvariableop1savev2_adam_conv2d_137_bias_m_read_readvariableop>savev2_adam_batch_normalization_47_gamma_m_read_readvariableop=savev2_adam_batch_normalization_47_beta_m_read_readvariableop3savev2_adam_conv2d_138_kernel_m_read_readvariableop1savev2_adam_conv2d_138_bias_m_read_readvariableop3savev2_adam_conv2d_139_kernel_m_read_readvariableop1savev2_adam_conv2d_139_bias_m_read_readvariableop>savev2_adam_batch_normalization_48_gamma_m_read_readvariableop=savev2_adam_batch_normalization_48_beta_m_read_readvariableop3savev2_adam_conv2d_140_kernel_m_read_readvariableop1savev2_adam_conv2d_140_bias_m_read_readvariableop>savev2_adam_batch_normalization_49_gamma_m_read_readvariableop=savev2_adam_batch_normalization_49_beta_m_read_readvariableop3savev2_adam_conv2d_141_kernel_m_read_readvariableop1savev2_adam_conv2d_141_bias_m_read_readvariableop3savev2_adam_conv2d_142_kernel_m_read_readvariableop1savev2_adam_conv2d_142_bias_m_read_readvariableop>savev2_adam_batch_normalization_50_gamma_m_read_readvariableop=savev2_adam_batch_normalization_50_beta_m_read_readvariableop3savev2_adam_conv2d_143_kernel_m_read_readvariableop1savev2_adam_conv2d_143_bias_m_read_readvariableop>savev2_adam_batch_normalization_51_gamma_m_read_readvariableop=savev2_adam_batch_normalization_51_beta_m_read_readvariableop3savev2_adam_conv2d_144_kernel_m_read_readvariableop1savev2_adam_conv2d_144_bias_m_read_readvariableop1savev2_adam_dense_58_kernel_m_read_readvariableop/savev2_adam_dense_58_bias_m_read_readvariableop1savev2_adam_dense_59_kernel_m_read_readvariableop/savev2_adam_dense_59_bias_m_read_readvariableop3savev2_adam_conv2d_136_kernel_v_read_readvariableop1savev2_adam_conv2d_136_bias_v_read_readvariableop>savev2_adam_batch_normalization_46_gamma_v_read_readvariableop=savev2_adam_batch_normalization_46_beta_v_read_readvariableop3savev2_adam_conv2d_137_kernel_v_read_readvariableop1savev2_adam_conv2d_137_bias_v_read_readvariableop>savev2_adam_batch_normalization_47_gamma_v_read_readvariableop=savev2_adam_batch_normalization_47_beta_v_read_readvariableop3savev2_adam_conv2d_138_kernel_v_read_readvariableop1savev2_adam_conv2d_138_bias_v_read_readvariableop3savev2_adam_conv2d_139_kernel_v_read_readvariableop1savev2_adam_conv2d_139_bias_v_read_readvariableop>savev2_adam_batch_normalization_48_gamma_v_read_readvariableop=savev2_adam_batch_normalization_48_beta_v_read_readvariableop3savev2_adam_conv2d_140_kernel_v_read_readvariableop1savev2_adam_conv2d_140_bias_v_read_readvariableop>savev2_adam_batch_normalization_49_gamma_v_read_readvariableop=savev2_adam_batch_normalization_49_beta_v_read_readvariableop3savev2_adam_conv2d_141_kernel_v_read_readvariableop1savev2_adam_conv2d_141_bias_v_read_readvariableop3savev2_adam_conv2d_142_kernel_v_read_readvariableop1savev2_adam_conv2d_142_bias_v_read_readvariableop>savev2_adam_batch_normalization_50_gamma_v_read_readvariableop=savev2_adam_batch_normalization_50_beta_v_read_readvariableop3savev2_adam_conv2d_143_kernel_v_read_readvariableop1savev2_adam_conv2d_143_bias_v_read_readvariableop>savev2_adam_batch_normalization_51_gamma_v_read_readvariableop=savev2_adam_batch_normalization_51_beta_v_read_readvariableop3savev2_adam_conv2d_144_kernel_v_read_readvariableop1savev2_adam_conv2d_144_bias_v_read_readvariableop1savev2_adam_dense_58_kernel_v_read_readvariableop/savev2_adam_dense_58_bias_v_read_readvariableop1savev2_adam_dense_59_kernel_v_read_readvariableop/savev2_adam_dense_59_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *?
dtypes?
~2|	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: ::::::::::::::::::::::::::::::::::::::::::: : : :: : : : : : : : : ::::::::::::::::::::::::::::::: : : :::::::::::::::::::::::::::::::: : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 	

_output_shapes
:: 


_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::  

_output_shapes
:: !

_output_shapes
:: "

_output_shapes
::,#(
&
_output_shapes
:: $

_output_shapes
:: %

_output_shapes
:: &

_output_shapes
:: '

_output_shapes
:: (

_output_shapes
::,)(
&
_output_shapes
:: *

_output_shapes
::$+ 

_output_shapes

: : ,

_output_shapes
: :$- 

_output_shapes

: : .

_output_shapes
::/

_output_shapes
: :0

_output_shapes
: :1

_output_shapes
: :2

_output_shapes
: :3

_output_shapes
: :4

_output_shapes
: :5

_output_shapes
: :6

_output_shapes
: :7

_output_shapes
: :,8(
&
_output_shapes
:: 9

_output_shapes
:: :

_output_shapes
:: ;

_output_shapes
::,<(
&
_output_shapes
:: =

_output_shapes
:: >

_output_shapes
:: ?

_output_shapes
::,@(
&
_output_shapes
:: A

_output_shapes
::,B(
&
_output_shapes
:: C

_output_shapes
:: D

_output_shapes
:: E

_output_shapes
::,F(
&
_output_shapes
:: G

_output_shapes
:: H

_output_shapes
:: I

_output_shapes
::,J(
&
_output_shapes
:: K

_output_shapes
::,L(
&
_output_shapes
:: M

_output_shapes
:: N

_output_shapes
:: O

_output_shapes
::,P(
&
_output_shapes
:: Q

_output_shapes
:: R

_output_shapes
:: S

_output_shapes
::,T(
&
_output_shapes
:: U

_output_shapes
::$V 

_output_shapes

: : W

_output_shapes
: :$X 

_output_shapes

: : Y

_output_shapes
::,Z(
&
_output_shapes
:: [

_output_shapes
:: \

_output_shapes
:: ]

_output_shapes
::,^(
&
_output_shapes
:: _

_output_shapes
:: `

_output_shapes
:: a

_output_shapes
::,b(
&
_output_shapes
:: c

_output_shapes
::,d(
&
_output_shapes
:: e

_output_shapes
:: f

_output_shapes
:: g

_output_shapes
::,h(
&
_output_shapes
:: i

_output_shapes
:: j

_output_shapes
:: k

_output_shapes
::,l(
&
_output_shapes
:: m

_output_shapes
::,n(
&
_output_shapes
:: o

_output_shapes
:: p

_output_shapes
:: q

_output_shapes
::,r(
&
_output_shapes
:: s

_output_shapes
:: t

_output_shapes
:: u

_output_shapes
::,v(
&
_output_shapes
:: w

_output_shapes
::$x 

_output_shapes

: : y

_output_shapes
: :$z 

_output_shapes

: : {

_output_shapes
::|

_output_shapes
: 
?	
?
E__inference_dense_59_layer_call_and_return_conditional_losses_2458862

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_51_layer_call_fn_2461032

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_51_layer_call_and_return_conditional_losses_24587132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
T__inference_CNN_aug_deep_bnorm_skip_layer_call_and_return_conditional_losses_2459122

inputs
conv2d_136_2459005
conv2d_136_2459007"
batch_normalization_46_2459010"
batch_normalization_46_2459012"
batch_normalization_46_2459014"
batch_normalization_46_2459016
conv2d_137_2459019
conv2d_137_2459021"
batch_normalization_47_2459024"
batch_normalization_47_2459026"
batch_normalization_47_2459028"
batch_normalization_47_2459030
conv2d_138_2459033
conv2d_138_2459035
conv2d_139_2459040
conv2d_139_2459042"
batch_normalization_48_2459045"
batch_normalization_48_2459047"
batch_normalization_48_2459049"
batch_normalization_48_2459051
conv2d_140_2459054
conv2d_140_2459056"
batch_normalization_49_2459059"
batch_normalization_49_2459061"
batch_normalization_49_2459063"
batch_normalization_49_2459065
conv2d_141_2459068
conv2d_141_2459070
conv2d_142_2459075
conv2d_142_2459077"
batch_normalization_50_2459080"
batch_normalization_50_2459082"
batch_normalization_50_2459084"
batch_normalization_50_2459086
conv2d_143_2459089
conv2d_143_2459091"
batch_normalization_51_2459094"
batch_normalization_51_2459096"
batch_normalization_51_2459098"
batch_normalization_51_2459100
conv2d_144_2459103
conv2d_144_2459105
dense_58_2459111
dense_58_2459113
dense_59_2459116
dense_59_2459118
identity??.batch_normalization_46/StatefulPartitionedCall?.batch_normalization_47/StatefulPartitionedCall?.batch_normalization_48/StatefulPartitionedCall?.batch_normalization_49/StatefulPartitionedCall?.batch_normalization_50/StatefulPartitionedCall?.batch_normalization_51/StatefulPartitionedCall?"conv2d_136/StatefulPartitionedCall?"conv2d_137/StatefulPartitionedCall?"conv2d_138/StatefulPartitionedCall?"conv2d_139/StatefulPartitionedCall?"conv2d_140/StatefulPartitionedCall?"conv2d_141/StatefulPartitionedCall?"conv2d_142/StatefulPartitionedCall?"conv2d_143/StatefulPartitionedCall?"conv2d_144/StatefulPartitionedCall? dense_58/StatefulPartitionedCall? dense_59/StatefulPartitionedCall?
"conv2d_136/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_136_2459005conv2d_136_2459007*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_136_layer_call_and_return_conditional_losses_24580922$
"conv2d_136/StatefulPartitionedCall?
.batch_normalization_46/StatefulPartitionedCallStatefulPartitionedCall+conv2d_136/StatefulPartitionedCall:output:0batch_normalization_46_2459010batch_normalization_46_2459012batch_normalization_46_2459014batch_normalization_46_2459016*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_46_layer_call_and_return_conditional_losses_245812720
.batch_normalization_46/StatefulPartitionedCall?
"conv2d_137/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_46/StatefulPartitionedCall:output:0conv2d_137_2459019conv2d_137_2459021*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_137_layer_call_and_return_conditional_losses_24581922$
"conv2d_137/StatefulPartitionedCall?
.batch_normalization_47/StatefulPartitionedCallStatefulPartitionedCall+conv2d_137/StatefulPartitionedCall:output:0batch_normalization_47_2459024batch_normalization_47_2459026batch_normalization_47_2459028batch_normalization_47_2459030*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_47_layer_call_and_return_conditional_losses_245822720
.batch_normalization_47/StatefulPartitionedCall?
"conv2d_138/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_47/StatefulPartitionedCall:output:0conv2d_138_2459033conv2d_138_2459035*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_138_layer_call_and_return_conditional_losses_24582922$
"conv2d_138/StatefulPartitionedCall?
add_26/PartitionedCallPartitionedCall+conv2d_138/StatefulPartitionedCall:output:0inputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_add_26_layer_call_and_return_conditional_losses_24583142
add_26/PartitionedCall?
 max_pooling2d_55/PartitionedCallPartitionedCalladd_26/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*** 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_max_pooling2d_55_layer_call_and_return_conditional_losses_24576312"
 max_pooling2d_55/PartitionedCall?
"conv2d_139/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_55/PartitionedCall:output:0conv2d_139_2459040conv2d_139_2459042*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????***$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_139_layer_call_and_return_conditional_losses_24583352$
"conv2d_139/StatefulPartitionedCall?
.batch_normalization_48/StatefulPartitionedCallStatefulPartitionedCall+conv2d_139/StatefulPartitionedCall:output:0batch_normalization_48_2459045batch_normalization_48_2459047batch_normalization_48_2459049batch_normalization_48_2459051*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????***$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_48_layer_call_and_return_conditional_losses_245837020
.batch_normalization_48/StatefulPartitionedCall?
"conv2d_140/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_48/StatefulPartitionedCall:output:0conv2d_140_2459054conv2d_140_2459056*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????***$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_140_layer_call_and_return_conditional_losses_24584352$
"conv2d_140/StatefulPartitionedCall?
.batch_normalization_49/StatefulPartitionedCallStatefulPartitionedCall+conv2d_140/StatefulPartitionedCall:output:0batch_normalization_49_2459059batch_normalization_49_2459061batch_normalization_49_2459063batch_normalization_49_2459065*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????***$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_49_layer_call_and_return_conditional_losses_245847020
.batch_normalization_49/StatefulPartitionedCall?
"conv2d_141/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_49/StatefulPartitionedCall:output:0conv2d_141_2459068conv2d_141_2459070*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????***$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_141_layer_call_and_return_conditional_losses_24585352$
"conv2d_141/StatefulPartitionedCall?
add_27/PartitionedCallPartitionedCall+conv2d_141/StatefulPartitionedCall:output:0)max_pooling2d_55/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*** 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_add_27_layer_call_and_return_conditional_losses_24585572
add_27/PartitionedCall?
 max_pooling2d_56/PartitionedCallPartitionedCalladd_27/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_max_pooling2d_56_layer_call_and_return_conditional_losses_24578512"
 max_pooling2d_56/PartitionedCall?
"conv2d_142/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_56/PartitionedCall:output:0conv2d_142_2459075conv2d_142_2459077*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_142_layer_call_and_return_conditional_losses_24585782$
"conv2d_142/StatefulPartitionedCall?
.batch_normalization_50/StatefulPartitionedCallStatefulPartitionedCall+conv2d_142/StatefulPartitionedCall:output:0batch_normalization_50_2459080batch_normalization_50_2459082batch_normalization_50_2459084batch_normalization_50_2459086*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_50_layer_call_and_return_conditional_losses_245861320
.batch_normalization_50/StatefulPartitionedCall?
"conv2d_143/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_50/StatefulPartitionedCall:output:0conv2d_143_2459089conv2d_143_2459091*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_143_layer_call_and_return_conditional_losses_24586782$
"conv2d_143/StatefulPartitionedCall?
.batch_normalization_51/StatefulPartitionedCallStatefulPartitionedCall+conv2d_143/StatefulPartitionedCall:output:0batch_normalization_51_2459094batch_normalization_51_2459096batch_normalization_51_2459098batch_normalization_51_2459100*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_51_layer_call_and_return_conditional_losses_245871320
.batch_normalization_51/StatefulPartitionedCall?
"conv2d_144/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_51/StatefulPartitionedCall:output:0conv2d_144_2459103conv2d_144_2459105*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_144_layer_call_and_return_conditional_losses_24587782$
"conv2d_144/StatefulPartitionedCall?
add_28/PartitionedCallPartitionedCall+conv2d_144/StatefulPartitionedCall:output:0)max_pooling2d_56/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_add_28_layer_call_and_return_conditional_losses_24588002
add_28/PartitionedCall?
 max_pooling2d_57/PartitionedCallPartitionedCalladd_28/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_max_pooling2d_57_layer_call_and_return_conditional_losses_24580712"
 max_pooling2d_57/PartitionedCall?
flatten_29/PartitionedCallPartitionedCall)max_pooling2d_57/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_flatten_29_layer_call_and_return_conditional_losses_24588162
flatten_29/PartitionedCall?
 dense_58/StatefulPartitionedCallStatefulPartitionedCall#flatten_29/PartitionedCall:output:0dense_58_2459111dense_58_2459113*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_58_layer_call_and_return_conditional_losses_24588352"
 dense_58/StatefulPartitionedCall?
 dense_59/StatefulPartitionedCallStatefulPartitionedCall)dense_58/StatefulPartitionedCall:output:0dense_59_2459116dense_59_2459118*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_59_layer_call_and_return_conditional_losses_24588622"
 dense_59/StatefulPartitionedCall?
IdentityIdentity)dense_59/StatefulPartitionedCall:output:0/^batch_normalization_46/StatefulPartitionedCall/^batch_normalization_47/StatefulPartitionedCall/^batch_normalization_48/StatefulPartitionedCall/^batch_normalization_49/StatefulPartitionedCall/^batch_normalization_50/StatefulPartitionedCall/^batch_normalization_51/StatefulPartitionedCall#^conv2d_136/StatefulPartitionedCall#^conv2d_137/StatefulPartitionedCall#^conv2d_138/StatefulPartitionedCall#^conv2d_139/StatefulPartitionedCall#^conv2d_140/StatefulPartitionedCall#^conv2d_141/StatefulPartitionedCall#^conv2d_142/StatefulPartitionedCall#^conv2d_143/StatefulPartitionedCall#^conv2d_144/StatefulPartitionedCall!^dense_58/StatefulPartitionedCall!^dense_59/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_46/StatefulPartitionedCall.batch_normalization_46/StatefulPartitionedCall2`
.batch_normalization_47/StatefulPartitionedCall.batch_normalization_47/StatefulPartitionedCall2`
.batch_normalization_48/StatefulPartitionedCall.batch_normalization_48/StatefulPartitionedCall2`
.batch_normalization_49/StatefulPartitionedCall.batch_normalization_49/StatefulPartitionedCall2`
.batch_normalization_50/StatefulPartitionedCall.batch_normalization_50/StatefulPartitionedCall2`
.batch_normalization_51/StatefulPartitionedCall.batch_normalization_51/StatefulPartitionedCall2H
"conv2d_136/StatefulPartitionedCall"conv2d_136/StatefulPartitionedCall2H
"conv2d_137/StatefulPartitionedCall"conv2d_137/StatefulPartitionedCall2H
"conv2d_138/StatefulPartitionedCall"conv2d_138/StatefulPartitionedCall2H
"conv2d_139/StatefulPartitionedCall"conv2d_139/StatefulPartitionedCall2H
"conv2d_140/StatefulPartitionedCall"conv2d_140/StatefulPartitionedCall2H
"conv2d_141/StatefulPartitionedCall"conv2d_141/StatefulPartitionedCall2H
"conv2d_142/StatefulPartitionedCall"conv2d_142/StatefulPartitionedCall2H
"conv2d_143/StatefulPartitionedCall"conv2d_143/StatefulPartitionedCall2H
"conv2d_144/StatefulPartitionedCall"conv2d_144/StatefulPartitionedCall2D
 dense_58/StatefulPartitionedCall dense_58/StatefulPartitionedCall2D
 dense_59/StatefulPartitionedCall dense_59/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_48_layer_call_and_return_conditional_losses_2460525

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????**:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????**2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????**::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????**
 
_user_specified_nameinputs
?
N
2__inference_max_pooling2d_56_layer_call_fn_2457857

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_max_pooling2d_56_layer_call_and_return_conditional_losses_24578512
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
,__inference_conv2d_138_layer_call_fn_2460409

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_138_layer_call_and_return_conditional_losses_24582922
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
o
C__inference_add_27_layer_call_and_return_conditional_losses_2460743
inputs_0
inputs_1
identitya
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:?????????**2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:?????????**2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:?????????**:?????????**:Y U
/
_output_shapes
:?????????**
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????**
"
_user_specified_name
inputs/1
?
?
9__inference_CNN_aug_deep_bnorm_skip_layer_call_fn_2459217
input_30
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_30unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*D
_read_only_resource_inputs&
$"	
 #$%&)*+,-.*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_CNN_aug_deep_bnorm_skip_layer_call_and_return_conditional_losses_24591222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:???????????
"
_user_specified_name
input_30
?

*__inference_dense_58_layer_call_fn_2461108

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_58_layer_call_and_return_conditional_losses_24588352
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_46_layer_call_and_return_conditional_losses_2457510

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_47_layer_call_and_return_conditional_losses_2458227

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_50_layer_call_and_return_conditional_losses_2457950

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_47_layer_call_fn_2460389

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_47_layer_call_and_return_conditional_losses_24582452
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_48_layer_call_and_return_conditional_losses_2458388

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????**:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????**2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????**::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????**
 
_user_specified_nameinputs
?	
?
E__inference_dense_58_layer_call_and_return_conditional_losses_2458835

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
G__inference_conv2d_137_layer_call_and_return_conditional_losses_2458192

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
c
G__inference_flatten_29_layer_call_and_return_conditional_losses_2458816

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
G__inference_conv2d_144_layer_call_and_return_conditional_losses_2458778

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_47_layer_call_and_return_conditional_losses_2460281

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
,__inference_conv2d_142_layer_call_fn_2460769

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_142_layer_call_and_return_conditional_losses_24585782
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_conv2d_140_layer_call_fn_2460589

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????***$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_140_layer_call_and_return_conditional_losses_24584352
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????**2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????**::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????**
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_50_layer_call_fn_2460884

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_50_layer_call_and_return_conditional_losses_24579192
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?

?
G__inference_conv2d_139_layer_call_and_return_conditional_losses_2458335

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????***
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????**2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????**2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????**::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????**
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_46_layer_call_and_return_conditional_losses_2458127

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_48_layer_call_fn_2460556

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????***$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_48_layer_call_and_return_conditional_losses_24583702
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????**2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????**::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????**
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_49_layer_call_and_return_conditional_losses_2457834

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_50_layer_call_and_return_conditional_losses_2460807

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
G__inference_conv2d_142_layer_call_and_return_conditional_losses_2460760

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_conv2d_143_layer_call_fn_2460917

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_143_layer_call_and_return_conditional_losses_24586782
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_50_layer_call_and_return_conditional_losses_2460853

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_51_layer_call_and_return_conditional_losses_2458023

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_46_layer_call_and_return_conditional_losses_2460151

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
9__inference_CNN_aug_deep_bnorm_skip_layer_call_fn_2460093

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_CNN_aug_deep_bnorm_skip_layer_call_and_return_conditional_losses_24593392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
??
?'
T__inference_CNN_aug_deep_bnorm_skip_layer_call_and_return_conditional_losses_2459726

inputs-
)conv2d_136_conv2d_readvariableop_resource.
*conv2d_136_biasadd_readvariableop_resource2
.batch_normalization_46_readvariableop_resource4
0batch_normalization_46_readvariableop_1_resourceC
?batch_normalization_46_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_46_fusedbatchnormv3_readvariableop_1_resource-
)conv2d_137_conv2d_readvariableop_resource.
*conv2d_137_biasadd_readvariableop_resource2
.batch_normalization_47_readvariableop_resource4
0batch_normalization_47_readvariableop_1_resourceC
?batch_normalization_47_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_47_fusedbatchnormv3_readvariableop_1_resource-
)conv2d_138_conv2d_readvariableop_resource.
*conv2d_138_biasadd_readvariableop_resource-
)conv2d_139_conv2d_readvariableop_resource.
*conv2d_139_biasadd_readvariableop_resource2
.batch_normalization_48_readvariableop_resource4
0batch_normalization_48_readvariableop_1_resourceC
?batch_normalization_48_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_48_fusedbatchnormv3_readvariableop_1_resource-
)conv2d_140_conv2d_readvariableop_resource.
*conv2d_140_biasadd_readvariableop_resource2
.batch_normalization_49_readvariableop_resource4
0batch_normalization_49_readvariableop_1_resourceC
?batch_normalization_49_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_49_fusedbatchnormv3_readvariableop_1_resource-
)conv2d_141_conv2d_readvariableop_resource.
*conv2d_141_biasadd_readvariableop_resource-
)conv2d_142_conv2d_readvariableop_resource.
*conv2d_142_biasadd_readvariableop_resource2
.batch_normalization_50_readvariableop_resource4
0batch_normalization_50_readvariableop_1_resourceC
?batch_normalization_50_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_50_fusedbatchnormv3_readvariableop_1_resource-
)conv2d_143_conv2d_readvariableop_resource.
*conv2d_143_biasadd_readvariableop_resource2
.batch_normalization_51_readvariableop_resource4
0batch_normalization_51_readvariableop_1_resourceC
?batch_normalization_51_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_51_fusedbatchnormv3_readvariableop_1_resource-
)conv2d_144_conv2d_readvariableop_resource.
*conv2d_144_biasadd_readvariableop_resource+
'dense_58_matmul_readvariableop_resource,
(dense_58_biasadd_readvariableop_resource+
'dense_59_matmul_readvariableop_resource,
(dense_59_biasadd_readvariableop_resource
identity??%batch_normalization_46/AssignNewValue?'batch_normalization_46/AssignNewValue_1?6batch_normalization_46/FusedBatchNormV3/ReadVariableOp?8batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_46/ReadVariableOp?'batch_normalization_46/ReadVariableOp_1?%batch_normalization_47/AssignNewValue?'batch_normalization_47/AssignNewValue_1?6batch_normalization_47/FusedBatchNormV3/ReadVariableOp?8batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_47/ReadVariableOp?'batch_normalization_47/ReadVariableOp_1?%batch_normalization_48/AssignNewValue?'batch_normalization_48/AssignNewValue_1?6batch_normalization_48/FusedBatchNormV3/ReadVariableOp?8batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_48/ReadVariableOp?'batch_normalization_48/ReadVariableOp_1?%batch_normalization_49/AssignNewValue?'batch_normalization_49/AssignNewValue_1?6batch_normalization_49/FusedBatchNormV3/ReadVariableOp?8batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_49/ReadVariableOp?'batch_normalization_49/ReadVariableOp_1?%batch_normalization_50/AssignNewValue?'batch_normalization_50/AssignNewValue_1?6batch_normalization_50/FusedBatchNormV3/ReadVariableOp?8batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_50/ReadVariableOp?'batch_normalization_50/ReadVariableOp_1?%batch_normalization_51/AssignNewValue?'batch_normalization_51/AssignNewValue_1?6batch_normalization_51/FusedBatchNormV3/ReadVariableOp?8batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_51/ReadVariableOp?'batch_normalization_51/ReadVariableOp_1?!conv2d_136/BiasAdd/ReadVariableOp? conv2d_136/Conv2D/ReadVariableOp?!conv2d_137/BiasAdd/ReadVariableOp? conv2d_137/Conv2D/ReadVariableOp?!conv2d_138/BiasAdd/ReadVariableOp? conv2d_138/Conv2D/ReadVariableOp?!conv2d_139/BiasAdd/ReadVariableOp? conv2d_139/Conv2D/ReadVariableOp?!conv2d_140/BiasAdd/ReadVariableOp? conv2d_140/Conv2D/ReadVariableOp?!conv2d_141/BiasAdd/ReadVariableOp? conv2d_141/Conv2D/ReadVariableOp?!conv2d_142/BiasAdd/ReadVariableOp? conv2d_142/Conv2D/ReadVariableOp?!conv2d_143/BiasAdd/ReadVariableOp? conv2d_143/Conv2D/ReadVariableOp?!conv2d_144/BiasAdd/ReadVariableOp? conv2d_144/Conv2D/ReadVariableOp?dense_58/BiasAdd/ReadVariableOp?dense_58/MatMul/ReadVariableOp?dense_59/BiasAdd/ReadVariableOp?dense_59/MatMul/ReadVariableOp?
 conv2d_136/Conv2D/ReadVariableOpReadVariableOp)conv2d_136_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_136/Conv2D/ReadVariableOp?
conv2d_136/Conv2DConv2Dinputs(conv2d_136/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_136/Conv2D?
!conv2d_136/BiasAdd/ReadVariableOpReadVariableOp*conv2d_136_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_136/BiasAdd/ReadVariableOp?
conv2d_136/BiasAddBiasAddconv2d_136/Conv2D:output:0)conv2d_136/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_136/BiasAdd?
conv2d_136/ReluReluconv2d_136/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_136/Relu?
%batch_normalization_46/ReadVariableOpReadVariableOp.batch_normalization_46_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_46/ReadVariableOp?
'batch_normalization_46/ReadVariableOp_1ReadVariableOp0batch_normalization_46_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_46/ReadVariableOp_1?
6batch_normalization_46/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_46_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_46/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_46_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_46/FusedBatchNormV3FusedBatchNormV3conv2d_136/Relu:activations:0-batch_normalization_46/ReadVariableOp:value:0/batch_normalization_46/ReadVariableOp_1:value:0>batch_normalization_46/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_46/FusedBatchNormV3?
%batch_normalization_46/AssignNewValueAssignVariableOp?batch_normalization_46_fusedbatchnormv3_readvariableop_resource4batch_normalization_46/FusedBatchNormV3:batch_mean:07^batch_normalization_46/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*R
_classH
FDloc:@batch_normalization_46/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_46/AssignNewValue?
'batch_normalization_46/AssignNewValue_1AssignVariableOpAbatch_normalization_46_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_46/FusedBatchNormV3:batch_variance:09^batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_46/AssignNewValue_1?
 conv2d_137/Conv2D/ReadVariableOpReadVariableOp)conv2d_137_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_137/Conv2D/ReadVariableOp?
conv2d_137/Conv2DConv2D+batch_normalization_46/FusedBatchNormV3:y:0(conv2d_137/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_137/Conv2D?
!conv2d_137/BiasAdd/ReadVariableOpReadVariableOp*conv2d_137_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_137/BiasAdd/ReadVariableOp?
conv2d_137/BiasAddBiasAddconv2d_137/Conv2D:output:0)conv2d_137/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_137/BiasAdd?
conv2d_137/ReluReluconv2d_137/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_137/Relu?
%batch_normalization_47/ReadVariableOpReadVariableOp.batch_normalization_47_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_47/ReadVariableOp?
'batch_normalization_47/ReadVariableOp_1ReadVariableOp0batch_normalization_47_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_47/ReadVariableOp_1?
6batch_normalization_47/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_47_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_47/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_47_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_47/FusedBatchNormV3FusedBatchNormV3conv2d_137/Relu:activations:0-batch_normalization_47/ReadVariableOp:value:0/batch_normalization_47/ReadVariableOp_1:value:0>batch_normalization_47/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_47/FusedBatchNormV3?
%batch_normalization_47/AssignNewValueAssignVariableOp?batch_normalization_47_fusedbatchnormv3_readvariableop_resource4batch_normalization_47/FusedBatchNormV3:batch_mean:07^batch_normalization_47/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*R
_classH
FDloc:@batch_normalization_47/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_47/AssignNewValue?
'batch_normalization_47/AssignNewValue_1AssignVariableOpAbatch_normalization_47_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_47/FusedBatchNormV3:batch_variance:09^batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_47/AssignNewValue_1?
 conv2d_138/Conv2D/ReadVariableOpReadVariableOp)conv2d_138_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_138/Conv2D/ReadVariableOp?
conv2d_138/Conv2DConv2D+batch_normalization_47/FusedBatchNormV3:y:0(conv2d_138/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_138/Conv2D?
!conv2d_138/BiasAdd/ReadVariableOpReadVariableOp*conv2d_138_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_138/BiasAdd/ReadVariableOp?
conv2d_138/BiasAddBiasAddconv2d_138/Conv2D:output:0)conv2d_138/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_138/BiasAdd?
conv2d_138/ReluReluconv2d_138/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_138/Relu?

add_26/addAddV2conv2d_138/Relu:activations:0inputs*
T0*1
_output_shapes
:???????????2

add_26/add?
max_pooling2d_55/MaxPoolMaxPooladd_26/add:z:0*/
_output_shapes
:?????????***
ksize
*
paddingVALID*
strides
2
max_pooling2d_55/MaxPool?
 conv2d_139/Conv2D/ReadVariableOpReadVariableOp)conv2d_139_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_139/Conv2D/ReadVariableOp?
conv2d_139/Conv2DConv2D!max_pooling2d_55/MaxPool:output:0(conv2d_139/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????***
paddingSAME*
strides
2
conv2d_139/Conv2D?
!conv2d_139/BiasAdd/ReadVariableOpReadVariableOp*conv2d_139_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_139/BiasAdd/ReadVariableOp?
conv2d_139/BiasAddBiasAddconv2d_139/Conv2D:output:0)conv2d_139/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**2
conv2d_139/BiasAdd?
conv2d_139/ReluReluconv2d_139/BiasAdd:output:0*
T0*/
_output_shapes
:?????????**2
conv2d_139/Relu?
%batch_normalization_48/ReadVariableOpReadVariableOp.batch_normalization_48_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_48/ReadVariableOp?
'batch_normalization_48/ReadVariableOp_1ReadVariableOp0batch_normalization_48_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_48/ReadVariableOp_1?
6batch_normalization_48/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_48_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_48/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_48_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_48/FusedBatchNormV3FusedBatchNormV3conv2d_139/Relu:activations:0-batch_normalization_48/ReadVariableOp:value:0/batch_normalization_48/ReadVariableOp_1:value:0>batch_normalization_48/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????**:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_48/FusedBatchNormV3?
%batch_normalization_48/AssignNewValueAssignVariableOp?batch_normalization_48_fusedbatchnormv3_readvariableop_resource4batch_normalization_48/FusedBatchNormV3:batch_mean:07^batch_normalization_48/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*R
_classH
FDloc:@batch_normalization_48/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_48/AssignNewValue?
'batch_normalization_48/AssignNewValue_1AssignVariableOpAbatch_normalization_48_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_48/FusedBatchNormV3:batch_variance:09^batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_48/AssignNewValue_1?
 conv2d_140/Conv2D/ReadVariableOpReadVariableOp)conv2d_140_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_140/Conv2D/ReadVariableOp?
conv2d_140/Conv2DConv2D+batch_normalization_48/FusedBatchNormV3:y:0(conv2d_140/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????***
paddingSAME*
strides
2
conv2d_140/Conv2D?
!conv2d_140/BiasAdd/ReadVariableOpReadVariableOp*conv2d_140_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_140/BiasAdd/ReadVariableOp?
conv2d_140/BiasAddBiasAddconv2d_140/Conv2D:output:0)conv2d_140/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**2
conv2d_140/BiasAdd?
conv2d_140/ReluReluconv2d_140/BiasAdd:output:0*
T0*/
_output_shapes
:?????????**2
conv2d_140/Relu?
%batch_normalization_49/ReadVariableOpReadVariableOp.batch_normalization_49_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_49/ReadVariableOp?
'batch_normalization_49/ReadVariableOp_1ReadVariableOp0batch_normalization_49_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_49/ReadVariableOp_1?
6batch_normalization_49/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_49_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_49/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_49_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_49/FusedBatchNormV3FusedBatchNormV3conv2d_140/Relu:activations:0-batch_normalization_49/ReadVariableOp:value:0/batch_normalization_49/ReadVariableOp_1:value:0>batch_normalization_49/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????**:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_49/FusedBatchNormV3?
%batch_normalization_49/AssignNewValueAssignVariableOp?batch_normalization_49_fusedbatchnormv3_readvariableop_resource4batch_normalization_49/FusedBatchNormV3:batch_mean:07^batch_normalization_49/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*R
_classH
FDloc:@batch_normalization_49/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_49/AssignNewValue?
'batch_normalization_49/AssignNewValue_1AssignVariableOpAbatch_normalization_49_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_49/FusedBatchNormV3:batch_variance:09^batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_49/AssignNewValue_1?
 conv2d_141/Conv2D/ReadVariableOpReadVariableOp)conv2d_141_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_141/Conv2D/ReadVariableOp?
conv2d_141/Conv2DConv2D+batch_normalization_49/FusedBatchNormV3:y:0(conv2d_141/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????***
paddingSAME*
strides
2
conv2d_141/Conv2D?
!conv2d_141/BiasAdd/ReadVariableOpReadVariableOp*conv2d_141_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_141/BiasAdd/ReadVariableOp?
conv2d_141/BiasAddBiasAddconv2d_141/Conv2D:output:0)conv2d_141/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**2
conv2d_141/BiasAdd?
conv2d_141/ReluReluconv2d_141/BiasAdd:output:0*
T0*/
_output_shapes
:?????????**2
conv2d_141/Relu?

add_27/addAddV2conv2d_141/Relu:activations:0!max_pooling2d_55/MaxPool:output:0*
T0*/
_output_shapes
:?????????**2

add_27/add?
max_pooling2d_56/MaxPoolMaxPooladd_27/add:z:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_56/MaxPool?
 conv2d_142/Conv2D/ReadVariableOpReadVariableOp)conv2d_142_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_142/Conv2D/ReadVariableOp?
conv2d_142/Conv2DConv2D!max_pooling2d_56/MaxPool:output:0(conv2d_142/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_142/Conv2D?
!conv2d_142/BiasAdd/ReadVariableOpReadVariableOp*conv2d_142_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_142/BiasAdd/ReadVariableOp?
conv2d_142/BiasAddBiasAddconv2d_142/Conv2D:output:0)conv2d_142/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_142/BiasAdd?
conv2d_142/ReluReluconv2d_142/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_142/Relu?
%batch_normalization_50/ReadVariableOpReadVariableOp.batch_normalization_50_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_50/ReadVariableOp?
'batch_normalization_50/ReadVariableOp_1ReadVariableOp0batch_normalization_50_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_50/ReadVariableOp_1?
6batch_normalization_50/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_50_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_50/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_50_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_50/FusedBatchNormV3FusedBatchNormV3conv2d_142/Relu:activations:0-batch_normalization_50/ReadVariableOp:value:0/batch_normalization_50/ReadVariableOp_1:value:0>batch_normalization_50/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_50/FusedBatchNormV3?
%batch_normalization_50/AssignNewValueAssignVariableOp?batch_normalization_50_fusedbatchnormv3_readvariableop_resource4batch_normalization_50/FusedBatchNormV3:batch_mean:07^batch_normalization_50/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*R
_classH
FDloc:@batch_normalization_50/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_50/AssignNewValue?
'batch_normalization_50/AssignNewValue_1AssignVariableOpAbatch_normalization_50_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_50/FusedBatchNormV3:batch_variance:09^batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_50/AssignNewValue_1?
 conv2d_143/Conv2D/ReadVariableOpReadVariableOp)conv2d_143_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_143/Conv2D/ReadVariableOp?
conv2d_143/Conv2DConv2D+batch_normalization_50/FusedBatchNormV3:y:0(conv2d_143/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_143/Conv2D?
!conv2d_143/BiasAdd/ReadVariableOpReadVariableOp*conv2d_143_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_143/BiasAdd/ReadVariableOp?
conv2d_143/BiasAddBiasAddconv2d_143/Conv2D:output:0)conv2d_143/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_143/BiasAdd?
conv2d_143/ReluReluconv2d_143/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_143/Relu?
%batch_normalization_51/ReadVariableOpReadVariableOp.batch_normalization_51_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_51/ReadVariableOp?
'batch_normalization_51/ReadVariableOp_1ReadVariableOp0batch_normalization_51_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_51/ReadVariableOp_1?
6batch_normalization_51/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_51_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_51/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_51_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_51/FusedBatchNormV3FusedBatchNormV3conv2d_143/Relu:activations:0-batch_normalization_51/ReadVariableOp:value:0/batch_normalization_51/ReadVariableOp_1:value:0>batch_normalization_51/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_51/FusedBatchNormV3?
%batch_normalization_51/AssignNewValueAssignVariableOp?batch_normalization_51_fusedbatchnormv3_readvariableop_resource4batch_normalization_51/FusedBatchNormV3:batch_mean:07^batch_normalization_51/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*R
_classH
FDloc:@batch_normalization_51/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_51/AssignNewValue?
'batch_normalization_51/AssignNewValue_1AssignVariableOpAbatch_normalization_51_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_51/FusedBatchNormV3:batch_variance:09^batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_51/AssignNewValue_1?
 conv2d_144/Conv2D/ReadVariableOpReadVariableOp)conv2d_144_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_144/Conv2D/ReadVariableOp?
conv2d_144/Conv2DConv2D+batch_normalization_51/FusedBatchNormV3:y:0(conv2d_144/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_144/Conv2D?
!conv2d_144/BiasAdd/ReadVariableOpReadVariableOp*conv2d_144_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_144/BiasAdd/ReadVariableOp?
conv2d_144/BiasAddBiasAddconv2d_144/Conv2D:output:0)conv2d_144/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_144/BiasAdd?
conv2d_144/ReluReluconv2d_144/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_144/Relu?

add_28/addAddV2conv2d_144/Relu:activations:0!max_pooling2d_56/MaxPool:output:0*
T0*/
_output_shapes
:?????????2

add_28/add?
max_pooling2d_57/MaxPoolMaxPooladd_28/add:z:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_57/MaxPoolu
flatten_29/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_29/Const?
flatten_29/ReshapeReshape!max_pooling2d_57/MaxPool:output:0flatten_29/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_29/Reshape?
dense_58/MatMul/ReadVariableOpReadVariableOp'dense_58_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_58/MatMul/ReadVariableOp?
dense_58/MatMulMatMulflatten_29/Reshape:output:0&dense_58/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_58/MatMul?
dense_58/BiasAdd/ReadVariableOpReadVariableOp(dense_58_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_58/BiasAdd/ReadVariableOp?
dense_58/BiasAddBiasAdddense_58/MatMul:product:0'dense_58/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_58/BiasAdds
dense_58/ReluReludense_58/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_58/Relu?
dense_59/MatMul/ReadVariableOpReadVariableOp'dense_59_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_59/MatMul/ReadVariableOp?
dense_59/MatMulMatMuldense_58/Relu:activations:0&dense_59/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_59/MatMul?
dense_59/BiasAdd/ReadVariableOpReadVariableOp(dense_59_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_59/BiasAdd/ReadVariableOp?
dense_59/BiasAddBiasAdddense_59/MatMul:product:0'dense_59/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_59/BiasAdd|
dense_59/SoftmaxSoftmaxdense_59/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_59/Softmax?
IdentityIdentitydense_59/Softmax:softmax:0&^batch_normalization_46/AssignNewValue(^batch_normalization_46/AssignNewValue_17^batch_normalization_46/FusedBatchNormV3/ReadVariableOp9^batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_46/ReadVariableOp(^batch_normalization_46/ReadVariableOp_1&^batch_normalization_47/AssignNewValue(^batch_normalization_47/AssignNewValue_17^batch_normalization_47/FusedBatchNormV3/ReadVariableOp9^batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_47/ReadVariableOp(^batch_normalization_47/ReadVariableOp_1&^batch_normalization_48/AssignNewValue(^batch_normalization_48/AssignNewValue_17^batch_normalization_48/FusedBatchNormV3/ReadVariableOp9^batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_48/ReadVariableOp(^batch_normalization_48/ReadVariableOp_1&^batch_normalization_49/AssignNewValue(^batch_normalization_49/AssignNewValue_17^batch_normalization_49/FusedBatchNormV3/ReadVariableOp9^batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_49/ReadVariableOp(^batch_normalization_49/ReadVariableOp_1&^batch_normalization_50/AssignNewValue(^batch_normalization_50/AssignNewValue_17^batch_normalization_50/FusedBatchNormV3/ReadVariableOp9^batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_50/ReadVariableOp(^batch_normalization_50/ReadVariableOp_1&^batch_normalization_51/AssignNewValue(^batch_normalization_51/AssignNewValue_17^batch_normalization_51/FusedBatchNormV3/ReadVariableOp9^batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_51/ReadVariableOp(^batch_normalization_51/ReadVariableOp_1"^conv2d_136/BiasAdd/ReadVariableOp!^conv2d_136/Conv2D/ReadVariableOp"^conv2d_137/BiasAdd/ReadVariableOp!^conv2d_137/Conv2D/ReadVariableOp"^conv2d_138/BiasAdd/ReadVariableOp!^conv2d_138/Conv2D/ReadVariableOp"^conv2d_139/BiasAdd/ReadVariableOp!^conv2d_139/Conv2D/ReadVariableOp"^conv2d_140/BiasAdd/ReadVariableOp!^conv2d_140/Conv2D/ReadVariableOp"^conv2d_141/BiasAdd/ReadVariableOp!^conv2d_141/Conv2D/ReadVariableOp"^conv2d_142/BiasAdd/ReadVariableOp!^conv2d_142/Conv2D/ReadVariableOp"^conv2d_143/BiasAdd/ReadVariableOp!^conv2d_143/Conv2D/ReadVariableOp"^conv2d_144/BiasAdd/ReadVariableOp!^conv2d_144/Conv2D/ReadVariableOp ^dense_58/BiasAdd/ReadVariableOp^dense_58/MatMul/ReadVariableOp ^dense_59/BiasAdd/ReadVariableOp^dense_59/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::::::::2N
%batch_normalization_46/AssignNewValue%batch_normalization_46/AssignNewValue2R
'batch_normalization_46/AssignNewValue_1'batch_normalization_46/AssignNewValue_12p
6batch_normalization_46/FusedBatchNormV3/ReadVariableOp6batch_normalization_46/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_46/FusedBatchNormV3/ReadVariableOp_18batch_normalization_46/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_46/ReadVariableOp%batch_normalization_46/ReadVariableOp2R
'batch_normalization_46/ReadVariableOp_1'batch_normalization_46/ReadVariableOp_12N
%batch_normalization_47/AssignNewValue%batch_normalization_47/AssignNewValue2R
'batch_normalization_47/AssignNewValue_1'batch_normalization_47/AssignNewValue_12p
6batch_normalization_47/FusedBatchNormV3/ReadVariableOp6batch_normalization_47/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_47/FusedBatchNormV3/ReadVariableOp_18batch_normalization_47/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_47/ReadVariableOp%batch_normalization_47/ReadVariableOp2R
'batch_normalization_47/ReadVariableOp_1'batch_normalization_47/ReadVariableOp_12N
%batch_normalization_48/AssignNewValue%batch_normalization_48/AssignNewValue2R
'batch_normalization_48/AssignNewValue_1'batch_normalization_48/AssignNewValue_12p
6batch_normalization_48/FusedBatchNormV3/ReadVariableOp6batch_normalization_48/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_48/FusedBatchNormV3/ReadVariableOp_18batch_normalization_48/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_48/ReadVariableOp%batch_normalization_48/ReadVariableOp2R
'batch_normalization_48/ReadVariableOp_1'batch_normalization_48/ReadVariableOp_12N
%batch_normalization_49/AssignNewValue%batch_normalization_49/AssignNewValue2R
'batch_normalization_49/AssignNewValue_1'batch_normalization_49/AssignNewValue_12p
6batch_normalization_49/FusedBatchNormV3/ReadVariableOp6batch_normalization_49/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_49/FusedBatchNormV3/ReadVariableOp_18batch_normalization_49/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_49/ReadVariableOp%batch_normalization_49/ReadVariableOp2R
'batch_normalization_49/ReadVariableOp_1'batch_normalization_49/ReadVariableOp_12N
%batch_normalization_50/AssignNewValue%batch_normalization_50/AssignNewValue2R
'batch_normalization_50/AssignNewValue_1'batch_normalization_50/AssignNewValue_12p
6batch_normalization_50/FusedBatchNormV3/ReadVariableOp6batch_normalization_50/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_50/FusedBatchNormV3/ReadVariableOp_18batch_normalization_50/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_50/ReadVariableOp%batch_normalization_50/ReadVariableOp2R
'batch_normalization_50/ReadVariableOp_1'batch_normalization_50/ReadVariableOp_12N
%batch_normalization_51/AssignNewValue%batch_normalization_51/AssignNewValue2R
'batch_normalization_51/AssignNewValue_1'batch_normalization_51/AssignNewValue_12p
6batch_normalization_51/FusedBatchNormV3/ReadVariableOp6batch_normalization_51/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_51/FusedBatchNormV3/ReadVariableOp_18batch_normalization_51/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_51/ReadVariableOp%batch_normalization_51/ReadVariableOp2R
'batch_normalization_51/ReadVariableOp_1'batch_normalization_51/ReadVariableOp_12F
!conv2d_136/BiasAdd/ReadVariableOp!conv2d_136/BiasAdd/ReadVariableOp2D
 conv2d_136/Conv2D/ReadVariableOp conv2d_136/Conv2D/ReadVariableOp2F
!conv2d_137/BiasAdd/ReadVariableOp!conv2d_137/BiasAdd/ReadVariableOp2D
 conv2d_137/Conv2D/ReadVariableOp conv2d_137/Conv2D/ReadVariableOp2F
!conv2d_138/BiasAdd/ReadVariableOp!conv2d_138/BiasAdd/ReadVariableOp2D
 conv2d_138/Conv2D/ReadVariableOp conv2d_138/Conv2D/ReadVariableOp2F
!conv2d_139/BiasAdd/ReadVariableOp!conv2d_139/BiasAdd/ReadVariableOp2D
 conv2d_139/Conv2D/ReadVariableOp conv2d_139/Conv2D/ReadVariableOp2F
!conv2d_140/BiasAdd/ReadVariableOp!conv2d_140/BiasAdd/ReadVariableOp2D
 conv2d_140/Conv2D/ReadVariableOp conv2d_140/Conv2D/ReadVariableOp2F
!conv2d_141/BiasAdd/ReadVariableOp!conv2d_141/BiasAdd/ReadVariableOp2D
 conv2d_141/Conv2D/ReadVariableOp conv2d_141/Conv2D/ReadVariableOp2F
!conv2d_142/BiasAdd/ReadVariableOp!conv2d_142/BiasAdd/ReadVariableOp2D
 conv2d_142/Conv2D/ReadVariableOp conv2d_142/Conv2D/ReadVariableOp2F
!conv2d_143/BiasAdd/ReadVariableOp!conv2d_143/BiasAdd/ReadVariableOp2D
 conv2d_143/Conv2D/ReadVariableOp conv2d_143/Conv2D/ReadVariableOp2F
!conv2d_144/BiasAdd/ReadVariableOp!conv2d_144/BiasAdd/ReadVariableOp2D
 conv2d_144/Conv2D/ReadVariableOp conv2d_144/Conv2D/ReadVariableOp2B
dense_58/BiasAdd/ReadVariableOpdense_58/BiasAdd/ReadVariableOp2@
dense_58/MatMul/ReadVariableOpdense_58/MatMul/ReadVariableOp2B
dense_59/BiasAdd/ReadVariableOpdense_59/BiasAdd/ReadVariableOp2@
dense_59/MatMul/ReadVariableOpdense_59/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
T__inference_CNN_aug_deep_bnorm_skip_layer_call_and_return_conditional_losses_2458879
input_30
conv2d_136_2458103
conv2d_136_2458105"
batch_normalization_46_2458172"
batch_normalization_46_2458174"
batch_normalization_46_2458176"
batch_normalization_46_2458178
conv2d_137_2458203
conv2d_137_2458205"
batch_normalization_47_2458272"
batch_normalization_47_2458274"
batch_normalization_47_2458276"
batch_normalization_47_2458278
conv2d_138_2458303
conv2d_138_2458305
conv2d_139_2458346
conv2d_139_2458348"
batch_normalization_48_2458415"
batch_normalization_48_2458417"
batch_normalization_48_2458419"
batch_normalization_48_2458421
conv2d_140_2458446
conv2d_140_2458448"
batch_normalization_49_2458515"
batch_normalization_49_2458517"
batch_normalization_49_2458519"
batch_normalization_49_2458521
conv2d_141_2458546
conv2d_141_2458548
conv2d_142_2458589
conv2d_142_2458591"
batch_normalization_50_2458658"
batch_normalization_50_2458660"
batch_normalization_50_2458662"
batch_normalization_50_2458664
conv2d_143_2458689
conv2d_143_2458691"
batch_normalization_51_2458758"
batch_normalization_51_2458760"
batch_normalization_51_2458762"
batch_normalization_51_2458764
conv2d_144_2458789
conv2d_144_2458791
dense_58_2458846
dense_58_2458848
dense_59_2458873
dense_59_2458875
identity??.batch_normalization_46/StatefulPartitionedCall?.batch_normalization_47/StatefulPartitionedCall?.batch_normalization_48/StatefulPartitionedCall?.batch_normalization_49/StatefulPartitionedCall?.batch_normalization_50/StatefulPartitionedCall?.batch_normalization_51/StatefulPartitionedCall?"conv2d_136/StatefulPartitionedCall?"conv2d_137/StatefulPartitionedCall?"conv2d_138/StatefulPartitionedCall?"conv2d_139/StatefulPartitionedCall?"conv2d_140/StatefulPartitionedCall?"conv2d_141/StatefulPartitionedCall?"conv2d_142/StatefulPartitionedCall?"conv2d_143/StatefulPartitionedCall?"conv2d_144/StatefulPartitionedCall? dense_58/StatefulPartitionedCall? dense_59/StatefulPartitionedCall?
"conv2d_136/StatefulPartitionedCallStatefulPartitionedCallinput_30conv2d_136_2458103conv2d_136_2458105*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_136_layer_call_and_return_conditional_losses_24580922$
"conv2d_136/StatefulPartitionedCall?
.batch_normalization_46/StatefulPartitionedCallStatefulPartitionedCall+conv2d_136/StatefulPartitionedCall:output:0batch_normalization_46_2458172batch_normalization_46_2458174batch_normalization_46_2458176batch_normalization_46_2458178*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_46_layer_call_and_return_conditional_losses_245812720
.batch_normalization_46/StatefulPartitionedCall?
"conv2d_137/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_46/StatefulPartitionedCall:output:0conv2d_137_2458203conv2d_137_2458205*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_137_layer_call_and_return_conditional_losses_24581922$
"conv2d_137/StatefulPartitionedCall?
.batch_normalization_47/StatefulPartitionedCallStatefulPartitionedCall+conv2d_137/StatefulPartitionedCall:output:0batch_normalization_47_2458272batch_normalization_47_2458274batch_normalization_47_2458276batch_normalization_47_2458278*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_47_layer_call_and_return_conditional_losses_245822720
.batch_normalization_47/StatefulPartitionedCall?
"conv2d_138/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_47/StatefulPartitionedCall:output:0conv2d_138_2458303conv2d_138_2458305*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_138_layer_call_and_return_conditional_losses_24582922$
"conv2d_138/StatefulPartitionedCall?
add_26/PartitionedCallPartitionedCall+conv2d_138/StatefulPartitionedCall:output:0input_30*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_add_26_layer_call_and_return_conditional_losses_24583142
add_26/PartitionedCall?
 max_pooling2d_55/PartitionedCallPartitionedCalladd_26/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*** 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_max_pooling2d_55_layer_call_and_return_conditional_losses_24576312"
 max_pooling2d_55/PartitionedCall?
"conv2d_139/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_55/PartitionedCall:output:0conv2d_139_2458346conv2d_139_2458348*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????***$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_139_layer_call_and_return_conditional_losses_24583352$
"conv2d_139/StatefulPartitionedCall?
.batch_normalization_48/StatefulPartitionedCallStatefulPartitionedCall+conv2d_139/StatefulPartitionedCall:output:0batch_normalization_48_2458415batch_normalization_48_2458417batch_normalization_48_2458419batch_normalization_48_2458421*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????***$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_48_layer_call_and_return_conditional_losses_245837020
.batch_normalization_48/StatefulPartitionedCall?
"conv2d_140/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_48/StatefulPartitionedCall:output:0conv2d_140_2458446conv2d_140_2458448*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????***$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_140_layer_call_and_return_conditional_losses_24584352$
"conv2d_140/StatefulPartitionedCall?
.batch_normalization_49/StatefulPartitionedCallStatefulPartitionedCall+conv2d_140/StatefulPartitionedCall:output:0batch_normalization_49_2458515batch_normalization_49_2458517batch_normalization_49_2458519batch_normalization_49_2458521*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????***$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_49_layer_call_and_return_conditional_losses_245847020
.batch_normalization_49/StatefulPartitionedCall?
"conv2d_141/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_49/StatefulPartitionedCall:output:0conv2d_141_2458546conv2d_141_2458548*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????***$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_141_layer_call_and_return_conditional_losses_24585352$
"conv2d_141/StatefulPartitionedCall?
add_27/PartitionedCallPartitionedCall+conv2d_141/StatefulPartitionedCall:output:0)max_pooling2d_55/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*** 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_add_27_layer_call_and_return_conditional_losses_24585572
add_27/PartitionedCall?
 max_pooling2d_56/PartitionedCallPartitionedCalladd_27/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_max_pooling2d_56_layer_call_and_return_conditional_losses_24578512"
 max_pooling2d_56/PartitionedCall?
"conv2d_142/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_56/PartitionedCall:output:0conv2d_142_2458589conv2d_142_2458591*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_142_layer_call_and_return_conditional_losses_24585782$
"conv2d_142/StatefulPartitionedCall?
.batch_normalization_50/StatefulPartitionedCallStatefulPartitionedCall+conv2d_142/StatefulPartitionedCall:output:0batch_normalization_50_2458658batch_normalization_50_2458660batch_normalization_50_2458662batch_normalization_50_2458664*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_50_layer_call_and_return_conditional_losses_245861320
.batch_normalization_50/StatefulPartitionedCall?
"conv2d_143/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_50/StatefulPartitionedCall:output:0conv2d_143_2458689conv2d_143_2458691*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_143_layer_call_and_return_conditional_losses_24586782$
"conv2d_143/StatefulPartitionedCall?
.batch_normalization_51/StatefulPartitionedCallStatefulPartitionedCall+conv2d_143/StatefulPartitionedCall:output:0batch_normalization_51_2458758batch_normalization_51_2458760batch_normalization_51_2458762batch_normalization_51_2458764*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_51_layer_call_and_return_conditional_losses_245871320
.batch_normalization_51/StatefulPartitionedCall?
"conv2d_144/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_51/StatefulPartitionedCall:output:0conv2d_144_2458789conv2d_144_2458791*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_144_layer_call_and_return_conditional_losses_24587782$
"conv2d_144/StatefulPartitionedCall?
add_28/PartitionedCallPartitionedCall+conv2d_144/StatefulPartitionedCall:output:0)max_pooling2d_56/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_add_28_layer_call_and_return_conditional_losses_24588002
add_28/PartitionedCall?
 max_pooling2d_57/PartitionedCallPartitionedCalladd_28/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_max_pooling2d_57_layer_call_and_return_conditional_losses_24580712"
 max_pooling2d_57/PartitionedCall?
flatten_29/PartitionedCallPartitionedCall)max_pooling2d_57/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_flatten_29_layer_call_and_return_conditional_losses_24588162
flatten_29/PartitionedCall?
 dense_58/StatefulPartitionedCallStatefulPartitionedCall#flatten_29/PartitionedCall:output:0dense_58_2458846dense_58_2458848*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_58_layer_call_and_return_conditional_losses_24588352"
 dense_58/StatefulPartitionedCall?
 dense_59/StatefulPartitionedCallStatefulPartitionedCall)dense_58/StatefulPartitionedCall:output:0dense_59_2458873dense_59_2458875*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_59_layer_call_and_return_conditional_losses_24588622"
 dense_59/StatefulPartitionedCall?
IdentityIdentity)dense_59/StatefulPartitionedCall:output:0/^batch_normalization_46/StatefulPartitionedCall/^batch_normalization_47/StatefulPartitionedCall/^batch_normalization_48/StatefulPartitionedCall/^batch_normalization_49/StatefulPartitionedCall/^batch_normalization_50/StatefulPartitionedCall/^batch_normalization_51/StatefulPartitionedCall#^conv2d_136/StatefulPartitionedCall#^conv2d_137/StatefulPartitionedCall#^conv2d_138/StatefulPartitionedCall#^conv2d_139/StatefulPartitionedCall#^conv2d_140/StatefulPartitionedCall#^conv2d_141/StatefulPartitionedCall#^conv2d_142/StatefulPartitionedCall#^conv2d_143/StatefulPartitionedCall#^conv2d_144/StatefulPartitionedCall!^dense_58/StatefulPartitionedCall!^dense_59/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_46/StatefulPartitionedCall.batch_normalization_46/StatefulPartitionedCall2`
.batch_normalization_47/StatefulPartitionedCall.batch_normalization_47/StatefulPartitionedCall2`
.batch_normalization_48/StatefulPartitionedCall.batch_normalization_48/StatefulPartitionedCall2`
.batch_normalization_49/StatefulPartitionedCall.batch_normalization_49/StatefulPartitionedCall2`
.batch_normalization_50/StatefulPartitionedCall.batch_normalization_50/StatefulPartitionedCall2`
.batch_normalization_51/StatefulPartitionedCall.batch_normalization_51/StatefulPartitionedCall2H
"conv2d_136/StatefulPartitionedCall"conv2d_136/StatefulPartitionedCall2H
"conv2d_137/StatefulPartitionedCall"conv2d_137/StatefulPartitionedCall2H
"conv2d_138/StatefulPartitionedCall"conv2d_138/StatefulPartitionedCall2H
"conv2d_139/StatefulPartitionedCall"conv2d_139/StatefulPartitionedCall2H
"conv2d_140/StatefulPartitionedCall"conv2d_140/StatefulPartitionedCall2H
"conv2d_141/StatefulPartitionedCall"conv2d_141/StatefulPartitionedCall2H
"conv2d_142/StatefulPartitionedCall"conv2d_142/StatefulPartitionedCall2H
"conv2d_143/StatefulPartitionedCall"conv2d_143/StatefulPartitionedCall2H
"conv2d_144/StatefulPartitionedCall"conv2d_144/StatefulPartitionedCall2D
 dense_58/StatefulPartitionedCall dense_58/StatefulPartitionedCall2D
 dense_59/StatefulPartitionedCall dense_59/StatefulPartitionedCall:[ W
1
_output_shapes
:???????????
"
_user_specified_name
input_30
?
?
8__inference_batch_normalization_50_layer_call_fn_2460820

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_50_layer_call_and_return_conditional_losses_24586132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_48_layer_call_and_return_conditional_losses_2460543

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????**:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????**2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????**::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????**
 
_user_specified_nameinputs
?	
?
E__inference_dense_58_layer_call_and_return_conditional_losses_2461099

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_50_layer_call_fn_2460833

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_50_layer_call_and_return_conditional_losses_24586312
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_46_layer_call_and_return_conditional_losses_2457479

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_48_layer_call_fn_2460505

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_48_layer_call_and_return_conditional_losses_24577302
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_51_layer_call_fn_2461045

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_51_layer_call_and_return_conditional_losses_24587312
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_48_layer_call_and_return_conditional_losses_2457699

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
m
C__inference_add_27_layer_call_and_return_conditional_losses_2458557

inputs
inputs_1
identity_
addAddV2inputsinputs_1*
T0*/
_output_shapes
:?????????**2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:?????????**2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:?????????**:?????????**:W S
/
_output_shapes
:?????????**
 
_user_specified_nameinputs:WS
/
_output_shapes
:?????????**
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_47_layer_call_fn_2460325

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_47_layer_call_and_return_conditional_losses_24576142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?

?
G__inference_conv2d_138_layer_call_and_return_conditional_losses_2460400

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_46_layer_call_fn_2460228

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_46_layer_call_and_return_conditional_losses_24574792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_46_layer_call_fn_2460241

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_46_layer_call_and_return_conditional_losses_24575102
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?

?
G__inference_conv2d_137_layer_call_and_return_conditional_losses_2460252

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_51_layer_call_and_return_conditional_losses_2458054

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_46_layer_call_and_return_conditional_losses_2458145

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
i
M__inference_max_pooling2d_55_layer_call_and_return_conditional_losses_2457631

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
??
?F
#__inference__traced_restore_2461899
file_prefix&
"assignvariableop_conv2d_136_kernel&
"assignvariableop_1_conv2d_136_bias3
/assignvariableop_2_batch_normalization_46_gamma2
.assignvariableop_3_batch_normalization_46_beta9
5assignvariableop_4_batch_normalization_46_moving_mean=
9assignvariableop_5_batch_normalization_46_moving_variance(
$assignvariableop_6_conv2d_137_kernel&
"assignvariableop_7_conv2d_137_bias3
/assignvariableop_8_batch_normalization_47_gamma2
.assignvariableop_9_batch_normalization_47_beta:
6assignvariableop_10_batch_normalization_47_moving_mean>
:assignvariableop_11_batch_normalization_47_moving_variance)
%assignvariableop_12_conv2d_138_kernel'
#assignvariableop_13_conv2d_138_bias)
%assignvariableop_14_conv2d_139_kernel'
#assignvariableop_15_conv2d_139_bias4
0assignvariableop_16_batch_normalization_48_gamma3
/assignvariableop_17_batch_normalization_48_beta:
6assignvariableop_18_batch_normalization_48_moving_mean>
:assignvariableop_19_batch_normalization_48_moving_variance)
%assignvariableop_20_conv2d_140_kernel'
#assignvariableop_21_conv2d_140_bias4
0assignvariableop_22_batch_normalization_49_gamma3
/assignvariableop_23_batch_normalization_49_beta:
6assignvariableop_24_batch_normalization_49_moving_mean>
:assignvariableop_25_batch_normalization_49_moving_variance)
%assignvariableop_26_conv2d_141_kernel'
#assignvariableop_27_conv2d_141_bias)
%assignvariableop_28_conv2d_142_kernel'
#assignvariableop_29_conv2d_142_bias4
0assignvariableop_30_batch_normalization_50_gamma3
/assignvariableop_31_batch_normalization_50_beta:
6assignvariableop_32_batch_normalization_50_moving_mean>
:assignvariableop_33_batch_normalization_50_moving_variance)
%assignvariableop_34_conv2d_143_kernel'
#assignvariableop_35_conv2d_143_bias4
0assignvariableop_36_batch_normalization_51_gamma3
/assignvariableop_37_batch_normalization_51_beta:
6assignvariableop_38_batch_normalization_51_moving_mean>
:assignvariableop_39_batch_normalization_51_moving_variance)
%assignvariableop_40_conv2d_144_kernel'
#assignvariableop_41_conv2d_144_bias'
#assignvariableop_42_dense_58_kernel%
!assignvariableop_43_dense_58_bias'
#assignvariableop_44_dense_59_kernel%
!assignvariableop_45_dense_59_bias!
assignvariableop_46_adam_iter#
assignvariableop_47_adam_beta_1#
assignvariableop_48_adam_beta_2"
assignvariableop_49_adam_decay*
&assignvariableop_50_adam_learning_rate
assignvariableop_51_total
assignvariableop_52_count
assignvariableop_53_total_1
assignvariableop_54_count_10
,assignvariableop_55_adam_conv2d_136_kernel_m.
*assignvariableop_56_adam_conv2d_136_bias_m;
7assignvariableop_57_adam_batch_normalization_46_gamma_m:
6assignvariableop_58_adam_batch_normalization_46_beta_m0
,assignvariableop_59_adam_conv2d_137_kernel_m.
*assignvariableop_60_adam_conv2d_137_bias_m;
7assignvariableop_61_adam_batch_normalization_47_gamma_m:
6assignvariableop_62_adam_batch_normalization_47_beta_m0
,assignvariableop_63_adam_conv2d_138_kernel_m.
*assignvariableop_64_adam_conv2d_138_bias_m0
,assignvariableop_65_adam_conv2d_139_kernel_m.
*assignvariableop_66_adam_conv2d_139_bias_m;
7assignvariableop_67_adam_batch_normalization_48_gamma_m:
6assignvariableop_68_adam_batch_normalization_48_beta_m0
,assignvariableop_69_adam_conv2d_140_kernel_m.
*assignvariableop_70_adam_conv2d_140_bias_m;
7assignvariableop_71_adam_batch_normalization_49_gamma_m:
6assignvariableop_72_adam_batch_normalization_49_beta_m0
,assignvariableop_73_adam_conv2d_141_kernel_m.
*assignvariableop_74_adam_conv2d_141_bias_m0
,assignvariableop_75_adam_conv2d_142_kernel_m.
*assignvariableop_76_adam_conv2d_142_bias_m;
7assignvariableop_77_adam_batch_normalization_50_gamma_m:
6assignvariableop_78_adam_batch_normalization_50_beta_m0
,assignvariableop_79_adam_conv2d_143_kernel_m.
*assignvariableop_80_adam_conv2d_143_bias_m;
7assignvariableop_81_adam_batch_normalization_51_gamma_m:
6assignvariableop_82_adam_batch_normalization_51_beta_m0
,assignvariableop_83_adam_conv2d_144_kernel_m.
*assignvariableop_84_adam_conv2d_144_bias_m.
*assignvariableop_85_adam_dense_58_kernel_m,
(assignvariableop_86_adam_dense_58_bias_m.
*assignvariableop_87_adam_dense_59_kernel_m,
(assignvariableop_88_adam_dense_59_bias_m0
,assignvariableop_89_adam_conv2d_136_kernel_v.
*assignvariableop_90_adam_conv2d_136_bias_v;
7assignvariableop_91_adam_batch_normalization_46_gamma_v:
6assignvariableop_92_adam_batch_normalization_46_beta_v0
,assignvariableop_93_adam_conv2d_137_kernel_v.
*assignvariableop_94_adam_conv2d_137_bias_v;
7assignvariableop_95_adam_batch_normalization_47_gamma_v:
6assignvariableop_96_adam_batch_normalization_47_beta_v0
,assignvariableop_97_adam_conv2d_138_kernel_v.
*assignvariableop_98_adam_conv2d_138_bias_v0
,assignvariableop_99_adam_conv2d_139_kernel_v/
+assignvariableop_100_adam_conv2d_139_bias_v<
8assignvariableop_101_adam_batch_normalization_48_gamma_v;
7assignvariableop_102_adam_batch_normalization_48_beta_v1
-assignvariableop_103_adam_conv2d_140_kernel_v/
+assignvariableop_104_adam_conv2d_140_bias_v<
8assignvariableop_105_adam_batch_normalization_49_gamma_v;
7assignvariableop_106_adam_batch_normalization_49_beta_v1
-assignvariableop_107_adam_conv2d_141_kernel_v/
+assignvariableop_108_adam_conv2d_141_bias_v1
-assignvariableop_109_adam_conv2d_142_kernel_v/
+assignvariableop_110_adam_conv2d_142_bias_v<
8assignvariableop_111_adam_batch_normalization_50_gamma_v;
7assignvariableop_112_adam_batch_normalization_50_beta_v1
-assignvariableop_113_adam_conv2d_143_kernel_v/
+assignvariableop_114_adam_conv2d_143_bias_v<
8assignvariableop_115_adam_batch_normalization_51_gamma_v;
7assignvariableop_116_adam_batch_normalization_51_beta_v1
-assignvariableop_117_adam_conv2d_144_kernel_v/
+assignvariableop_118_adam_conv2d_144_bias_v/
+assignvariableop_119_adam_dense_58_kernel_v-
)assignvariableop_120_adam_dense_58_bias_v/
+assignvariableop_121_adam_dense_59_kernel_v-
)assignvariableop_122_adam_dense_59_bias_v
identity_124??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_100?AssignVariableOp_101?AssignVariableOp_102?AssignVariableOp_103?AssignVariableOp_104?AssignVariableOp_105?AssignVariableOp_106?AssignVariableOp_107?AssignVariableOp_108?AssignVariableOp_109?AssignVariableOp_11?AssignVariableOp_110?AssignVariableOp_111?AssignVariableOp_112?AssignVariableOp_113?AssignVariableOp_114?AssignVariableOp_115?AssignVariableOp_116?AssignVariableOp_117?AssignVariableOp_118?AssignVariableOp_119?AssignVariableOp_12?AssignVariableOp_120?AssignVariableOp_121?AssignVariableOp_122?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_83?AssignVariableOp_84?AssignVariableOp_85?AssignVariableOp_86?AssignVariableOp_87?AssignVariableOp_88?AssignVariableOp_89?AssignVariableOp_9?AssignVariableOp_90?AssignVariableOp_91?AssignVariableOp_92?AssignVariableOp_93?AssignVariableOp_94?AssignVariableOp_95?AssignVariableOp_96?AssignVariableOp_97?AssignVariableOp_98?AssignVariableOp_99?E
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:|*
dtype0*?D
value?DB?D|B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:|*
dtype0*?
value?B?|B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*?
dtypes?
~2|	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp"assignvariableop_conv2d_136_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp"assignvariableop_1_conv2d_136_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp/assignvariableop_2_batch_normalization_46_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp.assignvariableop_3_batch_normalization_46_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp5assignvariableop_4_batch_normalization_46_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp9assignvariableop_5_batch_normalization_46_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp$assignvariableop_6_conv2d_137_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp"assignvariableop_7_conv2d_137_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp/assignvariableop_8_batch_normalization_47_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_47_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp6assignvariableop_10_batch_normalization_47_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_normalization_47_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp%assignvariableop_12_conv2d_138_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp#assignvariableop_13_conv2d_138_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp%assignvariableop_14_conv2d_139_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp#assignvariableop_15_conv2d_139_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp0assignvariableop_16_batch_normalization_48_gammaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp/assignvariableop_17_batch_normalization_48_betaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp6assignvariableop_18_batch_normalization_48_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp:assignvariableop_19_batch_normalization_48_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp%assignvariableop_20_conv2d_140_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp#assignvariableop_21_conv2d_140_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp0assignvariableop_22_batch_normalization_49_gammaIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp/assignvariableop_23_batch_normalization_49_betaIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp6assignvariableop_24_batch_normalization_49_moving_meanIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp:assignvariableop_25_batch_normalization_49_moving_varianceIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp%assignvariableop_26_conv2d_141_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp#assignvariableop_27_conv2d_141_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp%assignvariableop_28_conv2d_142_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp#assignvariableop_29_conv2d_142_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp0assignvariableop_30_batch_normalization_50_gammaIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp/assignvariableop_31_batch_normalization_50_betaIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp6assignvariableop_32_batch_normalization_50_moving_meanIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp:assignvariableop_33_batch_normalization_50_moving_varianceIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp%assignvariableop_34_conv2d_143_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp#assignvariableop_35_conv2d_143_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp0assignvariableop_36_batch_normalization_51_gammaIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp/assignvariableop_37_batch_normalization_51_betaIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp6assignvariableop_38_batch_normalization_51_moving_meanIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp:assignvariableop_39_batch_normalization_51_moving_varianceIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp%assignvariableop_40_conv2d_144_kernelIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp#assignvariableop_41_conv2d_144_biasIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp#assignvariableop_42_dense_58_kernelIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp!assignvariableop_43_dense_58_biasIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp#assignvariableop_44_dense_59_kernelIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp!assignvariableop_45_dense_59_biasIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOpassignvariableop_46_adam_iterIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOpassignvariableop_47_adam_beta_1Identity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOpassignvariableop_48_adam_beta_2Identity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOpassignvariableop_49_adam_decayIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp&assignvariableop_50_adam_learning_rateIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOpassignvariableop_51_totalIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOpassignvariableop_52_countIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOpassignvariableop_53_total_1Identity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOpassignvariableop_54_count_1Identity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp,assignvariableop_55_adam_conv2d_136_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp*assignvariableop_56_adam_conv2d_136_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp7assignvariableop_57_adam_batch_normalization_46_gamma_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp6assignvariableop_58_adam_batch_normalization_46_beta_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp,assignvariableop_59_adam_conv2d_137_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp*assignvariableop_60_adam_conv2d_137_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp7assignvariableop_61_adam_batch_normalization_47_gamma_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp6assignvariableop_62_adam_batch_normalization_47_beta_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp,assignvariableop_63_adam_conv2d_138_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp*assignvariableop_64_adam_conv2d_138_bias_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp,assignvariableop_65_adam_conv2d_139_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp*assignvariableop_66_adam_conv2d_139_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOp7assignvariableop_67_adam_batch_normalization_48_gamma_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp6assignvariableop_68_adam_batch_normalization_48_beta_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOp,assignvariableop_69_adam_conv2d_140_kernel_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp*assignvariableop_70_adam_conv2d_140_bias_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOp7assignvariableop_71_adam_batch_normalization_49_gamma_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOp6assignvariableop_72_adam_batch_normalization_49_beta_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73?
AssignVariableOp_73AssignVariableOp,assignvariableop_73_adam_conv2d_141_kernel_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOp*assignvariableop_74_adam_conv2d_141_bias_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75?
AssignVariableOp_75AssignVariableOp,assignvariableop_75_adam_conv2d_142_kernel_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76?
AssignVariableOp_76AssignVariableOp*assignvariableop_76_adam_conv2d_142_bias_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77?
AssignVariableOp_77AssignVariableOp7assignvariableop_77_adam_batch_normalization_50_gamma_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78?
AssignVariableOp_78AssignVariableOp6assignvariableop_78_adam_batch_normalization_50_beta_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79?
AssignVariableOp_79AssignVariableOp,assignvariableop_79_adam_conv2d_143_kernel_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80?
AssignVariableOp_80AssignVariableOp*assignvariableop_80_adam_conv2d_143_bias_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81?
AssignVariableOp_81AssignVariableOp7assignvariableop_81_adam_batch_normalization_51_gamma_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82?
AssignVariableOp_82AssignVariableOp6assignvariableop_82_adam_batch_normalization_51_beta_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83?
AssignVariableOp_83AssignVariableOp,assignvariableop_83_adam_conv2d_144_kernel_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84?
AssignVariableOp_84AssignVariableOp*assignvariableop_84_adam_conv2d_144_bias_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85?
AssignVariableOp_85AssignVariableOp*assignvariableop_85_adam_dense_58_kernel_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86?
AssignVariableOp_86AssignVariableOp(assignvariableop_86_adam_dense_58_bias_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87?
AssignVariableOp_87AssignVariableOp*assignvariableop_87_adam_dense_59_kernel_mIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88?
AssignVariableOp_88AssignVariableOp(assignvariableop_88_adam_dense_59_bias_mIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89?
AssignVariableOp_89AssignVariableOp,assignvariableop_89_adam_conv2d_136_kernel_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90?
AssignVariableOp_90AssignVariableOp*assignvariableop_90_adam_conv2d_136_bias_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91?
AssignVariableOp_91AssignVariableOp7assignvariableop_91_adam_batch_normalization_46_gamma_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92?
AssignVariableOp_92AssignVariableOp6assignvariableop_92_adam_batch_normalization_46_beta_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93?
AssignVariableOp_93AssignVariableOp,assignvariableop_93_adam_conv2d_137_kernel_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94?
AssignVariableOp_94AssignVariableOp*assignvariableop_94_adam_conv2d_137_bias_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95?
AssignVariableOp_95AssignVariableOp7assignvariableop_95_adam_batch_normalization_47_gamma_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96?
AssignVariableOp_96AssignVariableOp6assignvariableop_96_adam_batch_normalization_47_beta_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_96n
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:2
Identity_97?
AssignVariableOp_97AssignVariableOp,assignvariableop_97_adam_conv2d_138_kernel_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_97n
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:2
Identity_98?
AssignVariableOp_98AssignVariableOp*assignvariableop_98_adam_conv2d_138_bias_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_98n
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:2
Identity_99?
AssignVariableOp_99AssignVariableOp,assignvariableop_99_adam_conv2d_139_kernel_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99q
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:2
Identity_100?
AssignVariableOp_100AssignVariableOp+assignvariableop_100_adam_conv2d_139_bias_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_100q
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:2
Identity_101?
AssignVariableOp_101AssignVariableOp8assignvariableop_101_adam_batch_normalization_48_gamma_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_101q
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:2
Identity_102?
AssignVariableOp_102AssignVariableOp7assignvariableop_102_adam_batch_normalization_48_beta_vIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_102q
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:2
Identity_103?
AssignVariableOp_103AssignVariableOp-assignvariableop_103_adam_conv2d_140_kernel_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_103q
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:2
Identity_104?
AssignVariableOp_104AssignVariableOp+assignvariableop_104_adam_conv2d_140_bias_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_104q
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:2
Identity_105?
AssignVariableOp_105AssignVariableOp8assignvariableop_105_adam_batch_normalization_49_gamma_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_105q
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:2
Identity_106?
AssignVariableOp_106AssignVariableOp7assignvariableop_106_adam_batch_normalization_49_beta_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_106q
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:2
Identity_107?
AssignVariableOp_107AssignVariableOp-assignvariableop_107_adam_conv2d_141_kernel_vIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_107q
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:2
Identity_108?
AssignVariableOp_108AssignVariableOp+assignvariableop_108_adam_conv2d_141_bias_vIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_108q
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:2
Identity_109?
AssignVariableOp_109AssignVariableOp-assignvariableop_109_adam_conv2d_142_kernel_vIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_109q
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:2
Identity_110?
AssignVariableOp_110AssignVariableOp+assignvariableop_110_adam_conv2d_142_bias_vIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_110q
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:2
Identity_111?
AssignVariableOp_111AssignVariableOp8assignvariableop_111_adam_batch_normalization_50_gamma_vIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_111q
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:2
Identity_112?
AssignVariableOp_112AssignVariableOp7assignvariableop_112_adam_batch_normalization_50_beta_vIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_112q
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:2
Identity_113?
AssignVariableOp_113AssignVariableOp-assignvariableop_113_adam_conv2d_143_kernel_vIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_113q
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:2
Identity_114?
AssignVariableOp_114AssignVariableOp+assignvariableop_114_adam_conv2d_143_bias_vIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_114q
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:2
Identity_115?
AssignVariableOp_115AssignVariableOp8assignvariableop_115_adam_batch_normalization_51_gamma_vIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_115q
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:2
Identity_116?
AssignVariableOp_116AssignVariableOp7assignvariableop_116_adam_batch_normalization_51_beta_vIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_116q
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:2
Identity_117?
AssignVariableOp_117AssignVariableOp-assignvariableop_117_adam_conv2d_144_kernel_vIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_117q
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:2
Identity_118?
AssignVariableOp_118AssignVariableOp+assignvariableop_118_adam_conv2d_144_bias_vIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_118q
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:2
Identity_119?
AssignVariableOp_119AssignVariableOp+assignvariableop_119_adam_dense_58_kernel_vIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_119q
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:2
Identity_120?
AssignVariableOp_120AssignVariableOp)assignvariableop_120_adam_dense_58_bias_vIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_120q
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:2
Identity_121?
AssignVariableOp_121AssignVariableOp+assignvariableop_121_adam_dense_59_kernel_vIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_121q
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:2
Identity_122?
AssignVariableOp_122AssignVariableOp)assignvariableop_122_adam_dense_59_bias_vIdentity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1229
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_123Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_123?
Identity_124IdentityIdentity_123:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*
T0*
_output_shapes
: 2
Identity_124"%
identity_124Identity_124:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142,
AssignVariableOp_115AssignVariableOp_1152,
AssignVariableOp_116AssignVariableOp_1162,
AssignVariableOp_117AssignVariableOp_1172,
AssignVariableOp_118AssignVariableOp_1182,
AssignVariableOp_119AssignVariableOp_1192*
AssignVariableOp_12AssignVariableOp_122,
AssignVariableOp_120AssignVariableOp_1202,
AssignVariableOp_121AssignVariableOp_1212,
AssignVariableOp_122AssignVariableOp_1222*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
,__inference_conv2d_144_layer_call_fn_2461065

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_144_layer_call_and_return_conditional_losses_24587782
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_conv2d_137_layer_call_fn_2460261

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_137_layer_call_and_return_conditional_losses_24581922
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_47_layer_call_and_return_conditional_losses_2460299

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?

*__inference_dense_59_layer_call_fn_2461128

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_59_layer_call_and_return_conditional_losses_24588622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_49_layer_call_and_return_conditional_losses_2460627

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????**:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????**2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????**::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????**
 
_user_specified_nameinputs
?
N
2__inference_max_pooling2d_55_layer_call_fn_2457637

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_max_pooling2d_55_layer_call_and_return_conditional_losses_24576312
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_50_layer_call_and_return_conditional_losses_2457919

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_51_layer_call_fn_2460981

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_51_layer_call_and_return_conditional_losses_24580542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
m
C__inference_add_28_layer_call_and_return_conditional_losses_2458800

inputs
inputs_1
identity_
addAddV2inputsinputs_1*
T0*/
_output_shapes
:?????????2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:?????????:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs:WS
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
i
M__inference_max_pooling2d_57_layer_call_and_return_conditional_losses_2458071

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_47_layer_call_and_return_conditional_losses_2457583

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_51_layer_call_and_return_conditional_losses_2461019

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
c
G__inference_flatten_29_layer_call_and_return_conditional_losses_2461083

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
T__inference_CNN_aug_deep_bnorm_skip_layer_call_and_return_conditional_losses_2459339

inputs
conv2d_136_2459222
conv2d_136_2459224"
batch_normalization_46_2459227"
batch_normalization_46_2459229"
batch_normalization_46_2459231"
batch_normalization_46_2459233
conv2d_137_2459236
conv2d_137_2459238"
batch_normalization_47_2459241"
batch_normalization_47_2459243"
batch_normalization_47_2459245"
batch_normalization_47_2459247
conv2d_138_2459250
conv2d_138_2459252
conv2d_139_2459257
conv2d_139_2459259"
batch_normalization_48_2459262"
batch_normalization_48_2459264"
batch_normalization_48_2459266"
batch_normalization_48_2459268
conv2d_140_2459271
conv2d_140_2459273"
batch_normalization_49_2459276"
batch_normalization_49_2459278"
batch_normalization_49_2459280"
batch_normalization_49_2459282
conv2d_141_2459285
conv2d_141_2459287
conv2d_142_2459292
conv2d_142_2459294"
batch_normalization_50_2459297"
batch_normalization_50_2459299"
batch_normalization_50_2459301"
batch_normalization_50_2459303
conv2d_143_2459306
conv2d_143_2459308"
batch_normalization_51_2459311"
batch_normalization_51_2459313"
batch_normalization_51_2459315"
batch_normalization_51_2459317
conv2d_144_2459320
conv2d_144_2459322
dense_58_2459328
dense_58_2459330
dense_59_2459333
dense_59_2459335
identity??.batch_normalization_46/StatefulPartitionedCall?.batch_normalization_47/StatefulPartitionedCall?.batch_normalization_48/StatefulPartitionedCall?.batch_normalization_49/StatefulPartitionedCall?.batch_normalization_50/StatefulPartitionedCall?.batch_normalization_51/StatefulPartitionedCall?"conv2d_136/StatefulPartitionedCall?"conv2d_137/StatefulPartitionedCall?"conv2d_138/StatefulPartitionedCall?"conv2d_139/StatefulPartitionedCall?"conv2d_140/StatefulPartitionedCall?"conv2d_141/StatefulPartitionedCall?"conv2d_142/StatefulPartitionedCall?"conv2d_143/StatefulPartitionedCall?"conv2d_144/StatefulPartitionedCall? dense_58/StatefulPartitionedCall? dense_59/StatefulPartitionedCall?
"conv2d_136/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_136_2459222conv2d_136_2459224*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_136_layer_call_and_return_conditional_losses_24580922$
"conv2d_136/StatefulPartitionedCall?
.batch_normalization_46/StatefulPartitionedCallStatefulPartitionedCall+conv2d_136/StatefulPartitionedCall:output:0batch_normalization_46_2459227batch_normalization_46_2459229batch_normalization_46_2459231batch_normalization_46_2459233*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_46_layer_call_and_return_conditional_losses_245814520
.batch_normalization_46/StatefulPartitionedCall?
"conv2d_137/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_46/StatefulPartitionedCall:output:0conv2d_137_2459236conv2d_137_2459238*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_137_layer_call_and_return_conditional_losses_24581922$
"conv2d_137/StatefulPartitionedCall?
.batch_normalization_47/StatefulPartitionedCallStatefulPartitionedCall+conv2d_137/StatefulPartitionedCall:output:0batch_normalization_47_2459241batch_normalization_47_2459243batch_normalization_47_2459245batch_normalization_47_2459247*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_47_layer_call_and_return_conditional_losses_245824520
.batch_normalization_47/StatefulPartitionedCall?
"conv2d_138/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_47/StatefulPartitionedCall:output:0conv2d_138_2459250conv2d_138_2459252*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_138_layer_call_and_return_conditional_losses_24582922$
"conv2d_138/StatefulPartitionedCall?
add_26/PartitionedCallPartitionedCall+conv2d_138/StatefulPartitionedCall:output:0inputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_add_26_layer_call_and_return_conditional_losses_24583142
add_26/PartitionedCall?
 max_pooling2d_55/PartitionedCallPartitionedCalladd_26/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*** 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_max_pooling2d_55_layer_call_and_return_conditional_losses_24576312"
 max_pooling2d_55/PartitionedCall?
"conv2d_139/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_55/PartitionedCall:output:0conv2d_139_2459257conv2d_139_2459259*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????***$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_139_layer_call_and_return_conditional_losses_24583352$
"conv2d_139/StatefulPartitionedCall?
.batch_normalization_48/StatefulPartitionedCallStatefulPartitionedCall+conv2d_139/StatefulPartitionedCall:output:0batch_normalization_48_2459262batch_normalization_48_2459264batch_normalization_48_2459266batch_normalization_48_2459268*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????***&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_48_layer_call_and_return_conditional_losses_245838820
.batch_normalization_48/StatefulPartitionedCall?
"conv2d_140/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_48/StatefulPartitionedCall:output:0conv2d_140_2459271conv2d_140_2459273*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????***$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_140_layer_call_and_return_conditional_losses_24584352$
"conv2d_140/StatefulPartitionedCall?
.batch_normalization_49/StatefulPartitionedCallStatefulPartitionedCall+conv2d_140/StatefulPartitionedCall:output:0batch_normalization_49_2459276batch_normalization_49_2459278batch_normalization_49_2459280batch_normalization_49_2459282*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????***&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_49_layer_call_and_return_conditional_losses_245848820
.batch_normalization_49/StatefulPartitionedCall?
"conv2d_141/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_49/StatefulPartitionedCall:output:0conv2d_141_2459285conv2d_141_2459287*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????***$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_141_layer_call_and_return_conditional_losses_24585352$
"conv2d_141/StatefulPartitionedCall?
add_27/PartitionedCallPartitionedCall+conv2d_141/StatefulPartitionedCall:output:0)max_pooling2d_55/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*** 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_add_27_layer_call_and_return_conditional_losses_24585572
add_27/PartitionedCall?
 max_pooling2d_56/PartitionedCallPartitionedCalladd_27/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_max_pooling2d_56_layer_call_and_return_conditional_losses_24578512"
 max_pooling2d_56/PartitionedCall?
"conv2d_142/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_56/PartitionedCall:output:0conv2d_142_2459292conv2d_142_2459294*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_142_layer_call_and_return_conditional_losses_24585782$
"conv2d_142/StatefulPartitionedCall?
.batch_normalization_50/StatefulPartitionedCallStatefulPartitionedCall+conv2d_142/StatefulPartitionedCall:output:0batch_normalization_50_2459297batch_normalization_50_2459299batch_normalization_50_2459301batch_normalization_50_2459303*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_50_layer_call_and_return_conditional_losses_245863120
.batch_normalization_50/StatefulPartitionedCall?
"conv2d_143/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_50/StatefulPartitionedCall:output:0conv2d_143_2459306conv2d_143_2459308*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_143_layer_call_and_return_conditional_losses_24586782$
"conv2d_143/StatefulPartitionedCall?
.batch_normalization_51/StatefulPartitionedCallStatefulPartitionedCall+conv2d_143/StatefulPartitionedCall:output:0batch_normalization_51_2459311batch_normalization_51_2459313batch_normalization_51_2459315batch_normalization_51_2459317*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_51_layer_call_and_return_conditional_losses_245873120
.batch_normalization_51/StatefulPartitionedCall?
"conv2d_144/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_51/StatefulPartitionedCall:output:0conv2d_144_2459320conv2d_144_2459322*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_144_layer_call_and_return_conditional_losses_24587782$
"conv2d_144/StatefulPartitionedCall?
add_28/PartitionedCallPartitionedCall+conv2d_144/StatefulPartitionedCall:output:0)max_pooling2d_56/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_add_28_layer_call_and_return_conditional_losses_24588002
add_28/PartitionedCall?
 max_pooling2d_57/PartitionedCallPartitionedCalladd_28/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_max_pooling2d_57_layer_call_and_return_conditional_losses_24580712"
 max_pooling2d_57/PartitionedCall?
flatten_29/PartitionedCallPartitionedCall)max_pooling2d_57/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_flatten_29_layer_call_and_return_conditional_losses_24588162
flatten_29/PartitionedCall?
 dense_58/StatefulPartitionedCallStatefulPartitionedCall#flatten_29/PartitionedCall:output:0dense_58_2459328dense_58_2459330*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_58_layer_call_and_return_conditional_losses_24588352"
 dense_58/StatefulPartitionedCall?
 dense_59/StatefulPartitionedCallStatefulPartitionedCall)dense_58/StatefulPartitionedCall:output:0dense_59_2459333dense_59_2459335*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_59_layer_call_and_return_conditional_losses_24588622"
 dense_59/StatefulPartitionedCall?
IdentityIdentity)dense_59/StatefulPartitionedCall:output:0/^batch_normalization_46/StatefulPartitionedCall/^batch_normalization_47/StatefulPartitionedCall/^batch_normalization_48/StatefulPartitionedCall/^batch_normalization_49/StatefulPartitionedCall/^batch_normalization_50/StatefulPartitionedCall/^batch_normalization_51/StatefulPartitionedCall#^conv2d_136/StatefulPartitionedCall#^conv2d_137/StatefulPartitionedCall#^conv2d_138/StatefulPartitionedCall#^conv2d_139/StatefulPartitionedCall#^conv2d_140/StatefulPartitionedCall#^conv2d_141/StatefulPartitionedCall#^conv2d_142/StatefulPartitionedCall#^conv2d_143/StatefulPartitionedCall#^conv2d_144/StatefulPartitionedCall!^dense_58/StatefulPartitionedCall!^dense_59/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_46/StatefulPartitionedCall.batch_normalization_46/StatefulPartitionedCall2`
.batch_normalization_47/StatefulPartitionedCall.batch_normalization_47/StatefulPartitionedCall2`
.batch_normalization_48/StatefulPartitionedCall.batch_normalization_48/StatefulPartitionedCall2`
.batch_normalization_49/StatefulPartitionedCall.batch_normalization_49/StatefulPartitionedCall2`
.batch_normalization_50/StatefulPartitionedCall.batch_normalization_50/StatefulPartitionedCall2`
.batch_normalization_51/StatefulPartitionedCall.batch_normalization_51/StatefulPartitionedCall2H
"conv2d_136/StatefulPartitionedCall"conv2d_136/StatefulPartitionedCall2H
"conv2d_137/StatefulPartitionedCall"conv2d_137/StatefulPartitionedCall2H
"conv2d_138/StatefulPartitionedCall"conv2d_138/StatefulPartitionedCall2H
"conv2d_139/StatefulPartitionedCall"conv2d_139/StatefulPartitionedCall2H
"conv2d_140/StatefulPartitionedCall"conv2d_140/StatefulPartitionedCall2H
"conv2d_141/StatefulPartitionedCall"conv2d_141/StatefulPartitionedCall2H
"conv2d_142/StatefulPartitionedCall"conv2d_142/StatefulPartitionedCall2H
"conv2d_143/StatefulPartitionedCall"conv2d_143/StatefulPartitionedCall2H
"conv2d_144/StatefulPartitionedCall"conv2d_144/StatefulPartitionedCall2D
 dense_58/StatefulPartitionedCall dense_58/StatefulPartitionedCall2D
 dense_59/StatefulPartitionedCall dense_59/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
o
C__inference_add_26_layer_call_and_return_conditional_losses_2460415
inputs_0
inputs_1
identityc
addAddV2inputs_0inputs_1*
T0*1
_output_shapes
:???????????2
adde
IdentityIdentityadd:z:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*M
_input_shapes<
::???????????:???????????:[ W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/1
?
?
S__inference_batch_normalization_51_layer_call_and_return_conditional_losses_2460955

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
T__inference_CNN_aug_deep_bnorm_skip_layer_call_and_return_conditional_losses_2458999
input_30
conv2d_136_2458882
conv2d_136_2458884"
batch_normalization_46_2458887"
batch_normalization_46_2458889"
batch_normalization_46_2458891"
batch_normalization_46_2458893
conv2d_137_2458896
conv2d_137_2458898"
batch_normalization_47_2458901"
batch_normalization_47_2458903"
batch_normalization_47_2458905"
batch_normalization_47_2458907
conv2d_138_2458910
conv2d_138_2458912
conv2d_139_2458917
conv2d_139_2458919"
batch_normalization_48_2458922"
batch_normalization_48_2458924"
batch_normalization_48_2458926"
batch_normalization_48_2458928
conv2d_140_2458931
conv2d_140_2458933"
batch_normalization_49_2458936"
batch_normalization_49_2458938"
batch_normalization_49_2458940"
batch_normalization_49_2458942
conv2d_141_2458945
conv2d_141_2458947
conv2d_142_2458952
conv2d_142_2458954"
batch_normalization_50_2458957"
batch_normalization_50_2458959"
batch_normalization_50_2458961"
batch_normalization_50_2458963
conv2d_143_2458966
conv2d_143_2458968"
batch_normalization_51_2458971"
batch_normalization_51_2458973"
batch_normalization_51_2458975"
batch_normalization_51_2458977
conv2d_144_2458980
conv2d_144_2458982
dense_58_2458988
dense_58_2458990
dense_59_2458993
dense_59_2458995
identity??.batch_normalization_46/StatefulPartitionedCall?.batch_normalization_47/StatefulPartitionedCall?.batch_normalization_48/StatefulPartitionedCall?.batch_normalization_49/StatefulPartitionedCall?.batch_normalization_50/StatefulPartitionedCall?.batch_normalization_51/StatefulPartitionedCall?"conv2d_136/StatefulPartitionedCall?"conv2d_137/StatefulPartitionedCall?"conv2d_138/StatefulPartitionedCall?"conv2d_139/StatefulPartitionedCall?"conv2d_140/StatefulPartitionedCall?"conv2d_141/StatefulPartitionedCall?"conv2d_142/StatefulPartitionedCall?"conv2d_143/StatefulPartitionedCall?"conv2d_144/StatefulPartitionedCall? dense_58/StatefulPartitionedCall? dense_59/StatefulPartitionedCall?
"conv2d_136/StatefulPartitionedCallStatefulPartitionedCallinput_30conv2d_136_2458882conv2d_136_2458884*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_136_layer_call_and_return_conditional_losses_24580922$
"conv2d_136/StatefulPartitionedCall?
.batch_normalization_46/StatefulPartitionedCallStatefulPartitionedCall+conv2d_136/StatefulPartitionedCall:output:0batch_normalization_46_2458887batch_normalization_46_2458889batch_normalization_46_2458891batch_normalization_46_2458893*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_46_layer_call_and_return_conditional_losses_245814520
.batch_normalization_46/StatefulPartitionedCall?
"conv2d_137/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_46/StatefulPartitionedCall:output:0conv2d_137_2458896conv2d_137_2458898*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_137_layer_call_and_return_conditional_losses_24581922$
"conv2d_137/StatefulPartitionedCall?
.batch_normalization_47/StatefulPartitionedCallStatefulPartitionedCall+conv2d_137/StatefulPartitionedCall:output:0batch_normalization_47_2458901batch_normalization_47_2458903batch_normalization_47_2458905batch_normalization_47_2458907*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_47_layer_call_and_return_conditional_losses_245824520
.batch_normalization_47/StatefulPartitionedCall?
"conv2d_138/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_47/StatefulPartitionedCall:output:0conv2d_138_2458910conv2d_138_2458912*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_138_layer_call_and_return_conditional_losses_24582922$
"conv2d_138/StatefulPartitionedCall?
add_26/PartitionedCallPartitionedCall+conv2d_138/StatefulPartitionedCall:output:0input_30*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_add_26_layer_call_and_return_conditional_losses_24583142
add_26/PartitionedCall?
 max_pooling2d_55/PartitionedCallPartitionedCalladd_26/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*** 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_max_pooling2d_55_layer_call_and_return_conditional_losses_24576312"
 max_pooling2d_55/PartitionedCall?
"conv2d_139/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_55/PartitionedCall:output:0conv2d_139_2458917conv2d_139_2458919*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????***$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_139_layer_call_and_return_conditional_losses_24583352$
"conv2d_139/StatefulPartitionedCall?
.batch_normalization_48/StatefulPartitionedCallStatefulPartitionedCall+conv2d_139/StatefulPartitionedCall:output:0batch_normalization_48_2458922batch_normalization_48_2458924batch_normalization_48_2458926batch_normalization_48_2458928*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????***&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_48_layer_call_and_return_conditional_losses_245838820
.batch_normalization_48/StatefulPartitionedCall?
"conv2d_140/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_48/StatefulPartitionedCall:output:0conv2d_140_2458931conv2d_140_2458933*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????***$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_140_layer_call_and_return_conditional_losses_24584352$
"conv2d_140/StatefulPartitionedCall?
.batch_normalization_49/StatefulPartitionedCallStatefulPartitionedCall+conv2d_140/StatefulPartitionedCall:output:0batch_normalization_49_2458936batch_normalization_49_2458938batch_normalization_49_2458940batch_normalization_49_2458942*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????***&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_49_layer_call_and_return_conditional_losses_245848820
.batch_normalization_49/StatefulPartitionedCall?
"conv2d_141/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_49/StatefulPartitionedCall:output:0conv2d_141_2458945conv2d_141_2458947*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????***$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_141_layer_call_and_return_conditional_losses_24585352$
"conv2d_141/StatefulPartitionedCall?
add_27/PartitionedCallPartitionedCall+conv2d_141/StatefulPartitionedCall:output:0)max_pooling2d_55/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*** 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_add_27_layer_call_and_return_conditional_losses_24585572
add_27/PartitionedCall?
 max_pooling2d_56/PartitionedCallPartitionedCalladd_27/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_max_pooling2d_56_layer_call_and_return_conditional_losses_24578512"
 max_pooling2d_56/PartitionedCall?
"conv2d_142/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_56/PartitionedCall:output:0conv2d_142_2458952conv2d_142_2458954*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_142_layer_call_and_return_conditional_losses_24585782$
"conv2d_142/StatefulPartitionedCall?
.batch_normalization_50/StatefulPartitionedCallStatefulPartitionedCall+conv2d_142/StatefulPartitionedCall:output:0batch_normalization_50_2458957batch_normalization_50_2458959batch_normalization_50_2458961batch_normalization_50_2458963*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_50_layer_call_and_return_conditional_losses_245863120
.batch_normalization_50/StatefulPartitionedCall?
"conv2d_143/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_50/StatefulPartitionedCall:output:0conv2d_143_2458966conv2d_143_2458968*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_143_layer_call_and_return_conditional_losses_24586782$
"conv2d_143/StatefulPartitionedCall?
.batch_normalization_51/StatefulPartitionedCallStatefulPartitionedCall+conv2d_143/StatefulPartitionedCall:output:0batch_normalization_51_2458971batch_normalization_51_2458973batch_normalization_51_2458975batch_normalization_51_2458977*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_51_layer_call_and_return_conditional_losses_245873120
.batch_normalization_51/StatefulPartitionedCall?
"conv2d_144/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_51/StatefulPartitionedCall:output:0conv2d_144_2458980conv2d_144_2458982*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_144_layer_call_and_return_conditional_losses_24587782$
"conv2d_144/StatefulPartitionedCall?
add_28/PartitionedCallPartitionedCall+conv2d_144/StatefulPartitionedCall:output:0)max_pooling2d_56/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_add_28_layer_call_and_return_conditional_losses_24588002
add_28/PartitionedCall?
 max_pooling2d_57/PartitionedCallPartitionedCalladd_28/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_max_pooling2d_57_layer_call_and_return_conditional_losses_24580712"
 max_pooling2d_57/PartitionedCall?
flatten_29/PartitionedCallPartitionedCall)max_pooling2d_57/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_flatten_29_layer_call_and_return_conditional_losses_24588162
flatten_29/PartitionedCall?
 dense_58/StatefulPartitionedCallStatefulPartitionedCall#flatten_29/PartitionedCall:output:0dense_58_2458988dense_58_2458990*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_58_layer_call_and_return_conditional_losses_24588352"
 dense_58/StatefulPartitionedCall?
 dense_59/StatefulPartitionedCallStatefulPartitionedCall)dense_58/StatefulPartitionedCall:output:0dense_59_2458993dense_59_2458995*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_59_layer_call_and_return_conditional_losses_24588622"
 dense_59/StatefulPartitionedCall?
IdentityIdentity)dense_59/StatefulPartitionedCall:output:0/^batch_normalization_46/StatefulPartitionedCall/^batch_normalization_47/StatefulPartitionedCall/^batch_normalization_48/StatefulPartitionedCall/^batch_normalization_49/StatefulPartitionedCall/^batch_normalization_50/StatefulPartitionedCall/^batch_normalization_51/StatefulPartitionedCall#^conv2d_136/StatefulPartitionedCall#^conv2d_137/StatefulPartitionedCall#^conv2d_138/StatefulPartitionedCall#^conv2d_139/StatefulPartitionedCall#^conv2d_140/StatefulPartitionedCall#^conv2d_141/StatefulPartitionedCall#^conv2d_142/StatefulPartitionedCall#^conv2d_143/StatefulPartitionedCall#^conv2d_144/StatefulPartitionedCall!^dense_58/StatefulPartitionedCall!^dense_59/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_46/StatefulPartitionedCall.batch_normalization_46/StatefulPartitionedCall2`
.batch_normalization_47/StatefulPartitionedCall.batch_normalization_47/StatefulPartitionedCall2`
.batch_normalization_48/StatefulPartitionedCall.batch_normalization_48/StatefulPartitionedCall2`
.batch_normalization_49/StatefulPartitionedCall.batch_normalization_49/StatefulPartitionedCall2`
.batch_normalization_50/StatefulPartitionedCall.batch_normalization_50/StatefulPartitionedCall2`
.batch_normalization_51/StatefulPartitionedCall.batch_normalization_51/StatefulPartitionedCall2H
"conv2d_136/StatefulPartitionedCall"conv2d_136/StatefulPartitionedCall2H
"conv2d_137/StatefulPartitionedCall"conv2d_137/StatefulPartitionedCall2H
"conv2d_138/StatefulPartitionedCall"conv2d_138/StatefulPartitionedCall2H
"conv2d_139/StatefulPartitionedCall"conv2d_139/StatefulPartitionedCall2H
"conv2d_140/StatefulPartitionedCall"conv2d_140/StatefulPartitionedCall2H
"conv2d_141/StatefulPartitionedCall"conv2d_141/StatefulPartitionedCall2H
"conv2d_142/StatefulPartitionedCall"conv2d_142/StatefulPartitionedCall2H
"conv2d_143/StatefulPartitionedCall"conv2d_143/StatefulPartitionedCall2H
"conv2d_144/StatefulPartitionedCall"conv2d_144/StatefulPartitionedCall2D
 dense_58/StatefulPartitionedCall dense_58/StatefulPartitionedCall2D
 dense_59/StatefulPartitionedCall dense_59/StatefulPartitionedCall:[ W
1
_output_shapes
:???????????
"
_user_specified_name
input_30
?
?
S__inference_batch_normalization_51_layer_call_and_return_conditional_losses_2458731

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_49_layer_call_and_return_conditional_losses_2458488

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????**:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????**2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????**::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????**
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_48_layer_call_and_return_conditional_losses_2458370

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????**:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????**2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????**::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????**
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_46_layer_call_and_return_conditional_losses_2460215

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_49_layer_call_and_return_conditional_losses_2460609

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????**:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????**2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????**::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????**
 
_user_specified_nameinputs
?

?
G__inference_conv2d_143_layer_call_and_return_conditional_losses_2458678

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_48_layer_call_and_return_conditional_losses_2460461

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
o
C__inference_add_28_layer_call_and_return_conditional_losses_2461071
inputs_0
inputs_1
identitya
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:?????????2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:?????????:?????????:Y U
/
_output_shapes
:?????????
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?

?
G__inference_conv2d_136_layer_call_and_return_conditional_losses_2458092

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_50_layer_call_and_return_conditional_losses_2458631

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
G__inference_conv2d_143_layer_call_and_return_conditional_losses_2460908

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_49_layer_call_fn_2460640

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????***$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_49_layer_call_and_return_conditional_losses_24584702
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????**2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????**::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????**
 
_user_specified_nameinputs
?

?
G__inference_conv2d_141_layer_call_and_return_conditional_losses_2458535

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????***
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????**2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????**2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????**::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????**
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_48_layer_call_fn_2460569

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????***&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_48_layer_call_and_return_conditional_losses_24583882
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????**2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????**::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????**
 
_user_specified_nameinputs
?
?
%__inference_signature_wrapper_2459541
input_30
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_30unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__wrapped_model_24574172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:???????????
"
_user_specified_name
input_30
?
?
S__inference_batch_normalization_47_layer_call_and_return_conditional_losses_2458245

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
H
,__inference_flatten_29_layer_call_fn_2461088

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_flatten_29_layer_call_and_return_conditional_losses_24588162
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_47_layer_call_fn_2460376

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_47_layer_call_and_return_conditional_losses_24582272
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_49_layer_call_fn_2460653

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????***&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_49_layer_call_and_return_conditional_losses_24584882
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????**2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????**::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????**
 
_user_specified_nameinputs
?

?
G__inference_conv2d_144_layer_call_and_return_conditional_losses_2461056

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
G__inference_conv2d_136_layer_call_and_return_conditional_losses_2460104

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_47_layer_call_fn_2460312

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_47_layer_call_and_return_conditional_losses_24575832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
9__inference_CNN_aug_deep_bnorm_skip_layer_call_fn_2459434
input_30
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_30unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_CNN_aug_deep_bnorm_skip_layer_call_and_return_conditional_losses_24593392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:???????????
"
_user_specified_name
input_30
?
T
(__inference_add_27_layer_call_fn_2460749
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*** 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_add_27_layer_call_and_return_conditional_losses_24585572
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????**2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:?????????**:?????????**:Y U
/
_output_shapes
:?????????**
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????**
"
_user_specified_name
inputs/1
?
?
S__inference_batch_normalization_51_layer_call_and_return_conditional_losses_2460937

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
T
(__inference_add_26_layer_call_fn_2460421
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_add_26_layer_call_and_return_conditional_losses_24583142
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*M
_input_shapes<
::???????????:???????????:[ W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/1
?
?
S__inference_batch_normalization_51_layer_call_and_return_conditional_losses_2461001

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
G__inference_conv2d_138_layer_call_and_return_conditional_losses_2458292

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_49_layer_call_and_return_conditional_losses_2460691

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_47_layer_call_and_return_conditional_losses_2460345

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_50_layer_call_fn_2460897

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_50_layer_call_and_return_conditional_losses_24579502
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
T
(__inference_add_28_layer_call_fn_2461077
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_add_28_layer_call_and_return_conditional_losses_24588002
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:?????????:?????????:Y U
/
_output_shapes
:?????????
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?

?
G__inference_conv2d_140_layer_call_and_return_conditional_losses_2460580

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????***
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????**2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????**2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????**::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????**
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_49_layer_call_and_return_conditional_losses_2457803

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
i
M__inference_max_pooling2d_56_layer_call_and_return_conditional_losses_2457851

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_51_layer_call_and_return_conditional_losses_2458713

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_46_layer_call_and_return_conditional_losses_2460197

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
m
C__inference_add_26_layer_call_and_return_conditional_losses_2458314

inputs
inputs_1
identitya
addAddV2inputsinputs_1*
T0*1
_output_shapes
:???????????2
adde
IdentityIdentityadd:z:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*M
_input_shapes<
::???????????:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs:YU
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
,__inference_conv2d_136_layer_call_fn_2460113

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_136_layer_call_and_return_conditional_losses_24580922
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_49_layer_call_fn_2460704

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_49_layer_call_and_return_conditional_losses_24578032
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_46_layer_call_and_return_conditional_losses_2460133

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_50_layer_call_and_return_conditional_losses_2458613

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
N
2__inference_max_pooling2d_57_layer_call_fn_2458077

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_max_pooling2d_57_layer_call_and_return_conditional_losses_24580712
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
,__inference_conv2d_141_layer_call_fn_2460737

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????***$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_141_layer_call_and_return_conditional_losses_24585352
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????**2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????**::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????**
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_48_layer_call_fn_2460492

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_48_layer_call_and_return_conditional_losses_24576992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_49_layer_call_and_return_conditional_losses_2458470

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????**:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????**2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????**::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????**
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_51_layer_call_fn_2460968

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_51_layer_call_and_return_conditional_losses_24580232
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
,__inference_conv2d_139_layer_call_fn_2460441

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????***$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_139_layer_call_and_return_conditional_losses_24583352
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????**2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????**::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????**
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_47_layer_call_and_return_conditional_losses_2460363

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
G__inference_conv2d_140_layer_call_and_return_conditional_losses_2458435

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????***
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????**2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????**2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????**::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????**
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_50_layer_call_and_return_conditional_losses_2460871

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?

?
G__inference_conv2d_139_layer_call_and_return_conditional_losses_2460432

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????***
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????**2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????**2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????**::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????**
 
_user_specified_nameinputs
??
?#
T__inference_CNN_aug_deep_bnorm_skip_layer_call_and_return_conditional_losses_2459899

inputs-
)conv2d_136_conv2d_readvariableop_resource.
*conv2d_136_biasadd_readvariableop_resource2
.batch_normalization_46_readvariableop_resource4
0batch_normalization_46_readvariableop_1_resourceC
?batch_normalization_46_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_46_fusedbatchnormv3_readvariableop_1_resource-
)conv2d_137_conv2d_readvariableop_resource.
*conv2d_137_biasadd_readvariableop_resource2
.batch_normalization_47_readvariableop_resource4
0batch_normalization_47_readvariableop_1_resourceC
?batch_normalization_47_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_47_fusedbatchnormv3_readvariableop_1_resource-
)conv2d_138_conv2d_readvariableop_resource.
*conv2d_138_biasadd_readvariableop_resource-
)conv2d_139_conv2d_readvariableop_resource.
*conv2d_139_biasadd_readvariableop_resource2
.batch_normalization_48_readvariableop_resource4
0batch_normalization_48_readvariableop_1_resourceC
?batch_normalization_48_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_48_fusedbatchnormv3_readvariableop_1_resource-
)conv2d_140_conv2d_readvariableop_resource.
*conv2d_140_biasadd_readvariableop_resource2
.batch_normalization_49_readvariableop_resource4
0batch_normalization_49_readvariableop_1_resourceC
?batch_normalization_49_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_49_fusedbatchnormv3_readvariableop_1_resource-
)conv2d_141_conv2d_readvariableop_resource.
*conv2d_141_biasadd_readvariableop_resource-
)conv2d_142_conv2d_readvariableop_resource.
*conv2d_142_biasadd_readvariableop_resource2
.batch_normalization_50_readvariableop_resource4
0batch_normalization_50_readvariableop_1_resourceC
?batch_normalization_50_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_50_fusedbatchnormv3_readvariableop_1_resource-
)conv2d_143_conv2d_readvariableop_resource.
*conv2d_143_biasadd_readvariableop_resource2
.batch_normalization_51_readvariableop_resource4
0batch_normalization_51_readvariableop_1_resourceC
?batch_normalization_51_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_51_fusedbatchnormv3_readvariableop_1_resource-
)conv2d_144_conv2d_readvariableop_resource.
*conv2d_144_biasadd_readvariableop_resource+
'dense_58_matmul_readvariableop_resource,
(dense_58_biasadd_readvariableop_resource+
'dense_59_matmul_readvariableop_resource,
(dense_59_biasadd_readvariableop_resource
identity??6batch_normalization_46/FusedBatchNormV3/ReadVariableOp?8batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_46/ReadVariableOp?'batch_normalization_46/ReadVariableOp_1?6batch_normalization_47/FusedBatchNormV3/ReadVariableOp?8batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_47/ReadVariableOp?'batch_normalization_47/ReadVariableOp_1?6batch_normalization_48/FusedBatchNormV3/ReadVariableOp?8batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_48/ReadVariableOp?'batch_normalization_48/ReadVariableOp_1?6batch_normalization_49/FusedBatchNormV3/ReadVariableOp?8batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_49/ReadVariableOp?'batch_normalization_49/ReadVariableOp_1?6batch_normalization_50/FusedBatchNormV3/ReadVariableOp?8batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_50/ReadVariableOp?'batch_normalization_50/ReadVariableOp_1?6batch_normalization_51/FusedBatchNormV3/ReadVariableOp?8batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_51/ReadVariableOp?'batch_normalization_51/ReadVariableOp_1?!conv2d_136/BiasAdd/ReadVariableOp? conv2d_136/Conv2D/ReadVariableOp?!conv2d_137/BiasAdd/ReadVariableOp? conv2d_137/Conv2D/ReadVariableOp?!conv2d_138/BiasAdd/ReadVariableOp? conv2d_138/Conv2D/ReadVariableOp?!conv2d_139/BiasAdd/ReadVariableOp? conv2d_139/Conv2D/ReadVariableOp?!conv2d_140/BiasAdd/ReadVariableOp? conv2d_140/Conv2D/ReadVariableOp?!conv2d_141/BiasAdd/ReadVariableOp? conv2d_141/Conv2D/ReadVariableOp?!conv2d_142/BiasAdd/ReadVariableOp? conv2d_142/Conv2D/ReadVariableOp?!conv2d_143/BiasAdd/ReadVariableOp? conv2d_143/Conv2D/ReadVariableOp?!conv2d_144/BiasAdd/ReadVariableOp? conv2d_144/Conv2D/ReadVariableOp?dense_58/BiasAdd/ReadVariableOp?dense_58/MatMul/ReadVariableOp?dense_59/BiasAdd/ReadVariableOp?dense_59/MatMul/ReadVariableOp?
 conv2d_136/Conv2D/ReadVariableOpReadVariableOp)conv2d_136_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_136/Conv2D/ReadVariableOp?
conv2d_136/Conv2DConv2Dinputs(conv2d_136/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_136/Conv2D?
!conv2d_136/BiasAdd/ReadVariableOpReadVariableOp*conv2d_136_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_136/BiasAdd/ReadVariableOp?
conv2d_136/BiasAddBiasAddconv2d_136/Conv2D:output:0)conv2d_136/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_136/BiasAdd?
conv2d_136/ReluReluconv2d_136/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_136/Relu?
%batch_normalization_46/ReadVariableOpReadVariableOp.batch_normalization_46_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_46/ReadVariableOp?
'batch_normalization_46/ReadVariableOp_1ReadVariableOp0batch_normalization_46_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_46/ReadVariableOp_1?
6batch_normalization_46/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_46_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_46/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_46_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_46/FusedBatchNormV3FusedBatchNormV3conv2d_136/Relu:activations:0-batch_normalization_46/ReadVariableOp:value:0/batch_normalization_46/ReadVariableOp_1:value:0>batch_normalization_46/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2)
'batch_normalization_46/FusedBatchNormV3?
 conv2d_137/Conv2D/ReadVariableOpReadVariableOp)conv2d_137_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_137/Conv2D/ReadVariableOp?
conv2d_137/Conv2DConv2D+batch_normalization_46/FusedBatchNormV3:y:0(conv2d_137/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_137/Conv2D?
!conv2d_137/BiasAdd/ReadVariableOpReadVariableOp*conv2d_137_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_137/BiasAdd/ReadVariableOp?
conv2d_137/BiasAddBiasAddconv2d_137/Conv2D:output:0)conv2d_137/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_137/BiasAdd?
conv2d_137/ReluReluconv2d_137/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_137/Relu?
%batch_normalization_47/ReadVariableOpReadVariableOp.batch_normalization_47_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_47/ReadVariableOp?
'batch_normalization_47/ReadVariableOp_1ReadVariableOp0batch_normalization_47_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_47/ReadVariableOp_1?
6batch_normalization_47/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_47_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_47/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_47_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_47/FusedBatchNormV3FusedBatchNormV3conv2d_137/Relu:activations:0-batch_normalization_47/ReadVariableOp:value:0/batch_normalization_47/ReadVariableOp_1:value:0>batch_normalization_47/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2)
'batch_normalization_47/FusedBatchNormV3?
 conv2d_138/Conv2D/ReadVariableOpReadVariableOp)conv2d_138_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_138/Conv2D/ReadVariableOp?
conv2d_138/Conv2DConv2D+batch_normalization_47/FusedBatchNormV3:y:0(conv2d_138/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_138/Conv2D?
!conv2d_138/BiasAdd/ReadVariableOpReadVariableOp*conv2d_138_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_138/BiasAdd/ReadVariableOp?
conv2d_138/BiasAddBiasAddconv2d_138/Conv2D:output:0)conv2d_138/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_138/BiasAdd?
conv2d_138/ReluReluconv2d_138/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_138/Relu?

add_26/addAddV2conv2d_138/Relu:activations:0inputs*
T0*1
_output_shapes
:???????????2

add_26/add?
max_pooling2d_55/MaxPoolMaxPooladd_26/add:z:0*/
_output_shapes
:?????????***
ksize
*
paddingVALID*
strides
2
max_pooling2d_55/MaxPool?
 conv2d_139/Conv2D/ReadVariableOpReadVariableOp)conv2d_139_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_139/Conv2D/ReadVariableOp?
conv2d_139/Conv2DConv2D!max_pooling2d_55/MaxPool:output:0(conv2d_139/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????***
paddingSAME*
strides
2
conv2d_139/Conv2D?
!conv2d_139/BiasAdd/ReadVariableOpReadVariableOp*conv2d_139_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_139/BiasAdd/ReadVariableOp?
conv2d_139/BiasAddBiasAddconv2d_139/Conv2D:output:0)conv2d_139/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**2
conv2d_139/BiasAdd?
conv2d_139/ReluReluconv2d_139/BiasAdd:output:0*
T0*/
_output_shapes
:?????????**2
conv2d_139/Relu?
%batch_normalization_48/ReadVariableOpReadVariableOp.batch_normalization_48_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_48/ReadVariableOp?
'batch_normalization_48/ReadVariableOp_1ReadVariableOp0batch_normalization_48_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_48/ReadVariableOp_1?
6batch_normalization_48/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_48_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_48/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_48_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_48/FusedBatchNormV3FusedBatchNormV3conv2d_139/Relu:activations:0-batch_normalization_48/ReadVariableOp:value:0/batch_normalization_48/ReadVariableOp_1:value:0>batch_normalization_48/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????**:::::*
epsilon%o?:*
is_training( 2)
'batch_normalization_48/FusedBatchNormV3?
 conv2d_140/Conv2D/ReadVariableOpReadVariableOp)conv2d_140_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_140/Conv2D/ReadVariableOp?
conv2d_140/Conv2DConv2D+batch_normalization_48/FusedBatchNormV3:y:0(conv2d_140/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????***
paddingSAME*
strides
2
conv2d_140/Conv2D?
!conv2d_140/BiasAdd/ReadVariableOpReadVariableOp*conv2d_140_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_140/BiasAdd/ReadVariableOp?
conv2d_140/BiasAddBiasAddconv2d_140/Conv2D:output:0)conv2d_140/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**2
conv2d_140/BiasAdd?
conv2d_140/ReluReluconv2d_140/BiasAdd:output:0*
T0*/
_output_shapes
:?????????**2
conv2d_140/Relu?
%batch_normalization_49/ReadVariableOpReadVariableOp.batch_normalization_49_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_49/ReadVariableOp?
'batch_normalization_49/ReadVariableOp_1ReadVariableOp0batch_normalization_49_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_49/ReadVariableOp_1?
6batch_normalization_49/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_49_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_49/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_49_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_49/FusedBatchNormV3FusedBatchNormV3conv2d_140/Relu:activations:0-batch_normalization_49/ReadVariableOp:value:0/batch_normalization_49/ReadVariableOp_1:value:0>batch_normalization_49/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????**:::::*
epsilon%o?:*
is_training( 2)
'batch_normalization_49/FusedBatchNormV3?
 conv2d_141/Conv2D/ReadVariableOpReadVariableOp)conv2d_141_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_141/Conv2D/ReadVariableOp?
conv2d_141/Conv2DConv2D+batch_normalization_49/FusedBatchNormV3:y:0(conv2d_141/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????***
paddingSAME*
strides
2
conv2d_141/Conv2D?
!conv2d_141/BiasAdd/ReadVariableOpReadVariableOp*conv2d_141_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_141/BiasAdd/ReadVariableOp?
conv2d_141/BiasAddBiasAddconv2d_141/Conv2D:output:0)conv2d_141/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**2
conv2d_141/BiasAdd?
conv2d_141/ReluReluconv2d_141/BiasAdd:output:0*
T0*/
_output_shapes
:?????????**2
conv2d_141/Relu?

add_27/addAddV2conv2d_141/Relu:activations:0!max_pooling2d_55/MaxPool:output:0*
T0*/
_output_shapes
:?????????**2

add_27/add?
max_pooling2d_56/MaxPoolMaxPooladd_27/add:z:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_56/MaxPool?
 conv2d_142/Conv2D/ReadVariableOpReadVariableOp)conv2d_142_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_142/Conv2D/ReadVariableOp?
conv2d_142/Conv2DConv2D!max_pooling2d_56/MaxPool:output:0(conv2d_142/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_142/Conv2D?
!conv2d_142/BiasAdd/ReadVariableOpReadVariableOp*conv2d_142_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_142/BiasAdd/ReadVariableOp?
conv2d_142/BiasAddBiasAddconv2d_142/Conv2D:output:0)conv2d_142/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_142/BiasAdd?
conv2d_142/ReluReluconv2d_142/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_142/Relu?
%batch_normalization_50/ReadVariableOpReadVariableOp.batch_normalization_50_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_50/ReadVariableOp?
'batch_normalization_50/ReadVariableOp_1ReadVariableOp0batch_normalization_50_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_50/ReadVariableOp_1?
6batch_normalization_50/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_50_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_50/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_50_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_50/FusedBatchNormV3FusedBatchNormV3conv2d_142/Relu:activations:0-batch_normalization_50/ReadVariableOp:value:0/batch_normalization_50/ReadVariableOp_1:value:0>batch_normalization_50/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( 2)
'batch_normalization_50/FusedBatchNormV3?
 conv2d_143/Conv2D/ReadVariableOpReadVariableOp)conv2d_143_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_143/Conv2D/ReadVariableOp?
conv2d_143/Conv2DConv2D+batch_normalization_50/FusedBatchNormV3:y:0(conv2d_143/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_143/Conv2D?
!conv2d_143/BiasAdd/ReadVariableOpReadVariableOp*conv2d_143_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_143/BiasAdd/ReadVariableOp?
conv2d_143/BiasAddBiasAddconv2d_143/Conv2D:output:0)conv2d_143/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_143/BiasAdd?
conv2d_143/ReluReluconv2d_143/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_143/Relu?
%batch_normalization_51/ReadVariableOpReadVariableOp.batch_normalization_51_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_51/ReadVariableOp?
'batch_normalization_51/ReadVariableOp_1ReadVariableOp0batch_normalization_51_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_51/ReadVariableOp_1?
6batch_normalization_51/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_51_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_51/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_51_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_51/FusedBatchNormV3FusedBatchNormV3conv2d_143/Relu:activations:0-batch_normalization_51/ReadVariableOp:value:0/batch_normalization_51/ReadVariableOp_1:value:0>batch_normalization_51/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( 2)
'batch_normalization_51/FusedBatchNormV3?
 conv2d_144/Conv2D/ReadVariableOpReadVariableOp)conv2d_144_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_144/Conv2D/ReadVariableOp?
conv2d_144/Conv2DConv2D+batch_normalization_51/FusedBatchNormV3:y:0(conv2d_144/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_144/Conv2D?
!conv2d_144/BiasAdd/ReadVariableOpReadVariableOp*conv2d_144_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_144/BiasAdd/ReadVariableOp?
conv2d_144/BiasAddBiasAddconv2d_144/Conv2D:output:0)conv2d_144/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_144/BiasAdd?
conv2d_144/ReluReluconv2d_144/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_144/Relu?

add_28/addAddV2conv2d_144/Relu:activations:0!max_pooling2d_56/MaxPool:output:0*
T0*/
_output_shapes
:?????????2

add_28/add?
max_pooling2d_57/MaxPoolMaxPooladd_28/add:z:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_57/MaxPoolu
flatten_29/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_29/Const?
flatten_29/ReshapeReshape!max_pooling2d_57/MaxPool:output:0flatten_29/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_29/Reshape?
dense_58/MatMul/ReadVariableOpReadVariableOp'dense_58_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_58/MatMul/ReadVariableOp?
dense_58/MatMulMatMulflatten_29/Reshape:output:0&dense_58/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_58/MatMul?
dense_58/BiasAdd/ReadVariableOpReadVariableOp(dense_58_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_58/BiasAdd/ReadVariableOp?
dense_58/BiasAddBiasAdddense_58/MatMul:product:0'dense_58/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_58/BiasAdds
dense_58/ReluReludense_58/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_58/Relu?
dense_59/MatMul/ReadVariableOpReadVariableOp'dense_59_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_59/MatMul/ReadVariableOp?
dense_59/MatMulMatMuldense_58/Relu:activations:0&dense_59/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_59/MatMul?
dense_59/BiasAdd/ReadVariableOpReadVariableOp(dense_59_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_59/BiasAdd/ReadVariableOp?
dense_59/BiasAddBiasAdddense_59/MatMul:product:0'dense_59/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_59/BiasAdd|
dense_59/SoftmaxSoftmaxdense_59/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_59/Softmax?
IdentityIdentitydense_59/Softmax:softmax:07^batch_normalization_46/FusedBatchNormV3/ReadVariableOp9^batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_46/ReadVariableOp(^batch_normalization_46/ReadVariableOp_17^batch_normalization_47/FusedBatchNormV3/ReadVariableOp9^batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_47/ReadVariableOp(^batch_normalization_47/ReadVariableOp_17^batch_normalization_48/FusedBatchNormV3/ReadVariableOp9^batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_48/ReadVariableOp(^batch_normalization_48/ReadVariableOp_17^batch_normalization_49/FusedBatchNormV3/ReadVariableOp9^batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_49/ReadVariableOp(^batch_normalization_49/ReadVariableOp_17^batch_normalization_50/FusedBatchNormV3/ReadVariableOp9^batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_50/ReadVariableOp(^batch_normalization_50/ReadVariableOp_17^batch_normalization_51/FusedBatchNormV3/ReadVariableOp9^batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_51/ReadVariableOp(^batch_normalization_51/ReadVariableOp_1"^conv2d_136/BiasAdd/ReadVariableOp!^conv2d_136/Conv2D/ReadVariableOp"^conv2d_137/BiasAdd/ReadVariableOp!^conv2d_137/Conv2D/ReadVariableOp"^conv2d_138/BiasAdd/ReadVariableOp!^conv2d_138/Conv2D/ReadVariableOp"^conv2d_139/BiasAdd/ReadVariableOp!^conv2d_139/Conv2D/ReadVariableOp"^conv2d_140/BiasAdd/ReadVariableOp!^conv2d_140/Conv2D/ReadVariableOp"^conv2d_141/BiasAdd/ReadVariableOp!^conv2d_141/Conv2D/ReadVariableOp"^conv2d_142/BiasAdd/ReadVariableOp!^conv2d_142/Conv2D/ReadVariableOp"^conv2d_143/BiasAdd/ReadVariableOp!^conv2d_143/Conv2D/ReadVariableOp"^conv2d_144/BiasAdd/ReadVariableOp!^conv2d_144/Conv2D/ReadVariableOp ^dense_58/BiasAdd/ReadVariableOp^dense_58/MatMul/ReadVariableOp ^dense_59/BiasAdd/ReadVariableOp^dense_59/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::::::::2p
6batch_normalization_46/FusedBatchNormV3/ReadVariableOp6batch_normalization_46/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_46/FusedBatchNormV3/ReadVariableOp_18batch_normalization_46/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_46/ReadVariableOp%batch_normalization_46/ReadVariableOp2R
'batch_normalization_46/ReadVariableOp_1'batch_normalization_46/ReadVariableOp_12p
6batch_normalization_47/FusedBatchNormV3/ReadVariableOp6batch_normalization_47/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_47/FusedBatchNormV3/ReadVariableOp_18batch_normalization_47/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_47/ReadVariableOp%batch_normalization_47/ReadVariableOp2R
'batch_normalization_47/ReadVariableOp_1'batch_normalization_47/ReadVariableOp_12p
6batch_normalization_48/FusedBatchNormV3/ReadVariableOp6batch_normalization_48/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_48/FusedBatchNormV3/ReadVariableOp_18batch_normalization_48/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_48/ReadVariableOp%batch_normalization_48/ReadVariableOp2R
'batch_normalization_48/ReadVariableOp_1'batch_normalization_48/ReadVariableOp_12p
6batch_normalization_49/FusedBatchNormV3/ReadVariableOp6batch_normalization_49/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_49/FusedBatchNormV3/ReadVariableOp_18batch_normalization_49/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_49/ReadVariableOp%batch_normalization_49/ReadVariableOp2R
'batch_normalization_49/ReadVariableOp_1'batch_normalization_49/ReadVariableOp_12p
6batch_normalization_50/FusedBatchNormV3/ReadVariableOp6batch_normalization_50/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_50/FusedBatchNormV3/ReadVariableOp_18batch_normalization_50/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_50/ReadVariableOp%batch_normalization_50/ReadVariableOp2R
'batch_normalization_50/ReadVariableOp_1'batch_normalization_50/ReadVariableOp_12p
6batch_normalization_51/FusedBatchNormV3/ReadVariableOp6batch_normalization_51/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_51/FusedBatchNormV3/ReadVariableOp_18batch_normalization_51/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_51/ReadVariableOp%batch_normalization_51/ReadVariableOp2R
'batch_normalization_51/ReadVariableOp_1'batch_normalization_51/ReadVariableOp_12F
!conv2d_136/BiasAdd/ReadVariableOp!conv2d_136/BiasAdd/ReadVariableOp2D
 conv2d_136/Conv2D/ReadVariableOp conv2d_136/Conv2D/ReadVariableOp2F
!conv2d_137/BiasAdd/ReadVariableOp!conv2d_137/BiasAdd/ReadVariableOp2D
 conv2d_137/Conv2D/ReadVariableOp conv2d_137/Conv2D/ReadVariableOp2F
!conv2d_138/BiasAdd/ReadVariableOp!conv2d_138/BiasAdd/ReadVariableOp2D
 conv2d_138/Conv2D/ReadVariableOp conv2d_138/Conv2D/ReadVariableOp2F
!conv2d_139/BiasAdd/ReadVariableOp!conv2d_139/BiasAdd/ReadVariableOp2D
 conv2d_139/Conv2D/ReadVariableOp conv2d_139/Conv2D/ReadVariableOp2F
!conv2d_140/BiasAdd/ReadVariableOp!conv2d_140/BiasAdd/ReadVariableOp2D
 conv2d_140/Conv2D/ReadVariableOp conv2d_140/Conv2D/ReadVariableOp2F
!conv2d_141/BiasAdd/ReadVariableOp!conv2d_141/BiasAdd/ReadVariableOp2D
 conv2d_141/Conv2D/ReadVariableOp conv2d_141/Conv2D/ReadVariableOp2F
!conv2d_142/BiasAdd/ReadVariableOp!conv2d_142/BiasAdd/ReadVariableOp2D
 conv2d_142/Conv2D/ReadVariableOp conv2d_142/Conv2D/ReadVariableOp2F
!conv2d_143/BiasAdd/ReadVariableOp!conv2d_143/BiasAdd/ReadVariableOp2D
 conv2d_143/Conv2D/ReadVariableOp conv2d_143/Conv2D/ReadVariableOp2F
!conv2d_144/BiasAdd/ReadVariableOp!conv2d_144/BiasAdd/ReadVariableOp2D
 conv2d_144/Conv2D/ReadVariableOp conv2d_144/Conv2D/ReadVariableOp2B
dense_58/BiasAdd/ReadVariableOpdense_58/BiasAdd/ReadVariableOp2@
dense_58/MatMul/ReadVariableOpdense_58/MatMul/ReadVariableOp2B
dense_59/BiasAdd/ReadVariableOpdense_59/BiasAdd/ReadVariableOp2@
dense_59/MatMul/ReadVariableOpdense_59/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
9__inference_CNN_aug_deep_bnorm_skip_layer_call_fn_2459996

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*D
_read_only_resource_inputs&
$"	
 #$%&)*+,-.*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_CNN_aug_deep_bnorm_skip_layer_call_and_return_conditional_losses_24591222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
??
?4
"__inference__wrapped_model_2457417
input_30E
Acnn_aug_deep_bnorm_skip_conv2d_136_conv2d_readvariableop_resourceF
Bcnn_aug_deep_bnorm_skip_conv2d_136_biasadd_readvariableop_resourceJ
Fcnn_aug_deep_bnorm_skip_batch_normalization_46_readvariableop_resourceL
Hcnn_aug_deep_bnorm_skip_batch_normalization_46_readvariableop_1_resource[
Wcnn_aug_deep_bnorm_skip_batch_normalization_46_fusedbatchnormv3_readvariableop_resource]
Ycnn_aug_deep_bnorm_skip_batch_normalization_46_fusedbatchnormv3_readvariableop_1_resourceE
Acnn_aug_deep_bnorm_skip_conv2d_137_conv2d_readvariableop_resourceF
Bcnn_aug_deep_bnorm_skip_conv2d_137_biasadd_readvariableop_resourceJ
Fcnn_aug_deep_bnorm_skip_batch_normalization_47_readvariableop_resourceL
Hcnn_aug_deep_bnorm_skip_batch_normalization_47_readvariableop_1_resource[
Wcnn_aug_deep_bnorm_skip_batch_normalization_47_fusedbatchnormv3_readvariableop_resource]
Ycnn_aug_deep_bnorm_skip_batch_normalization_47_fusedbatchnormv3_readvariableop_1_resourceE
Acnn_aug_deep_bnorm_skip_conv2d_138_conv2d_readvariableop_resourceF
Bcnn_aug_deep_bnorm_skip_conv2d_138_biasadd_readvariableop_resourceE
Acnn_aug_deep_bnorm_skip_conv2d_139_conv2d_readvariableop_resourceF
Bcnn_aug_deep_bnorm_skip_conv2d_139_biasadd_readvariableop_resourceJ
Fcnn_aug_deep_bnorm_skip_batch_normalization_48_readvariableop_resourceL
Hcnn_aug_deep_bnorm_skip_batch_normalization_48_readvariableop_1_resource[
Wcnn_aug_deep_bnorm_skip_batch_normalization_48_fusedbatchnormv3_readvariableop_resource]
Ycnn_aug_deep_bnorm_skip_batch_normalization_48_fusedbatchnormv3_readvariableop_1_resourceE
Acnn_aug_deep_bnorm_skip_conv2d_140_conv2d_readvariableop_resourceF
Bcnn_aug_deep_bnorm_skip_conv2d_140_biasadd_readvariableop_resourceJ
Fcnn_aug_deep_bnorm_skip_batch_normalization_49_readvariableop_resourceL
Hcnn_aug_deep_bnorm_skip_batch_normalization_49_readvariableop_1_resource[
Wcnn_aug_deep_bnorm_skip_batch_normalization_49_fusedbatchnormv3_readvariableop_resource]
Ycnn_aug_deep_bnorm_skip_batch_normalization_49_fusedbatchnormv3_readvariableop_1_resourceE
Acnn_aug_deep_bnorm_skip_conv2d_141_conv2d_readvariableop_resourceF
Bcnn_aug_deep_bnorm_skip_conv2d_141_biasadd_readvariableop_resourceE
Acnn_aug_deep_bnorm_skip_conv2d_142_conv2d_readvariableop_resourceF
Bcnn_aug_deep_bnorm_skip_conv2d_142_biasadd_readvariableop_resourceJ
Fcnn_aug_deep_bnorm_skip_batch_normalization_50_readvariableop_resourceL
Hcnn_aug_deep_bnorm_skip_batch_normalization_50_readvariableop_1_resource[
Wcnn_aug_deep_bnorm_skip_batch_normalization_50_fusedbatchnormv3_readvariableop_resource]
Ycnn_aug_deep_bnorm_skip_batch_normalization_50_fusedbatchnormv3_readvariableop_1_resourceE
Acnn_aug_deep_bnorm_skip_conv2d_143_conv2d_readvariableop_resourceF
Bcnn_aug_deep_bnorm_skip_conv2d_143_biasadd_readvariableop_resourceJ
Fcnn_aug_deep_bnorm_skip_batch_normalization_51_readvariableop_resourceL
Hcnn_aug_deep_bnorm_skip_batch_normalization_51_readvariableop_1_resource[
Wcnn_aug_deep_bnorm_skip_batch_normalization_51_fusedbatchnormv3_readvariableop_resource]
Ycnn_aug_deep_bnorm_skip_batch_normalization_51_fusedbatchnormv3_readvariableop_1_resourceE
Acnn_aug_deep_bnorm_skip_conv2d_144_conv2d_readvariableop_resourceF
Bcnn_aug_deep_bnorm_skip_conv2d_144_biasadd_readvariableop_resourceC
?cnn_aug_deep_bnorm_skip_dense_58_matmul_readvariableop_resourceD
@cnn_aug_deep_bnorm_skip_dense_58_biasadd_readvariableop_resourceC
?cnn_aug_deep_bnorm_skip_dense_59_matmul_readvariableop_resourceD
@cnn_aug_deep_bnorm_skip_dense_59_biasadd_readvariableop_resource
identity??NCNN_aug_deep_bnorm_skip/batch_normalization_46/FusedBatchNormV3/ReadVariableOp?PCNN_aug_deep_bnorm_skip/batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1?=CNN_aug_deep_bnorm_skip/batch_normalization_46/ReadVariableOp??CNN_aug_deep_bnorm_skip/batch_normalization_46/ReadVariableOp_1?NCNN_aug_deep_bnorm_skip/batch_normalization_47/FusedBatchNormV3/ReadVariableOp?PCNN_aug_deep_bnorm_skip/batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1?=CNN_aug_deep_bnorm_skip/batch_normalization_47/ReadVariableOp??CNN_aug_deep_bnorm_skip/batch_normalization_47/ReadVariableOp_1?NCNN_aug_deep_bnorm_skip/batch_normalization_48/FusedBatchNormV3/ReadVariableOp?PCNN_aug_deep_bnorm_skip/batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1?=CNN_aug_deep_bnorm_skip/batch_normalization_48/ReadVariableOp??CNN_aug_deep_bnorm_skip/batch_normalization_48/ReadVariableOp_1?NCNN_aug_deep_bnorm_skip/batch_normalization_49/FusedBatchNormV3/ReadVariableOp?PCNN_aug_deep_bnorm_skip/batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1?=CNN_aug_deep_bnorm_skip/batch_normalization_49/ReadVariableOp??CNN_aug_deep_bnorm_skip/batch_normalization_49/ReadVariableOp_1?NCNN_aug_deep_bnorm_skip/batch_normalization_50/FusedBatchNormV3/ReadVariableOp?PCNN_aug_deep_bnorm_skip/batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1?=CNN_aug_deep_bnorm_skip/batch_normalization_50/ReadVariableOp??CNN_aug_deep_bnorm_skip/batch_normalization_50/ReadVariableOp_1?NCNN_aug_deep_bnorm_skip/batch_normalization_51/FusedBatchNormV3/ReadVariableOp?PCNN_aug_deep_bnorm_skip/batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1?=CNN_aug_deep_bnorm_skip/batch_normalization_51/ReadVariableOp??CNN_aug_deep_bnorm_skip/batch_normalization_51/ReadVariableOp_1?9CNN_aug_deep_bnorm_skip/conv2d_136/BiasAdd/ReadVariableOp?8CNN_aug_deep_bnorm_skip/conv2d_136/Conv2D/ReadVariableOp?9CNN_aug_deep_bnorm_skip/conv2d_137/BiasAdd/ReadVariableOp?8CNN_aug_deep_bnorm_skip/conv2d_137/Conv2D/ReadVariableOp?9CNN_aug_deep_bnorm_skip/conv2d_138/BiasAdd/ReadVariableOp?8CNN_aug_deep_bnorm_skip/conv2d_138/Conv2D/ReadVariableOp?9CNN_aug_deep_bnorm_skip/conv2d_139/BiasAdd/ReadVariableOp?8CNN_aug_deep_bnorm_skip/conv2d_139/Conv2D/ReadVariableOp?9CNN_aug_deep_bnorm_skip/conv2d_140/BiasAdd/ReadVariableOp?8CNN_aug_deep_bnorm_skip/conv2d_140/Conv2D/ReadVariableOp?9CNN_aug_deep_bnorm_skip/conv2d_141/BiasAdd/ReadVariableOp?8CNN_aug_deep_bnorm_skip/conv2d_141/Conv2D/ReadVariableOp?9CNN_aug_deep_bnorm_skip/conv2d_142/BiasAdd/ReadVariableOp?8CNN_aug_deep_bnorm_skip/conv2d_142/Conv2D/ReadVariableOp?9CNN_aug_deep_bnorm_skip/conv2d_143/BiasAdd/ReadVariableOp?8CNN_aug_deep_bnorm_skip/conv2d_143/Conv2D/ReadVariableOp?9CNN_aug_deep_bnorm_skip/conv2d_144/BiasAdd/ReadVariableOp?8CNN_aug_deep_bnorm_skip/conv2d_144/Conv2D/ReadVariableOp?7CNN_aug_deep_bnorm_skip/dense_58/BiasAdd/ReadVariableOp?6CNN_aug_deep_bnorm_skip/dense_58/MatMul/ReadVariableOp?7CNN_aug_deep_bnorm_skip/dense_59/BiasAdd/ReadVariableOp?6CNN_aug_deep_bnorm_skip/dense_59/MatMul/ReadVariableOp?
8CNN_aug_deep_bnorm_skip/conv2d_136/Conv2D/ReadVariableOpReadVariableOpAcnn_aug_deep_bnorm_skip_conv2d_136_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02:
8CNN_aug_deep_bnorm_skip/conv2d_136/Conv2D/ReadVariableOp?
)CNN_aug_deep_bnorm_skip/conv2d_136/Conv2DConv2Dinput_30@CNN_aug_deep_bnorm_skip/conv2d_136/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2+
)CNN_aug_deep_bnorm_skip/conv2d_136/Conv2D?
9CNN_aug_deep_bnorm_skip/conv2d_136/BiasAdd/ReadVariableOpReadVariableOpBcnn_aug_deep_bnorm_skip_conv2d_136_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02;
9CNN_aug_deep_bnorm_skip/conv2d_136/BiasAdd/ReadVariableOp?
*CNN_aug_deep_bnorm_skip/conv2d_136/BiasAddBiasAdd2CNN_aug_deep_bnorm_skip/conv2d_136/Conv2D:output:0ACNN_aug_deep_bnorm_skip/conv2d_136/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2,
*CNN_aug_deep_bnorm_skip/conv2d_136/BiasAdd?
'CNN_aug_deep_bnorm_skip/conv2d_136/ReluRelu3CNN_aug_deep_bnorm_skip/conv2d_136/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2)
'CNN_aug_deep_bnorm_skip/conv2d_136/Relu?
=CNN_aug_deep_bnorm_skip/batch_normalization_46/ReadVariableOpReadVariableOpFcnn_aug_deep_bnorm_skip_batch_normalization_46_readvariableop_resource*
_output_shapes
:*
dtype02?
=CNN_aug_deep_bnorm_skip/batch_normalization_46/ReadVariableOp?
?CNN_aug_deep_bnorm_skip/batch_normalization_46/ReadVariableOp_1ReadVariableOpHcnn_aug_deep_bnorm_skip_batch_normalization_46_readvariableop_1_resource*
_output_shapes
:*
dtype02A
?CNN_aug_deep_bnorm_skip/batch_normalization_46/ReadVariableOp_1?
NCNN_aug_deep_bnorm_skip/batch_normalization_46/FusedBatchNormV3/ReadVariableOpReadVariableOpWcnn_aug_deep_bnorm_skip_batch_normalization_46_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02P
NCNN_aug_deep_bnorm_skip/batch_normalization_46/FusedBatchNormV3/ReadVariableOp?
PCNN_aug_deep_bnorm_skip/batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpYcnn_aug_deep_bnorm_skip_batch_normalization_46_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02R
PCNN_aug_deep_bnorm_skip/batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1?
?CNN_aug_deep_bnorm_skip/batch_normalization_46/FusedBatchNormV3FusedBatchNormV35CNN_aug_deep_bnorm_skip/conv2d_136/Relu:activations:0ECNN_aug_deep_bnorm_skip/batch_normalization_46/ReadVariableOp:value:0GCNN_aug_deep_bnorm_skip/batch_normalization_46/ReadVariableOp_1:value:0VCNN_aug_deep_bnorm_skip/batch_normalization_46/FusedBatchNormV3/ReadVariableOp:value:0XCNN_aug_deep_bnorm_skip/batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2A
?CNN_aug_deep_bnorm_skip/batch_normalization_46/FusedBatchNormV3?
8CNN_aug_deep_bnorm_skip/conv2d_137/Conv2D/ReadVariableOpReadVariableOpAcnn_aug_deep_bnorm_skip_conv2d_137_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02:
8CNN_aug_deep_bnorm_skip/conv2d_137/Conv2D/ReadVariableOp?
)CNN_aug_deep_bnorm_skip/conv2d_137/Conv2DConv2DCCNN_aug_deep_bnorm_skip/batch_normalization_46/FusedBatchNormV3:y:0@CNN_aug_deep_bnorm_skip/conv2d_137/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2+
)CNN_aug_deep_bnorm_skip/conv2d_137/Conv2D?
9CNN_aug_deep_bnorm_skip/conv2d_137/BiasAdd/ReadVariableOpReadVariableOpBcnn_aug_deep_bnorm_skip_conv2d_137_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02;
9CNN_aug_deep_bnorm_skip/conv2d_137/BiasAdd/ReadVariableOp?
*CNN_aug_deep_bnorm_skip/conv2d_137/BiasAddBiasAdd2CNN_aug_deep_bnorm_skip/conv2d_137/Conv2D:output:0ACNN_aug_deep_bnorm_skip/conv2d_137/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2,
*CNN_aug_deep_bnorm_skip/conv2d_137/BiasAdd?
'CNN_aug_deep_bnorm_skip/conv2d_137/ReluRelu3CNN_aug_deep_bnorm_skip/conv2d_137/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2)
'CNN_aug_deep_bnorm_skip/conv2d_137/Relu?
=CNN_aug_deep_bnorm_skip/batch_normalization_47/ReadVariableOpReadVariableOpFcnn_aug_deep_bnorm_skip_batch_normalization_47_readvariableop_resource*
_output_shapes
:*
dtype02?
=CNN_aug_deep_bnorm_skip/batch_normalization_47/ReadVariableOp?
?CNN_aug_deep_bnorm_skip/batch_normalization_47/ReadVariableOp_1ReadVariableOpHcnn_aug_deep_bnorm_skip_batch_normalization_47_readvariableop_1_resource*
_output_shapes
:*
dtype02A
?CNN_aug_deep_bnorm_skip/batch_normalization_47/ReadVariableOp_1?
NCNN_aug_deep_bnorm_skip/batch_normalization_47/FusedBatchNormV3/ReadVariableOpReadVariableOpWcnn_aug_deep_bnorm_skip_batch_normalization_47_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02P
NCNN_aug_deep_bnorm_skip/batch_normalization_47/FusedBatchNormV3/ReadVariableOp?
PCNN_aug_deep_bnorm_skip/batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpYcnn_aug_deep_bnorm_skip_batch_normalization_47_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02R
PCNN_aug_deep_bnorm_skip/batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1?
?CNN_aug_deep_bnorm_skip/batch_normalization_47/FusedBatchNormV3FusedBatchNormV35CNN_aug_deep_bnorm_skip/conv2d_137/Relu:activations:0ECNN_aug_deep_bnorm_skip/batch_normalization_47/ReadVariableOp:value:0GCNN_aug_deep_bnorm_skip/batch_normalization_47/ReadVariableOp_1:value:0VCNN_aug_deep_bnorm_skip/batch_normalization_47/FusedBatchNormV3/ReadVariableOp:value:0XCNN_aug_deep_bnorm_skip/batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2A
?CNN_aug_deep_bnorm_skip/batch_normalization_47/FusedBatchNormV3?
8CNN_aug_deep_bnorm_skip/conv2d_138/Conv2D/ReadVariableOpReadVariableOpAcnn_aug_deep_bnorm_skip_conv2d_138_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02:
8CNN_aug_deep_bnorm_skip/conv2d_138/Conv2D/ReadVariableOp?
)CNN_aug_deep_bnorm_skip/conv2d_138/Conv2DConv2DCCNN_aug_deep_bnorm_skip/batch_normalization_47/FusedBatchNormV3:y:0@CNN_aug_deep_bnorm_skip/conv2d_138/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2+
)CNN_aug_deep_bnorm_skip/conv2d_138/Conv2D?
9CNN_aug_deep_bnorm_skip/conv2d_138/BiasAdd/ReadVariableOpReadVariableOpBcnn_aug_deep_bnorm_skip_conv2d_138_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02;
9CNN_aug_deep_bnorm_skip/conv2d_138/BiasAdd/ReadVariableOp?
*CNN_aug_deep_bnorm_skip/conv2d_138/BiasAddBiasAdd2CNN_aug_deep_bnorm_skip/conv2d_138/Conv2D:output:0ACNN_aug_deep_bnorm_skip/conv2d_138/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2,
*CNN_aug_deep_bnorm_skip/conv2d_138/BiasAdd?
'CNN_aug_deep_bnorm_skip/conv2d_138/ReluRelu3CNN_aug_deep_bnorm_skip/conv2d_138/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2)
'CNN_aug_deep_bnorm_skip/conv2d_138/Relu?
"CNN_aug_deep_bnorm_skip/add_26/addAddV25CNN_aug_deep_bnorm_skip/conv2d_138/Relu:activations:0input_30*
T0*1
_output_shapes
:???????????2$
"CNN_aug_deep_bnorm_skip/add_26/add?
0CNN_aug_deep_bnorm_skip/max_pooling2d_55/MaxPoolMaxPool&CNN_aug_deep_bnorm_skip/add_26/add:z:0*/
_output_shapes
:?????????***
ksize
*
paddingVALID*
strides
22
0CNN_aug_deep_bnorm_skip/max_pooling2d_55/MaxPool?
8CNN_aug_deep_bnorm_skip/conv2d_139/Conv2D/ReadVariableOpReadVariableOpAcnn_aug_deep_bnorm_skip_conv2d_139_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02:
8CNN_aug_deep_bnorm_skip/conv2d_139/Conv2D/ReadVariableOp?
)CNN_aug_deep_bnorm_skip/conv2d_139/Conv2DConv2D9CNN_aug_deep_bnorm_skip/max_pooling2d_55/MaxPool:output:0@CNN_aug_deep_bnorm_skip/conv2d_139/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????***
paddingSAME*
strides
2+
)CNN_aug_deep_bnorm_skip/conv2d_139/Conv2D?
9CNN_aug_deep_bnorm_skip/conv2d_139/BiasAdd/ReadVariableOpReadVariableOpBcnn_aug_deep_bnorm_skip_conv2d_139_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02;
9CNN_aug_deep_bnorm_skip/conv2d_139/BiasAdd/ReadVariableOp?
*CNN_aug_deep_bnorm_skip/conv2d_139/BiasAddBiasAdd2CNN_aug_deep_bnorm_skip/conv2d_139/Conv2D:output:0ACNN_aug_deep_bnorm_skip/conv2d_139/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**2,
*CNN_aug_deep_bnorm_skip/conv2d_139/BiasAdd?
'CNN_aug_deep_bnorm_skip/conv2d_139/ReluRelu3CNN_aug_deep_bnorm_skip/conv2d_139/BiasAdd:output:0*
T0*/
_output_shapes
:?????????**2)
'CNN_aug_deep_bnorm_skip/conv2d_139/Relu?
=CNN_aug_deep_bnorm_skip/batch_normalization_48/ReadVariableOpReadVariableOpFcnn_aug_deep_bnorm_skip_batch_normalization_48_readvariableop_resource*
_output_shapes
:*
dtype02?
=CNN_aug_deep_bnorm_skip/batch_normalization_48/ReadVariableOp?
?CNN_aug_deep_bnorm_skip/batch_normalization_48/ReadVariableOp_1ReadVariableOpHcnn_aug_deep_bnorm_skip_batch_normalization_48_readvariableop_1_resource*
_output_shapes
:*
dtype02A
?CNN_aug_deep_bnorm_skip/batch_normalization_48/ReadVariableOp_1?
NCNN_aug_deep_bnorm_skip/batch_normalization_48/FusedBatchNormV3/ReadVariableOpReadVariableOpWcnn_aug_deep_bnorm_skip_batch_normalization_48_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02P
NCNN_aug_deep_bnorm_skip/batch_normalization_48/FusedBatchNormV3/ReadVariableOp?
PCNN_aug_deep_bnorm_skip/batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpYcnn_aug_deep_bnorm_skip_batch_normalization_48_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02R
PCNN_aug_deep_bnorm_skip/batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1?
?CNN_aug_deep_bnorm_skip/batch_normalization_48/FusedBatchNormV3FusedBatchNormV35CNN_aug_deep_bnorm_skip/conv2d_139/Relu:activations:0ECNN_aug_deep_bnorm_skip/batch_normalization_48/ReadVariableOp:value:0GCNN_aug_deep_bnorm_skip/batch_normalization_48/ReadVariableOp_1:value:0VCNN_aug_deep_bnorm_skip/batch_normalization_48/FusedBatchNormV3/ReadVariableOp:value:0XCNN_aug_deep_bnorm_skip/batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????**:::::*
epsilon%o?:*
is_training( 2A
?CNN_aug_deep_bnorm_skip/batch_normalization_48/FusedBatchNormV3?
8CNN_aug_deep_bnorm_skip/conv2d_140/Conv2D/ReadVariableOpReadVariableOpAcnn_aug_deep_bnorm_skip_conv2d_140_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02:
8CNN_aug_deep_bnorm_skip/conv2d_140/Conv2D/ReadVariableOp?
)CNN_aug_deep_bnorm_skip/conv2d_140/Conv2DConv2DCCNN_aug_deep_bnorm_skip/batch_normalization_48/FusedBatchNormV3:y:0@CNN_aug_deep_bnorm_skip/conv2d_140/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????***
paddingSAME*
strides
2+
)CNN_aug_deep_bnorm_skip/conv2d_140/Conv2D?
9CNN_aug_deep_bnorm_skip/conv2d_140/BiasAdd/ReadVariableOpReadVariableOpBcnn_aug_deep_bnorm_skip_conv2d_140_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02;
9CNN_aug_deep_bnorm_skip/conv2d_140/BiasAdd/ReadVariableOp?
*CNN_aug_deep_bnorm_skip/conv2d_140/BiasAddBiasAdd2CNN_aug_deep_bnorm_skip/conv2d_140/Conv2D:output:0ACNN_aug_deep_bnorm_skip/conv2d_140/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**2,
*CNN_aug_deep_bnorm_skip/conv2d_140/BiasAdd?
'CNN_aug_deep_bnorm_skip/conv2d_140/ReluRelu3CNN_aug_deep_bnorm_skip/conv2d_140/BiasAdd:output:0*
T0*/
_output_shapes
:?????????**2)
'CNN_aug_deep_bnorm_skip/conv2d_140/Relu?
=CNN_aug_deep_bnorm_skip/batch_normalization_49/ReadVariableOpReadVariableOpFcnn_aug_deep_bnorm_skip_batch_normalization_49_readvariableop_resource*
_output_shapes
:*
dtype02?
=CNN_aug_deep_bnorm_skip/batch_normalization_49/ReadVariableOp?
?CNN_aug_deep_bnorm_skip/batch_normalization_49/ReadVariableOp_1ReadVariableOpHcnn_aug_deep_bnorm_skip_batch_normalization_49_readvariableop_1_resource*
_output_shapes
:*
dtype02A
?CNN_aug_deep_bnorm_skip/batch_normalization_49/ReadVariableOp_1?
NCNN_aug_deep_bnorm_skip/batch_normalization_49/FusedBatchNormV3/ReadVariableOpReadVariableOpWcnn_aug_deep_bnorm_skip_batch_normalization_49_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02P
NCNN_aug_deep_bnorm_skip/batch_normalization_49/FusedBatchNormV3/ReadVariableOp?
PCNN_aug_deep_bnorm_skip/batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpYcnn_aug_deep_bnorm_skip_batch_normalization_49_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02R
PCNN_aug_deep_bnorm_skip/batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1?
?CNN_aug_deep_bnorm_skip/batch_normalization_49/FusedBatchNormV3FusedBatchNormV35CNN_aug_deep_bnorm_skip/conv2d_140/Relu:activations:0ECNN_aug_deep_bnorm_skip/batch_normalization_49/ReadVariableOp:value:0GCNN_aug_deep_bnorm_skip/batch_normalization_49/ReadVariableOp_1:value:0VCNN_aug_deep_bnorm_skip/batch_normalization_49/FusedBatchNormV3/ReadVariableOp:value:0XCNN_aug_deep_bnorm_skip/batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????**:::::*
epsilon%o?:*
is_training( 2A
?CNN_aug_deep_bnorm_skip/batch_normalization_49/FusedBatchNormV3?
8CNN_aug_deep_bnorm_skip/conv2d_141/Conv2D/ReadVariableOpReadVariableOpAcnn_aug_deep_bnorm_skip_conv2d_141_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02:
8CNN_aug_deep_bnorm_skip/conv2d_141/Conv2D/ReadVariableOp?
)CNN_aug_deep_bnorm_skip/conv2d_141/Conv2DConv2DCCNN_aug_deep_bnorm_skip/batch_normalization_49/FusedBatchNormV3:y:0@CNN_aug_deep_bnorm_skip/conv2d_141/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????***
paddingSAME*
strides
2+
)CNN_aug_deep_bnorm_skip/conv2d_141/Conv2D?
9CNN_aug_deep_bnorm_skip/conv2d_141/BiasAdd/ReadVariableOpReadVariableOpBcnn_aug_deep_bnorm_skip_conv2d_141_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02;
9CNN_aug_deep_bnorm_skip/conv2d_141/BiasAdd/ReadVariableOp?
*CNN_aug_deep_bnorm_skip/conv2d_141/BiasAddBiasAdd2CNN_aug_deep_bnorm_skip/conv2d_141/Conv2D:output:0ACNN_aug_deep_bnorm_skip/conv2d_141/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**2,
*CNN_aug_deep_bnorm_skip/conv2d_141/BiasAdd?
'CNN_aug_deep_bnorm_skip/conv2d_141/ReluRelu3CNN_aug_deep_bnorm_skip/conv2d_141/BiasAdd:output:0*
T0*/
_output_shapes
:?????????**2)
'CNN_aug_deep_bnorm_skip/conv2d_141/Relu?
"CNN_aug_deep_bnorm_skip/add_27/addAddV25CNN_aug_deep_bnorm_skip/conv2d_141/Relu:activations:09CNN_aug_deep_bnorm_skip/max_pooling2d_55/MaxPool:output:0*
T0*/
_output_shapes
:?????????**2$
"CNN_aug_deep_bnorm_skip/add_27/add?
0CNN_aug_deep_bnorm_skip/max_pooling2d_56/MaxPoolMaxPool&CNN_aug_deep_bnorm_skip/add_27/add:z:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
22
0CNN_aug_deep_bnorm_skip/max_pooling2d_56/MaxPool?
8CNN_aug_deep_bnorm_skip/conv2d_142/Conv2D/ReadVariableOpReadVariableOpAcnn_aug_deep_bnorm_skip_conv2d_142_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02:
8CNN_aug_deep_bnorm_skip/conv2d_142/Conv2D/ReadVariableOp?
)CNN_aug_deep_bnorm_skip/conv2d_142/Conv2DConv2D9CNN_aug_deep_bnorm_skip/max_pooling2d_56/MaxPool:output:0@CNN_aug_deep_bnorm_skip/conv2d_142/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2+
)CNN_aug_deep_bnorm_skip/conv2d_142/Conv2D?
9CNN_aug_deep_bnorm_skip/conv2d_142/BiasAdd/ReadVariableOpReadVariableOpBcnn_aug_deep_bnorm_skip_conv2d_142_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02;
9CNN_aug_deep_bnorm_skip/conv2d_142/BiasAdd/ReadVariableOp?
*CNN_aug_deep_bnorm_skip/conv2d_142/BiasAddBiasAdd2CNN_aug_deep_bnorm_skip/conv2d_142/Conv2D:output:0ACNN_aug_deep_bnorm_skip/conv2d_142/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2,
*CNN_aug_deep_bnorm_skip/conv2d_142/BiasAdd?
'CNN_aug_deep_bnorm_skip/conv2d_142/ReluRelu3CNN_aug_deep_bnorm_skip/conv2d_142/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2)
'CNN_aug_deep_bnorm_skip/conv2d_142/Relu?
=CNN_aug_deep_bnorm_skip/batch_normalization_50/ReadVariableOpReadVariableOpFcnn_aug_deep_bnorm_skip_batch_normalization_50_readvariableop_resource*
_output_shapes
:*
dtype02?
=CNN_aug_deep_bnorm_skip/batch_normalization_50/ReadVariableOp?
?CNN_aug_deep_bnorm_skip/batch_normalization_50/ReadVariableOp_1ReadVariableOpHcnn_aug_deep_bnorm_skip_batch_normalization_50_readvariableop_1_resource*
_output_shapes
:*
dtype02A
?CNN_aug_deep_bnorm_skip/batch_normalization_50/ReadVariableOp_1?
NCNN_aug_deep_bnorm_skip/batch_normalization_50/FusedBatchNormV3/ReadVariableOpReadVariableOpWcnn_aug_deep_bnorm_skip_batch_normalization_50_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02P
NCNN_aug_deep_bnorm_skip/batch_normalization_50/FusedBatchNormV3/ReadVariableOp?
PCNN_aug_deep_bnorm_skip/batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpYcnn_aug_deep_bnorm_skip_batch_normalization_50_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02R
PCNN_aug_deep_bnorm_skip/batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1?
?CNN_aug_deep_bnorm_skip/batch_normalization_50/FusedBatchNormV3FusedBatchNormV35CNN_aug_deep_bnorm_skip/conv2d_142/Relu:activations:0ECNN_aug_deep_bnorm_skip/batch_normalization_50/ReadVariableOp:value:0GCNN_aug_deep_bnorm_skip/batch_normalization_50/ReadVariableOp_1:value:0VCNN_aug_deep_bnorm_skip/batch_normalization_50/FusedBatchNormV3/ReadVariableOp:value:0XCNN_aug_deep_bnorm_skip/batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( 2A
?CNN_aug_deep_bnorm_skip/batch_normalization_50/FusedBatchNormV3?
8CNN_aug_deep_bnorm_skip/conv2d_143/Conv2D/ReadVariableOpReadVariableOpAcnn_aug_deep_bnorm_skip_conv2d_143_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02:
8CNN_aug_deep_bnorm_skip/conv2d_143/Conv2D/ReadVariableOp?
)CNN_aug_deep_bnorm_skip/conv2d_143/Conv2DConv2DCCNN_aug_deep_bnorm_skip/batch_normalization_50/FusedBatchNormV3:y:0@CNN_aug_deep_bnorm_skip/conv2d_143/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2+
)CNN_aug_deep_bnorm_skip/conv2d_143/Conv2D?
9CNN_aug_deep_bnorm_skip/conv2d_143/BiasAdd/ReadVariableOpReadVariableOpBcnn_aug_deep_bnorm_skip_conv2d_143_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02;
9CNN_aug_deep_bnorm_skip/conv2d_143/BiasAdd/ReadVariableOp?
*CNN_aug_deep_bnorm_skip/conv2d_143/BiasAddBiasAdd2CNN_aug_deep_bnorm_skip/conv2d_143/Conv2D:output:0ACNN_aug_deep_bnorm_skip/conv2d_143/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2,
*CNN_aug_deep_bnorm_skip/conv2d_143/BiasAdd?
'CNN_aug_deep_bnorm_skip/conv2d_143/ReluRelu3CNN_aug_deep_bnorm_skip/conv2d_143/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2)
'CNN_aug_deep_bnorm_skip/conv2d_143/Relu?
=CNN_aug_deep_bnorm_skip/batch_normalization_51/ReadVariableOpReadVariableOpFcnn_aug_deep_bnorm_skip_batch_normalization_51_readvariableop_resource*
_output_shapes
:*
dtype02?
=CNN_aug_deep_bnorm_skip/batch_normalization_51/ReadVariableOp?
?CNN_aug_deep_bnorm_skip/batch_normalization_51/ReadVariableOp_1ReadVariableOpHcnn_aug_deep_bnorm_skip_batch_normalization_51_readvariableop_1_resource*
_output_shapes
:*
dtype02A
?CNN_aug_deep_bnorm_skip/batch_normalization_51/ReadVariableOp_1?
NCNN_aug_deep_bnorm_skip/batch_normalization_51/FusedBatchNormV3/ReadVariableOpReadVariableOpWcnn_aug_deep_bnorm_skip_batch_normalization_51_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02P
NCNN_aug_deep_bnorm_skip/batch_normalization_51/FusedBatchNormV3/ReadVariableOp?
PCNN_aug_deep_bnorm_skip/batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpYcnn_aug_deep_bnorm_skip_batch_normalization_51_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02R
PCNN_aug_deep_bnorm_skip/batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1?
?CNN_aug_deep_bnorm_skip/batch_normalization_51/FusedBatchNormV3FusedBatchNormV35CNN_aug_deep_bnorm_skip/conv2d_143/Relu:activations:0ECNN_aug_deep_bnorm_skip/batch_normalization_51/ReadVariableOp:value:0GCNN_aug_deep_bnorm_skip/batch_normalization_51/ReadVariableOp_1:value:0VCNN_aug_deep_bnorm_skip/batch_normalization_51/FusedBatchNormV3/ReadVariableOp:value:0XCNN_aug_deep_bnorm_skip/batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( 2A
?CNN_aug_deep_bnorm_skip/batch_normalization_51/FusedBatchNormV3?
8CNN_aug_deep_bnorm_skip/conv2d_144/Conv2D/ReadVariableOpReadVariableOpAcnn_aug_deep_bnorm_skip_conv2d_144_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02:
8CNN_aug_deep_bnorm_skip/conv2d_144/Conv2D/ReadVariableOp?
)CNN_aug_deep_bnorm_skip/conv2d_144/Conv2DConv2DCCNN_aug_deep_bnorm_skip/batch_normalization_51/FusedBatchNormV3:y:0@CNN_aug_deep_bnorm_skip/conv2d_144/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2+
)CNN_aug_deep_bnorm_skip/conv2d_144/Conv2D?
9CNN_aug_deep_bnorm_skip/conv2d_144/BiasAdd/ReadVariableOpReadVariableOpBcnn_aug_deep_bnorm_skip_conv2d_144_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02;
9CNN_aug_deep_bnorm_skip/conv2d_144/BiasAdd/ReadVariableOp?
*CNN_aug_deep_bnorm_skip/conv2d_144/BiasAddBiasAdd2CNN_aug_deep_bnorm_skip/conv2d_144/Conv2D:output:0ACNN_aug_deep_bnorm_skip/conv2d_144/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2,
*CNN_aug_deep_bnorm_skip/conv2d_144/BiasAdd?
'CNN_aug_deep_bnorm_skip/conv2d_144/ReluRelu3CNN_aug_deep_bnorm_skip/conv2d_144/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2)
'CNN_aug_deep_bnorm_skip/conv2d_144/Relu?
"CNN_aug_deep_bnorm_skip/add_28/addAddV25CNN_aug_deep_bnorm_skip/conv2d_144/Relu:activations:09CNN_aug_deep_bnorm_skip/max_pooling2d_56/MaxPool:output:0*
T0*/
_output_shapes
:?????????2$
"CNN_aug_deep_bnorm_skip/add_28/add?
0CNN_aug_deep_bnorm_skip/max_pooling2d_57/MaxPoolMaxPool&CNN_aug_deep_bnorm_skip/add_28/add:z:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
22
0CNN_aug_deep_bnorm_skip/max_pooling2d_57/MaxPool?
(CNN_aug_deep_bnorm_skip/flatten_29/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2*
(CNN_aug_deep_bnorm_skip/flatten_29/Const?
*CNN_aug_deep_bnorm_skip/flatten_29/ReshapeReshape9CNN_aug_deep_bnorm_skip/max_pooling2d_57/MaxPool:output:01CNN_aug_deep_bnorm_skip/flatten_29/Const:output:0*
T0*'
_output_shapes
:?????????2,
*CNN_aug_deep_bnorm_skip/flatten_29/Reshape?
6CNN_aug_deep_bnorm_skip/dense_58/MatMul/ReadVariableOpReadVariableOp?cnn_aug_deep_bnorm_skip_dense_58_matmul_readvariableop_resource*
_output_shapes

: *
dtype028
6CNN_aug_deep_bnorm_skip/dense_58/MatMul/ReadVariableOp?
'CNN_aug_deep_bnorm_skip/dense_58/MatMulMatMul3CNN_aug_deep_bnorm_skip/flatten_29/Reshape:output:0>CNN_aug_deep_bnorm_skip/dense_58/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2)
'CNN_aug_deep_bnorm_skip/dense_58/MatMul?
7CNN_aug_deep_bnorm_skip/dense_58/BiasAdd/ReadVariableOpReadVariableOp@cnn_aug_deep_bnorm_skip_dense_58_biasadd_readvariableop_resource*
_output_shapes
: *
dtype029
7CNN_aug_deep_bnorm_skip/dense_58/BiasAdd/ReadVariableOp?
(CNN_aug_deep_bnorm_skip/dense_58/BiasAddBiasAdd1CNN_aug_deep_bnorm_skip/dense_58/MatMul:product:0?CNN_aug_deep_bnorm_skip/dense_58/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2*
(CNN_aug_deep_bnorm_skip/dense_58/BiasAdd?
%CNN_aug_deep_bnorm_skip/dense_58/ReluRelu1CNN_aug_deep_bnorm_skip/dense_58/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2'
%CNN_aug_deep_bnorm_skip/dense_58/Relu?
6CNN_aug_deep_bnorm_skip/dense_59/MatMul/ReadVariableOpReadVariableOp?cnn_aug_deep_bnorm_skip_dense_59_matmul_readvariableop_resource*
_output_shapes

: *
dtype028
6CNN_aug_deep_bnorm_skip/dense_59/MatMul/ReadVariableOp?
'CNN_aug_deep_bnorm_skip/dense_59/MatMulMatMul3CNN_aug_deep_bnorm_skip/dense_58/Relu:activations:0>CNN_aug_deep_bnorm_skip/dense_59/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2)
'CNN_aug_deep_bnorm_skip/dense_59/MatMul?
7CNN_aug_deep_bnorm_skip/dense_59/BiasAdd/ReadVariableOpReadVariableOp@cnn_aug_deep_bnorm_skip_dense_59_biasadd_readvariableop_resource*
_output_shapes
:*
dtype029
7CNN_aug_deep_bnorm_skip/dense_59/BiasAdd/ReadVariableOp?
(CNN_aug_deep_bnorm_skip/dense_59/BiasAddBiasAdd1CNN_aug_deep_bnorm_skip/dense_59/MatMul:product:0?CNN_aug_deep_bnorm_skip/dense_59/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2*
(CNN_aug_deep_bnorm_skip/dense_59/BiasAdd?
(CNN_aug_deep_bnorm_skip/dense_59/SoftmaxSoftmax1CNN_aug_deep_bnorm_skip/dense_59/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2*
(CNN_aug_deep_bnorm_skip/dense_59/Softmax?
IdentityIdentity2CNN_aug_deep_bnorm_skip/dense_59/Softmax:softmax:0O^CNN_aug_deep_bnorm_skip/batch_normalization_46/FusedBatchNormV3/ReadVariableOpQ^CNN_aug_deep_bnorm_skip/batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1>^CNN_aug_deep_bnorm_skip/batch_normalization_46/ReadVariableOp@^CNN_aug_deep_bnorm_skip/batch_normalization_46/ReadVariableOp_1O^CNN_aug_deep_bnorm_skip/batch_normalization_47/FusedBatchNormV3/ReadVariableOpQ^CNN_aug_deep_bnorm_skip/batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1>^CNN_aug_deep_bnorm_skip/batch_normalization_47/ReadVariableOp@^CNN_aug_deep_bnorm_skip/batch_normalization_47/ReadVariableOp_1O^CNN_aug_deep_bnorm_skip/batch_normalization_48/FusedBatchNormV3/ReadVariableOpQ^CNN_aug_deep_bnorm_skip/batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1>^CNN_aug_deep_bnorm_skip/batch_normalization_48/ReadVariableOp@^CNN_aug_deep_bnorm_skip/batch_normalization_48/ReadVariableOp_1O^CNN_aug_deep_bnorm_skip/batch_normalization_49/FusedBatchNormV3/ReadVariableOpQ^CNN_aug_deep_bnorm_skip/batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1>^CNN_aug_deep_bnorm_skip/batch_normalization_49/ReadVariableOp@^CNN_aug_deep_bnorm_skip/batch_normalization_49/ReadVariableOp_1O^CNN_aug_deep_bnorm_skip/batch_normalization_50/FusedBatchNormV3/ReadVariableOpQ^CNN_aug_deep_bnorm_skip/batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1>^CNN_aug_deep_bnorm_skip/batch_normalization_50/ReadVariableOp@^CNN_aug_deep_bnorm_skip/batch_normalization_50/ReadVariableOp_1O^CNN_aug_deep_bnorm_skip/batch_normalization_51/FusedBatchNormV3/ReadVariableOpQ^CNN_aug_deep_bnorm_skip/batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1>^CNN_aug_deep_bnorm_skip/batch_normalization_51/ReadVariableOp@^CNN_aug_deep_bnorm_skip/batch_normalization_51/ReadVariableOp_1:^CNN_aug_deep_bnorm_skip/conv2d_136/BiasAdd/ReadVariableOp9^CNN_aug_deep_bnorm_skip/conv2d_136/Conv2D/ReadVariableOp:^CNN_aug_deep_bnorm_skip/conv2d_137/BiasAdd/ReadVariableOp9^CNN_aug_deep_bnorm_skip/conv2d_137/Conv2D/ReadVariableOp:^CNN_aug_deep_bnorm_skip/conv2d_138/BiasAdd/ReadVariableOp9^CNN_aug_deep_bnorm_skip/conv2d_138/Conv2D/ReadVariableOp:^CNN_aug_deep_bnorm_skip/conv2d_139/BiasAdd/ReadVariableOp9^CNN_aug_deep_bnorm_skip/conv2d_139/Conv2D/ReadVariableOp:^CNN_aug_deep_bnorm_skip/conv2d_140/BiasAdd/ReadVariableOp9^CNN_aug_deep_bnorm_skip/conv2d_140/Conv2D/ReadVariableOp:^CNN_aug_deep_bnorm_skip/conv2d_141/BiasAdd/ReadVariableOp9^CNN_aug_deep_bnorm_skip/conv2d_141/Conv2D/ReadVariableOp:^CNN_aug_deep_bnorm_skip/conv2d_142/BiasAdd/ReadVariableOp9^CNN_aug_deep_bnorm_skip/conv2d_142/Conv2D/ReadVariableOp:^CNN_aug_deep_bnorm_skip/conv2d_143/BiasAdd/ReadVariableOp9^CNN_aug_deep_bnorm_skip/conv2d_143/Conv2D/ReadVariableOp:^CNN_aug_deep_bnorm_skip/conv2d_144/BiasAdd/ReadVariableOp9^CNN_aug_deep_bnorm_skip/conv2d_144/Conv2D/ReadVariableOp8^CNN_aug_deep_bnorm_skip/dense_58/BiasAdd/ReadVariableOp7^CNN_aug_deep_bnorm_skip/dense_58/MatMul/ReadVariableOp8^CNN_aug_deep_bnorm_skip/dense_59/BiasAdd/ReadVariableOp7^CNN_aug_deep_bnorm_skip/dense_59/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::::::::2?
NCNN_aug_deep_bnorm_skip/batch_normalization_46/FusedBatchNormV3/ReadVariableOpNCNN_aug_deep_bnorm_skip/batch_normalization_46/FusedBatchNormV3/ReadVariableOp2?
PCNN_aug_deep_bnorm_skip/batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1PCNN_aug_deep_bnorm_skip/batch_normalization_46/FusedBatchNormV3/ReadVariableOp_12~
=CNN_aug_deep_bnorm_skip/batch_normalization_46/ReadVariableOp=CNN_aug_deep_bnorm_skip/batch_normalization_46/ReadVariableOp2?
?CNN_aug_deep_bnorm_skip/batch_normalization_46/ReadVariableOp_1?CNN_aug_deep_bnorm_skip/batch_normalization_46/ReadVariableOp_12?
NCNN_aug_deep_bnorm_skip/batch_normalization_47/FusedBatchNormV3/ReadVariableOpNCNN_aug_deep_bnorm_skip/batch_normalization_47/FusedBatchNormV3/ReadVariableOp2?
PCNN_aug_deep_bnorm_skip/batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1PCNN_aug_deep_bnorm_skip/batch_normalization_47/FusedBatchNormV3/ReadVariableOp_12~
=CNN_aug_deep_bnorm_skip/batch_normalization_47/ReadVariableOp=CNN_aug_deep_bnorm_skip/batch_normalization_47/ReadVariableOp2?
?CNN_aug_deep_bnorm_skip/batch_normalization_47/ReadVariableOp_1?CNN_aug_deep_bnorm_skip/batch_normalization_47/ReadVariableOp_12?
NCNN_aug_deep_bnorm_skip/batch_normalization_48/FusedBatchNormV3/ReadVariableOpNCNN_aug_deep_bnorm_skip/batch_normalization_48/FusedBatchNormV3/ReadVariableOp2?
PCNN_aug_deep_bnorm_skip/batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1PCNN_aug_deep_bnorm_skip/batch_normalization_48/FusedBatchNormV3/ReadVariableOp_12~
=CNN_aug_deep_bnorm_skip/batch_normalization_48/ReadVariableOp=CNN_aug_deep_bnorm_skip/batch_normalization_48/ReadVariableOp2?
?CNN_aug_deep_bnorm_skip/batch_normalization_48/ReadVariableOp_1?CNN_aug_deep_bnorm_skip/batch_normalization_48/ReadVariableOp_12?
NCNN_aug_deep_bnorm_skip/batch_normalization_49/FusedBatchNormV3/ReadVariableOpNCNN_aug_deep_bnorm_skip/batch_normalization_49/FusedBatchNormV3/ReadVariableOp2?
PCNN_aug_deep_bnorm_skip/batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1PCNN_aug_deep_bnorm_skip/batch_normalization_49/FusedBatchNormV3/ReadVariableOp_12~
=CNN_aug_deep_bnorm_skip/batch_normalization_49/ReadVariableOp=CNN_aug_deep_bnorm_skip/batch_normalization_49/ReadVariableOp2?
?CNN_aug_deep_bnorm_skip/batch_normalization_49/ReadVariableOp_1?CNN_aug_deep_bnorm_skip/batch_normalization_49/ReadVariableOp_12?
NCNN_aug_deep_bnorm_skip/batch_normalization_50/FusedBatchNormV3/ReadVariableOpNCNN_aug_deep_bnorm_skip/batch_normalization_50/FusedBatchNormV3/ReadVariableOp2?
PCNN_aug_deep_bnorm_skip/batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1PCNN_aug_deep_bnorm_skip/batch_normalization_50/FusedBatchNormV3/ReadVariableOp_12~
=CNN_aug_deep_bnorm_skip/batch_normalization_50/ReadVariableOp=CNN_aug_deep_bnorm_skip/batch_normalization_50/ReadVariableOp2?
?CNN_aug_deep_bnorm_skip/batch_normalization_50/ReadVariableOp_1?CNN_aug_deep_bnorm_skip/batch_normalization_50/ReadVariableOp_12?
NCNN_aug_deep_bnorm_skip/batch_normalization_51/FusedBatchNormV3/ReadVariableOpNCNN_aug_deep_bnorm_skip/batch_normalization_51/FusedBatchNormV3/ReadVariableOp2?
PCNN_aug_deep_bnorm_skip/batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1PCNN_aug_deep_bnorm_skip/batch_normalization_51/FusedBatchNormV3/ReadVariableOp_12~
=CNN_aug_deep_bnorm_skip/batch_normalization_51/ReadVariableOp=CNN_aug_deep_bnorm_skip/batch_normalization_51/ReadVariableOp2?
?CNN_aug_deep_bnorm_skip/batch_normalization_51/ReadVariableOp_1?CNN_aug_deep_bnorm_skip/batch_normalization_51/ReadVariableOp_12v
9CNN_aug_deep_bnorm_skip/conv2d_136/BiasAdd/ReadVariableOp9CNN_aug_deep_bnorm_skip/conv2d_136/BiasAdd/ReadVariableOp2t
8CNN_aug_deep_bnorm_skip/conv2d_136/Conv2D/ReadVariableOp8CNN_aug_deep_bnorm_skip/conv2d_136/Conv2D/ReadVariableOp2v
9CNN_aug_deep_bnorm_skip/conv2d_137/BiasAdd/ReadVariableOp9CNN_aug_deep_bnorm_skip/conv2d_137/BiasAdd/ReadVariableOp2t
8CNN_aug_deep_bnorm_skip/conv2d_137/Conv2D/ReadVariableOp8CNN_aug_deep_bnorm_skip/conv2d_137/Conv2D/ReadVariableOp2v
9CNN_aug_deep_bnorm_skip/conv2d_138/BiasAdd/ReadVariableOp9CNN_aug_deep_bnorm_skip/conv2d_138/BiasAdd/ReadVariableOp2t
8CNN_aug_deep_bnorm_skip/conv2d_138/Conv2D/ReadVariableOp8CNN_aug_deep_bnorm_skip/conv2d_138/Conv2D/ReadVariableOp2v
9CNN_aug_deep_bnorm_skip/conv2d_139/BiasAdd/ReadVariableOp9CNN_aug_deep_bnorm_skip/conv2d_139/BiasAdd/ReadVariableOp2t
8CNN_aug_deep_bnorm_skip/conv2d_139/Conv2D/ReadVariableOp8CNN_aug_deep_bnorm_skip/conv2d_139/Conv2D/ReadVariableOp2v
9CNN_aug_deep_bnorm_skip/conv2d_140/BiasAdd/ReadVariableOp9CNN_aug_deep_bnorm_skip/conv2d_140/BiasAdd/ReadVariableOp2t
8CNN_aug_deep_bnorm_skip/conv2d_140/Conv2D/ReadVariableOp8CNN_aug_deep_bnorm_skip/conv2d_140/Conv2D/ReadVariableOp2v
9CNN_aug_deep_bnorm_skip/conv2d_141/BiasAdd/ReadVariableOp9CNN_aug_deep_bnorm_skip/conv2d_141/BiasAdd/ReadVariableOp2t
8CNN_aug_deep_bnorm_skip/conv2d_141/Conv2D/ReadVariableOp8CNN_aug_deep_bnorm_skip/conv2d_141/Conv2D/ReadVariableOp2v
9CNN_aug_deep_bnorm_skip/conv2d_142/BiasAdd/ReadVariableOp9CNN_aug_deep_bnorm_skip/conv2d_142/BiasAdd/ReadVariableOp2t
8CNN_aug_deep_bnorm_skip/conv2d_142/Conv2D/ReadVariableOp8CNN_aug_deep_bnorm_skip/conv2d_142/Conv2D/ReadVariableOp2v
9CNN_aug_deep_bnorm_skip/conv2d_143/BiasAdd/ReadVariableOp9CNN_aug_deep_bnorm_skip/conv2d_143/BiasAdd/ReadVariableOp2t
8CNN_aug_deep_bnorm_skip/conv2d_143/Conv2D/ReadVariableOp8CNN_aug_deep_bnorm_skip/conv2d_143/Conv2D/ReadVariableOp2v
9CNN_aug_deep_bnorm_skip/conv2d_144/BiasAdd/ReadVariableOp9CNN_aug_deep_bnorm_skip/conv2d_144/BiasAdd/ReadVariableOp2t
8CNN_aug_deep_bnorm_skip/conv2d_144/Conv2D/ReadVariableOp8CNN_aug_deep_bnorm_skip/conv2d_144/Conv2D/ReadVariableOp2r
7CNN_aug_deep_bnorm_skip/dense_58/BiasAdd/ReadVariableOp7CNN_aug_deep_bnorm_skip/dense_58/BiasAdd/ReadVariableOp2p
6CNN_aug_deep_bnorm_skip/dense_58/MatMul/ReadVariableOp6CNN_aug_deep_bnorm_skip/dense_58/MatMul/ReadVariableOp2r
7CNN_aug_deep_bnorm_skip/dense_59/BiasAdd/ReadVariableOp7CNN_aug_deep_bnorm_skip/dense_59/BiasAdd/ReadVariableOp2p
6CNN_aug_deep_bnorm_skip/dense_59/MatMul/ReadVariableOp6CNN_aug_deep_bnorm_skip/dense_59/MatMul/ReadVariableOp:[ W
1
_output_shapes
:???????????
"
_user_specified_name
input_30
?
?
8__inference_batch_normalization_46_layer_call_fn_2460177

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_46_layer_call_and_return_conditional_losses_24581452
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_48_layer_call_and_return_conditional_losses_2457730

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_49_layer_call_fn_2460717

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_49_layer_call_and_return_conditional_losses_24578342
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?

?
G__inference_conv2d_141_layer_call_and_return_conditional_losses_2460728

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????***
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????**2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????**2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????**::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????**
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_48_layer_call_and_return_conditional_losses_2460479

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_46_layer_call_fn_2460164

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_46_layer_call_and_return_conditional_losses_24581272
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
G
input_30;
serving_default_input_30:0???????????<
dense_590
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
??
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer-6
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
layer_with_weights-8
layer-11
layer_with_weights-9
layer-12
layer-13
layer-14
layer_with_weights-10
layer-15
layer_with_weights-11
layer-16
layer_with_weights-12
layer-17
layer_with_weights-13
layer-18
layer_with_weights-14
layer-19
layer-20
layer-21
layer-22
layer_with_weights-15
layer-23
layer_with_weights-16
layer-24
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
?_default_save_signature
+?&call_and_return_all_conditional_losses
?__call__"??
_tf_keras_network??{"class_name": "Functional", "name": "CNN_aug_deep_bnorm_skip", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "CNN_aug_deep_bnorm_skip", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_30"}, "name": "input_30", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_136", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_136", "inbound_nodes": [[["input_30", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_46", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_46", "inbound_nodes": [[["conv2d_136", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_137", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_137", "inbound_nodes": [[["batch_normalization_46", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_47", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_47", "inbound_nodes": [[["conv2d_137", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_138", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_138", "inbound_nodes": [[["batch_normalization_47", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_26", "trainable": true, "dtype": "float32"}, "name": "add_26", "inbound_nodes": [[["conv2d_138", 0, 0, {}], ["input_30", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_55", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "name": "max_pooling2d_55", "inbound_nodes": [[["add_26", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_139", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_139", "inbound_nodes": [[["max_pooling2d_55", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_48", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_48", "inbound_nodes": [[["conv2d_139", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_140", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_140", "inbound_nodes": [[["batch_normalization_48", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_49", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_49", "inbound_nodes": [[["conv2d_140", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_141", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_141", "inbound_nodes": [[["batch_normalization_49", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_27", "trainable": true, "dtype": "float32"}, "name": "add_27", "inbound_nodes": [[["conv2d_141", 0, 0, {}], ["max_pooling2d_55", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_56", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "name": "max_pooling2d_56", "inbound_nodes": [[["add_27", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_142", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_142", "inbound_nodes": [[["max_pooling2d_56", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_50", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_50", "inbound_nodes": [[["conv2d_142", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_143", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_143", "inbound_nodes": [[["batch_normalization_50", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_51", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_51", "inbound_nodes": [[["conv2d_143", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_144", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_144", "inbound_nodes": [[["batch_normalization_51", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_28", "trainable": true, "dtype": "float32"}, "name": "add_28", "inbound_nodes": [[["conv2d_144", 0, 0, {}], ["max_pooling2d_56", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_57", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "name": "max_pooling2d_57", "inbound_nodes": [[["add_28", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_29", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_29", "inbound_nodes": [[["max_pooling2d_57", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_58", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_58", "inbound_nodes": [[["flatten_29", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_59", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_59", "inbound_nodes": [[["dense_58", 0, 0, {}]]]}], "input_layers": [["input_30", 0, 0]], "output_layers": [["dense_59", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "CNN_aug_deep_bnorm_skip", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_30"}, "name": "input_30", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_136", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_136", "inbound_nodes": [[["input_30", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_46", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_46", "inbound_nodes": [[["conv2d_136", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_137", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_137", "inbound_nodes": [[["batch_normalization_46", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_47", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_47", "inbound_nodes": [[["conv2d_137", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_138", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_138", "inbound_nodes": [[["batch_normalization_47", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_26", "trainable": true, "dtype": "float32"}, "name": "add_26", "inbound_nodes": [[["conv2d_138", 0, 0, {}], ["input_30", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_55", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "name": "max_pooling2d_55", "inbound_nodes": [[["add_26", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_139", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_139", "inbound_nodes": [[["max_pooling2d_55", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_48", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_48", "inbound_nodes": [[["conv2d_139", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_140", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_140", "inbound_nodes": [[["batch_normalization_48", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_49", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_49", "inbound_nodes": [[["conv2d_140", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_141", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_141", "inbound_nodes": [[["batch_normalization_49", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_27", "trainable": true, "dtype": "float32"}, "name": "add_27", "inbound_nodes": [[["conv2d_141", 0, 0, {}], ["max_pooling2d_55", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_56", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "name": "max_pooling2d_56", "inbound_nodes": [[["add_27", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_142", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_142", "inbound_nodes": [[["max_pooling2d_56", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_50", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_50", "inbound_nodes": [[["conv2d_142", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_143", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_143", "inbound_nodes": [[["batch_normalization_50", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_51", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_51", "inbound_nodes": [[["conv2d_143", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_144", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_144", "inbound_nodes": [[["batch_normalization_51", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_28", "trainable": true, "dtype": "float32"}, "name": "add_28", "inbound_nodes": [[["conv2d_144", 0, 0, {}], ["max_pooling2d_56", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_57", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "name": "max_pooling2d_57", "inbound_nodes": [[["add_28", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_29", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_29", "inbound_nodes": [[["max_pooling2d_57", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_58", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_58", "inbound_nodes": [[["flatten_29", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_59", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_59", "inbound_nodes": [[["dense_58", 0, 0, {}]]]}], "input_layers": [["input_30", 0, 0]], "output_layers": [["dense_59", 0, 0]]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_30", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_30"}}
?	

 kernel
!bias
"regularization_losses
#trainable_variables
$	variables
%	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_136", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_136", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 1]}}
?	
&axis
	'gamma
(beta
)moving_mean
*moving_variance
+regularization_losses
,trainable_variables
-	variables
.	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_46", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_46", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 8]}}
?	

/kernel
0bias
1regularization_losses
2trainable_variables
3	variables
4	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_137", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_137", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 8]}}
?	
5axis
	6gamma
7beta
8moving_mean
9moving_variance
:regularization_losses
;trainable_variables
<	variables
=	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_47", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_47", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 8]}}
?	

>kernel
?bias
@regularization_losses
Atrainable_variables
B	variables
C	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_138", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_138", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 8]}}
?
Dregularization_losses
Etrainable_variables
F	variables
G	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Add", "name": "add_26", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "add_26", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 128, 128, 1]}, {"class_name": "TensorShape", "items": [null, 128, 128, 1]}]}
?
Hregularization_losses
Itrainable_variables
J	variables
K	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_55", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_55", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

Lkernel
Mbias
Nregularization_losses
Otrainable_variables
P	variables
Q	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_139", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_139", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 42, 42, 1]}}
?	
Raxis
	Sgamma
Tbeta
Umoving_mean
Vmoving_variance
Wregularization_losses
Xtrainable_variables
Y	variables
Z	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_48", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_48", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 42, 42, 8]}}
?	

[kernel
\bias
]regularization_losses
^trainable_variables
_	variables
`	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_140", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_140", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 42, 42, 8]}}
?	
aaxis
	bgamma
cbeta
dmoving_mean
emoving_variance
fregularization_losses
gtrainable_variables
h	variables
i	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_49", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_49", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 42, 42, 8]}}
?	

jkernel
kbias
lregularization_losses
mtrainable_variables
n	variables
o	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_141", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_141", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 42, 42, 8]}}
?
pregularization_losses
qtrainable_variables
r	variables
s	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Add", "name": "add_27", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "add_27", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 42, 42, 1]}, {"class_name": "TensorShape", "items": [null, 42, 42, 1]}]}
?
tregularization_losses
utrainable_variables
v	variables
w	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_56", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_56", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

xkernel
ybias
zregularization_losses
{trainable_variables
|	variables
}	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_142", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_142", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 1]}}
?	
~axis
	gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_50", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_50", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 8]}}
?	
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_143", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_143", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 8]}}
?	
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_51", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_51", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 8]}}
?	
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_144", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_144", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 8]}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Add", "name": "add_28", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "add_28", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 14, 14, 1]}, {"class_name": "TensorShape", "items": [null, 14, 14, 1]}]}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_57", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_57", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_29", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_29", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_58", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_58", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
?
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_59", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_59", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
?
	?iter
?beta_1
?beta_2

?decay
?learning_rate m?!m?'m?(m?/m?0m?6m?7m?>m??m?Lm?Mm?Sm?Tm?[m?\m?bm?cm?jm?km?xm?ym?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m? v?!v?'v?(v?/v?0v?6v?7v?>v??v?Lv?Mv?Sv?Tv?[v?\v?bv?cv?jv?kv?xv?yv?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?"
	optimizer
 "
trackable_list_wrapper
?
 0
!1
'2
(3
/4
05
66
77
>8
?9
L10
M11
S12
T13
[14
\15
b16
c17
j18
k19
x20
y21
22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33"
trackable_list_wrapper
?
 0
!1
'2
(3
)4
*5
/6
07
68
79
810
911
>12
?13
L14
M15
S16
T17
U18
V19
[20
\21
b22
c23
d24
e25
j26
k27
x28
y29
30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43
?44
?45"
trackable_list_wrapper
?
regularization_losses
?layers
?layer_metrics
?metrics
trainable_variables
	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
+:)2conv2d_136/kernel
:2conv2d_136/bias
 "
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
?
"regularization_losses
?layers
?layer_metrics
?metrics
#trainable_variables
$	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_46/gamma
):'2batch_normalization_46/beta
2:0 (2"batch_normalization_46/moving_mean
6:4 (2&batch_normalization_46/moving_variance
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
<
'0
(1
)2
*3"
trackable_list_wrapper
?
+regularization_losses
?layers
?layer_metrics
?metrics
,trainable_variables
-	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)2conv2d_137/kernel
:2conv2d_137/bias
 "
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
?
1regularization_losses
?layers
?layer_metrics
?metrics
2trainable_variables
3	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_47/gamma
):'2batch_normalization_47/beta
2:0 (2"batch_normalization_47/moving_mean
6:4 (2&batch_normalization_47/moving_variance
 "
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
<
60
71
82
93"
trackable_list_wrapper
?
:regularization_losses
?layers
?layer_metrics
?metrics
;trainable_variables
<	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)2conv2d_138/kernel
:2conv2d_138/bias
 "
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
?
@regularization_losses
?layers
?layer_metrics
?metrics
Atrainable_variables
B	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Dregularization_losses
?layers
?layer_metrics
?metrics
Etrainable_variables
F	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Hregularization_losses
?layers
?layer_metrics
?metrics
Itrainable_variables
J	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)2conv2d_139/kernel
:2conv2d_139/bias
 "
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
?
Nregularization_losses
?layers
?layer_metrics
?metrics
Otrainable_variables
P	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_48/gamma
):'2batch_normalization_48/beta
2:0 (2"batch_normalization_48/moving_mean
6:4 (2&batch_normalization_48/moving_variance
 "
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
<
S0
T1
U2
V3"
trackable_list_wrapper
?
Wregularization_losses
?layers
?layer_metrics
?metrics
Xtrainable_variables
Y	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)2conv2d_140/kernel
:2conv2d_140/bias
 "
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
?
]regularization_losses
?layers
?layer_metrics
?metrics
^trainable_variables
_	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_49/gamma
):'2batch_normalization_49/beta
2:0 (2"batch_normalization_49/moving_mean
6:4 (2&batch_normalization_49/moving_variance
 "
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
<
b0
c1
d2
e3"
trackable_list_wrapper
?
fregularization_losses
?layers
?layer_metrics
?metrics
gtrainable_variables
h	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)2conv2d_141/kernel
:2conv2d_141/bias
 "
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
?
lregularization_losses
?layers
?layer_metrics
?metrics
mtrainable_variables
n	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
pregularization_losses
?layers
?layer_metrics
?metrics
qtrainable_variables
r	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
tregularization_losses
?layers
?layer_metrics
?metrics
utrainable_variables
v	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)2conv2d_142/kernel
:2conv2d_142/bias
 "
trackable_list_wrapper
.
x0
y1"
trackable_list_wrapper
.
x0
y1"
trackable_list_wrapper
?
zregularization_losses
?layers
?layer_metrics
?metrics
{trainable_variables
|	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_50/gamma
):'2batch_normalization_50/beta
2:0 (2"batch_normalization_50/moving_mean
6:4 (2&batch_normalization_50/moving_variance
 "
trackable_list_wrapper
/
0
?1"
trackable_list_wrapper
?
0
?1
?2
?3"
trackable_list_wrapper
?
?regularization_losses
?layers
?layer_metrics
?metrics
?trainable_variables
?	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)2conv2d_143/kernel
:2conv2d_143/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?regularization_losses
?layers
?layer_metrics
?metrics
?trainable_variables
?	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_51/gamma
):'2batch_normalization_51/beta
2:0 (2"batch_normalization_51/moving_mean
6:4 (2&batch_normalization_51/moving_variance
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
?
?regularization_losses
?layers
?layer_metrics
?metrics
?trainable_variables
?	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)2conv2d_144/kernel
:2conv2d_144/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?regularization_losses
?layers
?layer_metrics
?metrics
?trainable_variables
?	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?layers
?layer_metrics
?metrics
?trainable_variables
?	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?layers
?layer_metrics
?metrics
?trainable_variables
?	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?layers
?layer_metrics
?metrics
?trainable_variables
?	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!: 2dense_58/kernel
: 2dense_58/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?regularization_losses
?layers
?layer_metrics
?metrics
?trainable_variables
?	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!: 2dense_59/kernel
:2dense_59/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?regularization_losses
?layers
?layer_metrics
?metrics
?trainable_variables
?	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24"
trackable_list_wrapper
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
z
)0
*1
82
93
U4
V5
d6
e7
?8
?9
?10
?11"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
0:.2Adam/conv2d_136/kernel/m
": 2Adam/conv2d_136/bias/m
/:-2#Adam/batch_normalization_46/gamma/m
.:,2"Adam/batch_normalization_46/beta/m
0:.2Adam/conv2d_137/kernel/m
": 2Adam/conv2d_137/bias/m
/:-2#Adam/batch_normalization_47/gamma/m
.:,2"Adam/batch_normalization_47/beta/m
0:.2Adam/conv2d_138/kernel/m
": 2Adam/conv2d_138/bias/m
0:.2Adam/conv2d_139/kernel/m
": 2Adam/conv2d_139/bias/m
/:-2#Adam/batch_normalization_48/gamma/m
.:,2"Adam/batch_normalization_48/beta/m
0:.2Adam/conv2d_140/kernel/m
": 2Adam/conv2d_140/bias/m
/:-2#Adam/batch_normalization_49/gamma/m
.:,2"Adam/batch_normalization_49/beta/m
0:.2Adam/conv2d_141/kernel/m
": 2Adam/conv2d_141/bias/m
0:.2Adam/conv2d_142/kernel/m
": 2Adam/conv2d_142/bias/m
/:-2#Adam/batch_normalization_50/gamma/m
.:,2"Adam/batch_normalization_50/beta/m
0:.2Adam/conv2d_143/kernel/m
": 2Adam/conv2d_143/bias/m
/:-2#Adam/batch_normalization_51/gamma/m
.:,2"Adam/batch_normalization_51/beta/m
0:.2Adam/conv2d_144/kernel/m
": 2Adam/conv2d_144/bias/m
&:$ 2Adam/dense_58/kernel/m
 : 2Adam/dense_58/bias/m
&:$ 2Adam/dense_59/kernel/m
 :2Adam/dense_59/bias/m
0:.2Adam/conv2d_136/kernel/v
": 2Adam/conv2d_136/bias/v
/:-2#Adam/batch_normalization_46/gamma/v
.:,2"Adam/batch_normalization_46/beta/v
0:.2Adam/conv2d_137/kernel/v
": 2Adam/conv2d_137/bias/v
/:-2#Adam/batch_normalization_47/gamma/v
.:,2"Adam/batch_normalization_47/beta/v
0:.2Adam/conv2d_138/kernel/v
": 2Adam/conv2d_138/bias/v
0:.2Adam/conv2d_139/kernel/v
": 2Adam/conv2d_139/bias/v
/:-2#Adam/batch_normalization_48/gamma/v
.:,2"Adam/batch_normalization_48/beta/v
0:.2Adam/conv2d_140/kernel/v
": 2Adam/conv2d_140/bias/v
/:-2#Adam/batch_normalization_49/gamma/v
.:,2"Adam/batch_normalization_49/beta/v
0:.2Adam/conv2d_141/kernel/v
": 2Adam/conv2d_141/bias/v
0:.2Adam/conv2d_142/kernel/v
": 2Adam/conv2d_142/bias/v
/:-2#Adam/batch_normalization_50/gamma/v
.:,2"Adam/batch_normalization_50/beta/v
0:.2Adam/conv2d_143/kernel/v
": 2Adam/conv2d_143/bias/v
/:-2#Adam/batch_normalization_51/gamma/v
.:,2"Adam/batch_normalization_51/beta/v
0:.2Adam/conv2d_144/kernel/v
": 2Adam/conv2d_144/bias/v
&:$ 2Adam/dense_58/kernel/v
 : 2Adam/dense_58/bias/v
&:$ 2Adam/dense_59/kernel/v
 :2Adam/dense_59/bias/v
?2?
"__inference__wrapped_model_2457417?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *1?.
,?)
input_30???????????
?2?
T__inference_CNN_aug_deep_bnorm_skip_layer_call_and_return_conditional_losses_2458879
T__inference_CNN_aug_deep_bnorm_skip_layer_call_and_return_conditional_losses_2459899
T__inference_CNN_aug_deep_bnorm_skip_layer_call_and_return_conditional_losses_2459726
T__inference_CNN_aug_deep_bnorm_skip_layer_call_and_return_conditional_losses_2458999?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
9__inference_CNN_aug_deep_bnorm_skip_layer_call_fn_2459217
9__inference_CNN_aug_deep_bnorm_skip_layer_call_fn_2460093
9__inference_CNN_aug_deep_bnorm_skip_layer_call_fn_2459434
9__inference_CNN_aug_deep_bnorm_skip_layer_call_fn_2459996?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_conv2d_136_layer_call_and_return_conditional_losses_2460104?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_conv2d_136_layer_call_fn_2460113?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
S__inference_batch_normalization_46_layer_call_and_return_conditional_losses_2460151
S__inference_batch_normalization_46_layer_call_and_return_conditional_losses_2460197
S__inference_batch_normalization_46_layer_call_and_return_conditional_losses_2460133
S__inference_batch_normalization_46_layer_call_and_return_conditional_losses_2460215?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
8__inference_batch_normalization_46_layer_call_fn_2460164
8__inference_batch_normalization_46_layer_call_fn_2460228
8__inference_batch_normalization_46_layer_call_fn_2460241
8__inference_batch_normalization_46_layer_call_fn_2460177?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_conv2d_137_layer_call_and_return_conditional_losses_2460252?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_conv2d_137_layer_call_fn_2460261?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
S__inference_batch_normalization_47_layer_call_and_return_conditional_losses_2460281
S__inference_batch_normalization_47_layer_call_and_return_conditional_losses_2460363
S__inference_batch_normalization_47_layer_call_and_return_conditional_losses_2460299
S__inference_batch_normalization_47_layer_call_and_return_conditional_losses_2460345?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
8__inference_batch_normalization_47_layer_call_fn_2460312
8__inference_batch_normalization_47_layer_call_fn_2460389
8__inference_batch_normalization_47_layer_call_fn_2460376
8__inference_batch_normalization_47_layer_call_fn_2460325?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_conv2d_138_layer_call_and_return_conditional_losses_2460400?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_conv2d_138_layer_call_fn_2460409?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_add_26_layer_call_and_return_conditional_losses_2460415?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_add_26_layer_call_fn_2460421?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
M__inference_max_pooling2d_55_layer_call_and_return_conditional_losses_2457631?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
2__inference_max_pooling2d_55_layer_call_fn_2457637?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
G__inference_conv2d_139_layer_call_and_return_conditional_losses_2460432?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_conv2d_139_layer_call_fn_2460441?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
S__inference_batch_normalization_48_layer_call_and_return_conditional_losses_2460479
S__inference_batch_normalization_48_layer_call_and_return_conditional_losses_2460525
S__inference_batch_normalization_48_layer_call_and_return_conditional_losses_2460461
S__inference_batch_normalization_48_layer_call_and_return_conditional_losses_2460543?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
8__inference_batch_normalization_48_layer_call_fn_2460492
8__inference_batch_normalization_48_layer_call_fn_2460569
8__inference_batch_normalization_48_layer_call_fn_2460556
8__inference_batch_normalization_48_layer_call_fn_2460505?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_conv2d_140_layer_call_and_return_conditional_losses_2460580?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_conv2d_140_layer_call_fn_2460589?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
S__inference_batch_normalization_49_layer_call_and_return_conditional_losses_2460609
S__inference_batch_normalization_49_layer_call_and_return_conditional_losses_2460691
S__inference_batch_normalization_49_layer_call_and_return_conditional_losses_2460673
S__inference_batch_normalization_49_layer_call_and_return_conditional_losses_2460627?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
8__inference_batch_normalization_49_layer_call_fn_2460653
8__inference_batch_normalization_49_layer_call_fn_2460704
8__inference_batch_normalization_49_layer_call_fn_2460717
8__inference_batch_normalization_49_layer_call_fn_2460640?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_conv2d_141_layer_call_and_return_conditional_losses_2460728?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_conv2d_141_layer_call_fn_2460737?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_add_27_layer_call_and_return_conditional_losses_2460743?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_add_27_layer_call_fn_2460749?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
M__inference_max_pooling2d_56_layer_call_and_return_conditional_losses_2457851?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
2__inference_max_pooling2d_56_layer_call_fn_2457857?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
G__inference_conv2d_142_layer_call_and_return_conditional_losses_2460760?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_conv2d_142_layer_call_fn_2460769?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
S__inference_batch_normalization_50_layer_call_and_return_conditional_losses_2460807
S__inference_batch_normalization_50_layer_call_and_return_conditional_losses_2460853
S__inference_batch_normalization_50_layer_call_and_return_conditional_losses_2460789
S__inference_batch_normalization_50_layer_call_and_return_conditional_losses_2460871?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
8__inference_batch_normalization_50_layer_call_fn_2460833
8__inference_batch_normalization_50_layer_call_fn_2460820
8__inference_batch_normalization_50_layer_call_fn_2460897
8__inference_batch_normalization_50_layer_call_fn_2460884?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_conv2d_143_layer_call_and_return_conditional_losses_2460908?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_conv2d_143_layer_call_fn_2460917?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
S__inference_batch_normalization_51_layer_call_and_return_conditional_losses_2460955
S__inference_batch_normalization_51_layer_call_and_return_conditional_losses_2461019
S__inference_batch_normalization_51_layer_call_and_return_conditional_losses_2460937
S__inference_batch_normalization_51_layer_call_and_return_conditional_losses_2461001?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
8__inference_batch_normalization_51_layer_call_fn_2460981
8__inference_batch_normalization_51_layer_call_fn_2461032
8__inference_batch_normalization_51_layer_call_fn_2460968
8__inference_batch_normalization_51_layer_call_fn_2461045?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_conv2d_144_layer_call_and_return_conditional_losses_2461056?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_conv2d_144_layer_call_fn_2461065?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_add_28_layer_call_and_return_conditional_losses_2461071?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_add_28_layer_call_fn_2461077?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
M__inference_max_pooling2d_57_layer_call_and_return_conditional_losses_2458071?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
2__inference_max_pooling2d_57_layer_call_fn_2458077?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
G__inference_flatten_29_layer_call_and_return_conditional_losses_2461083?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_flatten_29_layer_call_fn_2461088?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_58_layer_call_and_return_conditional_losses_2461099?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_58_layer_call_fn_2461108?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_59_layer_call_and_return_conditional_losses_2461119?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_59_layer_call_fn_2461128?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
%__inference_signature_wrapper_2459541input_30"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
T__inference_CNN_aug_deep_bnorm_skip_layer_call_and_return_conditional_losses_2458879?= !'()*/06789>?LMSTUV[\bcdejkxy???????????????C?@
9?6
,?)
input_30???????????
p

 
? "%?"
?
0?????????
? ?
T__inference_CNN_aug_deep_bnorm_skip_layer_call_and_return_conditional_losses_2458999?= !'()*/06789>?LMSTUV[\bcdejkxy???????????????C?@
9?6
,?)
input_30???????????
p 

 
? "%?"
?
0?????????
? ?
T__inference_CNN_aug_deep_bnorm_skip_layer_call_and_return_conditional_losses_2459726?= !'()*/06789>?LMSTUV[\bcdejkxy???????????????A?>
7?4
*?'
inputs???????????
p

 
? "%?"
?
0?????????
? ?
T__inference_CNN_aug_deep_bnorm_skip_layer_call_and_return_conditional_losses_2459899?= !'()*/06789>?LMSTUV[\bcdejkxy???????????????A?>
7?4
*?'
inputs???????????
p 

 
? "%?"
?
0?????????
? ?
9__inference_CNN_aug_deep_bnorm_skip_layer_call_fn_2459217?= !'()*/06789>?LMSTUV[\bcdejkxy???????????????C?@
9?6
,?)
input_30???????????
p

 
? "???????????
9__inference_CNN_aug_deep_bnorm_skip_layer_call_fn_2459434?= !'()*/06789>?LMSTUV[\bcdejkxy???????????????C?@
9?6
,?)
input_30???????????
p 

 
? "???????????
9__inference_CNN_aug_deep_bnorm_skip_layer_call_fn_2459996?= !'()*/06789>?LMSTUV[\bcdejkxy???????????????A?>
7?4
*?'
inputs???????????
p

 
? "???????????
9__inference_CNN_aug_deep_bnorm_skip_layer_call_fn_2460093?= !'()*/06789>?LMSTUV[\bcdejkxy???????????????A?>
7?4
*?'
inputs???????????
p 

 
? "???????????
"__inference__wrapped_model_2457417?= !'()*/06789>?LMSTUV[\bcdejkxy???????????????;?8
1?.
,?)
input_30???????????
? "3?0
.
dense_59"?
dense_59??????????
C__inference_add_26_layer_call_and_return_conditional_losses_2460415?n?k
d?a
_?\
,?)
inputs/0???????????
,?)
inputs/1???????????
? "/?,
%?"
0???????????
? ?
(__inference_add_26_layer_call_fn_2460421?n?k
d?a
_?\
,?)
inputs/0???????????
,?)
inputs/1???????????
? ""?????????????
C__inference_add_27_layer_call_and_return_conditional_losses_2460743?j?g
`?]
[?X
*?'
inputs/0?????????**
*?'
inputs/1?????????**
? "-?*
#? 
0?????????**
? ?
(__inference_add_27_layer_call_fn_2460749?j?g
`?]
[?X
*?'
inputs/0?????????**
*?'
inputs/1?????????**
? " ??????????**?
C__inference_add_28_layer_call_and_return_conditional_losses_2461071?j?g
`?]
[?X
*?'
inputs/0?????????
*?'
inputs/1?????????
? "-?*
#? 
0?????????
? ?
(__inference_add_28_layer_call_fn_2461077?j?g
`?]
[?X
*?'
inputs/0?????????
*?'
inputs/1?????????
? " ???????????
S__inference_batch_normalization_46_layer_call_and_return_conditional_losses_2460133v'()*=?:
3?0
*?'
inputs???????????
p
? "/?,
%?"
0???????????
? ?
S__inference_batch_normalization_46_layer_call_and_return_conditional_losses_2460151v'()*=?:
3?0
*?'
inputs???????????
p 
? "/?,
%?"
0???????????
? ?
S__inference_batch_normalization_46_layer_call_and_return_conditional_losses_2460197?'()*M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_46_layer_call_and_return_conditional_losses_2460215?'()*M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
8__inference_batch_normalization_46_layer_call_fn_2460164i'()*=?:
3?0
*?'
inputs???????????
p
? ""?????????????
8__inference_batch_normalization_46_layer_call_fn_2460177i'()*=?:
3?0
*?'
inputs???????????
p 
? ""?????????????
8__inference_batch_normalization_46_layer_call_fn_2460228?'()*M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
8__inference_batch_normalization_46_layer_call_fn_2460241?'()*M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
S__inference_batch_normalization_47_layer_call_and_return_conditional_losses_2460281?6789M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_47_layer_call_and_return_conditional_losses_2460299?6789M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_47_layer_call_and_return_conditional_losses_2460345v6789=?:
3?0
*?'
inputs???????????
p
? "/?,
%?"
0???????????
? ?
S__inference_batch_normalization_47_layer_call_and_return_conditional_losses_2460363v6789=?:
3?0
*?'
inputs???????????
p 
? "/?,
%?"
0???????????
? ?
8__inference_batch_normalization_47_layer_call_fn_2460312?6789M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
8__inference_batch_normalization_47_layer_call_fn_2460325?6789M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
8__inference_batch_normalization_47_layer_call_fn_2460376i6789=?:
3?0
*?'
inputs???????????
p
? ""?????????????
8__inference_batch_normalization_47_layer_call_fn_2460389i6789=?:
3?0
*?'
inputs???????????
p 
? ""?????????????
S__inference_batch_normalization_48_layer_call_and_return_conditional_losses_2460461?STUVM?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_48_layer_call_and_return_conditional_losses_2460479?STUVM?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_48_layer_call_and_return_conditional_losses_2460525rSTUV;?8
1?.
(?%
inputs?????????**
p
? "-?*
#? 
0?????????**
? ?
S__inference_batch_normalization_48_layer_call_and_return_conditional_losses_2460543rSTUV;?8
1?.
(?%
inputs?????????**
p 
? "-?*
#? 
0?????????**
? ?
8__inference_batch_normalization_48_layer_call_fn_2460492?STUVM?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
8__inference_batch_normalization_48_layer_call_fn_2460505?STUVM?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
8__inference_batch_normalization_48_layer_call_fn_2460556eSTUV;?8
1?.
(?%
inputs?????????**
p
? " ??????????**?
8__inference_batch_normalization_48_layer_call_fn_2460569eSTUV;?8
1?.
(?%
inputs?????????**
p 
? " ??????????**?
S__inference_batch_normalization_49_layer_call_and_return_conditional_losses_2460609rbcde;?8
1?.
(?%
inputs?????????**
p
? "-?*
#? 
0?????????**
? ?
S__inference_batch_normalization_49_layer_call_and_return_conditional_losses_2460627rbcde;?8
1?.
(?%
inputs?????????**
p 
? "-?*
#? 
0?????????**
? ?
S__inference_batch_normalization_49_layer_call_and_return_conditional_losses_2460673?bcdeM?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_49_layer_call_and_return_conditional_losses_2460691?bcdeM?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
8__inference_batch_normalization_49_layer_call_fn_2460640ebcde;?8
1?.
(?%
inputs?????????**
p
? " ??????????**?
8__inference_batch_normalization_49_layer_call_fn_2460653ebcde;?8
1?.
(?%
inputs?????????**
p 
? " ??????????**?
8__inference_batch_normalization_49_layer_call_fn_2460704?bcdeM?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
8__inference_batch_normalization_49_layer_call_fn_2460717?bcdeM?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
S__inference_batch_normalization_50_layer_call_and_return_conditional_losses_2460789u???;?8
1?.
(?%
inputs?????????
p
? "-?*
#? 
0?????????
? ?
S__inference_batch_normalization_50_layer_call_and_return_conditional_losses_2460807u???;?8
1?.
(?%
inputs?????????
p 
? "-?*
#? 
0?????????
? ?
S__inference_batch_normalization_50_layer_call_and_return_conditional_losses_2460853????M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_50_layer_call_and_return_conditional_losses_2460871????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
8__inference_batch_normalization_50_layer_call_fn_2460820h???;?8
1?.
(?%
inputs?????????
p
? " ???????????
8__inference_batch_normalization_50_layer_call_fn_2460833h???;?8
1?.
(?%
inputs?????????
p 
? " ???????????
8__inference_batch_normalization_50_layer_call_fn_2460884????M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
8__inference_batch_normalization_50_layer_call_fn_2460897????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
S__inference_batch_normalization_51_layer_call_and_return_conditional_losses_2460937?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_51_layer_call_and_return_conditional_losses_2460955?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_51_layer_call_and_return_conditional_losses_2461001v????;?8
1?.
(?%
inputs?????????
p
? "-?*
#? 
0?????????
? ?
S__inference_batch_normalization_51_layer_call_and_return_conditional_losses_2461019v????;?8
1?.
(?%
inputs?????????
p 
? "-?*
#? 
0?????????
? ?
8__inference_batch_normalization_51_layer_call_fn_2460968?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
8__inference_batch_normalization_51_layer_call_fn_2460981?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
8__inference_batch_normalization_51_layer_call_fn_2461032i????;?8
1?.
(?%
inputs?????????
p
? " ???????????
8__inference_batch_normalization_51_layer_call_fn_2461045i????;?8
1?.
(?%
inputs?????????
p 
? " ???????????
G__inference_conv2d_136_layer_call_and_return_conditional_losses_2460104p !9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
,__inference_conv2d_136_layer_call_fn_2460113c !9?6
/?,
*?'
inputs???????????
? ""?????????????
G__inference_conv2d_137_layer_call_and_return_conditional_losses_2460252p/09?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
,__inference_conv2d_137_layer_call_fn_2460261c/09?6
/?,
*?'
inputs???????????
? ""?????????????
G__inference_conv2d_138_layer_call_and_return_conditional_losses_2460400p>?9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
,__inference_conv2d_138_layer_call_fn_2460409c>?9?6
/?,
*?'
inputs???????????
? ""?????????????
G__inference_conv2d_139_layer_call_and_return_conditional_losses_2460432lLM7?4
-?*
(?%
inputs?????????**
? "-?*
#? 
0?????????**
? ?
,__inference_conv2d_139_layer_call_fn_2460441_LM7?4
-?*
(?%
inputs?????????**
? " ??????????**?
G__inference_conv2d_140_layer_call_and_return_conditional_losses_2460580l[\7?4
-?*
(?%
inputs?????????**
? "-?*
#? 
0?????????**
? ?
,__inference_conv2d_140_layer_call_fn_2460589_[\7?4
-?*
(?%
inputs?????????**
? " ??????????**?
G__inference_conv2d_141_layer_call_and_return_conditional_losses_2460728ljk7?4
-?*
(?%
inputs?????????**
? "-?*
#? 
0?????????**
? ?
,__inference_conv2d_141_layer_call_fn_2460737_jk7?4
-?*
(?%
inputs?????????**
? " ??????????**?
G__inference_conv2d_142_layer_call_and_return_conditional_losses_2460760lxy7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
,__inference_conv2d_142_layer_call_fn_2460769_xy7?4
-?*
(?%
inputs?????????
? " ???????????
G__inference_conv2d_143_layer_call_and_return_conditional_losses_2460908n??7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
,__inference_conv2d_143_layer_call_fn_2460917a??7?4
-?*
(?%
inputs?????????
? " ???????????
G__inference_conv2d_144_layer_call_and_return_conditional_losses_2461056n??7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
,__inference_conv2d_144_layer_call_fn_2461065a??7?4
-?*
(?%
inputs?????????
? " ???????????
E__inference_dense_58_layer_call_and_return_conditional_losses_2461099^??/?,
%?"
 ?
inputs?????????
? "%?"
?
0????????? 
? 
*__inference_dense_58_layer_call_fn_2461108Q??/?,
%?"
 ?
inputs?????????
? "?????????? ?
E__inference_dense_59_layer_call_and_return_conditional_losses_2461119^??/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????
? 
*__inference_dense_59_layer_call_fn_2461128Q??/?,
%?"
 ?
inputs????????? 
? "???????????
G__inference_flatten_29_layer_call_and_return_conditional_losses_2461083`7?4
-?*
(?%
inputs?????????
? "%?"
?
0?????????
? ?
,__inference_flatten_29_layer_call_fn_2461088S7?4
-?*
(?%
inputs?????????
? "???????????
M__inference_max_pooling2d_55_layer_call_and_return_conditional_losses_2457631?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
2__inference_max_pooling2d_55_layer_call_fn_2457637?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
M__inference_max_pooling2d_56_layer_call_and_return_conditional_losses_2457851?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
2__inference_max_pooling2d_56_layer_call_fn_2457857?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
M__inference_max_pooling2d_57_layer_call_and_return_conditional_losses_2458071?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
2__inference_max_pooling2d_57_layer_call_fn_2458077?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
%__inference_signature_wrapper_2459541?= !'()*/06789>?LMSTUV[\bcdejkxy???????????????G?D
? 
=?:
8
input_30,?)
input_30???????????"3?0
.
dense_59"?
dense_59?????????