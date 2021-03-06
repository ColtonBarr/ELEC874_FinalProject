??"
??
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
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
?
conv2d_130/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_130/kernel

%conv2d_130/kernel/Read/ReadVariableOpReadVariableOpconv2d_130/kernel*&
_output_shapes
:*
dtype0
v
conv2d_130/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_130/bias
o
#conv2d_130/bias/Read/ReadVariableOpReadVariableOpconv2d_130/bias*
_output_shapes
:*
dtype0
?
batch_normalization_40/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_40/gamma
?
0batch_normalization_40/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_40/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_40/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_40/beta
?
/batch_normalization_40/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_40/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_40/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_40/moving_mean
?
6batch_normalization_40/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_40/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_40/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_40/moving_variance
?
:batch_normalization_40/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_40/moving_variance*
_output_shapes
:*
dtype0
?
conv2d_131/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_131/kernel

%conv2d_131/kernel/Read/ReadVariableOpReadVariableOpconv2d_131/kernel*&
_output_shapes
:*
dtype0
v
conv2d_131/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_131/bias
o
#conv2d_131/bias/Read/ReadVariableOpReadVariableOpconv2d_131/bias*
_output_shapes
:*
dtype0
?
batch_normalization_41/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_41/gamma
?
0batch_normalization_41/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_41/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_41/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_41/beta
?
/batch_normalization_41/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_41/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_41/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_41/moving_mean
?
6batch_normalization_41/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_41/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_41/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_41/moving_variance
?
:batch_normalization_41/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_41/moving_variance*
_output_shapes
:*
dtype0
?
conv2d_132/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_132/kernel

%conv2d_132/kernel/Read/ReadVariableOpReadVariableOpconv2d_132/kernel*&
_output_shapes
:*
dtype0
v
conv2d_132/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_132/bias
o
#conv2d_132/bias/Read/ReadVariableOpReadVariableOpconv2d_132/bias*
_output_shapes
:*
dtype0
?
batch_normalization_42/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_42/gamma
?
0batch_normalization_42/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_42/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_42/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_42/beta
?
/batch_normalization_42/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_42/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_42/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_42/moving_mean
?
6batch_normalization_42/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_42/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_42/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_42/moving_variance
?
:batch_normalization_42/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_42/moving_variance*
_output_shapes
:*
dtype0
?
conv2d_133/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_133/kernel

%conv2d_133/kernel/Read/ReadVariableOpReadVariableOpconv2d_133/kernel*&
_output_shapes
:*
dtype0
v
conv2d_133/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_133/bias
o
#conv2d_133/bias/Read/ReadVariableOpReadVariableOpconv2d_133/bias*
_output_shapes
:*
dtype0
?
batch_normalization_43/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_43/gamma
?
0batch_normalization_43/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_43/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_43/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_43/beta
?
/batch_normalization_43/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_43/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_43/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_43/moving_mean
?
6batch_normalization_43/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_43/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_43/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_43/moving_variance
?
:batch_normalization_43/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_43/moving_variance*
_output_shapes
:*
dtype0
?
conv2d_134/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_134/kernel

%conv2d_134/kernel/Read/ReadVariableOpReadVariableOpconv2d_134/kernel*&
_output_shapes
:*
dtype0
v
conv2d_134/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_134/bias
o
#conv2d_134/bias/Read/ReadVariableOpReadVariableOpconv2d_134/bias*
_output_shapes
:*
dtype0
?
batch_normalization_44/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_44/gamma
?
0batch_normalization_44/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_44/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_44/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_44/beta
?
/batch_normalization_44/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_44/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_44/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_44/moving_mean
?
6batch_normalization_44/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_44/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_44/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_44/moving_variance
?
:batch_normalization_44/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_44/moving_variance*
_output_shapes
:*
dtype0
?
conv2d_135/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_135/kernel

%conv2d_135/kernel/Read/ReadVariableOpReadVariableOpconv2d_135/kernel*&
_output_shapes
:*
dtype0
v
conv2d_135/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_135/bias
o
#conv2d_135/bias/Read/ReadVariableOpReadVariableOpconv2d_135/bias*
_output_shapes
:*
dtype0
?
batch_normalization_45/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_45/gamma
?
0batch_normalization_45/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_45/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_45/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_45/beta
?
/batch_normalization_45/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_45/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_45/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_45/moving_mean
?
6batch_normalization_45/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_45/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_45/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_45/moving_variance
?
:batch_normalization_45/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_45/moving_variance*
_output_shapes
:*
dtype0
{
dense_56/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	? * 
shared_namedense_56/kernel
t
#dense_56/kernel/Read/ReadVariableOpReadVariableOpdense_56/kernel*
_output_shapes
:	? *
dtype0
r
dense_56/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_56/bias
k
!dense_56/bias/Read/ReadVariableOpReadVariableOpdense_56/bias*
_output_shapes
: *
dtype0
z
dense_57/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_57/kernel
s
#dense_57/kernel/Read/ReadVariableOpReadVariableOpdense_57/kernel*
_output_shapes

: *
dtype0
r
dense_57/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_57/bias
k
!dense_57/bias/Read/ReadVariableOpReadVariableOpdense_57/bias*
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
Adam/conv2d_130/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_130/kernel/m
?
,Adam/conv2d_130/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_130/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_130/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_130/bias/m
}
*Adam/conv2d_130/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_130/bias/m*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_40/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_40/gamma/m
?
7Adam/batch_normalization_40/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_40/gamma/m*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_40/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_40/beta/m
?
6Adam/batch_normalization_40/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_40/beta/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_131/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_131/kernel/m
?
,Adam/conv2d_131/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_131/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_131/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_131/bias/m
}
*Adam/conv2d_131/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_131/bias/m*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_41/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_41/gamma/m
?
7Adam/batch_normalization_41/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_41/gamma/m*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_41/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_41/beta/m
?
6Adam/batch_normalization_41/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_41/beta/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_132/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_132/kernel/m
?
,Adam/conv2d_132/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_132/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_132/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_132/bias/m
}
*Adam/conv2d_132/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_132/bias/m*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_42/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_42/gamma/m
?
7Adam/batch_normalization_42/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_42/gamma/m*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_42/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_42/beta/m
?
6Adam/batch_normalization_42/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_42/beta/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_133/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_133/kernel/m
?
,Adam/conv2d_133/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_133/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_133/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_133/bias/m
}
*Adam/conv2d_133/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_133/bias/m*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_43/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_43/gamma/m
?
7Adam/batch_normalization_43/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_43/gamma/m*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_43/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_43/beta/m
?
6Adam/batch_normalization_43/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_43/beta/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_134/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_134/kernel/m
?
,Adam/conv2d_134/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_134/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_134/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_134/bias/m
}
*Adam/conv2d_134/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_134/bias/m*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_44/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_44/gamma/m
?
7Adam/batch_normalization_44/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_44/gamma/m*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_44/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_44/beta/m
?
6Adam/batch_normalization_44/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_44/beta/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_135/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_135/kernel/m
?
,Adam/conv2d_135/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_135/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_135/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_135/bias/m
}
*Adam/conv2d_135/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_135/bias/m*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_45/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_45/gamma/m
?
7Adam/batch_normalization_45/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_45/gamma/m*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_45/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_45/beta/m
?
6Adam/batch_normalization_45/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_45/beta/m*
_output_shapes
:*
dtype0
?
Adam/dense_56/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	? *'
shared_nameAdam/dense_56/kernel/m
?
*Adam/dense_56/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_56/kernel/m*
_output_shapes
:	? *
dtype0
?
Adam/dense_56/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_56/bias/m
y
(Adam/dense_56/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_56/bias/m*
_output_shapes
: *
dtype0
?
Adam/dense_57/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_57/kernel/m
?
*Adam/dense_57/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_57/kernel/m*
_output_shapes

: *
dtype0
?
Adam/dense_57/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_57/bias/m
y
(Adam/dense_57/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_57/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_130/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_130/kernel/v
?
,Adam/conv2d_130/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_130/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_130/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_130/bias/v
}
*Adam/conv2d_130/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_130/bias/v*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_40/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_40/gamma/v
?
7Adam/batch_normalization_40/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_40/gamma/v*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_40/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_40/beta/v
?
6Adam/batch_normalization_40/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_40/beta/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_131/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_131/kernel/v
?
,Adam/conv2d_131/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_131/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_131/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_131/bias/v
}
*Adam/conv2d_131/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_131/bias/v*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_41/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_41/gamma/v
?
7Adam/batch_normalization_41/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_41/gamma/v*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_41/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_41/beta/v
?
6Adam/batch_normalization_41/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_41/beta/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_132/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_132/kernel/v
?
,Adam/conv2d_132/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_132/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_132/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_132/bias/v
}
*Adam/conv2d_132/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_132/bias/v*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_42/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_42/gamma/v
?
7Adam/batch_normalization_42/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_42/gamma/v*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_42/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_42/beta/v
?
6Adam/batch_normalization_42/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_42/beta/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_133/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_133/kernel/v
?
,Adam/conv2d_133/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_133/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_133/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_133/bias/v
}
*Adam/conv2d_133/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_133/bias/v*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_43/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_43/gamma/v
?
7Adam/batch_normalization_43/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_43/gamma/v*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_43/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_43/beta/v
?
6Adam/batch_normalization_43/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_43/beta/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_134/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_134/kernel/v
?
,Adam/conv2d_134/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_134/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_134/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_134/bias/v
}
*Adam/conv2d_134/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_134/bias/v*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_44/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_44/gamma/v
?
7Adam/batch_normalization_44/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_44/gamma/v*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_44/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_44/beta/v
?
6Adam/batch_normalization_44/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_44/beta/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_135/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_135/kernel/v
?
,Adam/conv2d_135/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_135/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_135/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_135/bias/v
}
*Adam/conv2d_135/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_135/bias/v*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_45/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_45/gamma/v
?
7Adam/batch_normalization_45/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_45/gamma/v*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_45/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_45/beta/v
?
6Adam/batch_normalization_45/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_45/beta/v*
_output_shapes
:*
dtype0
?
Adam/dense_56/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	? *'
shared_nameAdam/dense_56/kernel/v
?
*Adam/dense_56/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_56/kernel/v*
_output_shapes
:	? *
dtype0
?
Adam/dense_56/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_56/bias/v
y
(Adam/dense_56/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_56/bias/v*
_output_shapes
: *
dtype0
?
Adam/dense_57/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_57/kernel/v
?
*Adam/dense_57/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_57/kernel/v*
_output_shapes

: *
dtype0
?
Adam/dense_57/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_57/bias/v
y
(Adam/dense_57/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_57/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
Ϧ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer_with_weights-7

layer-9
layer-10
layer_with_weights-8
layer-11
layer_with_weights-9
layer-12
layer_with_weights-10
layer-13
layer_with_weights-11
layer-14
layer-15
layer-16
layer_with_weights-12
layer-17
layer_with_weights-13
layer-18
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
 
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
?
 axis
	!gamma
"beta
#moving_mean
$moving_variance
%regularization_losses
&trainable_variables
'	variables
(	keras_api
h

)kernel
*bias
+regularization_losses
,trainable_variables
-	variables
.	keras_api
?
/axis
	0gamma
1beta
2moving_mean
3moving_variance
4regularization_losses
5trainable_variables
6	variables
7	keras_api
R
8regularization_losses
9trainable_variables
:	variables
;	keras_api
h

<kernel
=bias
>regularization_losses
?trainable_variables
@	variables
A	keras_api
?
Baxis
	Cgamma
Dbeta
Emoving_mean
Fmoving_variance
Gregularization_losses
Htrainable_variables
I	variables
J	keras_api
h

Kkernel
Lbias
Mregularization_losses
Ntrainable_variables
O	variables
P	keras_api
?
Qaxis
	Rgamma
Sbeta
Tmoving_mean
Umoving_variance
Vregularization_losses
Wtrainable_variables
X	variables
Y	keras_api
R
Zregularization_losses
[trainable_variables
\	variables
]	keras_api
h

^kernel
_bias
`regularization_losses
atrainable_variables
b	variables
c	keras_api
?
daxis
	egamma
fbeta
gmoving_mean
hmoving_variance
iregularization_losses
jtrainable_variables
k	variables
l	keras_api
h

mkernel
nbias
oregularization_losses
ptrainable_variables
q	variables
r	keras_api
?
saxis
	tgamma
ubeta
vmoving_mean
wmoving_variance
xregularization_losses
ytrainable_variables
z	variables
{	keras_api
R
|regularization_losses
}trainable_variables
~	variables
	keras_api
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
?
	?iter
?beta_1
?beta_2

?decay
?learning_ratem?m?!m?"m?)m?*m?0m?1m?<m?=m?Cm?Dm?Km?Lm?Rm?Sm?^m?_m?em?fm?mm?nm?tm?um?	?m?	?m?	?m?	?m?v?v?!v?"v?)v?*v?0v?1v?<v?=v?Cv?Dv?Kv?Lv?Rv?Sv?^v?_v?ev?fv?mv?nv?tv?uv?	?v?	?v?	?v?	?v?
 
?
0
1
!2
"3
)4
*5
06
17
<8
=9
C10
D11
K12
L13
R14
S15
^16
_17
e18
f19
m20
n21
t22
u23
?24
?25
?26
?27
?
0
1
!2
"3
#4
$5
)6
*7
08
19
210
311
<12
=13
C14
D15
E16
F17
K18
L19
R20
S21
T22
U23
^24
_25
e26
f27
g28
h29
m30
n31
t32
u33
v34
w35
?36
?37
?38
?39
?
regularization_losses
?layers
?layer_metrics
?metrics
trainable_variables
	variables
 ?layer_regularization_losses
?non_trainable_variables
 
][
VARIABLE_VALUEconv2d_130/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_130/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
regularization_losses
?layers
?layer_metrics
?metrics
trainable_variables
	variables
 ?layer_regularization_losses
?non_trainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_40/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_40/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_40/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_40/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

!0
"1

!0
"1
#2
$3
?
%regularization_losses
?layers
?layer_metrics
?metrics
&trainable_variables
'	variables
 ?layer_regularization_losses
?non_trainable_variables
][
VARIABLE_VALUEconv2d_131/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_131/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

)0
*1

)0
*1
?
+regularization_losses
?layers
?layer_metrics
?metrics
,trainable_variables
-	variables
 ?layer_regularization_losses
?non_trainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_41/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_41/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_41/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_41/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

00
11

00
11
22
33
?
4regularization_losses
?layers
?layer_metrics
?metrics
5trainable_variables
6	variables
 ?layer_regularization_losses
?non_trainable_variables
 
 
 
?
8regularization_losses
?layers
?layer_metrics
?metrics
9trainable_variables
:	variables
 ?layer_regularization_losses
?non_trainable_variables
][
VARIABLE_VALUEconv2d_132/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_132/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

<0
=1

<0
=1
?
>regularization_losses
?layers
?layer_metrics
?metrics
?trainable_variables
@	variables
 ?layer_regularization_losses
?non_trainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_42/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_42/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_42/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_42/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

C0
D1

C0
D1
E2
F3
?
Gregularization_losses
?layers
?layer_metrics
?metrics
Htrainable_variables
I	variables
 ?layer_regularization_losses
?non_trainable_variables
][
VARIABLE_VALUEconv2d_133/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_133/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

K0
L1

K0
L1
?
Mregularization_losses
?layers
?layer_metrics
?metrics
Ntrainable_variables
O	variables
 ?layer_regularization_losses
?non_trainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_43/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_43/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_43/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_43/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

R0
S1

R0
S1
T2
U3
?
Vregularization_losses
?layers
?layer_metrics
?metrics
Wtrainable_variables
X	variables
 ?layer_regularization_losses
?non_trainable_variables
 
 
 
?
Zregularization_losses
?layers
?layer_metrics
?metrics
[trainable_variables
\	variables
 ?layer_regularization_losses
?non_trainable_variables
][
VARIABLE_VALUEconv2d_134/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_134/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
 

^0
_1

^0
_1
?
`regularization_losses
?layers
?layer_metrics
?metrics
atrainable_variables
b	variables
 ?layer_regularization_losses
?non_trainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_44/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_44/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_44/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_44/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

e0
f1

e0
f1
g2
h3
?
iregularization_losses
?layers
?layer_metrics
?metrics
jtrainable_variables
k	variables
 ?layer_regularization_losses
?non_trainable_variables
^\
VARIABLE_VALUEconv2d_135/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEconv2d_135/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE
 

m0
n1

m0
n1
?
oregularization_losses
?layers
?layer_metrics
?metrics
ptrainable_variables
q	variables
 ?layer_regularization_losses
?non_trainable_variables
 
hf
VARIABLE_VALUEbatch_normalization_45/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_45/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_45/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_45/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

t0
u1

t0
u1
v2
w3
?
xregularization_losses
?layers
?layer_metrics
?metrics
ytrainable_variables
z	variables
 ?layer_regularization_losses
?non_trainable_variables
 
 
 
?
|regularization_losses
?layers
?layer_metrics
?metrics
}trainable_variables
~	variables
 ?layer_regularization_losses
?non_trainable_variables
 
 
 
?
?regularization_losses
?layers
?layer_metrics
?metrics
?trainable_variables
?	variables
 ?layer_regularization_losses
?non_trainable_variables
\Z
VARIABLE_VALUEdense_56/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_56/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?regularization_losses
?layers
?layer_metrics
?metrics
?trainable_variables
?	variables
 ?layer_regularization_losses
?non_trainable_variables
\Z
VARIABLE_VALUEdense_57/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_57/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?regularization_losses
?layers
?layer_metrics
?metrics
?trainable_variables
?	variables
 ?layer_regularization_losses
?non_trainable_variables
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
 

?0
?1
 
V
#0
$1
22
33
E4
F5
T6
U7
g8
h9
v10
w11
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
#0
$1
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
20
31
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
E0
F1
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
T0
U1
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
g0
h1
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
v0
w1
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

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
?~
VARIABLE_VALUEAdam/conv2d_130/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_130/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_40/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_40/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_131/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_131/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_41/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_41/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_132/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_132/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_42/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_42/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_133/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_133/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_43/gamma/mQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_43/beta/mPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_134/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_134/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_44/gamma/mQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_44/beta/mPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/conv2d_135/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/conv2d_135/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_45/gamma/mRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_45/beta/mQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_56/kernel/mSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_56/bias/mQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_57/kernel/mSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_57/bias/mQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_130/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_130/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_40/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_40/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_131/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_131/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_41/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_41/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_132/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_132/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_42/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_42/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_133/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_133/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_43/gamma/vQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_43/beta/vPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_134/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_134/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_44/gamma/vQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_44/beta/vPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/conv2d_135/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/conv2d_135/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_45/gamma/vRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_45/beta/vQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_56/kernel/vSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_56/bias/vQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_57/kernel/vSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_57/bias/vQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_29Placeholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_29conv2d_130/kernelconv2d_130/biasbatch_normalization_40/gammabatch_normalization_40/beta"batch_normalization_40/moving_mean&batch_normalization_40/moving_varianceconv2d_131/kernelconv2d_131/biasbatch_normalization_41/gammabatch_normalization_41/beta"batch_normalization_41/moving_mean&batch_normalization_41/moving_varianceconv2d_132/kernelconv2d_132/biasbatch_normalization_42/gammabatch_normalization_42/beta"batch_normalization_42/moving_mean&batch_normalization_42/moving_varianceconv2d_133/kernelconv2d_133/biasbatch_normalization_43/gammabatch_normalization_43/beta"batch_normalization_43/moving_mean&batch_normalization_43/moving_varianceconv2d_134/kernelconv2d_134/biasbatch_normalization_44/gammabatch_normalization_44/beta"batch_normalization_44/moving_mean&batch_normalization_44/moving_varianceconv2d_135/kernelconv2d_135/biasbatch_normalization_45/gammabatch_normalization_45/beta"batch_normalization_45/moving_mean&batch_normalization_45/moving_variancedense_56/kerneldense_56/biasdense_57/kerneldense_57/bias*4
Tin-
+2)*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*J
_read_only_resource_inputs,
*(	
 !"#$%&'(*-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_2304122
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?)
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%conv2d_130/kernel/Read/ReadVariableOp#conv2d_130/bias/Read/ReadVariableOp0batch_normalization_40/gamma/Read/ReadVariableOp/batch_normalization_40/beta/Read/ReadVariableOp6batch_normalization_40/moving_mean/Read/ReadVariableOp:batch_normalization_40/moving_variance/Read/ReadVariableOp%conv2d_131/kernel/Read/ReadVariableOp#conv2d_131/bias/Read/ReadVariableOp0batch_normalization_41/gamma/Read/ReadVariableOp/batch_normalization_41/beta/Read/ReadVariableOp6batch_normalization_41/moving_mean/Read/ReadVariableOp:batch_normalization_41/moving_variance/Read/ReadVariableOp%conv2d_132/kernel/Read/ReadVariableOp#conv2d_132/bias/Read/ReadVariableOp0batch_normalization_42/gamma/Read/ReadVariableOp/batch_normalization_42/beta/Read/ReadVariableOp6batch_normalization_42/moving_mean/Read/ReadVariableOp:batch_normalization_42/moving_variance/Read/ReadVariableOp%conv2d_133/kernel/Read/ReadVariableOp#conv2d_133/bias/Read/ReadVariableOp0batch_normalization_43/gamma/Read/ReadVariableOp/batch_normalization_43/beta/Read/ReadVariableOp6batch_normalization_43/moving_mean/Read/ReadVariableOp:batch_normalization_43/moving_variance/Read/ReadVariableOp%conv2d_134/kernel/Read/ReadVariableOp#conv2d_134/bias/Read/ReadVariableOp0batch_normalization_44/gamma/Read/ReadVariableOp/batch_normalization_44/beta/Read/ReadVariableOp6batch_normalization_44/moving_mean/Read/ReadVariableOp:batch_normalization_44/moving_variance/Read/ReadVariableOp%conv2d_135/kernel/Read/ReadVariableOp#conv2d_135/bias/Read/ReadVariableOp0batch_normalization_45/gamma/Read/ReadVariableOp/batch_normalization_45/beta/Read/ReadVariableOp6batch_normalization_45/moving_mean/Read/ReadVariableOp:batch_normalization_45/moving_variance/Read/ReadVariableOp#dense_56/kernel/Read/ReadVariableOp!dense_56/bias/Read/ReadVariableOp#dense_57/kernel/Read/ReadVariableOp!dense_57/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp,Adam/conv2d_130/kernel/m/Read/ReadVariableOp*Adam/conv2d_130/bias/m/Read/ReadVariableOp7Adam/batch_normalization_40/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_40/beta/m/Read/ReadVariableOp,Adam/conv2d_131/kernel/m/Read/ReadVariableOp*Adam/conv2d_131/bias/m/Read/ReadVariableOp7Adam/batch_normalization_41/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_41/beta/m/Read/ReadVariableOp,Adam/conv2d_132/kernel/m/Read/ReadVariableOp*Adam/conv2d_132/bias/m/Read/ReadVariableOp7Adam/batch_normalization_42/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_42/beta/m/Read/ReadVariableOp,Adam/conv2d_133/kernel/m/Read/ReadVariableOp*Adam/conv2d_133/bias/m/Read/ReadVariableOp7Adam/batch_normalization_43/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_43/beta/m/Read/ReadVariableOp,Adam/conv2d_134/kernel/m/Read/ReadVariableOp*Adam/conv2d_134/bias/m/Read/ReadVariableOp7Adam/batch_normalization_44/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_44/beta/m/Read/ReadVariableOp,Adam/conv2d_135/kernel/m/Read/ReadVariableOp*Adam/conv2d_135/bias/m/Read/ReadVariableOp7Adam/batch_normalization_45/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_45/beta/m/Read/ReadVariableOp*Adam/dense_56/kernel/m/Read/ReadVariableOp(Adam/dense_56/bias/m/Read/ReadVariableOp*Adam/dense_57/kernel/m/Read/ReadVariableOp(Adam/dense_57/bias/m/Read/ReadVariableOp,Adam/conv2d_130/kernel/v/Read/ReadVariableOp*Adam/conv2d_130/bias/v/Read/ReadVariableOp7Adam/batch_normalization_40/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_40/beta/v/Read/ReadVariableOp,Adam/conv2d_131/kernel/v/Read/ReadVariableOp*Adam/conv2d_131/bias/v/Read/ReadVariableOp7Adam/batch_normalization_41/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_41/beta/v/Read/ReadVariableOp,Adam/conv2d_132/kernel/v/Read/ReadVariableOp*Adam/conv2d_132/bias/v/Read/ReadVariableOp7Adam/batch_normalization_42/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_42/beta/v/Read/ReadVariableOp,Adam/conv2d_133/kernel/v/Read/ReadVariableOp*Adam/conv2d_133/bias/v/Read/ReadVariableOp7Adam/batch_normalization_43/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_43/beta/v/Read/ReadVariableOp,Adam/conv2d_134/kernel/v/Read/ReadVariableOp*Adam/conv2d_134/bias/v/Read/ReadVariableOp7Adam/batch_normalization_44/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_44/beta/v/Read/ReadVariableOp,Adam/conv2d_135/kernel/v/Read/ReadVariableOp*Adam/conv2d_135/bias/v/Read/ReadVariableOp7Adam/batch_normalization_45/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_45/beta/v/Read/ReadVariableOp*Adam/dense_56/kernel/v/Read/ReadVariableOp(Adam/dense_56/bias/v/Read/ReadVariableOp*Adam/dense_57/kernel/v/Read/ReadVariableOp(Adam/dense_57/bias/v/Read/ReadVariableOpConst*v
Tino
m2k	*
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
 __inference__traced_save_2305879
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_130/kernelconv2d_130/biasbatch_normalization_40/gammabatch_normalization_40/beta"batch_normalization_40/moving_mean&batch_normalization_40/moving_varianceconv2d_131/kernelconv2d_131/biasbatch_normalization_41/gammabatch_normalization_41/beta"batch_normalization_41/moving_mean&batch_normalization_41/moving_varianceconv2d_132/kernelconv2d_132/biasbatch_normalization_42/gammabatch_normalization_42/beta"batch_normalization_42/moving_mean&batch_normalization_42/moving_varianceconv2d_133/kernelconv2d_133/biasbatch_normalization_43/gammabatch_normalization_43/beta"batch_normalization_43/moving_mean&batch_normalization_43/moving_varianceconv2d_134/kernelconv2d_134/biasbatch_normalization_44/gammabatch_normalization_44/beta"batch_normalization_44/moving_mean&batch_normalization_44/moving_varianceconv2d_135/kernelconv2d_135/biasbatch_normalization_45/gammabatch_normalization_45/beta"batch_normalization_45/moving_mean&batch_normalization_45/moving_variancedense_56/kerneldense_56/biasdense_57/kerneldense_57/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv2d_130/kernel/mAdam/conv2d_130/bias/m#Adam/batch_normalization_40/gamma/m"Adam/batch_normalization_40/beta/mAdam/conv2d_131/kernel/mAdam/conv2d_131/bias/m#Adam/batch_normalization_41/gamma/m"Adam/batch_normalization_41/beta/mAdam/conv2d_132/kernel/mAdam/conv2d_132/bias/m#Adam/batch_normalization_42/gamma/m"Adam/batch_normalization_42/beta/mAdam/conv2d_133/kernel/mAdam/conv2d_133/bias/m#Adam/batch_normalization_43/gamma/m"Adam/batch_normalization_43/beta/mAdam/conv2d_134/kernel/mAdam/conv2d_134/bias/m#Adam/batch_normalization_44/gamma/m"Adam/batch_normalization_44/beta/mAdam/conv2d_135/kernel/mAdam/conv2d_135/bias/m#Adam/batch_normalization_45/gamma/m"Adam/batch_normalization_45/beta/mAdam/dense_56/kernel/mAdam/dense_56/bias/mAdam/dense_57/kernel/mAdam/dense_57/bias/mAdam/conv2d_130/kernel/vAdam/conv2d_130/bias/v#Adam/batch_normalization_40/gamma/v"Adam/batch_normalization_40/beta/vAdam/conv2d_131/kernel/vAdam/conv2d_131/bias/v#Adam/batch_normalization_41/gamma/v"Adam/batch_normalization_41/beta/vAdam/conv2d_132/kernel/vAdam/conv2d_132/bias/v#Adam/batch_normalization_42/gamma/v"Adam/batch_normalization_42/beta/vAdam/conv2d_133/kernel/vAdam/conv2d_133/bias/v#Adam/batch_normalization_43/gamma/v"Adam/batch_normalization_43/beta/vAdam/conv2d_134/kernel/vAdam/conv2d_134/bias/v#Adam/batch_normalization_44/gamma/v"Adam/batch_normalization_44/beta/vAdam/conv2d_135/kernel/vAdam/conv2d_135/bias/v#Adam/batch_normalization_45/gamma/v"Adam/batch_normalization_45/beta/vAdam/dense_56/kernel/vAdam/dense_56/bias/vAdam/dense_57/kernel/vAdam/dense_57/bias/v*u
Tinn
l2j*
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
#__inference__traced_restore_2306204??
??
?0
 __inference__traced_save_2305879
file_prefix0
,savev2_conv2d_130_kernel_read_readvariableop.
*savev2_conv2d_130_bias_read_readvariableop;
7savev2_batch_normalization_40_gamma_read_readvariableop:
6savev2_batch_normalization_40_beta_read_readvariableopA
=savev2_batch_normalization_40_moving_mean_read_readvariableopE
Asavev2_batch_normalization_40_moving_variance_read_readvariableop0
,savev2_conv2d_131_kernel_read_readvariableop.
*savev2_conv2d_131_bias_read_readvariableop;
7savev2_batch_normalization_41_gamma_read_readvariableop:
6savev2_batch_normalization_41_beta_read_readvariableopA
=savev2_batch_normalization_41_moving_mean_read_readvariableopE
Asavev2_batch_normalization_41_moving_variance_read_readvariableop0
,savev2_conv2d_132_kernel_read_readvariableop.
*savev2_conv2d_132_bias_read_readvariableop;
7savev2_batch_normalization_42_gamma_read_readvariableop:
6savev2_batch_normalization_42_beta_read_readvariableopA
=savev2_batch_normalization_42_moving_mean_read_readvariableopE
Asavev2_batch_normalization_42_moving_variance_read_readvariableop0
,savev2_conv2d_133_kernel_read_readvariableop.
*savev2_conv2d_133_bias_read_readvariableop;
7savev2_batch_normalization_43_gamma_read_readvariableop:
6savev2_batch_normalization_43_beta_read_readvariableopA
=savev2_batch_normalization_43_moving_mean_read_readvariableopE
Asavev2_batch_normalization_43_moving_variance_read_readvariableop0
,savev2_conv2d_134_kernel_read_readvariableop.
*savev2_conv2d_134_bias_read_readvariableop;
7savev2_batch_normalization_44_gamma_read_readvariableop:
6savev2_batch_normalization_44_beta_read_readvariableopA
=savev2_batch_normalization_44_moving_mean_read_readvariableopE
Asavev2_batch_normalization_44_moving_variance_read_readvariableop0
,savev2_conv2d_135_kernel_read_readvariableop.
*savev2_conv2d_135_bias_read_readvariableop;
7savev2_batch_normalization_45_gamma_read_readvariableop:
6savev2_batch_normalization_45_beta_read_readvariableopA
=savev2_batch_normalization_45_moving_mean_read_readvariableopE
Asavev2_batch_normalization_45_moving_variance_read_readvariableop.
*savev2_dense_56_kernel_read_readvariableop,
(savev2_dense_56_bias_read_readvariableop.
*savev2_dense_57_kernel_read_readvariableop,
(savev2_dense_57_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop7
3savev2_adam_conv2d_130_kernel_m_read_readvariableop5
1savev2_adam_conv2d_130_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_40_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_40_beta_m_read_readvariableop7
3savev2_adam_conv2d_131_kernel_m_read_readvariableop5
1savev2_adam_conv2d_131_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_41_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_41_beta_m_read_readvariableop7
3savev2_adam_conv2d_132_kernel_m_read_readvariableop5
1savev2_adam_conv2d_132_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_42_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_42_beta_m_read_readvariableop7
3savev2_adam_conv2d_133_kernel_m_read_readvariableop5
1savev2_adam_conv2d_133_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_43_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_43_beta_m_read_readvariableop7
3savev2_adam_conv2d_134_kernel_m_read_readvariableop5
1savev2_adam_conv2d_134_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_44_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_44_beta_m_read_readvariableop7
3savev2_adam_conv2d_135_kernel_m_read_readvariableop5
1savev2_adam_conv2d_135_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_45_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_45_beta_m_read_readvariableop5
1savev2_adam_dense_56_kernel_m_read_readvariableop3
/savev2_adam_dense_56_bias_m_read_readvariableop5
1savev2_adam_dense_57_kernel_m_read_readvariableop3
/savev2_adam_dense_57_bias_m_read_readvariableop7
3savev2_adam_conv2d_130_kernel_v_read_readvariableop5
1savev2_adam_conv2d_130_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_40_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_40_beta_v_read_readvariableop7
3savev2_adam_conv2d_131_kernel_v_read_readvariableop5
1savev2_adam_conv2d_131_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_41_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_41_beta_v_read_readvariableop7
3savev2_adam_conv2d_132_kernel_v_read_readvariableop5
1savev2_adam_conv2d_132_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_42_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_42_beta_v_read_readvariableop7
3savev2_adam_conv2d_133_kernel_v_read_readvariableop5
1savev2_adam_conv2d_133_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_43_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_43_beta_v_read_readvariableop7
3savev2_adam_conv2d_134_kernel_v_read_readvariableop5
1savev2_adam_conv2d_134_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_44_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_44_beta_v_read_readvariableop7
3savev2_adam_conv2d_135_kernel_v_read_readvariableop5
1savev2_adam_conv2d_135_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_45_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_45_beta_v_read_readvariableop5
1savev2_adam_dense_56_kernel_v_read_readvariableop3
/savev2_adam_dense_56_bias_v_read_readvariableop5
1savev2_adam_dense_57_kernel_v_read_readvariableop3
/savev2_adam_dense_57_bias_v_read_readvariableop
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
ShardedFilename?;
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:j*
dtype0*?:
value?:B?:jB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:j*
dtype0*?
value?B?jB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?.
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_conv2d_130_kernel_read_readvariableop*savev2_conv2d_130_bias_read_readvariableop7savev2_batch_normalization_40_gamma_read_readvariableop6savev2_batch_normalization_40_beta_read_readvariableop=savev2_batch_normalization_40_moving_mean_read_readvariableopAsavev2_batch_normalization_40_moving_variance_read_readvariableop,savev2_conv2d_131_kernel_read_readvariableop*savev2_conv2d_131_bias_read_readvariableop7savev2_batch_normalization_41_gamma_read_readvariableop6savev2_batch_normalization_41_beta_read_readvariableop=savev2_batch_normalization_41_moving_mean_read_readvariableopAsavev2_batch_normalization_41_moving_variance_read_readvariableop,savev2_conv2d_132_kernel_read_readvariableop*savev2_conv2d_132_bias_read_readvariableop7savev2_batch_normalization_42_gamma_read_readvariableop6savev2_batch_normalization_42_beta_read_readvariableop=savev2_batch_normalization_42_moving_mean_read_readvariableopAsavev2_batch_normalization_42_moving_variance_read_readvariableop,savev2_conv2d_133_kernel_read_readvariableop*savev2_conv2d_133_bias_read_readvariableop7savev2_batch_normalization_43_gamma_read_readvariableop6savev2_batch_normalization_43_beta_read_readvariableop=savev2_batch_normalization_43_moving_mean_read_readvariableopAsavev2_batch_normalization_43_moving_variance_read_readvariableop,savev2_conv2d_134_kernel_read_readvariableop*savev2_conv2d_134_bias_read_readvariableop7savev2_batch_normalization_44_gamma_read_readvariableop6savev2_batch_normalization_44_beta_read_readvariableop=savev2_batch_normalization_44_moving_mean_read_readvariableopAsavev2_batch_normalization_44_moving_variance_read_readvariableop,savev2_conv2d_135_kernel_read_readvariableop*savev2_conv2d_135_bias_read_readvariableop7savev2_batch_normalization_45_gamma_read_readvariableop6savev2_batch_normalization_45_beta_read_readvariableop=savev2_batch_normalization_45_moving_mean_read_readvariableopAsavev2_batch_normalization_45_moving_variance_read_readvariableop*savev2_dense_56_kernel_read_readvariableop(savev2_dense_56_bias_read_readvariableop*savev2_dense_57_kernel_read_readvariableop(savev2_dense_57_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop3savev2_adam_conv2d_130_kernel_m_read_readvariableop1savev2_adam_conv2d_130_bias_m_read_readvariableop>savev2_adam_batch_normalization_40_gamma_m_read_readvariableop=savev2_adam_batch_normalization_40_beta_m_read_readvariableop3savev2_adam_conv2d_131_kernel_m_read_readvariableop1savev2_adam_conv2d_131_bias_m_read_readvariableop>savev2_adam_batch_normalization_41_gamma_m_read_readvariableop=savev2_adam_batch_normalization_41_beta_m_read_readvariableop3savev2_adam_conv2d_132_kernel_m_read_readvariableop1savev2_adam_conv2d_132_bias_m_read_readvariableop>savev2_adam_batch_normalization_42_gamma_m_read_readvariableop=savev2_adam_batch_normalization_42_beta_m_read_readvariableop3savev2_adam_conv2d_133_kernel_m_read_readvariableop1savev2_adam_conv2d_133_bias_m_read_readvariableop>savev2_adam_batch_normalization_43_gamma_m_read_readvariableop=savev2_adam_batch_normalization_43_beta_m_read_readvariableop3savev2_adam_conv2d_134_kernel_m_read_readvariableop1savev2_adam_conv2d_134_bias_m_read_readvariableop>savev2_adam_batch_normalization_44_gamma_m_read_readvariableop=savev2_adam_batch_normalization_44_beta_m_read_readvariableop3savev2_adam_conv2d_135_kernel_m_read_readvariableop1savev2_adam_conv2d_135_bias_m_read_readvariableop>savev2_adam_batch_normalization_45_gamma_m_read_readvariableop=savev2_adam_batch_normalization_45_beta_m_read_readvariableop1savev2_adam_dense_56_kernel_m_read_readvariableop/savev2_adam_dense_56_bias_m_read_readvariableop1savev2_adam_dense_57_kernel_m_read_readvariableop/savev2_adam_dense_57_bias_m_read_readvariableop3savev2_adam_conv2d_130_kernel_v_read_readvariableop1savev2_adam_conv2d_130_bias_v_read_readvariableop>savev2_adam_batch_normalization_40_gamma_v_read_readvariableop=savev2_adam_batch_normalization_40_beta_v_read_readvariableop3savev2_adam_conv2d_131_kernel_v_read_readvariableop1savev2_adam_conv2d_131_bias_v_read_readvariableop>savev2_adam_batch_normalization_41_gamma_v_read_readvariableop=savev2_adam_batch_normalization_41_beta_v_read_readvariableop3savev2_adam_conv2d_132_kernel_v_read_readvariableop1savev2_adam_conv2d_132_bias_v_read_readvariableop>savev2_adam_batch_normalization_42_gamma_v_read_readvariableop=savev2_adam_batch_normalization_42_beta_v_read_readvariableop3savev2_adam_conv2d_133_kernel_v_read_readvariableop1savev2_adam_conv2d_133_bias_v_read_readvariableop>savev2_adam_batch_normalization_43_gamma_v_read_readvariableop=savev2_adam_batch_normalization_43_beta_v_read_readvariableop3savev2_adam_conv2d_134_kernel_v_read_readvariableop1savev2_adam_conv2d_134_bias_v_read_readvariableop>savev2_adam_batch_normalization_44_gamma_v_read_readvariableop=savev2_adam_batch_normalization_44_beta_v_read_readvariableop3savev2_adam_conv2d_135_kernel_v_read_readvariableop1savev2_adam_conv2d_135_bias_v_read_readvariableop>savev2_adam_batch_normalization_45_gamma_v_read_readvariableop=savev2_adam_batch_normalization_45_beta_v_read_readvariableop1savev2_adam_dense_56_kernel_v_read_readvariableop/savev2_adam_dense_56_bias_v_read_readvariableop1savev2_adam_dense_57_kernel_v_read_readvariableop/savev2_adam_dense_57_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *x
dtypesn
l2j	2
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::	? : : :: : : : : : : : : :::::::::::::::::::::::::	? : : ::::::::::::::::::::::::::	? : : :: 2(
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
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
::  

_output_shapes
:: !

_output_shapes
:: "

_output_shapes
:: #

_output_shapes
:: $

_output_shapes
::%%!

_output_shapes
:	? : &

_output_shapes
: :$' 

_output_shapes

: : (

_output_shapes
::)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: :1

_output_shapes
: :,2(
&
_output_shapes
:: 3

_output_shapes
:: 4

_output_shapes
:: 5

_output_shapes
::,6(
&
_output_shapes
:: 7

_output_shapes
:: 8

_output_shapes
:: 9

_output_shapes
::,:(
&
_output_shapes
:: ;

_output_shapes
:: <

_output_shapes
:: =

_output_shapes
::,>(
&
_output_shapes
:: ?

_output_shapes
:: @

_output_shapes
:: A

_output_shapes
::,B(
&
_output_shapes
:: C
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
::%J!

_output_shapes
:	? : K

_output_shapes
: :$L 

_output_shapes

: : M

_output_shapes
::,N(
&
_output_shapes
:: O

_output_shapes
:: P

_output_shapes
:: Q

_output_shapes
::,R(
&
_output_shapes
:: S

_output_shapes
:: T

_output_shapes
:: U

_output_shapes
::,V(
&
_output_shapes
:: W

_output_shapes
:: X

_output_shapes
:: Y

_output_shapes
::,Z(
&
_output_shapes
:: [
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
:: c

_output_shapes
:: d

_output_shapes
:: e

_output_shapes
::%f!

_output_shapes
:	? : g

_output_shapes
: :$h 

_output_shapes

: : i

_output_shapes
::j

_output_shapes
: 
?
c
G__inference_flatten_28_layer_call_and_return_conditional_losses_2305496

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?e
?
O__inference_CNN_aug_deep_bnorm_layer_call_and_return_conditional_losses_2303757

inputs
conv2d_130_2303658
conv2d_130_2303660"
batch_normalization_40_2303663"
batch_normalization_40_2303665"
batch_normalization_40_2303667"
batch_normalization_40_2303669
conv2d_131_2303672
conv2d_131_2303674"
batch_normalization_41_2303677"
batch_normalization_41_2303679"
batch_normalization_41_2303681"
batch_normalization_41_2303683
conv2d_132_2303687
conv2d_132_2303689"
batch_normalization_42_2303692"
batch_normalization_42_2303694"
batch_normalization_42_2303696"
batch_normalization_42_2303698
conv2d_133_2303701
conv2d_133_2303703"
batch_normalization_43_2303706"
batch_normalization_43_2303708"
batch_normalization_43_2303710"
batch_normalization_43_2303712
conv2d_134_2303716
conv2d_134_2303718"
batch_normalization_44_2303721"
batch_normalization_44_2303723"
batch_normalization_44_2303725"
batch_normalization_44_2303727
conv2d_135_2303730
conv2d_135_2303732"
batch_normalization_45_2303735"
batch_normalization_45_2303737"
batch_normalization_45_2303739"
batch_normalization_45_2303741
dense_56_2303746
dense_56_2303748
dense_57_2303751
dense_57_2303753
identity??.batch_normalization_40/StatefulPartitionedCall?.batch_normalization_41/StatefulPartitionedCall?.batch_normalization_42/StatefulPartitionedCall?.batch_normalization_43/StatefulPartitionedCall?.batch_normalization_44/StatefulPartitionedCall?.batch_normalization_45/StatefulPartitionedCall?"conv2d_130/StatefulPartitionedCall?"conv2d_131/StatefulPartitionedCall?"conv2d_132/StatefulPartitionedCall?"conv2d_133/StatefulPartitionedCall?"conv2d_134/StatefulPartitionedCall?"conv2d_135/StatefulPartitionedCall? dense_56/StatefulPartitionedCall? dense_57/StatefulPartitionedCall?
"conv2d_130/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_130_2303658conv2d_130_2303660*
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
G__inference_conv2d_130_layer_call_and_return_conditional_losses_23028892$
"conv2d_130/StatefulPartitionedCall?
.batch_normalization_40/StatefulPartitionedCallStatefulPartitionedCall+conv2d_130/StatefulPartitionedCall:output:0batch_normalization_40_2303663batch_normalization_40_2303665batch_normalization_40_2303667batch_normalization_40_2303669*
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
S__inference_batch_normalization_40_layer_call_and_return_conditional_losses_230292420
.batch_normalization_40/StatefulPartitionedCall?
"conv2d_131/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_40/StatefulPartitionedCall:output:0conv2d_131_2303672conv2d_131_2303674*
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
G__inference_conv2d_131_layer_call_and_return_conditional_losses_23029892$
"conv2d_131/StatefulPartitionedCall?
.batch_normalization_41/StatefulPartitionedCallStatefulPartitionedCall+conv2d_131/StatefulPartitionedCall:output:0batch_normalization_41_2303677batch_normalization_41_2303679batch_normalization_41_2303681batch_normalization_41_2303683*
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
S__inference_batch_normalization_41_layer_call_and_return_conditional_losses_230302420
.batch_normalization_41/StatefulPartitionedCall?
 max_pooling2d_52/PartitionedCallPartitionedCall7batch_normalization_41/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*** 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_max_pooling2d_52_layer_call_and_return_conditional_losses_23024282"
 max_pooling2d_52/PartitionedCall?
"conv2d_132/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_52/PartitionedCall:output:0conv2d_132_2303687conv2d_132_2303689*
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
G__inference_conv2d_132_layer_call_and_return_conditional_losses_23030902$
"conv2d_132/StatefulPartitionedCall?
.batch_normalization_42/StatefulPartitionedCallStatefulPartitionedCall+conv2d_132/StatefulPartitionedCall:output:0batch_normalization_42_2303692batch_normalization_42_2303694batch_normalization_42_2303696batch_normalization_42_2303698*
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
S__inference_batch_normalization_42_layer_call_and_return_conditional_losses_230312520
.batch_normalization_42/StatefulPartitionedCall?
"conv2d_133/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_42/StatefulPartitionedCall:output:0conv2d_133_2303701conv2d_133_2303703*
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
G__inference_conv2d_133_layer_call_and_return_conditional_losses_23031902$
"conv2d_133/StatefulPartitionedCall?
.batch_normalization_43/StatefulPartitionedCallStatefulPartitionedCall+conv2d_133/StatefulPartitionedCall:output:0batch_normalization_43_2303706batch_normalization_43_2303708batch_normalization_43_2303710batch_normalization_43_2303712*
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
S__inference_batch_normalization_43_layer_call_and_return_conditional_losses_230322520
.batch_normalization_43/StatefulPartitionedCall?
 max_pooling2d_53/PartitionedCallPartitionedCall7batch_normalization_43/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_max_pooling2d_53_layer_call_and_return_conditional_losses_23026482"
 max_pooling2d_53/PartitionedCall?
"conv2d_134/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_53/PartitionedCall:output:0conv2d_134_2303716conv2d_134_2303718*
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
G__inference_conv2d_134_layer_call_and_return_conditional_losses_23032912$
"conv2d_134/StatefulPartitionedCall?
.batch_normalization_44/StatefulPartitionedCallStatefulPartitionedCall+conv2d_134/StatefulPartitionedCall:output:0batch_normalization_44_2303721batch_normalization_44_2303723batch_normalization_44_2303725batch_normalization_44_2303727*
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
S__inference_batch_normalization_44_layer_call_and_return_conditional_losses_230332620
.batch_normalization_44/StatefulPartitionedCall?
"conv2d_135/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_44/StatefulPartitionedCall:output:0conv2d_135_2303730conv2d_135_2303732*
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
G__inference_conv2d_135_layer_call_and_return_conditional_losses_23033912$
"conv2d_135/StatefulPartitionedCall?
.batch_normalization_45/StatefulPartitionedCallStatefulPartitionedCall+conv2d_135/StatefulPartitionedCall:output:0batch_normalization_45_2303735batch_normalization_45_2303737batch_normalization_45_2303739batch_normalization_45_2303741*
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
S__inference_batch_normalization_45_layer_call_and_return_conditional_losses_230342620
.batch_normalization_45/StatefulPartitionedCall?
 max_pooling2d_54/PartitionedCallPartitionedCall7batch_normalization_45/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_max_pooling2d_54_layer_call_and_return_conditional_losses_23028682"
 max_pooling2d_54/PartitionedCall?
flatten_28/PartitionedCallPartitionedCall)max_pooling2d_54/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_flatten_28_layer_call_and_return_conditional_losses_23034872
flatten_28/PartitionedCall?
 dense_56/StatefulPartitionedCallStatefulPartitionedCall#flatten_28/PartitionedCall:output:0dense_56_2303746dense_56_2303748*
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
E__inference_dense_56_layer_call_and_return_conditional_losses_23035062"
 dense_56/StatefulPartitionedCall?
 dense_57/StatefulPartitionedCallStatefulPartitionedCall)dense_56/StatefulPartitionedCall:output:0dense_57_2303751dense_57_2303753*
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
E__inference_dense_57_layer_call_and_return_conditional_losses_23035332"
 dense_57/StatefulPartitionedCall?
IdentityIdentity)dense_57/StatefulPartitionedCall:output:0/^batch_normalization_40/StatefulPartitionedCall/^batch_normalization_41/StatefulPartitionedCall/^batch_normalization_42/StatefulPartitionedCall/^batch_normalization_43/StatefulPartitionedCall/^batch_normalization_44/StatefulPartitionedCall/^batch_normalization_45/StatefulPartitionedCall#^conv2d_130/StatefulPartitionedCall#^conv2d_131/StatefulPartitionedCall#^conv2d_132/StatefulPartitionedCall#^conv2d_133/StatefulPartitionedCall#^conv2d_134/StatefulPartitionedCall#^conv2d_135/StatefulPartitionedCall!^dense_56/StatefulPartitionedCall!^dense_57/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_40/StatefulPartitionedCall.batch_normalization_40/StatefulPartitionedCall2`
.batch_normalization_41/StatefulPartitionedCall.batch_normalization_41/StatefulPartitionedCall2`
.batch_normalization_42/StatefulPartitionedCall.batch_normalization_42/StatefulPartitionedCall2`
.batch_normalization_43/StatefulPartitionedCall.batch_normalization_43/StatefulPartitionedCall2`
.batch_normalization_44/StatefulPartitionedCall.batch_normalization_44/StatefulPartitionedCall2`
.batch_normalization_45/StatefulPartitionedCall.batch_normalization_45/StatefulPartitionedCall2H
"conv2d_130/StatefulPartitionedCall"conv2d_130/StatefulPartitionedCall2H
"conv2d_131/StatefulPartitionedCall"conv2d_131/StatefulPartitionedCall2H
"conv2d_132/StatefulPartitionedCall"conv2d_132/StatefulPartitionedCall2H
"conv2d_133/StatefulPartitionedCall"conv2d_133/StatefulPartitionedCall2H
"conv2d_134/StatefulPartitionedCall"conv2d_134/StatefulPartitionedCall2H
"conv2d_135/StatefulPartitionedCall"conv2d_135/StatefulPartitionedCall2D
 dense_56/StatefulPartitionedCall dense_56/StatefulPartitionedCall2D
 dense_57/StatefulPartitionedCall dense_57/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
4__inference_CNN_aug_deep_bnorm_layer_call_fn_2304027
input_29
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

unknown_38
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_29unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_38*4
Tin-
+2)*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*J
_read_only_resource_inputs,
*(	
 !"#$%&'(*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_CNN_aug_deep_bnorm_layer_call_and_return_conditional_losses_23039442
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:???????????
"
_user_specified_name
input_29
?f
?
O__inference_CNN_aug_deep_bnorm_layer_call_and_return_conditional_losses_2303944

inputs
conv2d_130_2303845
conv2d_130_2303847"
batch_normalization_40_2303850"
batch_normalization_40_2303852"
batch_normalization_40_2303854"
batch_normalization_40_2303856
conv2d_131_2303859
conv2d_131_2303861"
batch_normalization_41_2303864"
batch_normalization_41_2303866"
batch_normalization_41_2303868"
batch_normalization_41_2303870
conv2d_132_2303874
conv2d_132_2303876"
batch_normalization_42_2303879"
batch_normalization_42_2303881"
batch_normalization_42_2303883"
batch_normalization_42_2303885
conv2d_133_2303888
conv2d_133_2303890"
batch_normalization_43_2303893"
batch_normalization_43_2303895"
batch_normalization_43_2303897"
batch_normalization_43_2303899
conv2d_134_2303903
conv2d_134_2303905"
batch_normalization_44_2303908"
batch_normalization_44_2303910"
batch_normalization_44_2303912"
batch_normalization_44_2303914
conv2d_135_2303917
conv2d_135_2303919"
batch_normalization_45_2303922"
batch_normalization_45_2303924"
batch_normalization_45_2303926"
batch_normalization_45_2303928
dense_56_2303933
dense_56_2303935
dense_57_2303938
dense_57_2303940
identity??.batch_normalization_40/StatefulPartitionedCall?.batch_normalization_41/StatefulPartitionedCall?.batch_normalization_42/StatefulPartitionedCall?.batch_normalization_43/StatefulPartitionedCall?.batch_normalization_44/StatefulPartitionedCall?.batch_normalization_45/StatefulPartitionedCall?"conv2d_130/StatefulPartitionedCall?"conv2d_131/StatefulPartitionedCall?"conv2d_132/StatefulPartitionedCall?"conv2d_133/StatefulPartitionedCall?"conv2d_134/StatefulPartitionedCall?"conv2d_135/StatefulPartitionedCall? dense_56/StatefulPartitionedCall? dense_57/StatefulPartitionedCall?
"conv2d_130/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_130_2303845conv2d_130_2303847*
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
G__inference_conv2d_130_layer_call_and_return_conditional_losses_23028892$
"conv2d_130/StatefulPartitionedCall?
.batch_normalization_40/StatefulPartitionedCallStatefulPartitionedCall+conv2d_130/StatefulPartitionedCall:output:0batch_normalization_40_2303850batch_normalization_40_2303852batch_normalization_40_2303854batch_normalization_40_2303856*
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
S__inference_batch_normalization_40_layer_call_and_return_conditional_losses_230294220
.batch_normalization_40/StatefulPartitionedCall?
"conv2d_131/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_40/StatefulPartitionedCall:output:0conv2d_131_2303859conv2d_131_2303861*
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
G__inference_conv2d_131_layer_call_and_return_conditional_losses_23029892$
"conv2d_131/StatefulPartitionedCall?
.batch_normalization_41/StatefulPartitionedCallStatefulPartitionedCall+conv2d_131/StatefulPartitionedCall:output:0batch_normalization_41_2303864batch_normalization_41_2303866batch_normalization_41_2303868batch_normalization_41_2303870*
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
S__inference_batch_normalization_41_layer_call_and_return_conditional_losses_230304220
.batch_normalization_41/StatefulPartitionedCall?
 max_pooling2d_52/PartitionedCallPartitionedCall7batch_normalization_41/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*** 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_max_pooling2d_52_layer_call_and_return_conditional_losses_23024282"
 max_pooling2d_52/PartitionedCall?
"conv2d_132/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_52/PartitionedCall:output:0conv2d_132_2303874conv2d_132_2303876*
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
G__inference_conv2d_132_layer_call_and_return_conditional_losses_23030902$
"conv2d_132/StatefulPartitionedCall?
.batch_normalization_42/StatefulPartitionedCallStatefulPartitionedCall+conv2d_132/StatefulPartitionedCall:output:0batch_normalization_42_2303879batch_normalization_42_2303881batch_normalization_42_2303883batch_normalization_42_2303885*
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
S__inference_batch_normalization_42_layer_call_and_return_conditional_losses_230314320
.batch_normalization_42/StatefulPartitionedCall?
"conv2d_133/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_42/StatefulPartitionedCall:output:0conv2d_133_2303888conv2d_133_2303890*
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
G__inference_conv2d_133_layer_call_and_return_conditional_losses_23031902$
"conv2d_133/StatefulPartitionedCall?
.batch_normalization_43/StatefulPartitionedCallStatefulPartitionedCall+conv2d_133/StatefulPartitionedCall:output:0batch_normalization_43_2303893batch_normalization_43_2303895batch_normalization_43_2303897batch_normalization_43_2303899*
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
S__inference_batch_normalization_43_layer_call_and_return_conditional_losses_230324320
.batch_normalization_43/StatefulPartitionedCall?
 max_pooling2d_53/PartitionedCallPartitionedCall7batch_normalization_43/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_max_pooling2d_53_layer_call_and_return_conditional_losses_23026482"
 max_pooling2d_53/PartitionedCall?
"conv2d_134/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_53/PartitionedCall:output:0conv2d_134_2303903conv2d_134_2303905*
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
G__inference_conv2d_134_layer_call_and_return_conditional_losses_23032912$
"conv2d_134/StatefulPartitionedCall?
.batch_normalization_44/StatefulPartitionedCallStatefulPartitionedCall+conv2d_134/StatefulPartitionedCall:output:0batch_normalization_44_2303908batch_normalization_44_2303910batch_normalization_44_2303912batch_normalization_44_2303914*
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
S__inference_batch_normalization_44_layer_call_and_return_conditional_losses_230334420
.batch_normalization_44/StatefulPartitionedCall?
"conv2d_135/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_44/StatefulPartitionedCall:output:0conv2d_135_2303917conv2d_135_2303919*
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
G__inference_conv2d_135_layer_call_and_return_conditional_losses_23033912$
"conv2d_135/StatefulPartitionedCall?
.batch_normalization_45/StatefulPartitionedCallStatefulPartitionedCall+conv2d_135/StatefulPartitionedCall:output:0batch_normalization_45_2303922batch_normalization_45_2303924batch_normalization_45_2303926batch_normalization_45_2303928*
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
S__inference_batch_normalization_45_layer_call_and_return_conditional_losses_230344420
.batch_normalization_45/StatefulPartitionedCall?
 max_pooling2d_54/PartitionedCallPartitionedCall7batch_normalization_45/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_max_pooling2d_54_layer_call_and_return_conditional_losses_23028682"
 max_pooling2d_54/PartitionedCall?
flatten_28/PartitionedCallPartitionedCall)max_pooling2d_54/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_flatten_28_layer_call_and_return_conditional_losses_23034872
flatten_28/PartitionedCall?
 dense_56/StatefulPartitionedCallStatefulPartitionedCall#flatten_28/PartitionedCall:output:0dense_56_2303933dense_56_2303935*
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
E__inference_dense_56_layer_call_and_return_conditional_losses_23035062"
 dense_56/StatefulPartitionedCall?
 dense_57/StatefulPartitionedCallStatefulPartitionedCall)dense_56/StatefulPartitionedCall:output:0dense_57_2303938dense_57_2303940*
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
E__inference_dense_57_layer_call_and_return_conditional_losses_23035332"
 dense_57/StatefulPartitionedCall?
IdentityIdentity)dense_57/StatefulPartitionedCall:output:0/^batch_normalization_40/StatefulPartitionedCall/^batch_normalization_41/StatefulPartitionedCall/^batch_normalization_42/StatefulPartitionedCall/^batch_normalization_43/StatefulPartitionedCall/^batch_normalization_44/StatefulPartitionedCall/^batch_normalization_45/StatefulPartitionedCall#^conv2d_130/StatefulPartitionedCall#^conv2d_131/StatefulPartitionedCall#^conv2d_132/StatefulPartitionedCall#^conv2d_133/StatefulPartitionedCall#^conv2d_134/StatefulPartitionedCall#^conv2d_135/StatefulPartitionedCall!^dense_56/StatefulPartitionedCall!^dense_57/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_40/StatefulPartitionedCall.batch_normalization_40/StatefulPartitionedCall2`
.batch_normalization_41/StatefulPartitionedCall.batch_normalization_41/StatefulPartitionedCall2`
.batch_normalization_42/StatefulPartitionedCall.batch_normalization_42/StatefulPartitionedCall2`
.batch_normalization_43/StatefulPartitionedCall.batch_normalization_43/StatefulPartitionedCall2`
.batch_normalization_44/StatefulPartitionedCall.batch_normalization_44/StatefulPartitionedCall2`
.batch_normalization_45/StatefulPartitionedCall.batch_normalization_45/StatefulPartitionedCall2H
"conv2d_130/StatefulPartitionedCall"conv2d_130/StatefulPartitionedCall2H
"conv2d_131/StatefulPartitionedCall"conv2d_131/StatefulPartitionedCall2H
"conv2d_132/StatefulPartitionedCall"conv2d_132/StatefulPartitionedCall2H
"conv2d_133/StatefulPartitionedCall"conv2d_133/StatefulPartitionedCall2H
"conv2d_134/StatefulPartitionedCall"conv2d_134/StatefulPartitionedCall2H
"conv2d_135/StatefulPartitionedCall"conv2d_135/StatefulPartitionedCall2D
 dense_56/StatefulPartitionedCall dense_56/StatefulPartitionedCall2D
 dense_57/StatefulPartitionedCall dense_57/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
,__inference_conv2d_132_layer_call_fn_2304918

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
G__inference_conv2d_132_layer_call_and_return_conditional_losses_23030902
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
?
H
,__inference_flatten_28_layer_call_fn_2305501

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_flatten_28_layer_call_and_return_conditional_losses_23034872
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_45_layer_call_fn_2305413

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
S__inference_batch_normalization_45_layer_call_and_return_conditional_losses_23034262
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
?
?
8__inference_batch_normalization_42_layer_call_fn_2304969

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
S__inference_batch_normalization_42_layer_call_and_return_conditional_losses_23024962
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
??
?<
#__inference__traced_restore_2306204
file_prefix&
"assignvariableop_conv2d_130_kernel&
"assignvariableop_1_conv2d_130_bias3
/assignvariableop_2_batch_normalization_40_gamma2
.assignvariableop_3_batch_normalization_40_beta9
5assignvariableop_4_batch_normalization_40_moving_mean=
9assignvariableop_5_batch_normalization_40_moving_variance(
$assignvariableop_6_conv2d_131_kernel&
"assignvariableop_7_conv2d_131_bias3
/assignvariableop_8_batch_normalization_41_gamma2
.assignvariableop_9_batch_normalization_41_beta:
6assignvariableop_10_batch_normalization_41_moving_mean>
:assignvariableop_11_batch_normalization_41_moving_variance)
%assignvariableop_12_conv2d_132_kernel'
#assignvariableop_13_conv2d_132_bias4
0assignvariableop_14_batch_normalization_42_gamma3
/assignvariableop_15_batch_normalization_42_beta:
6assignvariableop_16_batch_normalization_42_moving_mean>
:assignvariableop_17_batch_normalization_42_moving_variance)
%assignvariableop_18_conv2d_133_kernel'
#assignvariableop_19_conv2d_133_bias4
0assignvariableop_20_batch_normalization_43_gamma3
/assignvariableop_21_batch_normalization_43_beta:
6assignvariableop_22_batch_normalization_43_moving_mean>
:assignvariableop_23_batch_normalization_43_moving_variance)
%assignvariableop_24_conv2d_134_kernel'
#assignvariableop_25_conv2d_134_bias4
0assignvariableop_26_batch_normalization_44_gamma3
/assignvariableop_27_batch_normalization_44_beta:
6assignvariableop_28_batch_normalization_44_moving_mean>
:assignvariableop_29_batch_normalization_44_moving_variance)
%assignvariableop_30_conv2d_135_kernel'
#assignvariableop_31_conv2d_135_bias4
0assignvariableop_32_batch_normalization_45_gamma3
/assignvariableop_33_batch_normalization_45_beta:
6assignvariableop_34_batch_normalization_45_moving_mean>
:assignvariableop_35_batch_normalization_45_moving_variance'
#assignvariableop_36_dense_56_kernel%
!assignvariableop_37_dense_56_bias'
#assignvariableop_38_dense_57_kernel%
!assignvariableop_39_dense_57_bias!
assignvariableop_40_adam_iter#
assignvariableop_41_adam_beta_1#
assignvariableop_42_adam_beta_2"
assignvariableop_43_adam_decay*
&assignvariableop_44_adam_learning_rate
assignvariableop_45_total
assignvariableop_46_count
assignvariableop_47_total_1
assignvariableop_48_count_10
,assignvariableop_49_adam_conv2d_130_kernel_m.
*assignvariableop_50_adam_conv2d_130_bias_m;
7assignvariableop_51_adam_batch_normalization_40_gamma_m:
6assignvariableop_52_adam_batch_normalization_40_beta_m0
,assignvariableop_53_adam_conv2d_131_kernel_m.
*assignvariableop_54_adam_conv2d_131_bias_m;
7assignvariableop_55_adam_batch_normalization_41_gamma_m:
6assignvariableop_56_adam_batch_normalization_41_beta_m0
,assignvariableop_57_adam_conv2d_132_kernel_m.
*assignvariableop_58_adam_conv2d_132_bias_m;
7assignvariableop_59_adam_batch_normalization_42_gamma_m:
6assignvariableop_60_adam_batch_normalization_42_beta_m0
,assignvariableop_61_adam_conv2d_133_kernel_m.
*assignvariableop_62_adam_conv2d_133_bias_m;
7assignvariableop_63_adam_batch_normalization_43_gamma_m:
6assignvariableop_64_adam_batch_normalization_43_beta_m0
,assignvariableop_65_adam_conv2d_134_kernel_m.
*assignvariableop_66_adam_conv2d_134_bias_m;
7assignvariableop_67_adam_batch_normalization_44_gamma_m:
6assignvariableop_68_adam_batch_normalization_44_beta_m0
,assignvariableop_69_adam_conv2d_135_kernel_m.
*assignvariableop_70_adam_conv2d_135_bias_m;
7assignvariableop_71_adam_batch_normalization_45_gamma_m:
6assignvariableop_72_adam_batch_normalization_45_beta_m.
*assignvariableop_73_adam_dense_56_kernel_m,
(assignvariableop_74_adam_dense_56_bias_m.
*assignvariableop_75_adam_dense_57_kernel_m,
(assignvariableop_76_adam_dense_57_bias_m0
,assignvariableop_77_adam_conv2d_130_kernel_v.
*assignvariableop_78_adam_conv2d_130_bias_v;
7assignvariableop_79_adam_batch_normalization_40_gamma_v:
6assignvariableop_80_adam_batch_normalization_40_beta_v0
,assignvariableop_81_adam_conv2d_131_kernel_v.
*assignvariableop_82_adam_conv2d_131_bias_v;
7assignvariableop_83_adam_batch_normalization_41_gamma_v:
6assignvariableop_84_adam_batch_normalization_41_beta_v0
,assignvariableop_85_adam_conv2d_132_kernel_v.
*assignvariableop_86_adam_conv2d_132_bias_v;
7assignvariableop_87_adam_batch_normalization_42_gamma_v:
6assignvariableop_88_adam_batch_normalization_42_beta_v0
,assignvariableop_89_adam_conv2d_133_kernel_v.
*assignvariableop_90_adam_conv2d_133_bias_v;
7assignvariableop_91_adam_batch_normalization_43_gamma_v:
6assignvariableop_92_adam_batch_normalization_43_beta_v0
,assignvariableop_93_adam_conv2d_134_kernel_v.
*assignvariableop_94_adam_conv2d_134_bias_v;
7assignvariableop_95_adam_batch_normalization_44_gamma_v:
6assignvariableop_96_adam_batch_normalization_44_beta_v0
,assignvariableop_97_adam_conv2d_135_kernel_v.
*assignvariableop_98_adam_conv2d_135_bias_v;
7assignvariableop_99_adam_batch_normalization_45_gamma_v;
7assignvariableop_100_adam_batch_normalization_45_beta_v/
+assignvariableop_101_adam_dense_56_kernel_v-
)assignvariableop_102_adam_dense_56_bias_v/
+assignvariableop_103_adam_dense_57_kernel_v-
)assignvariableop_104_adam_dense_57_bias_v
identity_106??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_100?AssignVariableOp_101?AssignVariableOp_102?AssignVariableOp_103?AssignVariableOp_104?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_83?AssignVariableOp_84?AssignVariableOp_85?AssignVariableOp_86?AssignVariableOp_87?AssignVariableOp_88?AssignVariableOp_89?AssignVariableOp_9?AssignVariableOp_90?AssignVariableOp_91?AssignVariableOp_92?AssignVariableOp_93?AssignVariableOp_94?AssignVariableOp_95?AssignVariableOp_96?AssignVariableOp_97?AssignVariableOp_98?AssignVariableOp_99?;
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:j*
dtype0*?:
value?:B?:jB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:j*
dtype0*?
value?B?jB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*x
dtypesn
l2j	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp"assignvariableop_conv2d_130_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp"assignvariableop_1_conv2d_130_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp/assignvariableop_2_batch_normalization_40_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp.assignvariableop_3_batch_normalization_40_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp5assignvariableop_4_batch_normalization_40_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp9assignvariableop_5_batch_normalization_40_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp$assignvariableop_6_conv2d_131_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp"assignvariableop_7_conv2d_131_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp/assignvariableop_8_batch_normalization_41_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_41_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp6assignvariableop_10_batch_normalization_41_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_normalization_41_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp%assignvariableop_12_conv2d_132_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp#assignvariableop_13_conv2d_132_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp0assignvariableop_14_batch_normalization_42_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp/assignvariableop_15_batch_normalization_42_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp6assignvariableop_16_batch_normalization_42_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp:assignvariableop_17_batch_normalization_42_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp%assignvariableop_18_conv2d_133_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp#assignvariableop_19_conv2d_133_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp0assignvariableop_20_batch_normalization_43_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp/assignvariableop_21_batch_normalization_43_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp6assignvariableop_22_batch_normalization_43_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp:assignvariableop_23_batch_normalization_43_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp%assignvariableop_24_conv2d_134_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp#assignvariableop_25_conv2d_134_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp0assignvariableop_26_batch_normalization_44_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp/assignvariableop_27_batch_normalization_44_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp6assignvariableop_28_batch_normalization_44_moving_meanIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp:assignvariableop_29_batch_normalization_44_moving_varianceIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp%assignvariableop_30_conv2d_135_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp#assignvariableop_31_conv2d_135_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp0assignvariableop_32_batch_normalization_45_gammaIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp/assignvariableop_33_batch_normalization_45_betaIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp6assignvariableop_34_batch_normalization_45_moving_meanIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp:assignvariableop_35_batch_normalization_45_moving_varianceIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp#assignvariableop_36_dense_56_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp!assignvariableop_37_dense_56_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp#assignvariableop_38_dense_57_kernelIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp!assignvariableop_39_dense_57_biasIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOpassignvariableop_40_adam_iterIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOpassignvariableop_41_adam_beta_1Identity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOpassignvariableop_42_adam_beta_2Identity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOpassignvariableop_43_adam_decayIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp&assignvariableop_44_adam_learning_rateIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOpassignvariableop_45_totalIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOpassignvariableop_46_countIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOpassignvariableop_47_total_1Identity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOpassignvariableop_48_count_1Identity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp,assignvariableop_49_adam_conv2d_130_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp*assignvariableop_50_adam_conv2d_130_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp7assignvariableop_51_adam_batch_normalization_40_gamma_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp6assignvariableop_52_adam_batch_normalization_40_beta_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp,assignvariableop_53_adam_conv2d_131_kernel_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp*assignvariableop_54_adam_conv2d_131_bias_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp7assignvariableop_55_adam_batch_normalization_41_gamma_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp6assignvariableop_56_adam_batch_normalization_41_beta_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp,assignvariableop_57_adam_conv2d_132_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp*assignvariableop_58_adam_conv2d_132_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp7assignvariableop_59_adam_batch_normalization_42_gamma_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp6assignvariableop_60_adam_batch_normalization_42_beta_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp,assignvariableop_61_adam_conv2d_133_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp*assignvariableop_62_adam_conv2d_133_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp7assignvariableop_63_adam_batch_normalization_43_gamma_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp6assignvariableop_64_adam_batch_normalization_43_beta_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp,assignvariableop_65_adam_conv2d_134_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp*assignvariableop_66_adam_conv2d_134_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOp7assignvariableop_67_adam_batch_normalization_44_gamma_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp6assignvariableop_68_adam_batch_normalization_44_beta_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOp,assignvariableop_69_adam_conv2d_135_kernel_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp*assignvariableop_70_adam_conv2d_135_bias_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOp7assignvariableop_71_adam_batch_normalization_45_gamma_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOp6assignvariableop_72_adam_batch_normalization_45_beta_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73?
AssignVariableOp_73AssignVariableOp*assignvariableop_73_adam_dense_56_kernel_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOp(assignvariableop_74_adam_dense_56_bias_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75?
AssignVariableOp_75AssignVariableOp*assignvariableop_75_adam_dense_57_kernel_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76?
AssignVariableOp_76AssignVariableOp(assignvariableop_76_adam_dense_57_bias_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77?
AssignVariableOp_77AssignVariableOp,assignvariableop_77_adam_conv2d_130_kernel_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78?
AssignVariableOp_78AssignVariableOp*assignvariableop_78_adam_conv2d_130_bias_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79?
AssignVariableOp_79AssignVariableOp7assignvariableop_79_adam_batch_normalization_40_gamma_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80?
AssignVariableOp_80AssignVariableOp6assignvariableop_80_adam_batch_normalization_40_beta_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81?
AssignVariableOp_81AssignVariableOp,assignvariableop_81_adam_conv2d_131_kernel_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82?
AssignVariableOp_82AssignVariableOp*assignvariableop_82_adam_conv2d_131_bias_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83?
AssignVariableOp_83AssignVariableOp7assignvariableop_83_adam_batch_normalization_41_gamma_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84?
AssignVariableOp_84AssignVariableOp6assignvariableop_84_adam_batch_normalization_41_beta_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85?
AssignVariableOp_85AssignVariableOp,assignvariableop_85_adam_conv2d_132_kernel_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86?
AssignVariableOp_86AssignVariableOp*assignvariableop_86_adam_conv2d_132_bias_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87?
AssignVariableOp_87AssignVariableOp7assignvariableop_87_adam_batch_normalization_42_gamma_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88?
AssignVariableOp_88AssignVariableOp6assignvariableop_88_adam_batch_normalization_42_beta_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89?
AssignVariableOp_89AssignVariableOp,assignvariableop_89_adam_conv2d_133_kernel_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90?
AssignVariableOp_90AssignVariableOp*assignvariableop_90_adam_conv2d_133_bias_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91?
AssignVariableOp_91AssignVariableOp7assignvariableop_91_adam_batch_normalization_43_gamma_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92?
AssignVariableOp_92AssignVariableOp6assignvariableop_92_adam_batch_normalization_43_beta_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93?
AssignVariableOp_93AssignVariableOp,assignvariableop_93_adam_conv2d_134_kernel_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94?
AssignVariableOp_94AssignVariableOp*assignvariableop_94_adam_conv2d_134_bias_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95?
AssignVariableOp_95AssignVariableOp7assignvariableop_95_adam_batch_normalization_44_gamma_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96?
AssignVariableOp_96AssignVariableOp6assignvariableop_96_adam_batch_normalization_44_beta_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_96n
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:2
Identity_97?
AssignVariableOp_97AssignVariableOp,assignvariableop_97_adam_conv2d_135_kernel_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_97n
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:2
Identity_98?
AssignVariableOp_98AssignVariableOp*assignvariableop_98_adam_conv2d_135_bias_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_98n
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:2
Identity_99?
AssignVariableOp_99AssignVariableOp7assignvariableop_99_adam_batch_normalization_45_gamma_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99q
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:2
Identity_100?
AssignVariableOp_100AssignVariableOp7assignvariableop_100_adam_batch_normalization_45_beta_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_100q
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:2
Identity_101?
AssignVariableOp_101AssignVariableOp+assignvariableop_101_adam_dense_56_kernel_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_101q
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:2
Identity_102?
AssignVariableOp_102AssignVariableOp)assignvariableop_102_adam_dense_56_bias_vIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_102q
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:2
Identity_103?
AssignVariableOp_103AssignVariableOp+assignvariableop_103_adam_dense_57_kernel_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_103q
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:2
Identity_104?
AssignVariableOp_104AssignVariableOp)assignvariableop_104_adam_dense_57_bias_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1049
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_105Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_105?
Identity_106IdentityIdentity_105:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*
T0*
_output_shapes
: 2
Identity_106"%
identity_106Identity_106:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
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
,__inference_conv2d_131_layer_call_fn_2304770

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
G__inference_conv2d_131_layer_call_and_return_conditional_losses_23029892
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
?
?
S__inference_batch_normalization_43_layer_call_and_return_conditional_losses_2303243

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
S__inference_batch_normalization_44_layer_call_and_return_conditional_losses_2305298

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
8__inference_batch_normalization_40_layer_call_fn_2304686

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
S__inference_batch_normalization_40_layer_call_and_return_conditional_losses_23023072
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
8__inference_batch_normalization_41_layer_call_fn_2304821

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
S__inference_batch_normalization_41_layer_call_and_return_conditional_losses_23023802
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
?
i
M__inference_max_pooling2d_54_layer_call_and_return_conditional_losses_2302868

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
?

*__inference_dense_57_layer_call_fn_2305541

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
E__inference_dense_57_layer_call_and_return_conditional_losses_23035332
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
?
?
8__inference_batch_normalization_43_layer_call_fn_2305194

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
S__inference_batch_normalization_43_layer_call_and_return_conditional_losses_23032432
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
S__inference_batch_normalization_41_layer_call_and_return_conditional_losses_2304872

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
?
?
,__inference_conv2d_134_layer_call_fn_2305214

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
G__inference_conv2d_134_layer_call_and_return_conditional_losses_23032912
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
?

?
G__inference_conv2d_135_layer_call_and_return_conditional_losses_2305353

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
S__inference_batch_normalization_45_layer_call_and_return_conditional_losses_2305382

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
S__inference_batch_normalization_45_layer_call_and_return_conditional_losses_2303426

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
2__inference_max_pooling2d_54_layer_call_fn_2302874

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
M__inference_max_pooling2d_54_layer_call_and_return_conditional_losses_23028682
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
?
?
8__inference_batch_normalization_45_layer_call_fn_2305477

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
S__inference_batch_normalization_45_layer_call_and_return_conditional_losses_23028202
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
G__inference_conv2d_134_layer_call_and_return_conditional_losses_2305205

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
?

?
G__inference_conv2d_131_layer_call_and_return_conditional_losses_2304761

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
G__inference_flatten_28_layer_call_and_return_conditional_losses_2303487

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_44_layer_call_and_return_conditional_losses_2305234

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
?
?
S__inference_batch_normalization_42_layer_call_and_return_conditional_losses_2302527

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
N
2__inference_max_pooling2d_53_layer_call_fn_2302654

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
M__inference_max_pooling2d_53_layer_call_and_return_conditional_losses_23026482
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
?
?
8__inference_batch_normalization_43_layer_call_fn_2305117

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
S__inference_batch_normalization_43_layer_call_and_return_conditional_losses_23026002
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
G__inference_conv2d_135_layer_call_and_return_conditional_losses_2303391

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
?
?
S__inference_batch_normalization_45_layer_call_and_return_conditional_losses_2305400

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
?
?
S__inference_batch_normalization_40_layer_call_and_return_conditional_losses_2304706

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
8__inference_batch_normalization_41_layer_call_fn_2304834

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
S__inference_batch_normalization_41_layer_call_and_return_conditional_losses_23024112
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
?
?
S__inference_batch_normalization_43_layer_call_and_return_conditional_losses_2305168

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
?
?
S__inference_batch_normalization_40_layer_call_and_return_conditional_losses_2302307

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
G__inference_conv2d_132_layer_call_and_return_conditional_losses_2303090

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
S__inference_batch_normalization_44_layer_call_and_return_conditional_losses_2303326

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
?
?
,__inference_conv2d_133_layer_call_fn_2305066

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
G__inference_conv2d_133_layer_call_and_return_conditional_losses_23031902
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
?f
?
O__inference_CNN_aug_deep_bnorm_layer_call_and_return_conditional_losses_2303652
input_29
conv2d_130_2303553
conv2d_130_2303555"
batch_normalization_40_2303558"
batch_normalization_40_2303560"
batch_normalization_40_2303562"
batch_normalization_40_2303564
conv2d_131_2303567
conv2d_131_2303569"
batch_normalization_41_2303572"
batch_normalization_41_2303574"
batch_normalization_41_2303576"
batch_normalization_41_2303578
conv2d_132_2303582
conv2d_132_2303584"
batch_normalization_42_2303587"
batch_normalization_42_2303589"
batch_normalization_42_2303591"
batch_normalization_42_2303593
conv2d_133_2303596
conv2d_133_2303598"
batch_normalization_43_2303601"
batch_normalization_43_2303603"
batch_normalization_43_2303605"
batch_normalization_43_2303607
conv2d_134_2303611
conv2d_134_2303613"
batch_normalization_44_2303616"
batch_normalization_44_2303618"
batch_normalization_44_2303620"
batch_normalization_44_2303622
conv2d_135_2303625
conv2d_135_2303627"
batch_normalization_45_2303630"
batch_normalization_45_2303632"
batch_normalization_45_2303634"
batch_normalization_45_2303636
dense_56_2303641
dense_56_2303643
dense_57_2303646
dense_57_2303648
identity??.batch_normalization_40/StatefulPartitionedCall?.batch_normalization_41/StatefulPartitionedCall?.batch_normalization_42/StatefulPartitionedCall?.batch_normalization_43/StatefulPartitionedCall?.batch_normalization_44/StatefulPartitionedCall?.batch_normalization_45/StatefulPartitionedCall?"conv2d_130/StatefulPartitionedCall?"conv2d_131/StatefulPartitionedCall?"conv2d_132/StatefulPartitionedCall?"conv2d_133/StatefulPartitionedCall?"conv2d_134/StatefulPartitionedCall?"conv2d_135/StatefulPartitionedCall? dense_56/StatefulPartitionedCall? dense_57/StatefulPartitionedCall?
"conv2d_130/StatefulPartitionedCallStatefulPartitionedCallinput_29conv2d_130_2303553conv2d_130_2303555*
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
G__inference_conv2d_130_layer_call_and_return_conditional_losses_23028892$
"conv2d_130/StatefulPartitionedCall?
.batch_normalization_40/StatefulPartitionedCallStatefulPartitionedCall+conv2d_130/StatefulPartitionedCall:output:0batch_normalization_40_2303558batch_normalization_40_2303560batch_normalization_40_2303562batch_normalization_40_2303564*
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
S__inference_batch_normalization_40_layer_call_and_return_conditional_losses_230294220
.batch_normalization_40/StatefulPartitionedCall?
"conv2d_131/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_40/StatefulPartitionedCall:output:0conv2d_131_2303567conv2d_131_2303569*
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
G__inference_conv2d_131_layer_call_and_return_conditional_losses_23029892$
"conv2d_131/StatefulPartitionedCall?
.batch_normalization_41/StatefulPartitionedCallStatefulPartitionedCall+conv2d_131/StatefulPartitionedCall:output:0batch_normalization_41_2303572batch_normalization_41_2303574batch_normalization_41_2303576batch_normalization_41_2303578*
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
S__inference_batch_normalization_41_layer_call_and_return_conditional_losses_230304220
.batch_normalization_41/StatefulPartitionedCall?
 max_pooling2d_52/PartitionedCallPartitionedCall7batch_normalization_41/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*** 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_max_pooling2d_52_layer_call_and_return_conditional_losses_23024282"
 max_pooling2d_52/PartitionedCall?
"conv2d_132/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_52/PartitionedCall:output:0conv2d_132_2303582conv2d_132_2303584*
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
G__inference_conv2d_132_layer_call_and_return_conditional_losses_23030902$
"conv2d_132/StatefulPartitionedCall?
.batch_normalization_42/StatefulPartitionedCallStatefulPartitionedCall+conv2d_132/StatefulPartitionedCall:output:0batch_normalization_42_2303587batch_normalization_42_2303589batch_normalization_42_2303591batch_normalization_42_2303593*
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
S__inference_batch_normalization_42_layer_call_and_return_conditional_losses_230314320
.batch_normalization_42/StatefulPartitionedCall?
"conv2d_133/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_42/StatefulPartitionedCall:output:0conv2d_133_2303596conv2d_133_2303598*
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
G__inference_conv2d_133_layer_call_and_return_conditional_losses_23031902$
"conv2d_133/StatefulPartitionedCall?
.batch_normalization_43/StatefulPartitionedCallStatefulPartitionedCall+conv2d_133/StatefulPartitionedCall:output:0batch_normalization_43_2303601batch_normalization_43_2303603batch_normalization_43_2303605batch_normalization_43_2303607*
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
S__inference_batch_normalization_43_layer_call_and_return_conditional_losses_230324320
.batch_normalization_43/StatefulPartitionedCall?
 max_pooling2d_53/PartitionedCallPartitionedCall7batch_normalization_43/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_max_pooling2d_53_layer_call_and_return_conditional_losses_23026482"
 max_pooling2d_53/PartitionedCall?
"conv2d_134/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_53/PartitionedCall:output:0conv2d_134_2303611conv2d_134_2303613*
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
G__inference_conv2d_134_layer_call_and_return_conditional_losses_23032912$
"conv2d_134/StatefulPartitionedCall?
.batch_normalization_44/StatefulPartitionedCallStatefulPartitionedCall+conv2d_134/StatefulPartitionedCall:output:0batch_normalization_44_2303616batch_normalization_44_2303618batch_normalization_44_2303620batch_normalization_44_2303622*
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
S__inference_batch_normalization_44_layer_call_and_return_conditional_losses_230334420
.batch_normalization_44/StatefulPartitionedCall?
"conv2d_135/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_44/StatefulPartitionedCall:output:0conv2d_135_2303625conv2d_135_2303627*
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
G__inference_conv2d_135_layer_call_and_return_conditional_losses_23033912$
"conv2d_135/StatefulPartitionedCall?
.batch_normalization_45/StatefulPartitionedCallStatefulPartitionedCall+conv2d_135/StatefulPartitionedCall:output:0batch_normalization_45_2303630batch_normalization_45_2303632batch_normalization_45_2303634batch_normalization_45_2303636*
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
S__inference_batch_normalization_45_layer_call_and_return_conditional_losses_230344420
.batch_normalization_45/StatefulPartitionedCall?
 max_pooling2d_54/PartitionedCallPartitionedCall7batch_normalization_45/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_max_pooling2d_54_layer_call_and_return_conditional_losses_23028682"
 max_pooling2d_54/PartitionedCall?
flatten_28/PartitionedCallPartitionedCall)max_pooling2d_54/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_flatten_28_layer_call_and_return_conditional_losses_23034872
flatten_28/PartitionedCall?
 dense_56/StatefulPartitionedCallStatefulPartitionedCall#flatten_28/PartitionedCall:output:0dense_56_2303641dense_56_2303643*
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
E__inference_dense_56_layer_call_and_return_conditional_losses_23035062"
 dense_56/StatefulPartitionedCall?
 dense_57/StatefulPartitionedCallStatefulPartitionedCall)dense_56/StatefulPartitionedCall:output:0dense_57_2303646dense_57_2303648*
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
E__inference_dense_57_layer_call_and_return_conditional_losses_23035332"
 dense_57/StatefulPartitionedCall?
IdentityIdentity)dense_57/StatefulPartitionedCall:output:0/^batch_normalization_40/StatefulPartitionedCall/^batch_normalization_41/StatefulPartitionedCall/^batch_normalization_42/StatefulPartitionedCall/^batch_normalization_43/StatefulPartitionedCall/^batch_normalization_44/StatefulPartitionedCall/^batch_normalization_45/StatefulPartitionedCall#^conv2d_130/StatefulPartitionedCall#^conv2d_131/StatefulPartitionedCall#^conv2d_132/StatefulPartitionedCall#^conv2d_133/StatefulPartitionedCall#^conv2d_134/StatefulPartitionedCall#^conv2d_135/StatefulPartitionedCall!^dense_56/StatefulPartitionedCall!^dense_57/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_40/StatefulPartitionedCall.batch_normalization_40/StatefulPartitionedCall2`
.batch_normalization_41/StatefulPartitionedCall.batch_normalization_41/StatefulPartitionedCall2`
.batch_normalization_42/StatefulPartitionedCall.batch_normalization_42/StatefulPartitionedCall2`
.batch_normalization_43/StatefulPartitionedCall.batch_normalization_43/StatefulPartitionedCall2`
.batch_normalization_44/StatefulPartitionedCall.batch_normalization_44/StatefulPartitionedCall2`
.batch_normalization_45/StatefulPartitionedCall.batch_normalization_45/StatefulPartitionedCall2H
"conv2d_130/StatefulPartitionedCall"conv2d_130/StatefulPartitionedCall2H
"conv2d_131/StatefulPartitionedCall"conv2d_131/StatefulPartitionedCall2H
"conv2d_132/StatefulPartitionedCall"conv2d_132/StatefulPartitionedCall2H
"conv2d_133/StatefulPartitionedCall"conv2d_133/StatefulPartitionedCall2H
"conv2d_134/StatefulPartitionedCall"conv2d_134/StatefulPartitionedCall2H
"conv2d_135/StatefulPartitionedCall"conv2d_135/StatefulPartitionedCall2D
 dense_56/StatefulPartitionedCall dense_56/StatefulPartitionedCall2D
 dense_57/StatefulPartitionedCall dense_57/StatefulPartitionedCall:[ W
1
_output_shapes
:???????????
"
_user_specified_name
input_29
?

?
G__inference_conv2d_134_layer_call_and_return_conditional_losses_2303291

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
S__inference_batch_normalization_40_layer_call_and_return_conditional_losses_2302924

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
?

?
G__inference_conv2d_133_layer_call_and_return_conditional_losses_2303190

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
?

?
G__inference_conv2d_133_layer_call_and_return_conditional_losses_2305057

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
S__inference_batch_normalization_41_layer_call_and_return_conditional_losses_2304854

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
?

?
G__inference_conv2d_130_layer_call_and_return_conditional_losses_2304613

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
8__inference_batch_normalization_43_layer_call_fn_2305130

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
S__inference_batch_normalization_43_layer_call_and_return_conditional_losses_23026312
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
?
i
M__inference_max_pooling2d_52_layer_call_and_return_conditional_losses_2302428

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
?
?
S__inference_batch_normalization_41_layer_call_and_return_conditional_losses_2304808

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
??
?+
"__inference__wrapped_model_2302214
input_29@
<cnn_aug_deep_bnorm_conv2d_130_conv2d_readvariableop_resourceA
=cnn_aug_deep_bnorm_conv2d_130_biasadd_readvariableop_resourceE
Acnn_aug_deep_bnorm_batch_normalization_40_readvariableop_resourceG
Ccnn_aug_deep_bnorm_batch_normalization_40_readvariableop_1_resourceV
Rcnn_aug_deep_bnorm_batch_normalization_40_fusedbatchnormv3_readvariableop_resourceX
Tcnn_aug_deep_bnorm_batch_normalization_40_fusedbatchnormv3_readvariableop_1_resource@
<cnn_aug_deep_bnorm_conv2d_131_conv2d_readvariableop_resourceA
=cnn_aug_deep_bnorm_conv2d_131_biasadd_readvariableop_resourceE
Acnn_aug_deep_bnorm_batch_normalization_41_readvariableop_resourceG
Ccnn_aug_deep_bnorm_batch_normalization_41_readvariableop_1_resourceV
Rcnn_aug_deep_bnorm_batch_normalization_41_fusedbatchnormv3_readvariableop_resourceX
Tcnn_aug_deep_bnorm_batch_normalization_41_fusedbatchnormv3_readvariableop_1_resource@
<cnn_aug_deep_bnorm_conv2d_132_conv2d_readvariableop_resourceA
=cnn_aug_deep_bnorm_conv2d_132_biasadd_readvariableop_resourceE
Acnn_aug_deep_bnorm_batch_normalization_42_readvariableop_resourceG
Ccnn_aug_deep_bnorm_batch_normalization_42_readvariableop_1_resourceV
Rcnn_aug_deep_bnorm_batch_normalization_42_fusedbatchnormv3_readvariableop_resourceX
Tcnn_aug_deep_bnorm_batch_normalization_42_fusedbatchnormv3_readvariableop_1_resource@
<cnn_aug_deep_bnorm_conv2d_133_conv2d_readvariableop_resourceA
=cnn_aug_deep_bnorm_conv2d_133_biasadd_readvariableop_resourceE
Acnn_aug_deep_bnorm_batch_normalization_43_readvariableop_resourceG
Ccnn_aug_deep_bnorm_batch_normalization_43_readvariableop_1_resourceV
Rcnn_aug_deep_bnorm_batch_normalization_43_fusedbatchnormv3_readvariableop_resourceX
Tcnn_aug_deep_bnorm_batch_normalization_43_fusedbatchnormv3_readvariableop_1_resource@
<cnn_aug_deep_bnorm_conv2d_134_conv2d_readvariableop_resourceA
=cnn_aug_deep_bnorm_conv2d_134_biasadd_readvariableop_resourceE
Acnn_aug_deep_bnorm_batch_normalization_44_readvariableop_resourceG
Ccnn_aug_deep_bnorm_batch_normalization_44_readvariableop_1_resourceV
Rcnn_aug_deep_bnorm_batch_normalization_44_fusedbatchnormv3_readvariableop_resourceX
Tcnn_aug_deep_bnorm_batch_normalization_44_fusedbatchnormv3_readvariableop_1_resource@
<cnn_aug_deep_bnorm_conv2d_135_conv2d_readvariableop_resourceA
=cnn_aug_deep_bnorm_conv2d_135_biasadd_readvariableop_resourceE
Acnn_aug_deep_bnorm_batch_normalization_45_readvariableop_resourceG
Ccnn_aug_deep_bnorm_batch_normalization_45_readvariableop_1_resourceV
Rcnn_aug_deep_bnorm_batch_normalization_45_fusedbatchnormv3_readvariableop_resourceX
Tcnn_aug_deep_bnorm_batch_normalization_45_fusedbatchnormv3_readvariableop_1_resource>
:cnn_aug_deep_bnorm_dense_56_matmul_readvariableop_resource?
;cnn_aug_deep_bnorm_dense_56_biasadd_readvariableop_resource>
:cnn_aug_deep_bnorm_dense_57_matmul_readvariableop_resource?
;cnn_aug_deep_bnorm_dense_57_biasadd_readvariableop_resource
identity??ICNN_aug_deep_bnorm/batch_normalization_40/FusedBatchNormV3/ReadVariableOp?KCNN_aug_deep_bnorm/batch_normalization_40/FusedBatchNormV3/ReadVariableOp_1?8CNN_aug_deep_bnorm/batch_normalization_40/ReadVariableOp?:CNN_aug_deep_bnorm/batch_normalization_40/ReadVariableOp_1?ICNN_aug_deep_bnorm/batch_normalization_41/FusedBatchNormV3/ReadVariableOp?KCNN_aug_deep_bnorm/batch_normalization_41/FusedBatchNormV3/ReadVariableOp_1?8CNN_aug_deep_bnorm/batch_normalization_41/ReadVariableOp?:CNN_aug_deep_bnorm/batch_normalization_41/ReadVariableOp_1?ICNN_aug_deep_bnorm/batch_normalization_42/FusedBatchNormV3/ReadVariableOp?KCNN_aug_deep_bnorm/batch_normalization_42/FusedBatchNormV3/ReadVariableOp_1?8CNN_aug_deep_bnorm/batch_normalization_42/ReadVariableOp?:CNN_aug_deep_bnorm/batch_normalization_42/ReadVariableOp_1?ICNN_aug_deep_bnorm/batch_normalization_43/FusedBatchNormV3/ReadVariableOp?KCNN_aug_deep_bnorm/batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1?8CNN_aug_deep_bnorm/batch_normalization_43/ReadVariableOp?:CNN_aug_deep_bnorm/batch_normalization_43/ReadVariableOp_1?ICNN_aug_deep_bnorm/batch_normalization_44/FusedBatchNormV3/ReadVariableOp?KCNN_aug_deep_bnorm/batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1?8CNN_aug_deep_bnorm/batch_normalization_44/ReadVariableOp?:CNN_aug_deep_bnorm/batch_normalization_44/ReadVariableOp_1?ICNN_aug_deep_bnorm/batch_normalization_45/FusedBatchNormV3/ReadVariableOp?KCNN_aug_deep_bnorm/batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1?8CNN_aug_deep_bnorm/batch_normalization_45/ReadVariableOp?:CNN_aug_deep_bnorm/batch_normalization_45/ReadVariableOp_1?4CNN_aug_deep_bnorm/conv2d_130/BiasAdd/ReadVariableOp?3CNN_aug_deep_bnorm/conv2d_130/Conv2D/ReadVariableOp?4CNN_aug_deep_bnorm/conv2d_131/BiasAdd/ReadVariableOp?3CNN_aug_deep_bnorm/conv2d_131/Conv2D/ReadVariableOp?4CNN_aug_deep_bnorm/conv2d_132/BiasAdd/ReadVariableOp?3CNN_aug_deep_bnorm/conv2d_132/Conv2D/ReadVariableOp?4CNN_aug_deep_bnorm/conv2d_133/BiasAdd/ReadVariableOp?3CNN_aug_deep_bnorm/conv2d_133/Conv2D/ReadVariableOp?4CNN_aug_deep_bnorm/conv2d_134/BiasAdd/ReadVariableOp?3CNN_aug_deep_bnorm/conv2d_134/Conv2D/ReadVariableOp?4CNN_aug_deep_bnorm/conv2d_135/BiasAdd/ReadVariableOp?3CNN_aug_deep_bnorm/conv2d_135/Conv2D/ReadVariableOp?2CNN_aug_deep_bnorm/dense_56/BiasAdd/ReadVariableOp?1CNN_aug_deep_bnorm/dense_56/MatMul/ReadVariableOp?2CNN_aug_deep_bnorm/dense_57/BiasAdd/ReadVariableOp?1CNN_aug_deep_bnorm/dense_57/MatMul/ReadVariableOp?
3CNN_aug_deep_bnorm/conv2d_130/Conv2D/ReadVariableOpReadVariableOp<cnn_aug_deep_bnorm_conv2d_130_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype025
3CNN_aug_deep_bnorm/conv2d_130/Conv2D/ReadVariableOp?
$CNN_aug_deep_bnorm/conv2d_130/Conv2DConv2Dinput_29;CNN_aug_deep_bnorm/conv2d_130/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2&
$CNN_aug_deep_bnorm/conv2d_130/Conv2D?
4CNN_aug_deep_bnorm/conv2d_130/BiasAdd/ReadVariableOpReadVariableOp=cnn_aug_deep_bnorm_conv2d_130_biasadd_readvariableop_resource*
_output_shapes
:*
dtype026
4CNN_aug_deep_bnorm/conv2d_130/BiasAdd/ReadVariableOp?
%CNN_aug_deep_bnorm/conv2d_130/BiasAddBiasAdd-CNN_aug_deep_bnorm/conv2d_130/Conv2D:output:0<CNN_aug_deep_bnorm/conv2d_130/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2'
%CNN_aug_deep_bnorm/conv2d_130/BiasAdd?
"CNN_aug_deep_bnorm/conv2d_130/ReluRelu.CNN_aug_deep_bnorm/conv2d_130/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2$
"CNN_aug_deep_bnorm/conv2d_130/Relu?
8CNN_aug_deep_bnorm/batch_normalization_40/ReadVariableOpReadVariableOpAcnn_aug_deep_bnorm_batch_normalization_40_readvariableop_resource*
_output_shapes
:*
dtype02:
8CNN_aug_deep_bnorm/batch_normalization_40/ReadVariableOp?
:CNN_aug_deep_bnorm/batch_normalization_40/ReadVariableOp_1ReadVariableOpCcnn_aug_deep_bnorm_batch_normalization_40_readvariableop_1_resource*
_output_shapes
:*
dtype02<
:CNN_aug_deep_bnorm/batch_normalization_40/ReadVariableOp_1?
ICNN_aug_deep_bnorm/batch_normalization_40/FusedBatchNormV3/ReadVariableOpReadVariableOpRcnn_aug_deep_bnorm_batch_normalization_40_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02K
ICNN_aug_deep_bnorm/batch_normalization_40/FusedBatchNormV3/ReadVariableOp?
KCNN_aug_deep_bnorm/batch_normalization_40/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpTcnn_aug_deep_bnorm_batch_normalization_40_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02M
KCNN_aug_deep_bnorm/batch_normalization_40/FusedBatchNormV3/ReadVariableOp_1?
:CNN_aug_deep_bnorm/batch_normalization_40/FusedBatchNormV3FusedBatchNormV30CNN_aug_deep_bnorm/conv2d_130/Relu:activations:0@CNN_aug_deep_bnorm/batch_normalization_40/ReadVariableOp:value:0BCNN_aug_deep_bnorm/batch_normalization_40/ReadVariableOp_1:value:0QCNN_aug_deep_bnorm/batch_normalization_40/FusedBatchNormV3/ReadVariableOp:value:0SCNN_aug_deep_bnorm/batch_normalization_40/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2<
:CNN_aug_deep_bnorm/batch_normalization_40/FusedBatchNormV3?
3CNN_aug_deep_bnorm/conv2d_131/Conv2D/ReadVariableOpReadVariableOp<cnn_aug_deep_bnorm_conv2d_131_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype025
3CNN_aug_deep_bnorm/conv2d_131/Conv2D/ReadVariableOp?
$CNN_aug_deep_bnorm/conv2d_131/Conv2DConv2D>CNN_aug_deep_bnorm/batch_normalization_40/FusedBatchNormV3:y:0;CNN_aug_deep_bnorm/conv2d_131/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2&
$CNN_aug_deep_bnorm/conv2d_131/Conv2D?
4CNN_aug_deep_bnorm/conv2d_131/BiasAdd/ReadVariableOpReadVariableOp=cnn_aug_deep_bnorm_conv2d_131_biasadd_readvariableop_resource*
_output_shapes
:*
dtype026
4CNN_aug_deep_bnorm/conv2d_131/BiasAdd/ReadVariableOp?
%CNN_aug_deep_bnorm/conv2d_131/BiasAddBiasAdd-CNN_aug_deep_bnorm/conv2d_131/Conv2D:output:0<CNN_aug_deep_bnorm/conv2d_131/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2'
%CNN_aug_deep_bnorm/conv2d_131/BiasAdd?
"CNN_aug_deep_bnorm/conv2d_131/ReluRelu.CNN_aug_deep_bnorm/conv2d_131/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2$
"CNN_aug_deep_bnorm/conv2d_131/Relu?
8CNN_aug_deep_bnorm/batch_normalization_41/ReadVariableOpReadVariableOpAcnn_aug_deep_bnorm_batch_normalization_41_readvariableop_resource*
_output_shapes
:*
dtype02:
8CNN_aug_deep_bnorm/batch_normalization_41/ReadVariableOp?
:CNN_aug_deep_bnorm/batch_normalization_41/ReadVariableOp_1ReadVariableOpCcnn_aug_deep_bnorm_batch_normalization_41_readvariableop_1_resource*
_output_shapes
:*
dtype02<
:CNN_aug_deep_bnorm/batch_normalization_41/ReadVariableOp_1?
ICNN_aug_deep_bnorm/batch_normalization_41/FusedBatchNormV3/ReadVariableOpReadVariableOpRcnn_aug_deep_bnorm_batch_normalization_41_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02K
ICNN_aug_deep_bnorm/batch_normalization_41/FusedBatchNormV3/ReadVariableOp?
KCNN_aug_deep_bnorm/batch_normalization_41/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpTcnn_aug_deep_bnorm_batch_normalization_41_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02M
KCNN_aug_deep_bnorm/batch_normalization_41/FusedBatchNormV3/ReadVariableOp_1?
:CNN_aug_deep_bnorm/batch_normalization_41/FusedBatchNormV3FusedBatchNormV30CNN_aug_deep_bnorm/conv2d_131/Relu:activations:0@CNN_aug_deep_bnorm/batch_normalization_41/ReadVariableOp:value:0BCNN_aug_deep_bnorm/batch_normalization_41/ReadVariableOp_1:value:0QCNN_aug_deep_bnorm/batch_normalization_41/FusedBatchNormV3/ReadVariableOp:value:0SCNN_aug_deep_bnorm/batch_normalization_41/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2<
:CNN_aug_deep_bnorm/batch_normalization_41/FusedBatchNormV3?
+CNN_aug_deep_bnorm/max_pooling2d_52/MaxPoolMaxPool>CNN_aug_deep_bnorm/batch_normalization_41/FusedBatchNormV3:y:0*/
_output_shapes
:?????????***
ksize
*
paddingVALID*
strides
2-
+CNN_aug_deep_bnorm/max_pooling2d_52/MaxPool?
3CNN_aug_deep_bnorm/conv2d_132/Conv2D/ReadVariableOpReadVariableOp<cnn_aug_deep_bnorm_conv2d_132_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype025
3CNN_aug_deep_bnorm/conv2d_132/Conv2D/ReadVariableOp?
$CNN_aug_deep_bnorm/conv2d_132/Conv2DConv2D4CNN_aug_deep_bnorm/max_pooling2d_52/MaxPool:output:0;CNN_aug_deep_bnorm/conv2d_132/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????***
paddingSAME*
strides
2&
$CNN_aug_deep_bnorm/conv2d_132/Conv2D?
4CNN_aug_deep_bnorm/conv2d_132/BiasAdd/ReadVariableOpReadVariableOp=cnn_aug_deep_bnorm_conv2d_132_biasadd_readvariableop_resource*
_output_shapes
:*
dtype026
4CNN_aug_deep_bnorm/conv2d_132/BiasAdd/ReadVariableOp?
%CNN_aug_deep_bnorm/conv2d_132/BiasAddBiasAdd-CNN_aug_deep_bnorm/conv2d_132/Conv2D:output:0<CNN_aug_deep_bnorm/conv2d_132/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**2'
%CNN_aug_deep_bnorm/conv2d_132/BiasAdd?
"CNN_aug_deep_bnorm/conv2d_132/ReluRelu.CNN_aug_deep_bnorm/conv2d_132/BiasAdd:output:0*
T0*/
_output_shapes
:?????????**2$
"CNN_aug_deep_bnorm/conv2d_132/Relu?
8CNN_aug_deep_bnorm/batch_normalization_42/ReadVariableOpReadVariableOpAcnn_aug_deep_bnorm_batch_normalization_42_readvariableop_resource*
_output_shapes
:*
dtype02:
8CNN_aug_deep_bnorm/batch_normalization_42/ReadVariableOp?
:CNN_aug_deep_bnorm/batch_normalization_42/ReadVariableOp_1ReadVariableOpCcnn_aug_deep_bnorm_batch_normalization_42_readvariableop_1_resource*
_output_shapes
:*
dtype02<
:CNN_aug_deep_bnorm/batch_normalization_42/ReadVariableOp_1?
ICNN_aug_deep_bnorm/batch_normalization_42/FusedBatchNormV3/ReadVariableOpReadVariableOpRcnn_aug_deep_bnorm_batch_normalization_42_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02K
ICNN_aug_deep_bnorm/batch_normalization_42/FusedBatchNormV3/ReadVariableOp?
KCNN_aug_deep_bnorm/batch_normalization_42/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpTcnn_aug_deep_bnorm_batch_normalization_42_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02M
KCNN_aug_deep_bnorm/batch_normalization_42/FusedBatchNormV3/ReadVariableOp_1?
:CNN_aug_deep_bnorm/batch_normalization_42/FusedBatchNormV3FusedBatchNormV30CNN_aug_deep_bnorm/conv2d_132/Relu:activations:0@CNN_aug_deep_bnorm/batch_normalization_42/ReadVariableOp:value:0BCNN_aug_deep_bnorm/batch_normalization_42/ReadVariableOp_1:value:0QCNN_aug_deep_bnorm/batch_normalization_42/FusedBatchNormV3/ReadVariableOp:value:0SCNN_aug_deep_bnorm/batch_normalization_42/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????**:::::*
epsilon%o?:*
is_training( 2<
:CNN_aug_deep_bnorm/batch_normalization_42/FusedBatchNormV3?
3CNN_aug_deep_bnorm/conv2d_133/Conv2D/ReadVariableOpReadVariableOp<cnn_aug_deep_bnorm_conv2d_133_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype025
3CNN_aug_deep_bnorm/conv2d_133/Conv2D/ReadVariableOp?
$CNN_aug_deep_bnorm/conv2d_133/Conv2DConv2D>CNN_aug_deep_bnorm/batch_normalization_42/FusedBatchNormV3:y:0;CNN_aug_deep_bnorm/conv2d_133/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????***
paddingSAME*
strides
2&
$CNN_aug_deep_bnorm/conv2d_133/Conv2D?
4CNN_aug_deep_bnorm/conv2d_133/BiasAdd/ReadVariableOpReadVariableOp=cnn_aug_deep_bnorm_conv2d_133_biasadd_readvariableop_resource*
_output_shapes
:*
dtype026
4CNN_aug_deep_bnorm/conv2d_133/BiasAdd/ReadVariableOp?
%CNN_aug_deep_bnorm/conv2d_133/BiasAddBiasAdd-CNN_aug_deep_bnorm/conv2d_133/Conv2D:output:0<CNN_aug_deep_bnorm/conv2d_133/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**2'
%CNN_aug_deep_bnorm/conv2d_133/BiasAdd?
"CNN_aug_deep_bnorm/conv2d_133/ReluRelu.CNN_aug_deep_bnorm/conv2d_133/BiasAdd:output:0*
T0*/
_output_shapes
:?????????**2$
"CNN_aug_deep_bnorm/conv2d_133/Relu?
8CNN_aug_deep_bnorm/batch_normalization_43/ReadVariableOpReadVariableOpAcnn_aug_deep_bnorm_batch_normalization_43_readvariableop_resource*
_output_shapes
:*
dtype02:
8CNN_aug_deep_bnorm/batch_normalization_43/ReadVariableOp?
:CNN_aug_deep_bnorm/batch_normalization_43/ReadVariableOp_1ReadVariableOpCcnn_aug_deep_bnorm_batch_normalization_43_readvariableop_1_resource*
_output_shapes
:*
dtype02<
:CNN_aug_deep_bnorm/batch_normalization_43/ReadVariableOp_1?
ICNN_aug_deep_bnorm/batch_normalization_43/FusedBatchNormV3/ReadVariableOpReadVariableOpRcnn_aug_deep_bnorm_batch_normalization_43_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02K
ICNN_aug_deep_bnorm/batch_normalization_43/FusedBatchNormV3/ReadVariableOp?
KCNN_aug_deep_bnorm/batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpTcnn_aug_deep_bnorm_batch_normalization_43_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02M
KCNN_aug_deep_bnorm/batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1?
:CNN_aug_deep_bnorm/batch_normalization_43/FusedBatchNormV3FusedBatchNormV30CNN_aug_deep_bnorm/conv2d_133/Relu:activations:0@CNN_aug_deep_bnorm/batch_normalization_43/ReadVariableOp:value:0BCNN_aug_deep_bnorm/batch_normalization_43/ReadVariableOp_1:value:0QCNN_aug_deep_bnorm/batch_normalization_43/FusedBatchNormV3/ReadVariableOp:value:0SCNN_aug_deep_bnorm/batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????**:::::*
epsilon%o?:*
is_training( 2<
:CNN_aug_deep_bnorm/batch_normalization_43/FusedBatchNormV3?
+CNN_aug_deep_bnorm/max_pooling2d_53/MaxPoolMaxPool>CNN_aug_deep_bnorm/batch_normalization_43/FusedBatchNormV3:y:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2-
+CNN_aug_deep_bnorm/max_pooling2d_53/MaxPool?
3CNN_aug_deep_bnorm/conv2d_134/Conv2D/ReadVariableOpReadVariableOp<cnn_aug_deep_bnorm_conv2d_134_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype025
3CNN_aug_deep_bnorm/conv2d_134/Conv2D/ReadVariableOp?
$CNN_aug_deep_bnorm/conv2d_134/Conv2DConv2D4CNN_aug_deep_bnorm/max_pooling2d_53/MaxPool:output:0;CNN_aug_deep_bnorm/conv2d_134/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2&
$CNN_aug_deep_bnorm/conv2d_134/Conv2D?
4CNN_aug_deep_bnorm/conv2d_134/BiasAdd/ReadVariableOpReadVariableOp=cnn_aug_deep_bnorm_conv2d_134_biasadd_readvariableop_resource*
_output_shapes
:*
dtype026
4CNN_aug_deep_bnorm/conv2d_134/BiasAdd/ReadVariableOp?
%CNN_aug_deep_bnorm/conv2d_134/BiasAddBiasAdd-CNN_aug_deep_bnorm/conv2d_134/Conv2D:output:0<CNN_aug_deep_bnorm/conv2d_134/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2'
%CNN_aug_deep_bnorm/conv2d_134/BiasAdd?
"CNN_aug_deep_bnorm/conv2d_134/ReluRelu.CNN_aug_deep_bnorm/conv2d_134/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2$
"CNN_aug_deep_bnorm/conv2d_134/Relu?
8CNN_aug_deep_bnorm/batch_normalization_44/ReadVariableOpReadVariableOpAcnn_aug_deep_bnorm_batch_normalization_44_readvariableop_resource*
_output_shapes
:*
dtype02:
8CNN_aug_deep_bnorm/batch_normalization_44/ReadVariableOp?
:CNN_aug_deep_bnorm/batch_normalization_44/ReadVariableOp_1ReadVariableOpCcnn_aug_deep_bnorm_batch_normalization_44_readvariableop_1_resource*
_output_shapes
:*
dtype02<
:CNN_aug_deep_bnorm/batch_normalization_44/ReadVariableOp_1?
ICNN_aug_deep_bnorm/batch_normalization_44/FusedBatchNormV3/ReadVariableOpReadVariableOpRcnn_aug_deep_bnorm_batch_normalization_44_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02K
ICNN_aug_deep_bnorm/batch_normalization_44/FusedBatchNormV3/ReadVariableOp?
KCNN_aug_deep_bnorm/batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpTcnn_aug_deep_bnorm_batch_normalization_44_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02M
KCNN_aug_deep_bnorm/batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1?
:CNN_aug_deep_bnorm/batch_normalization_44/FusedBatchNormV3FusedBatchNormV30CNN_aug_deep_bnorm/conv2d_134/Relu:activations:0@CNN_aug_deep_bnorm/batch_normalization_44/ReadVariableOp:value:0BCNN_aug_deep_bnorm/batch_normalization_44/ReadVariableOp_1:value:0QCNN_aug_deep_bnorm/batch_normalization_44/FusedBatchNormV3/ReadVariableOp:value:0SCNN_aug_deep_bnorm/batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( 2<
:CNN_aug_deep_bnorm/batch_normalization_44/FusedBatchNormV3?
3CNN_aug_deep_bnorm/conv2d_135/Conv2D/ReadVariableOpReadVariableOp<cnn_aug_deep_bnorm_conv2d_135_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype025
3CNN_aug_deep_bnorm/conv2d_135/Conv2D/ReadVariableOp?
$CNN_aug_deep_bnorm/conv2d_135/Conv2DConv2D>CNN_aug_deep_bnorm/batch_normalization_44/FusedBatchNormV3:y:0;CNN_aug_deep_bnorm/conv2d_135/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2&
$CNN_aug_deep_bnorm/conv2d_135/Conv2D?
4CNN_aug_deep_bnorm/conv2d_135/BiasAdd/ReadVariableOpReadVariableOp=cnn_aug_deep_bnorm_conv2d_135_biasadd_readvariableop_resource*
_output_shapes
:*
dtype026
4CNN_aug_deep_bnorm/conv2d_135/BiasAdd/ReadVariableOp?
%CNN_aug_deep_bnorm/conv2d_135/BiasAddBiasAdd-CNN_aug_deep_bnorm/conv2d_135/Conv2D:output:0<CNN_aug_deep_bnorm/conv2d_135/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2'
%CNN_aug_deep_bnorm/conv2d_135/BiasAdd?
"CNN_aug_deep_bnorm/conv2d_135/ReluRelu.CNN_aug_deep_bnorm/conv2d_135/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2$
"CNN_aug_deep_bnorm/conv2d_135/Relu?
8CNN_aug_deep_bnorm/batch_normalization_45/ReadVariableOpReadVariableOpAcnn_aug_deep_bnorm_batch_normalization_45_readvariableop_resource*
_output_shapes
:*
dtype02:
8CNN_aug_deep_bnorm/batch_normalization_45/ReadVariableOp?
:CNN_aug_deep_bnorm/batch_normalization_45/ReadVariableOp_1ReadVariableOpCcnn_aug_deep_bnorm_batch_normalization_45_readvariableop_1_resource*
_output_shapes
:*
dtype02<
:CNN_aug_deep_bnorm/batch_normalization_45/ReadVariableOp_1?
ICNN_aug_deep_bnorm/batch_normalization_45/FusedBatchNormV3/ReadVariableOpReadVariableOpRcnn_aug_deep_bnorm_batch_normalization_45_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02K
ICNN_aug_deep_bnorm/batch_normalization_45/FusedBatchNormV3/ReadVariableOp?
KCNN_aug_deep_bnorm/batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpTcnn_aug_deep_bnorm_batch_normalization_45_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02M
KCNN_aug_deep_bnorm/batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1?
:CNN_aug_deep_bnorm/batch_normalization_45/FusedBatchNormV3FusedBatchNormV30CNN_aug_deep_bnorm/conv2d_135/Relu:activations:0@CNN_aug_deep_bnorm/batch_normalization_45/ReadVariableOp:value:0BCNN_aug_deep_bnorm/batch_normalization_45/ReadVariableOp_1:value:0QCNN_aug_deep_bnorm/batch_normalization_45/FusedBatchNormV3/ReadVariableOp:value:0SCNN_aug_deep_bnorm/batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( 2<
:CNN_aug_deep_bnorm/batch_normalization_45/FusedBatchNormV3?
+CNN_aug_deep_bnorm/max_pooling2d_54/MaxPoolMaxPool>CNN_aug_deep_bnorm/batch_normalization_45/FusedBatchNormV3:y:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2-
+CNN_aug_deep_bnorm/max_pooling2d_54/MaxPool?
#CNN_aug_deep_bnorm/flatten_28/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   2%
#CNN_aug_deep_bnorm/flatten_28/Const?
%CNN_aug_deep_bnorm/flatten_28/ReshapeReshape4CNN_aug_deep_bnorm/max_pooling2d_54/MaxPool:output:0,CNN_aug_deep_bnorm/flatten_28/Const:output:0*
T0*(
_output_shapes
:??????????2'
%CNN_aug_deep_bnorm/flatten_28/Reshape?
1CNN_aug_deep_bnorm/dense_56/MatMul/ReadVariableOpReadVariableOp:cnn_aug_deep_bnorm_dense_56_matmul_readvariableop_resource*
_output_shapes
:	? *
dtype023
1CNN_aug_deep_bnorm/dense_56/MatMul/ReadVariableOp?
"CNN_aug_deep_bnorm/dense_56/MatMulMatMul.CNN_aug_deep_bnorm/flatten_28/Reshape:output:09CNN_aug_deep_bnorm/dense_56/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2$
"CNN_aug_deep_bnorm/dense_56/MatMul?
2CNN_aug_deep_bnorm/dense_56/BiasAdd/ReadVariableOpReadVariableOp;cnn_aug_deep_bnorm_dense_56_biasadd_readvariableop_resource*
_output_shapes
: *
dtype024
2CNN_aug_deep_bnorm/dense_56/BiasAdd/ReadVariableOp?
#CNN_aug_deep_bnorm/dense_56/BiasAddBiasAdd,CNN_aug_deep_bnorm/dense_56/MatMul:product:0:CNN_aug_deep_bnorm/dense_56/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2%
#CNN_aug_deep_bnorm/dense_56/BiasAdd?
 CNN_aug_deep_bnorm/dense_56/ReluRelu,CNN_aug_deep_bnorm/dense_56/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2"
 CNN_aug_deep_bnorm/dense_56/Relu?
1CNN_aug_deep_bnorm/dense_57/MatMul/ReadVariableOpReadVariableOp:cnn_aug_deep_bnorm_dense_57_matmul_readvariableop_resource*
_output_shapes

: *
dtype023
1CNN_aug_deep_bnorm/dense_57/MatMul/ReadVariableOp?
"CNN_aug_deep_bnorm/dense_57/MatMulMatMul.CNN_aug_deep_bnorm/dense_56/Relu:activations:09CNN_aug_deep_bnorm/dense_57/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2$
"CNN_aug_deep_bnorm/dense_57/MatMul?
2CNN_aug_deep_bnorm/dense_57/BiasAdd/ReadVariableOpReadVariableOp;cnn_aug_deep_bnorm_dense_57_biasadd_readvariableop_resource*
_output_shapes
:*
dtype024
2CNN_aug_deep_bnorm/dense_57/BiasAdd/ReadVariableOp?
#CNN_aug_deep_bnorm/dense_57/BiasAddBiasAdd,CNN_aug_deep_bnorm/dense_57/MatMul:product:0:CNN_aug_deep_bnorm/dense_57/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2%
#CNN_aug_deep_bnorm/dense_57/BiasAdd?
#CNN_aug_deep_bnorm/dense_57/SoftmaxSoftmax,CNN_aug_deep_bnorm/dense_57/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2%
#CNN_aug_deep_bnorm/dense_57/Softmax?
IdentityIdentity-CNN_aug_deep_bnorm/dense_57/Softmax:softmax:0J^CNN_aug_deep_bnorm/batch_normalization_40/FusedBatchNormV3/ReadVariableOpL^CNN_aug_deep_bnorm/batch_normalization_40/FusedBatchNormV3/ReadVariableOp_19^CNN_aug_deep_bnorm/batch_normalization_40/ReadVariableOp;^CNN_aug_deep_bnorm/batch_normalization_40/ReadVariableOp_1J^CNN_aug_deep_bnorm/batch_normalization_41/FusedBatchNormV3/ReadVariableOpL^CNN_aug_deep_bnorm/batch_normalization_41/FusedBatchNormV3/ReadVariableOp_19^CNN_aug_deep_bnorm/batch_normalization_41/ReadVariableOp;^CNN_aug_deep_bnorm/batch_normalization_41/ReadVariableOp_1J^CNN_aug_deep_bnorm/batch_normalization_42/FusedBatchNormV3/ReadVariableOpL^CNN_aug_deep_bnorm/batch_normalization_42/FusedBatchNormV3/ReadVariableOp_19^CNN_aug_deep_bnorm/batch_normalization_42/ReadVariableOp;^CNN_aug_deep_bnorm/batch_normalization_42/ReadVariableOp_1J^CNN_aug_deep_bnorm/batch_normalization_43/FusedBatchNormV3/ReadVariableOpL^CNN_aug_deep_bnorm/batch_normalization_43/FusedBatchNormV3/ReadVariableOp_19^CNN_aug_deep_bnorm/batch_normalization_43/ReadVariableOp;^CNN_aug_deep_bnorm/batch_normalization_43/ReadVariableOp_1J^CNN_aug_deep_bnorm/batch_normalization_44/FusedBatchNormV3/ReadVariableOpL^CNN_aug_deep_bnorm/batch_normalization_44/FusedBatchNormV3/ReadVariableOp_19^CNN_aug_deep_bnorm/batch_normalization_44/ReadVariableOp;^CNN_aug_deep_bnorm/batch_normalization_44/ReadVariableOp_1J^CNN_aug_deep_bnorm/batch_normalization_45/FusedBatchNormV3/ReadVariableOpL^CNN_aug_deep_bnorm/batch_normalization_45/FusedBatchNormV3/ReadVariableOp_19^CNN_aug_deep_bnorm/batch_normalization_45/ReadVariableOp;^CNN_aug_deep_bnorm/batch_normalization_45/ReadVariableOp_15^CNN_aug_deep_bnorm/conv2d_130/BiasAdd/ReadVariableOp4^CNN_aug_deep_bnorm/conv2d_130/Conv2D/ReadVariableOp5^CNN_aug_deep_bnorm/conv2d_131/BiasAdd/ReadVariableOp4^CNN_aug_deep_bnorm/conv2d_131/Conv2D/ReadVariableOp5^CNN_aug_deep_bnorm/conv2d_132/BiasAdd/ReadVariableOp4^CNN_aug_deep_bnorm/conv2d_132/Conv2D/ReadVariableOp5^CNN_aug_deep_bnorm/conv2d_133/BiasAdd/ReadVariableOp4^CNN_aug_deep_bnorm/conv2d_133/Conv2D/ReadVariableOp5^CNN_aug_deep_bnorm/conv2d_134/BiasAdd/ReadVariableOp4^CNN_aug_deep_bnorm/conv2d_134/Conv2D/ReadVariableOp5^CNN_aug_deep_bnorm/conv2d_135/BiasAdd/ReadVariableOp4^CNN_aug_deep_bnorm/conv2d_135/Conv2D/ReadVariableOp3^CNN_aug_deep_bnorm/dense_56/BiasAdd/ReadVariableOp2^CNN_aug_deep_bnorm/dense_56/MatMul/ReadVariableOp3^CNN_aug_deep_bnorm/dense_57/BiasAdd/ReadVariableOp2^CNN_aug_deep_bnorm/dense_57/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::2?
ICNN_aug_deep_bnorm/batch_normalization_40/FusedBatchNormV3/ReadVariableOpICNN_aug_deep_bnorm/batch_normalization_40/FusedBatchNormV3/ReadVariableOp2?
KCNN_aug_deep_bnorm/batch_normalization_40/FusedBatchNormV3/ReadVariableOp_1KCNN_aug_deep_bnorm/batch_normalization_40/FusedBatchNormV3/ReadVariableOp_12t
8CNN_aug_deep_bnorm/batch_normalization_40/ReadVariableOp8CNN_aug_deep_bnorm/batch_normalization_40/ReadVariableOp2x
:CNN_aug_deep_bnorm/batch_normalization_40/ReadVariableOp_1:CNN_aug_deep_bnorm/batch_normalization_40/ReadVariableOp_12?
ICNN_aug_deep_bnorm/batch_normalization_41/FusedBatchNormV3/ReadVariableOpICNN_aug_deep_bnorm/batch_normalization_41/FusedBatchNormV3/ReadVariableOp2?
KCNN_aug_deep_bnorm/batch_normalization_41/FusedBatchNormV3/ReadVariableOp_1KCNN_aug_deep_bnorm/batch_normalization_41/FusedBatchNormV3/ReadVariableOp_12t
8CNN_aug_deep_bnorm/batch_normalization_41/ReadVariableOp8CNN_aug_deep_bnorm/batch_normalization_41/ReadVariableOp2x
:CNN_aug_deep_bnorm/batch_normalization_41/ReadVariableOp_1:CNN_aug_deep_bnorm/batch_normalization_41/ReadVariableOp_12?
ICNN_aug_deep_bnorm/batch_normalization_42/FusedBatchNormV3/ReadVariableOpICNN_aug_deep_bnorm/batch_normalization_42/FusedBatchNormV3/ReadVariableOp2?
KCNN_aug_deep_bnorm/batch_normalization_42/FusedBatchNormV3/ReadVariableOp_1KCNN_aug_deep_bnorm/batch_normalization_42/FusedBatchNormV3/ReadVariableOp_12t
8CNN_aug_deep_bnorm/batch_normalization_42/ReadVariableOp8CNN_aug_deep_bnorm/batch_normalization_42/ReadVariableOp2x
:CNN_aug_deep_bnorm/batch_normalization_42/ReadVariableOp_1:CNN_aug_deep_bnorm/batch_normalization_42/ReadVariableOp_12?
ICNN_aug_deep_bnorm/batch_normalization_43/FusedBatchNormV3/ReadVariableOpICNN_aug_deep_bnorm/batch_normalization_43/FusedBatchNormV3/ReadVariableOp2?
KCNN_aug_deep_bnorm/batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1KCNN_aug_deep_bnorm/batch_normalization_43/FusedBatchNormV3/ReadVariableOp_12t
8CNN_aug_deep_bnorm/batch_normalization_43/ReadVariableOp8CNN_aug_deep_bnorm/batch_normalization_43/ReadVariableOp2x
:CNN_aug_deep_bnorm/batch_normalization_43/ReadVariableOp_1:CNN_aug_deep_bnorm/batch_normalization_43/ReadVariableOp_12?
ICNN_aug_deep_bnorm/batch_normalization_44/FusedBatchNormV3/ReadVariableOpICNN_aug_deep_bnorm/batch_normalization_44/FusedBatchNormV3/ReadVariableOp2?
KCNN_aug_deep_bnorm/batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1KCNN_aug_deep_bnorm/batch_normalization_44/FusedBatchNormV3/ReadVariableOp_12t
8CNN_aug_deep_bnorm/batch_normalization_44/ReadVariableOp8CNN_aug_deep_bnorm/batch_normalization_44/ReadVariableOp2x
:CNN_aug_deep_bnorm/batch_normalization_44/ReadVariableOp_1:CNN_aug_deep_bnorm/batch_normalization_44/ReadVariableOp_12?
ICNN_aug_deep_bnorm/batch_normalization_45/FusedBatchNormV3/ReadVariableOpICNN_aug_deep_bnorm/batch_normalization_45/FusedBatchNormV3/ReadVariableOp2?
KCNN_aug_deep_bnorm/batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1KCNN_aug_deep_bnorm/batch_normalization_45/FusedBatchNormV3/ReadVariableOp_12t
8CNN_aug_deep_bnorm/batch_normalization_45/ReadVariableOp8CNN_aug_deep_bnorm/batch_normalization_45/ReadVariableOp2x
:CNN_aug_deep_bnorm/batch_normalization_45/ReadVariableOp_1:CNN_aug_deep_bnorm/batch_normalization_45/ReadVariableOp_12l
4CNN_aug_deep_bnorm/conv2d_130/BiasAdd/ReadVariableOp4CNN_aug_deep_bnorm/conv2d_130/BiasAdd/ReadVariableOp2j
3CNN_aug_deep_bnorm/conv2d_130/Conv2D/ReadVariableOp3CNN_aug_deep_bnorm/conv2d_130/Conv2D/ReadVariableOp2l
4CNN_aug_deep_bnorm/conv2d_131/BiasAdd/ReadVariableOp4CNN_aug_deep_bnorm/conv2d_131/BiasAdd/ReadVariableOp2j
3CNN_aug_deep_bnorm/conv2d_131/Conv2D/ReadVariableOp3CNN_aug_deep_bnorm/conv2d_131/Conv2D/ReadVariableOp2l
4CNN_aug_deep_bnorm/conv2d_132/BiasAdd/ReadVariableOp4CNN_aug_deep_bnorm/conv2d_132/BiasAdd/ReadVariableOp2j
3CNN_aug_deep_bnorm/conv2d_132/Conv2D/ReadVariableOp3CNN_aug_deep_bnorm/conv2d_132/Conv2D/ReadVariableOp2l
4CNN_aug_deep_bnorm/conv2d_133/BiasAdd/ReadVariableOp4CNN_aug_deep_bnorm/conv2d_133/BiasAdd/ReadVariableOp2j
3CNN_aug_deep_bnorm/conv2d_133/Conv2D/ReadVariableOp3CNN_aug_deep_bnorm/conv2d_133/Conv2D/ReadVariableOp2l
4CNN_aug_deep_bnorm/conv2d_134/BiasAdd/ReadVariableOp4CNN_aug_deep_bnorm/conv2d_134/BiasAdd/ReadVariableOp2j
3CNN_aug_deep_bnorm/conv2d_134/Conv2D/ReadVariableOp3CNN_aug_deep_bnorm/conv2d_134/Conv2D/ReadVariableOp2l
4CNN_aug_deep_bnorm/conv2d_135/BiasAdd/ReadVariableOp4CNN_aug_deep_bnorm/conv2d_135/BiasAdd/ReadVariableOp2j
3CNN_aug_deep_bnorm/conv2d_135/Conv2D/ReadVariableOp3CNN_aug_deep_bnorm/conv2d_135/Conv2D/ReadVariableOp2h
2CNN_aug_deep_bnorm/dense_56/BiasAdd/ReadVariableOp2CNN_aug_deep_bnorm/dense_56/BiasAdd/ReadVariableOp2f
1CNN_aug_deep_bnorm/dense_56/MatMul/ReadVariableOp1CNN_aug_deep_bnorm/dense_56/MatMul/ReadVariableOp2h
2CNN_aug_deep_bnorm/dense_57/BiasAdd/ReadVariableOp2CNN_aug_deep_bnorm/dense_57/BiasAdd/ReadVariableOp2f
1CNN_aug_deep_bnorm/dense_57/MatMul/ReadVariableOp1CNN_aug_deep_bnorm/dense_57/MatMul/ReadVariableOp:[ W
1
_output_shapes
:???????????
"
_user_specified_name
input_29
?
?
8__inference_batch_normalization_42_layer_call_fn_2304982

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
S__inference_batch_normalization_42_layer_call_and_return_conditional_losses_23025272
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
8__inference_batch_normalization_45_layer_call_fn_2305490

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
S__inference_batch_normalization_45_layer_call_and_return_conditional_losses_23028512
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
S__inference_batch_normalization_42_layer_call_and_return_conditional_losses_2302496

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
?e
?
O__inference_CNN_aug_deep_bnorm_layer_call_and_return_conditional_losses_2303550
input_29
conv2d_130_2302900
conv2d_130_2302902"
batch_normalization_40_2302969"
batch_normalization_40_2302971"
batch_normalization_40_2302973"
batch_normalization_40_2302975
conv2d_131_2303000
conv2d_131_2303002"
batch_normalization_41_2303069"
batch_normalization_41_2303071"
batch_normalization_41_2303073"
batch_normalization_41_2303075
conv2d_132_2303101
conv2d_132_2303103"
batch_normalization_42_2303170"
batch_normalization_42_2303172"
batch_normalization_42_2303174"
batch_normalization_42_2303176
conv2d_133_2303201
conv2d_133_2303203"
batch_normalization_43_2303270"
batch_normalization_43_2303272"
batch_normalization_43_2303274"
batch_normalization_43_2303276
conv2d_134_2303302
conv2d_134_2303304"
batch_normalization_44_2303371"
batch_normalization_44_2303373"
batch_normalization_44_2303375"
batch_normalization_44_2303377
conv2d_135_2303402
conv2d_135_2303404"
batch_normalization_45_2303471"
batch_normalization_45_2303473"
batch_normalization_45_2303475"
batch_normalization_45_2303477
dense_56_2303517
dense_56_2303519
dense_57_2303544
dense_57_2303546
identity??.batch_normalization_40/StatefulPartitionedCall?.batch_normalization_41/StatefulPartitionedCall?.batch_normalization_42/StatefulPartitionedCall?.batch_normalization_43/StatefulPartitionedCall?.batch_normalization_44/StatefulPartitionedCall?.batch_normalization_45/StatefulPartitionedCall?"conv2d_130/StatefulPartitionedCall?"conv2d_131/StatefulPartitionedCall?"conv2d_132/StatefulPartitionedCall?"conv2d_133/StatefulPartitionedCall?"conv2d_134/StatefulPartitionedCall?"conv2d_135/StatefulPartitionedCall? dense_56/StatefulPartitionedCall? dense_57/StatefulPartitionedCall?
"conv2d_130/StatefulPartitionedCallStatefulPartitionedCallinput_29conv2d_130_2302900conv2d_130_2302902*
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
G__inference_conv2d_130_layer_call_and_return_conditional_losses_23028892$
"conv2d_130/StatefulPartitionedCall?
.batch_normalization_40/StatefulPartitionedCallStatefulPartitionedCall+conv2d_130/StatefulPartitionedCall:output:0batch_normalization_40_2302969batch_normalization_40_2302971batch_normalization_40_2302973batch_normalization_40_2302975*
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
S__inference_batch_normalization_40_layer_call_and_return_conditional_losses_230292420
.batch_normalization_40/StatefulPartitionedCall?
"conv2d_131/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_40/StatefulPartitionedCall:output:0conv2d_131_2303000conv2d_131_2303002*
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
G__inference_conv2d_131_layer_call_and_return_conditional_losses_23029892$
"conv2d_131/StatefulPartitionedCall?
.batch_normalization_41/StatefulPartitionedCallStatefulPartitionedCall+conv2d_131/StatefulPartitionedCall:output:0batch_normalization_41_2303069batch_normalization_41_2303071batch_normalization_41_2303073batch_normalization_41_2303075*
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
S__inference_batch_normalization_41_layer_call_and_return_conditional_losses_230302420
.batch_normalization_41/StatefulPartitionedCall?
 max_pooling2d_52/PartitionedCallPartitionedCall7batch_normalization_41/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*** 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_max_pooling2d_52_layer_call_and_return_conditional_losses_23024282"
 max_pooling2d_52/PartitionedCall?
"conv2d_132/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_52/PartitionedCall:output:0conv2d_132_2303101conv2d_132_2303103*
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
G__inference_conv2d_132_layer_call_and_return_conditional_losses_23030902$
"conv2d_132/StatefulPartitionedCall?
.batch_normalization_42/StatefulPartitionedCallStatefulPartitionedCall+conv2d_132/StatefulPartitionedCall:output:0batch_normalization_42_2303170batch_normalization_42_2303172batch_normalization_42_2303174batch_normalization_42_2303176*
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
S__inference_batch_normalization_42_layer_call_and_return_conditional_losses_230312520
.batch_normalization_42/StatefulPartitionedCall?
"conv2d_133/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_42/StatefulPartitionedCall:output:0conv2d_133_2303201conv2d_133_2303203*
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
G__inference_conv2d_133_layer_call_and_return_conditional_losses_23031902$
"conv2d_133/StatefulPartitionedCall?
.batch_normalization_43/StatefulPartitionedCallStatefulPartitionedCall+conv2d_133/StatefulPartitionedCall:output:0batch_normalization_43_2303270batch_normalization_43_2303272batch_normalization_43_2303274batch_normalization_43_2303276*
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
S__inference_batch_normalization_43_layer_call_and_return_conditional_losses_230322520
.batch_normalization_43/StatefulPartitionedCall?
 max_pooling2d_53/PartitionedCallPartitionedCall7batch_normalization_43/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_max_pooling2d_53_layer_call_and_return_conditional_losses_23026482"
 max_pooling2d_53/PartitionedCall?
"conv2d_134/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_53/PartitionedCall:output:0conv2d_134_2303302conv2d_134_2303304*
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
G__inference_conv2d_134_layer_call_and_return_conditional_losses_23032912$
"conv2d_134/StatefulPartitionedCall?
.batch_normalization_44/StatefulPartitionedCallStatefulPartitionedCall+conv2d_134/StatefulPartitionedCall:output:0batch_normalization_44_2303371batch_normalization_44_2303373batch_normalization_44_2303375batch_normalization_44_2303377*
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
S__inference_batch_normalization_44_layer_call_and_return_conditional_losses_230332620
.batch_normalization_44/StatefulPartitionedCall?
"conv2d_135/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_44/StatefulPartitionedCall:output:0conv2d_135_2303402conv2d_135_2303404*
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
G__inference_conv2d_135_layer_call_and_return_conditional_losses_23033912$
"conv2d_135/StatefulPartitionedCall?
.batch_normalization_45/StatefulPartitionedCallStatefulPartitionedCall+conv2d_135/StatefulPartitionedCall:output:0batch_normalization_45_2303471batch_normalization_45_2303473batch_normalization_45_2303475batch_normalization_45_2303477*
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
S__inference_batch_normalization_45_layer_call_and_return_conditional_losses_230342620
.batch_normalization_45/StatefulPartitionedCall?
 max_pooling2d_54/PartitionedCallPartitionedCall7batch_normalization_45/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_max_pooling2d_54_layer_call_and_return_conditional_losses_23028682"
 max_pooling2d_54/PartitionedCall?
flatten_28/PartitionedCallPartitionedCall)max_pooling2d_54/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_flatten_28_layer_call_and_return_conditional_losses_23034872
flatten_28/PartitionedCall?
 dense_56/StatefulPartitionedCallStatefulPartitionedCall#flatten_28/PartitionedCall:output:0dense_56_2303517dense_56_2303519*
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
E__inference_dense_56_layer_call_and_return_conditional_losses_23035062"
 dense_56/StatefulPartitionedCall?
 dense_57/StatefulPartitionedCallStatefulPartitionedCall)dense_56/StatefulPartitionedCall:output:0dense_57_2303544dense_57_2303546*
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
E__inference_dense_57_layer_call_and_return_conditional_losses_23035332"
 dense_57/StatefulPartitionedCall?
IdentityIdentity)dense_57/StatefulPartitionedCall:output:0/^batch_normalization_40/StatefulPartitionedCall/^batch_normalization_41/StatefulPartitionedCall/^batch_normalization_42/StatefulPartitionedCall/^batch_normalization_43/StatefulPartitionedCall/^batch_normalization_44/StatefulPartitionedCall/^batch_normalization_45/StatefulPartitionedCall#^conv2d_130/StatefulPartitionedCall#^conv2d_131/StatefulPartitionedCall#^conv2d_132/StatefulPartitionedCall#^conv2d_133/StatefulPartitionedCall#^conv2d_134/StatefulPartitionedCall#^conv2d_135/StatefulPartitionedCall!^dense_56/StatefulPartitionedCall!^dense_57/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_40/StatefulPartitionedCall.batch_normalization_40/StatefulPartitionedCall2`
.batch_normalization_41/StatefulPartitionedCall.batch_normalization_41/StatefulPartitionedCall2`
.batch_normalization_42/StatefulPartitionedCall.batch_normalization_42/StatefulPartitionedCall2`
.batch_normalization_43/StatefulPartitionedCall.batch_normalization_43/StatefulPartitionedCall2`
.batch_normalization_44/StatefulPartitionedCall.batch_normalization_44/StatefulPartitionedCall2`
.batch_normalization_45/StatefulPartitionedCall.batch_normalization_45/StatefulPartitionedCall2H
"conv2d_130/StatefulPartitionedCall"conv2d_130/StatefulPartitionedCall2H
"conv2d_131/StatefulPartitionedCall"conv2d_131/StatefulPartitionedCall2H
"conv2d_132/StatefulPartitionedCall"conv2d_132/StatefulPartitionedCall2H
"conv2d_133/StatefulPartitionedCall"conv2d_133/StatefulPartitionedCall2H
"conv2d_134/StatefulPartitionedCall"conv2d_134/StatefulPartitionedCall2H
"conv2d_135/StatefulPartitionedCall"conv2d_135/StatefulPartitionedCall2D
 dense_56/StatefulPartitionedCall dense_56/StatefulPartitionedCall2D
 dense_57/StatefulPartitionedCall dense_57/StatefulPartitionedCall:[ W
1
_output_shapes
:???????????
"
_user_specified_name
input_29
?
?
4__inference_CNN_aug_deep_bnorm_layer_call_fn_2304602

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

unknown_38
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
unknown_38*4
Tin-
+2)*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*J
_read_only_resource_inputs,
*(	
 !"#$%&'(*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_CNN_aug_deep_bnorm_layer_call_and_return_conditional_losses_23039442
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
G__inference_conv2d_132_layer_call_and_return_conditional_losses_2304909

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
S__inference_batch_normalization_41_layer_call_and_return_conditional_losses_2303042

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
?
?
8__inference_batch_normalization_41_layer_call_fn_2304885

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
S__inference_batch_normalization_41_layer_call_and_return_conditional_losses_23030242
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
?
?
S__inference_batch_normalization_41_layer_call_and_return_conditional_losses_2303024

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
S__inference_batch_normalization_45_layer_call_and_return_conditional_losses_2302851

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
S__inference_batch_normalization_41_layer_call_and_return_conditional_losses_2302380

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
S__inference_batch_normalization_40_layer_call_and_return_conditional_losses_2304724

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
?
?
S__inference_batch_normalization_43_layer_call_and_return_conditional_losses_2305104

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
S__inference_batch_normalization_44_layer_call_and_return_conditional_losses_2303344

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
?
?
S__inference_batch_normalization_43_layer_call_and_return_conditional_losses_2305150

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
?
i
M__inference_max_pooling2d_53_layer_call_and_return_conditional_losses_2302648

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
?
?
,__inference_conv2d_130_layer_call_fn_2304622

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
G__inference_conv2d_130_layer_call_and_return_conditional_losses_23028892
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
8__inference_batch_normalization_40_layer_call_fn_2304673

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
S__inference_batch_normalization_40_layer_call_and_return_conditional_losses_23022762
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
??
?
O__inference_CNN_aug_deep_bnorm_layer_call_and_return_conditional_losses_2304432

inputs-
)conv2d_130_conv2d_readvariableop_resource.
*conv2d_130_biasadd_readvariableop_resource2
.batch_normalization_40_readvariableop_resource4
0batch_normalization_40_readvariableop_1_resourceC
?batch_normalization_40_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_40_fusedbatchnormv3_readvariableop_1_resource-
)conv2d_131_conv2d_readvariableop_resource.
*conv2d_131_biasadd_readvariableop_resource2
.batch_normalization_41_readvariableop_resource4
0batch_normalization_41_readvariableop_1_resourceC
?batch_normalization_41_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_41_fusedbatchnormv3_readvariableop_1_resource-
)conv2d_132_conv2d_readvariableop_resource.
*conv2d_132_biasadd_readvariableop_resource2
.batch_normalization_42_readvariableop_resource4
0batch_normalization_42_readvariableop_1_resourceC
?batch_normalization_42_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_42_fusedbatchnormv3_readvariableop_1_resource-
)conv2d_133_conv2d_readvariableop_resource.
*conv2d_133_biasadd_readvariableop_resource2
.batch_normalization_43_readvariableop_resource4
0batch_normalization_43_readvariableop_1_resourceC
?batch_normalization_43_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_43_fusedbatchnormv3_readvariableop_1_resource-
)conv2d_134_conv2d_readvariableop_resource.
*conv2d_134_biasadd_readvariableop_resource2
.batch_normalization_44_readvariableop_resource4
0batch_normalization_44_readvariableop_1_resourceC
?batch_normalization_44_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_44_fusedbatchnormv3_readvariableop_1_resource-
)conv2d_135_conv2d_readvariableop_resource.
*conv2d_135_biasadd_readvariableop_resource2
.batch_normalization_45_readvariableop_resource4
0batch_normalization_45_readvariableop_1_resourceC
?batch_normalization_45_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_45_fusedbatchnormv3_readvariableop_1_resource+
'dense_56_matmul_readvariableop_resource,
(dense_56_biasadd_readvariableop_resource+
'dense_57_matmul_readvariableop_resource,
(dense_57_biasadd_readvariableop_resource
identity??6batch_normalization_40/FusedBatchNormV3/ReadVariableOp?8batch_normalization_40/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_40/ReadVariableOp?'batch_normalization_40/ReadVariableOp_1?6batch_normalization_41/FusedBatchNormV3/ReadVariableOp?8batch_normalization_41/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_41/ReadVariableOp?'batch_normalization_41/ReadVariableOp_1?6batch_normalization_42/FusedBatchNormV3/ReadVariableOp?8batch_normalization_42/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_42/ReadVariableOp?'batch_normalization_42/ReadVariableOp_1?6batch_normalization_43/FusedBatchNormV3/ReadVariableOp?8batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_43/ReadVariableOp?'batch_normalization_43/ReadVariableOp_1?6batch_normalization_44/FusedBatchNormV3/ReadVariableOp?8batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_44/ReadVariableOp?'batch_normalization_44/ReadVariableOp_1?6batch_normalization_45/FusedBatchNormV3/ReadVariableOp?8batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_45/ReadVariableOp?'batch_normalization_45/ReadVariableOp_1?!conv2d_130/BiasAdd/ReadVariableOp? conv2d_130/Conv2D/ReadVariableOp?!conv2d_131/BiasAdd/ReadVariableOp? conv2d_131/Conv2D/ReadVariableOp?!conv2d_132/BiasAdd/ReadVariableOp? conv2d_132/Conv2D/ReadVariableOp?!conv2d_133/BiasAdd/ReadVariableOp? conv2d_133/Conv2D/ReadVariableOp?!conv2d_134/BiasAdd/ReadVariableOp? conv2d_134/Conv2D/ReadVariableOp?!conv2d_135/BiasAdd/ReadVariableOp? conv2d_135/Conv2D/ReadVariableOp?dense_56/BiasAdd/ReadVariableOp?dense_56/MatMul/ReadVariableOp?dense_57/BiasAdd/ReadVariableOp?dense_57/MatMul/ReadVariableOp?
 conv2d_130/Conv2D/ReadVariableOpReadVariableOp)conv2d_130_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_130/Conv2D/ReadVariableOp?
conv2d_130/Conv2DConv2Dinputs(conv2d_130/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_130/Conv2D?
!conv2d_130/BiasAdd/ReadVariableOpReadVariableOp*conv2d_130_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_130/BiasAdd/ReadVariableOp?
conv2d_130/BiasAddBiasAddconv2d_130/Conv2D:output:0)conv2d_130/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_130/BiasAdd?
conv2d_130/ReluReluconv2d_130/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_130/Relu?
%batch_normalization_40/ReadVariableOpReadVariableOp.batch_normalization_40_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_40/ReadVariableOp?
'batch_normalization_40/ReadVariableOp_1ReadVariableOp0batch_normalization_40_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_40/ReadVariableOp_1?
6batch_normalization_40/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_40_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_40/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_40/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_40_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_40/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_40/FusedBatchNormV3FusedBatchNormV3conv2d_130/Relu:activations:0-batch_normalization_40/ReadVariableOp:value:0/batch_normalization_40/ReadVariableOp_1:value:0>batch_normalization_40/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_40/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2)
'batch_normalization_40/FusedBatchNormV3?
 conv2d_131/Conv2D/ReadVariableOpReadVariableOp)conv2d_131_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_131/Conv2D/ReadVariableOp?
conv2d_131/Conv2DConv2D+batch_normalization_40/FusedBatchNormV3:y:0(conv2d_131/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_131/Conv2D?
!conv2d_131/BiasAdd/ReadVariableOpReadVariableOp*conv2d_131_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_131/BiasAdd/ReadVariableOp?
conv2d_131/BiasAddBiasAddconv2d_131/Conv2D:output:0)conv2d_131/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_131/BiasAdd?
conv2d_131/ReluReluconv2d_131/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_131/Relu?
%batch_normalization_41/ReadVariableOpReadVariableOp.batch_normalization_41_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_41/ReadVariableOp?
'batch_normalization_41/ReadVariableOp_1ReadVariableOp0batch_normalization_41_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_41/ReadVariableOp_1?
6batch_normalization_41/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_41_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_41/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_41/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_41_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_41/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_41/FusedBatchNormV3FusedBatchNormV3conv2d_131/Relu:activations:0-batch_normalization_41/ReadVariableOp:value:0/batch_normalization_41/ReadVariableOp_1:value:0>batch_normalization_41/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_41/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2)
'batch_normalization_41/FusedBatchNormV3?
max_pooling2d_52/MaxPoolMaxPool+batch_normalization_41/FusedBatchNormV3:y:0*/
_output_shapes
:?????????***
ksize
*
paddingVALID*
strides
2
max_pooling2d_52/MaxPool?
 conv2d_132/Conv2D/ReadVariableOpReadVariableOp)conv2d_132_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_132/Conv2D/ReadVariableOp?
conv2d_132/Conv2DConv2D!max_pooling2d_52/MaxPool:output:0(conv2d_132/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????***
paddingSAME*
strides
2
conv2d_132/Conv2D?
!conv2d_132/BiasAdd/ReadVariableOpReadVariableOp*conv2d_132_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_132/BiasAdd/ReadVariableOp?
conv2d_132/BiasAddBiasAddconv2d_132/Conv2D:output:0)conv2d_132/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**2
conv2d_132/BiasAdd?
conv2d_132/ReluReluconv2d_132/BiasAdd:output:0*
T0*/
_output_shapes
:?????????**2
conv2d_132/Relu?
%batch_normalization_42/ReadVariableOpReadVariableOp.batch_normalization_42_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_42/ReadVariableOp?
'batch_normalization_42/ReadVariableOp_1ReadVariableOp0batch_normalization_42_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_42/ReadVariableOp_1?
6batch_normalization_42/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_42_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_42/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_42/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_42_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_42/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_42/FusedBatchNormV3FusedBatchNormV3conv2d_132/Relu:activations:0-batch_normalization_42/ReadVariableOp:value:0/batch_normalization_42/ReadVariableOp_1:value:0>batch_normalization_42/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_42/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????**:::::*
epsilon%o?:*
is_training( 2)
'batch_normalization_42/FusedBatchNormV3?
 conv2d_133/Conv2D/ReadVariableOpReadVariableOp)conv2d_133_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_133/Conv2D/ReadVariableOp?
conv2d_133/Conv2DConv2D+batch_normalization_42/FusedBatchNormV3:y:0(conv2d_133/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????***
paddingSAME*
strides
2
conv2d_133/Conv2D?
!conv2d_133/BiasAdd/ReadVariableOpReadVariableOp*conv2d_133_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_133/BiasAdd/ReadVariableOp?
conv2d_133/BiasAddBiasAddconv2d_133/Conv2D:output:0)conv2d_133/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**2
conv2d_133/BiasAdd?
conv2d_133/ReluReluconv2d_133/BiasAdd:output:0*
T0*/
_output_shapes
:?????????**2
conv2d_133/Relu?
%batch_normalization_43/ReadVariableOpReadVariableOp.batch_normalization_43_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_43/ReadVariableOp?
'batch_normalization_43/ReadVariableOp_1ReadVariableOp0batch_normalization_43_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_43/ReadVariableOp_1?
6batch_normalization_43/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_43_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_43/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_43_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_43/FusedBatchNormV3FusedBatchNormV3conv2d_133/Relu:activations:0-batch_normalization_43/ReadVariableOp:value:0/batch_normalization_43/ReadVariableOp_1:value:0>batch_normalization_43/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????**:::::*
epsilon%o?:*
is_training( 2)
'batch_normalization_43/FusedBatchNormV3?
max_pooling2d_53/MaxPoolMaxPool+batch_normalization_43/FusedBatchNormV3:y:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_53/MaxPool?
 conv2d_134/Conv2D/ReadVariableOpReadVariableOp)conv2d_134_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_134/Conv2D/ReadVariableOp?
conv2d_134/Conv2DConv2D!max_pooling2d_53/MaxPool:output:0(conv2d_134/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_134/Conv2D?
!conv2d_134/BiasAdd/ReadVariableOpReadVariableOp*conv2d_134_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_134/BiasAdd/ReadVariableOp?
conv2d_134/BiasAddBiasAddconv2d_134/Conv2D:output:0)conv2d_134/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_134/BiasAdd?
conv2d_134/ReluReluconv2d_134/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_134/Relu?
%batch_normalization_44/ReadVariableOpReadVariableOp.batch_normalization_44_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_44/ReadVariableOp?
'batch_normalization_44/ReadVariableOp_1ReadVariableOp0batch_normalization_44_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_44/ReadVariableOp_1?
6batch_normalization_44/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_44_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_44/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_44_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_44/FusedBatchNormV3FusedBatchNormV3conv2d_134/Relu:activations:0-batch_normalization_44/ReadVariableOp:value:0/batch_normalization_44/ReadVariableOp_1:value:0>batch_normalization_44/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( 2)
'batch_normalization_44/FusedBatchNormV3?
 conv2d_135/Conv2D/ReadVariableOpReadVariableOp)conv2d_135_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_135/Conv2D/ReadVariableOp?
conv2d_135/Conv2DConv2D+batch_normalization_44/FusedBatchNormV3:y:0(conv2d_135/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_135/Conv2D?
!conv2d_135/BiasAdd/ReadVariableOpReadVariableOp*conv2d_135_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_135/BiasAdd/ReadVariableOp?
conv2d_135/BiasAddBiasAddconv2d_135/Conv2D:output:0)conv2d_135/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_135/BiasAdd?
conv2d_135/ReluReluconv2d_135/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_135/Relu?
%batch_normalization_45/ReadVariableOpReadVariableOp.batch_normalization_45_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_45/ReadVariableOp?
'batch_normalization_45/ReadVariableOp_1ReadVariableOp0batch_normalization_45_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_45/ReadVariableOp_1?
6batch_normalization_45/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_45_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_45/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_45_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_45/FusedBatchNormV3FusedBatchNormV3conv2d_135/Relu:activations:0-batch_normalization_45/ReadVariableOp:value:0/batch_normalization_45/ReadVariableOp_1:value:0>batch_normalization_45/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( 2)
'batch_normalization_45/FusedBatchNormV3?
max_pooling2d_54/MaxPoolMaxPool+batch_normalization_45/FusedBatchNormV3:y:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_54/MaxPoolu
flatten_28/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   2
flatten_28/Const?
flatten_28/ReshapeReshape!max_pooling2d_54/MaxPool:output:0flatten_28/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_28/Reshape?
dense_56/MatMul/ReadVariableOpReadVariableOp'dense_56_matmul_readvariableop_resource*
_output_shapes
:	? *
dtype02 
dense_56/MatMul/ReadVariableOp?
dense_56/MatMulMatMulflatten_28/Reshape:output:0&dense_56/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_56/MatMul?
dense_56/BiasAdd/ReadVariableOpReadVariableOp(dense_56_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_56/BiasAdd/ReadVariableOp?
dense_56/BiasAddBiasAdddense_56/MatMul:product:0'dense_56/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_56/BiasAdds
dense_56/ReluReludense_56/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_56/Relu?
dense_57/MatMul/ReadVariableOpReadVariableOp'dense_57_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_57/MatMul/ReadVariableOp?
dense_57/MatMulMatMuldense_56/Relu:activations:0&dense_57/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_57/MatMul?
dense_57/BiasAdd/ReadVariableOpReadVariableOp(dense_57_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_57/BiasAdd/ReadVariableOp?
dense_57/BiasAddBiasAdddense_57/MatMul:product:0'dense_57/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_57/BiasAdd|
dense_57/SoftmaxSoftmaxdense_57/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_57/Softmax?
IdentityIdentitydense_57/Softmax:softmax:07^batch_normalization_40/FusedBatchNormV3/ReadVariableOp9^batch_normalization_40/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_40/ReadVariableOp(^batch_normalization_40/ReadVariableOp_17^batch_normalization_41/FusedBatchNormV3/ReadVariableOp9^batch_normalization_41/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_41/ReadVariableOp(^batch_normalization_41/ReadVariableOp_17^batch_normalization_42/FusedBatchNormV3/ReadVariableOp9^batch_normalization_42/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_42/ReadVariableOp(^batch_normalization_42/ReadVariableOp_17^batch_normalization_43/FusedBatchNormV3/ReadVariableOp9^batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_43/ReadVariableOp(^batch_normalization_43/ReadVariableOp_17^batch_normalization_44/FusedBatchNormV3/ReadVariableOp9^batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_44/ReadVariableOp(^batch_normalization_44/ReadVariableOp_17^batch_normalization_45/FusedBatchNormV3/ReadVariableOp9^batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_45/ReadVariableOp(^batch_normalization_45/ReadVariableOp_1"^conv2d_130/BiasAdd/ReadVariableOp!^conv2d_130/Conv2D/ReadVariableOp"^conv2d_131/BiasAdd/ReadVariableOp!^conv2d_131/Conv2D/ReadVariableOp"^conv2d_132/BiasAdd/ReadVariableOp!^conv2d_132/Conv2D/ReadVariableOp"^conv2d_133/BiasAdd/ReadVariableOp!^conv2d_133/Conv2D/ReadVariableOp"^conv2d_134/BiasAdd/ReadVariableOp!^conv2d_134/Conv2D/ReadVariableOp"^conv2d_135/BiasAdd/ReadVariableOp!^conv2d_135/Conv2D/ReadVariableOp ^dense_56/BiasAdd/ReadVariableOp^dense_56/MatMul/ReadVariableOp ^dense_57/BiasAdd/ReadVariableOp^dense_57/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::2p
6batch_normalization_40/FusedBatchNormV3/ReadVariableOp6batch_normalization_40/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_40/FusedBatchNormV3/ReadVariableOp_18batch_normalization_40/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_40/ReadVariableOp%batch_normalization_40/ReadVariableOp2R
'batch_normalization_40/ReadVariableOp_1'batch_normalization_40/ReadVariableOp_12p
6batch_normalization_41/FusedBatchNormV3/ReadVariableOp6batch_normalization_41/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_41/FusedBatchNormV3/ReadVariableOp_18batch_normalization_41/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_41/ReadVariableOp%batch_normalization_41/ReadVariableOp2R
'batch_normalization_41/ReadVariableOp_1'batch_normalization_41/ReadVariableOp_12p
6batch_normalization_42/FusedBatchNormV3/ReadVariableOp6batch_normalization_42/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_42/FusedBatchNormV3/ReadVariableOp_18batch_normalization_42/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_42/ReadVariableOp%batch_normalization_42/ReadVariableOp2R
'batch_normalization_42/ReadVariableOp_1'batch_normalization_42/ReadVariableOp_12p
6batch_normalization_43/FusedBatchNormV3/ReadVariableOp6batch_normalization_43/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_43/FusedBatchNormV3/ReadVariableOp_18batch_normalization_43/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_43/ReadVariableOp%batch_normalization_43/ReadVariableOp2R
'batch_normalization_43/ReadVariableOp_1'batch_normalization_43/ReadVariableOp_12p
6batch_normalization_44/FusedBatchNormV3/ReadVariableOp6batch_normalization_44/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_44/FusedBatchNormV3/ReadVariableOp_18batch_normalization_44/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_44/ReadVariableOp%batch_normalization_44/ReadVariableOp2R
'batch_normalization_44/ReadVariableOp_1'batch_normalization_44/ReadVariableOp_12p
6batch_normalization_45/FusedBatchNormV3/ReadVariableOp6batch_normalization_45/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_45/FusedBatchNormV3/ReadVariableOp_18batch_normalization_45/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_45/ReadVariableOp%batch_normalization_45/ReadVariableOp2R
'batch_normalization_45/ReadVariableOp_1'batch_normalization_45/ReadVariableOp_12F
!conv2d_130/BiasAdd/ReadVariableOp!conv2d_130/BiasAdd/ReadVariableOp2D
 conv2d_130/Conv2D/ReadVariableOp conv2d_130/Conv2D/ReadVariableOp2F
!conv2d_131/BiasAdd/ReadVariableOp!conv2d_131/BiasAdd/ReadVariableOp2D
 conv2d_131/Conv2D/ReadVariableOp conv2d_131/Conv2D/ReadVariableOp2F
!conv2d_132/BiasAdd/ReadVariableOp!conv2d_132/BiasAdd/ReadVariableOp2D
 conv2d_132/Conv2D/ReadVariableOp conv2d_132/Conv2D/ReadVariableOp2F
!conv2d_133/BiasAdd/ReadVariableOp!conv2d_133/BiasAdd/ReadVariableOp2D
 conv2d_133/Conv2D/ReadVariableOp conv2d_133/Conv2D/ReadVariableOp2F
!conv2d_134/BiasAdd/ReadVariableOp!conv2d_134/BiasAdd/ReadVariableOp2D
 conv2d_134/Conv2D/ReadVariableOp conv2d_134/Conv2D/ReadVariableOp2F
!conv2d_135/BiasAdd/ReadVariableOp!conv2d_135/BiasAdd/ReadVariableOp2D
 conv2d_135/Conv2D/ReadVariableOp conv2d_135/Conv2D/ReadVariableOp2B
dense_56/BiasAdd/ReadVariableOpdense_56/BiasAdd/ReadVariableOp2@
dense_56/MatMul/ReadVariableOpdense_56/MatMul/ReadVariableOp2B
dense_57/BiasAdd/ReadVariableOpdense_57/BiasAdd/ReadVariableOp2@
dense_57/MatMul/ReadVariableOpdense_57/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
4__inference_CNN_aug_deep_bnorm_layer_call_fn_2303840
input_29
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

unknown_38
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_29unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_38*4
Tin-
+2)*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*>
_read_only_resource_inputs 
	
 !"%&'(*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_CNN_aug_deep_bnorm_layer_call_and_return_conditional_losses_23037572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:???????????
"
_user_specified_name
input_29
?
?
8__inference_batch_normalization_45_layer_call_fn_2305426

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
S__inference_batch_normalization_45_layer_call_and_return_conditional_losses_23034442
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
S__inference_batch_normalization_41_layer_call_and_return_conditional_losses_2304790

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
8__inference_batch_normalization_44_layer_call_fn_2305342

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
S__inference_batch_normalization_44_layer_call_and_return_conditional_losses_23027472
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
?
?
S__inference_batch_normalization_45_layer_call_and_return_conditional_losses_2305464

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
S__inference_batch_normalization_44_layer_call_and_return_conditional_losses_2302716

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
S__inference_batch_normalization_42_layer_call_and_return_conditional_losses_2305002

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
E__inference_dense_57_layer_call_and_return_conditional_losses_2303533

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
?
?
S__inference_batch_normalization_40_layer_call_and_return_conditional_losses_2302942

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
?
?
S__inference_batch_normalization_45_layer_call_and_return_conditional_losses_2305446

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
?	
?
E__inference_dense_56_layer_call_and_return_conditional_losses_2303506

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	? *
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
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_44_layer_call_fn_2305265

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
S__inference_batch_normalization_44_layer_call_and_return_conditional_losses_23033262
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
?
N
2__inference_max_pooling2d_52_layer_call_fn_2302434

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
M__inference_max_pooling2d_52_layer_call_and_return_conditional_losses_23024282
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
8__inference_batch_normalization_42_layer_call_fn_2305046

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
S__inference_batch_normalization_42_layer_call_and_return_conditional_losses_23031432
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
S__inference_batch_normalization_42_layer_call_and_return_conditional_losses_2304956

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
8__inference_batch_normalization_42_layer_call_fn_2305033

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
S__inference_batch_normalization_42_layer_call_and_return_conditional_losses_23031252
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
?
?
S__inference_batch_normalization_44_layer_call_and_return_conditional_losses_2305252

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
S__inference_batch_normalization_45_layer_call_and_return_conditional_losses_2303444

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
Ȅ
?#
O__inference_CNN_aug_deep_bnorm_layer_call_and_return_conditional_losses_2304283

inputs-
)conv2d_130_conv2d_readvariableop_resource.
*conv2d_130_biasadd_readvariableop_resource2
.batch_normalization_40_readvariableop_resource4
0batch_normalization_40_readvariableop_1_resourceC
?batch_normalization_40_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_40_fusedbatchnormv3_readvariableop_1_resource-
)conv2d_131_conv2d_readvariableop_resource.
*conv2d_131_biasadd_readvariableop_resource2
.batch_normalization_41_readvariableop_resource4
0batch_normalization_41_readvariableop_1_resourceC
?batch_normalization_41_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_41_fusedbatchnormv3_readvariableop_1_resource-
)conv2d_132_conv2d_readvariableop_resource.
*conv2d_132_biasadd_readvariableop_resource2
.batch_normalization_42_readvariableop_resource4
0batch_normalization_42_readvariableop_1_resourceC
?batch_normalization_42_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_42_fusedbatchnormv3_readvariableop_1_resource-
)conv2d_133_conv2d_readvariableop_resource.
*conv2d_133_biasadd_readvariableop_resource2
.batch_normalization_43_readvariableop_resource4
0batch_normalization_43_readvariableop_1_resourceC
?batch_normalization_43_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_43_fusedbatchnormv3_readvariableop_1_resource-
)conv2d_134_conv2d_readvariableop_resource.
*conv2d_134_biasadd_readvariableop_resource2
.batch_normalization_44_readvariableop_resource4
0batch_normalization_44_readvariableop_1_resourceC
?batch_normalization_44_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_44_fusedbatchnormv3_readvariableop_1_resource-
)conv2d_135_conv2d_readvariableop_resource.
*conv2d_135_biasadd_readvariableop_resource2
.batch_normalization_45_readvariableop_resource4
0batch_normalization_45_readvariableop_1_resourceC
?batch_normalization_45_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_45_fusedbatchnormv3_readvariableop_1_resource+
'dense_56_matmul_readvariableop_resource,
(dense_56_biasadd_readvariableop_resource+
'dense_57_matmul_readvariableop_resource,
(dense_57_biasadd_readvariableop_resource
identity??%batch_normalization_40/AssignNewValue?'batch_normalization_40/AssignNewValue_1?6batch_normalization_40/FusedBatchNormV3/ReadVariableOp?8batch_normalization_40/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_40/ReadVariableOp?'batch_normalization_40/ReadVariableOp_1?%batch_normalization_41/AssignNewValue?'batch_normalization_41/AssignNewValue_1?6batch_normalization_41/FusedBatchNormV3/ReadVariableOp?8batch_normalization_41/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_41/ReadVariableOp?'batch_normalization_41/ReadVariableOp_1?%batch_normalization_42/AssignNewValue?'batch_normalization_42/AssignNewValue_1?6batch_normalization_42/FusedBatchNormV3/ReadVariableOp?8batch_normalization_42/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_42/ReadVariableOp?'batch_normalization_42/ReadVariableOp_1?%batch_normalization_43/AssignNewValue?'batch_normalization_43/AssignNewValue_1?6batch_normalization_43/FusedBatchNormV3/ReadVariableOp?8batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_43/ReadVariableOp?'batch_normalization_43/ReadVariableOp_1?%batch_normalization_44/AssignNewValue?'batch_normalization_44/AssignNewValue_1?6batch_normalization_44/FusedBatchNormV3/ReadVariableOp?8batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_44/ReadVariableOp?'batch_normalization_44/ReadVariableOp_1?%batch_normalization_45/AssignNewValue?'batch_normalization_45/AssignNewValue_1?6batch_normalization_45/FusedBatchNormV3/ReadVariableOp?8batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_45/ReadVariableOp?'batch_normalization_45/ReadVariableOp_1?!conv2d_130/BiasAdd/ReadVariableOp? conv2d_130/Conv2D/ReadVariableOp?!conv2d_131/BiasAdd/ReadVariableOp? conv2d_131/Conv2D/ReadVariableOp?!conv2d_132/BiasAdd/ReadVariableOp? conv2d_132/Conv2D/ReadVariableOp?!conv2d_133/BiasAdd/ReadVariableOp? conv2d_133/Conv2D/ReadVariableOp?!conv2d_134/BiasAdd/ReadVariableOp? conv2d_134/Conv2D/ReadVariableOp?!conv2d_135/BiasAdd/ReadVariableOp? conv2d_135/Conv2D/ReadVariableOp?dense_56/BiasAdd/ReadVariableOp?dense_56/MatMul/ReadVariableOp?dense_57/BiasAdd/ReadVariableOp?dense_57/MatMul/ReadVariableOp?
 conv2d_130/Conv2D/ReadVariableOpReadVariableOp)conv2d_130_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_130/Conv2D/ReadVariableOp?
conv2d_130/Conv2DConv2Dinputs(conv2d_130/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_130/Conv2D?
!conv2d_130/BiasAdd/ReadVariableOpReadVariableOp*conv2d_130_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_130/BiasAdd/ReadVariableOp?
conv2d_130/BiasAddBiasAddconv2d_130/Conv2D:output:0)conv2d_130/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_130/BiasAdd?
conv2d_130/ReluReluconv2d_130/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_130/Relu?
%batch_normalization_40/ReadVariableOpReadVariableOp.batch_normalization_40_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_40/ReadVariableOp?
'batch_normalization_40/ReadVariableOp_1ReadVariableOp0batch_normalization_40_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_40/ReadVariableOp_1?
6batch_normalization_40/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_40_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_40/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_40/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_40_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_40/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_40/FusedBatchNormV3FusedBatchNormV3conv2d_130/Relu:activations:0-batch_normalization_40/ReadVariableOp:value:0/batch_normalization_40/ReadVariableOp_1:value:0>batch_normalization_40/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_40/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_40/FusedBatchNormV3?
%batch_normalization_40/AssignNewValueAssignVariableOp?batch_normalization_40_fusedbatchnormv3_readvariableop_resource4batch_normalization_40/FusedBatchNormV3:batch_mean:07^batch_normalization_40/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*R
_classH
FDloc:@batch_normalization_40/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_40/AssignNewValue?
'batch_normalization_40/AssignNewValue_1AssignVariableOpAbatch_normalization_40_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_40/FusedBatchNormV3:batch_variance:09^batch_normalization_40/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_40/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_40/AssignNewValue_1?
 conv2d_131/Conv2D/ReadVariableOpReadVariableOp)conv2d_131_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_131/Conv2D/ReadVariableOp?
conv2d_131/Conv2DConv2D+batch_normalization_40/FusedBatchNormV3:y:0(conv2d_131/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_131/Conv2D?
!conv2d_131/BiasAdd/ReadVariableOpReadVariableOp*conv2d_131_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_131/BiasAdd/ReadVariableOp?
conv2d_131/BiasAddBiasAddconv2d_131/Conv2D:output:0)conv2d_131/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_131/BiasAdd?
conv2d_131/ReluReluconv2d_131/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_131/Relu?
%batch_normalization_41/ReadVariableOpReadVariableOp.batch_normalization_41_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_41/ReadVariableOp?
'batch_normalization_41/ReadVariableOp_1ReadVariableOp0batch_normalization_41_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_41/ReadVariableOp_1?
6batch_normalization_41/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_41_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_41/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_41/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_41_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_41/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_41/FusedBatchNormV3FusedBatchNormV3conv2d_131/Relu:activations:0-batch_normalization_41/ReadVariableOp:value:0/batch_normalization_41/ReadVariableOp_1:value:0>batch_normalization_41/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_41/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_41/FusedBatchNormV3?
%batch_normalization_41/AssignNewValueAssignVariableOp?batch_normalization_41_fusedbatchnormv3_readvariableop_resource4batch_normalization_41/FusedBatchNormV3:batch_mean:07^batch_normalization_41/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*R
_classH
FDloc:@batch_normalization_41/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_41/AssignNewValue?
'batch_normalization_41/AssignNewValue_1AssignVariableOpAbatch_normalization_41_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_41/FusedBatchNormV3:batch_variance:09^batch_normalization_41/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_41/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_41/AssignNewValue_1?
max_pooling2d_52/MaxPoolMaxPool+batch_normalization_41/FusedBatchNormV3:y:0*/
_output_shapes
:?????????***
ksize
*
paddingVALID*
strides
2
max_pooling2d_52/MaxPool?
 conv2d_132/Conv2D/ReadVariableOpReadVariableOp)conv2d_132_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_132/Conv2D/ReadVariableOp?
conv2d_132/Conv2DConv2D!max_pooling2d_52/MaxPool:output:0(conv2d_132/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????***
paddingSAME*
strides
2
conv2d_132/Conv2D?
!conv2d_132/BiasAdd/ReadVariableOpReadVariableOp*conv2d_132_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_132/BiasAdd/ReadVariableOp?
conv2d_132/BiasAddBiasAddconv2d_132/Conv2D:output:0)conv2d_132/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**2
conv2d_132/BiasAdd?
conv2d_132/ReluReluconv2d_132/BiasAdd:output:0*
T0*/
_output_shapes
:?????????**2
conv2d_132/Relu?
%batch_normalization_42/ReadVariableOpReadVariableOp.batch_normalization_42_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_42/ReadVariableOp?
'batch_normalization_42/ReadVariableOp_1ReadVariableOp0batch_normalization_42_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_42/ReadVariableOp_1?
6batch_normalization_42/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_42_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_42/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_42/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_42_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_42/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_42/FusedBatchNormV3FusedBatchNormV3conv2d_132/Relu:activations:0-batch_normalization_42/ReadVariableOp:value:0/batch_normalization_42/ReadVariableOp_1:value:0>batch_normalization_42/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_42/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????**:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_42/FusedBatchNormV3?
%batch_normalization_42/AssignNewValueAssignVariableOp?batch_normalization_42_fusedbatchnormv3_readvariableop_resource4batch_normalization_42/FusedBatchNormV3:batch_mean:07^batch_normalization_42/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*R
_classH
FDloc:@batch_normalization_42/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_42/AssignNewValue?
'batch_normalization_42/AssignNewValue_1AssignVariableOpAbatch_normalization_42_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_42/FusedBatchNormV3:batch_variance:09^batch_normalization_42/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_42/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_42/AssignNewValue_1?
 conv2d_133/Conv2D/ReadVariableOpReadVariableOp)conv2d_133_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_133/Conv2D/ReadVariableOp?
conv2d_133/Conv2DConv2D+batch_normalization_42/FusedBatchNormV3:y:0(conv2d_133/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????***
paddingSAME*
strides
2
conv2d_133/Conv2D?
!conv2d_133/BiasAdd/ReadVariableOpReadVariableOp*conv2d_133_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_133/BiasAdd/ReadVariableOp?
conv2d_133/BiasAddBiasAddconv2d_133/Conv2D:output:0)conv2d_133/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**2
conv2d_133/BiasAdd?
conv2d_133/ReluReluconv2d_133/BiasAdd:output:0*
T0*/
_output_shapes
:?????????**2
conv2d_133/Relu?
%batch_normalization_43/ReadVariableOpReadVariableOp.batch_normalization_43_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_43/ReadVariableOp?
'batch_normalization_43/ReadVariableOp_1ReadVariableOp0batch_normalization_43_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_43/ReadVariableOp_1?
6batch_normalization_43/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_43_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_43/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_43_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_43/FusedBatchNormV3FusedBatchNormV3conv2d_133/Relu:activations:0-batch_normalization_43/ReadVariableOp:value:0/batch_normalization_43/ReadVariableOp_1:value:0>batch_normalization_43/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????**:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_43/FusedBatchNormV3?
%batch_normalization_43/AssignNewValueAssignVariableOp?batch_normalization_43_fusedbatchnormv3_readvariableop_resource4batch_normalization_43/FusedBatchNormV3:batch_mean:07^batch_normalization_43/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*R
_classH
FDloc:@batch_normalization_43/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_43/AssignNewValue?
'batch_normalization_43/AssignNewValue_1AssignVariableOpAbatch_normalization_43_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_43/FusedBatchNormV3:batch_variance:09^batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_43/AssignNewValue_1?
max_pooling2d_53/MaxPoolMaxPool+batch_normalization_43/FusedBatchNormV3:y:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_53/MaxPool?
 conv2d_134/Conv2D/ReadVariableOpReadVariableOp)conv2d_134_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_134/Conv2D/ReadVariableOp?
conv2d_134/Conv2DConv2D!max_pooling2d_53/MaxPool:output:0(conv2d_134/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_134/Conv2D?
!conv2d_134/BiasAdd/ReadVariableOpReadVariableOp*conv2d_134_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_134/BiasAdd/ReadVariableOp?
conv2d_134/BiasAddBiasAddconv2d_134/Conv2D:output:0)conv2d_134/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_134/BiasAdd?
conv2d_134/ReluReluconv2d_134/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_134/Relu?
%batch_normalization_44/ReadVariableOpReadVariableOp.batch_normalization_44_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_44/ReadVariableOp?
'batch_normalization_44/ReadVariableOp_1ReadVariableOp0batch_normalization_44_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_44/ReadVariableOp_1?
6batch_normalization_44/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_44_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_44/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_44_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_44/FusedBatchNormV3FusedBatchNormV3conv2d_134/Relu:activations:0-batch_normalization_44/ReadVariableOp:value:0/batch_normalization_44/ReadVariableOp_1:value:0>batch_normalization_44/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_44/FusedBatchNormV3?
%batch_normalization_44/AssignNewValueAssignVariableOp?batch_normalization_44_fusedbatchnormv3_readvariableop_resource4batch_normalization_44/FusedBatchNormV3:batch_mean:07^batch_normalization_44/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*R
_classH
FDloc:@batch_normalization_44/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_44/AssignNewValue?
'batch_normalization_44/AssignNewValue_1AssignVariableOpAbatch_normalization_44_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_44/FusedBatchNormV3:batch_variance:09^batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_44/AssignNewValue_1?
 conv2d_135/Conv2D/ReadVariableOpReadVariableOp)conv2d_135_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_135/Conv2D/ReadVariableOp?
conv2d_135/Conv2DConv2D+batch_normalization_44/FusedBatchNormV3:y:0(conv2d_135/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_135/Conv2D?
!conv2d_135/BiasAdd/ReadVariableOpReadVariableOp*conv2d_135_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_135/BiasAdd/ReadVariableOp?
conv2d_135/BiasAddBiasAddconv2d_135/Conv2D:output:0)conv2d_135/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_135/BiasAdd?
conv2d_135/ReluReluconv2d_135/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_135/Relu?
%batch_normalization_45/ReadVariableOpReadVariableOp.batch_normalization_45_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_45/ReadVariableOp?
'batch_normalization_45/ReadVariableOp_1ReadVariableOp0batch_normalization_45_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_45/ReadVariableOp_1?
6batch_normalization_45/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_45_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_45/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_45_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_45/FusedBatchNormV3FusedBatchNormV3conv2d_135/Relu:activations:0-batch_normalization_45/ReadVariableOp:value:0/batch_normalization_45/ReadVariableOp_1:value:0>batch_normalization_45/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_45/FusedBatchNormV3?
%batch_normalization_45/AssignNewValueAssignVariableOp?batch_normalization_45_fusedbatchnormv3_readvariableop_resource4batch_normalization_45/FusedBatchNormV3:batch_mean:07^batch_normalization_45/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*R
_classH
FDloc:@batch_normalization_45/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_45/AssignNewValue?
'batch_normalization_45/AssignNewValue_1AssignVariableOpAbatch_normalization_45_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_45/FusedBatchNormV3:batch_variance:09^batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_45/AssignNewValue_1?
max_pooling2d_54/MaxPoolMaxPool+batch_normalization_45/FusedBatchNormV3:y:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_54/MaxPoolu
flatten_28/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   2
flatten_28/Const?
flatten_28/ReshapeReshape!max_pooling2d_54/MaxPool:output:0flatten_28/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_28/Reshape?
dense_56/MatMul/ReadVariableOpReadVariableOp'dense_56_matmul_readvariableop_resource*
_output_shapes
:	? *
dtype02 
dense_56/MatMul/ReadVariableOp?
dense_56/MatMulMatMulflatten_28/Reshape:output:0&dense_56/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_56/MatMul?
dense_56/BiasAdd/ReadVariableOpReadVariableOp(dense_56_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_56/BiasAdd/ReadVariableOp?
dense_56/BiasAddBiasAdddense_56/MatMul:product:0'dense_56/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_56/BiasAdds
dense_56/ReluReludense_56/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_56/Relu?
dense_57/MatMul/ReadVariableOpReadVariableOp'dense_57_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_57/MatMul/ReadVariableOp?
dense_57/MatMulMatMuldense_56/Relu:activations:0&dense_57/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_57/MatMul?
dense_57/BiasAdd/ReadVariableOpReadVariableOp(dense_57_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_57/BiasAdd/ReadVariableOp?
dense_57/BiasAddBiasAdddense_57/MatMul:product:0'dense_57/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_57/BiasAdd|
dense_57/SoftmaxSoftmaxdense_57/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_57/Softmax?
IdentityIdentitydense_57/Softmax:softmax:0&^batch_normalization_40/AssignNewValue(^batch_normalization_40/AssignNewValue_17^batch_normalization_40/FusedBatchNormV3/ReadVariableOp9^batch_normalization_40/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_40/ReadVariableOp(^batch_normalization_40/ReadVariableOp_1&^batch_normalization_41/AssignNewValue(^batch_normalization_41/AssignNewValue_17^batch_normalization_41/FusedBatchNormV3/ReadVariableOp9^batch_normalization_41/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_41/ReadVariableOp(^batch_normalization_41/ReadVariableOp_1&^batch_normalization_42/AssignNewValue(^batch_normalization_42/AssignNewValue_17^batch_normalization_42/FusedBatchNormV3/ReadVariableOp9^batch_normalization_42/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_42/ReadVariableOp(^batch_normalization_42/ReadVariableOp_1&^batch_normalization_43/AssignNewValue(^batch_normalization_43/AssignNewValue_17^batch_normalization_43/FusedBatchNormV3/ReadVariableOp9^batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_43/ReadVariableOp(^batch_normalization_43/ReadVariableOp_1&^batch_normalization_44/AssignNewValue(^batch_normalization_44/AssignNewValue_17^batch_normalization_44/FusedBatchNormV3/ReadVariableOp9^batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_44/ReadVariableOp(^batch_normalization_44/ReadVariableOp_1&^batch_normalization_45/AssignNewValue(^batch_normalization_45/AssignNewValue_17^batch_normalization_45/FusedBatchNormV3/ReadVariableOp9^batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_45/ReadVariableOp(^batch_normalization_45/ReadVariableOp_1"^conv2d_130/BiasAdd/ReadVariableOp!^conv2d_130/Conv2D/ReadVariableOp"^conv2d_131/BiasAdd/ReadVariableOp!^conv2d_131/Conv2D/ReadVariableOp"^conv2d_132/BiasAdd/ReadVariableOp!^conv2d_132/Conv2D/ReadVariableOp"^conv2d_133/BiasAdd/ReadVariableOp!^conv2d_133/Conv2D/ReadVariableOp"^conv2d_134/BiasAdd/ReadVariableOp!^conv2d_134/Conv2D/ReadVariableOp"^conv2d_135/BiasAdd/ReadVariableOp!^conv2d_135/Conv2D/ReadVariableOp ^dense_56/BiasAdd/ReadVariableOp^dense_56/MatMul/ReadVariableOp ^dense_57/BiasAdd/ReadVariableOp^dense_57/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::2N
%batch_normalization_40/AssignNewValue%batch_normalization_40/AssignNewValue2R
'batch_normalization_40/AssignNewValue_1'batch_normalization_40/AssignNewValue_12p
6batch_normalization_40/FusedBatchNormV3/ReadVariableOp6batch_normalization_40/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_40/FusedBatchNormV3/ReadVariableOp_18batch_normalization_40/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_40/ReadVariableOp%batch_normalization_40/ReadVariableOp2R
'batch_normalization_40/ReadVariableOp_1'batch_normalization_40/ReadVariableOp_12N
%batch_normalization_41/AssignNewValue%batch_normalization_41/AssignNewValue2R
'batch_normalization_41/AssignNewValue_1'batch_normalization_41/AssignNewValue_12p
6batch_normalization_41/FusedBatchNormV3/ReadVariableOp6batch_normalization_41/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_41/FusedBatchNormV3/ReadVariableOp_18batch_normalization_41/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_41/ReadVariableOp%batch_normalization_41/ReadVariableOp2R
'batch_normalization_41/ReadVariableOp_1'batch_normalization_41/ReadVariableOp_12N
%batch_normalization_42/AssignNewValue%batch_normalization_42/AssignNewValue2R
'batch_normalization_42/AssignNewValue_1'batch_normalization_42/AssignNewValue_12p
6batch_normalization_42/FusedBatchNormV3/ReadVariableOp6batch_normalization_42/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_42/FusedBatchNormV3/ReadVariableOp_18batch_normalization_42/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_42/ReadVariableOp%batch_normalization_42/ReadVariableOp2R
'batch_normalization_42/ReadVariableOp_1'batch_normalization_42/ReadVariableOp_12N
%batch_normalization_43/AssignNewValue%batch_normalization_43/AssignNewValue2R
'batch_normalization_43/AssignNewValue_1'batch_normalization_43/AssignNewValue_12p
6batch_normalization_43/FusedBatchNormV3/ReadVariableOp6batch_normalization_43/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_43/FusedBatchNormV3/ReadVariableOp_18batch_normalization_43/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_43/ReadVariableOp%batch_normalization_43/ReadVariableOp2R
'batch_normalization_43/ReadVariableOp_1'batch_normalization_43/ReadVariableOp_12N
%batch_normalization_44/AssignNewValue%batch_normalization_44/AssignNewValue2R
'batch_normalization_44/AssignNewValue_1'batch_normalization_44/AssignNewValue_12p
6batch_normalization_44/FusedBatchNormV3/ReadVariableOp6batch_normalization_44/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_44/FusedBatchNormV3/ReadVariableOp_18batch_normalization_44/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_44/ReadVariableOp%batch_normalization_44/ReadVariableOp2R
'batch_normalization_44/ReadVariableOp_1'batch_normalization_44/ReadVariableOp_12N
%batch_normalization_45/AssignNewValue%batch_normalization_45/AssignNewValue2R
'batch_normalization_45/AssignNewValue_1'batch_normalization_45/AssignNewValue_12p
6batch_normalization_45/FusedBatchNormV3/ReadVariableOp6batch_normalization_45/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_45/FusedBatchNormV3/ReadVariableOp_18batch_normalization_45/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_45/ReadVariableOp%batch_normalization_45/ReadVariableOp2R
'batch_normalization_45/ReadVariableOp_1'batch_normalization_45/ReadVariableOp_12F
!conv2d_130/BiasAdd/ReadVariableOp!conv2d_130/BiasAdd/ReadVariableOp2D
 conv2d_130/Conv2D/ReadVariableOp conv2d_130/Conv2D/ReadVariableOp2F
!conv2d_131/BiasAdd/ReadVariableOp!conv2d_131/BiasAdd/ReadVariableOp2D
 conv2d_131/Conv2D/ReadVariableOp conv2d_131/Conv2D/ReadVariableOp2F
!conv2d_132/BiasAdd/ReadVariableOp!conv2d_132/BiasAdd/ReadVariableOp2D
 conv2d_132/Conv2D/ReadVariableOp conv2d_132/Conv2D/ReadVariableOp2F
!conv2d_133/BiasAdd/ReadVariableOp!conv2d_133/BiasAdd/ReadVariableOp2D
 conv2d_133/Conv2D/ReadVariableOp conv2d_133/Conv2D/ReadVariableOp2F
!conv2d_134/BiasAdd/ReadVariableOp!conv2d_134/BiasAdd/ReadVariableOp2D
 conv2d_134/Conv2D/ReadVariableOp conv2d_134/Conv2D/ReadVariableOp2F
!conv2d_135/BiasAdd/ReadVariableOp!conv2d_135/BiasAdd/ReadVariableOp2D
 conv2d_135/Conv2D/ReadVariableOp conv2d_135/Conv2D/ReadVariableOp2B
dense_56/BiasAdd/ReadVariableOpdense_56/BiasAdd/ReadVariableOp2@
dense_56/MatMul/ReadVariableOpdense_56/MatMul/ReadVariableOp2B
dense_57/BiasAdd/ReadVariableOpdense_57/BiasAdd/ReadVariableOp2@
dense_57/MatMul/ReadVariableOpdense_57/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_43_layer_call_and_return_conditional_losses_2305086

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
8__inference_batch_normalization_40_layer_call_fn_2304750

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
S__inference_batch_normalization_40_layer_call_and_return_conditional_losses_23029422
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
?

*__inference_dense_56_layer_call_fn_2305521

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
E__inference_dense_56_layer_call_and_return_conditional_losses_23035062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_42_layer_call_and_return_conditional_losses_2305020

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
S__inference_batch_normalization_43_layer_call_and_return_conditional_losses_2303225

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
E__inference_dense_56_layer_call_and_return_conditional_losses_2305512

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	? *
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
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_41_layer_call_fn_2304898

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
S__inference_batch_normalization_41_layer_call_and_return_conditional_losses_23030422
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
S__inference_batch_normalization_44_layer_call_and_return_conditional_losses_2302747

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
?
?
4__inference_CNN_aug_deep_bnorm_layer_call_fn_2304517

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

unknown_38
identity??StatefulPartitionedCall?
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
unknown_38*4
Tin-
+2)*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*>
_read_only_resource_inputs 
	
 !"%&'(*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_CNN_aug_deep_bnorm_layer_call_and_return_conditional_losses_23037572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_41_layer_call_and_return_conditional_losses_2302411

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
E__inference_dense_57_layer_call_and_return_conditional_losses_2305532

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
8__inference_batch_normalization_40_layer_call_fn_2304737

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
S__inference_batch_normalization_40_layer_call_and_return_conditional_losses_23029242
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
S__inference_batch_normalization_42_layer_call_and_return_conditional_losses_2303143

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
?
?
S__inference_batch_normalization_40_layer_call_and_return_conditional_losses_2304660

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
S__inference_batch_normalization_43_layer_call_and_return_conditional_losses_2302631

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
S__inference_batch_normalization_42_layer_call_and_return_conditional_losses_2304938

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
S__inference_batch_normalization_40_layer_call_and_return_conditional_losses_2304642

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
8__inference_batch_normalization_44_layer_call_fn_2305329

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
S__inference_batch_normalization_44_layer_call_and_return_conditional_losses_23027162
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
G__inference_conv2d_131_layer_call_and_return_conditional_losses_2302989

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
?
?
S__inference_batch_normalization_42_layer_call_and_return_conditional_losses_2303125

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
?
?
%__inference_signature_wrapper_2304122
input_29
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

unknown_38
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_29unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_38*4
Tin-
+2)*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*J
_read_only_resource_inputs,
*(	
 !"#$%&'(*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__wrapped_model_23022142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:???????????
"
_user_specified_name
input_29
?
?
S__inference_batch_normalization_45_layer_call_and_return_conditional_losses_2302820

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
S__inference_batch_normalization_44_layer_call_and_return_conditional_losses_2305316

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
,__inference_conv2d_135_layer_call_fn_2305362

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
G__inference_conv2d_135_layer_call_and_return_conditional_losses_23033912
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
?
?
8__inference_batch_normalization_43_layer_call_fn_2305181

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
S__inference_batch_normalization_43_layer_call_and_return_conditional_losses_23032252
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
?
?
S__inference_batch_normalization_40_layer_call_and_return_conditional_losses_2302276

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
S__inference_batch_normalization_43_layer_call_and_return_conditional_losses_2302600

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
8__inference_batch_normalization_44_layer_call_fn_2305278

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
S__inference_batch_normalization_44_layer_call_and_return_conditional_losses_23033442
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
?

?
G__inference_conv2d_130_layer_call_and_return_conditional_losses_2302889

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
input_29;
serving_default_input_29:0???????????<
dense_570
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
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
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer_with_weights-7

layer-9
layer-10
layer_with_weights-8
layer-11
layer_with_weights-9
layer-12
layer_with_weights-10
layer-13
layer_with_weights-11
layer-14
layer-15
layer-16
layer_with_weights-12
layer-17
layer_with_weights-13
layer-18
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
?_default_save_signature
+?&call_and_return_all_conditional_losses
?__call__"??
_tf_keras_network??{"class_name": "Functional", "name": "CNN_aug_deep_bnorm", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "CNN_aug_deep_bnorm", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_29"}, "name": "input_29", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_130", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_130", "inbound_nodes": [[["input_29", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_40", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_40", "inbound_nodes": [[["conv2d_130", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_131", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_131", "inbound_nodes": [[["batch_normalization_40", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_41", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_41", "inbound_nodes": [[["conv2d_131", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_52", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "name": "max_pooling2d_52", "inbound_nodes": [[["batch_normalization_41", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_132", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_132", "inbound_nodes": [[["max_pooling2d_52", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_42", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_42", "inbound_nodes": [[["conv2d_132", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_133", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_133", "inbound_nodes": [[["batch_normalization_42", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_43", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_43", "inbound_nodes": [[["conv2d_133", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_53", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "name": "max_pooling2d_53", "inbound_nodes": [[["batch_normalization_43", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_134", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_134", "inbound_nodes": [[["max_pooling2d_53", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_44", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_44", "inbound_nodes": [[["conv2d_134", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_135", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_135", "inbound_nodes": [[["batch_normalization_44", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_45", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_45", "inbound_nodes": [[["conv2d_135", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_54", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "name": "max_pooling2d_54", "inbound_nodes": [[["batch_normalization_45", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_28", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_28", "inbound_nodes": [[["max_pooling2d_54", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_56", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_56", "inbound_nodes": [[["flatten_28", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_57", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_57", "inbound_nodes": [[["dense_56", 0, 0, {}]]]}], "input_layers": [["input_29", 0, 0]], "output_layers": [["dense_57", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "CNN_aug_deep_bnorm", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_29"}, "name": "input_29", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_130", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_130", "inbound_nodes": [[["input_29", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_40", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_40", "inbound_nodes": [[["conv2d_130", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_131", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_131", "inbound_nodes": [[["batch_normalization_40", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_41", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_41", "inbound_nodes": [[["conv2d_131", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_52", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "name": "max_pooling2d_52", "inbound_nodes": [[["batch_normalization_41", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_132", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_132", "inbound_nodes": [[["max_pooling2d_52", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_42", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_42", "inbound_nodes": [[["conv2d_132", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_133", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_133", "inbound_nodes": [[["batch_normalization_42", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_43", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_43", "inbound_nodes": [[["conv2d_133", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_53", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "name": "max_pooling2d_53", "inbound_nodes": [[["batch_normalization_43", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_134", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_134", "inbound_nodes": [[["max_pooling2d_53", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_44", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_44", "inbound_nodes": [[["conv2d_134", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_135", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_135", "inbound_nodes": [[["batch_normalization_44", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_45", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_45", "inbound_nodes": [[["conv2d_135", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_54", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "name": "max_pooling2d_54", "inbound_nodes": [[["batch_normalization_45", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_28", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_28", "inbound_nodes": [[["max_pooling2d_54", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_56", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_56", "inbound_nodes": [[["flatten_28", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_57", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_57", "inbound_nodes": [[["dense_56", 0, 0, {}]]]}], "input_layers": [["input_29", 0, 0]], "output_layers": [["dense_57", 0, 0]]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_29", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_29"}}
?	

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_130", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_130", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 1]}}
?	
 axis
	!gamma
"beta
#moving_mean
$moving_variance
%regularization_losses
&trainable_variables
'	variables
(	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_40", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_40", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 8]}}
?	

)kernel
*bias
+regularization_losses
,trainable_variables
-	variables
.	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_131", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_131", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 8]}}
?	
/axis
	0gamma
1beta
2moving_mean
3moving_variance
4regularization_losses
5trainable_variables
6	variables
7	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_41", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_41", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 8]}}
?
8regularization_losses
9trainable_variables
:	variables
;	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_52", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_52", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

<kernel
=bias
>regularization_losses
?trainable_variables
@	variables
A	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_132", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_132", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 42, 42, 8]}}
?	
Baxis
	Cgamma
Dbeta
Emoving_mean
Fmoving_variance
Gregularization_losses
Htrainable_variables
I	variables
J	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_42", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_42", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 42, 42, 8]}}
?	

Kkernel
Lbias
Mregularization_losses
Ntrainable_variables
O	variables
P	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_133", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_133", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 42, 42, 8]}}
?	
Qaxis
	Rgamma
Sbeta
Tmoving_mean
Umoving_variance
Vregularization_losses
Wtrainable_variables
X	variables
Y	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_43", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_43", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 42, 42, 8]}}
?
Zregularization_losses
[trainable_variables
\	variables
]	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_53", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_53", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

^kernel
_bias
`regularization_losses
atrainable_variables
b	variables
c	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_134", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_134", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 8]}}
?	
daxis
	egamma
fbeta
gmoving_mean
hmoving_variance
iregularization_losses
jtrainable_variables
k	variables
l	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_44", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_44", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 8]}}
?	

mkernel
nbias
oregularization_losses
ptrainable_variables
q	variables
r	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_135", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_135", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 8]}}
?	
saxis
	tgamma
ubeta
vmoving_mean
wmoving_variance
xregularization_losses
ytrainable_variables
z	variables
{	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_45", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_45", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 8]}}
?
|regularization_losses
}trainable_variables
~	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_54", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_54", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_28", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_28", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_56", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_56", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
?
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_57", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_57", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
?
	?iter
?beta_1
?beta_2

?decay
?learning_ratem?m?!m?"m?)m?*m?0m?1m?<m?=m?Cm?Dm?Km?Lm?Rm?Sm?^m?_m?em?fm?mm?nm?tm?um?	?m?	?m?	?m?	?m?v?v?!v?"v?)v?*v?0v?1v?<v?=v?Cv?Dv?Kv?Lv?Rv?Sv?^v?_v?ev?fv?mv?nv?tv?uv?	?v?	?v?	?v?	?v?"
	optimizer
 "
trackable_list_wrapper
?
0
1
!2
"3
)4
*5
06
17
<8
=9
C10
D11
K12
L13
R14
S15
^16
_17
e18
f19
m20
n21
t22
u23
?24
?25
?26
?27"
trackable_list_wrapper
?
0
1
!2
"3
#4
$5
)6
*7
08
19
210
311
<12
=13
C14
D15
E16
F17
K18
L19
R20
S21
T22
U23
^24
_25
e26
f27
g28
h29
m30
n31
t32
u33
v34
w35
?36
?37
?38
?39"
trackable_list_wrapper
?
regularization_losses
?layers
?layer_metrics
?metrics
trainable_variables
	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
+:)2conv2d_130/kernel
:2conv2d_130/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
regularization_losses
?layers
?layer_metrics
?metrics
trainable_variables
	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_40/gamma
):'2batch_normalization_40/beta
2:0 (2"batch_normalization_40/moving_mean
6:4 (2&batch_normalization_40/moving_variance
 "
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
<
!0
"1
#2
$3"
trackable_list_wrapper
?
%regularization_losses
?layers
?layer_metrics
?metrics
&trainable_variables
'	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)2conv2d_131/kernel
:2conv2d_131/bias
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
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
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_41/gamma
):'2batch_normalization_41/beta
2:0 (2"batch_normalization_41/moving_mean
6:4 (2&batch_normalization_41/moving_variance
 "
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
<
00
11
22
33"
trackable_list_wrapper
?
4regularization_losses
?layers
?layer_metrics
?metrics
5trainable_variables
6	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
8regularization_losses
?layers
?layer_metrics
?metrics
9trainable_variables
:	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)2conv2d_132/kernel
:2conv2d_132/bias
 "
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
?
>regularization_losses
?layers
?layer_metrics
?metrics
?trainable_variables
@	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_42/gamma
):'2batch_normalization_42/beta
2:0 (2"batch_normalization_42/moving_mean
6:4 (2&batch_normalization_42/moving_variance
 "
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
<
C0
D1
E2
F3"
trackable_list_wrapper
?
Gregularization_losses
?layers
?layer_metrics
?metrics
Htrainable_variables
I	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)2conv2d_133/kernel
:2conv2d_133/bias
 "
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
?
Mregularization_losses
?layers
?layer_metrics
?metrics
Ntrainable_variables
O	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_43/gamma
):'2batch_normalization_43/beta
2:0 (2"batch_normalization_43/moving_mean
6:4 (2&batch_normalization_43/moving_variance
 "
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
<
R0
S1
T2
U3"
trackable_list_wrapper
?
Vregularization_losses
?layers
?layer_metrics
?metrics
Wtrainable_variables
X	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Zregularization_losses
?layers
?layer_metrics
?metrics
[trainable_variables
\	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)2conv2d_134/kernel
:2conv2d_134/bias
 "
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
?
`regularization_losses
?layers
?layer_metrics
?metrics
atrainable_variables
b	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_44/gamma
):'2batch_normalization_44/beta
2:0 (2"batch_normalization_44/moving_mean
6:4 (2&batch_normalization_44/moving_variance
 "
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
<
e0
f1
g2
h3"
trackable_list_wrapper
?
iregularization_losses
?layers
?layer_metrics
?metrics
jtrainable_variables
k	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)2conv2d_135/kernel
:2conv2d_135/bias
 "
trackable_list_wrapper
.
m0
n1"
trackable_list_wrapper
.
m0
n1"
trackable_list_wrapper
?
oregularization_losses
?layers
?layer_metrics
?metrics
ptrainable_variables
q	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_45/gamma
):'2batch_normalization_45/beta
2:0 (2"batch_normalization_45/moving_mean
6:4 (2&batch_normalization_45/moving_variance
 "
trackable_list_wrapper
.
t0
u1"
trackable_list_wrapper
<
t0
u1
v2
w3"
trackable_list_wrapper
?
xregularization_losses
?layers
?layer_metrics
?metrics
ytrainable_variables
z	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
|regularization_losses
?layers
?layer_metrics
?metrics
}trainable_variables
~	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?layers
?layer_metrics
?metrics
?trainable_variables
?	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	? 2dense_56/kernel
: 2dense_56/bias
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
?layers
?layer_metrics
?metrics
?trainable_variables
?	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!: 2dense_57/kernel
:2dense_57/bias
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
?layers
?layer_metrics
?metrics
?trainable_variables
?	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
18"
trackable_list_wrapper
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
v
#0
$1
22
33
E4
F5
T6
U7
g8
h9
v10
w11"
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
#0
$1"
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
20
31"
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
E0
F1"
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
T0
U1"
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
g0
h1"
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
v0
w1"
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

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
0:.2Adam/conv2d_130/kernel/m
": 2Adam/conv2d_130/bias/m
/:-2#Adam/batch_normalization_40/gamma/m
.:,2"Adam/batch_normalization_40/beta/m
0:.2Adam/conv2d_131/kernel/m
": 2Adam/conv2d_131/bias/m
/:-2#Adam/batch_normalization_41/gamma/m
.:,2"Adam/batch_normalization_41/beta/m
0:.2Adam/conv2d_132/kernel/m
": 2Adam/conv2d_132/bias/m
/:-2#Adam/batch_normalization_42/gamma/m
.:,2"Adam/batch_normalization_42/beta/m
0:.2Adam/conv2d_133/kernel/m
": 2Adam/conv2d_133/bias/m
/:-2#Adam/batch_normalization_43/gamma/m
.:,2"Adam/batch_normalization_43/beta/m
0:.2Adam/conv2d_134/kernel/m
": 2Adam/conv2d_134/bias/m
/:-2#Adam/batch_normalization_44/gamma/m
.:,2"Adam/batch_normalization_44/beta/m
0:.2Adam/conv2d_135/kernel/m
": 2Adam/conv2d_135/bias/m
/:-2#Adam/batch_normalization_45/gamma/m
.:,2"Adam/batch_normalization_45/beta/m
':%	? 2Adam/dense_56/kernel/m
 : 2Adam/dense_56/bias/m
&:$ 2Adam/dense_57/kernel/m
 :2Adam/dense_57/bias/m
0:.2Adam/conv2d_130/kernel/v
": 2Adam/conv2d_130/bias/v
/:-2#Adam/batch_normalization_40/gamma/v
.:,2"Adam/batch_normalization_40/beta/v
0:.2Adam/conv2d_131/kernel/v
": 2Adam/conv2d_131/bias/v
/:-2#Adam/batch_normalization_41/gamma/v
.:,2"Adam/batch_normalization_41/beta/v
0:.2Adam/conv2d_132/kernel/v
": 2Adam/conv2d_132/bias/v
/:-2#Adam/batch_normalization_42/gamma/v
.:,2"Adam/batch_normalization_42/beta/v
0:.2Adam/conv2d_133/kernel/v
": 2Adam/conv2d_133/bias/v
/:-2#Adam/batch_normalization_43/gamma/v
.:,2"Adam/batch_normalization_43/beta/v
0:.2Adam/conv2d_134/kernel/v
": 2Adam/conv2d_134/bias/v
/:-2#Adam/batch_normalization_44/gamma/v
.:,2"Adam/batch_normalization_44/beta/v
0:.2Adam/conv2d_135/kernel/v
": 2Adam/conv2d_135/bias/v
/:-2#Adam/batch_normalization_45/gamma/v
.:,2"Adam/batch_normalization_45/beta/v
':%	? 2Adam/dense_56/kernel/v
 : 2Adam/dense_56/bias/v
&:$ 2Adam/dense_57/kernel/v
 :2Adam/dense_57/bias/v
?2?
"__inference__wrapped_model_2302214?
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
input_29???????????
?2?
O__inference_CNN_aug_deep_bnorm_layer_call_and_return_conditional_losses_2304283
O__inference_CNN_aug_deep_bnorm_layer_call_and_return_conditional_losses_2304432
O__inference_CNN_aug_deep_bnorm_layer_call_and_return_conditional_losses_2303652
O__inference_CNN_aug_deep_bnorm_layer_call_and_return_conditional_losses_2303550?
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
4__inference_CNN_aug_deep_bnorm_layer_call_fn_2303840
4__inference_CNN_aug_deep_bnorm_layer_call_fn_2304517
4__inference_CNN_aug_deep_bnorm_layer_call_fn_2304027
4__inference_CNN_aug_deep_bnorm_layer_call_fn_2304602?
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
G__inference_conv2d_130_layer_call_and_return_conditional_losses_2304613?
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
,__inference_conv2d_130_layer_call_fn_2304622?
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
S__inference_batch_normalization_40_layer_call_and_return_conditional_losses_2304660
S__inference_batch_normalization_40_layer_call_and_return_conditional_losses_2304642
S__inference_batch_normalization_40_layer_call_and_return_conditional_losses_2304724
S__inference_batch_normalization_40_layer_call_and_return_conditional_losses_2304706?
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
8__inference_batch_normalization_40_layer_call_fn_2304737
8__inference_batch_normalization_40_layer_call_fn_2304686
8__inference_batch_normalization_40_layer_call_fn_2304673
8__inference_batch_normalization_40_layer_call_fn_2304750?
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
G__inference_conv2d_131_layer_call_and_return_conditional_losses_2304761?
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
,__inference_conv2d_131_layer_call_fn_2304770?
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
S__inference_batch_normalization_41_layer_call_and_return_conditional_losses_2304808
S__inference_batch_normalization_41_layer_call_and_return_conditional_losses_2304790
S__inference_batch_normalization_41_layer_call_and_return_conditional_losses_2304854
S__inference_batch_normalization_41_layer_call_and_return_conditional_losses_2304872?
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
8__inference_batch_normalization_41_layer_call_fn_2304885
8__inference_batch_normalization_41_layer_call_fn_2304821
8__inference_batch_normalization_41_layer_call_fn_2304834
8__inference_batch_normalization_41_layer_call_fn_2304898?
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
?2?
M__inference_max_pooling2d_52_layer_call_and_return_conditional_losses_2302428?
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
2__inference_max_pooling2d_52_layer_call_fn_2302434?
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
G__inference_conv2d_132_layer_call_and_return_conditional_losses_2304909?
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
,__inference_conv2d_132_layer_call_fn_2304918?
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
S__inference_batch_normalization_42_layer_call_and_return_conditional_losses_2304956
S__inference_batch_normalization_42_layer_call_and_return_conditional_losses_2304938
S__inference_batch_normalization_42_layer_call_and_return_conditional_losses_2305002
S__inference_batch_normalization_42_layer_call_and_return_conditional_losses_2305020?
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
8__inference_batch_normalization_42_layer_call_fn_2304982
8__inference_batch_normalization_42_layer_call_fn_2305046
8__inference_batch_normalization_42_layer_call_fn_2304969
8__inference_batch_normalization_42_layer_call_fn_2305033?
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
G__inference_conv2d_133_layer_call_and_return_conditional_losses_2305057?
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
,__inference_conv2d_133_layer_call_fn_2305066?
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
S__inference_batch_normalization_43_layer_call_and_return_conditional_losses_2305086
S__inference_batch_normalization_43_layer_call_and_return_conditional_losses_2305104
S__inference_batch_normalization_43_layer_call_and_return_conditional_losses_2305150
S__inference_batch_normalization_43_layer_call_and_return_conditional_losses_2305168?
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
8__inference_batch_normalization_43_layer_call_fn_2305130
8__inference_batch_normalization_43_layer_call_fn_2305181
8__inference_batch_normalization_43_layer_call_fn_2305194
8__inference_batch_normalization_43_layer_call_fn_2305117?
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
?2?
M__inference_max_pooling2d_53_layer_call_and_return_conditional_losses_2302648?
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
2__inference_max_pooling2d_53_layer_call_fn_2302654?
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
G__inference_conv2d_134_layer_call_and_return_conditional_losses_2305205?
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
,__inference_conv2d_134_layer_call_fn_2305214?
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
S__inference_batch_normalization_44_layer_call_and_return_conditional_losses_2305316
S__inference_batch_normalization_44_layer_call_and_return_conditional_losses_2305234
S__inference_batch_normalization_44_layer_call_and_return_conditional_losses_2305298
S__inference_batch_normalization_44_layer_call_and_return_conditional_losses_2305252?
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
8__inference_batch_normalization_44_layer_call_fn_2305329
8__inference_batch_normalization_44_layer_call_fn_2305265
8__inference_batch_normalization_44_layer_call_fn_2305278
8__inference_batch_normalization_44_layer_call_fn_2305342?
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
G__inference_conv2d_135_layer_call_and_return_conditional_losses_2305353?
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
,__inference_conv2d_135_layer_call_fn_2305362?
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
S__inference_batch_normalization_45_layer_call_and_return_conditional_losses_2305382
S__inference_batch_normalization_45_layer_call_and_return_conditional_losses_2305464
S__inference_batch_normalization_45_layer_call_and_return_conditional_losses_2305446
S__inference_batch_normalization_45_layer_call_and_return_conditional_losses_2305400?
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
8__inference_batch_normalization_45_layer_call_fn_2305426
8__inference_batch_normalization_45_layer_call_fn_2305413
8__inference_batch_normalization_45_layer_call_fn_2305477
8__inference_batch_normalization_45_layer_call_fn_2305490?
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
?2?
M__inference_max_pooling2d_54_layer_call_and_return_conditional_losses_2302868?
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
2__inference_max_pooling2d_54_layer_call_fn_2302874?
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
G__inference_flatten_28_layer_call_and_return_conditional_losses_2305496?
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
,__inference_flatten_28_layer_call_fn_2305501?
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
E__inference_dense_56_layer_call_and_return_conditional_losses_2305512?
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
*__inference_dense_56_layer_call_fn_2305521?
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
E__inference_dense_57_layer_call_and_return_conditional_losses_2305532?
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
*__inference_dense_57_layer_call_fn_2305541?
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
%__inference_signature_wrapper_2304122input_29"?
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
 ?
O__inference_CNN_aug_deep_bnorm_layer_call_and_return_conditional_losses_2303550?,!"#$)*0123<=CDEFKLRSTU^_efghmntuvw????C?@
9?6
,?)
input_29???????????
p

 
? "%?"
?
0?????????
? ?
O__inference_CNN_aug_deep_bnorm_layer_call_and_return_conditional_losses_2303652?,!"#$)*0123<=CDEFKLRSTU^_efghmntuvw????C?@
9?6
,?)
input_29???????????
p 

 
? "%?"
?
0?????????
? ?
O__inference_CNN_aug_deep_bnorm_layer_call_and_return_conditional_losses_2304283?,!"#$)*0123<=CDEFKLRSTU^_efghmntuvw????A?>
7?4
*?'
inputs???????????
p

 
? "%?"
?
0?????????
? ?
O__inference_CNN_aug_deep_bnorm_layer_call_and_return_conditional_losses_2304432?,!"#$)*0123<=CDEFKLRSTU^_efghmntuvw????A?>
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
4__inference_CNN_aug_deep_bnorm_layer_call_fn_2303840?,!"#$)*0123<=CDEFKLRSTU^_efghmntuvw????C?@
9?6
,?)
input_29???????????
p

 
? "???????????
4__inference_CNN_aug_deep_bnorm_layer_call_fn_2304027?,!"#$)*0123<=CDEFKLRSTU^_efghmntuvw????C?@
9?6
,?)
input_29???????????
p 

 
? "???????????
4__inference_CNN_aug_deep_bnorm_layer_call_fn_2304517?,!"#$)*0123<=CDEFKLRSTU^_efghmntuvw????A?>
7?4
*?'
inputs???????????
p

 
? "???????????
4__inference_CNN_aug_deep_bnorm_layer_call_fn_2304602?,!"#$)*0123<=CDEFKLRSTU^_efghmntuvw????A?>
7?4
*?'
inputs???????????
p 

 
? "???????????
"__inference__wrapped_model_2302214?,!"#$)*0123<=CDEFKLRSTU^_efghmntuvw????;?8
1?.
,?)
input_29???????????
? "3?0
.
dense_57"?
dense_57??????????
S__inference_batch_normalization_40_layer_call_and_return_conditional_losses_2304642?!"#$M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_40_layer_call_and_return_conditional_losses_2304660?!"#$M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_40_layer_call_and_return_conditional_losses_2304706v!"#$=?:
3?0
*?'
inputs???????????
p
? "/?,
%?"
0???????????
? ?
S__inference_batch_normalization_40_layer_call_and_return_conditional_losses_2304724v!"#$=?:
3?0
*?'
inputs???????????
p 
? "/?,
%?"
0???????????
? ?
8__inference_batch_normalization_40_layer_call_fn_2304673?!"#$M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
8__inference_batch_normalization_40_layer_call_fn_2304686?!"#$M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
8__inference_batch_normalization_40_layer_call_fn_2304737i!"#$=?:
3?0
*?'
inputs???????????
p
? ""?????????????
8__inference_batch_normalization_40_layer_call_fn_2304750i!"#$=?:
3?0
*?'
inputs???????????
p 
? ""?????????????
S__inference_batch_normalization_41_layer_call_and_return_conditional_losses_2304790?0123M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_41_layer_call_and_return_conditional_losses_2304808?0123M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_41_layer_call_and_return_conditional_losses_2304854v0123=?:
3?0
*?'
inputs???????????
p
? "/?,
%?"
0???????????
? ?
S__inference_batch_normalization_41_layer_call_and_return_conditional_losses_2304872v0123=?:
3?0
*?'
inputs???????????
p 
? "/?,
%?"
0???????????
? ?
8__inference_batch_normalization_41_layer_call_fn_2304821?0123M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
8__inference_batch_normalization_41_layer_call_fn_2304834?0123M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
8__inference_batch_normalization_41_layer_call_fn_2304885i0123=?:
3?0
*?'
inputs???????????
p
? ""?????????????
8__inference_batch_normalization_41_layer_call_fn_2304898i0123=?:
3?0
*?'
inputs???????????
p 
? ""?????????????
S__inference_batch_normalization_42_layer_call_and_return_conditional_losses_2304938?CDEFM?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_42_layer_call_and_return_conditional_losses_2304956?CDEFM?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_42_layer_call_and_return_conditional_losses_2305002rCDEF;?8
1?.
(?%
inputs?????????**
p
? "-?*
#? 
0?????????**
? ?
S__inference_batch_normalization_42_layer_call_and_return_conditional_losses_2305020rCDEF;?8
1?.
(?%
inputs?????????**
p 
? "-?*
#? 
0?????????**
? ?
8__inference_batch_normalization_42_layer_call_fn_2304969?CDEFM?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
8__inference_batch_normalization_42_layer_call_fn_2304982?CDEFM?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
8__inference_batch_normalization_42_layer_call_fn_2305033eCDEF;?8
1?.
(?%
inputs?????????**
p
? " ??????????**?
8__inference_batch_normalization_42_layer_call_fn_2305046eCDEF;?8
1?.
(?%
inputs?????????**
p 
? " ??????????**?
S__inference_batch_normalization_43_layer_call_and_return_conditional_losses_2305086?RSTUM?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_43_layer_call_and_return_conditional_losses_2305104?RSTUM?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_43_layer_call_and_return_conditional_losses_2305150rRSTU;?8
1?.
(?%
inputs?????????**
p
? "-?*
#? 
0?????????**
? ?
S__inference_batch_normalization_43_layer_call_and_return_conditional_losses_2305168rRSTU;?8
1?.
(?%
inputs?????????**
p 
? "-?*
#? 
0?????????**
? ?
8__inference_batch_normalization_43_layer_call_fn_2305117?RSTUM?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
8__inference_batch_normalization_43_layer_call_fn_2305130?RSTUM?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
8__inference_batch_normalization_43_layer_call_fn_2305181eRSTU;?8
1?.
(?%
inputs?????????**
p
? " ??????????**?
8__inference_batch_normalization_43_layer_call_fn_2305194eRSTU;?8
1?.
(?%
inputs?????????**
p 
? " ??????????**?
S__inference_batch_normalization_44_layer_call_and_return_conditional_losses_2305234refgh;?8
1?.
(?%
inputs?????????
p
? "-?*
#? 
0?????????
? ?
S__inference_batch_normalization_44_layer_call_and_return_conditional_losses_2305252refgh;?8
1?.
(?%
inputs?????????
p 
? "-?*
#? 
0?????????
? ?
S__inference_batch_normalization_44_layer_call_and_return_conditional_losses_2305298?efghM?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_44_layer_call_and_return_conditional_losses_2305316?efghM?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
8__inference_batch_normalization_44_layer_call_fn_2305265eefgh;?8
1?.
(?%
inputs?????????
p
? " ???????????
8__inference_batch_normalization_44_layer_call_fn_2305278eefgh;?8
1?.
(?%
inputs?????????
p 
? " ???????????
8__inference_batch_normalization_44_layer_call_fn_2305329?efghM?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
8__inference_batch_normalization_44_layer_call_fn_2305342?efghM?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
S__inference_batch_normalization_45_layer_call_and_return_conditional_losses_2305382rtuvw;?8
1?.
(?%
inputs?????????
p
? "-?*
#? 
0?????????
? ?
S__inference_batch_normalization_45_layer_call_and_return_conditional_losses_2305400rtuvw;?8
1?.
(?%
inputs?????????
p 
? "-?*
#? 
0?????????
? ?
S__inference_batch_normalization_45_layer_call_and_return_conditional_losses_2305446?tuvwM?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_45_layer_call_and_return_conditional_losses_2305464?tuvwM?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
8__inference_batch_normalization_45_layer_call_fn_2305413etuvw;?8
1?.
(?%
inputs?????????
p
? " ???????????
8__inference_batch_normalization_45_layer_call_fn_2305426etuvw;?8
1?.
(?%
inputs?????????
p 
? " ???????????
8__inference_batch_normalization_45_layer_call_fn_2305477?tuvwM?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
8__inference_batch_normalization_45_layer_call_fn_2305490?tuvwM?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
G__inference_conv2d_130_layer_call_and_return_conditional_losses_2304613p9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
,__inference_conv2d_130_layer_call_fn_2304622c9?6
/?,
*?'
inputs???????????
? ""?????????????
G__inference_conv2d_131_layer_call_and_return_conditional_losses_2304761p)*9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
,__inference_conv2d_131_layer_call_fn_2304770c)*9?6
/?,
*?'
inputs???????????
? ""?????????????
G__inference_conv2d_132_layer_call_and_return_conditional_losses_2304909l<=7?4
-?*
(?%
inputs?????????**
? "-?*
#? 
0?????????**
? ?
,__inference_conv2d_132_layer_call_fn_2304918_<=7?4
-?*
(?%
inputs?????????**
? " ??????????**?
G__inference_conv2d_133_layer_call_and_return_conditional_losses_2305057lKL7?4
-?*
(?%
inputs?????????**
? "-?*
#? 
0?????????**
? ?
,__inference_conv2d_133_layer_call_fn_2305066_KL7?4
-?*
(?%
inputs?????????**
? " ??????????**?
G__inference_conv2d_134_layer_call_and_return_conditional_losses_2305205l^_7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
,__inference_conv2d_134_layer_call_fn_2305214_^_7?4
-?*
(?%
inputs?????????
? " ???????????
G__inference_conv2d_135_layer_call_and_return_conditional_losses_2305353lmn7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
,__inference_conv2d_135_layer_call_fn_2305362_mn7?4
-?*
(?%
inputs?????????
? " ???????????
E__inference_dense_56_layer_call_and_return_conditional_losses_2305512_??0?-
&?#
!?
inputs??????????
? "%?"
?
0????????? 
? ?
*__inference_dense_56_layer_call_fn_2305521R??0?-
&?#
!?
inputs??????????
? "?????????? ?
E__inference_dense_57_layer_call_and_return_conditional_losses_2305532^??/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????
? 
*__inference_dense_57_layer_call_fn_2305541Q??/?,
%?"
 ?
inputs????????? 
? "???????????
G__inference_flatten_28_layer_call_and_return_conditional_losses_2305496a7?4
-?*
(?%
inputs?????????
? "&?#
?
0??????????
? ?
,__inference_flatten_28_layer_call_fn_2305501T7?4
-?*
(?%
inputs?????????
? "????????????
M__inference_max_pooling2d_52_layer_call_and_return_conditional_losses_2302428?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
2__inference_max_pooling2d_52_layer_call_fn_2302434?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
M__inference_max_pooling2d_53_layer_call_and_return_conditional_losses_2302648?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
2__inference_max_pooling2d_53_layer_call_fn_2302654?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
M__inference_max_pooling2d_54_layer_call_and_return_conditional_losses_2302868?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
2__inference_max_pooling2d_54_layer_call_fn_2302874?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
%__inference_signature_wrapper_2304122?,!"#$)*0123<=CDEFKLRSTU^_efghmntuvw????G?D
? 
=?:
8
input_29,?)
input_29???????????"3?0
.
dense_57"?
dense_57?????????