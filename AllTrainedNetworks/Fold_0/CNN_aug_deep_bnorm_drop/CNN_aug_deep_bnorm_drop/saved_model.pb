??'
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
conv2d_145/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_145/kernel

%conv2d_145/kernel/Read/ReadVariableOpReadVariableOpconv2d_145/kernel*&
_output_shapes
:*
dtype0
v
conv2d_145/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_145/bias
o
#conv2d_145/bias/Read/ReadVariableOpReadVariableOpconv2d_145/bias*
_output_shapes
:*
dtype0
?
batch_normalization_52/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_52/gamma
?
0batch_normalization_52/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_52/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_52/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_52/beta
?
/batch_normalization_52/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_52/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_52/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_52/moving_mean
?
6batch_normalization_52/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_52/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_52/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_52/moving_variance
?
:batch_normalization_52/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_52/moving_variance*
_output_shapes
:*
dtype0
?
conv2d_146/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_146/kernel

%conv2d_146/kernel/Read/ReadVariableOpReadVariableOpconv2d_146/kernel*&
_output_shapes
:*
dtype0
v
conv2d_146/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_146/bias
o
#conv2d_146/bias/Read/ReadVariableOpReadVariableOpconv2d_146/bias*
_output_shapes
:*
dtype0
?
batch_normalization_53/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_53/gamma
?
0batch_normalization_53/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_53/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_53/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_53/beta
?
/batch_normalization_53/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_53/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_53/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_53/moving_mean
?
6batch_normalization_53/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_53/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_53/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_53/moving_variance
?
:batch_normalization_53/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_53/moving_variance*
_output_shapes
:*
dtype0
?
conv2d_147/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_147/kernel

%conv2d_147/kernel/Read/ReadVariableOpReadVariableOpconv2d_147/kernel*&
_output_shapes
:*
dtype0
v
conv2d_147/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_147/bias
o
#conv2d_147/bias/Read/ReadVariableOpReadVariableOpconv2d_147/bias*
_output_shapes
:*
dtype0
?
batch_normalization_54/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_54/gamma
?
0batch_normalization_54/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_54/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_54/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_54/beta
?
/batch_normalization_54/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_54/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_54/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_54/moving_mean
?
6batch_normalization_54/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_54/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_54/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_54/moving_variance
?
:batch_normalization_54/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_54/moving_variance*
_output_shapes
:*
dtype0
?
conv2d_148/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_148/kernel

%conv2d_148/kernel/Read/ReadVariableOpReadVariableOpconv2d_148/kernel*&
_output_shapes
:*
dtype0
v
conv2d_148/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_148/bias
o
#conv2d_148/bias/Read/ReadVariableOpReadVariableOpconv2d_148/bias*
_output_shapes
:*
dtype0
?
batch_normalization_55/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_55/gamma
?
0batch_normalization_55/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_55/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_55/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_55/beta
?
/batch_normalization_55/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_55/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_55/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_55/moving_mean
?
6batch_normalization_55/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_55/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_55/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_55/moving_variance
?
:batch_normalization_55/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_55/moving_variance*
_output_shapes
:*
dtype0
?
conv2d_149/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_149/kernel

%conv2d_149/kernel/Read/ReadVariableOpReadVariableOpconv2d_149/kernel*&
_output_shapes
:*
dtype0
v
conv2d_149/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_149/bias
o
#conv2d_149/bias/Read/ReadVariableOpReadVariableOpconv2d_149/bias*
_output_shapes
:*
dtype0
?
batch_normalization_56/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_56/gamma
?
0batch_normalization_56/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_56/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_56/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_56/beta
?
/batch_normalization_56/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_56/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_56/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_56/moving_mean
?
6batch_normalization_56/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_56/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_56/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_56/moving_variance
?
:batch_normalization_56/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_56/moving_variance*
_output_shapes
:*
dtype0
?
conv2d_150/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_150/kernel

%conv2d_150/kernel/Read/ReadVariableOpReadVariableOpconv2d_150/kernel*&
_output_shapes
:*
dtype0
v
conv2d_150/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_150/bias
o
#conv2d_150/bias/Read/ReadVariableOpReadVariableOpconv2d_150/bias*
_output_shapes
:*
dtype0
?
batch_normalization_57/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_57/gamma
?
0batch_normalization_57/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_57/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_57/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_57/beta
?
/batch_normalization_57/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_57/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_57/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_57/moving_mean
?
6batch_normalization_57/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_57/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_57/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_57/moving_variance
?
:batch_normalization_57/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_57/moving_variance*
_output_shapes
:*
dtype0
{
dense_60/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	? * 
shared_namedense_60/kernel
t
#dense_60/kernel/Read/ReadVariableOpReadVariableOpdense_60/kernel*
_output_shapes
:	? *
dtype0
r
dense_60/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_60/bias
k
!dense_60/bias/Read/ReadVariableOpReadVariableOpdense_60/bias*
_output_shapes
: *
dtype0
z
dense_61/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_61/kernel
s
#dense_61/kernel/Read/ReadVariableOpReadVariableOpdense_61/kernel*
_output_shapes

: *
dtype0
r
dense_61/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_61/bias
k
!dense_61/bias/Read/ReadVariableOpReadVariableOpdense_61/bias*
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
Adam/conv2d_145/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_145/kernel/m
?
,Adam/conv2d_145/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_145/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_145/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_145/bias/m
}
*Adam/conv2d_145/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_145/bias/m*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_52/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_52/gamma/m
?
7Adam/batch_normalization_52/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_52/gamma/m*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_52/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_52/beta/m
?
6Adam/batch_normalization_52/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_52/beta/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_146/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_146/kernel/m
?
,Adam/conv2d_146/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_146/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_146/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_146/bias/m
}
*Adam/conv2d_146/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_146/bias/m*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_53/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_53/gamma/m
?
7Adam/batch_normalization_53/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_53/gamma/m*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_53/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_53/beta/m
?
6Adam/batch_normalization_53/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_53/beta/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_147/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_147/kernel/m
?
,Adam/conv2d_147/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_147/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_147/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_147/bias/m
}
*Adam/conv2d_147/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_147/bias/m*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_54/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_54/gamma/m
?
7Adam/batch_normalization_54/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_54/gamma/m*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_54/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_54/beta/m
?
6Adam/batch_normalization_54/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_54/beta/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_148/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_148/kernel/m
?
,Adam/conv2d_148/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_148/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_148/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_148/bias/m
}
*Adam/conv2d_148/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_148/bias/m*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_55/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_55/gamma/m
?
7Adam/batch_normalization_55/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_55/gamma/m*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_55/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_55/beta/m
?
6Adam/batch_normalization_55/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_55/beta/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_149/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_149/kernel/m
?
,Adam/conv2d_149/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_149/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_149/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_149/bias/m
}
*Adam/conv2d_149/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_149/bias/m*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_56/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_56/gamma/m
?
7Adam/batch_normalization_56/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_56/gamma/m*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_56/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_56/beta/m
?
6Adam/batch_normalization_56/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_56/beta/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_150/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_150/kernel/m
?
,Adam/conv2d_150/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_150/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_150/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_150/bias/m
}
*Adam/conv2d_150/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_150/bias/m*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_57/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_57/gamma/m
?
7Adam/batch_normalization_57/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_57/gamma/m*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_57/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_57/beta/m
?
6Adam/batch_normalization_57/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_57/beta/m*
_output_shapes
:*
dtype0
?
Adam/dense_60/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	? *'
shared_nameAdam/dense_60/kernel/m
?
*Adam/dense_60/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_60/kernel/m*
_output_shapes
:	? *
dtype0
?
Adam/dense_60/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_60/bias/m
y
(Adam/dense_60/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_60/bias/m*
_output_shapes
: *
dtype0
?
Adam/dense_61/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_61/kernel/m
?
*Adam/dense_61/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_61/kernel/m*
_output_shapes

: *
dtype0
?
Adam/dense_61/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_61/bias/m
y
(Adam/dense_61/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_61/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_145/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_145/kernel/v
?
,Adam/conv2d_145/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_145/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_145/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_145/bias/v
}
*Adam/conv2d_145/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_145/bias/v*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_52/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_52/gamma/v
?
7Adam/batch_normalization_52/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_52/gamma/v*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_52/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_52/beta/v
?
6Adam/batch_normalization_52/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_52/beta/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_146/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_146/kernel/v
?
,Adam/conv2d_146/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_146/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_146/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_146/bias/v
}
*Adam/conv2d_146/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_146/bias/v*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_53/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_53/gamma/v
?
7Adam/batch_normalization_53/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_53/gamma/v*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_53/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_53/beta/v
?
6Adam/batch_normalization_53/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_53/beta/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_147/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_147/kernel/v
?
,Adam/conv2d_147/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_147/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_147/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_147/bias/v
}
*Adam/conv2d_147/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_147/bias/v*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_54/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_54/gamma/v
?
7Adam/batch_normalization_54/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_54/gamma/v*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_54/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_54/beta/v
?
6Adam/batch_normalization_54/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_54/beta/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_148/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_148/kernel/v
?
,Adam/conv2d_148/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_148/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_148/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_148/bias/v
}
*Adam/conv2d_148/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_148/bias/v*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_55/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_55/gamma/v
?
7Adam/batch_normalization_55/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_55/gamma/v*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_55/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_55/beta/v
?
6Adam/batch_normalization_55/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_55/beta/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_149/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_149/kernel/v
?
,Adam/conv2d_149/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_149/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_149/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_149/bias/v
}
*Adam/conv2d_149/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_149/bias/v*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_56/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_56/gamma/v
?
7Adam/batch_normalization_56/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_56/gamma/v*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_56/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_56/beta/v
?
6Adam/batch_normalization_56/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_56/beta/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_150/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_150/kernel/v
?
,Adam/conv2d_150/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_150/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_150/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_150/bias/v
}
*Adam/conv2d_150/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_150/bias/v*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_57/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_57/gamma/v
?
7Adam/batch_normalization_57/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_57/gamma/v*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_57/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_57/beta/v
?
6Adam/batch_normalization_57/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_57/beta/v*
_output_shapes
:*
dtype0
?
Adam/dense_60/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	? *'
shared_nameAdam/dense_60/kernel/v
?
*Adam/dense_60/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_60/kernel/v*
_output_shapes
:	? *
dtype0
?
Adam/dense_60/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_60/bias/v
y
(Adam/dense_60/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_60/bias/v*
_output_shapes
: *
dtype0
?
Adam/dense_61/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_61/kernel/v
?
*Adam/dense_61/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_61/kernel/v*
_output_shapes

: *
dtype0
?
Adam/dense_61/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_61/bias/v
y
(Adam/dense_61/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_61/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B۴
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer_with_weights-6
layer-11
layer_with_weights-7
layer-12
layer-13
layer-14
layer_with_weights-8
layer-15
layer_with_weights-9
layer-16
layer-17
layer_with_weights-10
layer-18
layer_with_weights-11
layer-19
layer-20
layer-21
layer-22
layer_with_weights-12
layer-23
layer_with_weights-13
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
R
/regularization_losses
0trainable_variables
1	variables
2	keras_api
h

3kernel
4bias
5regularization_losses
6trainable_variables
7	variables
8	keras_api
?
9axis
	:gamma
;beta
<moving_mean
=moving_variance
>regularization_losses
?trainable_variables
@	variables
A	keras_api
R
Bregularization_losses
Ctrainable_variables
D	variables
E	keras_api
R
Fregularization_losses
Gtrainable_variables
H	variables
I	keras_api
h

Jkernel
Kbias
Lregularization_losses
Mtrainable_variables
N	variables
O	keras_api
?
Paxis
	Qgamma
Rbeta
Smoving_mean
Tmoving_variance
Uregularization_losses
Vtrainable_variables
W	variables
X	keras_api
R
Yregularization_losses
Ztrainable_variables
[	variables
\	keras_api
h

]kernel
^bias
_regularization_losses
`trainable_variables
a	variables
b	keras_api
?
caxis
	dgamma
ebeta
fmoving_mean
gmoving_variance
hregularization_losses
itrainable_variables
j	variables
k	keras_api
R
lregularization_losses
mtrainable_variables
n	variables
o	keras_api
R
pregularization_losses
qtrainable_variables
r	variables
s	keras_api
h

tkernel
ubias
vregularization_losses
wtrainable_variables
x	variables
y	keras_api
?
zaxis
	{gamma
|beta
}moving_mean
~moving_variance
regularization_losses
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
?
	?iter
?beta_1
?beta_2

?decay
?learning_rate m?!m?'m?(m?3m?4m?:m?;m?Jm?Km?Qm?Rm?]m?^m?dm?em?tm?um?{m?|m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m? v?!v?'v?(v?3v?4v?:v?;v?Jv?Kv?Qv?Rv?]v?^v?dv?ev?tv?uv?{v?|v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?
 
?
 0
!1
'2
(3
34
45
:6
;7
J8
K9
Q10
R11
]12
^13
d14
e15
t16
u17
{18
|19
?20
?21
?22
?23
?24
?25
?26
?27
?
 0
!1
'2
(3
)4
*5
36
47
:8
;9
<10
=11
J12
K13
Q14
R15
S16
T17
]18
^19
d20
e21
f22
g23
t24
u25
{26
|27
}28
~29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
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
VARIABLE_VALUEconv2d_145/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_145/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEbatch_normalization_52/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_52/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_52/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_52/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
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
 
 
 
?
/regularization_losses
?layers
?layer_metrics
?metrics
0trainable_variables
1	variables
 ?layer_regularization_losses
?non_trainable_variables
][
VARIABLE_VALUEconv2d_146/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_146/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

30
41

30
41
?
5regularization_losses
?layers
?layer_metrics
?metrics
6trainable_variables
7	variables
 ?layer_regularization_losses
?non_trainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_53/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_53/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_53/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_53/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

:0
;1

:0
;1
<2
=3
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
 
 
?
Bregularization_losses
?layers
?layer_metrics
?metrics
Ctrainable_variables
D	variables
 ?layer_regularization_losses
?non_trainable_variables
 
 
 
?
Fregularization_losses
?layers
?layer_metrics
?metrics
Gtrainable_variables
H	variables
 ?layer_regularization_losses
?non_trainable_variables
][
VARIABLE_VALUEconv2d_147/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_147/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

J0
K1

J0
K1
?
Lregularization_losses
?layers
?layer_metrics
?metrics
Mtrainable_variables
N	variables
 ?layer_regularization_losses
?non_trainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_54/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_54/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_54/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_54/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

Q0
R1

Q0
R1
S2
T3
?
Uregularization_losses
?layers
?layer_metrics
?metrics
Vtrainable_variables
W	variables
 ?layer_regularization_losses
?non_trainable_variables
 
 
 
?
Yregularization_losses
?layers
?layer_metrics
?metrics
Ztrainable_variables
[	variables
 ?layer_regularization_losses
?non_trainable_variables
][
VARIABLE_VALUEconv2d_148/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_148/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

]0
^1

]0
^1
?
_regularization_losses
?layers
?layer_metrics
?metrics
`trainable_variables
a	variables
 ?layer_regularization_losses
?non_trainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_55/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_55/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_55/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_55/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

d0
e1

d0
e1
f2
g3
?
hregularization_losses
?layers
?layer_metrics
?metrics
itrainable_variables
j	variables
 ?layer_regularization_losses
?non_trainable_variables
 
 
 
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
][
VARIABLE_VALUEconv2d_149/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_149/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
 

t0
u1

t0
u1
?
vregularization_losses
?layers
?layer_metrics
?metrics
wtrainable_variables
x	variables
 ?layer_regularization_losses
?non_trainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_56/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_56/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_56/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_56/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

{0
|1

{0
|1
}2
~3
?
regularization_losses
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
^\
VARIABLE_VALUEconv2d_150/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEconv2d_150/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEbatch_normalization_57/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_57/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_57/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_57/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEdense_60/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_60/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEdense_61/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_61/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE
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
X
)0
*1
<2
=3
S4
T5
f6
g7
}8
~9
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
 
 
 
 
 

<0
=1
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
S0
T1
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
f0
g1
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
}0
~1
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
VARIABLE_VALUEAdam/conv2d_145/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_145/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_52/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_52/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_146/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_146/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_53/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_53/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_147/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_147/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_54/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_54/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_148/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_148/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_55/gamma/mQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_55/beta/mPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_149/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_149/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_56/gamma/mQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_56/beta/mPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/conv2d_150/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/conv2d_150/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_57/gamma/mRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_57/beta/mQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_60/kernel/mSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_60/bias/mQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_61/kernel/mSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_61/bias/mQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_145/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_145/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_52/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_52/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_146/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_146/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_53/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_53/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_147/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_147/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_54/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_54/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_148/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_148/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_55/gamma/vQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_55/beta/vPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_149/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_149/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_56/gamma/vQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_56/beta/vPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/conv2d_150/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/conv2d_150/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_57/gamma/vRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_57/beta/vQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_60/kernel/vSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_60/bias/vQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_61/kernel/vSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_61/bias/vQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_31Placeholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_31conv2d_145/kernelconv2d_145/biasbatch_normalization_52/gammabatch_normalization_52/beta"batch_normalization_52/moving_mean&batch_normalization_52/moving_varianceconv2d_146/kernelconv2d_146/biasbatch_normalization_53/gammabatch_normalization_53/beta"batch_normalization_53/moving_mean&batch_normalization_53/moving_varianceconv2d_147/kernelconv2d_147/biasbatch_normalization_54/gammabatch_normalization_54/beta"batch_normalization_54/moving_mean&batch_normalization_54/moving_varianceconv2d_148/kernelconv2d_148/biasbatch_normalization_55/gammabatch_normalization_55/beta"batch_normalization_55/moving_mean&batch_normalization_55/moving_varianceconv2d_149/kernelconv2d_149/biasbatch_normalization_56/gammabatch_normalization_56/beta"batch_normalization_56/moving_mean&batch_normalization_56/moving_varianceconv2d_150/kernelconv2d_150/biasbatch_normalization_57/gammabatch_normalization_57/beta"batch_normalization_57/moving_mean&batch_normalization_57/moving_variancedense_60/kerneldense_60/biasdense_61/kerneldense_61/bias*4
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
%__inference_signature_wrapper_2706757
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?)
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%conv2d_145/kernel/Read/ReadVariableOp#conv2d_145/bias/Read/ReadVariableOp0batch_normalization_52/gamma/Read/ReadVariableOp/batch_normalization_52/beta/Read/ReadVariableOp6batch_normalization_52/moving_mean/Read/ReadVariableOp:batch_normalization_52/moving_variance/Read/ReadVariableOp%conv2d_146/kernel/Read/ReadVariableOp#conv2d_146/bias/Read/ReadVariableOp0batch_normalization_53/gamma/Read/ReadVariableOp/batch_normalization_53/beta/Read/ReadVariableOp6batch_normalization_53/moving_mean/Read/ReadVariableOp:batch_normalization_53/moving_variance/Read/ReadVariableOp%conv2d_147/kernel/Read/ReadVariableOp#conv2d_147/bias/Read/ReadVariableOp0batch_normalization_54/gamma/Read/ReadVariableOp/batch_normalization_54/beta/Read/ReadVariableOp6batch_normalization_54/moving_mean/Read/ReadVariableOp:batch_normalization_54/moving_variance/Read/ReadVariableOp%conv2d_148/kernel/Read/ReadVariableOp#conv2d_148/bias/Read/ReadVariableOp0batch_normalization_55/gamma/Read/ReadVariableOp/batch_normalization_55/beta/Read/ReadVariableOp6batch_normalization_55/moving_mean/Read/ReadVariableOp:batch_normalization_55/moving_variance/Read/ReadVariableOp%conv2d_149/kernel/Read/ReadVariableOp#conv2d_149/bias/Read/ReadVariableOp0batch_normalization_56/gamma/Read/ReadVariableOp/batch_normalization_56/beta/Read/ReadVariableOp6batch_normalization_56/moving_mean/Read/ReadVariableOp:batch_normalization_56/moving_variance/Read/ReadVariableOp%conv2d_150/kernel/Read/ReadVariableOp#conv2d_150/bias/Read/ReadVariableOp0batch_normalization_57/gamma/Read/ReadVariableOp/batch_normalization_57/beta/Read/ReadVariableOp6batch_normalization_57/moving_mean/Read/ReadVariableOp:batch_normalization_57/moving_variance/Read/ReadVariableOp#dense_60/kernel/Read/ReadVariableOp!dense_60/bias/Read/ReadVariableOp#dense_61/kernel/Read/ReadVariableOp!dense_61/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp,Adam/conv2d_145/kernel/m/Read/ReadVariableOp*Adam/conv2d_145/bias/m/Read/ReadVariableOp7Adam/batch_normalization_52/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_52/beta/m/Read/ReadVariableOp,Adam/conv2d_146/kernel/m/Read/ReadVariableOp*Adam/conv2d_146/bias/m/Read/ReadVariableOp7Adam/batch_normalization_53/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_53/beta/m/Read/ReadVariableOp,Adam/conv2d_147/kernel/m/Read/ReadVariableOp*Adam/conv2d_147/bias/m/Read/ReadVariableOp7Adam/batch_normalization_54/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_54/beta/m/Read/ReadVariableOp,Adam/conv2d_148/kernel/m/Read/ReadVariableOp*Adam/conv2d_148/bias/m/Read/ReadVariableOp7Adam/batch_normalization_55/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_55/beta/m/Read/ReadVariableOp,Adam/conv2d_149/kernel/m/Read/ReadVariableOp*Adam/conv2d_149/bias/m/Read/ReadVariableOp7Adam/batch_normalization_56/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_56/beta/m/Read/ReadVariableOp,Adam/conv2d_150/kernel/m/Read/ReadVariableOp*Adam/conv2d_150/bias/m/Read/ReadVariableOp7Adam/batch_normalization_57/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_57/beta/m/Read/ReadVariableOp*Adam/dense_60/kernel/m/Read/ReadVariableOp(Adam/dense_60/bias/m/Read/ReadVariableOp*Adam/dense_61/kernel/m/Read/ReadVariableOp(Adam/dense_61/bias/m/Read/ReadVariableOp,Adam/conv2d_145/kernel/v/Read/ReadVariableOp*Adam/conv2d_145/bias/v/Read/ReadVariableOp7Adam/batch_normalization_52/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_52/beta/v/Read/ReadVariableOp,Adam/conv2d_146/kernel/v/Read/ReadVariableOp*Adam/conv2d_146/bias/v/Read/ReadVariableOp7Adam/batch_normalization_53/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_53/beta/v/Read/ReadVariableOp,Adam/conv2d_147/kernel/v/Read/ReadVariableOp*Adam/conv2d_147/bias/v/Read/ReadVariableOp7Adam/batch_normalization_54/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_54/beta/v/Read/ReadVariableOp,Adam/conv2d_148/kernel/v/Read/ReadVariableOp*Adam/conv2d_148/bias/v/Read/ReadVariableOp7Adam/batch_normalization_55/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_55/beta/v/Read/ReadVariableOp,Adam/conv2d_149/kernel/v/Read/ReadVariableOp*Adam/conv2d_149/bias/v/Read/ReadVariableOp7Adam/batch_normalization_56/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_56/beta/v/Read/ReadVariableOp,Adam/conv2d_150/kernel/v/Read/ReadVariableOp*Adam/conv2d_150/bias/v/Read/ReadVariableOp7Adam/batch_normalization_57/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_57/beta/v/Read/ReadVariableOp*Adam/dense_60/kernel/v/Read/ReadVariableOp(Adam/dense_60/bias/v/Read/ReadVariableOp*Adam/dense_61/kernel/v/Read/ReadVariableOp(Adam/dense_61/bias/v/Read/ReadVariableOpConst*v
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
 __inference__traced_save_2708730
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_145/kernelconv2d_145/biasbatch_normalization_52/gammabatch_normalization_52/beta"batch_normalization_52/moving_mean&batch_normalization_52/moving_varianceconv2d_146/kernelconv2d_146/biasbatch_normalization_53/gammabatch_normalization_53/beta"batch_normalization_53/moving_mean&batch_normalization_53/moving_varianceconv2d_147/kernelconv2d_147/biasbatch_normalization_54/gammabatch_normalization_54/beta"batch_normalization_54/moving_mean&batch_normalization_54/moving_varianceconv2d_148/kernelconv2d_148/biasbatch_normalization_55/gammabatch_normalization_55/beta"batch_normalization_55/moving_mean&batch_normalization_55/moving_varianceconv2d_149/kernelconv2d_149/biasbatch_normalization_56/gammabatch_normalization_56/beta"batch_normalization_56/moving_mean&batch_normalization_56/moving_varianceconv2d_150/kernelconv2d_150/biasbatch_normalization_57/gammabatch_normalization_57/beta"batch_normalization_57/moving_mean&batch_normalization_57/moving_variancedense_60/kerneldense_60/biasdense_61/kerneldense_61/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv2d_145/kernel/mAdam/conv2d_145/bias/m#Adam/batch_normalization_52/gamma/m"Adam/batch_normalization_52/beta/mAdam/conv2d_146/kernel/mAdam/conv2d_146/bias/m#Adam/batch_normalization_53/gamma/m"Adam/batch_normalization_53/beta/mAdam/conv2d_147/kernel/mAdam/conv2d_147/bias/m#Adam/batch_normalization_54/gamma/m"Adam/batch_normalization_54/beta/mAdam/conv2d_148/kernel/mAdam/conv2d_148/bias/m#Adam/batch_normalization_55/gamma/m"Adam/batch_normalization_55/beta/mAdam/conv2d_149/kernel/mAdam/conv2d_149/bias/m#Adam/batch_normalization_56/gamma/m"Adam/batch_normalization_56/beta/mAdam/conv2d_150/kernel/mAdam/conv2d_150/bias/m#Adam/batch_normalization_57/gamma/m"Adam/batch_normalization_57/beta/mAdam/dense_60/kernel/mAdam/dense_60/bias/mAdam/dense_61/kernel/mAdam/dense_61/bias/mAdam/conv2d_145/kernel/vAdam/conv2d_145/bias/v#Adam/batch_normalization_52/gamma/v"Adam/batch_normalization_52/beta/vAdam/conv2d_146/kernel/vAdam/conv2d_146/bias/v#Adam/batch_normalization_53/gamma/v"Adam/batch_normalization_53/beta/vAdam/conv2d_147/kernel/vAdam/conv2d_147/bias/v#Adam/batch_normalization_54/gamma/v"Adam/batch_normalization_54/beta/vAdam/conv2d_148/kernel/vAdam/conv2d_148/bias/v#Adam/batch_normalization_55/gamma/v"Adam/batch_normalization_55/beta/vAdam/conv2d_149/kernel/vAdam/conv2d_149/bias/v#Adam/batch_normalization_56/gamma/v"Adam/batch_normalization_56/beta/vAdam/conv2d_150/kernel/vAdam/conv2d_150/bias/v#Adam/batch_normalization_57/gamma/v"Adam/batch_normalization_57/beta/vAdam/dense_60/kernel/vAdam/dense_60/bias/vAdam/dense_61/kernel/vAdam/dense_61/bias/v*u
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
#__inference__traced_restore_2709055??
?
i
M__inference_max_pooling2d_58_layer_call_and_return_conditional_losses_2704865

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
?
i
M__inference_max_pooling2d_59_layer_call_and_return_conditional_losses_2705085

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
??
?#
T__inference_CNN_aug_deep_bnorm_drop_layer_call_and_return_conditional_losses_2706966

inputs-
)conv2d_145_conv2d_readvariableop_resource.
*conv2d_145_biasadd_readvariableop_resource2
.batch_normalization_52_readvariableop_resource4
0batch_normalization_52_readvariableop_1_resourceC
?batch_normalization_52_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_52_fusedbatchnormv3_readvariableop_1_resource-
)conv2d_146_conv2d_readvariableop_resource.
*conv2d_146_biasadd_readvariableop_resource2
.batch_normalization_53_readvariableop_resource4
0batch_normalization_53_readvariableop_1_resourceC
?batch_normalization_53_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_53_fusedbatchnormv3_readvariableop_1_resource-
)conv2d_147_conv2d_readvariableop_resource.
*conv2d_147_biasadd_readvariableop_resource2
.batch_normalization_54_readvariableop_resource4
0batch_normalization_54_readvariableop_1_resourceC
?batch_normalization_54_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_54_fusedbatchnormv3_readvariableop_1_resource-
)conv2d_148_conv2d_readvariableop_resource.
*conv2d_148_biasadd_readvariableop_resource2
.batch_normalization_55_readvariableop_resource4
0batch_normalization_55_readvariableop_1_resourceC
?batch_normalization_55_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_55_fusedbatchnormv3_readvariableop_1_resource-
)conv2d_149_conv2d_readvariableop_resource.
*conv2d_149_biasadd_readvariableop_resource2
.batch_normalization_56_readvariableop_resource4
0batch_normalization_56_readvariableop_1_resourceC
?batch_normalization_56_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_56_fusedbatchnormv3_readvariableop_1_resource-
)conv2d_150_conv2d_readvariableop_resource.
*conv2d_150_biasadd_readvariableop_resource2
.batch_normalization_57_readvariableop_resource4
0batch_normalization_57_readvariableop_1_resourceC
?batch_normalization_57_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_57_fusedbatchnormv3_readvariableop_1_resource+
'dense_60_matmul_readvariableop_resource,
(dense_60_biasadd_readvariableop_resource+
'dense_61_matmul_readvariableop_resource,
(dense_61_biasadd_readvariableop_resource
identity??%batch_normalization_52/AssignNewValue?'batch_normalization_52/AssignNewValue_1?6batch_normalization_52/FusedBatchNormV3/ReadVariableOp?8batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_52/ReadVariableOp?'batch_normalization_52/ReadVariableOp_1?%batch_normalization_53/AssignNewValue?'batch_normalization_53/AssignNewValue_1?6batch_normalization_53/FusedBatchNormV3/ReadVariableOp?8batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_53/ReadVariableOp?'batch_normalization_53/ReadVariableOp_1?%batch_normalization_54/AssignNewValue?'batch_normalization_54/AssignNewValue_1?6batch_normalization_54/FusedBatchNormV3/ReadVariableOp?8batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_54/ReadVariableOp?'batch_normalization_54/ReadVariableOp_1?%batch_normalization_55/AssignNewValue?'batch_normalization_55/AssignNewValue_1?6batch_normalization_55/FusedBatchNormV3/ReadVariableOp?8batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_55/ReadVariableOp?'batch_normalization_55/ReadVariableOp_1?%batch_normalization_56/AssignNewValue?'batch_normalization_56/AssignNewValue_1?6batch_normalization_56/FusedBatchNormV3/ReadVariableOp?8batch_normalization_56/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_56/ReadVariableOp?'batch_normalization_56/ReadVariableOp_1?%batch_normalization_57/AssignNewValue?'batch_normalization_57/AssignNewValue_1?6batch_normalization_57/FusedBatchNormV3/ReadVariableOp?8batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_57/ReadVariableOp?'batch_normalization_57/ReadVariableOp_1?!conv2d_145/BiasAdd/ReadVariableOp? conv2d_145/Conv2D/ReadVariableOp?!conv2d_146/BiasAdd/ReadVariableOp? conv2d_146/Conv2D/ReadVariableOp?!conv2d_147/BiasAdd/ReadVariableOp? conv2d_147/Conv2D/ReadVariableOp?!conv2d_148/BiasAdd/ReadVariableOp? conv2d_148/Conv2D/ReadVariableOp?!conv2d_149/BiasAdd/ReadVariableOp? conv2d_149/Conv2D/ReadVariableOp?!conv2d_150/BiasAdd/ReadVariableOp? conv2d_150/Conv2D/ReadVariableOp?dense_60/BiasAdd/ReadVariableOp?dense_60/MatMul/ReadVariableOp?dense_61/BiasAdd/ReadVariableOp?dense_61/MatMul/ReadVariableOp?
 conv2d_145/Conv2D/ReadVariableOpReadVariableOp)conv2d_145_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_145/Conv2D/ReadVariableOp?
conv2d_145/Conv2DConv2Dinputs(conv2d_145/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_145/Conv2D?
!conv2d_145/BiasAdd/ReadVariableOpReadVariableOp*conv2d_145_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_145/BiasAdd/ReadVariableOp?
conv2d_145/BiasAddBiasAddconv2d_145/Conv2D:output:0)conv2d_145/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_145/BiasAdd?
conv2d_145/ReluReluconv2d_145/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_145/Relu?
%batch_normalization_52/ReadVariableOpReadVariableOp.batch_normalization_52_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_52/ReadVariableOp?
'batch_normalization_52/ReadVariableOp_1ReadVariableOp0batch_normalization_52_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_52/ReadVariableOp_1?
6batch_normalization_52/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_52_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_52/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_52_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_52/FusedBatchNormV3FusedBatchNormV3conv2d_145/Relu:activations:0-batch_normalization_52/ReadVariableOp:value:0/batch_normalization_52/ReadVariableOp_1:value:0>batch_normalization_52/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_52/FusedBatchNormV3?
%batch_normalization_52/AssignNewValueAssignVariableOp?batch_normalization_52_fusedbatchnormv3_readvariableop_resource4batch_normalization_52/FusedBatchNormV3:batch_mean:07^batch_normalization_52/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*R
_classH
FDloc:@batch_normalization_52/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_52/AssignNewValue?
'batch_normalization_52/AssignNewValue_1AssignVariableOpAbatch_normalization_52_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_52/FusedBatchNormV3:batch_variance:09^batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_52/AssignNewValue_1y
dropout_52/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_52/dropout/Const?
dropout_52/dropout/MulMul+batch_normalization_52/FusedBatchNormV3:y:0!dropout_52/dropout/Const:output:0*
T0*1
_output_shapes
:???????????2
dropout_52/dropout/Mul?
dropout_52/dropout/ShapeShape+batch_normalization_52/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
dropout_52/dropout/Shape?
/dropout_52/dropout/random_uniform/RandomUniformRandomUniform!dropout_52/dropout/Shape:output:0*
T0*1
_output_shapes
:???????????*
dtype021
/dropout_52/dropout/random_uniform/RandomUniform?
!dropout_52/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2#
!dropout_52/dropout/GreaterEqual/y?
dropout_52/dropout/GreaterEqualGreaterEqual8dropout_52/dropout/random_uniform/RandomUniform:output:0*dropout_52/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:???????????2!
dropout_52/dropout/GreaterEqual?
dropout_52/dropout/CastCast#dropout_52/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:???????????2
dropout_52/dropout/Cast?
dropout_52/dropout/Mul_1Muldropout_52/dropout/Mul:z:0dropout_52/dropout/Cast:y:0*
T0*1
_output_shapes
:???????????2
dropout_52/dropout/Mul_1?
 conv2d_146/Conv2D/ReadVariableOpReadVariableOp)conv2d_146_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_146/Conv2D/ReadVariableOp?
conv2d_146/Conv2DConv2Ddropout_52/dropout/Mul_1:z:0(conv2d_146/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_146/Conv2D?
!conv2d_146/BiasAdd/ReadVariableOpReadVariableOp*conv2d_146_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_146/BiasAdd/ReadVariableOp?
conv2d_146/BiasAddBiasAddconv2d_146/Conv2D:output:0)conv2d_146/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_146/BiasAdd?
conv2d_146/ReluReluconv2d_146/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_146/Relu?
%batch_normalization_53/ReadVariableOpReadVariableOp.batch_normalization_53_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_53/ReadVariableOp?
'batch_normalization_53/ReadVariableOp_1ReadVariableOp0batch_normalization_53_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_53/ReadVariableOp_1?
6batch_normalization_53/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_53_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_53/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_53_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_53/FusedBatchNormV3FusedBatchNormV3conv2d_146/Relu:activations:0-batch_normalization_53/ReadVariableOp:value:0/batch_normalization_53/ReadVariableOp_1:value:0>batch_normalization_53/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_53/FusedBatchNormV3?
%batch_normalization_53/AssignNewValueAssignVariableOp?batch_normalization_53_fusedbatchnormv3_readvariableop_resource4batch_normalization_53/FusedBatchNormV3:batch_mean:07^batch_normalization_53/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*R
_classH
FDloc:@batch_normalization_53/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_53/AssignNewValue?
'batch_normalization_53/AssignNewValue_1AssignVariableOpAbatch_normalization_53_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_53/FusedBatchNormV3:batch_variance:09^batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_53/AssignNewValue_1y
dropout_53/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_53/dropout/Const?
dropout_53/dropout/MulMul+batch_normalization_53/FusedBatchNormV3:y:0!dropout_53/dropout/Const:output:0*
T0*1
_output_shapes
:???????????2
dropout_53/dropout/Mul?
dropout_53/dropout/ShapeShape+batch_normalization_53/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
dropout_53/dropout/Shape?
/dropout_53/dropout/random_uniform/RandomUniformRandomUniform!dropout_53/dropout/Shape:output:0*
T0*1
_output_shapes
:???????????*
dtype021
/dropout_53/dropout/random_uniform/RandomUniform?
!dropout_53/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2#
!dropout_53/dropout/GreaterEqual/y?
dropout_53/dropout/GreaterEqualGreaterEqual8dropout_53/dropout/random_uniform/RandomUniform:output:0*dropout_53/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:???????????2!
dropout_53/dropout/GreaterEqual?
dropout_53/dropout/CastCast#dropout_53/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:???????????2
dropout_53/dropout/Cast?
dropout_53/dropout/Mul_1Muldropout_53/dropout/Mul:z:0dropout_53/dropout/Cast:y:0*
T0*1
_output_shapes
:???????????2
dropout_53/dropout/Mul_1?
max_pooling2d_58/MaxPoolMaxPooldropout_53/dropout/Mul_1:z:0*/
_output_shapes
:?????????***
ksize
*
paddingVALID*
strides
2
max_pooling2d_58/MaxPool?
 conv2d_147/Conv2D/ReadVariableOpReadVariableOp)conv2d_147_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_147/Conv2D/ReadVariableOp?
conv2d_147/Conv2DConv2D!max_pooling2d_58/MaxPool:output:0(conv2d_147/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????***
paddingSAME*
strides
2
conv2d_147/Conv2D?
!conv2d_147/BiasAdd/ReadVariableOpReadVariableOp*conv2d_147_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_147/BiasAdd/ReadVariableOp?
conv2d_147/BiasAddBiasAddconv2d_147/Conv2D:output:0)conv2d_147/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**2
conv2d_147/BiasAdd?
conv2d_147/ReluReluconv2d_147/BiasAdd:output:0*
T0*/
_output_shapes
:?????????**2
conv2d_147/Relu?
%batch_normalization_54/ReadVariableOpReadVariableOp.batch_normalization_54_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_54/ReadVariableOp?
'batch_normalization_54/ReadVariableOp_1ReadVariableOp0batch_normalization_54_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_54/ReadVariableOp_1?
6batch_normalization_54/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_54_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_54/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_54_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_54/FusedBatchNormV3FusedBatchNormV3conv2d_147/Relu:activations:0-batch_normalization_54/ReadVariableOp:value:0/batch_normalization_54/ReadVariableOp_1:value:0>batch_normalization_54/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????**:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_54/FusedBatchNormV3?
%batch_normalization_54/AssignNewValueAssignVariableOp?batch_normalization_54_fusedbatchnormv3_readvariableop_resource4batch_normalization_54/FusedBatchNormV3:batch_mean:07^batch_normalization_54/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*R
_classH
FDloc:@batch_normalization_54/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_54/AssignNewValue?
'batch_normalization_54/AssignNewValue_1AssignVariableOpAbatch_normalization_54_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_54/FusedBatchNormV3:batch_variance:09^batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_54/AssignNewValue_1y
dropout_54/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_54/dropout/Const?
dropout_54/dropout/MulMul+batch_normalization_54/FusedBatchNormV3:y:0!dropout_54/dropout/Const:output:0*
T0*/
_output_shapes
:?????????**2
dropout_54/dropout/Mul?
dropout_54/dropout/ShapeShape+batch_normalization_54/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
dropout_54/dropout/Shape?
/dropout_54/dropout/random_uniform/RandomUniformRandomUniform!dropout_54/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????***
dtype021
/dropout_54/dropout/random_uniform/RandomUniform?
!dropout_54/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2#
!dropout_54/dropout/GreaterEqual/y?
dropout_54/dropout/GreaterEqualGreaterEqual8dropout_54/dropout/random_uniform/RandomUniform:output:0*dropout_54/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????**2!
dropout_54/dropout/GreaterEqual?
dropout_54/dropout/CastCast#dropout_54/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????**2
dropout_54/dropout/Cast?
dropout_54/dropout/Mul_1Muldropout_54/dropout/Mul:z:0dropout_54/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????**2
dropout_54/dropout/Mul_1?
 conv2d_148/Conv2D/ReadVariableOpReadVariableOp)conv2d_148_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_148/Conv2D/ReadVariableOp?
conv2d_148/Conv2DConv2Ddropout_54/dropout/Mul_1:z:0(conv2d_148/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????***
paddingSAME*
strides
2
conv2d_148/Conv2D?
!conv2d_148/BiasAdd/ReadVariableOpReadVariableOp*conv2d_148_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_148/BiasAdd/ReadVariableOp?
conv2d_148/BiasAddBiasAddconv2d_148/Conv2D:output:0)conv2d_148/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**2
conv2d_148/BiasAdd?
conv2d_148/ReluReluconv2d_148/BiasAdd:output:0*
T0*/
_output_shapes
:?????????**2
conv2d_148/Relu?
%batch_normalization_55/ReadVariableOpReadVariableOp.batch_normalization_55_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_55/ReadVariableOp?
'batch_normalization_55/ReadVariableOp_1ReadVariableOp0batch_normalization_55_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_55/ReadVariableOp_1?
6batch_normalization_55/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_55_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_55/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_55_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_55/FusedBatchNormV3FusedBatchNormV3conv2d_148/Relu:activations:0-batch_normalization_55/ReadVariableOp:value:0/batch_normalization_55/ReadVariableOp_1:value:0>batch_normalization_55/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????**:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_55/FusedBatchNormV3?
%batch_normalization_55/AssignNewValueAssignVariableOp?batch_normalization_55_fusedbatchnormv3_readvariableop_resource4batch_normalization_55/FusedBatchNormV3:batch_mean:07^batch_normalization_55/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*R
_classH
FDloc:@batch_normalization_55/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_55/AssignNewValue?
'batch_normalization_55/AssignNewValue_1AssignVariableOpAbatch_normalization_55_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_55/FusedBatchNormV3:batch_variance:09^batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_55/AssignNewValue_1y
dropout_55/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_55/dropout/Const?
dropout_55/dropout/MulMul+batch_normalization_55/FusedBatchNormV3:y:0!dropout_55/dropout/Const:output:0*
T0*/
_output_shapes
:?????????**2
dropout_55/dropout/Mul?
dropout_55/dropout/ShapeShape+batch_normalization_55/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
dropout_55/dropout/Shape?
/dropout_55/dropout/random_uniform/RandomUniformRandomUniform!dropout_55/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????***
dtype021
/dropout_55/dropout/random_uniform/RandomUniform?
!dropout_55/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2#
!dropout_55/dropout/GreaterEqual/y?
dropout_55/dropout/GreaterEqualGreaterEqual8dropout_55/dropout/random_uniform/RandomUniform:output:0*dropout_55/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????**2!
dropout_55/dropout/GreaterEqual?
dropout_55/dropout/CastCast#dropout_55/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????**2
dropout_55/dropout/Cast?
dropout_55/dropout/Mul_1Muldropout_55/dropout/Mul:z:0dropout_55/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????**2
dropout_55/dropout/Mul_1?
max_pooling2d_59/MaxPoolMaxPooldropout_55/dropout/Mul_1:z:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_59/MaxPool?
 conv2d_149/Conv2D/ReadVariableOpReadVariableOp)conv2d_149_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_149/Conv2D/ReadVariableOp?
conv2d_149/Conv2DConv2D!max_pooling2d_59/MaxPool:output:0(conv2d_149/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_149/Conv2D?
!conv2d_149/BiasAdd/ReadVariableOpReadVariableOp*conv2d_149_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_149/BiasAdd/ReadVariableOp?
conv2d_149/BiasAddBiasAddconv2d_149/Conv2D:output:0)conv2d_149/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_149/BiasAdd?
conv2d_149/ReluReluconv2d_149/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_149/Relu?
%batch_normalization_56/ReadVariableOpReadVariableOp.batch_normalization_56_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_56/ReadVariableOp?
'batch_normalization_56/ReadVariableOp_1ReadVariableOp0batch_normalization_56_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_56/ReadVariableOp_1?
6batch_normalization_56/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_56_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_56/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_56/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_56_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_56/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_56/FusedBatchNormV3FusedBatchNormV3conv2d_149/Relu:activations:0-batch_normalization_56/ReadVariableOp:value:0/batch_normalization_56/ReadVariableOp_1:value:0>batch_normalization_56/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_56/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_56/FusedBatchNormV3?
%batch_normalization_56/AssignNewValueAssignVariableOp?batch_normalization_56_fusedbatchnormv3_readvariableop_resource4batch_normalization_56/FusedBatchNormV3:batch_mean:07^batch_normalization_56/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*R
_classH
FDloc:@batch_normalization_56/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_56/AssignNewValue?
'batch_normalization_56/AssignNewValue_1AssignVariableOpAbatch_normalization_56_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_56/FusedBatchNormV3:batch_variance:09^batch_normalization_56/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_56/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_56/AssignNewValue_1y
dropout_56/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_56/dropout/Const?
dropout_56/dropout/MulMul+batch_normalization_56/FusedBatchNormV3:y:0!dropout_56/dropout/Const:output:0*
T0*/
_output_shapes
:?????????2
dropout_56/dropout/Mul?
dropout_56/dropout/ShapeShape+batch_normalization_56/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
dropout_56/dropout/Shape?
/dropout_56/dropout/random_uniform/RandomUniformRandomUniform!dropout_56/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype021
/dropout_56/dropout/random_uniform/RandomUniform?
!dropout_56/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2#
!dropout_56/dropout/GreaterEqual/y?
dropout_56/dropout/GreaterEqualGreaterEqual8dropout_56/dropout/random_uniform/RandomUniform:output:0*dropout_56/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????2!
dropout_56/dropout/GreaterEqual?
dropout_56/dropout/CastCast#dropout_56/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout_56/dropout/Cast?
dropout_56/dropout/Mul_1Muldropout_56/dropout/Mul:z:0dropout_56/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout_56/dropout/Mul_1?
 conv2d_150/Conv2D/ReadVariableOpReadVariableOp)conv2d_150_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_150/Conv2D/ReadVariableOp?
conv2d_150/Conv2DConv2Ddropout_56/dropout/Mul_1:z:0(conv2d_150/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_150/Conv2D?
!conv2d_150/BiasAdd/ReadVariableOpReadVariableOp*conv2d_150_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_150/BiasAdd/ReadVariableOp?
conv2d_150/BiasAddBiasAddconv2d_150/Conv2D:output:0)conv2d_150/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_150/BiasAdd?
conv2d_150/ReluReluconv2d_150/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_150/Relu?
%batch_normalization_57/ReadVariableOpReadVariableOp.batch_normalization_57_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_57/ReadVariableOp?
'batch_normalization_57/ReadVariableOp_1ReadVariableOp0batch_normalization_57_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_57/ReadVariableOp_1?
6batch_normalization_57/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_57_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_57/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_57_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_57/FusedBatchNormV3FusedBatchNormV3conv2d_150/Relu:activations:0-batch_normalization_57/ReadVariableOp:value:0/batch_normalization_57/ReadVariableOp_1:value:0>batch_normalization_57/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_57/FusedBatchNormV3?
%batch_normalization_57/AssignNewValueAssignVariableOp?batch_normalization_57_fusedbatchnormv3_readvariableop_resource4batch_normalization_57/FusedBatchNormV3:batch_mean:07^batch_normalization_57/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*R
_classH
FDloc:@batch_normalization_57/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_57/AssignNewValue?
'batch_normalization_57/AssignNewValue_1AssignVariableOpAbatch_normalization_57_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_57/FusedBatchNormV3:batch_variance:09^batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_57/AssignNewValue_1y
dropout_57/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_57/dropout/Const?
dropout_57/dropout/MulMul+batch_normalization_57/FusedBatchNormV3:y:0!dropout_57/dropout/Const:output:0*
T0*/
_output_shapes
:?????????2
dropout_57/dropout/Mul?
dropout_57/dropout/ShapeShape+batch_normalization_57/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
dropout_57/dropout/Shape?
/dropout_57/dropout/random_uniform/RandomUniformRandomUniform!dropout_57/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype021
/dropout_57/dropout/random_uniform/RandomUniform?
!dropout_57/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2#
!dropout_57/dropout/GreaterEqual/y?
dropout_57/dropout/GreaterEqualGreaterEqual8dropout_57/dropout/random_uniform/RandomUniform:output:0*dropout_57/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????2!
dropout_57/dropout/GreaterEqual?
dropout_57/dropout/CastCast#dropout_57/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout_57/dropout/Cast?
dropout_57/dropout/Mul_1Muldropout_57/dropout/Mul:z:0dropout_57/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout_57/dropout/Mul_1?
max_pooling2d_60/MaxPoolMaxPooldropout_57/dropout/Mul_1:z:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_60/MaxPoolu
flatten_30/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   2
flatten_30/Const?
flatten_30/ReshapeReshape!max_pooling2d_60/MaxPool:output:0flatten_30/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_30/Reshape?
dense_60/MatMul/ReadVariableOpReadVariableOp'dense_60_matmul_readvariableop_resource*
_output_shapes
:	? *
dtype02 
dense_60/MatMul/ReadVariableOp?
dense_60/MatMulMatMulflatten_30/Reshape:output:0&dense_60/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_60/MatMul?
dense_60/BiasAdd/ReadVariableOpReadVariableOp(dense_60_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_60/BiasAdd/ReadVariableOp?
dense_60/BiasAddBiasAdddense_60/MatMul:product:0'dense_60/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_60/BiasAdds
dense_60/ReluReludense_60/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_60/Relu?
dense_61/MatMul/ReadVariableOpReadVariableOp'dense_61_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_61/MatMul/ReadVariableOp?
dense_61/MatMulMatMuldense_60/Relu:activations:0&dense_61/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_61/MatMul?
dense_61/BiasAdd/ReadVariableOpReadVariableOp(dense_61_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_61/BiasAdd/ReadVariableOp?
dense_61/BiasAddBiasAdddense_61/MatMul:product:0'dense_61/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_61/BiasAdd|
dense_61/SoftmaxSoftmaxdense_61/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_61/Softmax?
IdentityIdentitydense_61/Softmax:softmax:0&^batch_normalization_52/AssignNewValue(^batch_normalization_52/AssignNewValue_17^batch_normalization_52/FusedBatchNormV3/ReadVariableOp9^batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_52/ReadVariableOp(^batch_normalization_52/ReadVariableOp_1&^batch_normalization_53/AssignNewValue(^batch_normalization_53/AssignNewValue_17^batch_normalization_53/FusedBatchNormV3/ReadVariableOp9^batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_53/ReadVariableOp(^batch_normalization_53/ReadVariableOp_1&^batch_normalization_54/AssignNewValue(^batch_normalization_54/AssignNewValue_17^batch_normalization_54/FusedBatchNormV3/ReadVariableOp9^batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_54/ReadVariableOp(^batch_normalization_54/ReadVariableOp_1&^batch_normalization_55/AssignNewValue(^batch_normalization_55/AssignNewValue_17^batch_normalization_55/FusedBatchNormV3/ReadVariableOp9^batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_55/ReadVariableOp(^batch_normalization_55/ReadVariableOp_1&^batch_normalization_56/AssignNewValue(^batch_normalization_56/AssignNewValue_17^batch_normalization_56/FusedBatchNormV3/ReadVariableOp9^batch_normalization_56/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_56/ReadVariableOp(^batch_normalization_56/ReadVariableOp_1&^batch_normalization_57/AssignNewValue(^batch_normalization_57/AssignNewValue_17^batch_normalization_57/FusedBatchNormV3/ReadVariableOp9^batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_57/ReadVariableOp(^batch_normalization_57/ReadVariableOp_1"^conv2d_145/BiasAdd/ReadVariableOp!^conv2d_145/Conv2D/ReadVariableOp"^conv2d_146/BiasAdd/ReadVariableOp!^conv2d_146/Conv2D/ReadVariableOp"^conv2d_147/BiasAdd/ReadVariableOp!^conv2d_147/Conv2D/ReadVariableOp"^conv2d_148/BiasAdd/ReadVariableOp!^conv2d_148/Conv2D/ReadVariableOp"^conv2d_149/BiasAdd/ReadVariableOp!^conv2d_149/Conv2D/ReadVariableOp"^conv2d_150/BiasAdd/ReadVariableOp!^conv2d_150/Conv2D/ReadVariableOp ^dense_60/BiasAdd/ReadVariableOp^dense_60/MatMul/ReadVariableOp ^dense_61/BiasAdd/ReadVariableOp^dense_61/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::2N
%batch_normalization_52/AssignNewValue%batch_normalization_52/AssignNewValue2R
'batch_normalization_52/AssignNewValue_1'batch_normalization_52/AssignNewValue_12p
6batch_normalization_52/FusedBatchNormV3/ReadVariableOp6batch_normalization_52/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_52/FusedBatchNormV3/ReadVariableOp_18batch_normalization_52/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_52/ReadVariableOp%batch_normalization_52/ReadVariableOp2R
'batch_normalization_52/ReadVariableOp_1'batch_normalization_52/ReadVariableOp_12N
%batch_normalization_53/AssignNewValue%batch_normalization_53/AssignNewValue2R
'batch_normalization_53/AssignNewValue_1'batch_normalization_53/AssignNewValue_12p
6batch_normalization_53/FusedBatchNormV3/ReadVariableOp6batch_normalization_53/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_53/FusedBatchNormV3/ReadVariableOp_18batch_normalization_53/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_53/ReadVariableOp%batch_normalization_53/ReadVariableOp2R
'batch_normalization_53/ReadVariableOp_1'batch_normalization_53/ReadVariableOp_12N
%batch_normalization_54/AssignNewValue%batch_normalization_54/AssignNewValue2R
'batch_normalization_54/AssignNewValue_1'batch_normalization_54/AssignNewValue_12p
6batch_normalization_54/FusedBatchNormV3/ReadVariableOp6batch_normalization_54/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_54/FusedBatchNormV3/ReadVariableOp_18batch_normalization_54/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_54/ReadVariableOp%batch_normalization_54/ReadVariableOp2R
'batch_normalization_54/ReadVariableOp_1'batch_normalization_54/ReadVariableOp_12N
%batch_normalization_55/AssignNewValue%batch_normalization_55/AssignNewValue2R
'batch_normalization_55/AssignNewValue_1'batch_normalization_55/AssignNewValue_12p
6batch_normalization_55/FusedBatchNormV3/ReadVariableOp6batch_normalization_55/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_55/FusedBatchNormV3/ReadVariableOp_18batch_normalization_55/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_55/ReadVariableOp%batch_normalization_55/ReadVariableOp2R
'batch_normalization_55/ReadVariableOp_1'batch_normalization_55/ReadVariableOp_12N
%batch_normalization_56/AssignNewValue%batch_normalization_56/AssignNewValue2R
'batch_normalization_56/AssignNewValue_1'batch_normalization_56/AssignNewValue_12p
6batch_normalization_56/FusedBatchNormV3/ReadVariableOp6batch_normalization_56/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_56/FusedBatchNormV3/ReadVariableOp_18batch_normalization_56/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_56/ReadVariableOp%batch_normalization_56/ReadVariableOp2R
'batch_normalization_56/ReadVariableOp_1'batch_normalization_56/ReadVariableOp_12N
%batch_normalization_57/AssignNewValue%batch_normalization_57/AssignNewValue2R
'batch_normalization_57/AssignNewValue_1'batch_normalization_57/AssignNewValue_12p
6batch_normalization_57/FusedBatchNormV3/ReadVariableOp6batch_normalization_57/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_57/FusedBatchNormV3/ReadVariableOp_18batch_normalization_57/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_57/ReadVariableOp%batch_normalization_57/ReadVariableOp2R
'batch_normalization_57/ReadVariableOp_1'batch_normalization_57/ReadVariableOp_12F
!conv2d_145/BiasAdd/ReadVariableOp!conv2d_145/BiasAdd/ReadVariableOp2D
 conv2d_145/Conv2D/ReadVariableOp conv2d_145/Conv2D/ReadVariableOp2F
!conv2d_146/BiasAdd/ReadVariableOp!conv2d_146/BiasAdd/ReadVariableOp2D
 conv2d_146/Conv2D/ReadVariableOp conv2d_146/Conv2D/ReadVariableOp2F
!conv2d_147/BiasAdd/ReadVariableOp!conv2d_147/BiasAdd/ReadVariableOp2D
 conv2d_147/Conv2D/ReadVariableOp conv2d_147/Conv2D/ReadVariableOp2F
!conv2d_148/BiasAdd/ReadVariableOp!conv2d_148/BiasAdd/ReadVariableOp2D
 conv2d_148/Conv2D/ReadVariableOp conv2d_148/Conv2D/ReadVariableOp2F
!conv2d_149/BiasAdd/ReadVariableOp!conv2d_149/BiasAdd/ReadVariableOp2D
 conv2d_149/Conv2D/ReadVariableOp conv2d_149/Conv2D/ReadVariableOp2F
!conv2d_150/BiasAdd/ReadVariableOp!conv2d_150/BiasAdd/ReadVariableOp2D
 conv2d_150/Conv2D/ReadVariableOp conv2d_150/Conv2D/ReadVariableOp2B
dense_60/BiasAdd/ReadVariableOpdense_60/BiasAdd/ReadVariableOp2@
dense_60/MatMul/ReadVariableOpdense_60/MatMul/ReadVariableOp2B
dense_61/BiasAdd/ReadVariableOpdense_61/BiasAdd/ReadVariableOp2@
dense_61/MatMul/ReadVariableOpdense_61/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_55_layer_call_fn_2707887

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
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_27050372
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
e
,__inference_dropout_56_layer_call_fn_2708161

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
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
GPU 2J 8? *P
fKRI
G__inference_dropout_56_layer_call_and_return_conditional_losses_27059492
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_54_layer_call_and_return_conditional_losses_2707699

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
f
G__inference_dropout_54_layer_call_and_return_conditional_losses_2705688

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????**2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????***
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????**2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????**2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????**2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????**2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????**:W S
/
_output_shapes
:?????????**
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_57_layer_call_and_return_conditional_losses_2705288

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
G__inference_conv2d_149_layer_call_and_return_conditional_losses_2708002

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
S__inference_batch_normalization_54_layer_call_and_return_conditional_losses_2704933

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
ث
?.
"__inference__wrapped_model_2704651
input_31E
Acnn_aug_deep_bnorm_drop_conv2d_145_conv2d_readvariableop_resourceF
Bcnn_aug_deep_bnorm_drop_conv2d_145_biasadd_readvariableop_resourceJ
Fcnn_aug_deep_bnorm_drop_batch_normalization_52_readvariableop_resourceL
Hcnn_aug_deep_bnorm_drop_batch_normalization_52_readvariableop_1_resource[
Wcnn_aug_deep_bnorm_drop_batch_normalization_52_fusedbatchnormv3_readvariableop_resource]
Ycnn_aug_deep_bnorm_drop_batch_normalization_52_fusedbatchnormv3_readvariableop_1_resourceE
Acnn_aug_deep_bnorm_drop_conv2d_146_conv2d_readvariableop_resourceF
Bcnn_aug_deep_bnorm_drop_conv2d_146_biasadd_readvariableop_resourceJ
Fcnn_aug_deep_bnorm_drop_batch_normalization_53_readvariableop_resourceL
Hcnn_aug_deep_bnorm_drop_batch_normalization_53_readvariableop_1_resource[
Wcnn_aug_deep_bnorm_drop_batch_normalization_53_fusedbatchnormv3_readvariableop_resource]
Ycnn_aug_deep_bnorm_drop_batch_normalization_53_fusedbatchnormv3_readvariableop_1_resourceE
Acnn_aug_deep_bnorm_drop_conv2d_147_conv2d_readvariableop_resourceF
Bcnn_aug_deep_bnorm_drop_conv2d_147_biasadd_readvariableop_resourceJ
Fcnn_aug_deep_bnorm_drop_batch_normalization_54_readvariableop_resourceL
Hcnn_aug_deep_bnorm_drop_batch_normalization_54_readvariableop_1_resource[
Wcnn_aug_deep_bnorm_drop_batch_normalization_54_fusedbatchnormv3_readvariableop_resource]
Ycnn_aug_deep_bnorm_drop_batch_normalization_54_fusedbatchnormv3_readvariableop_1_resourceE
Acnn_aug_deep_bnorm_drop_conv2d_148_conv2d_readvariableop_resourceF
Bcnn_aug_deep_bnorm_drop_conv2d_148_biasadd_readvariableop_resourceJ
Fcnn_aug_deep_bnorm_drop_batch_normalization_55_readvariableop_resourceL
Hcnn_aug_deep_bnorm_drop_batch_normalization_55_readvariableop_1_resource[
Wcnn_aug_deep_bnorm_drop_batch_normalization_55_fusedbatchnormv3_readvariableop_resource]
Ycnn_aug_deep_bnorm_drop_batch_normalization_55_fusedbatchnormv3_readvariableop_1_resourceE
Acnn_aug_deep_bnorm_drop_conv2d_149_conv2d_readvariableop_resourceF
Bcnn_aug_deep_bnorm_drop_conv2d_149_biasadd_readvariableop_resourceJ
Fcnn_aug_deep_bnorm_drop_batch_normalization_56_readvariableop_resourceL
Hcnn_aug_deep_bnorm_drop_batch_normalization_56_readvariableop_1_resource[
Wcnn_aug_deep_bnorm_drop_batch_normalization_56_fusedbatchnormv3_readvariableop_resource]
Ycnn_aug_deep_bnorm_drop_batch_normalization_56_fusedbatchnormv3_readvariableop_1_resourceE
Acnn_aug_deep_bnorm_drop_conv2d_150_conv2d_readvariableop_resourceF
Bcnn_aug_deep_bnorm_drop_conv2d_150_biasadd_readvariableop_resourceJ
Fcnn_aug_deep_bnorm_drop_batch_normalization_57_readvariableop_resourceL
Hcnn_aug_deep_bnorm_drop_batch_normalization_57_readvariableop_1_resource[
Wcnn_aug_deep_bnorm_drop_batch_normalization_57_fusedbatchnormv3_readvariableop_resource]
Ycnn_aug_deep_bnorm_drop_batch_normalization_57_fusedbatchnormv3_readvariableop_1_resourceC
?cnn_aug_deep_bnorm_drop_dense_60_matmul_readvariableop_resourceD
@cnn_aug_deep_bnorm_drop_dense_60_biasadd_readvariableop_resourceC
?cnn_aug_deep_bnorm_drop_dense_61_matmul_readvariableop_resourceD
@cnn_aug_deep_bnorm_drop_dense_61_biasadd_readvariableop_resource
identity??NCNN_aug_deep_bnorm_drop/batch_normalization_52/FusedBatchNormV3/ReadVariableOp?PCNN_aug_deep_bnorm_drop/batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1?=CNN_aug_deep_bnorm_drop/batch_normalization_52/ReadVariableOp??CNN_aug_deep_bnorm_drop/batch_normalization_52/ReadVariableOp_1?NCNN_aug_deep_bnorm_drop/batch_normalization_53/FusedBatchNormV3/ReadVariableOp?PCNN_aug_deep_bnorm_drop/batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1?=CNN_aug_deep_bnorm_drop/batch_normalization_53/ReadVariableOp??CNN_aug_deep_bnorm_drop/batch_normalization_53/ReadVariableOp_1?NCNN_aug_deep_bnorm_drop/batch_normalization_54/FusedBatchNormV3/ReadVariableOp?PCNN_aug_deep_bnorm_drop/batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1?=CNN_aug_deep_bnorm_drop/batch_normalization_54/ReadVariableOp??CNN_aug_deep_bnorm_drop/batch_normalization_54/ReadVariableOp_1?NCNN_aug_deep_bnorm_drop/batch_normalization_55/FusedBatchNormV3/ReadVariableOp?PCNN_aug_deep_bnorm_drop/batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1?=CNN_aug_deep_bnorm_drop/batch_normalization_55/ReadVariableOp??CNN_aug_deep_bnorm_drop/batch_normalization_55/ReadVariableOp_1?NCNN_aug_deep_bnorm_drop/batch_normalization_56/FusedBatchNormV3/ReadVariableOp?PCNN_aug_deep_bnorm_drop/batch_normalization_56/FusedBatchNormV3/ReadVariableOp_1?=CNN_aug_deep_bnorm_drop/batch_normalization_56/ReadVariableOp??CNN_aug_deep_bnorm_drop/batch_normalization_56/ReadVariableOp_1?NCNN_aug_deep_bnorm_drop/batch_normalization_57/FusedBatchNormV3/ReadVariableOp?PCNN_aug_deep_bnorm_drop/batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1?=CNN_aug_deep_bnorm_drop/batch_normalization_57/ReadVariableOp??CNN_aug_deep_bnorm_drop/batch_normalization_57/ReadVariableOp_1?9CNN_aug_deep_bnorm_drop/conv2d_145/BiasAdd/ReadVariableOp?8CNN_aug_deep_bnorm_drop/conv2d_145/Conv2D/ReadVariableOp?9CNN_aug_deep_bnorm_drop/conv2d_146/BiasAdd/ReadVariableOp?8CNN_aug_deep_bnorm_drop/conv2d_146/Conv2D/ReadVariableOp?9CNN_aug_deep_bnorm_drop/conv2d_147/BiasAdd/ReadVariableOp?8CNN_aug_deep_bnorm_drop/conv2d_147/Conv2D/ReadVariableOp?9CNN_aug_deep_bnorm_drop/conv2d_148/BiasAdd/ReadVariableOp?8CNN_aug_deep_bnorm_drop/conv2d_148/Conv2D/ReadVariableOp?9CNN_aug_deep_bnorm_drop/conv2d_149/BiasAdd/ReadVariableOp?8CNN_aug_deep_bnorm_drop/conv2d_149/Conv2D/ReadVariableOp?9CNN_aug_deep_bnorm_drop/conv2d_150/BiasAdd/ReadVariableOp?8CNN_aug_deep_bnorm_drop/conv2d_150/Conv2D/ReadVariableOp?7CNN_aug_deep_bnorm_drop/dense_60/BiasAdd/ReadVariableOp?6CNN_aug_deep_bnorm_drop/dense_60/MatMul/ReadVariableOp?7CNN_aug_deep_bnorm_drop/dense_61/BiasAdd/ReadVariableOp?6CNN_aug_deep_bnorm_drop/dense_61/MatMul/ReadVariableOp?
8CNN_aug_deep_bnorm_drop/conv2d_145/Conv2D/ReadVariableOpReadVariableOpAcnn_aug_deep_bnorm_drop_conv2d_145_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02:
8CNN_aug_deep_bnorm_drop/conv2d_145/Conv2D/ReadVariableOp?
)CNN_aug_deep_bnorm_drop/conv2d_145/Conv2DConv2Dinput_31@CNN_aug_deep_bnorm_drop/conv2d_145/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2+
)CNN_aug_deep_bnorm_drop/conv2d_145/Conv2D?
9CNN_aug_deep_bnorm_drop/conv2d_145/BiasAdd/ReadVariableOpReadVariableOpBcnn_aug_deep_bnorm_drop_conv2d_145_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02;
9CNN_aug_deep_bnorm_drop/conv2d_145/BiasAdd/ReadVariableOp?
*CNN_aug_deep_bnorm_drop/conv2d_145/BiasAddBiasAdd2CNN_aug_deep_bnorm_drop/conv2d_145/Conv2D:output:0ACNN_aug_deep_bnorm_drop/conv2d_145/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2,
*CNN_aug_deep_bnorm_drop/conv2d_145/BiasAdd?
'CNN_aug_deep_bnorm_drop/conv2d_145/ReluRelu3CNN_aug_deep_bnorm_drop/conv2d_145/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2)
'CNN_aug_deep_bnorm_drop/conv2d_145/Relu?
=CNN_aug_deep_bnorm_drop/batch_normalization_52/ReadVariableOpReadVariableOpFcnn_aug_deep_bnorm_drop_batch_normalization_52_readvariableop_resource*
_output_shapes
:*
dtype02?
=CNN_aug_deep_bnorm_drop/batch_normalization_52/ReadVariableOp?
?CNN_aug_deep_bnorm_drop/batch_normalization_52/ReadVariableOp_1ReadVariableOpHcnn_aug_deep_bnorm_drop_batch_normalization_52_readvariableop_1_resource*
_output_shapes
:*
dtype02A
?CNN_aug_deep_bnorm_drop/batch_normalization_52/ReadVariableOp_1?
NCNN_aug_deep_bnorm_drop/batch_normalization_52/FusedBatchNormV3/ReadVariableOpReadVariableOpWcnn_aug_deep_bnorm_drop_batch_normalization_52_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02P
NCNN_aug_deep_bnorm_drop/batch_normalization_52/FusedBatchNormV3/ReadVariableOp?
PCNN_aug_deep_bnorm_drop/batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpYcnn_aug_deep_bnorm_drop_batch_normalization_52_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02R
PCNN_aug_deep_bnorm_drop/batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1?
?CNN_aug_deep_bnorm_drop/batch_normalization_52/FusedBatchNormV3FusedBatchNormV35CNN_aug_deep_bnorm_drop/conv2d_145/Relu:activations:0ECNN_aug_deep_bnorm_drop/batch_normalization_52/ReadVariableOp:value:0GCNN_aug_deep_bnorm_drop/batch_normalization_52/ReadVariableOp_1:value:0VCNN_aug_deep_bnorm_drop/batch_normalization_52/FusedBatchNormV3/ReadVariableOp:value:0XCNN_aug_deep_bnorm_drop/batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2A
?CNN_aug_deep_bnorm_drop/batch_normalization_52/FusedBatchNormV3?
+CNN_aug_deep_bnorm_drop/dropout_52/IdentityIdentityCCNN_aug_deep_bnorm_drop/batch_normalization_52/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2-
+CNN_aug_deep_bnorm_drop/dropout_52/Identity?
8CNN_aug_deep_bnorm_drop/conv2d_146/Conv2D/ReadVariableOpReadVariableOpAcnn_aug_deep_bnorm_drop_conv2d_146_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02:
8CNN_aug_deep_bnorm_drop/conv2d_146/Conv2D/ReadVariableOp?
)CNN_aug_deep_bnorm_drop/conv2d_146/Conv2DConv2D4CNN_aug_deep_bnorm_drop/dropout_52/Identity:output:0@CNN_aug_deep_bnorm_drop/conv2d_146/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2+
)CNN_aug_deep_bnorm_drop/conv2d_146/Conv2D?
9CNN_aug_deep_bnorm_drop/conv2d_146/BiasAdd/ReadVariableOpReadVariableOpBcnn_aug_deep_bnorm_drop_conv2d_146_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02;
9CNN_aug_deep_bnorm_drop/conv2d_146/BiasAdd/ReadVariableOp?
*CNN_aug_deep_bnorm_drop/conv2d_146/BiasAddBiasAdd2CNN_aug_deep_bnorm_drop/conv2d_146/Conv2D:output:0ACNN_aug_deep_bnorm_drop/conv2d_146/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2,
*CNN_aug_deep_bnorm_drop/conv2d_146/BiasAdd?
'CNN_aug_deep_bnorm_drop/conv2d_146/ReluRelu3CNN_aug_deep_bnorm_drop/conv2d_146/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2)
'CNN_aug_deep_bnorm_drop/conv2d_146/Relu?
=CNN_aug_deep_bnorm_drop/batch_normalization_53/ReadVariableOpReadVariableOpFcnn_aug_deep_bnorm_drop_batch_normalization_53_readvariableop_resource*
_output_shapes
:*
dtype02?
=CNN_aug_deep_bnorm_drop/batch_normalization_53/ReadVariableOp?
?CNN_aug_deep_bnorm_drop/batch_normalization_53/ReadVariableOp_1ReadVariableOpHcnn_aug_deep_bnorm_drop_batch_normalization_53_readvariableop_1_resource*
_output_shapes
:*
dtype02A
?CNN_aug_deep_bnorm_drop/batch_normalization_53/ReadVariableOp_1?
NCNN_aug_deep_bnorm_drop/batch_normalization_53/FusedBatchNormV3/ReadVariableOpReadVariableOpWcnn_aug_deep_bnorm_drop_batch_normalization_53_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02P
NCNN_aug_deep_bnorm_drop/batch_normalization_53/FusedBatchNormV3/ReadVariableOp?
PCNN_aug_deep_bnorm_drop/batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpYcnn_aug_deep_bnorm_drop_batch_normalization_53_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02R
PCNN_aug_deep_bnorm_drop/batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1?
?CNN_aug_deep_bnorm_drop/batch_normalization_53/FusedBatchNormV3FusedBatchNormV35CNN_aug_deep_bnorm_drop/conv2d_146/Relu:activations:0ECNN_aug_deep_bnorm_drop/batch_normalization_53/ReadVariableOp:value:0GCNN_aug_deep_bnorm_drop/batch_normalization_53/ReadVariableOp_1:value:0VCNN_aug_deep_bnorm_drop/batch_normalization_53/FusedBatchNormV3/ReadVariableOp:value:0XCNN_aug_deep_bnorm_drop/batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2A
?CNN_aug_deep_bnorm_drop/batch_normalization_53/FusedBatchNormV3?
+CNN_aug_deep_bnorm_drop/dropout_53/IdentityIdentityCCNN_aug_deep_bnorm_drop/batch_normalization_53/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2-
+CNN_aug_deep_bnorm_drop/dropout_53/Identity?
0CNN_aug_deep_bnorm_drop/max_pooling2d_58/MaxPoolMaxPool4CNN_aug_deep_bnorm_drop/dropout_53/Identity:output:0*/
_output_shapes
:?????????***
ksize
*
paddingVALID*
strides
22
0CNN_aug_deep_bnorm_drop/max_pooling2d_58/MaxPool?
8CNN_aug_deep_bnorm_drop/conv2d_147/Conv2D/ReadVariableOpReadVariableOpAcnn_aug_deep_bnorm_drop_conv2d_147_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02:
8CNN_aug_deep_bnorm_drop/conv2d_147/Conv2D/ReadVariableOp?
)CNN_aug_deep_bnorm_drop/conv2d_147/Conv2DConv2D9CNN_aug_deep_bnorm_drop/max_pooling2d_58/MaxPool:output:0@CNN_aug_deep_bnorm_drop/conv2d_147/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????***
paddingSAME*
strides
2+
)CNN_aug_deep_bnorm_drop/conv2d_147/Conv2D?
9CNN_aug_deep_bnorm_drop/conv2d_147/BiasAdd/ReadVariableOpReadVariableOpBcnn_aug_deep_bnorm_drop_conv2d_147_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02;
9CNN_aug_deep_bnorm_drop/conv2d_147/BiasAdd/ReadVariableOp?
*CNN_aug_deep_bnorm_drop/conv2d_147/BiasAddBiasAdd2CNN_aug_deep_bnorm_drop/conv2d_147/Conv2D:output:0ACNN_aug_deep_bnorm_drop/conv2d_147/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**2,
*CNN_aug_deep_bnorm_drop/conv2d_147/BiasAdd?
'CNN_aug_deep_bnorm_drop/conv2d_147/ReluRelu3CNN_aug_deep_bnorm_drop/conv2d_147/BiasAdd:output:0*
T0*/
_output_shapes
:?????????**2)
'CNN_aug_deep_bnorm_drop/conv2d_147/Relu?
=CNN_aug_deep_bnorm_drop/batch_normalization_54/ReadVariableOpReadVariableOpFcnn_aug_deep_bnorm_drop_batch_normalization_54_readvariableop_resource*
_output_shapes
:*
dtype02?
=CNN_aug_deep_bnorm_drop/batch_normalization_54/ReadVariableOp?
?CNN_aug_deep_bnorm_drop/batch_normalization_54/ReadVariableOp_1ReadVariableOpHcnn_aug_deep_bnorm_drop_batch_normalization_54_readvariableop_1_resource*
_output_shapes
:*
dtype02A
?CNN_aug_deep_bnorm_drop/batch_normalization_54/ReadVariableOp_1?
NCNN_aug_deep_bnorm_drop/batch_normalization_54/FusedBatchNormV3/ReadVariableOpReadVariableOpWcnn_aug_deep_bnorm_drop_batch_normalization_54_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02P
NCNN_aug_deep_bnorm_drop/batch_normalization_54/FusedBatchNormV3/ReadVariableOp?
PCNN_aug_deep_bnorm_drop/batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpYcnn_aug_deep_bnorm_drop_batch_normalization_54_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02R
PCNN_aug_deep_bnorm_drop/batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1?
?CNN_aug_deep_bnorm_drop/batch_normalization_54/FusedBatchNormV3FusedBatchNormV35CNN_aug_deep_bnorm_drop/conv2d_147/Relu:activations:0ECNN_aug_deep_bnorm_drop/batch_normalization_54/ReadVariableOp:value:0GCNN_aug_deep_bnorm_drop/batch_normalization_54/ReadVariableOp_1:value:0VCNN_aug_deep_bnorm_drop/batch_normalization_54/FusedBatchNormV3/ReadVariableOp:value:0XCNN_aug_deep_bnorm_drop/batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????**:::::*
epsilon%o?:*
is_training( 2A
?CNN_aug_deep_bnorm_drop/batch_normalization_54/FusedBatchNormV3?
+CNN_aug_deep_bnorm_drop/dropout_54/IdentityIdentityCCNN_aug_deep_bnorm_drop/batch_normalization_54/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????**2-
+CNN_aug_deep_bnorm_drop/dropout_54/Identity?
8CNN_aug_deep_bnorm_drop/conv2d_148/Conv2D/ReadVariableOpReadVariableOpAcnn_aug_deep_bnorm_drop_conv2d_148_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02:
8CNN_aug_deep_bnorm_drop/conv2d_148/Conv2D/ReadVariableOp?
)CNN_aug_deep_bnorm_drop/conv2d_148/Conv2DConv2D4CNN_aug_deep_bnorm_drop/dropout_54/Identity:output:0@CNN_aug_deep_bnorm_drop/conv2d_148/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????***
paddingSAME*
strides
2+
)CNN_aug_deep_bnorm_drop/conv2d_148/Conv2D?
9CNN_aug_deep_bnorm_drop/conv2d_148/BiasAdd/ReadVariableOpReadVariableOpBcnn_aug_deep_bnorm_drop_conv2d_148_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02;
9CNN_aug_deep_bnorm_drop/conv2d_148/BiasAdd/ReadVariableOp?
*CNN_aug_deep_bnorm_drop/conv2d_148/BiasAddBiasAdd2CNN_aug_deep_bnorm_drop/conv2d_148/Conv2D:output:0ACNN_aug_deep_bnorm_drop/conv2d_148/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**2,
*CNN_aug_deep_bnorm_drop/conv2d_148/BiasAdd?
'CNN_aug_deep_bnorm_drop/conv2d_148/ReluRelu3CNN_aug_deep_bnorm_drop/conv2d_148/BiasAdd:output:0*
T0*/
_output_shapes
:?????????**2)
'CNN_aug_deep_bnorm_drop/conv2d_148/Relu?
=CNN_aug_deep_bnorm_drop/batch_normalization_55/ReadVariableOpReadVariableOpFcnn_aug_deep_bnorm_drop_batch_normalization_55_readvariableop_resource*
_output_shapes
:*
dtype02?
=CNN_aug_deep_bnorm_drop/batch_normalization_55/ReadVariableOp?
?CNN_aug_deep_bnorm_drop/batch_normalization_55/ReadVariableOp_1ReadVariableOpHcnn_aug_deep_bnorm_drop_batch_normalization_55_readvariableop_1_resource*
_output_shapes
:*
dtype02A
?CNN_aug_deep_bnorm_drop/batch_normalization_55/ReadVariableOp_1?
NCNN_aug_deep_bnorm_drop/batch_normalization_55/FusedBatchNormV3/ReadVariableOpReadVariableOpWcnn_aug_deep_bnorm_drop_batch_normalization_55_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02P
NCNN_aug_deep_bnorm_drop/batch_normalization_55/FusedBatchNormV3/ReadVariableOp?
PCNN_aug_deep_bnorm_drop/batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpYcnn_aug_deep_bnorm_drop_batch_normalization_55_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02R
PCNN_aug_deep_bnorm_drop/batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1?
?CNN_aug_deep_bnorm_drop/batch_normalization_55/FusedBatchNormV3FusedBatchNormV35CNN_aug_deep_bnorm_drop/conv2d_148/Relu:activations:0ECNN_aug_deep_bnorm_drop/batch_normalization_55/ReadVariableOp:value:0GCNN_aug_deep_bnorm_drop/batch_normalization_55/ReadVariableOp_1:value:0VCNN_aug_deep_bnorm_drop/batch_normalization_55/FusedBatchNormV3/ReadVariableOp:value:0XCNN_aug_deep_bnorm_drop/batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????**:::::*
epsilon%o?:*
is_training( 2A
?CNN_aug_deep_bnorm_drop/batch_normalization_55/FusedBatchNormV3?
+CNN_aug_deep_bnorm_drop/dropout_55/IdentityIdentityCCNN_aug_deep_bnorm_drop/batch_normalization_55/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????**2-
+CNN_aug_deep_bnorm_drop/dropout_55/Identity?
0CNN_aug_deep_bnorm_drop/max_pooling2d_59/MaxPoolMaxPool4CNN_aug_deep_bnorm_drop/dropout_55/Identity:output:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
22
0CNN_aug_deep_bnorm_drop/max_pooling2d_59/MaxPool?
8CNN_aug_deep_bnorm_drop/conv2d_149/Conv2D/ReadVariableOpReadVariableOpAcnn_aug_deep_bnorm_drop_conv2d_149_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02:
8CNN_aug_deep_bnorm_drop/conv2d_149/Conv2D/ReadVariableOp?
)CNN_aug_deep_bnorm_drop/conv2d_149/Conv2DConv2D9CNN_aug_deep_bnorm_drop/max_pooling2d_59/MaxPool:output:0@CNN_aug_deep_bnorm_drop/conv2d_149/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2+
)CNN_aug_deep_bnorm_drop/conv2d_149/Conv2D?
9CNN_aug_deep_bnorm_drop/conv2d_149/BiasAdd/ReadVariableOpReadVariableOpBcnn_aug_deep_bnorm_drop_conv2d_149_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02;
9CNN_aug_deep_bnorm_drop/conv2d_149/BiasAdd/ReadVariableOp?
*CNN_aug_deep_bnorm_drop/conv2d_149/BiasAddBiasAdd2CNN_aug_deep_bnorm_drop/conv2d_149/Conv2D:output:0ACNN_aug_deep_bnorm_drop/conv2d_149/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2,
*CNN_aug_deep_bnorm_drop/conv2d_149/BiasAdd?
'CNN_aug_deep_bnorm_drop/conv2d_149/ReluRelu3CNN_aug_deep_bnorm_drop/conv2d_149/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2)
'CNN_aug_deep_bnorm_drop/conv2d_149/Relu?
=CNN_aug_deep_bnorm_drop/batch_normalization_56/ReadVariableOpReadVariableOpFcnn_aug_deep_bnorm_drop_batch_normalization_56_readvariableop_resource*
_output_shapes
:*
dtype02?
=CNN_aug_deep_bnorm_drop/batch_normalization_56/ReadVariableOp?
?CNN_aug_deep_bnorm_drop/batch_normalization_56/ReadVariableOp_1ReadVariableOpHcnn_aug_deep_bnorm_drop_batch_normalization_56_readvariableop_1_resource*
_output_shapes
:*
dtype02A
?CNN_aug_deep_bnorm_drop/batch_normalization_56/ReadVariableOp_1?
NCNN_aug_deep_bnorm_drop/batch_normalization_56/FusedBatchNormV3/ReadVariableOpReadVariableOpWcnn_aug_deep_bnorm_drop_batch_normalization_56_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02P
NCNN_aug_deep_bnorm_drop/batch_normalization_56/FusedBatchNormV3/ReadVariableOp?
PCNN_aug_deep_bnorm_drop/batch_normalization_56/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpYcnn_aug_deep_bnorm_drop_batch_normalization_56_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02R
PCNN_aug_deep_bnorm_drop/batch_normalization_56/FusedBatchNormV3/ReadVariableOp_1?
?CNN_aug_deep_bnorm_drop/batch_normalization_56/FusedBatchNormV3FusedBatchNormV35CNN_aug_deep_bnorm_drop/conv2d_149/Relu:activations:0ECNN_aug_deep_bnorm_drop/batch_normalization_56/ReadVariableOp:value:0GCNN_aug_deep_bnorm_drop/batch_normalization_56/ReadVariableOp_1:value:0VCNN_aug_deep_bnorm_drop/batch_normalization_56/FusedBatchNormV3/ReadVariableOp:value:0XCNN_aug_deep_bnorm_drop/batch_normalization_56/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( 2A
?CNN_aug_deep_bnorm_drop/batch_normalization_56/FusedBatchNormV3?
+CNN_aug_deep_bnorm_drop/dropout_56/IdentityIdentityCCNN_aug_deep_bnorm_drop/batch_normalization_56/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????2-
+CNN_aug_deep_bnorm_drop/dropout_56/Identity?
8CNN_aug_deep_bnorm_drop/conv2d_150/Conv2D/ReadVariableOpReadVariableOpAcnn_aug_deep_bnorm_drop_conv2d_150_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02:
8CNN_aug_deep_bnorm_drop/conv2d_150/Conv2D/ReadVariableOp?
)CNN_aug_deep_bnorm_drop/conv2d_150/Conv2DConv2D4CNN_aug_deep_bnorm_drop/dropout_56/Identity:output:0@CNN_aug_deep_bnorm_drop/conv2d_150/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2+
)CNN_aug_deep_bnorm_drop/conv2d_150/Conv2D?
9CNN_aug_deep_bnorm_drop/conv2d_150/BiasAdd/ReadVariableOpReadVariableOpBcnn_aug_deep_bnorm_drop_conv2d_150_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02;
9CNN_aug_deep_bnorm_drop/conv2d_150/BiasAdd/ReadVariableOp?
*CNN_aug_deep_bnorm_drop/conv2d_150/BiasAddBiasAdd2CNN_aug_deep_bnorm_drop/conv2d_150/Conv2D:output:0ACNN_aug_deep_bnorm_drop/conv2d_150/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2,
*CNN_aug_deep_bnorm_drop/conv2d_150/BiasAdd?
'CNN_aug_deep_bnorm_drop/conv2d_150/ReluRelu3CNN_aug_deep_bnorm_drop/conv2d_150/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2)
'CNN_aug_deep_bnorm_drop/conv2d_150/Relu?
=CNN_aug_deep_bnorm_drop/batch_normalization_57/ReadVariableOpReadVariableOpFcnn_aug_deep_bnorm_drop_batch_normalization_57_readvariableop_resource*
_output_shapes
:*
dtype02?
=CNN_aug_deep_bnorm_drop/batch_normalization_57/ReadVariableOp?
?CNN_aug_deep_bnorm_drop/batch_normalization_57/ReadVariableOp_1ReadVariableOpHcnn_aug_deep_bnorm_drop_batch_normalization_57_readvariableop_1_resource*
_output_shapes
:*
dtype02A
?CNN_aug_deep_bnorm_drop/batch_normalization_57/ReadVariableOp_1?
NCNN_aug_deep_bnorm_drop/batch_normalization_57/FusedBatchNormV3/ReadVariableOpReadVariableOpWcnn_aug_deep_bnorm_drop_batch_normalization_57_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02P
NCNN_aug_deep_bnorm_drop/batch_normalization_57/FusedBatchNormV3/ReadVariableOp?
PCNN_aug_deep_bnorm_drop/batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpYcnn_aug_deep_bnorm_drop_batch_normalization_57_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02R
PCNN_aug_deep_bnorm_drop/batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1?
?CNN_aug_deep_bnorm_drop/batch_normalization_57/FusedBatchNormV3FusedBatchNormV35CNN_aug_deep_bnorm_drop/conv2d_150/Relu:activations:0ECNN_aug_deep_bnorm_drop/batch_normalization_57/ReadVariableOp:value:0GCNN_aug_deep_bnorm_drop/batch_normalization_57/ReadVariableOp_1:value:0VCNN_aug_deep_bnorm_drop/batch_normalization_57/FusedBatchNormV3/ReadVariableOp:value:0XCNN_aug_deep_bnorm_drop/batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( 2A
?CNN_aug_deep_bnorm_drop/batch_normalization_57/FusedBatchNormV3?
+CNN_aug_deep_bnorm_drop/dropout_57/IdentityIdentityCCNN_aug_deep_bnorm_drop/batch_normalization_57/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????2-
+CNN_aug_deep_bnorm_drop/dropout_57/Identity?
0CNN_aug_deep_bnorm_drop/max_pooling2d_60/MaxPoolMaxPool4CNN_aug_deep_bnorm_drop/dropout_57/Identity:output:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
22
0CNN_aug_deep_bnorm_drop/max_pooling2d_60/MaxPool?
(CNN_aug_deep_bnorm_drop/flatten_30/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   2*
(CNN_aug_deep_bnorm_drop/flatten_30/Const?
*CNN_aug_deep_bnorm_drop/flatten_30/ReshapeReshape9CNN_aug_deep_bnorm_drop/max_pooling2d_60/MaxPool:output:01CNN_aug_deep_bnorm_drop/flatten_30/Const:output:0*
T0*(
_output_shapes
:??????????2,
*CNN_aug_deep_bnorm_drop/flatten_30/Reshape?
6CNN_aug_deep_bnorm_drop/dense_60/MatMul/ReadVariableOpReadVariableOp?cnn_aug_deep_bnorm_drop_dense_60_matmul_readvariableop_resource*
_output_shapes
:	? *
dtype028
6CNN_aug_deep_bnorm_drop/dense_60/MatMul/ReadVariableOp?
'CNN_aug_deep_bnorm_drop/dense_60/MatMulMatMul3CNN_aug_deep_bnorm_drop/flatten_30/Reshape:output:0>CNN_aug_deep_bnorm_drop/dense_60/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2)
'CNN_aug_deep_bnorm_drop/dense_60/MatMul?
7CNN_aug_deep_bnorm_drop/dense_60/BiasAdd/ReadVariableOpReadVariableOp@cnn_aug_deep_bnorm_drop_dense_60_biasadd_readvariableop_resource*
_output_shapes
: *
dtype029
7CNN_aug_deep_bnorm_drop/dense_60/BiasAdd/ReadVariableOp?
(CNN_aug_deep_bnorm_drop/dense_60/BiasAddBiasAdd1CNN_aug_deep_bnorm_drop/dense_60/MatMul:product:0?CNN_aug_deep_bnorm_drop/dense_60/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2*
(CNN_aug_deep_bnorm_drop/dense_60/BiasAdd?
%CNN_aug_deep_bnorm_drop/dense_60/ReluRelu1CNN_aug_deep_bnorm_drop/dense_60/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2'
%CNN_aug_deep_bnorm_drop/dense_60/Relu?
6CNN_aug_deep_bnorm_drop/dense_61/MatMul/ReadVariableOpReadVariableOp?cnn_aug_deep_bnorm_drop_dense_61_matmul_readvariableop_resource*
_output_shapes

: *
dtype028
6CNN_aug_deep_bnorm_drop/dense_61/MatMul/ReadVariableOp?
'CNN_aug_deep_bnorm_drop/dense_61/MatMulMatMul3CNN_aug_deep_bnorm_drop/dense_60/Relu:activations:0>CNN_aug_deep_bnorm_drop/dense_61/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2)
'CNN_aug_deep_bnorm_drop/dense_61/MatMul?
7CNN_aug_deep_bnorm_drop/dense_61/BiasAdd/ReadVariableOpReadVariableOp@cnn_aug_deep_bnorm_drop_dense_61_biasadd_readvariableop_resource*
_output_shapes
:*
dtype029
7CNN_aug_deep_bnorm_drop/dense_61/BiasAdd/ReadVariableOp?
(CNN_aug_deep_bnorm_drop/dense_61/BiasAddBiasAdd1CNN_aug_deep_bnorm_drop/dense_61/MatMul:product:0?CNN_aug_deep_bnorm_drop/dense_61/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2*
(CNN_aug_deep_bnorm_drop/dense_61/BiasAdd?
(CNN_aug_deep_bnorm_drop/dense_61/SoftmaxSoftmax1CNN_aug_deep_bnorm_drop/dense_61/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2*
(CNN_aug_deep_bnorm_drop/dense_61/Softmax?
IdentityIdentity2CNN_aug_deep_bnorm_drop/dense_61/Softmax:softmax:0O^CNN_aug_deep_bnorm_drop/batch_normalization_52/FusedBatchNormV3/ReadVariableOpQ^CNN_aug_deep_bnorm_drop/batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1>^CNN_aug_deep_bnorm_drop/batch_normalization_52/ReadVariableOp@^CNN_aug_deep_bnorm_drop/batch_normalization_52/ReadVariableOp_1O^CNN_aug_deep_bnorm_drop/batch_normalization_53/FusedBatchNormV3/ReadVariableOpQ^CNN_aug_deep_bnorm_drop/batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1>^CNN_aug_deep_bnorm_drop/batch_normalization_53/ReadVariableOp@^CNN_aug_deep_bnorm_drop/batch_normalization_53/ReadVariableOp_1O^CNN_aug_deep_bnorm_drop/batch_normalization_54/FusedBatchNormV3/ReadVariableOpQ^CNN_aug_deep_bnorm_drop/batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1>^CNN_aug_deep_bnorm_drop/batch_normalization_54/ReadVariableOp@^CNN_aug_deep_bnorm_drop/batch_normalization_54/ReadVariableOp_1O^CNN_aug_deep_bnorm_drop/batch_normalization_55/FusedBatchNormV3/ReadVariableOpQ^CNN_aug_deep_bnorm_drop/batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1>^CNN_aug_deep_bnorm_drop/batch_normalization_55/ReadVariableOp@^CNN_aug_deep_bnorm_drop/batch_normalization_55/ReadVariableOp_1O^CNN_aug_deep_bnorm_drop/batch_normalization_56/FusedBatchNormV3/ReadVariableOpQ^CNN_aug_deep_bnorm_drop/batch_normalization_56/FusedBatchNormV3/ReadVariableOp_1>^CNN_aug_deep_bnorm_drop/batch_normalization_56/ReadVariableOp@^CNN_aug_deep_bnorm_drop/batch_normalization_56/ReadVariableOp_1O^CNN_aug_deep_bnorm_drop/batch_normalization_57/FusedBatchNormV3/ReadVariableOpQ^CNN_aug_deep_bnorm_drop/batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1>^CNN_aug_deep_bnorm_drop/batch_normalization_57/ReadVariableOp@^CNN_aug_deep_bnorm_drop/batch_normalization_57/ReadVariableOp_1:^CNN_aug_deep_bnorm_drop/conv2d_145/BiasAdd/ReadVariableOp9^CNN_aug_deep_bnorm_drop/conv2d_145/Conv2D/ReadVariableOp:^CNN_aug_deep_bnorm_drop/conv2d_146/BiasAdd/ReadVariableOp9^CNN_aug_deep_bnorm_drop/conv2d_146/Conv2D/ReadVariableOp:^CNN_aug_deep_bnorm_drop/conv2d_147/BiasAdd/ReadVariableOp9^CNN_aug_deep_bnorm_drop/conv2d_147/Conv2D/ReadVariableOp:^CNN_aug_deep_bnorm_drop/conv2d_148/BiasAdd/ReadVariableOp9^CNN_aug_deep_bnorm_drop/conv2d_148/Conv2D/ReadVariableOp:^CNN_aug_deep_bnorm_drop/conv2d_149/BiasAdd/ReadVariableOp9^CNN_aug_deep_bnorm_drop/conv2d_149/Conv2D/ReadVariableOp:^CNN_aug_deep_bnorm_drop/conv2d_150/BiasAdd/ReadVariableOp9^CNN_aug_deep_bnorm_drop/conv2d_150/Conv2D/ReadVariableOp8^CNN_aug_deep_bnorm_drop/dense_60/BiasAdd/ReadVariableOp7^CNN_aug_deep_bnorm_drop/dense_60/MatMul/ReadVariableOp8^CNN_aug_deep_bnorm_drop/dense_61/BiasAdd/ReadVariableOp7^CNN_aug_deep_bnorm_drop/dense_61/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::2?
NCNN_aug_deep_bnorm_drop/batch_normalization_52/FusedBatchNormV3/ReadVariableOpNCNN_aug_deep_bnorm_drop/batch_normalization_52/FusedBatchNormV3/ReadVariableOp2?
PCNN_aug_deep_bnorm_drop/batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1PCNN_aug_deep_bnorm_drop/batch_normalization_52/FusedBatchNormV3/ReadVariableOp_12~
=CNN_aug_deep_bnorm_drop/batch_normalization_52/ReadVariableOp=CNN_aug_deep_bnorm_drop/batch_normalization_52/ReadVariableOp2?
?CNN_aug_deep_bnorm_drop/batch_normalization_52/ReadVariableOp_1?CNN_aug_deep_bnorm_drop/batch_normalization_52/ReadVariableOp_12?
NCNN_aug_deep_bnorm_drop/batch_normalization_53/FusedBatchNormV3/ReadVariableOpNCNN_aug_deep_bnorm_drop/batch_normalization_53/FusedBatchNormV3/ReadVariableOp2?
PCNN_aug_deep_bnorm_drop/batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1PCNN_aug_deep_bnorm_drop/batch_normalization_53/FusedBatchNormV3/ReadVariableOp_12~
=CNN_aug_deep_bnorm_drop/batch_normalization_53/ReadVariableOp=CNN_aug_deep_bnorm_drop/batch_normalization_53/ReadVariableOp2?
?CNN_aug_deep_bnorm_drop/batch_normalization_53/ReadVariableOp_1?CNN_aug_deep_bnorm_drop/batch_normalization_53/ReadVariableOp_12?
NCNN_aug_deep_bnorm_drop/batch_normalization_54/FusedBatchNormV3/ReadVariableOpNCNN_aug_deep_bnorm_drop/batch_normalization_54/FusedBatchNormV3/ReadVariableOp2?
PCNN_aug_deep_bnorm_drop/batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1PCNN_aug_deep_bnorm_drop/batch_normalization_54/FusedBatchNormV3/ReadVariableOp_12~
=CNN_aug_deep_bnorm_drop/batch_normalization_54/ReadVariableOp=CNN_aug_deep_bnorm_drop/batch_normalization_54/ReadVariableOp2?
?CNN_aug_deep_bnorm_drop/batch_normalization_54/ReadVariableOp_1?CNN_aug_deep_bnorm_drop/batch_normalization_54/ReadVariableOp_12?
NCNN_aug_deep_bnorm_drop/batch_normalization_55/FusedBatchNormV3/ReadVariableOpNCNN_aug_deep_bnorm_drop/batch_normalization_55/FusedBatchNormV3/ReadVariableOp2?
PCNN_aug_deep_bnorm_drop/batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1PCNN_aug_deep_bnorm_drop/batch_normalization_55/FusedBatchNormV3/ReadVariableOp_12~
=CNN_aug_deep_bnorm_drop/batch_normalization_55/ReadVariableOp=CNN_aug_deep_bnorm_drop/batch_normalization_55/ReadVariableOp2?
?CNN_aug_deep_bnorm_drop/batch_normalization_55/ReadVariableOp_1?CNN_aug_deep_bnorm_drop/batch_normalization_55/ReadVariableOp_12?
NCNN_aug_deep_bnorm_drop/batch_normalization_56/FusedBatchNormV3/ReadVariableOpNCNN_aug_deep_bnorm_drop/batch_normalization_56/FusedBatchNormV3/ReadVariableOp2?
PCNN_aug_deep_bnorm_drop/batch_normalization_56/FusedBatchNormV3/ReadVariableOp_1PCNN_aug_deep_bnorm_drop/batch_normalization_56/FusedBatchNormV3/ReadVariableOp_12~
=CNN_aug_deep_bnorm_drop/batch_normalization_56/ReadVariableOp=CNN_aug_deep_bnorm_drop/batch_normalization_56/ReadVariableOp2?
?CNN_aug_deep_bnorm_drop/batch_normalization_56/ReadVariableOp_1?CNN_aug_deep_bnorm_drop/batch_normalization_56/ReadVariableOp_12?
NCNN_aug_deep_bnorm_drop/batch_normalization_57/FusedBatchNormV3/ReadVariableOpNCNN_aug_deep_bnorm_drop/batch_normalization_57/FusedBatchNormV3/ReadVariableOp2?
PCNN_aug_deep_bnorm_drop/batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1PCNN_aug_deep_bnorm_drop/batch_normalization_57/FusedBatchNormV3/ReadVariableOp_12~
=CNN_aug_deep_bnorm_drop/batch_normalization_57/ReadVariableOp=CNN_aug_deep_bnorm_drop/batch_normalization_57/ReadVariableOp2?
?CNN_aug_deep_bnorm_drop/batch_normalization_57/ReadVariableOp_1?CNN_aug_deep_bnorm_drop/batch_normalization_57/ReadVariableOp_12v
9CNN_aug_deep_bnorm_drop/conv2d_145/BiasAdd/ReadVariableOp9CNN_aug_deep_bnorm_drop/conv2d_145/BiasAdd/ReadVariableOp2t
8CNN_aug_deep_bnorm_drop/conv2d_145/Conv2D/ReadVariableOp8CNN_aug_deep_bnorm_drop/conv2d_145/Conv2D/ReadVariableOp2v
9CNN_aug_deep_bnorm_drop/conv2d_146/BiasAdd/ReadVariableOp9CNN_aug_deep_bnorm_drop/conv2d_146/BiasAdd/ReadVariableOp2t
8CNN_aug_deep_bnorm_drop/conv2d_146/Conv2D/ReadVariableOp8CNN_aug_deep_bnorm_drop/conv2d_146/Conv2D/ReadVariableOp2v
9CNN_aug_deep_bnorm_drop/conv2d_147/BiasAdd/ReadVariableOp9CNN_aug_deep_bnorm_drop/conv2d_147/BiasAdd/ReadVariableOp2t
8CNN_aug_deep_bnorm_drop/conv2d_147/Conv2D/ReadVariableOp8CNN_aug_deep_bnorm_drop/conv2d_147/Conv2D/ReadVariableOp2v
9CNN_aug_deep_bnorm_drop/conv2d_148/BiasAdd/ReadVariableOp9CNN_aug_deep_bnorm_drop/conv2d_148/BiasAdd/ReadVariableOp2t
8CNN_aug_deep_bnorm_drop/conv2d_148/Conv2D/ReadVariableOp8CNN_aug_deep_bnorm_drop/conv2d_148/Conv2D/ReadVariableOp2v
9CNN_aug_deep_bnorm_drop/conv2d_149/BiasAdd/ReadVariableOp9CNN_aug_deep_bnorm_drop/conv2d_149/BiasAdd/ReadVariableOp2t
8CNN_aug_deep_bnorm_drop/conv2d_149/Conv2D/ReadVariableOp8CNN_aug_deep_bnorm_drop/conv2d_149/Conv2D/ReadVariableOp2v
9CNN_aug_deep_bnorm_drop/conv2d_150/BiasAdd/ReadVariableOp9CNN_aug_deep_bnorm_drop/conv2d_150/BiasAdd/ReadVariableOp2t
8CNN_aug_deep_bnorm_drop/conv2d_150/Conv2D/ReadVariableOp8CNN_aug_deep_bnorm_drop/conv2d_150/Conv2D/ReadVariableOp2r
7CNN_aug_deep_bnorm_drop/dense_60/BiasAdd/ReadVariableOp7CNN_aug_deep_bnorm_drop/dense_60/BiasAdd/ReadVariableOp2p
6CNN_aug_deep_bnorm_drop/dense_60/MatMul/ReadVariableOp6CNN_aug_deep_bnorm_drop/dense_60/MatMul/ReadVariableOp2r
7CNN_aug_deep_bnorm_drop/dense_61/BiasAdd/ReadVariableOp7CNN_aug_deep_bnorm_drop/dense_61/BiasAdd/ReadVariableOp2p
6CNN_aug_deep_bnorm_drop/dense_61/MatMul/ReadVariableOp6CNN_aug_deep_bnorm_drop/dense_61/MatMul/ReadVariableOp:[ W
1
_output_shapes
:???????????
"
_user_specified_name
input_31
?	
?
E__inference_dense_61_layer_call_and_return_conditional_losses_2706150

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
?
?
S__inference_batch_normalization_56_layer_call_and_return_conditional_losses_2705901

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
?
?
8__inference_batch_normalization_56_layer_call_fn_2708139

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
S__inference_batch_normalization_56_layer_call_and_return_conditional_losses_27059012
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
?
?
9__inference_CNN_aug_deep_bnorm_drop_layer_call_fn_2707206

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
:?????????*>
_read_only_resource_inputs 
	
 !"%&'(*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_CNN_aug_deep_bnorm_drop_layer_call_and_return_conditional_losses_27063862
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
?
?
S__inference_batch_normalization_52_layer_call_and_return_conditional_losses_2707395

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
?
H
,__inference_dropout_54_layer_call_fn_2707816

inputs
identity?
PartitionedCallPartitionedCallinputs*
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
GPU 2J 8? *P
fKRI
G__inference_dropout_54_layer_call_and_return_conditional_losses_27056932
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????**2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????**:W S
/
_output_shapes
:?????????**
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_56_layer_call_and_return_conditional_losses_2708031

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
?
e
G__inference_dropout_57_layer_call_and_return_conditional_losses_2708331

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_56_layer_call_fn_2708126

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
S__inference_batch_normalization_56_layer_call_and_return_conditional_losses_27058832
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
?
?
S__inference_batch_normalization_53_layer_call_and_return_conditional_losses_2707524

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
G__inference_conv2d_145_layer_call_and_return_conditional_losses_2705326

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
?

?
G__inference_conv2d_148_layer_call_and_return_conditional_losses_2707827

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
G__inference_conv2d_146_layer_call_and_return_conditional_losses_2705456

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
?
?
8__inference_batch_normalization_54_layer_call_fn_2707789

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
S__inference_batch_normalization_54_layer_call_and_return_conditional_losses_27049642
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
S__inference_batch_normalization_57_layer_call_and_return_conditional_losses_2706013

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
?
e
G__inference_dropout_57_layer_call_and_return_conditional_losses_2706084

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_54_layer_call_and_return_conditional_losses_2704964

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
S__inference_batch_normalization_57_layer_call_and_return_conditional_losses_2705257

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
H
,__inference_dropout_53_layer_call_fn_2707641

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_53_layer_call_and_return_conditional_losses_27055622
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
e
G__inference_dropout_56_layer_call_and_return_conditional_losses_2708156

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_53_layer_call_and_return_conditional_losses_2705491

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
8__inference_batch_normalization_54_layer_call_fn_2707776

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
S__inference_batch_normalization_54_layer_call_and_return_conditional_losses_27049332
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
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_2705752

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
?
e
G__inference_dropout_54_layer_call_and_return_conditional_losses_2707806

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????**2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????**2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????**:W S
/
_output_shapes
:?????????**
 
_user_specified_nameinputs
?
f
G__inference_dropout_57_layer_call_and_return_conditional_losses_2708326

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
f
G__inference_dropout_52_layer_call_and_return_conditional_losses_2705427

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout/Const}
dropout/MulMulinputsdropout/Const:output:0*
T0*1
_output_shapes
:???????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*1
_output_shapes
:???????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:???????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:???????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*1
_output_shapes
:???????????2
dropout/Mul_1o
IdentityIdentitydropout/Mul_1:z:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_52_layer_call_and_return_conditional_losses_2707349

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
,__inference_conv2d_146_layer_call_fn_2707486

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
G__inference_conv2d_146_layer_call_and_return_conditional_losses_27054562
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
?	
?
E__inference_dense_60_layer_call_and_return_conditional_losses_2708363

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
8__inference_batch_normalization_55_layer_call_fn_2707964

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
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_27057702
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
?
e
G__inference_dropout_52_layer_call_and_return_conditional_losses_2707456

inputs

identity_1d
IdentityIdentityinputs*
T0*1
_output_shapes
:???????????2

Identitys

Identity_1IdentityIdentity:output:0*
T0*1
_output_shapes
:???????????2

Identity_1"!

identity_1Identity_1:output:0*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_56_layer_call_and_return_conditional_losses_2705883

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
8__inference_batch_normalization_55_layer_call_fn_2707951

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
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_27057522
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
f
G__inference_dropout_55_layer_call_and_return_conditional_losses_2705818

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????**2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????***
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????**2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????**2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????**2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????**2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????**:W S
/
_output_shapes
:?????????**
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_54_layer_call_and_return_conditional_losses_2707745

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
e
,__inference_dropout_53_layer_call_fn_2707636

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_53_layer_call_and_return_conditional_losses_27055572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

*__inference_dense_60_layer_call_fn_2708372

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
E__inference_dense_60_layer_call_and_return_conditional_losses_27061232
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
?
?
S__inference_batch_normalization_53_layer_call_and_return_conditional_losses_2704817

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
?
e
G__inference_dropout_54_layer_call_and_return_conditional_losses_2705693

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????**2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????**2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????**:W S
/
_output_shapes
:?????????**
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_54_layer_call_and_return_conditional_losses_2707763

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
?
e
G__inference_dropout_52_layer_call_and_return_conditional_losses_2705432

inputs

identity_1d
IdentityIdentityinputs*
T0*1
_output_shapes
:???????????2

Identitys

Identity_1IdentityIdentity:output:0*
T0*1
_output_shapes
:???????????2

Identity_1"!

identity_1Identity_1:output:0*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?	
?
E__inference_dense_61_layer_call_and_return_conditional_losses_2708383

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
f
G__inference_dropout_52_layer_call_and_return_conditional_losses_2707451

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout/Const}
dropout/MulMulinputsdropout/Const:output:0*
T0*1
_output_shapes
:???????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*1
_output_shapes
:???????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:???????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:???????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*1
_output_shapes
:???????????2
dropout/Mul_1o
IdentityIdentitydropout/Mul_1:z:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
9__inference_CNN_aug_deep_bnorm_drop_layer_call_fn_2707291

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
GPU 2J 8? *]
fXRV
T__inference_CNN_aug_deep_bnorm_drop_layer_call_and_return_conditional_losses_27065792
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
S__inference_batch_normalization_53_layer_call_and_return_conditional_losses_2704848

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
?
H
,__inference_dropout_56_layer_call_fn_2708166

inputs
identity?
PartitionedCallPartitionedCallinputs*
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
GPU 2J 8? *P
fKRI
G__inference_dropout_56_layer_call_and_return_conditional_losses_27059542
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_53_layer_call_fn_2707614

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
S__inference_batch_normalization_53_layer_call_and_return_conditional_losses_27055092
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
8__inference_batch_normalization_52_layer_call_fn_2707439

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
S__inference_batch_normalization_52_layer_call_and_return_conditional_losses_27053792
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
S__inference_batch_normalization_56_layer_call_and_return_conditional_losses_2708049

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
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_2707938

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
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_2707920

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
8__inference_batch_normalization_53_layer_call_fn_2707550

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
S__inference_batch_normalization_53_layer_call_and_return_conditional_losses_27048482
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
c
G__inference_flatten_30_layer_call_and_return_conditional_losses_2708347

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
?
e
G__inference_dropout_56_layer_call_and_return_conditional_losses_2705954

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_52_layer_call_fn_2707375

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
S__inference_batch_normalization_52_layer_call_and_return_conditional_losses_27047442
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
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_2707856

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
N
2__inference_max_pooling2d_59_layer_call_fn_2705091

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
M__inference_max_pooling2d_59_layer_call_and_return_conditional_losses_27050852
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
S__inference_batch_normalization_56_layer_call_and_return_conditional_losses_2708095

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
?
?
8__inference_batch_normalization_56_layer_call_fn_2708075

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
S__inference_batch_normalization_56_layer_call_and_return_conditional_losses_27051842
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
S__inference_batch_normalization_53_layer_call_and_return_conditional_losses_2707506

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
?
?
9__inference_CNN_aug_deep_bnorm_drop_layer_call_fn_2706662
input_31
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
StatefulPartitionedCallStatefulPartitionedCallinput_31unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU 2J 8? *]
fXRV
T__inference_CNN_aug_deep_bnorm_drop_layer_call_and_return_conditional_losses_27065792
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
input_31
?
?
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_2707874

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
?x
?
T__inference_CNN_aug_deep_bnorm_drop_layer_call_and_return_conditional_losses_2706275
input_31
conv2d_145_2706170
conv2d_145_2706172"
batch_normalization_52_2706175"
batch_normalization_52_2706177"
batch_normalization_52_2706179"
batch_normalization_52_2706181
conv2d_146_2706185
conv2d_146_2706187"
batch_normalization_53_2706190"
batch_normalization_53_2706192"
batch_normalization_53_2706194"
batch_normalization_53_2706196
conv2d_147_2706201
conv2d_147_2706203"
batch_normalization_54_2706206"
batch_normalization_54_2706208"
batch_normalization_54_2706210"
batch_normalization_54_2706212
conv2d_148_2706216
conv2d_148_2706218"
batch_normalization_55_2706221"
batch_normalization_55_2706223"
batch_normalization_55_2706225"
batch_normalization_55_2706227
conv2d_149_2706232
conv2d_149_2706234"
batch_normalization_56_2706237"
batch_normalization_56_2706239"
batch_normalization_56_2706241"
batch_normalization_56_2706243
conv2d_150_2706247
conv2d_150_2706249"
batch_normalization_57_2706252"
batch_normalization_57_2706254"
batch_normalization_57_2706256"
batch_normalization_57_2706258
dense_60_2706264
dense_60_2706266
dense_61_2706269
dense_61_2706271
identity??.batch_normalization_52/StatefulPartitionedCall?.batch_normalization_53/StatefulPartitionedCall?.batch_normalization_54/StatefulPartitionedCall?.batch_normalization_55/StatefulPartitionedCall?.batch_normalization_56/StatefulPartitionedCall?.batch_normalization_57/StatefulPartitionedCall?"conv2d_145/StatefulPartitionedCall?"conv2d_146/StatefulPartitionedCall?"conv2d_147/StatefulPartitionedCall?"conv2d_148/StatefulPartitionedCall?"conv2d_149/StatefulPartitionedCall?"conv2d_150/StatefulPartitionedCall? dense_60/StatefulPartitionedCall? dense_61/StatefulPartitionedCall?
"conv2d_145/StatefulPartitionedCallStatefulPartitionedCallinput_31conv2d_145_2706170conv2d_145_2706172*
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
G__inference_conv2d_145_layer_call_and_return_conditional_losses_27053262$
"conv2d_145/StatefulPartitionedCall?
.batch_normalization_52/StatefulPartitionedCallStatefulPartitionedCall+conv2d_145/StatefulPartitionedCall:output:0batch_normalization_52_2706175batch_normalization_52_2706177batch_normalization_52_2706179batch_normalization_52_2706181*
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
S__inference_batch_normalization_52_layer_call_and_return_conditional_losses_270537920
.batch_normalization_52/StatefulPartitionedCall?
dropout_52/PartitionedCallPartitionedCall7batch_normalization_52/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_52_layer_call_and_return_conditional_losses_27054322
dropout_52/PartitionedCall?
"conv2d_146/StatefulPartitionedCallStatefulPartitionedCall#dropout_52/PartitionedCall:output:0conv2d_146_2706185conv2d_146_2706187*
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
G__inference_conv2d_146_layer_call_and_return_conditional_losses_27054562$
"conv2d_146/StatefulPartitionedCall?
.batch_normalization_53/StatefulPartitionedCallStatefulPartitionedCall+conv2d_146/StatefulPartitionedCall:output:0batch_normalization_53_2706190batch_normalization_53_2706192batch_normalization_53_2706194batch_normalization_53_2706196*
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
S__inference_batch_normalization_53_layer_call_and_return_conditional_losses_270550920
.batch_normalization_53/StatefulPartitionedCall?
dropout_53/PartitionedCallPartitionedCall7batch_normalization_53/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_53_layer_call_and_return_conditional_losses_27055622
dropout_53/PartitionedCall?
 max_pooling2d_58/PartitionedCallPartitionedCall#dropout_53/PartitionedCall:output:0*
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
M__inference_max_pooling2d_58_layer_call_and_return_conditional_losses_27048652"
 max_pooling2d_58/PartitionedCall?
"conv2d_147/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_58/PartitionedCall:output:0conv2d_147_2706201conv2d_147_2706203*
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
G__inference_conv2d_147_layer_call_and_return_conditional_losses_27055872$
"conv2d_147/StatefulPartitionedCall?
.batch_normalization_54/StatefulPartitionedCallStatefulPartitionedCall+conv2d_147/StatefulPartitionedCall:output:0batch_normalization_54_2706206batch_normalization_54_2706208batch_normalization_54_2706210batch_normalization_54_2706212*
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
S__inference_batch_normalization_54_layer_call_and_return_conditional_losses_270564020
.batch_normalization_54/StatefulPartitionedCall?
dropout_54/PartitionedCallPartitionedCall7batch_normalization_54/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *P
fKRI
G__inference_dropout_54_layer_call_and_return_conditional_losses_27056932
dropout_54/PartitionedCall?
"conv2d_148/StatefulPartitionedCallStatefulPartitionedCall#dropout_54/PartitionedCall:output:0conv2d_148_2706216conv2d_148_2706218*
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
G__inference_conv2d_148_layer_call_and_return_conditional_losses_27057172$
"conv2d_148/StatefulPartitionedCall?
.batch_normalization_55/StatefulPartitionedCallStatefulPartitionedCall+conv2d_148/StatefulPartitionedCall:output:0batch_normalization_55_2706221batch_normalization_55_2706223batch_normalization_55_2706225batch_normalization_55_2706227*
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
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_270577020
.batch_normalization_55/StatefulPartitionedCall?
dropout_55/PartitionedCallPartitionedCall7batch_normalization_55/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *P
fKRI
G__inference_dropout_55_layer_call_and_return_conditional_losses_27058232
dropout_55/PartitionedCall?
 max_pooling2d_59/PartitionedCallPartitionedCall#dropout_55/PartitionedCall:output:0*
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
M__inference_max_pooling2d_59_layer_call_and_return_conditional_losses_27050852"
 max_pooling2d_59/PartitionedCall?
"conv2d_149/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_59/PartitionedCall:output:0conv2d_149_2706232conv2d_149_2706234*
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
G__inference_conv2d_149_layer_call_and_return_conditional_losses_27058482$
"conv2d_149/StatefulPartitionedCall?
.batch_normalization_56/StatefulPartitionedCallStatefulPartitionedCall+conv2d_149/StatefulPartitionedCall:output:0batch_normalization_56_2706237batch_normalization_56_2706239batch_normalization_56_2706241batch_normalization_56_2706243*
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
S__inference_batch_normalization_56_layer_call_and_return_conditional_losses_270590120
.batch_normalization_56/StatefulPartitionedCall?
dropout_56/PartitionedCallPartitionedCall7batch_normalization_56/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *P
fKRI
G__inference_dropout_56_layer_call_and_return_conditional_losses_27059542
dropout_56/PartitionedCall?
"conv2d_150/StatefulPartitionedCallStatefulPartitionedCall#dropout_56/PartitionedCall:output:0conv2d_150_2706247conv2d_150_2706249*
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
G__inference_conv2d_150_layer_call_and_return_conditional_losses_27059782$
"conv2d_150/StatefulPartitionedCall?
.batch_normalization_57/StatefulPartitionedCallStatefulPartitionedCall+conv2d_150/StatefulPartitionedCall:output:0batch_normalization_57_2706252batch_normalization_57_2706254batch_normalization_57_2706256batch_normalization_57_2706258*
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
S__inference_batch_normalization_57_layer_call_and_return_conditional_losses_270603120
.batch_normalization_57/StatefulPartitionedCall?
dropout_57/PartitionedCallPartitionedCall7batch_normalization_57/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *P
fKRI
G__inference_dropout_57_layer_call_and_return_conditional_losses_27060842
dropout_57/PartitionedCall?
 max_pooling2d_60/PartitionedCallPartitionedCall#dropout_57/PartitionedCall:output:0*
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
M__inference_max_pooling2d_60_layer_call_and_return_conditional_losses_27053052"
 max_pooling2d_60/PartitionedCall?
flatten_30/PartitionedCallPartitionedCall)max_pooling2d_60/PartitionedCall:output:0*
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
G__inference_flatten_30_layer_call_and_return_conditional_losses_27061042
flatten_30/PartitionedCall?
 dense_60/StatefulPartitionedCallStatefulPartitionedCall#flatten_30/PartitionedCall:output:0dense_60_2706264dense_60_2706266*
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
E__inference_dense_60_layer_call_and_return_conditional_losses_27061232"
 dense_60/StatefulPartitionedCall?
 dense_61/StatefulPartitionedCallStatefulPartitionedCall)dense_60/StatefulPartitionedCall:output:0dense_61_2706269dense_61_2706271*
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
E__inference_dense_61_layer_call_and_return_conditional_losses_27061502"
 dense_61/StatefulPartitionedCall?
IdentityIdentity)dense_61/StatefulPartitionedCall:output:0/^batch_normalization_52/StatefulPartitionedCall/^batch_normalization_53/StatefulPartitionedCall/^batch_normalization_54/StatefulPartitionedCall/^batch_normalization_55/StatefulPartitionedCall/^batch_normalization_56/StatefulPartitionedCall/^batch_normalization_57/StatefulPartitionedCall#^conv2d_145/StatefulPartitionedCall#^conv2d_146/StatefulPartitionedCall#^conv2d_147/StatefulPartitionedCall#^conv2d_148/StatefulPartitionedCall#^conv2d_149/StatefulPartitionedCall#^conv2d_150/StatefulPartitionedCall!^dense_60/StatefulPartitionedCall!^dense_61/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_52/StatefulPartitionedCall.batch_normalization_52/StatefulPartitionedCall2`
.batch_normalization_53/StatefulPartitionedCall.batch_normalization_53/StatefulPartitionedCall2`
.batch_normalization_54/StatefulPartitionedCall.batch_normalization_54/StatefulPartitionedCall2`
.batch_normalization_55/StatefulPartitionedCall.batch_normalization_55/StatefulPartitionedCall2`
.batch_normalization_56/StatefulPartitionedCall.batch_normalization_56/StatefulPartitionedCall2`
.batch_normalization_57/StatefulPartitionedCall.batch_normalization_57/StatefulPartitionedCall2H
"conv2d_145/StatefulPartitionedCall"conv2d_145/StatefulPartitionedCall2H
"conv2d_146/StatefulPartitionedCall"conv2d_146/StatefulPartitionedCall2H
"conv2d_147/StatefulPartitionedCall"conv2d_147/StatefulPartitionedCall2H
"conv2d_148/StatefulPartitionedCall"conv2d_148/StatefulPartitionedCall2H
"conv2d_149/StatefulPartitionedCall"conv2d_149/StatefulPartitionedCall2H
"conv2d_150/StatefulPartitionedCall"conv2d_150/StatefulPartitionedCall2D
 dense_60/StatefulPartitionedCall dense_60/StatefulPartitionedCall2D
 dense_61/StatefulPartitionedCall dense_61/StatefulPartitionedCall:[ W
1
_output_shapes
:???????????
"
_user_specified_name
input_31
?	
?
E__inference_dense_60_layer_call_and_return_conditional_losses_2706123

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
?
?
S__inference_batch_normalization_57_layer_call_and_return_conditional_losses_2706031

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
?
?
,__inference_conv2d_147_layer_call_fn_2707661

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
G__inference_conv2d_147_layer_call_and_return_conditional_losses_27055872
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
?
e
,__inference_dropout_52_layer_call_fn_2707461

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_52_layer_call_and_return_conditional_losses_27054272
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_54_layer_call_fn_2707725

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
S__inference_batch_normalization_54_layer_call_and_return_conditional_losses_27056402
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
?
?
8__inference_batch_normalization_53_layer_call_fn_2707537

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
S__inference_batch_normalization_53_layer_call_and_return_conditional_losses_27048172
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
H
,__inference_dropout_57_layer_call_fn_2708341

inputs
identity?
PartitionedCallPartitionedCallinputs*
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
GPU 2J 8? *P
fKRI
G__inference_dropout_57_layer_call_and_return_conditional_losses_27060842
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_53_layer_call_and_return_conditional_losses_2705509

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
f
G__inference_dropout_53_layer_call_and_return_conditional_losses_2705557

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout/Const}
dropout/MulMulinputsdropout/Const:output:0*
T0*1
_output_shapes
:???????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*1
_output_shapes
:???????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:???????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:???????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*1
_output_shapes
:???????????2
dropout/Mul_1o
IdentityIdentitydropout/Mul_1:z:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_56_layer_call_and_return_conditional_losses_2705184

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
8__inference_batch_normalization_53_layer_call_fn_2707601

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
S__inference_batch_normalization_53_layer_call_and_return_conditional_losses_27054912
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
?
?
8__inference_batch_normalization_57_layer_call_fn_2708250

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
S__inference_batch_normalization_57_layer_call_and_return_conditional_losses_27052882
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
,__inference_conv2d_148_layer_call_fn_2707836

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
G__inference_conv2d_148_layer_call_and_return_conditional_losses_27057172
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
?
?
S__inference_batch_normalization_53_layer_call_and_return_conditional_losses_2707570

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
G__inference_conv2d_149_layer_call_and_return_conditional_losses_2705848

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
?
e
,__inference_dropout_57_layer_call_fn_2708336

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
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
GPU 2J 8? *P
fKRI
G__inference_dropout_57_layer_call_and_return_conditional_losses_27060792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
N
2__inference_max_pooling2d_58_layer_call_fn_2704871

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
M__inference_max_pooling2d_58_layer_call_and_return_conditional_losses_27048652
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
?

?
G__inference_conv2d_146_layer_call_and_return_conditional_losses_2707477

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
??
?<
#__inference__traced_restore_2709055
file_prefix&
"assignvariableop_conv2d_145_kernel&
"assignvariableop_1_conv2d_145_bias3
/assignvariableop_2_batch_normalization_52_gamma2
.assignvariableop_3_batch_normalization_52_beta9
5assignvariableop_4_batch_normalization_52_moving_mean=
9assignvariableop_5_batch_normalization_52_moving_variance(
$assignvariableop_6_conv2d_146_kernel&
"assignvariableop_7_conv2d_146_bias3
/assignvariableop_8_batch_normalization_53_gamma2
.assignvariableop_9_batch_normalization_53_beta:
6assignvariableop_10_batch_normalization_53_moving_mean>
:assignvariableop_11_batch_normalization_53_moving_variance)
%assignvariableop_12_conv2d_147_kernel'
#assignvariableop_13_conv2d_147_bias4
0assignvariableop_14_batch_normalization_54_gamma3
/assignvariableop_15_batch_normalization_54_beta:
6assignvariableop_16_batch_normalization_54_moving_mean>
:assignvariableop_17_batch_normalization_54_moving_variance)
%assignvariableop_18_conv2d_148_kernel'
#assignvariableop_19_conv2d_148_bias4
0assignvariableop_20_batch_normalization_55_gamma3
/assignvariableop_21_batch_normalization_55_beta:
6assignvariableop_22_batch_normalization_55_moving_mean>
:assignvariableop_23_batch_normalization_55_moving_variance)
%assignvariableop_24_conv2d_149_kernel'
#assignvariableop_25_conv2d_149_bias4
0assignvariableop_26_batch_normalization_56_gamma3
/assignvariableop_27_batch_normalization_56_beta:
6assignvariableop_28_batch_normalization_56_moving_mean>
:assignvariableop_29_batch_normalization_56_moving_variance)
%assignvariableop_30_conv2d_150_kernel'
#assignvariableop_31_conv2d_150_bias4
0assignvariableop_32_batch_normalization_57_gamma3
/assignvariableop_33_batch_normalization_57_beta:
6assignvariableop_34_batch_normalization_57_moving_mean>
:assignvariableop_35_batch_normalization_57_moving_variance'
#assignvariableop_36_dense_60_kernel%
!assignvariableop_37_dense_60_bias'
#assignvariableop_38_dense_61_kernel%
!assignvariableop_39_dense_61_bias!
assignvariableop_40_adam_iter#
assignvariableop_41_adam_beta_1#
assignvariableop_42_adam_beta_2"
assignvariableop_43_adam_decay*
&assignvariableop_44_adam_learning_rate
assignvariableop_45_total
assignvariableop_46_count
assignvariableop_47_total_1
assignvariableop_48_count_10
,assignvariableop_49_adam_conv2d_145_kernel_m.
*assignvariableop_50_adam_conv2d_145_bias_m;
7assignvariableop_51_adam_batch_normalization_52_gamma_m:
6assignvariableop_52_adam_batch_normalization_52_beta_m0
,assignvariableop_53_adam_conv2d_146_kernel_m.
*assignvariableop_54_adam_conv2d_146_bias_m;
7assignvariableop_55_adam_batch_normalization_53_gamma_m:
6assignvariableop_56_adam_batch_normalization_53_beta_m0
,assignvariableop_57_adam_conv2d_147_kernel_m.
*assignvariableop_58_adam_conv2d_147_bias_m;
7assignvariableop_59_adam_batch_normalization_54_gamma_m:
6assignvariableop_60_adam_batch_normalization_54_beta_m0
,assignvariableop_61_adam_conv2d_148_kernel_m.
*assignvariableop_62_adam_conv2d_148_bias_m;
7assignvariableop_63_adam_batch_normalization_55_gamma_m:
6assignvariableop_64_adam_batch_normalization_55_beta_m0
,assignvariableop_65_adam_conv2d_149_kernel_m.
*assignvariableop_66_adam_conv2d_149_bias_m;
7assignvariableop_67_adam_batch_normalization_56_gamma_m:
6assignvariableop_68_adam_batch_normalization_56_beta_m0
,assignvariableop_69_adam_conv2d_150_kernel_m.
*assignvariableop_70_adam_conv2d_150_bias_m;
7assignvariableop_71_adam_batch_normalization_57_gamma_m:
6assignvariableop_72_adam_batch_normalization_57_beta_m.
*assignvariableop_73_adam_dense_60_kernel_m,
(assignvariableop_74_adam_dense_60_bias_m.
*assignvariableop_75_adam_dense_61_kernel_m,
(assignvariableop_76_adam_dense_61_bias_m0
,assignvariableop_77_adam_conv2d_145_kernel_v.
*assignvariableop_78_adam_conv2d_145_bias_v;
7assignvariableop_79_adam_batch_normalization_52_gamma_v:
6assignvariableop_80_adam_batch_normalization_52_beta_v0
,assignvariableop_81_adam_conv2d_146_kernel_v.
*assignvariableop_82_adam_conv2d_146_bias_v;
7assignvariableop_83_adam_batch_normalization_53_gamma_v:
6assignvariableop_84_adam_batch_normalization_53_beta_v0
,assignvariableop_85_adam_conv2d_147_kernel_v.
*assignvariableop_86_adam_conv2d_147_bias_v;
7assignvariableop_87_adam_batch_normalization_54_gamma_v:
6assignvariableop_88_adam_batch_normalization_54_beta_v0
,assignvariableop_89_adam_conv2d_148_kernel_v.
*assignvariableop_90_adam_conv2d_148_bias_v;
7assignvariableop_91_adam_batch_normalization_55_gamma_v:
6assignvariableop_92_adam_batch_normalization_55_beta_v0
,assignvariableop_93_adam_conv2d_149_kernel_v.
*assignvariableop_94_adam_conv2d_149_bias_v;
7assignvariableop_95_adam_batch_normalization_56_gamma_v:
6assignvariableop_96_adam_batch_normalization_56_beta_v0
,assignvariableop_97_adam_conv2d_150_kernel_v.
*assignvariableop_98_adam_conv2d_150_bias_v;
7assignvariableop_99_adam_batch_normalization_57_gamma_v;
7assignvariableop_100_adam_batch_normalization_57_beta_v/
+assignvariableop_101_adam_dense_60_kernel_v-
)assignvariableop_102_adam_dense_60_bias_v/
+assignvariableop_103_adam_dense_61_kernel_v-
)assignvariableop_104_adam_dense_61_bias_v
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
AssignVariableOpAssignVariableOp"assignvariableop_conv2d_145_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp"assignvariableop_1_conv2d_145_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp/assignvariableop_2_batch_normalization_52_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp.assignvariableop_3_batch_normalization_52_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp5assignvariableop_4_batch_normalization_52_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp9assignvariableop_5_batch_normalization_52_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp$assignvariableop_6_conv2d_146_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp"assignvariableop_7_conv2d_146_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp/assignvariableop_8_batch_normalization_53_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_53_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp6assignvariableop_10_batch_normalization_53_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_normalization_53_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp%assignvariableop_12_conv2d_147_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp#assignvariableop_13_conv2d_147_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp0assignvariableop_14_batch_normalization_54_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp/assignvariableop_15_batch_normalization_54_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp6assignvariableop_16_batch_normalization_54_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp:assignvariableop_17_batch_normalization_54_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp%assignvariableop_18_conv2d_148_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp#assignvariableop_19_conv2d_148_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp0assignvariableop_20_batch_normalization_55_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp/assignvariableop_21_batch_normalization_55_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp6assignvariableop_22_batch_normalization_55_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp:assignvariableop_23_batch_normalization_55_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp%assignvariableop_24_conv2d_149_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp#assignvariableop_25_conv2d_149_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp0assignvariableop_26_batch_normalization_56_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp/assignvariableop_27_batch_normalization_56_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp6assignvariableop_28_batch_normalization_56_moving_meanIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp:assignvariableop_29_batch_normalization_56_moving_varianceIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp%assignvariableop_30_conv2d_150_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp#assignvariableop_31_conv2d_150_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp0assignvariableop_32_batch_normalization_57_gammaIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp/assignvariableop_33_batch_normalization_57_betaIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp6assignvariableop_34_batch_normalization_57_moving_meanIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp:assignvariableop_35_batch_normalization_57_moving_varianceIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp#assignvariableop_36_dense_60_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp!assignvariableop_37_dense_60_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp#assignvariableop_38_dense_61_kernelIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp!assignvariableop_39_dense_61_biasIdentity_39:output:0"/device:CPU:0*
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
AssignVariableOp_49AssignVariableOp,assignvariableop_49_adam_conv2d_145_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp*assignvariableop_50_adam_conv2d_145_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp7assignvariableop_51_adam_batch_normalization_52_gamma_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp6assignvariableop_52_adam_batch_normalization_52_beta_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp,assignvariableop_53_adam_conv2d_146_kernel_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp*assignvariableop_54_adam_conv2d_146_bias_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp7assignvariableop_55_adam_batch_normalization_53_gamma_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp6assignvariableop_56_adam_batch_normalization_53_beta_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp,assignvariableop_57_adam_conv2d_147_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp*assignvariableop_58_adam_conv2d_147_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp7assignvariableop_59_adam_batch_normalization_54_gamma_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp6assignvariableop_60_adam_batch_normalization_54_beta_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp,assignvariableop_61_adam_conv2d_148_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp*assignvariableop_62_adam_conv2d_148_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp7assignvariableop_63_adam_batch_normalization_55_gamma_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp6assignvariableop_64_adam_batch_normalization_55_beta_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp,assignvariableop_65_adam_conv2d_149_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp*assignvariableop_66_adam_conv2d_149_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOp7assignvariableop_67_adam_batch_normalization_56_gamma_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp6assignvariableop_68_adam_batch_normalization_56_beta_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOp,assignvariableop_69_adam_conv2d_150_kernel_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp*assignvariableop_70_adam_conv2d_150_bias_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOp7assignvariableop_71_adam_batch_normalization_57_gamma_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOp6assignvariableop_72_adam_batch_normalization_57_beta_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73?
AssignVariableOp_73AssignVariableOp*assignvariableop_73_adam_dense_60_kernel_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOp(assignvariableop_74_adam_dense_60_bias_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75?
AssignVariableOp_75AssignVariableOp*assignvariableop_75_adam_dense_61_kernel_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76?
AssignVariableOp_76AssignVariableOp(assignvariableop_76_adam_dense_61_bias_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77?
AssignVariableOp_77AssignVariableOp,assignvariableop_77_adam_conv2d_145_kernel_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78?
AssignVariableOp_78AssignVariableOp*assignvariableop_78_adam_conv2d_145_bias_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79?
AssignVariableOp_79AssignVariableOp7assignvariableop_79_adam_batch_normalization_52_gamma_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80?
AssignVariableOp_80AssignVariableOp6assignvariableop_80_adam_batch_normalization_52_beta_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81?
AssignVariableOp_81AssignVariableOp,assignvariableop_81_adam_conv2d_146_kernel_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82?
AssignVariableOp_82AssignVariableOp*assignvariableop_82_adam_conv2d_146_bias_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83?
AssignVariableOp_83AssignVariableOp7assignvariableop_83_adam_batch_normalization_53_gamma_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84?
AssignVariableOp_84AssignVariableOp6assignvariableop_84_adam_batch_normalization_53_beta_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85?
AssignVariableOp_85AssignVariableOp,assignvariableop_85_adam_conv2d_147_kernel_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86?
AssignVariableOp_86AssignVariableOp*assignvariableop_86_adam_conv2d_147_bias_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87?
AssignVariableOp_87AssignVariableOp7assignvariableop_87_adam_batch_normalization_54_gamma_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88?
AssignVariableOp_88AssignVariableOp6assignvariableop_88_adam_batch_normalization_54_beta_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89?
AssignVariableOp_89AssignVariableOp,assignvariableop_89_adam_conv2d_148_kernel_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90?
AssignVariableOp_90AssignVariableOp*assignvariableop_90_adam_conv2d_148_bias_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91?
AssignVariableOp_91AssignVariableOp7assignvariableop_91_adam_batch_normalization_55_gamma_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92?
AssignVariableOp_92AssignVariableOp6assignvariableop_92_adam_batch_normalization_55_beta_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93?
AssignVariableOp_93AssignVariableOp,assignvariableop_93_adam_conv2d_149_kernel_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94?
AssignVariableOp_94AssignVariableOp*assignvariableop_94_adam_conv2d_149_bias_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95?
AssignVariableOp_95AssignVariableOp7assignvariableop_95_adam_batch_normalization_56_gamma_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96?
AssignVariableOp_96AssignVariableOp6assignvariableop_96_adam_batch_normalization_56_beta_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_96n
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:2
Identity_97?
AssignVariableOp_97AssignVariableOp,assignvariableop_97_adam_conv2d_150_kernel_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_97n
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:2
Identity_98?
AssignVariableOp_98AssignVariableOp*assignvariableop_98_adam_conv2d_150_bias_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_98n
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:2
Identity_99?
AssignVariableOp_99AssignVariableOp7assignvariableop_99_adam_batch_normalization_57_gamma_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99q
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:2
Identity_100?
AssignVariableOp_100AssignVariableOp7assignvariableop_100_adam_batch_normalization_57_beta_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_100q
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:2
Identity_101?
AssignVariableOp_101AssignVariableOp+assignvariableop_101_adam_dense_60_kernel_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_101q
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:2
Identity_102?
AssignVariableOp_102AssignVariableOp)assignvariableop_102_adam_dense_60_bias_vIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_102q
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:2
Identity_103?
AssignVariableOp_103AssignVariableOp+assignvariableop_103_adam_dense_61_kernel_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_103q
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:2
Identity_104?
AssignVariableOp_104AssignVariableOp)assignvariableop_104_adam_dense_61_bias_vIdentity_104:output:0"/device:CPU:0*
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
?
e
G__inference_dropout_53_layer_call_and_return_conditional_losses_2707631

inputs

identity_1d
IdentityIdentityinputs*
T0*1
_output_shapes
:???????????2

Identitys

Identity_1IdentityIdentity:output:0*
T0*1
_output_shapes
:???????????2

Identity_1"!

identity_1Identity_1:output:0*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
H
,__inference_flatten_30_layer_call_fn_2708352

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
G__inference_flatten_30_layer_call_and_return_conditional_losses_27061042
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
?
?
S__inference_batch_normalization_52_layer_call_and_return_conditional_losses_2704744

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
8__inference_batch_normalization_54_layer_call_fn_2707712

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
S__inference_batch_normalization_54_layer_call_and_return_conditional_losses_27056222
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
?
?
,__inference_conv2d_145_layer_call_fn_2707311

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
G__inference_conv2d_145_layer_call_and_return_conditional_losses_27053262
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
?
e
,__inference_dropout_55_layer_call_fn_2707986

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
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
GPU 2J 8? *P
fKRI
G__inference_dropout_55_layer_call_and_return_conditional_losses_27058182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????**2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????**22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????**
 
_user_specified_nameinputs
??
?
T__inference_CNN_aug_deep_bnorm_drop_layer_call_and_return_conditional_losses_2706167
input_31
conv2d_145_2705337
conv2d_145_2705339"
batch_normalization_52_2705406"
batch_normalization_52_2705408"
batch_normalization_52_2705410"
batch_normalization_52_2705412
conv2d_146_2705467
conv2d_146_2705469"
batch_normalization_53_2705536"
batch_normalization_53_2705538"
batch_normalization_53_2705540"
batch_normalization_53_2705542
conv2d_147_2705598
conv2d_147_2705600"
batch_normalization_54_2705667"
batch_normalization_54_2705669"
batch_normalization_54_2705671"
batch_normalization_54_2705673
conv2d_148_2705728
conv2d_148_2705730"
batch_normalization_55_2705797"
batch_normalization_55_2705799"
batch_normalization_55_2705801"
batch_normalization_55_2705803
conv2d_149_2705859
conv2d_149_2705861"
batch_normalization_56_2705928"
batch_normalization_56_2705930"
batch_normalization_56_2705932"
batch_normalization_56_2705934
conv2d_150_2705989
conv2d_150_2705991"
batch_normalization_57_2706058"
batch_normalization_57_2706060"
batch_normalization_57_2706062"
batch_normalization_57_2706064
dense_60_2706134
dense_60_2706136
dense_61_2706161
dense_61_2706163
identity??.batch_normalization_52/StatefulPartitionedCall?.batch_normalization_53/StatefulPartitionedCall?.batch_normalization_54/StatefulPartitionedCall?.batch_normalization_55/StatefulPartitionedCall?.batch_normalization_56/StatefulPartitionedCall?.batch_normalization_57/StatefulPartitionedCall?"conv2d_145/StatefulPartitionedCall?"conv2d_146/StatefulPartitionedCall?"conv2d_147/StatefulPartitionedCall?"conv2d_148/StatefulPartitionedCall?"conv2d_149/StatefulPartitionedCall?"conv2d_150/StatefulPartitionedCall? dense_60/StatefulPartitionedCall? dense_61/StatefulPartitionedCall?"dropout_52/StatefulPartitionedCall?"dropout_53/StatefulPartitionedCall?"dropout_54/StatefulPartitionedCall?"dropout_55/StatefulPartitionedCall?"dropout_56/StatefulPartitionedCall?"dropout_57/StatefulPartitionedCall?
"conv2d_145/StatefulPartitionedCallStatefulPartitionedCallinput_31conv2d_145_2705337conv2d_145_2705339*
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
G__inference_conv2d_145_layer_call_and_return_conditional_losses_27053262$
"conv2d_145/StatefulPartitionedCall?
.batch_normalization_52/StatefulPartitionedCallStatefulPartitionedCall+conv2d_145/StatefulPartitionedCall:output:0batch_normalization_52_2705406batch_normalization_52_2705408batch_normalization_52_2705410batch_normalization_52_2705412*
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
S__inference_batch_normalization_52_layer_call_and_return_conditional_losses_270536120
.batch_normalization_52/StatefulPartitionedCall?
"dropout_52/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_52/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_52_layer_call_and_return_conditional_losses_27054272$
"dropout_52/StatefulPartitionedCall?
"conv2d_146/StatefulPartitionedCallStatefulPartitionedCall+dropout_52/StatefulPartitionedCall:output:0conv2d_146_2705467conv2d_146_2705469*
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
G__inference_conv2d_146_layer_call_and_return_conditional_losses_27054562$
"conv2d_146/StatefulPartitionedCall?
.batch_normalization_53/StatefulPartitionedCallStatefulPartitionedCall+conv2d_146/StatefulPartitionedCall:output:0batch_normalization_53_2705536batch_normalization_53_2705538batch_normalization_53_2705540batch_normalization_53_2705542*
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
S__inference_batch_normalization_53_layer_call_and_return_conditional_losses_270549120
.batch_normalization_53/StatefulPartitionedCall?
"dropout_53/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_53/StatefulPartitionedCall:output:0#^dropout_52/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_53_layer_call_and_return_conditional_losses_27055572$
"dropout_53/StatefulPartitionedCall?
 max_pooling2d_58/PartitionedCallPartitionedCall+dropout_53/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_58_layer_call_and_return_conditional_losses_27048652"
 max_pooling2d_58/PartitionedCall?
"conv2d_147/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_58/PartitionedCall:output:0conv2d_147_2705598conv2d_147_2705600*
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
G__inference_conv2d_147_layer_call_and_return_conditional_losses_27055872$
"conv2d_147/StatefulPartitionedCall?
.batch_normalization_54/StatefulPartitionedCallStatefulPartitionedCall+conv2d_147/StatefulPartitionedCall:output:0batch_normalization_54_2705667batch_normalization_54_2705669batch_normalization_54_2705671batch_normalization_54_2705673*
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
S__inference_batch_normalization_54_layer_call_and_return_conditional_losses_270562220
.batch_normalization_54/StatefulPartitionedCall?
"dropout_54/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_54/StatefulPartitionedCall:output:0#^dropout_53/StatefulPartitionedCall*
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
GPU 2J 8? *P
fKRI
G__inference_dropout_54_layer_call_and_return_conditional_losses_27056882$
"dropout_54/StatefulPartitionedCall?
"conv2d_148/StatefulPartitionedCallStatefulPartitionedCall+dropout_54/StatefulPartitionedCall:output:0conv2d_148_2705728conv2d_148_2705730*
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
G__inference_conv2d_148_layer_call_and_return_conditional_losses_27057172$
"conv2d_148/StatefulPartitionedCall?
.batch_normalization_55/StatefulPartitionedCallStatefulPartitionedCall+conv2d_148/StatefulPartitionedCall:output:0batch_normalization_55_2705797batch_normalization_55_2705799batch_normalization_55_2705801batch_normalization_55_2705803*
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
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_270575220
.batch_normalization_55/StatefulPartitionedCall?
"dropout_55/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_55/StatefulPartitionedCall:output:0#^dropout_54/StatefulPartitionedCall*
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
GPU 2J 8? *P
fKRI
G__inference_dropout_55_layer_call_and_return_conditional_losses_27058182$
"dropout_55/StatefulPartitionedCall?
 max_pooling2d_59/PartitionedCallPartitionedCall+dropout_55/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_59_layer_call_and_return_conditional_losses_27050852"
 max_pooling2d_59/PartitionedCall?
"conv2d_149/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_59/PartitionedCall:output:0conv2d_149_2705859conv2d_149_2705861*
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
G__inference_conv2d_149_layer_call_and_return_conditional_losses_27058482$
"conv2d_149/StatefulPartitionedCall?
.batch_normalization_56/StatefulPartitionedCallStatefulPartitionedCall+conv2d_149/StatefulPartitionedCall:output:0batch_normalization_56_2705928batch_normalization_56_2705930batch_normalization_56_2705932batch_normalization_56_2705934*
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
S__inference_batch_normalization_56_layer_call_and_return_conditional_losses_270588320
.batch_normalization_56/StatefulPartitionedCall?
"dropout_56/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_56/StatefulPartitionedCall:output:0#^dropout_55/StatefulPartitionedCall*
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
GPU 2J 8? *P
fKRI
G__inference_dropout_56_layer_call_and_return_conditional_losses_27059492$
"dropout_56/StatefulPartitionedCall?
"conv2d_150/StatefulPartitionedCallStatefulPartitionedCall+dropout_56/StatefulPartitionedCall:output:0conv2d_150_2705989conv2d_150_2705991*
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
G__inference_conv2d_150_layer_call_and_return_conditional_losses_27059782$
"conv2d_150/StatefulPartitionedCall?
.batch_normalization_57/StatefulPartitionedCallStatefulPartitionedCall+conv2d_150/StatefulPartitionedCall:output:0batch_normalization_57_2706058batch_normalization_57_2706060batch_normalization_57_2706062batch_normalization_57_2706064*
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
S__inference_batch_normalization_57_layer_call_and_return_conditional_losses_270601320
.batch_normalization_57/StatefulPartitionedCall?
"dropout_57/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_57/StatefulPartitionedCall:output:0#^dropout_56/StatefulPartitionedCall*
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
GPU 2J 8? *P
fKRI
G__inference_dropout_57_layer_call_and_return_conditional_losses_27060792$
"dropout_57/StatefulPartitionedCall?
 max_pooling2d_60/PartitionedCallPartitionedCall+dropout_57/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_60_layer_call_and_return_conditional_losses_27053052"
 max_pooling2d_60/PartitionedCall?
flatten_30/PartitionedCallPartitionedCall)max_pooling2d_60/PartitionedCall:output:0*
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
G__inference_flatten_30_layer_call_and_return_conditional_losses_27061042
flatten_30/PartitionedCall?
 dense_60/StatefulPartitionedCallStatefulPartitionedCall#flatten_30/PartitionedCall:output:0dense_60_2706134dense_60_2706136*
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
E__inference_dense_60_layer_call_and_return_conditional_losses_27061232"
 dense_60/StatefulPartitionedCall?
 dense_61/StatefulPartitionedCallStatefulPartitionedCall)dense_60/StatefulPartitionedCall:output:0dense_61_2706161dense_61_2706163*
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
E__inference_dense_61_layer_call_and_return_conditional_losses_27061502"
 dense_61/StatefulPartitionedCall?
IdentityIdentity)dense_61/StatefulPartitionedCall:output:0/^batch_normalization_52/StatefulPartitionedCall/^batch_normalization_53/StatefulPartitionedCall/^batch_normalization_54/StatefulPartitionedCall/^batch_normalization_55/StatefulPartitionedCall/^batch_normalization_56/StatefulPartitionedCall/^batch_normalization_57/StatefulPartitionedCall#^conv2d_145/StatefulPartitionedCall#^conv2d_146/StatefulPartitionedCall#^conv2d_147/StatefulPartitionedCall#^conv2d_148/StatefulPartitionedCall#^conv2d_149/StatefulPartitionedCall#^conv2d_150/StatefulPartitionedCall!^dense_60/StatefulPartitionedCall!^dense_61/StatefulPartitionedCall#^dropout_52/StatefulPartitionedCall#^dropout_53/StatefulPartitionedCall#^dropout_54/StatefulPartitionedCall#^dropout_55/StatefulPartitionedCall#^dropout_56/StatefulPartitionedCall#^dropout_57/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_52/StatefulPartitionedCall.batch_normalization_52/StatefulPartitionedCall2`
.batch_normalization_53/StatefulPartitionedCall.batch_normalization_53/StatefulPartitionedCall2`
.batch_normalization_54/StatefulPartitionedCall.batch_normalization_54/StatefulPartitionedCall2`
.batch_normalization_55/StatefulPartitionedCall.batch_normalization_55/StatefulPartitionedCall2`
.batch_normalization_56/StatefulPartitionedCall.batch_normalization_56/StatefulPartitionedCall2`
.batch_normalization_57/StatefulPartitionedCall.batch_normalization_57/StatefulPartitionedCall2H
"conv2d_145/StatefulPartitionedCall"conv2d_145/StatefulPartitionedCall2H
"conv2d_146/StatefulPartitionedCall"conv2d_146/StatefulPartitionedCall2H
"conv2d_147/StatefulPartitionedCall"conv2d_147/StatefulPartitionedCall2H
"conv2d_148/StatefulPartitionedCall"conv2d_148/StatefulPartitionedCall2H
"conv2d_149/StatefulPartitionedCall"conv2d_149/StatefulPartitionedCall2H
"conv2d_150/StatefulPartitionedCall"conv2d_150/StatefulPartitionedCall2D
 dense_60/StatefulPartitionedCall dense_60/StatefulPartitionedCall2D
 dense_61/StatefulPartitionedCall dense_61/StatefulPartitionedCall2H
"dropout_52/StatefulPartitionedCall"dropout_52/StatefulPartitionedCall2H
"dropout_53/StatefulPartitionedCall"dropout_53/StatefulPartitionedCall2H
"dropout_54/StatefulPartitionedCall"dropout_54/StatefulPartitionedCall2H
"dropout_55/StatefulPartitionedCall"dropout_55/StatefulPartitionedCall2H
"dropout_56/StatefulPartitionedCall"dropout_56/StatefulPartitionedCall2H
"dropout_57/StatefulPartitionedCall"dropout_57/StatefulPartitionedCall:[ W
1
_output_shapes
:???????????
"
_user_specified_name
input_31
?
?
S__inference_batch_normalization_52_layer_call_and_return_conditional_losses_2707331

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
S__inference_batch_normalization_56_layer_call_and_return_conditional_losses_2705153

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
c
G__inference_flatten_30_layer_call_and_return_conditional_losses_2706104

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
?

?
G__inference_conv2d_150_layer_call_and_return_conditional_losses_2705978

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
?

*__inference_dense_61_layer_call_fn_2708392

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
E__inference_dense_61_layer_call_and_return_conditional_losses_27061502
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
??
?
T__inference_CNN_aug_deep_bnorm_drop_layer_call_and_return_conditional_losses_2706386

inputs
conv2d_145_2706281
conv2d_145_2706283"
batch_normalization_52_2706286"
batch_normalization_52_2706288"
batch_normalization_52_2706290"
batch_normalization_52_2706292
conv2d_146_2706296
conv2d_146_2706298"
batch_normalization_53_2706301"
batch_normalization_53_2706303"
batch_normalization_53_2706305"
batch_normalization_53_2706307
conv2d_147_2706312
conv2d_147_2706314"
batch_normalization_54_2706317"
batch_normalization_54_2706319"
batch_normalization_54_2706321"
batch_normalization_54_2706323
conv2d_148_2706327
conv2d_148_2706329"
batch_normalization_55_2706332"
batch_normalization_55_2706334"
batch_normalization_55_2706336"
batch_normalization_55_2706338
conv2d_149_2706343
conv2d_149_2706345"
batch_normalization_56_2706348"
batch_normalization_56_2706350"
batch_normalization_56_2706352"
batch_normalization_56_2706354
conv2d_150_2706358
conv2d_150_2706360"
batch_normalization_57_2706363"
batch_normalization_57_2706365"
batch_normalization_57_2706367"
batch_normalization_57_2706369
dense_60_2706375
dense_60_2706377
dense_61_2706380
dense_61_2706382
identity??.batch_normalization_52/StatefulPartitionedCall?.batch_normalization_53/StatefulPartitionedCall?.batch_normalization_54/StatefulPartitionedCall?.batch_normalization_55/StatefulPartitionedCall?.batch_normalization_56/StatefulPartitionedCall?.batch_normalization_57/StatefulPartitionedCall?"conv2d_145/StatefulPartitionedCall?"conv2d_146/StatefulPartitionedCall?"conv2d_147/StatefulPartitionedCall?"conv2d_148/StatefulPartitionedCall?"conv2d_149/StatefulPartitionedCall?"conv2d_150/StatefulPartitionedCall? dense_60/StatefulPartitionedCall? dense_61/StatefulPartitionedCall?"dropout_52/StatefulPartitionedCall?"dropout_53/StatefulPartitionedCall?"dropout_54/StatefulPartitionedCall?"dropout_55/StatefulPartitionedCall?"dropout_56/StatefulPartitionedCall?"dropout_57/StatefulPartitionedCall?
"conv2d_145/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_145_2706281conv2d_145_2706283*
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
G__inference_conv2d_145_layer_call_and_return_conditional_losses_27053262$
"conv2d_145/StatefulPartitionedCall?
.batch_normalization_52/StatefulPartitionedCallStatefulPartitionedCall+conv2d_145/StatefulPartitionedCall:output:0batch_normalization_52_2706286batch_normalization_52_2706288batch_normalization_52_2706290batch_normalization_52_2706292*
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
S__inference_batch_normalization_52_layer_call_and_return_conditional_losses_270536120
.batch_normalization_52/StatefulPartitionedCall?
"dropout_52/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_52/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_52_layer_call_and_return_conditional_losses_27054272$
"dropout_52/StatefulPartitionedCall?
"conv2d_146/StatefulPartitionedCallStatefulPartitionedCall+dropout_52/StatefulPartitionedCall:output:0conv2d_146_2706296conv2d_146_2706298*
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
G__inference_conv2d_146_layer_call_and_return_conditional_losses_27054562$
"conv2d_146/StatefulPartitionedCall?
.batch_normalization_53/StatefulPartitionedCallStatefulPartitionedCall+conv2d_146/StatefulPartitionedCall:output:0batch_normalization_53_2706301batch_normalization_53_2706303batch_normalization_53_2706305batch_normalization_53_2706307*
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
S__inference_batch_normalization_53_layer_call_and_return_conditional_losses_270549120
.batch_normalization_53/StatefulPartitionedCall?
"dropout_53/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_53/StatefulPartitionedCall:output:0#^dropout_52/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_53_layer_call_and_return_conditional_losses_27055572$
"dropout_53/StatefulPartitionedCall?
 max_pooling2d_58/PartitionedCallPartitionedCall+dropout_53/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_58_layer_call_and_return_conditional_losses_27048652"
 max_pooling2d_58/PartitionedCall?
"conv2d_147/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_58/PartitionedCall:output:0conv2d_147_2706312conv2d_147_2706314*
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
G__inference_conv2d_147_layer_call_and_return_conditional_losses_27055872$
"conv2d_147/StatefulPartitionedCall?
.batch_normalization_54/StatefulPartitionedCallStatefulPartitionedCall+conv2d_147/StatefulPartitionedCall:output:0batch_normalization_54_2706317batch_normalization_54_2706319batch_normalization_54_2706321batch_normalization_54_2706323*
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
S__inference_batch_normalization_54_layer_call_and_return_conditional_losses_270562220
.batch_normalization_54/StatefulPartitionedCall?
"dropout_54/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_54/StatefulPartitionedCall:output:0#^dropout_53/StatefulPartitionedCall*
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
GPU 2J 8? *P
fKRI
G__inference_dropout_54_layer_call_and_return_conditional_losses_27056882$
"dropout_54/StatefulPartitionedCall?
"conv2d_148/StatefulPartitionedCallStatefulPartitionedCall+dropout_54/StatefulPartitionedCall:output:0conv2d_148_2706327conv2d_148_2706329*
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
G__inference_conv2d_148_layer_call_and_return_conditional_losses_27057172$
"conv2d_148/StatefulPartitionedCall?
.batch_normalization_55/StatefulPartitionedCallStatefulPartitionedCall+conv2d_148/StatefulPartitionedCall:output:0batch_normalization_55_2706332batch_normalization_55_2706334batch_normalization_55_2706336batch_normalization_55_2706338*
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
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_270575220
.batch_normalization_55/StatefulPartitionedCall?
"dropout_55/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_55/StatefulPartitionedCall:output:0#^dropout_54/StatefulPartitionedCall*
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
GPU 2J 8? *P
fKRI
G__inference_dropout_55_layer_call_and_return_conditional_losses_27058182$
"dropout_55/StatefulPartitionedCall?
 max_pooling2d_59/PartitionedCallPartitionedCall+dropout_55/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_59_layer_call_and_return_conditional_losses_27050852"
 max_pooling2d_59/PartitionedCall?
"conv2d_149/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_59/PartitionedCall:output:0conv2d_149_2706343conv2d_149_2706345*
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
G__inference_conv2d_149_layer_call_and_return_conditional_losses_27058482$
"conv2d_149/StatefulPartitionedCall?
.batch_normalization_56/StatefulPartitionedCallStatefulPartitionedCall+conv2d_149/StatefulPartitionedCall:output:0batch_normalization_56_2706348batch_normalization_56_2706350batch_normalization_56_2706352batch_normalization_56_2706354*
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
S__inference_batch_normalization_56_layer_call_and_return_conditional_losses_270588320
.batch_normalization_56/StatefulPartitionedCall?
"dropout_56/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_56/StatefulPartitionedCall:output:0#^dropout_55/StatefulPartitionedCall*
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
GPU 2J 8? *P
fKRI
G__inference_dropout_56_layer_call_and_return_conditional_losses_27059492$
"dropout_56/StatefulPartitionedCall?
"conv2d_150/StatefulPartitionedCallStatefulPartitionedCall+dropout_56/StatefulPartitionedCall:output:0conv2d_150_2706358conv2d_150_2706360*
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
G__inference_conv2d_150_layer_call_and_return_conditional_losses_27059782$
"conv2d_150/StatefulPartitionedCall?
.batch_normalization_57/StatefulPartitionedCallStatefulPartitionedCall+conv2d_150/StatefulPartitionedCall:output:0batch_normalization_57_2706363batch_normalization_57_2706365batch_normalization_57_2706367batch_normalization_57_2706369*
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
S__inference_batch_normalization_57_layer_call_and_return_conditional_losses_270601320
.batch_normalization_57/StatefulPartitionedCall?
"dropout_57/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_57/StatefulPartitionedCall:output:0#^dropout_56/StatefulPartitionedCall*
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
GPU 2J 8? *P
fKRI
G__inference_dropout_57_layer_call_and_return_conditional_losses_27060792$
"dropout_57/StatefulPartitionedCall?
 max_pooling2d_60/PartitionedCallPartitionedCall+dropout_57/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_60_layer_call_and_return_conditional_losses_27053052"
 max_pooling2d_60/PartitionedCall?
flatten_30/PartitionedCallPartitionedCall)max_pooling2d_60/PartitionedCall:output:0*
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
G__inference_flatten_30_layer_call_and_return_conditional_losses_27061042
flatten_30/PartitionedCall?
 dense_60/StatefulPartitionedCallStatefulPartitionedCall#flatten_30/PartitionedCall:output:0dense_60_2706375dense_60_2706377*
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
E__inference_dense_60_layer_call_and_return_conditional_losses_27061232"
 dense_60/StatefulPartitionedCall?
 dense_61/StatefulPartitionedCallStatefulPartitionedCall)dense_60/StatefulPartitionedCall:output:0dense_61_2706380dense_61_2706382*
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
E__inference_dense_61_layer_call_and_return_conditional_losses_27061502"
 dense_61/StatefulPartitionedCall?
IdentityIdentity)dense_61/StatefulPartitionedCall:output:0/^batch_normalization_52/StatefulPartitionedCall/^batch_normalization_53/StatefulPartitionedCall/^batch_normalization_54/StatefulPartitionedCall/^batch_normalization_55/StatefulPartitionedCall/^batch_normalization_56/StatefulPartitionedCall/^batch_normalization_57/StatefulPartitionedCall#^conv2d_145/StatefulPartitionedCall#^conv2d_146/StatefulPartitionedCall#^conv2d_147/StatefulPartitionedCall#^conv2d_148/StatefulPartitionedCall#^conv2d_149/StatefulPartitionedCall#^conv2d_150/StatefulPartitionedCall!^dense_60/StatefulPartitionedCall!^dense_61/StatefulPartitionedCall#^dropout_52/StatefulPartitionedCall#^dropout_53/StatefulPartitionedCall#^dropout_54/StatefulPartitionedCall#^dropout_55/StatefulPartitionedCall#^dropout_56/StatefulPartitionedCall#^dropout_57/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_52/StatefulPartitionedCall.batch_normalization_52/StatefulPartitionedCall2`
.batch_normalization_53/StatefulPartitionedCall.batch_normalization_53/StatefulPartitionedCall2`
.batch_normalization_54/StatefulPartitionedCall.batch_normalization_54/StatefulPartitionedCall2`
.batch_normalization_55/StatefulPartitionedCall.batch_normalization_55/StatefulPartitionedCall2`
.batch_normalization_56/StatefulPartitionedCall.batch_normalization_56/StatefulPartitionedCall2`
.batch_normalization_57/StatefulPartitionedCall.batch_normalization_57/StatefulPartitionedCall2H
"conv2d_145/StatefulPartitionedCall"conv2d_145/StatefulPartitionedCall2H
"conv2d_146/StatefulPartitionedCall"conv2d_146/StatefulPartitionedCall2H
"conv2d_147/StatefulPartitionedCall"conv2d_147/StatefulPartitionedCall2H
"conv2d_148/StatefulPartitionedCall"conv2d_148/StatefulPartitionedCall2H
"conv2d_149/StatefulPartitionedCall"conv2d_149/StatefulPartitionedCall2H
"conv2d_150/StatefulPartitionedCall"conv2d_150/StatefulPartitionedCall2D
 dense_60/StatefulPartitionedCall dense_60/StatefulPartitionedCall2D
 dense_61/StatefulPartitionedCall dense_61/StatefulPartitionedCall2H
"dropout_52/StatefulPartitionedCall"dropout_52/StatefulPartitionedCall2H
"dropout_53/StatefulPartitionedCall"dropout_53/StatefulPartitionedCall2H
"dropout_54/StatefulPartitionedCall"dropout_54/StatefulPartitionedCall2H
"dropout_55/StatefulPartitionedCall"dropout_55/StatefulPartitionedCall2H
"dropout_56/StatefulPartitionedCall"dropout_56/StatefulPartitionedCall2H
"dropout_57/StatefulPartitionedCall"dropout_57/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
e
G__inference_dropout_53_layer_call_and_return_conditional_losses_2705562

inputs

identity_1d
IdentityIdentityinputs*
T0*1
_output_shapes
:???????????2

Identitys

Identity_1IdentityIdentity:output:0*
T0*1
_output_shapes
:???????????2

Identity_1"!

identity_1Identity_1:output:0*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
f
G__inference_dropout_55_layer_call_and_return_conditional_losses_2707976

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????**2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????***
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????**2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????**2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????**2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????**2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????**:W S
/
_output_shapes
:?????????**
 
_user_specified_nameinputs
?

?
G__inference_conv2d_147_layer_call_and_return_conditional_losses_2705587

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
??
?0
 __inference__traced_save_2708730
file_prefix0
,savev2_conv2d_145_kernel_read_readvariableop.
*savev2_conv2d_145_bias_read_readvariableop;
7savev2_batch_normalization_52_gamma_read_readvariableop:
6savev2_batch_normalization_52_beta_read_readvariableopA
=savev2_batch_normalization_52_moving_mean_read_readvariableopE
Asavev2_batch_normalization_52_moving_variance_read_readvariableop0
,savev2_conv2d_146_kernel_read_readvariableop.
*savev2_conv2d_146_bias_read_readvariableop;
7savev2_batch_normalization_53_gamma_read_readvariableop:
6savev2_batch_normalization_53_beta_read_readvariableopA
=savev2_batch_normalization_53_moving_mean_read_readvariableopE
Asavev2_batch_normalization_53_moving_variance_read_readvariableop0
,savev2_conv2d_147_kernel_read_readvariableop.
*savev2_conv2d_147_bias_read_readvariableop;
7savev2_batch_normalization_54_gamma_read_readvariableop:
6savev2_batch_normalization_54_beta_read_readvariableopA
=savev2_batch_normalization_54_moving_mean_read_readvariableopE
Asavev2_batch_normalization_54_moving_variance_read_readvariableop0
,savev2_conv2d_148_kernel_read_readvariableop.
*savev2_conv2d_148_bias_read_readvariableop;
7savev2_batch_normalization_55_gamma_read_readvariableop:
6savev2_batch_normalization_55_beta_read_readvariableopA
=savev2_batch_normalization_55_moving_mean_read_readvariableopE
Asavev2_batch_normalization_55_moving_variance_read_readvariableop0
,savev2_conv2d_149_kernel_read_readvariableop.
*savev2_conv2d_149_bias_read_readvariableop;
7savev2_batch_normalization_56_gamma_read_readvariableop:
6savev2_batch_normalization_56_beta_read_readvariableopA
=savev2_batch_normalization_56_moving_mean_read_readvariableopE
Asavev2_batch_normalization_56_moving_variance_read_readvariableop0
,savev2_conv2d_150_kernel_read_readvariableop.
*savev2_conv2d_150_bias_read_readvariableop;
7savev2_batch_normalization_57_gamma_read_readvariableop:
6savev2_batch_normalization_57_beta_read_readvariableopA
=savev2_batch_normalization_57_moving_mean_read_readvariableopE
Asavev2_batch_normalization_57_moving_variance_read_readvariableop.
*savev2_dense_60_kernel_read_readvariableop,
(savev2_dense_60_bias_read_readvariableop.
*savev2_dense_61_kernel_read_readvariableop,
(savev2_dense_61_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop7
3savev2_adam_conv2d_145_kernel_m_read_readvariableop5
1savev2_adam_conv2d_145_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_52_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_52_beta_m_read_readvariableop7
3savev2_adam_conv2d_146_kernel_m_read_readvariableop5
1savev2_adam_conv2d_146_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_53_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_53_beta_m_read_readvariableop7
3savev2_adam_conv2d_147_kernel_m_read_readvariableop5
1savev2_adam_conv2d_147_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_54_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_54_beta_m_read_readvariableop7
3savev2_adam_conv2d_148_kernel_m_read_readvariableop5
1savev2_adam_conv2d_148_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_55_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_55_beta_m_read_readvariableop7
3savev2_adam_conv2d_149_kernel_m_read_readvariableop5
1savev2_adam_conv2d_149_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_56_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_56_beta_m_read_readvariableop7
3savev2_adam_conv2d_150_kernel_m_read_readvariableop5
1savev2_adam_conv2d_150_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_57_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_57_beta_m_read_readvariableop5
1savev2_adam_dense_60_kernel_m_read_readvariableop3
/savev2_adam_dense_60_bias_m_read_readvariableop5
1savev2_adam_dense_61_kernel_m_read_readvariableop3
/savev2_adam_dense_61_bias_m_read_readvariableop7
3savev2_adam_conv2d_145_kernel_v_read_readvariableop5
1savev2_adam_conv2d_145_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_52_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_52_beta_v_read_readvariableop7
3savev2_adam_conv2d_146_kernel_v_read_readvariableop5
1savev2_adam_conv2d_146_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_53_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_53_beta_v_read_readvariableop7
3savev2_adam_conv2d_147_kernel_v_read_readvariableop5
1savev2_adam_conv2d_147_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_54_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_54_beta_v_read_readvariableop7
3savev2_adam_conv2d_148_kernel_v_read_readvariableop5
1savev2_adam_conv2d_148_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_55_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_55_beta_v_read_readvariableop7
3savev2_adam_conv2d_149_kernel_v_read_readvariableop5
1savev2_adam_conv2d_149_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_56_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_56_beta_v_read_readvariableop7
3savev2_adam_conv2d_150_kernel_v_read_readvariableop5
1savev2_adam_conv2d_150_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_57_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_57_beta_v_read_readvariableop5
1savev2_adam_dense_60_kernel_v_read_readvariableop3
/savev2_adam_dense_60_bias_v_read_readvariableop5
1savev2_adam_dense_61_kernel_v_read_readvariableop3
/savev2_adam_dense_61_bias_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_conv2d_145_kernel_read_readvariableop*savev2_conv2d_145_bias_read_readvariableop7savev2_batch_normalization_52_gamma_read_readvariableop6savev2_batch_normalization_52_beta_read_readvariableop=savev2_batch_normalization_52_moving_mean_read_readvariableopAsavev2_batch_normalization_52_moving_variance_read_readvariableop,savev2_conv2d_146_kernel_read_readvariableop*savev2_conv2d_146_bias_read_readvariableop7savev2_batch_normalization_53_gamma_read_readvariableop6savev2_batch_normalization_53_beta_read_readvariableop=savev2_batch_normalization_53_moving_mean_read_readvariableopAsavev2_batch_normalization_53_moving_variance_read_readvariableop,savev2_conv2d_147_kernel_read_readvariableop*savev2_conv2d_147_bias_read_readvariableop7savev2_batch_normalization_54_gamma_read_readvariableop6savev2_batch_normalization_54_beta_read_readvariableop=savev2_batch_normalization_54_moving_mean_read_readvariableopAsavev2_batch_normalization_54_moving_variance_read_readvariableop,savev2_conv2d_148_kernel_read_readvariableop*savev2_conv2d_148_bias_read_readvariableop7savev2_batch_normalization_55_gamma_read_readvariableop6savev2_batch_normalization_55_beta_read_readvariableop=savev2_batch_normalization_55_moving_mean_read_readvariableopAsavev2_batch_normalization_55_moving_variance_read_readvariableop,savev2_conv2d_149_kernel_read_readvariableop*savev2_conv2d_149_bias_read_readvariableop7savev2_batch_normalization_56_gamma_read_readvariableop6savev2_batch_normalization_56_beta_read_readvariableop=savev2_batch_normalization_56_moving_mean_read_readvariableopAsavev2_batch_normalization_56_moving_variance_read_readvariableop,savev2_conv2d_150_kernel_read_readvariableop*savev2_conv2d_150_bias_read_readvariableop7savev2_batch_normalization_57_gamma_read_readvariableop6savev2_batch_normalization_57_beta_read_readvariableop=savev2_batch_normalization_57_moving_mean_read_readvariableopAsavev2_batch_normalization_57_moving_variance_read_readvariableop*savev2_dense_60_kernel_read_readvariableop(savev2_dense_60_bias_read_readvariableop*savev2_dense_61_kernel_read_readvariableop(savev2_dense_61_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop3savev2_adam_conv2d_145_kernel_m_read_readvariableop1savev2_adam_conv2d_145_bias_m_read_readvariableop>savev2_adam_batch_normalization_52_gamma_m_read_readvariableop=savev2_adam_batch_normalization_52_beta_m_read_readvariableop3savev2_adam_conv2d_146_kernel_m_read_readvariableop1savev2_adam_conv2d_146_bias_m_read_readvariableop>savev2_adam_batch_normalization_53_gamma_m_read_readvariableop=savev2_adam_batch_normalization_53_beta_m_read_readvariableop3savev2_adam_conv2d_147_kernel_m_read_readvariableop1savev2_adam_conv2d_147_bias_m_read_readvariableop>savev2_adam_batch_normalization_54_gamma_m_read_readvariableop=savev2_adam_batch_normalization_54_beta_m_read_readvariableop3savev2_adam_conv2d_148_kernel_m_read_readvariableop1savev2_adam_conv2d_148_bias_m_read_readvariableop>savev2_adam_batch_normalization_55_gamma_m_read_readvariableop=savev2_adam_batch_normalization_55_beta_m_read_readvariableop3savev2_adam_conv2d_149_kernel_m_read_readvariableop1savev2_adam_conv2d_149_bias_m_read_readvariableop>savev2_adam_batch_normalization_56_gamma_m_read_readvariableop=savev2_adam_batch_normalization_56_beta_m_read_readvariableop3savev2_adam_conv2d_150_kernel_m_read_readvariableop1savev2_adam_conv2d_150_bias_m_read_readvariableop>savev2_adam_batch_normalization_57_gamma_m_read_readvariableop=savev2_adam_batch_normalization_57_beta_m_read_readvariableop1savev2_adam_dense_60_kernel_m_read_readvariableop/savev2_adam_dense_60_bias_m_read_readvariableop1savev2_adam_dense_61_kernel_m_read_readvariableop/savev2_adam_dense_61_bias_m_read_readvariableop3savev2_adam_conv2d_145_kernel_v_read_readvariableop1savev2_adam_conv2d_145_bias_v_read_readvariableop>savev2_adam_batch_normalization_52_gamma_v_read_readvariableop=savev2_adam_batch_normalization_52_beta_v_read_readvariableop3savev2_adam_conv2d_146_kernel_v_read_readvariableop1savev2_adam_conv2d_146_bias_v_read_readvariableop>savev2_adam_batch_normalization_53_gamma_v_read_readvariableop=savev2_adam_batch_normalization_53_beta_v_read_readvariableop3savev2_adam_conv2d_147_kernel_v_read_readvariableop1savev2_adam_conv2d_147_bias_v_read_readvariableop>savev2_adam_batch_normalization_54_gamma_v_read_readvariableop=savev2_adam_batch_normalization_54_beta_v_read_readvariableop3savev2_adam_conv2d_148_kernel_v_read_readvariableop1savev2_adam_conv2d_148_bias_v_read_readvariableop>savev2_adam_batch_normalization_55_gamma_v_read_readvariableop=savev2_adam_batch_normalization_55_beta_v_read_readvariableop3savev2_adam_conv2d_149_kernel_v_read_readvariableop1savev2_adam_conv2d_149_bias_v_read_readvariableop>savev2_adam_batch_normalization_56_gamma_v_read_readvariableop=savev2_adam_batch_normalization_56_beta_v_read_readvariableop3savev2_adam_conv2d_150_kernel_v_read_readvariableop1savev2_adam_conv2d_150_bias_v_read_readvariableop>savev2_adam_batch_normalization_57_gamma_v_read_readvariableop=savev2_adam_batch_normalization_57_beta_v_read_readvariableop1savev2_adam_dense_60_kernel_v_read_readvariableop/savev2_adam_dense_60_bias_v_read_readvariableop1savev2_adam_dense_61_kernel_v_read_readvariableop/savev2_adam_dense_61_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
?
?
S__inference_batch_normalization_54_layer_call_and_return_conditional_losses_2705640

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
S__inference_batch_normalization_53_layer_call_and_return_conditional_losses_2707588

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
?
?
8__inference_batch_normalization_57_layer_call_fn_2708237

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
S__inference_batch_normalization_57_layer_call_and_return_conditional_losses_27052572
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
G__inference_conv2d_150_layer_call_and_return_conditional_losses_2708177

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
f
G__inference_dropout_56_layer_call_and_return_conditional_losses_2708151

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_56_layer_call_fn_2708062

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
S__inference_batch_normalization_56_layer_call_and_return_conditional_losses_27051532
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
e
,__inference_dropout_54_layer_call_fn_2707811

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
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
GPU 2J 8? *P
fKRI
G__inference_dropout_54_layer_call_and_return_conditional_losses_27056882
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????**2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????**22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????**
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_52_layer_call_and_return_conditional_losses_2707413

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
f
G__inference_dropout_53_layer_call_and_return_conditional_losses_2707626

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout/Const}
dropout/MulMulinputsdropout/Const:output:0*
T0*1
_output_shapes
:???????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*1
_output_shapes
:???????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:???????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:???????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*1
_output_shapes
:???????????2
dropout/Mul_1o
IdentityIdentitydropout/Mul_1:z:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
f
G__inference_dropout_54_layer_call_and_return_conditional_losses_2707801

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????**2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????***
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????**2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????**2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????**2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????**2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????**:W S
/
_output_shapes
:?????????**
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_52_layer_call_and_return_conditional_losses_2705379

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
M__inference_max_pooling2d_60_layer_call_and_return_conditional_losses_2705305

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
?
?
S__inference_batch_normalization_57_layer_call_and_return_conditional_losses_2708288

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
G__inference_conv2d_145_layer_call_and_return_conditional_losses_2707302

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
?
?
,__inference_conv2d_149_layer_call_fn_2708011

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
G__inference_conv2d_149_layer_call_and_return_conditional_losses_27058482
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
?
?
S__inference_batch_normalization_57_layer_call_and_return_conditional_losses_2708224

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
8__inference_batch_normalization_52_layer_call_fn_2707362

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
S__inference_batch_normalization_52_layer_call_and_return_conditional_losses_27047132
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
G__inference_conv2d_147_layer_call_and_return_conditional_losses_2707652

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
?
e
G__inference_dropout_55_layer_call_and_return_conditional_losses_2707981

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????**2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????**2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????**:W S
/
_output_shapes
:?????????**
 
_user_specified_nameinputs
?
H
,__inference_dropout_52_layer_call_fn_2707466

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_52_layer_call_and_return_conditional_losses_27054322
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_54_layer_call_and_return_conditional_losses_2705622

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
?
?
S__inference_batch_normalization_54_layer_call_and_return_conditional_losses_2707681

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
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_2705068

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
G__inference_conv2d_148_layer_call_and_return_conditional_losses_2705717

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
f
G__inference_dropout_56_layer_call_and_return_conditional_losses_2705949

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_52_layer_call_fn_2707426

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
S__inference_batch_normalization_52_layer_call_and_return_conditional_losses_27053612
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
?
?
9__inference_CNN_aug_deep_bnorm_drop_layer_call_fn_2706469
input_31
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
StatefulPartitionedCallStatefulPartitionedCallinput_31unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU 2J 8? *]
fXRV
T__inference_CNN_aug_deep_bnorm_drop_layer_call_and_return_conditional_losses_27063862
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
input_31
?
?
8__inference_batch_normalization_57_layer_call_fn_2708301

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
S__inference_batch_normalization_57_layer_call_and_return_conditional_losses_27060132
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
?
H
,__inference_dropout_55_layer_call_fn_2707991

inputs
identity?
PartitionedCallPartitionedCallinputs*
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
GPU 2J 8? *P
fKRI
G__inference_dropout_55_layer_call_and_return_conditional_losses_27058232
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????**2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????**:W S
/
_output_shapes
:?????????**
 
_user_specified_nameinputs
?x
?
T__inference_CNN_aug_deep_bnorm_drop_layer_call_and_return_conditional_losses_2706579

inputs
conv2d_145_2706474
conv2d_145_2706476"
batch_normalization_52_2706479"
batch_normalization_52_2706481"
batch_normalization_52_2706483"
batch_normalization_52_2706485
conv2d_146_2706489
conv2d_146_2706491"
batch_normalization_53_2706494"
batch_normalization_53_2706496"
batch_normalization_53_2706498"
batch_normalization_53_2706500
conv2d_147_2706505
conv2d_147_2706507"
batch_normalization_54_2706510"
batch_normalization_54_2706512"
batch_normalization_54_2706514"
batch_normalization_54_2706516
conv2d_148_2706520
conv2d_148_2706522"
batch_normalization_55_2706525"
batch_normalization_55_2706527"
batch_normalization_55_2706529"
batch_normalization_55_2706531
conv2d_149_2706536
conv2d_149_2706538"
batch_normalization_56_2706541"
batch_normalization_56_2706543"
batch_normalization_56_2706545"
batch_normalization_56_2706547
conv2d_150_2706551
conv2d_150_2706553"
batch_normalization_57_2706556"
batch_normalization_57_2706558"
batch_normalization_57_2706560"
batch_normalization_57_2706562
dense_60_2706568
dense_60_2706570
dense_61_2706573
dense_61_2706575
identity??.batch_normalization_52/StatefulPartitionedCall?.batch_normalization_53/StatefulPartitionedCall?.batch_normalization_54/StatefulPartitionedCall?.batch_normalization_55/StatefulPartitionedCall?.batch_normalization_56/StatefulPartitionedCall?.batch_normalization_57/StatefulPartitionedCall?"conv2d_145/StatefulPartitionedCall?"conv2d_146/StatefulPartitionedCall?"conv2d_147/StatefulPartitionedCall?"conv2d_148/StatefulPartitionedCall?"conv2d_149/StatefulPartitionedCall?"conv2d_150/StatefulPartitionedCall? dense_60/StatefulPartitionedCall? dense_61/StatefulPartitionedCall?
"conv2d_145/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_145_2706474conv2d_145_2706476*
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
G__inference_conv2d_145_layer_call_and_return_conditional_losses_27053262$
"conv2d_145/StatefulPartitionedCall?
.batch_normalization_52/StatefulPartitionedCallStatefulPartitionedCall+conv2d_145/StatefulPartitionedCall:output:0batch_normalization_52_2706479batch_normalization_52_2706481batch_normalization_52_2706483batch_normalization_52_2706485*
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
S__inference_batch_normalization_52_layer_call_and_return_conditional_losses_270537920
.batch_normalization_52/StatefulPartitionedCall?
dropout_52/PartitionedCallPartitionedCall7batch_normalization_52/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_52_layer_call_and_return_conditional_losses_27054322
dropout_52/PartitionedCall?
"conv2d_146/StatefulPartitionedCallStatefulPartitionedCall#dropout_52/PartitionedCall:output:0conv2d_146_2706489conv2d_146_2706491*
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
G__inference_conv2d_146_layer_call_and_return_conditional_losses_27054562$
"conv2d_146/StatefulPartitionedCall?
.batch_normalization_53/StatefulPartitionedCallStatefulPartitionedCall+conv2d_146/StatefulPartitionedCall:output:0batch_normalization_53_2706494batch_normalization_53_2706496batch_normalization_53_2706498batch_normalization_53_2706500*
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
S__inference_batch_normalization_53_layer_call_and_return_conditional_losses_270550920
.batch_normalization_53/StatefulPartitionedCall?
dropout_53/PartitionedCallPartitionedCall7batch_normalization_53/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_53_layer_call_and_return_conditional_losses_27055622
dropout_53/PartitionedCall?
 max_pooling2d_58/PartitionedCallPartitionedCall#dropout_53/PartitionedCall:output:0*
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
M__inference_max_pooling2d_58_layer_call_and_return_conditional_losses_27048652"
 max_pooling2d_58/PartitionedCall?
"conv2d_147/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_58/PartitionedCall:output:0conv2d_147_2706505conv2d_147_2706507*
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
G__inference_conv2d_147_layer_call_and_return_conditional_losses_27055872$
"conv2d_147/StatefulPartitionedCall?
.batch_normalization_54/StatefulPartitionedCallStatefulPartitionedCall+conv2d_147/StatefulPartitionedCall:output:0batch_normalization_54_2706510batch_normalization_54_2706512batch_normalization_54_2706514batch_normalization_54_2706516*
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
S__inference_batch_normalization_54_layer_call_and_return_conditional_losses_270564020
.batch_normalization_54/StatefulPartitionedCall?
dropout_54/PartitionedCallPartitionedCall7batch_normalization_54/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *P
fKRI
G__inference_dropout_54_layer_call_and_return_conditional_losses_27056932
dropout_54/PartitionedCall?
"conv2d_148/StatefulPartitionedCallStatefulPartitionedCall#dropout_54/PartitionedCall:output:0conv2d_148_2706520conv2d_148_2706522*
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
G__inference_conv2d_148_layer_call_and_return_conditional_losses_27057172$
"conv2d_148/StatefulPartitionedCall?
.batch_normalization_55/StatefulPartitionedCallStatefulPartitionedCall+conv2d_148/StatefulPartitionedCall:output:0batch_normalization_55_2706525batch_normalization_55_2706527batch_normalization_55_2706529batch_normalization_55_2706531*
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
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_270577020
.batch_normalization_55/StatefulPartitionedCall?
dropout_55/PartitionedCallPartitionedCall7batch_normalization_55/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *P
fKRI
G__inference_dropout_55_layer_call_and_return_conditional_losses_27058232
dropout_55/PartitionedCall?
 max_pooling2d_59/PartitionedCallPartitionedCall#dropout_55/PartitionedCall:output:0*
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
M__inference_max_pooling2d_59_layer_call_and_return_conditional_losses_27050852"
 max_pooling2d_59/PartitionedCall?
"conv2d_149/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_59/PartitionedCall:output:0conv2d_149_2706536conv2d_149_2706538*
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
G__inference_conv2d_149_layer_call_and_return_conditional_losses_27058482$
"conv2d_149/StatefulPartitionedCall?
.batch_normalization_56/StatefulPartitionedCallStatefulPartitionedCall+conv2d_149/StatefulPartitionedCall:output:0batch_normalization_56_2706541batch_normalization_56_2706543batch_normalization_56_2706545batch_normalization_56_2706547*
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
S__inference_batch_normalization_56_layer_call_and_return_conditional_losses_270590120
.batch_normalization_56/StatefulPartitionedCall?
dropout_56/PartitionedCallPartitionedCall7batch_normalization_56/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *P
fKRI
G__inference_dropout_56_layer_call_and_return_conditional_losses_27059542
dropout_56/PartitionedCall?
"conv2d_150/StatefulPartitionedCallStatefulPartitionedCall#dropout_56/PartitionedCall:output:0conv2d_150_2706551conv2d_150_2706553*
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
G__inference_conv2d_150_layer_call_and_return_conditional_losses_27059782$
"conv2d_150/StatefulPartitionedCall?
.batch_normalization_57/StatefulPartitionedCallStatefulPartitionedCall+conv2d_150/StatefulPartitionedCall:output:0batch_normalization_57_2706556batch_normalization_57_2706558batch_normalization_57_2706560batch_normalization_57_2706562*
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
S__inference_batch_normalization_57_layer_call_and_return_conditional_losses_270603120
.batch_normalization_57/StatefulPartitionedCall?
dropout_57/PartitionedCallPartitionedCall7batch_normalization_57/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *P
fKRI
G__inference_dropout_57_layer_call_and_return_conditional_losses_27060842
dropout_57/PartitionedCall?
 max_pooling2d_60/PartitionedCallPartitionedCall#dropout_57/PartitionedCall:output:0*
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
M__inference_max_pooling2d_60_layer_call_and_return_conditional_losses_27053052"
 max_pooling2d_60/PartitionedCall?
flatten_30/PartitionedCallPartitionedCall)max_pooling2d_60/PartitionedCall:output:0*
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
G__inference_flatten_30_layer_call_and_return_conditional_losses_27061042
flatten_30/PartitionedCall?
 dense_60/StatefulPartitionedCallStatefulPartitionedCall#flatten_30/PartitionedCall:output:0dense_60_2706568dense_60_2706570*
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
E__inference_dense_60_layer_call_and_return_conditional_losses_27061232"
 dense_60/StatefulPartitionedCall?
 dense_61/StatefulPartitionedCallStatefulPartitionedCall)dense_60/StatefulPartitionedCall:output:0dense_61_2706573dense_61_2706575*
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
E__inference_dense_61_layer_call_and_return_conditional_losses_27061502"
 dense_61/StatefulPartitionedCall?
IdentityIdentity)dense_61/StatefulPartitionedCall:output:0/^batch_normalization_52/StatefulPartitionedCall/^batch_normalization_53/StatefulPartitionedCall/^batch_normalization_54/StatefulPartitionedCall/^batch_normalization_55/StatefulPartitionedCall/^batch_normalization_56/StatefulPartitionedCall/^batch_normalization_57/StatefulPartitionedCall#^conv2d_145/StatefulPartitionedCall#^conv2d_146/StatefulPartitionedCall#^conv2d_147/StatefulPartitionedCall#^conv2d_148/StatefulPartitionedCall#^conv2d_149/StatefulPartitionedCall#^conv2d_150/StatefulPartitionedCall!^dense_60/StatefulPartitionedCall!^dense_61/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_52/StatefulPartitionedCall.batch_normalization_52/StatefulPartitionedCall2`
.batch_normalization_53/StatefulPartitionedCall.batch_normalization_53/StatefulPartitionedCall2`
.batch_normalization_54/StatefulPartitionedCall.batch_normalization_54/StatefulPartitionedCall2`
.batch_normalization_55/StatefulPartitionedCall.batch_normalization_55/StatefulPartitionedCall2`
.batch_normalization_56/StatefulPartitionedCall.batch_normalization_56/StatefulPartitionedCall2`
.batch_normalization_57/StatefulPartitionedCall.batch_normalization_57/StatefulPartitionedCall2H
"conv2d_145/StatefulPartitionedCall"conv2d_145/StatefulPartitionedCall2H
"conv2d_146/StatefulPartitionedCall"conv2d_146/StatefulPartitionedCall2H
"conv2d_147/StatefulPartitionedCall"conv2d_147/StatefulPartitionedCall2H
"conv2d_148/StatefulPartitionedCall"conv2d_148/StatefulPartitionedCall2H
"conv2d_149/StatefulPartitionedCall"conv2d_149/StatefulPartitionedCall2H
"conv2d_150/StatefulPartitionedCall"conv2d_150/StatefulPartitionedCall2D
 dense_60/StatefulPartitionedCall dense_60/StatefulPartitionedCall2D
 dense_61/StatefulPartitionedCall dense_61/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_52_layer_call_and_return_conditional_losses_2704713

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
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_2705037

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
f
G__inference_dropout_57_layer_call_and_return_conditional_losses_2706079

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_conv2d_150_layer_call_fn_2708186

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
G__inference_conv2d_150_layer_call_and_return_conditional_losses_27059782
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
?
?
S__inference_batch_normalization_56_layer_call_and_return_conditional_losses_2708113

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
?
?
8__inference_batch_normalization_57_layer_call_fn_2708314

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
S__inference_batch_normalization_57_layer_call_and_return_conditional_losses_27060312
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
2__inference_max_pooling2d_60_layer_call_fn_2705311

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
M__inference_max_pooling2d_60_layer_call_and_return_conditional_losses_27053052
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
??
?
T__inference_CNN_aug_deep_bnorm_drop_layer_call_and_return_conditional_losses_2707121

inputs-
)conv2d_145_conv2d_readvariableop_resource.
*conv2d_145_biasadd_readvariableop_resource2
.batch_normalization_52_readvariableop_resource4
0batch_normalization_52_readvariableop_1_resourceC
?batch_normalization_52_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_52_fusedbatchnormv3_readvariableop_1_resource-
)conv2d_146_conv2d_readvariableop_resource.
*conv2d_146_biasadd_readvariableop_resource2
.batch_normalization_53_readvariableop_resource4
0batch_normalization_53_readvariableop_1_resourceC
?batch_normalization_53_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_53_fusedbatchnormv3_readvariableop_1_resource-
)conv2d_147_conv2d_readvariableop_resource.
*conv2d_147_biasadd_readvariableop_resource2
.batch_normalization_54_readvariableop_resource4
0batch_normalization_54_readvariableop_1_resourceC
?batch_normalization_54_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_54_fusedbatchnormv3_readvariableop_1_resource-
)conv2d_148_conv2d_readvariableop_resource.
*conv2d_148_biasadd_readvariableop_resource2
.batch_normalization_55_readvariableop_resource4
0batch_normalization_55_readvariableop_1_resourceC
?batch_normalization_55_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_55_fusedbatchnormv3_readvariableop_1_resource-
)conv2d_149_conv2d_readvariableop_resource.
*conv2d_149_biasadd_readvariableop_resource2
.batch_normalization_56_readvariableop_resource4
0batch_normalization_56_readvariableop_1_resourceC
?batch_normalization_56_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_56_fusedbatchnormv3_readvariableop_1_resource-
)conv2d_150_conv2d_readvariableop_resource.
*conv2d_150_biasadd_readvariableop_resource2
.batch_normalization_57_readvariableop_resource4
0batch_normalization_57_readvariableop_1_resourceC
?batch_normalization_57_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_57_fusedbatchnormv3_readvariableop_1_resource+
'dense_60_matmul_readvariableop_resource,
(dense_60_biasadd_readvariableop_resource+
'dense_61_matmul_readvariableop_resource,
(dense_61_biasadd_readvariableop_resource
identity??6batch_normalization_52/FusedBatchNormV3/ReadVariableOp?8batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_52/ReadVariableOp?'batch_normalization_52/ReadVariableOp_1?6batch_normalization_53/FusedBatchNormV3/ReadVariableOp?8batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_53/ReadVariableOp?'batch_normalization_53/ReadVariableOp_1?6batch_normalization_54/FusedBatchNormV3/ReadVariableOp?8batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_54/ReadVariableOp?'batch_normalization_54/ReadVariableOp_1?6batch_normalization_55/FusedBatchNormV3/ReadVariableOp?8batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_55/ReadVariableOp?'batch_normalization_55/ReadVariableOp_1?6batch_normalization_56/FusedBatchNormV3/ReadVariableOp?8batch_normalization_56/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_56/ReadVariableOp?'batch_normalization_56/ReadVariableOp_1?6batch_normalization_57/FusedBatchNormV3/ReadVariableOp?8batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_57/ReadVariableOp?'batch_normalization_57/ReadVariableOp_1?!conv2d_145/BiasAdd/ReadVariableOp? conv2d_145/Conv2D/ReadVariableOp?!conv2d_146/BiasAdd/ReadVariableOp? conv2d_146/Conv2D/ReadVariableOp?!conv2d_147/BiasAdd/ReadVariableOp? conv2d_147/Conv2D/ReadVariableOp?!conv2d_148/BiasAdd/ReadVariableOp? conv2d_148/Conv2D/ReadVariableOp?!conv2d_149/BiasAdd/ReadVariableOp? conv2d_149/Conv2D/ReadVariableOp?!conv2d_150/BiasAdd/ReadVariableOp? conv2d_150/Conv2D/ReadVariableOp?dense_60/BiasAdd/ReadVariableOp?dense_60/MatMul/ReadVariableOp?dense_61/BiasAdd/ReadVariableOp?dense_61/MatMul/ReadVariableOp?
 conv2d_145/Conv2D/ReadVariableOpReadVariableOp)conv2d_145_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_145/Conv2D/ReadVariableOp?
conv2d_145/Conv2DConv2Dinputs(conv2d_145/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_145/Conv2D?
!conv2d_145/BiasAdd/ReadVariableOpReadVariableOp*conv2d_145_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_145/BiasAdd/ReadVariableOp?
conv2d_145/BiasAddBiasAddconv2d_145/Conv2D:output:0)conv2d_145/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_145/BiasAdd?
conv2d_145/ReluReluconv2d_145/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_145/Relu?
%batch_normalization_52/ReadVariableOpReadVariableOp.batch_normalization_52_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_52/ReadVariableOp?
'batch_normalization_52/ReadVariableOp_1ReadVariableOp0batch_normalization_52_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_52/ReadVariableOp_1?
6batch_normalization_52/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_52_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_52/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_52_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_52/FusedBatchNormV3FusedBatchNormV3conv2d_145/Relu:activations:0-batch_normalization_52/ReadVariableOp:value:0/batch_normalization_52/ReadVariableOp_1:value:0>batch_normalization_52/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2)
'batch_normalization_52/FusedBatchNormV3?
dropout_52/IdentityIdentity+batch_normalization_52/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
dropout_52/Identity?
 conv2d_146/Conv2D/ReadVariableOpReadVariableOp)conv2d_146_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_146/Conv2D/ReadVariableOp?
conv2d_146/Conv2DConv2Ddropout_52/Identity:output:0(conv2d_146/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_146/Conv2D?
!conv2d_146/BiasAdd/ReadVariableOpReadVariableOp*conv2d_146_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_146/BiasAdd/ReadVariableOp?
conv2d_146/BiasAddBiasAddconv2d_146/Conv2D:output:0)conv2d_146/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_146/BiasAdd?
conv2d_146/ReluReluconv2d_146/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_146/Relu?
%batch_normalization_53/ReadVariableOpReadVariableOp.batch_normalization_53_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_53/ReadVariableOp?
'batch_normalization_53/ReadVariableOp_1ReadVariableOp0batch_normalization_53_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_53/ReadVariableOp_1?
6batch_normalization_53/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_53_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_53/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_53_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_53/FusedBatchNormV3FusedBatchNormV3conv2d_146/Relu:activations:0-batch_normalization_53/ReadVariableOp:value:0/batch_normalization_53/ReadVariableOp_1:value:0>batch_normalization_53/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2)
'batch_normalization_53/FusedBatchNormV3?
dropout_53/IdentityIdentity+batch_normalization_53/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
dropout_53/Identity?
max_pooling2d_58/MaxPoolMaxPooldropout_53/Identity:output:0*/
_output_shapes
:?????????***
ksize
*
paddingVALID*
strides
2
max_pooling2d_58/MaxPool?
 conv2d_147/Conv2D/ReadVariableOpReadVariableOp)conv2d_147_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_147/Conv2D/ReadVariableOp?
conv2d_147/Conv2DConv2D!max_pooling2d_58/MaxPool:output:0(conv2d_147/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????***
paddingSAME*
strides
2
conv2d_147/Conv2D?
!conv2d_147/BiasAdd/ReadVariableOpReadVariableOp*conv2d_147_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_147/BiasAdd/ReadVariableOp?
conv2d_147/BiasAddBiasAddconv2d_147/Conv2D:output:0)conv2d_147/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**2
conv2d_147/BiasAdd?
conv2d_147/ReluReluconv2d_147/BiasAdd:output:0*
T0*/
_output_shapes
:?????????**2
conv2d_147/Relu?
%batch_normalization_54/ReadVariableOpReadVariableOp.batch_normalization_54_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_54/ReadVariableOp?
'batch_normalization_54/ReadVariableOp_1ReadVariableOp0batch_normalization_54_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_54/ReadVariableOp_1?
6batch_normalization_54/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_54_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_54/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_54_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_54/FusedBatchNormV3FusedBatchNormV3conv2d_147/Relu:activations:0-batch_normalization_54/ReadVariableOp:value:0/batch_normalization_54/ReadVariableOp_1:value:0>batch_normalization_54/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????**:::::*
epsilon%o?:*
is_training( 2)
'batch_normalization_54/FusedBatchNormV3?
dropout_54/IdentityIdentity+batch_normalization_54/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????**2
dropout_54/Identity?
 conv2d_148/Conv2D/ReadVariableOpReadVariableOp)conv2d_148_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_148/Conv2D/ReadVariableOp?
conv2d_148/Conv2DConv2Ddropout_54/Identity:output:0(conv2d_148/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????***
paddingSAME*
strides
2
conv2d_148/Conv2D?
!conv2d_148/BiasAdd/ReadVariableOpReadVariableOp*conv2d_148_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_148/BiasAdd/ReadVariableOp?
conv2d_148/BiasAddBiasAddconv2d_148/Conv2D:output:0)conv2d_148/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**2
conv2d_148/BiasAdd?
conv2d_148/ReluReluconv2d_148/BiasAdd:output:0*
T0*/
_output_shapes
:?????????**2
conv2d_148/Relu?
%batch_normalization_55/ReadVariableOpReadVariableOp.batch_normalization_55_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_55/ReadVariableOp?
'batch_normalization_55/ReadVariableOp_1ReadVariableOp0batch_normalization_55_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_55/ReadVariableOp_1?
6batch_normalization_55/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_55_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_55/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_55_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_55/FusedBatchNormV3FusedBatchNormV3conv2d_148/Relu:activations:0-batch_normalization_55/ReadVariableOp:value:0/batch_normalization_55/ReadVariableOp_1:value:0>batch_normalization_55/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????**:::::*
epsilon%o?:*
is_training( 2)
'batch_normalization_55/FusedBatchNormV3?
dropout_55/IdentityIdentity+batch_normalization_55/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????**2
dropout_55/Identity?
max_pooling2d_59/MaxPoolMaxPooldropout_55/Identity:output:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_59/MaxPool?
 conv2d_149/Conv2D/ReadVariableOpReadVariableOp)conv2d_149_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_149/Conv2D/ReadVariableOp?
conv2d_149/Conv2DConv2D!max_pooling2d_59/MaxPool:output:0(conv2d_149/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_149/Conv2D?
!conv2d_149/BiasAdd/ReadVariableOpReadVariableOp*conv2d_149_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_149/BiasAdd/ReadVariableOp?
conv2d_149/BiasAddBiasAddconv2d_149/Conv2D:output:0)conv2d_149/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_149/BiasAdd?
conv2d_149/ReluReluconv2d_149/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_149/Relu?
%batch_normalization_56/ReadVariableOpReadVariableOp.batch_normalization_56_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_56/ReadVariableOp?
'batch_normalization_56/ReadVariableOp_1ReadVariableOp0batch_normalization_56_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_56/ReadVariableOp_1?
6batch_normalization_56/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_56_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_56/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_56/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_56_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_56/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_56/FusedBatchNormV3FusedBatchNormV3conv2d_149/Relu:activations:0-batch_normalization_56/ReadVariableOp:value:0/batch_normalization_56/ReadVariableOp_1:value:0>batch_normalization_56/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_56/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( 2)
'batch_normalization_56/FusedBatchNormV3?
dropout_56/IdentityIdentity+batch_normalization_56/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????2
dropout_56/Identity?
 conv2d_150/Conv2D/ReadVariableOpReadVariableOp)conv2d_150_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_150/Conv2D/ReadVariableOp?
conv2d_150/Conv2DConv2Ddropout_56/Identity:output:0(conv2d_150/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_150/Conv2D?
!conv2d_150/BiasAdd/ReadVariableOpReadVariableOp*conv2d_150_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_150/BiasAdd/ReadVariableOp?
conv2d_150/BiasAddBiasAddconv2d_150/Conv2D:output:0)conv2d_150/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_150/BiasAdd?
conv2d_150/ReluReluconv2d_150/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_150/Relu?
%batch_normalization_57/ReadVariableOpReadVariableOp.batch_normalization_57_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_57/ReadVariableOp?
'batch_normalization_57/ReadVariableOp_1ReadVariableOp0batch_normalization_57_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_57/ReadVariableOp_1?
6batch_normalization_57/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_57_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_57/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_57_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_57/FusedBatchNormV3FusedBatchNormV3conv2d_150/Relu:activations:0-batch_normalization_57/ReadVariableOp:value:0/batch_normalization_57/ReadVariableOp_1:value:0>batch_normalization_57/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( 2)
'batch_normalization_57/FusedBatchNormV3?
dropout_57/IdentityIdentity+batch_normalization_57/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????2
dropout_57/Identity?
max_pooling2d_60/MaxPoolMaxPooldropout_57/Identity:output:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_60/MaxPoolu
flatten_30/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   2
flatten_30/Const?
flatten_30/ReshapeReshape!max_pooling2d_60/MaxPool:output:0flatten_30/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_30/Reshape?
dense_60/MatMul/ReadVariableOpReadVariableOp'dense_60_matmul_readvariableop_resource*
_output_shapes
:	? *
dtype02 
dense_60/MatMul/ReadVariableOp?
dense_60/MatMulMatMulflatten_30/Reshape:output:0&dense_60/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_60/MatMul?
dense_60/BiasAdd/ReadVariableOpReadVariableOp(dense_60_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_60/BiasAdd/ReadVariableOp?
dense_60/BiasAddBiasAdddense_60/MatMul:product:0'dense_60/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_60/BiasAdds
dense_60/ReluReludense_60/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_60/Relu?
dense_61/MatMul/ReadVariableOpReadVariableOp'dense_61_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_61/MatMul/ReadVariableOp?
dense_61/MatMulMatMuldense_60/Relu:activations:0&dense_61/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_61/MatMul?
dense_61/BiasAdd/ReadVariableOpReadVariableOp(dense_61_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_61/BiasAdd/ReadVariableOp?
dense_61/BiasAddBiasAdddense_61/MatMul:product:0'dense_61/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_61/BiasAdd|
dense_61/SoftmaxSoftmaxdense_61/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_61/Softmax?
IdentityIdentitydense_61/Softmax:softmax:07^batch_normalization_52/FusedBatchNormV3/ReadVariableOp9^batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_52/ReadVariableOp(^batch_normalization_52/ReadVariableOp_17^batch_normalization_53/FusedBatchNormV3/ReadVariableOp9^batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_53/ReadVariableOp(^batch_normalization_53/ReadVariableOp_17^batch_normalization_54/FusedBatchNormV3/ReadVariableOp9^batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_54/ReadVariableOp(^batch_normalization_54/ReadVariableOp_17^batch_normalization_55/FusedBatchNormV3/ReadVariableOp9^batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_55/ReadVariableOp(^batch_normalization_55/ReadVariableOp_17^batch_normalization_56/FusedBatchNormV3/ReadVariableOp9^batch_normalization_56/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_56/ReadVariableOp(^batch_normalization_56/ReadVariableOp_17^batch_normalization_57/FusedBatchNormV3/ReadVariableOp9^batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_57/ReadVariableOp(^batch_normalization_57/ReadVariableOp_1"^conv2d_145/BiasAdd/ReadVariableOp!^conv2d_145/Conv2D/ReadVariableOp"^conv2d_146/BiasAdd/ReadVariableOp!^conv2d_146/Conv2D/ReadVariableOp"^conv2d_147/BiasAdd/ReadVariableOp!^conv2d_147/Conv2D/ReadVariableOp"^conv2d_148/BiasAdd/ReadVariableOp!^conv2d_148/Conv2D/ReadVariableOp"^conv2d_149/BiasAdd/ReadVariableOp!^conv2d_149/Conv2D/ReadVariableOp"^conv2d_150/BiasAdd/ReadVariableOp!^conv2d_150/Conv2D/ReadVariableOp ^dense_60/BiasAdd/ReadVariableOp^dense_60/MatMul/ReadVariableOp ^dense_61/BiasAdd/ReadVariableOp^dense_61/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::2p
6batch_normalization_52/FusedBatchNormV3/ReadVariableOp6batch_normalization_52/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_52/FusedBatchNormV3/ReadVariableOp_18batch_normalization_52/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_52/ReadVariableOp%batch_normalization_52/ReadVariableOp2R
'batch_normalization_52/ReadVariableOp_1'batch_normalization_52/ReadVariableOp_12p
6batch_normalization_53/FusedBatchNormV3/ReadVariableOp6batch_normalization_53/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_53/FusedBatchNormV3/ReadVariableOp_18batch_normalization_53/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_53/ReadVariableOp%batch_normalization_53/ReadVariableOp2R
'batch_normalization_53/ReadVariableOp_1'batch_normalization_53/ReadVariableOp_12p
6batch_normalization_54/FusedBatchNormV3/ReadVariableOp6batch_normalization_54/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_54/FusedBatchNormV3/ReadVariableOp_18batch_normalization_54/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_54/ReadVariableOp%batch_normalization_54/ReadVariableOp2R
'batch_normalization_54/ReadVariableOp_1'batch_normalization_54/ReadVariableOp_12p
6batch_normalization_55/FusedBatchNormV3/ReadVariableOp6batch_normalization_55/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_55/FusedBatchNormV3/ReadVariableOp_18batch_normalization_55/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_55/ReadVariableOp%batch_normalization_55/ReadVariableOp2R
'batch_normalization_55/ReadVariableOp_1'batch_normalization_55/ReadVariableOp_12p
6batch_normalization_56/FusedBatchNormV3/ReadVariableOp6batch_normalization_56/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_56/FusedBatchNormV3/ReadVariableOp_18batch_normalization_56/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_56/ReadVariableOp%batch_normalization_56/ReadVariableOp2R
'batch_normalization_56/ReadVariableOp_1'batch_normalization_56/ReadVariableOp_12p
6batch_normalization_57/FusedBatchNormV3/ReadVariableOp6batch_normalization_57/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_57/FusedBatchNormV3/ReadVariableOp_18batch_normalization_57/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_57/ReadVariableOp%batch_normalization_57/ReadVariableOp2R
'batch_normalization_57/ReadVariableOp_1'batch_normalization_57/ReadVariableOp_12F
!conv2d_145/BiasAdd/ReadVariableOp!conv2d_145/BiasAdd/ReadVariableOp2D
 conv2d_145/Conv2D/ReadVariableOp conv2d_145/Conv2D/ReadVariableOp2F
!conv2d_146/BiasAdd/ReadVariableOp!conv2d_146/BiasAdd/ReadVariableOp2D
 conv2d_146/Conv2D/ReadVariableOp conv2d_146/Conv2D/ReadVariableOp2F
!conv2d_147/BiasAdd/ReadVariableOp!conv2d_147/BiasAdd/ReadVariableOp2D
 conv2d_147/Conv2D/ReadVariableOp conv2d_147/Conv2D/ReadVariableOp2F
!conv2d_148/BiasAdd/ReadVariableOp!conv2d_148/BiasAdd/ReadVariableOp2D
 conv2d_148/Conv2D/ReadVariableOp conv2d_148/Conv2D/ReadVariableOp2F
!conv2d_149/BiasAdd/ReadVariableOp!conv2d_149/BiasAdd/ReadVariableOp2D
 conv2d_149/Conv2D/ReadVariableOp conv2d_149/Conv2D/ReadVariableOp2F
!conv2d_150/BiasAdd/ReadVariableOp!conv2d_150/BiasAdd/ReadVariableOp2D
 conv2d_150/Conv2D/ReadVariableOp conv2d_150/Conv2D/ReadVariableOp2B
dense_60/BiasAdd/ReadVariableOpdense_60/BiasAdd/ReadVariableOp2@
dense_60/MatMul/ReadVariableOpdense_60/MatMul/ReadVariableOp2B
dense_61/BiasAdd/ReadVariableOpdense_61/BiasAdd/ReadVariableOp2@
dense_61/MatMul/ReadVariableOpdense_61/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_57_layer_call_and_return_conditional_losses_2708270

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
S__inference_batch_normalization_52_layer_call_and_return_conditional_losses_2705361

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
?
?
%__inference_signature_wrapper_2706757
input_31
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
StatefulPartitionedCallStatefulPartitionedCallinput_31unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
"__inference__wrapped_model_27046512
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
input_31
?
?
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_2705770

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
?
?
8__inference_batch_normalization_55_layer_call_fn_2707900

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
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_27050682
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
S__inference_batch_normalization_57_layer_call_and_return_conditional_losses_2708206

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
?
e
G__inference_dropout_55_layer_call_and_return_conditional_losses_2705823

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????**2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????**2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????**:W S
/
_output_shapes
:?????????**
 
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
input_31;
serving_default_input_31:0???????????<
dense_610
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
??
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer_with_weights-6
layer-11
layer_with_weights-7
layer-12
layer-13
layer-14
layer_with_weights-8
layer-15
layer_with_weights-9
layer-16
layer-17
layer_with_weights-10
layer-18
layer_with_weights-11
layer-19
layer-20
layer-21
layer-22
layer_with_weights-12
layer-23
layer_with_weights-13
layer-24
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
?_default_save_signature
+?&call_and_return_all_conditional_losses
?__call__"??
_tf_keras_network??{"class_name": "Functional", "name": "CNN_aug_deep_bnorm_drop", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "CNN_aug_deep_bnorm_drop", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_31"}, "name": "input_31", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_145", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_145", "inbound_nodes": [[["input_31", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_52", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_52", "inbound_nodes": [[["conv2d_145", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_52", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "name": "dropout_52", "inbound_nodes": [[["batch_normalization_52", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_146", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_146", "inbound_nodes": [[["dropout_52", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_53", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_53", "inbound_nodes": [[["conv2d_146", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_53", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "name": "dropout_53", "inbound_nodes": [[["batch_normalization_53", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_58", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "name": "max_pooling2d_58", "inbound_nodes": [[["dropout_53", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_147", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_147", "inbound_nodes": [[["max_pooling2d_58", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_54", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_54", "inbound_nodes": [[["conv2d_147", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_54", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "name": "dropout_54", "inbound_nodes": [[["batch_normalization_54", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_148", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_148", "inbound_nodes": [[["dropout_54", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_55", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_55", "inbound_nodes": [[["conv2d_148", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_55", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "name": "dropout_55", "inbound_nodes": [[["batch_normalization_55", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_59", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "name": "max_pooling2d_59", "inbound_nodes": [[["dropout_55", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_149", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_149", "inbound_nodes": [[["max_pooling2d_59", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_56", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_56", "inbound_nodes": [[["conv2d_149", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_56", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "name": "dropout_56", "inbound_nodes": [[["batch_normalization_56", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_150", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_150", "inbound_nodes": [[["dropout_56", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_57", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_57", "inbound_nodes": [[["conv2d_150", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_57", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "name": "dropout_57", "inbound_nodes": [[["batch_normalization_57", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_60", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "name": "max_pooling2d_60", "inbound_nodes": [[["dropout_57", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_30", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_30", "inbound_nodes": [[["max_pooling2d_60", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_60", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_60", "inbound_nodes": [[["flatten_30", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_61", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_61", "inbound_nodes": [[["dense_60", 0, 0, {}]]]}], "input_layers": [["input_31", 0, 0]], "output_layers": [["dense_61", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "CNN_aug_deep_bnorm_drop", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_31"}, "name": "input_31", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_145", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_145", "inbound_nodes": [[["input_31", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_52", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_52", "inbound_nodes": [[["conv2d_145", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_52", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "name": "dropout_52", "inbound_nodes": [[["batch_normalization_52", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_146", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_146", "inbound_nodes": [[["dropout_52", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_53", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_53", "inbound_nodes": [[["conv2d_146", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_53", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "name": "dropout_53", "inbound_nodes": [[["batch_normalization_53", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_58", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "name": "max_pooling2d_58", "inbound_nodes": [[["dropout_53", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_147", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_147", "inbound_nodes": [[["max_pooling2d_58", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_54", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_54", "inbound_nodes": [[["conv2d_147", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_54", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "name": "dropout_54", "inbound_nodes": [[["batch_normalization_54", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_148", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_148", "inbound_nodes": [[["dropout_54", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_55", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_55", "inbound_nodes": [[["conv2d_148", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_55", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "name": "dropout_55", "inbound_nodes": [[["batch_normalization_55", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_59", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "name": "max_pooling2d_59", "inbound_nodes": [[["dropout_55", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_149", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_149", "inbound_nodes": [[["max_pooling2d_59", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_56", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_56", "inbound_nodes": [[["conv2d_149", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_56", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "name": "dropout_56", "inbound_nodes": [[["batch_normalization_56", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_150", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_150", "inbound_nodes": [[["dropout_56", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_57", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_57", "inbound_nodes": [[["conv2d_150", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_57", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "name": "dropout_57", "inbound_nodes": [[["batch_normalization_57", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_60", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "name": "max_pooling2d_60", "inbound_nodes": [[["dropout_57", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_30", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_30", "inbound_nodes": [[["max_pooling2d_60", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_60", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_60", "inbound_nodes": [[["flatten_30", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_61", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_61", "inbound_nodes": [[["dense_60", 0, 0, {}]]]}], "input_layers": [["input_31", 0, 0]], "output_layers": [["dense_61", 0, 0]]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_31", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_31"}}
?	

 kernel
!bias
"regularization_losses
#trainable_variables
$	variables
%	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_145", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_145", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 1]}}
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
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_52", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_52", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 8]}}
?
/regularization_losses
0trainable_variables
1	variables
2	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_52", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_52", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}
?	

3kernel
4bias
5regularization_losses
6trainable_variables
7	variables
8	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_146", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_146", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 8]}}
?	
9axis
	:gamma
;beta
<moving_mean
=moving_variance
>regularization_losses
?trainable_variables
@	variables
A	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_53", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_53", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 8]}}
?
Bregularization_losses
Ctrainable_variables
D	variables
E	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_53", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_53", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}
?
Fregularization_losses
Gtrainable_variables
H	variables
I	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_58", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_58", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

Jkernel
Kbias
Lregularization_losses
Mtrainable_variables
N	variables
O	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_147", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_147", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 42, 42, 8]}}
?	
Paxis
	Qgamma
Rbeta
Smoving_mean
Tmoving_variance
Uregularization_losses
Vtrainable_variables
W	variables
X	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_54", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_54", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 42, 42, 8]}}
?
Yregularization_losses
Ztrainable_variables
[	variables
\	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_54", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_54", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}
?	

]kernel
^bias
_regularization_losses
`trainable_variables
a	variables
b	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_148", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_148", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 42, 42, 8]}}
?	
caxis
	dgamma
ebeta
fmoving_mean
gmoving_variance
hregularization_losses
itrainable_variables
j	variables
k	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_55", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_55", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 42, 42, 8]}}
?
lregularization_losses
mtrainable_variables
n	variables
o	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_55", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_55", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}
?
pregularization_losses
qtrainable_variables
r	variables
s	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_59", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_59", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

tkernel
ubias
vregularization_losses
wtrainable_variables
x	variables
y	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_149", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_149", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 8]}}
?	
zaxis
	{gamma
|beta
}moving_mean
~moving_variance
regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_56", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_56", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 8]}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_56", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_56", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}
?	
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_150", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_150", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 8]}}
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
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_57", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_57", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 8]}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_57", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_57", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_60", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_60", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_30", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_30", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_60", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_60", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
?
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_61", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_61", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
?
	?iter
?beta_1
?beta_2

?decay
?learning_rate m?!m?'m?(m?3m?4m?:m?;m?Jm?Km?Qm?Rm?]m?^m?dm?em?tm?um?{m?|m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m? v?!v?'v?(v?3v?4v?:v?;v?Jv?Kv?Qv?Rv?]v?^v?dv?ev?tv?uv?{v?|v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?"
	optimizer
 "
trackable_list_wrapper
?
 0
!1
'2
(3
34
45
:6
;7
J8
K9
Q10
R11
]12
^13
d14
e15
t16
u17
{18
|19
?20
?21
?22
?23
?24
?25
?26
?27"
trackable_list_wrapper
?
 0
!1
'2
(3
)4
*5
36
47
:8
;9
<10
=11
J12
K13
Q14
R15
S16
T17
]18
^19
d20
e21
f22
g23
t24
u25
{26
|27
}28
~29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39"
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
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
+:)2conv2d_145/kernel
:2conv2d_145/bias
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
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_52/gamma
):'2batch_normalization_52/beta
2:0 (2"batch_normalization_52/moving_mean
6:4 (2&batch_normalization_52/moving_variance
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
/regularization_losses
?layers
?layer_metrics
?metrics
0trainable_variables
1	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)2conv2d_146/kernel
:2conv2d_146/bias
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
?
5regularization_losses
?layers
?layer_metrics
?metrics
6trainable_variables
7	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_53/gamma
):'2batch_normalization_53/beta
2:0 (2"batch_normalization_53/moving_mean
6:4 (2&batch_normalization_53/moving_variance
 "
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
<
:0
;1
<2
=3"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Bregularization_losses
?layers
?layer_metrics
?metrics
Ctrainable_variables
D	variables
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
Fregularization_losses
?layers
?layer_metrics
?metrics
Gtrainable_variables
H	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)2conv2d_147/kernel
:2conv2d_147/bias
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
?
Lregularization_losses
?layers
?layer_metrics
?metrics
Mtrainable_variables
N	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_54/gamma
):'2batch_normalization_54/beta
2:0 (2"batch_normalization_54/moving_mean
6:4 (2&batch_normalization_54/moving_variance
 "
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
<
Q0
R1
S2
T3"
trackable_list_wrapper
?
Uregularization_losses
?layers
?layer_metrics
?metrics
Vtrainable_variables
W	variables
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
Yregularization_losses
?layers
?layer_metrics
?metrics
Ztrainable_variables
[	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)2conv2d_148/kernel
:2conv2d_148/bias
 "
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
?
_regularization_losses
?layers
?layer_metrics
?metrics
`trainable_variables
a	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_55/gamma
):'2batch_normalization_55/beta
2:0 (2"batch_normalization_55/moving_mean
6:4 (2&batch_normalization_55/moving_variance
 "
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
<
d0
e1
f2
g3"
trackable_list_wrapper
?
hregularization_losses
?layers
?layer_metrics
?metrics
itrainable_variables
j	variables
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
+:)2conv2d_149/kernel
:2conv2d_149/bias
 "
trackable_list_wrapper
.
t0
u1"
trackable_list_wrapper
.
t0
u1"
trackable_list_wrapper
?
vregularization_losses
?layers
?layer_metrics
?metrics
wtrainable_variables
x	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_56/gamma
):'2batch_normalization_56/beta
2:0 (2"batch_normalization_56/moving_mean
6:4 (2&batch_normalization_56/moving_variance
 "
trackable_list_wrapper
.
{0
|1"
trackable_list_wrapper
<
{0
|1
}2
~3"
trackable_list_wrapper
?
regularization_losses
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
+:)2conv2d_150/kernel
:2conv2d_150/bias
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
*:(2batch_normalization_57/gamma
):'2batch_normalization_57/beta
2:0 (2"batch_normalization_57/moving_mean
6:4 (2&batch_normalization_57/moving_variance
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
": 	? 2dense_60/kernel
: 2dense_60/bias
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
!: 2dense_61/kernel
:2dense_61/bias
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
x
)0
*1
<2
=3
S4
T5
f6
g7
}8
~9
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
<0
=1"
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
S0
T1"
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
f0
g1"
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
}0
~1"
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
0:.2Adam/conv2d_145/kernel/m
": 2Adam/conv2d_145/bias/m
/:-2#Adam/batch_normalization_52/gamma/m
.:,2"Adam/batch_normalization_52/beta/m
0:.2Adam/conv2d_146/kernel/m
": 2Adam/conv2d_146/bias/m
/:-2#Adam/batch_normalization_53/gamma/m
.:,2"Adam/batch_normalization_53/beta/m
0:.2Adam/conv2d_147/kernel/m
": 2Adam/conv2d_147/bias/m
/:-2#Adam/batch_normalization_54/gamma/m
.:,2"Adam/batch_normalization_54/beta/m
0:.2Adam/conv2d_148/kernel/m
": 2Adam/conv2d_148/bias/m
/:-2#Adam/batch_normalization_55/gamma/m
.:,2"Adam/batch_normalization_55/beta/m
0:.2Adam/conv2d_149/kernel/m
": 2Adam/conv2d_149/bias/m
/:-2#Adam/batch_normalization_56/gamma/m
.:,2"Adam/batch_normalization_56/beta/m
0:.2Adam/conv2d_150/kernel/m
": 2Adam/conv2d_150/bias/m
/:-2#Adam/batch_normalization_57/gamma/m
.:,2"Adam/batch_normalization_57/beta/m
':%	? 2Adam/dense_60/kernel/m
 : 2Adam/dense_60/bias/m
&:$ 2Adam/dense_61/kernel/m
 :2Adam/dense_61/bias/m
0:.2Adam/conv2d_145/kernel/v
": 2Adam/conv2d_145/bias/v
/:-2#Adam/batch_normalization_52/gamma/v
.:,2"Adam/batch_normalization_52/beta/v
0:.2Adam/conv2d_146/kernel/v
": 2Adam/conv2d_146/bias/v
/:-2#Adam/batch_normalization_53/gamma/v
.:,2"Adam/batch_normalization_53/beta/v
0:.2Adam/conv2d_147/kernel/v
": 2Adam/conv2d_147/bias/v
/:-2#Adam/batch_normalization_54/gamma/v
.:,2"Adam/batch_normalization_54/beta/v
0:.2Adam/conv2d_148/kernel/v
": 2Adam/conv2d_148/bias/v
/:-2#Adam/batch_normalization_55/gamma/v
.:,2"Adam/batch_normalization_55/beta/v
0:.2Adam/conv2d_149/kernel/v
": 2Adam/conv2d_149/bias/v
/:-2#Adam/batch_normalization_56/gamma/v
.:,2"Adam/batch_normalization_56/beta/v
0:.2Adam/conv2d_150/kernel/v
": 2Adam/conv2d_150/bias/v
/:-2#Adam/batch_normalization_57/gamma/v
.:,2"Adam/batch_normalization_57/beta/v
':%	? 2Adam/dense_60/kernel/v
 : 2Adam/dense_60/bias/v
&:$ 2Adam/dense_61/kernel/v
 :2Adam/dense_61/bias/v
?2?
"__inference__wrapped_model_2704651?
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
input_31???????????
?2?
T__inference_CNN_aug_deep_bnorm_drop_layer_call_and_return_conditional_losses_2706167
T__inference_CNN_aug_deep_bnorm_drop_layer_call_and_return_conditional_losses_2706966
T__inference_CNN_aug_deep_bnorm_drop_layer_call_and_return_conditional_losses_2707121
T__inference_CNN_aug_deep_bnorm_drop_layer_call_and_return_conditional_losses_2706275?
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
9__inference_CNN_aug_deep_bnorm_drop_layer_call_fn_2706662
9__inference_CNN_aug_deep_bnorm_drop_layer_call_fn_2707291
9__inference_CNN_aug_deep_bnorm_drop_layer_call_fn_2706469
9__inference_CNN_aug_deep_bnorm_drop_layer_call_fn_2707206?
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
G__inference_conv2d_145_layer_call_and_return_conditional_losses_2707302?
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
,__inference_conv2d_145_layer_call_fn_2707311?
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
S__inference_batch_normalization_52_layer_call_and_return_conditional_losses_2707349
S__inference_batch_normalization_52_layer_call_and_return_conditional_losses_2707331
S__inference_batch_normalization_52_layer_call_and_return_conditional_losses_2707395
S__inference_batch_normalization_52_layer_call_and_return_conditional_losses_2707413?
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
8__inference_batch_normalization_52_layer_call_fn_2707362
8__inference_batch_normalization_52_layer_call_fn_2707426
8__inference_batch_normalization_52_layer_call_fn_2707439
8__inference_batch_normalization_52_layer_call_fn_2707375?
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
G__inference_dropout_52_layer_call_and_return_conditional_losses_2707451
G__inference_dropout_52_layer_call_and_return_conditional_losses_2707456?
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
,__inference_dropout_52_layer_call_fn_2707461
,__inference_dropout_52_layer_call_fn_2707466?
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
G__inference_conv2d_146_layer_call_and_return_conditional_losses_2707477?
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
,__inference_conv2d_146_layer_call_fn_2707486?
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
S__inference_batch_normalization_53_layer_call_and_return_conditional_losses_2707570
S__inference_batch_normalization_53_layer_call_and_return_conditional_losses_2707506
S__inference_batch_normalization_53_layer_call_and_return_conditional_losses_2707588
S__inference_batch_normalization_53_layer_call_and_return_conditional_losses_2707524?
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
8__inference_batch_normalization_53_layer_call_fn_2707537
8__inference_batch_normalization_53_layer_call_fn_2707550
8__inference_batch_normalization_53_layer_call_fn_2707614
8__inference_batch_normalization_53_layer_call_fn_2707601?
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
G__inference_dropout_53_layer_call_and_return_conditional_losses_2707626
G__inference_dropout_53_layer_call_and_return_conditional_losses_2707631?
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
,__inference_dropout_53_layer_call_fn_2707636
,__inference_dropout_53_layer_call_fn_2707641?
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
M__inference_max_pooling2d_58_layer_call_and_return_conditional_losses_2704865?
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
2__inference_max_pooling2d_58_layer_call_fn_2704871?
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
G__inference_conv2d_147_layer_call_and_return_conditional_losses_2707652?
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
,__inference_conv2d_147_layer_call_fn_2707661?
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
S__inference_batch_normalization_54_layer_call_and_return_conditional_losses_2707763
S__inference_batch_normalization_54_layer_call_and_return_conditional_losses_2707699
S__inference_batch_normalization_54_layer_call_and_return_conditional_losses_2707681
S__inference_batch_normalization_54_layer_call_and_return_conditional_losses_2707745?
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
8__inference_batch_normalization_54_layer_call_fn_2707712
8__inference_batch_normalization_54_layer_call_fn_2707725
8__inference_batch_normalization_54_layer_call_fn_2707776
8__inference_batch_normalization_54_layer_call_fn_2707789?
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
G__inference_dropout_54_layer_call_and_return_conditional_losses_2707801
G__inference_dropout_54_layer_call_and_return_conditional_losses_2707806?
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
,__inference_dropout_54_layer_call_fn_2707811
,__inference_dropout_54_layer_call_fn_2707816?
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
G__inference_conv2d_148_layer_call_and_return_conditional_losses_2707827?
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
,__inference_conv2d_148_layer_call_fn_2707836?
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
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_2707856
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_2707874
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_2707920
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_2707938?
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
8__inference_batch_normalization_55_layer_call_fn_2707887
8__inference_batch_normalization_55_layer_call_fn_2707900
8__inference_batch_normalization_55_layer_call_fn_2707964
8__inference_batch_normalization_55_layer_call_fn_2707951?
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
G__inference_dropout_55_layer_call_and_return_conditional_losses_2707976
G__inference_dropout_55_layer_call_and_return_conditional_losses_2707981?
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
,__inference_dropout_55_layer_call_fn_2707986
,__inference_dropout_55_layer_call_fn_2707991?
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
M__inference_max_pooling2d_59_layer_call_and_return_conditional_losses_2705085?
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
2__inference_max_pooling2d_59_layer_call_fn_2705091?
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
G__inference_conv2d_149_layer_call_and_return_conditional_losses_2708002?
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
,__inference_conv2d_149_layer_call_fn_2708011?
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
S__inference_batch_normalization_56_layer_call_and_return_conditional_losses_2708031
S__inference_batch_normalization_56_layer_call_and_return_conditional_losses_2708113
S__inference_batch_normalization_56_layer_call_and_return_conditional_losses_2708095
S__inference_batch_normalization_56_layer_call_and_return_conditional_losses_2708049?
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
8__inference_batch_normalization_56_layer_call_fn_2708139
8__inference_batch_normalization_56_layer_call_fn_2708126
8__inference_batch_normalization_56_layer_call_fn_2708075
8__inference_batch_normalization_56_layer_call_fn_2708062?
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
G__inference_dropout_56_layer_call_and_return_conditional_losses_2708151
G__inference_dropout_56_layer_call_and_return_conditional_losses_2708156?
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
,__inference_dropout_56_layer_call_fn_2708161
,__inference_dropout_56_layer_call_fn_2708166?
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
G__inference_conv2d_150_layer_call_and_return_conditional_losses_2708177?
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
,__inference_conv2d_150_layer_call_fn_2708186?
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
S__inference_batch_normalization_57_layer_call_and_return_conditional_losses_2708206
S__inference_batch_normalization_57_layer_call_and_return_conditional_losses_2708224
S__inference_batch_normalization_57_layer_call_and_return_conditional_losses_2708270
S__inference_batch_normalization_57_layer_call_and_return_conditional_losses_2708288?
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
8__inference_batch_normalization_57_layer_call_fn_2708314
8__inference_batch_normalization_57_layer_call_fn_2708301
8__inference_batch_normalization_57_layer_call_fn_2708237
8__inference_batch_normalization_57_layer_call_fn_2708250?
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
G__inference_dropout_57_layer_call_and_return_conditional_losses_2708326
G__inference_dropout_57_layer_call_and_return_conditional_losses_2708331?
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
,__inference_dropout_57_layer_call_fn_2708336
,__inference_dropout_57_layer_call_fn_2708341?
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
M__inference_max_pooling2d_60_layer_call_and_return_conditional_losses_2705305?
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
2__inference_max_pooling2d_60_layer_call_fn_2705311?
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
G__inference_flatten_30_layer_call_and_return_conditional_losses_2708347?
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
,__inference_flatten_30_layer_call_fn_2708352?
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
E__inference_dense_60_layer_call_and_return_conditional_losses_2708363?
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
*__inference_dense_60_layer_call_fn_2708372?
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
E__inference_dense_61_layer_call_and_return_conditional_losses_2708383?
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
*__inference_dense_61_layer_call_fn_2708392?
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
%__inference_signature_wrapper_2706757input_31"?
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
T__inference_CNN_aug_deep_bnorm_drop_layer_call_and_return_conditional_losses_2706167?2 !'()*34:;<=JKQRST]^defgtu{|}~??????????C?@
9?6
,?)
input_31???????????
p

 
? "%?"
?
0?????????
? ?
T__inference_CNN_aug_deep_bnorm_drop_layer_call_and_return_conditional_losses_2706275?2 !'()*34:;<=JKQRST]^defgtu{|}~??????????C?@
9?6
,?)
input_31???????????
p 

 
? "%?"
?
0?????????
? ?
T__inference_CNN_aug_deep_bnorm_drop_layer_call_and_return_conditional_losses_2706966?2 !'()*34:;<=JKQRST]^defgtu{|}~??????????A?>
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
T__inference_CNN_aug_deep_bnorm_drop_layer_call_and_return_conditional_losses_2707121?2 !'()*34:;<=JKQRST]^defgtu{|}~??????????A?>
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
9__inference_CNN_aug_deep_bnorm_drop_layer_call_fn_2706469?2 !'()*34:;<=JKQRST]^defgtu{|}~??????????C?@
9?6
,?)
input_31???????????
p

 
? "???????????
9__inference_CNN_aug_deep_bnorm_drop_layer_call_fn_2706662?2 !'()*34:;<=JKQRST]^defgtu{|}~??????????C?@
9?6
,?)
input_31???????????
p 

 
? "???????????
9__inference_CNN_aug_deep_bnorm_drop_layer_call_fn_2707206?2 !'()*34:;<=JKQRST]^defgtu{|}~??????????A?>
7?4
*?'
inputs???????????
p

 
? "???????????
9__inference_CNN_aug_deep_bnorm_drop_layer_call_fn_2707291?2 !'()*34:;<=JKQRST]^defgtu{|}~??????????A?>
7?4
*?'
inputs???????????
p 

 
? "???????????
"__inference__wrapped_model_2704651?2 !'()*34:;<=JKQRST]^defgtu{|}~??????????;?8
1?.
,?)
input_31???????????
? "3?0
.
dense_61"?
dense_61??????????
S__inference_batch_normalization_52_layer_call_and_return_conditional_losses_2707331?'()*M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_52_layer_call_and_return_conditional_losses_2707349?'()*M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_52_layer_call_and_return_conditional_losses_2707395v'()*=?:
3?0
*?'
inputs???????????
p
? "/?,
%?"
0???????????
? ?
S__inference_batch_normalization_52_layer_call_and_return_conditional_losses_2707413v'()*=?:
3?0
*?'
inputs???????????
p 
? "/?,
%?"
0???????????
? ?
8__inference_batch_normalization_52_layer_call_fn_2707362?'()*M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
8__inference_batch_normalization_52_layer_call_fn_2707375?'()*M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
8__inference_batch_normalization_52_layer_call_fn_2707426i'()*=?:
3?0
*?'
inputs???????????
p
? ""?????????????
8__inference_batch_normalization_52_layer_call_fn_2707439i'()*=?:
3?0
*?'
inputs???????????
p 
? ""?????????????
S__inference_batch_normalization_53_layer_call_and_return_conditional_losses_2707506?:;<=M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_53_layer_call_and_return_conditional_losses_2707524?:;<=M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_53_layer_call_and_return_conditional_losses_2707570v:;<==?:
3?0
*?'
inputs???????????
p
? "/?,
%?"
0???????????
? ?
S__inference_batch_normalization_53_layer_call_and_return_conditional_losses_2707588v:;<==?:
3?0
*?'
inputs???????????
p 
? "/?,
%?"
0???????????
? ?
8__inference_batch_normalization_53_layer_call_fn_2707537?:;<=M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
8__inference_batch_normalization_53_layer_call_fn_2707550?:;<=M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
8__inference_batch_normalization_53_layer_call_fn_2707601i:;<==?:
3?0
*?'
inputs???????????
p
? ""?????????????
8__inference_batch_normalization_53_layer_call_fn_2707614i:;<==?:
3?0
*?'
inputs???????????
p 
? ""?????????????
S__inference_batch_normalization_54_layer_call_and_return_conditional_losses_2707681rQRST;?8
1?.
(?%
inputs?????????**
p
? "-?*
#? 
0?????????**
? ?
S__inference_batch_normalization_54_layer_call_and_return_conditional_losses_2707699rQRST;?8
1?.
(?%
inputs?????????**
p 
? "-?*
#? 
0?????????**
? ?
S__inference_batch_normalization_54_layer_call_and_return_conditional_losses_2707745?QRSTM?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_54_layer_call_and_return_conditional_losses_2707763?QRSTM?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
8__inference_batch_normalization_54_layer_call_fn_2707712eQRST;?8
1?.
(?%
inputs?????????**
p
? " ??????????**?
8__inference_batch_normalization_54_layer_call_fn_2707725eQRST;?8
1?.
(?%
inputs?????????**
p 
? " ??????????**?
8__inference_batch_normalization_54_layer_call_fn_2707776?QRSTM?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
8__inference_batch_normalization_54_layer_call_fn_2707789?QRSTM?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_2707856?defgM?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_2707874?defgM?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_2707920rdefg;?8
1?.
(?%
inputs?????????**
p
? "-?*
#? 
0?????????**
? ?
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_2707938rdefg;?8
1?.
(?%
inputs?????????**
p 
? "-?*
#? 
0?????????**
? ?
8__inference_batch_normalization_55_layer_call_fn_2707887?defgM?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
8__inference_batch_normalization_55_layer_call_fn_2707900?defgM?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
8__inference_batch_normalization_55_layer_call_fn_2707951edefg;?8
1?.
(?%
inputs?????????**
p
? " ??????????**?
8__inference_batch_normalization_55_layer_call_fn_2707964edefg;?8
1?.
(?%
inputs?????????**
p 
? " ??????????**?
S__inference_batch_normalization_56_layer_call_and_return_conditional_losses_2708031?{|}~M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_56_layer_call_and_return_conditional_losses_2708049?{|}~M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_56_layer_call_and_return_conditional_losses_2708095r{|}~;?8
1?.
(?%
inputs?????????
p
? "-?*
#? 
0?????????
? ?
S__inference_batch_normalization_56_layer_call_and_return_conditional_losses_2708113r{|}~;?8
1?.
(?%
inputs?????????
p 
? "-?*
#? 
0?????????
? ?
8__inference_batch_normalization_56_layer_call_fn_2708062?{|}~M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
8__inference_batch_normalization_56_layer_call_fn_2708075?{|}~M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
8__inference_batch_normalization_56_layer_call_fn_2708126e{|}~;?8
1?.
(?%
inputs?????????
p
? " ???????????
8__inference_batch_normalization_56_layer_call_fn_2708139e{|}~;?8
1?.
(?%
inputs?????????
p 
? " ???????????
S__inference_batch_normalization_57_layer_call_and_return_conditional_losses_2708206?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_57_layer_call_and_return_conditional_losses_2708224?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_57_layer_call_and_return_conditional_losses_2708270v????;?8
1?.
(?%
inputs?????????
p
? "-?*
#? 
0?????????
? ?
S__inference_batch_normalization_57_layer_call_and_return_conditional_losses_2708288v????;?8
1?.
(?%
inputs?????????
p 
? "-?*
#? 
0?????????
? ?
8__inference_batch_normalization_57_layer_call_fn_2708237?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
8__inference_batch_normalization_57_layer_call_fn_2708250?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
8__inference_batch_normalization_57_layer_call_fn_2708301i????;?8
1?.
(?%
inputs?????????
p
? " ???????????
8__inference_batch_normalization_57_layer_call_fn_2708314i????;?8
1?.
(?%
inputs?????????
p 
? " ???????????
G__inference_conv2d_145_layer_call_and_return_conditional_losses_2707302p !9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
,__inference_conv2d_145_layer_call_fn_2707311c !9?6
/?,
*?'
inputs???????????
? ""?????????????
G__inference_conv2d_146_layer_call_and_return_conditional_losses_2707477p349?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
,__inference_conv2d_146_layer_call_fn_2707486c349?6
/?,
*?'
inputs???????????
? ""?????????????
G__inference_conv2d_147_layer_call_and_return_conditional_losses_2707652lJK7?4
-?*
(?%
inputs?????????**
? "-?*
#? 
0?????????**
? ?
,__inference_conv2d_147_layer_call_fn_2707661_JK7?4
-?*
(?%
inputs?????????**
? " ??????????**?
G__inference_conv2d_148_layer_call_and_return_conditional_losses_2707827l]^7?4
-?*
(?%
inputs?????????**
? "-?*
#? 
0?????????**
? ?
,__inference_conv2d_148_layer_call_fn_2707836_]^7?4
-?*
(?%
inputs?????????**
? " ??????????**?
G__inference_conv2d_149_layer_call_and_return_conditional_losses_2708002ltu7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
,__inference_conv2d_149_layer_call_fn_2708011_tu7?4
-?*
(?%
inputs?????????
? " ???????????
G__inference_conv2d_150_layer_call_and_return_conditional_losses_2708177n??7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
,__inference_conv2d_150_layer_call_fn_2708186a??7?4
-?*
(?%
inputs?????????
? " ???????????
E__inference_dense_60_layer_call_and_return_conditional_losses_2708363_??0?-
&?#
!?
inputs??????????
? "%?"
?
0????????? 
? ?
*__inference_dense_60_layer_call_fn_2708372R??0?-
&?#
!?
inputs??????????
? "?????????? ?
E__inference_dense_61_layer_call_and_return_conditional_losses_2708383^??/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????
? 
*__inference_dense_61_layer_call_fn_2708392Q??/?,
%?"
 ?
inputs????????? 
? "???????????
G__inference_dropout_52_layer_call_and_return_conditional_losses_2707451p=?:
3?0
*?'
inputs???????????
p
? "/?,
%?"
0???????????
? ?
G__inference_dropout_52_layer_call_and_return_conditional_losses_2707456p=?:
3?0
*?'
inputs???????????
p 
? "/?,
%?"
0???????????
? ?
,__inference_dropout_52_layer_call_fn_2707461c=?:
3?0
*?'
inputs???????????
p
? ""?????????????
,__inference_dropout_52_layer_call_fn_2707466c=?:
3?0
*?'
inputs???????????
p 
? ""?????????????
G__inference_dropout_53_layer_call_and_return_conditional_losses_2707626p=?:
3?0
*?'
inputs???????????
p
? "/?,
%?"
0???????????
? ?
G__inference_dropout_53_layer_call_and_return_conditional_losses_2707631p=?:
3?0
*?'
inputs???????????
p 
? "/?,
%?"
0???????????
? ?
,__inference_dropout_53_layer_call_fn_2707636c=?:
3?0
*?'
inputs???????????
p
? ""?????????????
,__inference_dropout_53_layer_call_fn_2707641c=?:
3?0
*?'
inputs???????????
p 
? ""?????????????
G__inference_dropout_54_layer_call_and_return_conditional_losses_2707801l;?8
1?.
(?%
inputs?????????**
p
? "-?*
#? 
0?????????**
? ?
G__inference_dropout_54_layer_call_and_return_conditional_losses_2707806l;?8
1?.
(?%
inputs?????????**
p 
? "-?*
#? 
0?????????**
? ?
,__inference_dropout_54_layer_call_fn_2707811_;?8
1?.
(?%
inputs?????????**
p
? " ??????????**?
,__inference_dropout_54_layer_call_fn_2707816_;?8
1?.
(?%
inputs?????????**
p 
? " ??????????**?
G__inference_dropout_55_layer_call_and_return_conditional_losses_2707976l;?8
1?.
(?%
inputs?????????**
p
? "-?*
#? 
0?????????**
? ?
G__inference_dropout_55_layer_call_and_return_conditional_losses_2707981l;?8
1?.
(?%
inputs?????????**
p 
? "-?*
#? 
0?????????**
? ?
,__inference_dropout_55_layer_call_fn_2707986_;?8
1?.
(?%
inputs?????????**
p
? " ??????????**?
,__inference_dropout_55_layer_call_fn_2707991_;?8
1?.
(?%
inputs?????????**
p 
? " ??????????**?
G__inference_dropout_56_layer_call_and_return_conditional_losses_2708151l;?8
1?.
(?%
inputs?????????
p
? "-?*
#? 
0?????????
? ?
G__inference_dropout_56_layer_call_and_return_conditional_losses_2708156l;?8
1?.
(?%
inputs?????????
p 
? "-?*
#? 
0?????????
? ?
,__inference_dropout_56_layer_call_fn_2708161_;?8
1?.
(?%
inputs?????????
p
? " ???????????
,__inference_dropout_56_layer_call_fn_2708166_;?8
1?.
(?%
inputs?????????
p 
? " ???????????
G__inference_dropout_57_layer_call_and_return_conditional_losses_2708326l;?8
1?.
(?%
inputs?????????
p
? "-?*
#? 
0?????????
? ?
G__inference_dropout_57_layer_call_and_return_conditional_losses_2708331l;?8
1?.
(?%
inputs?????????
p 
? "-?*
#? 
0?????????
? ?
,__inference_dropout_57_layer_call_fn_2708336_;?8
1?.
(?%
inputs?????????
p
? " ???????????
,__inference_dropout_57_layer_call_fn_2708341_;?8
1?.
(?%
inputs?????????
p 
? " ???????????
G__inference_flatten_30_layer_call_and_return_conditional_losses_2708347a7?4
-?*
(?%
inputs?????????
? "&?#
?
0??????????
? ?
,__inference_flatten_30_layer_call_fn_2708352T7?4
-?*
(?%
inputs?????????
? "????????????
M__inference_max_pooling2d_58_layer_call_and_return_conditional_losses_2704865?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
2__inference_max_pooling2d_58_layer_call_fn_2704871?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
M__inference_max_pooling2d_59_layer_call_and_return_conditional_losses_2705085?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
2__inference_max_pooling2d_59_layer_call_fn_2705091?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
M__inference_max_pooling2d_60_layer_call_and_return_conditional_losses_2705305?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
2__inference_max_pooling2d_60_layer_call_fn_2705311?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
%__inference_signature_wrapper_2706757?2 !'()*34:;<=JKQRST]^defgtu{|}~??????????G?D
? 
=?:
8
input_31,?)
input_31???????????"3?0
.
dense_61"?
dense_61?????????