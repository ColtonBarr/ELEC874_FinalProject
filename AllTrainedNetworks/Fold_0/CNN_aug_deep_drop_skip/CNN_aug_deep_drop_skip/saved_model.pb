??
??
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
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8ƚ
?
conv2d_121/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_121/kernel

%conv2d_121/kernel/Read/ReadVariableOpReadVariableOpconv2d_121/kernel*&
_output_shapes
:*
dtype0
v
conv2d_121/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_121/bias
o
#conv2d_121/bias/Read/ReadVariableOpReadVariableOpconv2d_121/bias*
_output_shapes
:*
dtype0
?
conv2d_122/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_122/kernel

%conv2d_122/kernel/Read/ReadVariableOpReadVariableOpconv2d_122/kernel*&
_output_shapes
:*
dtype0
v
conv2d_122/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_122/bias
o
#conv2d_122/bias/Read/ReadVariableOpReadVariableOpconv2d_122/bias*
_output_shapes
:*
dtype0
?
conv2d_123/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_123/kernel

%conv2d_123/kernel/Read/ReadVariableOpReadVariableOpconv2d_123/kernel*&
_output_shapes
:*
dtype0
v
conv2d_123/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_123/bias
o
#conv2d_123/bias/Read/ReadVariableOpReadVariableOpconv2d_123/bias*
_output_shapes
:*
dtype0
?
conv2d_124/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_124/kernel

%conv2d_124/kernel/Read/ReadVariableOpReadVariableOpconv2d_124/kernel*&
_output_shapes
:*
dtype0
v
conv2d_124/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_124/bias
o
#conv2d_124/bias/Read/ReadVariableOpReadVariableOpconv2d_124/bias*
_output_shapes
:*
dtype0
?
conv2d_125/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_125/kernel

%conv2d_125/kernel/Read/ReadVariableOpReadVariableOpconv2d_125/kernel*&
_output_shapes
:*
dtype0
v
conv2d_125/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_125/bias
o
#conv2d_125/bias/Read/ReadVariableOpReadVariableOpconv2d_125/bias*
_output_shapes
:*
dtype0
?
conv2d_126/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_126/kernel

%conv2d_126/kernel/Read/ReadVariableOpReadVariableOpconv2d_126/kernel*&
_output_shapes
:*
dtype0
v
conv2d_126/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_126/bias
o
#conv2d_126/bias/Read/ReadVariableOpReadVariableOpconv2d_126/bias*
_output_shapes
:*
dtype0
?
conv2d_127/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_127/kernel

%conv2d_127/kernel/Read/ReadVariableOpReadVariableOpconv2d_127/kernel*&
_output_shapes
:*
dtype0
v
conv2d_127/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_127/bias
o
#conv2d_127/bias/Read/ReadVariableOpReadVariableOpconv2d_127/bias*
_output_shapes
:*
dtype0
?
conv2d_128/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_128/kernel

%conv2d_128/kernel/Read/ReadVariableOpReadVariableOpconv2d_128/kernel*&
_output_shapes
:*
dtype0
v
conv2d_128/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_128/bias
o
#conv2d_128/bias/Read/ReadVariableOpReadVariableOpconv2d_128/bias*
_output_shapes
:*
dtype0
?
conv2d_129/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_129/kernel

%conv2d_129/kernel/Read/ReadVariableOpReadVariableOpconv2d_129/kernel*&
_output_shapes
:*
dtype0
v
conv2d_129/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_129/bias
o
#conv2d_129/bias/Read/ReadVariableOpReadVariableOpconv2d_129/bias*
_output_shapes
:*
dtype0
z
dense_54/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_54/kernel
s
#dense_54/kernel/Read/ReadVariableOpReadVariableOpdense_54/kernel*
_output_shapes

: *
dtype0
r
dense_54/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_54/bias
k
!dense_54/bias/Read/ReadVariableOpReadVariableOpdense_54/bias*
_output_shapes
: *
dtype0
z
dense_55/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_55/kernel
s
#dense_55/kernel/Read/ReadVariableOpReadVariableOpdense_55/kernel*
_output_shapes

: *
dtype0
r
dense_55/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_55/bias
k
!dense_55/bias/Read/ReadVariableOpReadVariableOpdense_55/bias*
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
Adam/conv2d_121/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_121/kernel/m
?
,Adam/conv2d_121/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_121/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_121/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_121/bias/m
}
*Adam/conv2d_121/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_121/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_122/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_122/kernel/m
?
,Adam/conv2d_122/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_122/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_122/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_122/bias/m
}
*Adam/conv2d_122/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_122/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_123/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_123/kernel/m
?
,Adam/conv2d_123/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_123/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_123/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_123/bias/m
}
*Adam/conv2d_123/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_123/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_124/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_124/kernel/m
?
,Adam/conv2d_124/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_124/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_124/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_124/bias/m
}
*Adam/conv2d_124/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_124/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_125/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_125/kernel/m
?
,Adam/conv2d_125/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_125/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_125/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_125/bias/m
}
*Adam/conv2d_125/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_125/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_126/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_126/kernel/m
?
,Adam/conv2d_126/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_126/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_126/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_126/bias/m
}
*Adam/conv2d_126/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_126/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_127/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_127/kernel/m
?
,Adam/conv2d_127/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_127/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_127/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_127/bias/m
}
*Adam/conv2d_127/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_127/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_128/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_128/kernel/m
?
,Adam/conv2d_128/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_128/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_128/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_128/bias/m
}
*Adam/conv2d_128/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_128/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_129/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_129/kernel/m
?
,Adam/conv2d_129/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_129/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_129/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_129/bias/m
}
*Adam/conv2d_129/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_129/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_54/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_54/kernel/m
?
*Adam/dense_54/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_54/kernel/m*
_output_shapes

: *
dtype0
?
Adam/dense_54/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_54/bias/m
y
(Adam/dense_54/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_54/bias/m*
_output_shapes
: *
dtype0
?
Adam/dense_55/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_55/kernel/m
?
*Adam/dense_55/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_55/kernel/m*
_output_shapes

: *
dtype0
?
Adam/dense_55/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_55/bias/m
y
(Adam/dense_55/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_55/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_121/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_121/kernel/v
?
,Adam/conv2d_121/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_121/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_121/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_121/bias/v
}
*Adam/conv2d_121/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_121/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_122/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_122/kernel/v
?
,Adam/conv2d_122/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_122/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_122/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_122/bias/v
}
*Adam/conv2d_122/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_122/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_123/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_123/kernel/v
?
,Adam/conv2d_123/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_123/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_123/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_123/bias/v
}
*Adam/conv2d_123/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_123/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_124/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_124/kernel/v
?
,Adam/conv2d_124/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_124/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_124/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_124/bias/v
}
*Adam/conv2d_124/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_124/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_125/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_125/kernel/v
?
,Adam/conv2d_125/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_125/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_125/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_125/bias/v
}
*Adam/conv2d_125/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_125/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_126/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_126/kernel/v
?
,Adam/conv2d_126/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_126/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_126/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_126/bias/v
}
*Adam/conv2d_126/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_126/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_127/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_127/kernel/v
?
,Adam/conv2d_127/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_127/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_127/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_127/bias/v
}
*Adam/conv2d_127/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_127/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_128/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_128/kernel/v
?
,Adam/conv2d_128/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_128/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_128/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_128/bias/v
}
*Adam/conv2d_128/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_128/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_129/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_129/kernel/v
?
,Adam/conv2d_129/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_129/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_129/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_129/bias/v
}
*Adam/conv2d_129/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_129/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_54/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_54/kernel/v
?
*Adam/dense_54/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_54/kernel/v*
_output_shapes

: *
dtype0
?
Adam/dense_54/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_54/bias/v
y
(Adam/dense_54/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_54/bias/v*
_output_shapes
: *
dtype0
?
Adam/dense_55/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_55/kernel/v
?
*Adam/dense_55/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_55/kernel/v*
_output_shapes

: *
dtype0
?
Adam/dense_55/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_55/bias/v
y
(Adam/dense_55/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_55/bias/v*
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
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer_with_weights-4
layer-10
layer-11
layer_with_weights-5
layer-12
layer-13
layer-14
layer_with_weights-6
layer-15
layer-16
layer_with_weights-7
layer-17
layer-18
layer_with_weights-8
layer-19
layer-20
layer-21
layer-22
layer_with_weights-9
layer-23
layer_with_weights-10
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
R
&regularization_losses
'trainable_variables
(	variables
)	keras_api
h

*kernel
+bias
,regularization_losses
-trainable_variables
.	variables
/	keras_api
R
0regularization_losses
1trainable_variables
2	variables
3	keras_api
h

4kernel
5bias
6regularization_losses
7trainable_variables
8	variables
9	keras_api
R
:regularization_losses
;trainable_variables
<	variables
=	keras_api
R
>regularization_losses
?trainable_variables
@	variables
A	keras_api
h

Bkernel
Cbias
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
R
Rregularization_losses
Strainable_variables
T	variables
U	keras_api
h

Vkernel
Wbias
Xregularization_losses
Ytrainable_variables
Z	variables
[	keras_api
R
\regularization_losses
]trainable_variables
^	variables
_	keras_api
R
`regularization_losses
atrainable_variables
b	variables
c	keras_api
h

dkernel
ebias
fregularization_losses
gtrainable_variables
h	variables
i	keras_api
R
jregularization_losses
ktrainable_variables
l	variables
m	keras_api
h

nkernel
obias
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
T
~regularization_losses
trainable_variables
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
?
	?iter
?beta_1
?beta_2

?decay
?learning_rate m?!m?*m?+m?4m?5m?Bm?Cm?Lm?Mm?Vm?Wm?dm?em?nm?om?xm?ym?	?m?	?m?	?m?	?m? v?!v?*v?+v?4v?5v?Bv?Cv?Lv?Mv?Vv?Wv?dv?ev?nv?ov?xv?yv?	?v?	?v?	?v?	?v?
 
?
 0
!1
*2
+3
44
55
B6
C7
L8
M9
V10
W11
d12
e13
n14
o15
x16
y17
?18
?19
?20
?21
?
 0
!1
*2
+3
44
55
B6
C7
L8
M9
V10
W11
d12
e13
n14
o15
x16
y17
?18
?19
?20
?21
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
VARIABLE_VALUEconv2d_121/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_121/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
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
 
 
?
&regularization_losses
?layers
?layer_metrics
?metrics
'trainable_variables
(	variables
 ?layer_regularization_losses
?non_trainable_variables
][
VARIABLE_VALUEconv2d_122/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_122/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

*0
+1

*0
+1
?
,regularization_losses
?layers
?layer_metrics
?metrics
-trainable_variables
.	variables
 ?layer_regularization_losses
?non_trainable_variables
 
 
 
?
0regularization_losses
?layers
?layer_metrics
?metrics
1trainable_variables
2	variables
 ?layer_regularization_losses
?non_trainable_variables
][
VARIABLE_VALUEconv2d_123/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_123/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

40
51

40
51
?
6regularization_losses
?layers
?layer_metrics
?metrics
7trainable_variables
8	variables
 ?layer_regularization_losses
?non_trainable_variables
 
 
 
?
:regularization_losses
?layers
?layer_metrics
?metrics
;trainable_variables
<	variables
 ?layer_regularization_losses
?non_trainable_variables
 
 
 
?
>regularization_losses
?layers
?layer_metrics
?metrics
?trainable_variables
@	variables
 ?layer_regularization_losses
?non_trainable_variables
][
VARIABLE_VALUEconv2d_124/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_124/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

B0
C1

B0
C1
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
VARIABLE_VALUEconv2d_125/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_125/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
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
 
 
?
Rregularization_losses
?layers
?layer_metrics
?metrics
Strainable_variables
T	variables
 ?layer_regularization_losses
?non_trainable_variables
][
VARIABLE_VALUEconv2d_126/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_126/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

V0
W1

V0
W1
?
Xregularization_losses
?layers
?layer_metrics
?metrics
Ytrainable_variables
Z	variables
 ?layer_regularization_losses
?non_trainable_variables
 
 
 
?
\regularization_losses
?layers
?layer_metrics
?metrics
]trainable_variables
^	variables
 ?layer_regularization_losses
?non_trainable_variables
 
 
 
?
`regularization_losses
?layers
?layer_metrics
?metrics
atrainable_variables
b	variables
 ?layer_regularization_losses
?non_trainable_variables
][
VARIABLE_VALUEconv2d_127/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_127/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

d0
e1

d0
e1
?
fregularization_losses
?layers
?layer_metrics
?metrics
gtrainable_variables
h	variables
 ?layer_regularization_losses
?non_trainable_variables
 
 
 
?
jregularization_losses
?layers
?layer_metrics
?metrics
ktrainable_variables
l	variables
 ?layer_regularization_losses
?non_trainable_variables
][
VARIABLE_VALUEconv2d_128/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_128/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 

n0
o1

n0
o1
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
?layer_metrics
?metrics
utrainable_variables
v	variables
 ?layer_regularization_losses
?non_trainable_variables
][
VARIABLE_VALUEconv2d_129/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_129/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
 

x0
y1

x0
y1
?
zregularization_losses
?layers
?layer_metrics
?metrics
{trainable_variables
|	variables
 ?layer_regularization_losses
?non_trainable_variables
 
 
 
?
~regularization_losses
?layers
?layer_metrics
?metrics
trainable_variables
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
[Y
VARIABLE_VALUEdense_54/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_54/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEdense_55/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_55/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEAdam/conv2d_121/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_121/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_122/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_122/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_123/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_123/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_124/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_124/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_125/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_125/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_126/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_126/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_127/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_127/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_128/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_128/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_129/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_129/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_54/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_54/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_55/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_55/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_121/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_121/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_122/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_122/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_123/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_123/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_124/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_124/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_125/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_125/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_126/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_126/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_127/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_127/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_128/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_128/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_129/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_129/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_54/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_54/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_55/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_55/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_28Placeholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_28conv2d_121/kernelconv2d_121/biasconv2d_122/kernelconv2d_122/biasconv2d_123/kernelconv2d_123/biasconv2d_124/kernelconv2d_124/biasconv2d_125/kernelconv2d_125/biasconv2d_126/kernelconv2d_126/biasconv2d_127/kernelconv2d_127/biasconv2d_128/kernelconv2d_128/biasconv2d_129/kernelconv2d_129/biasdense_54/kerneldense_54/biasdense_55/kerneldense_55/bias*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_2127448
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%conv2d_121/kernel/Read/ReadVariableOp#conv2d_121/bias/Read/ReadVariableOp%conv2d_122/kernel/Read/ReadVariableOp#conv2d_122/bias/Read/ReadVariableOp%conv2d_123/kernel/Read/ReadVariableOp#conv2d_123/bias/Read/ReadVariableOp%conv2d_124/kernel/Read/ReadVariableOp#conv2d_124/bias/Read/ReadVariableOp%conv2d_125/kernel/Read/ReadVariableOp#conv2d_125/bias/Read/ReadVariableOp%conv2d_126/kernel/Read/ReadVariableOp#conv2d_126/bias/Read/ReadVariableOp%conv2d_127/kernel/Read/ReadVariableOp#conv2d_127/bias/Read/ReadVariableOp%conv2d_128/kernel/Read/ReadVariableOp#conv2d_128/bias/Read/ReadVariableOp%conv2d_129/kernel/Read/ReadVariableOp#conv2d_129/bias/Read/ReadVariableOp#dense_54/kernel/Read/ReadVariableOp!dense_54/bias/Read/ReadVariableOp#dense_55/kernel/Read/ReadVariableOp!dense_55/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp,Adam/conv2d_121/kernel/m/Read/ReadVariableOp*Adam/conv2d_121/bias/m/Read/ReadVariableOp,Adam/conv2d_122/kernel/m/Read/ReadVariableOp*Adam/conv2d_122/bias/m/Read/ReadVariableOp,Adam/conv2d_123/kernel/m/Read/ReadVariableOp*Adam/conv2d_123/bias/m/Read/ReadVariableOp,Adam/conv2d_124/kernel/m/Read/ReadVariableOp*Adam/conv2d_124/bias/m/Read/ReadVariableOp,Adam/conv2d_125/kernel/m/Read/ReadVariableOp*Adam/conv2d_125/bias/m/Read/ReadVariableOp,Adam/conv2d_126/kernel/m/Read/ReadVariableOp*Adam/conv2d_126/bias/m/Read/ReadVariableOp,Adam/conv2d_127/kernel/m/Read/ReadVariableOp*Adam/conv2d_127/bias/m/Read/ReadVariableOp,Adam/conv2d_128/kernel/m/Read/ReadVariableOp*Adam/conv2d_128/bias/m/Read/ReadVariableOp,Adam/conv2d_129/kernel/m/Read/ReadVariableOp*Adam/conv2d_129/bias/m/Read/ReadVariableOp*Adam/dense_54/kernel/m/Read/ReadVariableOp(Adam/dense_54/bias/m/Read/ReadVariableOp*Adam/dense_55/kernel/m/Read/ReadVariableOp(Adam/dense_55/bias/m/Read/ReadVariableOp,Adam/conv2d_121/kernel/v/Read/ReadVariableOp*Adam/conv2d_121/bias/v/Read/ReadVariableOp,Adam/conv2d_122/kernel/v/Read/ReadVariableOp*Adam/conv2d_122/bias/v/Read/ReadVariableOp,Adam/conv2d_123/kernel/v/Read/ReadVariableOp*Adam/conv2d_123/bias/v/Read/ReadVariableOp,Adam/conv2d_124/kernel/v/Read/ReadVariableOp*Adam/conv2d_124/bias/v/Read/ReadVariableOp,Adam/conv2d_125/kernel/v/Read/ReadVariableOp*Adam/conv2d_125/bias/v/Read/ReadVariableOp,Adam/conv2d_126/kernel/v/Read/ReadVariableOp*Adam/conv2d_126/bias/v/Read/ReadVariableOp,Adam/conv2d_127/kernel/v/Read/ReadVariableOp*Adam/conv2d_127/bias/v/Read/ReadVariableOp,Adam/conv2d_128/kernel/v/Read/ReadVariableOp*Adam/conv2d_128/bias/v/Read/ReadVariableOp,Adam/conv2d_129/kernel/v/Read/ReadVariableOp*Adam/conv2d_129/bias/v/Read/ReadVariableOp*Adam/dense_54/kernel/v/Read/ReadVariableOp(Adam/dense_54/bias/v/Read/ReadVariableOp*Adam/dense_55/kernel/v/Read/ReadVariableOp(Adam/dense_55/bias/v/Read/ReadVariableOpConst*X
TinQ
O2M	*
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
 __inference__traced_save_2128455
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_121/kernelconv2d_121/biasconv2d_122/kernelconv2d_122/biasconv2d_123/kernelconv2d_123/biasconv2d_124/kernelconv2d_124/biasconv2d_125/kernelconv2d_125/biasconv2d_126/kernelconv2d_126/biasconv2d_127/kernelconv2d_127/biasconv2d_128/kernelconv2d_128/biasconv2d_129/kernelconv2d_129/biasdense_54/kerneldense_54/biasdense_55/kerneldense_55/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv2d_121/kernel/mAdam/conv2d_121/bias/mAdam/conv2d_122/kernel/mAdam/conv2d_122/bias/mAdam/conv2d_123/kernel/mAdam/conv2d_123/bias/mAdam/conv2d_124/kernel/mAdam/conv2d_124/bias/mAdam/conv2d_125/kernel/mAdam/conv2d_125/bias/mAdam/conv2d_126/kernel/mAdam/conv2d_126/bias/mAdam/conv2d_127/kernel/mAdam/conv2d_127/bias/mAdam/conv2d_128/kernel/mAdam/conv2d_128/bias/mAdam/conv2d_129/kernel/mAdam/conv2d_129/bias/mAdam/dense_54/kernel/mAdam/dense_54/bias/mAdam/dense_55/kernel/mAdam/dense_55/bias/mAdam/conv2d_121/kernel/vAdam/conv2d_121/bias/vAdam/conv2d_122/kernel/vAdam/conv2d_122/bias/vAdam/conv2d_123/kernel/vAdam/conv2d_123/bias/vAdam/conv2d_124/kernel/vAdam/conv2d_124/bias/vAdam/conv2d_125/kernel/vAdam/conv2d_125/bias/vAdam/conv2d_126/kernel/vAdam/conv2d_126/bias/vAdam/conv2d_127/kernel/vAdam/conv2d_127/bias/vAdam/conv2d_128/kernel/vAdam/conv2d_128/bias/vAdam/conv2d_129/kernel/vAdam/conv2d_129/bias/vAdam/dense_54/kernel/vAdam/dense_54/bias/vAdam/dense_55/kernel/vAdam/dense_55/bias/v*W
TinP
N2L*
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
#__inference__traced_restore_2128690??
?

?
G__inference_conv2d_127_layer_call_and_return_conditional_losses_2126859

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
?
e
G__inference_dropout_47_layer_call_and_return_conditional_losses_2127862

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
?
N
2__inference_max_pooling2d_50_layer_call_fn_2126518

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
M__inference_max_pooling2d_50_layer_call_and_return_conditional_losses_21265122
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
?
e
G__inference_dropout_50_layer_call_and_return_conditional_losses_2128067

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
?

?
G__inference_conv2d_123_layer_call_and_return_conditional_losses_2126659

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
?
?
,__inference_conv2d_125_layer_call_fn_2127971

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
G__inference_conv2d_125_layer_call_and_return_conditional_losses_21267592
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
?
f
G__inference_dropout_48_layer_call_and_return_conditional_losses_2127936

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
?
i
M__inference_max_pooling2d_50_layer_call_and_return_conditional_losses_2126512

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
,__inference_conv2d_122_layer_call_fn_2127845

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
G__inference_conv2d_122_layer_call_and_return_conditional_losses_21266022
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
?
H
,__inference_dropout_50_layer_call_fn_2128077

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
G__inference_dropout_50_layer_call_and_return_conditional_losses_21268922
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
?n
?	
S__inference_CNN_aug_deep_drop_skip_layer_call_and_return_conditional_losses_2127074
input_28
conv2d_121_2126556
conv2d_121_2126558
conv2d_122_2126613
conv2d_122_2126615
conv2d_123_2126670
conv2d_123_2126672
conv2d_124_2126713
conv2d_124_2126715
conv2d_125_2126770
conv2d_125_2126772
conv2d_126_2126827
conv2d_126_2126829
conv2d_127_2126870
conv2d_127_2126872
conv2d_128_2126927
conv2d_128_2126929
conv2d_129_2126984
conv2d_129_2126986
dense_54_2127041
dense_54_2127043
dense_55_2127068
dense_55_2127070
identity??"conv2d_121/StatefulPartitionedCall?"conv2d_122/StatefulPartitionedCall?"conv2d_123/StatefulPartitionedCall?"conv2d_124/StatefulPartitionedCall?"conv2d_125/StatefulPartitionedCall?"conv2d_126/StatefulPartitionedCall?"conv2d_127/StatefulPartitionedCall?"conv2d_128/StatefulPartitionedCall?"conv2d_129/StatefulPartitionedCall? dense_54/StatefulPartitionedCall? dense_55/StatefulPartitionedCall?"dropout_46/StatefulPartitionedCall?"dropout_47/StatefulPartitionedCall?"dropout_48/StatefulPartitionedCall?"dropout_49/StatefulPartitionedCall?"dropout_50/StatefulPartitionedCall?"dropout_51/StatefulPartitionedCall?
"conv2d_121/StatefulPartitionedCallStatefulPartitionedCallinput_28conv2d_121_2126556conv2d_121_2126558*
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
G__inference_conv2d_121_layer_call_and_return_conditional_losses_21265452$
"conv2d_121/StatefulPartitionedCall?
"dropout_46/StatefulPartitionedCallStatefulPartitionedCall+conv2d_121/StatefulPartitionedCall:output:0*
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
G__inference_dropout_46_layer_call_and_return_conditional_losses_21265732$
"dropout_46/StatefulPartitionedCall?
"conv2d_122/StatefulPartitionedCallStatefulPartitionedCall+dropout_46/StatefulPartitionedCall:output:0conv2d_122_2126613conv2d_122_2126615*
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
G__inference_conv2d_122_layer_call_and_return_conditional_losses_21266022$
"conv2d_122/StatefulPartitionedCall?
"dropout_47/StatefulPartitionedCallStatefulPartitionedCall+conv2d_122/StatefulPartitionedCall:output:0#^dropout_46/StatefulPartitionedCall*
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
G__inference_dropout_47_layer_call_and_return_conditional_losses_21266302$
"dropout_47/StatefulPartitionedCall?
"conv2d_123/StatefulPartitionedCallStatefulPartitionedCall+dropout_47/StatefulPartitionedCall:output:0conv2d_123_2126670conv2d_123_2126672*
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
G__inference_conv2d_123_layer_call_and_return_conditional_losses_21266592$
"conv2d_123/StatefulPartitionedCall?
add_23/PartitionedCallPartitionedCall+conv2d_123/StatefulPartitionedCall:output:0input_28*
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
C__inference_add_23_layer_call_and_return_conditional_losses_21266812
add_23/PartitionedCall?
 max_pooling2d_49/PartitionedCallPartitionedCalladd_23/PartitionedCall:output:0*
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
M__inference_max_pooling2d_49_layer_call_and_return_conditional_losses_21265002"
 max_pooling2d_49/PartitionedCall?
"conv2d_124/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_49/PartitionedCall:output:0conv2d_124_2126713conv2d_124_2126715*
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
G__inference_conv2d_124_layer_call_and_return_conditional_losses_21267022$
"conv2d_124/StatefulPartitionedCall?
"dropout_48/StatefulPartitionedCallStatefulPartitionedCall+conv2d_124/StatefulPartitionedCall:output:0#^dropout_47/StatefulPartitionedCall*
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
G__inference_dropout_48_layer_call_and_return_conditional_losses_21267302$
"dropout_48/StatefulPartitionedCall?
"conv2d_125/StatefulPartitionedCallStatefulPartitionedCall+dropout_48/StatefulPartitionedCall:output:0conv2d_125_2126770conv2d_125_2126772*
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
G__inference_conv2d_125_layer_call_and_return_conditional_losses_21267592$
"conv2d_125/StatefulPartitionedCall?
"dropout_49/StatefulPartitionedCallStatefulPartitionedCall+conv2d_125/StatefulPartitionedCall:output:0#^dropout_48/StatefulPartitionedCall*
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
G__inference_dropout_49_layer_call_and_return_conditional_losses_21267872$
"dropout_49/StatefulPartitionedCall?
"conv2d_126/StatefulPartitionedCallStatefulPartitionedCall+dropout_49/StatefulPartitionedCall:output:0conv2d_126_2126827conv2d_126_2126829*
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
G__inference_conv2d_126_layer_call_and_return_conditional_losses_21268162$
"conv2d_126/StatefulPartitionedCall?
add_24/PartitionedCallPartitionedCall+conv2d_126/StatefulPartitionedCall:output:0)max_pooling2d_49/PartitionedCall:output:0*
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
C__inference_add_24_layer_call_and_return_conditional_losses_21268382
add_24/PartitionedCall?
 max_pooling2d_50/PartitionedCallPartitionedCalladd_24/PartitionedCall:output:0*
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
M__inference_max_pooling2d_50_layer_call_and_return_conditional_losses_21265122"
 max_pooling2d_50/PartitionedCall?
"conv2d_127/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_50/PartitionedCall:output:0conv2d_127_2126870conv2d_127_2126872*
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
G__inference_conv2d_127_layer_call_and_return_conditional_losses_21268592$
"conv2d_127/StatefulPartitionedCall?
"dropout_50/StatefulPartitionedCallStatefulPartitionedCall+conv2d_127/StatefulPartitionedCall:output:0#^dropout_49/StatefulPartitionedCall*
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
G__inference_dropout_50_layer_call_and_return_conditional_losses_21268872$
"dropout_50/StatefulPartitionedCall?
"conv2d_128/StatefulPartitionedCallStatefulPartitionedCall+dropout_50/StatefulPartitionedCall:output:0conv2d_128_2126927conv2d_128_2126929*
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
G__inference_conv2d_128_layer_call_and_return_conditional_losses_21269162$
"conv2d_128/StatefulPartitionedCall?
"dropout_51/StatefulPartitionedCallStatefulPartitionedCall+conv2d_128/StatefulPartitionedCall:output:0#^dropout_50/StatefulPartitionedCall*
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
G__inference_dropout_51_layer_call_and_return_conditional_losses_21269442$
"dropout_51/StatefulPartitionedCall?
"conv2d_129/StatefulPartitionedCallStatefulPartitionedCall+dropout_51/StatefulPartitionedCall:output:0conv2d_129_2126984conv2d_129_2126986*
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
G__inference_conv2d_129_layer_call_and_return_conditional_losses_21269732$
"conv2d_129/StatefulPartitionedCall?
add_25/PartitionedCallPartitionedCall+conv2d_129/StatefulPartitionedCall:output:0)max_pooling2d_50/PartitionedCall:output:0*
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
C__inference_add_25_layer_call_and_return_conditional_losses_21269952
add_25/PartitionedCall?
 max_pooling2d_51/PartitionedCallPartitionedCalladd_25/PartitionedCall:output:0*
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
M__inference_max_pooling2d_51_layer_call_and_return_conditional_losses_21265242"
 max_pooling2d_51/PartitionedCall?
flatten_27/PartitionedCallPartitionedCall)max_pooling2d_51/PartitionedCall:output:0*
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
G__inference_flatten_27_layer_call_and_return_conditional_losses_21270112
flatten_27/PartitionedCall?
 dense_54/StatefulPartitionedCallStatefulPartitionedCall#flatten_27/PartitionedCall:output:0dense_54_2127041dense_54_2127043*
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
E__inference_dense_54_layer_call_and_return_conditional_losses_21270302"
 dense_54/StatefulPartitionedCall?
 dense_55/StatefulPartitionedCallStatefulPartitionedCall)dense_54/StatefulPartitionedCall:output:0dense_55_2127068dense_55_2127070*
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
E__inference_dense_55_layer_call_and_return_conditional_losses_21270572"
 dense_55/StatefulPartitionedCall?
IdentityIdentity)dense_55/StatefulPartitionedCall:output:0#^conv2d_121/StatefulPartitionedCall#^conv2d_122/StatefulPartitionedCall#^conv2d_123/StatefulPartitionedCall#^conv2d_124/StatefulPartitionedCall#^conv2d_125/StatefulPartitionedCall#^conv2d_126/StatefulPartitionedCall#^conv2d_127/StatefulPartitionedCall#^conv2d_128/StatefulPartitionedCall#^conv2d_129/StatefulPartitionedCall!^dense_54/StatefulPartitionedCall!^dense_55/StatefulPartitionedCall#^dropout_46/StatefulPartitionedCall#^dropout_47/StatefulPartitionedCall#^dropout_48/StatefulPartitionedCall#^dropout_49/StatefulPartitionedCall#^dropout_50/StatefulPartitionedCall#^dropout_51/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesw
u:???????????::::::::::::::::::::::2H
"conv2d_121/StatefulPartitionedCall"conv2d_121/StatefulPartitionedCall2H
"conv2d_122/StatefulPartitionedCall"conv2d_122/StatefulPartitionedCall2H
"conv2d_123/StatefulPartitionedCall"conv2d_123/StatefulPartitionedCall2H
"conv2d_124/StatefulPartitionedCall"conv2d_124/StatefulPartitionedCall2H
"conv2d_125/StatefulPartitionedCall"conv2d_125/StatefulPartitionedCall2H
"conv2d_126/StatefulPartitionedCall"conv2d_126/StatefulPartitionedCall2H
"conv2d_127/StatefulPartitionedCall"conv2d_127/StatefulPartitionedCall2H
"conv2d_128/StatefulPartitionedCall"conv2d_128/StatefulPartitionedCall2H
"conv2d_129/StatefulPartitionedCall"conv2d_129/StatefulPartitionedCall2D
 dense_54/StatefulPartitionedCall dense_54/StatefulPartitionedCall2D
 dense_55/StatefulPartitionedCall dense_55/StatefulPartitionedCall2H
"dropout_46/StatefulPartitionedCall"dropout_46/StatefulPartitionedCall2H
"dropout_47/StatefulPartitionedCall"dropout_47/StatefulPartitionedCall2H
"dropout_48/StatefulPartitionedCall"dropout_48/StatefulPartitionedCall2H
"dropout_49/StatefulPartitionedCall"dropout_49/StatefulPartitionedCall2H
"dropout_50/StatefulPartitionedCall"dropout_50/StatefulPartitionedCall2H
"dropout_51/StatefulPartitionedCall"dropout_51/StatefulPartitionedCall:[ W
1
_output_shapes
:???????????
"
_user_specified_name
input_28
??
?
"__inference__wrapped_model_2126494
input_28D
@cnn_aug_deep_drop_skip_conv2d_121_conv2d_readvariableop_resourceE
Acnn_aug_deep_drop_skip_conv2d_121_biasadd_readvariableop_resourceD
@cnn_aug_deep_drop_skip_conv2d_122_conv2d_readvariableop_resourceE
Acnn_aug_deep_drop_skip_conv2d_122_biasadd_readvariableop_resourceD
@cnn_aug_deep_drop_skip_conv2d_123_conv2d_readvariableop_resourceE
Acnn_aug_deep_drop_skip_conv2d_123_biasadd_readvariableop_resourceD
@cnn_aug_deep_drop_skip_conv2d_124_conv2d_readvariableop_resourceE
Acnn_aug_deep_drop_skip_conv2d_124_biasadd_readvariableop_resourceD
@cnn_aug_deep_drop_skip_conv2d_125_conv2d_readvariableop_resourceE
Acnn_aug_deep_drop_skip_conv2d_125_biasadd_readvariableop_resourceD
@cnn_aug_deep_drop_skip_conv2d_126_conv2d_readvariableop_resourceE
Acnn_aug_deep_drop_skip_conv2d_126_biasadd_readvariableop_resourceD
@cnn_aug_deep_drop_skip_conv2d_127_conv2d_readvariableop_resourceE
Acnn_aug_deep_drop_skip_conv2d_127_biasadd_readvariableop_resourceD
@cnn_aug_deep_drop_skip_conv2d_128_conv2d_readvariableop_resourceE
Acnn_aug_deep_drop_skip_conv2d_128_biasadd_readvariableop_resourceD
@cnn_aug_deep_drop_skip_conv2d_129_conv2d_readvariableop_resourceE
Acnn_aug_deep_drop_skip_conv2d_129_biasadd_readvariableop_resourceB
>cnn_aug_deep_drop_skip_dense_54_matmul_readvariableop_resourceC
?cnn_aug_deep_drop_skip_dense_54_biasadd_readvariableop_resourceB
>cnn_aug_deep_drop_skip_dense_55_matmul_readvariableop_resourceC
?cnn_aug_deep_drop_skip_dense_55_biasadd_readvariableop_resource
identity??8CNN_aug_deep_drop_skip/conv2d_121/BiasAdd/ReadVariableOp?7CNN_aug_deep_drop_skip/conv2d_121/Conv2D/ReadVariableOp?8CNN_aug_deep_drop_skip/conv2d_122/BiasAdd/ReadVariableOp?7CNN_aug_deep_drop_skip/conv2d_122/Conv2D/ReadVariableOp?8CNN_aug_deep_drop_skip/conv2d_123/BiasAdd/ReadVariableOp?7CNN_aug_deep_drop_skip/conv2d_123/Conv2D/ReadVariableOp?8CNN_aug_deep_drop_skip/conv2d_124/BiasAdd/ReadVariableOp?7CNN_aug_deep_drop_skip/conv2d_124/Conv2D/ReadVariableOp?8CNN_aug_deep_drop_skip/conv2d_125/BiasAdd/ReadVariableOp?7CNN_aug_deep_drop_skip/conv2d_125/Conv2D/ReadVariableOp?8CNN_aug_deep_drop_skip/conv2d_126/BiasAdd/ReadVariableOp?7CNN_aug_deep_drop_skip/conv2d_126/Conv2D/ReadVariableOp?8CNN_aug_deep_drop_skip/conv2d_127/BiasAdd/ReadVariableOp?7CNN_aug_deep_drop_skip/conv2d_127/Conv2D/ReadVariableOp?8CNN_aug_deep_drop_skip/conv2d_128/BiasAdd/ReadVariableOp?7CNN_aug_deep_drop_skip/conv2d_128/Conv2D/ReadVariableOp?8CNN_aug_deep_drop_skip/conv2d_129/BiasAdd/ReadVariableOp?7CNN_aug_deep_drop_skip/conv2d_129/Conv2D/ReadVariableOp?6CNN_aug_deep_drop_skip/dense_54/BiasAdd/ReadVariableOp?5CNN_aug_deep_drop_skip/dense_54/MatMul/ReadVariableOp?6CNN_aug_deep_drop_skip/dense_55/BiasAdd/ReadVariableOp?5CNN_aug_deep_drop_skip/dense_55/MatMul/ReadVariableOp?
7CNN_aug_deep_drop_skip/conv2d_121/Conv2D/ReadVariableOpReadVariableOp@cnn_aug_deep_drop_skip_conv2d_121_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype029
7CNN_aug_deep_drop_skip/conv2d_121/Conv2D/ReadVariableOp?
(CNN_aug_deep_drop_skip/conv2d_121/Conv2DConv2Dinput_28?CNN_aug_deep_drop_skip/conv2d_121/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2*
(CNN_aug_deep_drop_skip/conv2d_121/Conv2D?
8CNN_aug_deep_drop_skip/conv2d_121/BiasAdd/ReadVariableOpReadVariableOpAcnn_aug_deep_drop_skip_conv2d_121_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02:
8CNN_aug_deep_drop_skip/conv2d_121/BiasAdd/ReadVariableOp?
)CNN_aug_deep_drop_skip/conv2d_121/BiasAddBiasAdd1CNN_aug_deep_drop_skip/conv2d_121/Conv2D:output:0@CNN_aug_deep_drop_skip/conv2d_121/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2+
)CNN_aug_deep_drop_skip/conv2d_121/BiasAdd?
&CNN_aug_deep_drop_skip/conv2d_121/ReluRelu2CNN_aug_deep_drop_skip/conv2d_121/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2(
&CNN_aug_deep_drop_skip/conv2d_121/Relu?
*CNN_aug_deep_drop_skip/dropout_46/IdentityIdentity4CNN_aug_deep_drop_skip/conv2d_121/Relu:activations:0*
T0*1
_output_shapes
:???????????2,
*CNN_aug_deep_drop_skip/dropout_46/Identity?
7CNN_aug_deep_drop_skip/conv2d_122/Conv2D/ReadVariableOpReadVariableOp@cnn_aug_deep_drop_skip_conv2d_122_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype029
7CNN_aug_deep_drop_skip/conv2d_122/Conv2D/ReadVariableOp?
(CNN_aug_deep_drop_skip/conv2d_122/Conv2DConv2D3CNN_aug_deep_drop_skip/dropout_46/Identity:output:0?CNN_aug_deep_drop_skip/conv2d_122/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2*
(CNN_aug_deep_drop_skip/conv2d_122/Conv2D?
8CNN_aug_deep_drop_skip/conv2d_122/BiasAdd/ReadVariableOpReadVariableOpAcnn_aug_deep_drop_skip_conv2d_122_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02:
8CNN_aug_deep_drop_skip/conv2d_122/BiasAdd/ReadVariableOp?
)CNN_aug_deep_drop_skip/conv2d_122/BiasAddBiasAdd1CNN_aug_deep_drop_skip/conv2d_122/Conv2D:output:0@CNN_aug_deep_drop_skip/conv2d_122/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2+
)CNN_aug_deep_drop_skip/conv2d_122/BiasAdd?
&CNN_aug_deep_drop_skip/conv2d_122/ReluRelu2CNN_aug_deep_drop_skip/conv2d_122/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2(
&CNN_aug_deep_drop_skip/conv2d_122/Relu?
*CNN_aug_deep_drop_skip/dropout_47/IdentityIdentity4CNN_aug_deep_drop_skip/conv2d_122/Relu:activations:0*
T0*1
_output_shapes
:???????????2,
*CNN_aug_deep_drop_skip/dropout_47/Identity?
7CNN_aug_deep_drop_skip/conv2d_123/Conv2D/ReadVariableOpReadVariableOp@cnn_aug_deep_drop_skip_conv2d_123_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype029
7CNN_aug_deep_drop_skip/conv2d_123/Conv2D/ReadVariableOp?
(CNN_aug_deep_drop_skip/conv2d_123/Conv2DConv2D3CNN_aug_deep_drop_skip/dropout_47/Identity:output:0?CNN_aug_deep_drop_skip/conv2d_123/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2*
(CNN_aug_deep_drop_skip/conv2d_123/Conv2D?
8CNN_aug_deep_drop_skip/conv2d_123/BiasAdd/ReadVariableOpReadVariableOpAcnn_aug_deep_drop_skip_conv2d_123_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02:
8CNN_aug_deep_drop_skip/conv2d_123/BiasAdd/ReadVariableOp?
)CNN_aug_deep_drop_skip/conv2d_123/BiasAddBiasAdd1CNN_aug_deep_drop_skip/conv2d_123/Conv2D:output:0@CNN_aug_deep_drop_skip/conv2d_123/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2+
)CNN_aug_deep_drop_skip/conv2d_123/BiasAdd?
&CNN_aug_deep_drop_skip/conv2d_123/ReluRelu2CNN_aug_deep_drop_skip/conv2d_123/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2(
&CNN_aug_deep_drop_skip/conv2d_123/Relu?
!CNN_aug_deep_drop_skip/add_23/addAddV24CNN_aug_deep_drop_skip/conv2d_123/Relu:activations:0input_28*
T0*1
_output_shapes
:???????????2#
!CNN_aug_deep_drop_skip/add_23/add?
/CNN_aug_deep_drop_skip/max_pooling2d_49/MaxPoolMaxPool%CNN_aug_deep_drop_skip/add_23/add:z:0*/
_output_shapes
:?????????***
ksize
*
paddingVALID*
strides
21
/CNN_aug_deep_drop_skip/max_pooling2d_49/MaxPool?
7CNN_aug_deep_drop_skip/conv2d_124/Conv2D/ReadVariableOpReadVariableOp@cnn_aug_deep_drop_skip_conv2d_124_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype029
7CNN_aug_deep_drop_skip/conv2d_124/Conv2D/ReadVariableOp?
(CNN_aug_deep_drop_skip/conv2d_124/Conv2DConv2D8CNN_aug_deep_drop_skip/max_pooling2d_49/MaxPool:output:0?CNN_aug_deep_drop_skip/conv2d_124/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????***
paddingSAME*
strides
2*
(CNN_aug_deep_drop_skip/conv2d_124/Conv2D?
8CNN_aug_deep_drop_skip/conv2d_124/BiasAdd/ReadVariableOpReadVariableOpAcnn_aug_deep_drop_skip_conv2d_124_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02:
8CNN_aug_deep_drop_skip/conv2d_124/BiasAdd/ReadVariableOp?
)CNN_aug_deep_drop_skip/conv2d_124/BiasAddBiasAdd1CNN_aug_deep_drop_skip/conv2d_124/Conv2D:output:0@CNN_aug_deep_drop_skip/conv2d_124/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**2+
)CNN_aug_deep_drop_skip/conv2d_124/BiasAdd?
&CNN_aug_deep_drop_skip/conv2d_124/ReluRelu2CNN_aug_deep_drop_skip/conv2d_124/BiasAdd:output:0*
T0*/
_output_shapes
:?????????**2(
&CNN_aug_deep_drop_skip/conv2d_124/Relu?
*CNN_aug_deep_drop_skip/dropout_48/IdentityIdentity4CNN_aug_deep_drop_skip/conv2d_124/Relu:activations:0*
T0*/
_output_shapes
:?????????**2,
*CNN_aug_deep_drop_skip/dropout_48/Identity?
7CNN_aug_deep_drop_skip/conv2d_125/Conv2D/ReadVariableOpReadVariableOp@cnn_aug_deep_drop_skip_conv2d_125_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype029
7CNN_aug_deep_drop_skip/conv2d_125/Conv2D/ReadVariableOp?
(CNN_aug_deep_drop_skip/conv2d_125/Conv2DConv2D3CNN_aug_deep_drop_skip/dropout_48/Identity:output:0?CNN_aug_deep_drop_skip/conv2d_125/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????***
paddingSAME*
strides
2*
(CNN_aug_deep_drop_skip/conv2d_125/Conv2D?
8CNN_aug_deep_drop_skip/conv2d_125/BiasAdd/ReadVariableOpReadVariableOpAcnn_aug_deep_drop_skip_conv2d_125_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02:
8CNN_aug_deep_drop_skip/conv2d_125/BiasAdd/ReadVariableOp?
)CNN_aug_deep_drop_skip/conv2d_125/BiasAddBiasAdd1CNN_aug_deep_drop_skip/conv2d_125/Conv2D:output:0@CNN_aug_deep_drop_skip/conv2d_125/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**2+
)CNN_aug_deep_drop_skip/conv2d_125/BiasAdd?
&CNN_aug_deep_drop_skip/conv2d_125/ReluRelu2CNN_aug_deep_drop_skip/conv2d_125/BiasAdd:output:0*
T0*/
_output_shapes
:?????????**2(
&CNN_aug_deep_drop_skip/conv2d_125/Relu?
*CNN_aug_deep_drop_skip/dropout_49/IdentityIdentity4CNN_aug_deep_drop_skip/conv2d_125/Relu:activations:0*
T0*/
_output_shapes
:?????????**2,
*CNN_aug_deep_drop_skip/dropout_49/Identity?
7CNN_aug_deep_drop_skip/conv2d_126/Conv2D/ReadVariableOpReadVariableOp@cnn_aug_deep_drop_skip_conv2d_126_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype029
7CNN_aug_deep_drop_skip/conv2d_126/Conv2D/ReadVariableOp?
(CNN_aug_deep_drop_skip/conv2d_126/Conv2DConv2D3CNN_aug_deep_drop_skip/dropout_49/Identity:output:0?CNN_aug_deep_drop_skip/conv2d_126/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????***
paddingSAME*
strides
2*
(CNN_aug_deep_drop_skip/conv2d_126/Conv2D?
8CNN_aug_deep_drop_skip/conv2d_126/BiasAdd/ReadVariableOpReadVariableOpAcnn_aug_deep_drop_skip_conv2d_126_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02:
8CNN_aug_deep_drop_skip/conv2d_126/BiasAdd/ReadVariableOp?
)CNN_aug_deep_drop_skip/conv2d_126/BiasAddBiasAdd1CNN_aug_deep_drop_skip/conv2d_126/Conv2D:output:0@CNN_aug_deep_drop_skip/conv2d_126/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**2+
)CNN_aug_deep_drop_skip/conv2d_126/BiasAdd?
&CNN_aug_deep_drop_skip/conv2d_126/ReluRelu2CNN_aug_deep_drop_skip/conv2d_126/BiasAdd:output:0*
T0*/
_output_shapes
:?????????**2(
&CNN_aug_deep_drop_skip/conv2d_126/Relu?
!CNN_aug_deep_drop_skip/add_24/addAddV24CNN_aug_deep_drop_skip/conv2d_126/Relu:activations:08CNN_aug_deep_drop_skip/max_pooling2d_49/MaxPool:output:0*
T0*/
_output_shapes
:?????????**2#
!CNN_aug_deep_drop_skip/add_24/add?
/CNN_aug_deep_drop_skip/max_pooling2d_50/MaxPoolMaxPool%CNN_aug_deep_drop_skip/add_24/add:z:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
21
/CNN_aug_deep_drop_skip/max_pooling2d_50/MaxPool?
7CNN_aug_deep_drop_skip/conv2d_127/Conv2D/ReadVariableOpReadVariableOp@cnn_aug_deep_drop_skip_conv2d_127_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype029
7CNN_aug_deep_drop_skip/conv2d_127/Conv2D/ReadVariableOp?
(CNN_aug_deep_drop_skip/conv2d_127/Conv2DConv2D8CNN_aug_deep_drop_skip/max_pooling2d_50/MaxPool:output:0?CNN_aug_deep_drop_skip/conv2d_127/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2*
(CNN_aug_deep_drop_skip/conv2d_127/Conv2D?
8CNN_aug_deep_drop_skip/conv2d_127/BiasAdd/ReadVariableOpReadVariableOpAcnn_aug_deep_drop_skip_conv2d_127_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02:
8CNN_aug_deep_drop_skip/conv2d_127/BiasAdd/ReadVariableOp?
)CNN_aug_deep_drop_skip/conv2d_127/BiasAddBiasAdd1CNN_aug_deep_drop_skip/conv2d_127/Conv2D:output:0@CNN_aug_deep_drop_skip/conv2d_127/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2+
)CNN_aug_deep_drop_skip/conv2d_127/BiasAdd?
&CNN_aug_deep_drop_skip/conv2d_127/ReluRelu2CNN_aug_deep_drop_skip/conv2d_127/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2(
&CNN_aug_deep_drop_skip/conv2d_127/Relu?
*CNN_aug_deep_drop_skip/dropout_50/IdentityIdentity4CNN_aug_deep_drop_skip/conv2d_127/Relu:activations:0*
T0*/
_output_shapes
:?????????2,
*CNN_aug_deep_drop_skip/dropout_50/Identity?
7CNN_aug_deep_drop_skip/conv2d_128/Conv2D/ReadVariableOpReadVariableOp@cnn_aug_deep_drop_skip_conv2d_128_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype029
7CNN_aug_deep_drop_skip/conv2d_128/Conv2D/ReadVariableOp?
(CNN_aug_deep_drop_skip/conv2d_128/Conv2DConv2D3CNN_aug_deep_drop_skip/dropout_50/Identity:output:0?CNN_aug_deep_drop_skip/conv2d_128/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2*
(CNN_aug_deep_drop_skip/conv2d_128/Conv2D?
8CNN_aug_deep_drop_skip/conv2d_128/BiasAdd/ReadVariableOpReadVariableOpAcnn_aug_deep_drop_skip_conv2d_128_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02:
8CNN_aug_deep_drop_skip/conv2d_128/BiasAdd/ReadVariableOp?
)CNN_aug_deep_drop_skip/conv2d_128/BiasAddBiasAdd1CNN_aug_deep_drop_skip/conv2d_128/Conv2D:output:0@CNN_aug_deep_drop_skip/conv2d_128/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2+
)CNN_aug_deep_drop_skip/conv2d_128/BiasAdd?
&CNN_aug_deep_drop_skip/conv2d_128/ReluRelu2CNN_aug_deep_drop_skip/conv2d_128/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2(
&CNN_aug_deep_drop_skip/conv2d_128/Relu?
*CNN_aug_deep_drop_skip/dropout_51/IdentityIdentity4CNN_aug_deep_drop_skip/conv2d_128/Relu:activations:0*
T0*/
_output_shapes
:?????????2,
*CNN_aug_deep_drop_skip/dropout_51/Identity?
7CNN_aug_deep_drop_skip/conv2d_129/Conv2D/ReadVariableOpReadVariableOp@cnn_aug_deep_drop_skip_conv2d_129_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype029
7CNN_aug_deep_drop_skip/conv2d_129/Conv2D/ReadVariableOp?
(CNN_aug_deep_drop_skip/conv2d_129/Conv2DConv2D3CNN_aug_deep_drop_skip/dropout_51/Identity:output:0?CNN_aug_deep_drop_skip/conv2d_129/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2*
(CNN_aug_deep_drop_skip/conv2d_129/Conv2D?
8CNN_aug_deep_drop_skip/conv2d_129/BiasAdd/ReadVariableOpReadVariableOpAcnn_aug_deep_drop_skip_conv2d_129_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02:
8CNN_aug_deep_drop_skip/conv2d_129/BiasAdd/ReadVariableOp?
)CNN_aug_deep_drop_skip/conv2d_129/BiasAddBiasAdd1CNN_aug_deep_drop_skip/conv2d_129/Conv2D:output:0@CNN_aug_deep_drop_skip/conv2d_129/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2+
)CNN_aug_deep_drop_skip/conv2d_129/BiasAdd?
&CNN_aug_deep_drop_skip/conv2d_129/ReluRelu2CNN_aug_deep_drop_skip/conv2d_129/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2(
&CNN_aug_deep_drop_skip/conv2d_129/Relu?
!CNN_aug_deep_drop_skip/add_25/addAddV24CNN_aug_deep_drop_skip/conv2d_129/Relu:activations:08CNN_aug_deep_drop_skip/max_pooling2d_50/MaxPool:output:0*
T0*/
_output_shapes
:?????????2#
!CNN_aug_deep_drop_skip/add_25/add?
/CNN_aug_deep_drop_skip/max_pooling2d_51/MaxPoolMaxPool%CNN_aug_deep_drop_skip/add_25/add:z:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
21
/CNN_aug_deep_drop_skip/max_pooling2d_51/MaxPool?
'CNN_aug_deep_drop_skip/flatten_27/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2)
'CNN_aug_deep_drop_skip/flatten_27/Const?
)CNN_aug_deep_drop_skip/flatten_27/ReshapeReshape8CNN_aug_deep_drop_skip/max_pooling2d_51/MaxPool:output:00CNN_aug_deep_drop_skip/flatten_27/Const:output:0*
T0*'
_output_shapes
:?????????2+
)CNN_aug_deep_drop_skip/flatten_27/Reshape?
5CNN_aug_deep_drop_skip/dense_54/MatMul/ReadVariableOpReadVariableOp>cnn_aug_deep_drop_skip_dense_54_matmul_readvariableop_resource*
_output_shapes

: *
dtype027
5CNN_aug_deep_drop_skip/dense_54/MatMul/ReadVariableOp?
&CNN_aug_deep_drop_skip/dense_54/MatMulMatMul2CNN_aug_deep_drop_skip/flatten_27/Reshape:output:0=CNN_aug_deep_drop_skip/dense_54/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2(
&CNN_aug_deep_drop_skip/dense_54/MatMul?
6CNN_aug_deep_drop_skip/dense_54/BiasAdd/ReadVariableOpReadVariableOp?cnn_aug_deep_drop_skip_dense_54_biasadd_readvariableop_resource*
_output_shapes
: *
dtype028
6CNN_aug_deep_drop_skip/dense_54/BiasAdd/ReadVariableOp?
'CNN_aug_deep_drop_skip/dense_54/BiasAddBiasAdd0CNN_aug_deep_drop_skip/dense_54/MatMul:product:0>CNN_aug_deep_drop_skip/dense_54/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2)
'CNN_aug_deep_drop_skip/dense_54/BiasAdd?
$CNN_aug_deep_drop_skip/dense_54/ReluRelu0CNN_aug_deep_drop_skip/dense_54/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2&
$CNN_aug_deep_drop_skip/dense_54/Relu?
5CNN_aug_deep_drop_skip/dense_55/MatMul/ReadVariableOpReadVariableOp>cnn_aug_deep_drop_skip_dense_55_matmul_readvariableop_resource*
_output_shapes

: *
dtype027
5CNN_aug_deep_drop_skip/dense_55/MatMul/ReadVariableOp?
&CNN_aug_deep_drop_skip/dense_55/MatMulMatMul2CNN_aug_deep_drop_skip/dense_54/Relu:activations:0=CNN_aug_deep_drop_skip/dense_55/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2(
&CNN_aug_deep_drop_skip/dense_55/MatMul?
6CNN_aug_deep_drop_skip/dense_55/BiasAdd/ReadVariableOpReadVariableOp?cnn_aug_deep_drop_skip_dense_55_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
6CNN_aug_deep_drop_skip/dense_55/BiasAdd/ReadVariableOp?
'CNN_aug_deep_drop_skip/dense_55/BiasAddBiasAdd0CNN_aug_deep_drop_skip/dense_55/MatMul:product:0>CNN_aug_deep_drop_skip/dense_55/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2)
'CNN_aug_deep_drop_skip/dense_55/BiasAdd?
'CNN_aug_deep_drop_skip/dense_55/SoftmaxSoftmax0CNN_aug_deep_drop_skip/dense_55/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2)
'CNN_aug_deep_drop_skip/dense_55/Softmax?
IdentityIdentity1CNN_aug_deep_drop_skip/dense_55/Softmax:softmax:09^CNN_aug_deep_drop_skip/conv2d_121/BiasAdd/ReadVariableOp8^CNN_aug_deep_drop_skip/conv2d_121/Conv2D/ReadVariableOp9^CNN_aug_deep_drop_skip/conv2d_122/BiasAdd/ReadVariableOp8^CNN_aug_deep_drop_skip/conv2d_122/Conv2D/ReadVariableOp9^CNN_aug_deep_drop_skip/conv2d_123/BiasAdd/ReadVariableOp8^CNN_aug_deep_drop_skip/conv2d_123/Conv2D/ReadVariableOp9^CNN_aug_deep_drop_skip/conv2d_124/BiasAdd/ReadVariableOp8^CNN_aug_deep_drop_skip/conv2d_124/Conv2D/ReadVariableOp9^CNN_aug_deep_drop_skip/conv2d_125/BiasAdd/ReadVariableOp8^CNN_aug_deep_drop_skip/conv2d_125/Conv2D/ReadVariableOp9^CNN_aug_deep_drop_skip/conv2d_126/BiasAdd/ReadVariableOp8^CNN_aug_deep_drop_skip/conv2d_126/Conv2D/ReadVariableOp9^CNN_aug_deep_drop_skip/conv2d_127/BiasAdd/ReadVariableOp8^CNN_aug_deep_drop_skip/conv2d_127/Conv2D/ReadVariableOp9^CNN_aug_deep_drop_skip/conv2d_128/BiasAdd/ReadVariableOp8^CNN_aug_deep_drop_skip/conv2d_128/Conv2D/ReadVariableOp9^CNN_aug_deep_drop_skip/conv2d_129/BiasAdd/ReadVariableOp8^CNN_aug_deep_drop_skip/conv2d_129/Conv2D/ReadVariableOp7^CNN_aug_deep_drop_skip/dense_54/BiasAdd/ReadVariableOp6^CNN_aug_deep_drop_skip/dense_54/MatMul/ReadVariableOp7^CNN_aug_deep_drop_skip/dense_55/BiasAdd/ReadVariableOp6^CNN_aug_deep_drop_skip/dense_55/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesw
u:???????????::::::::::::::::::::::2t
8CNN_aug_deep_drop_skip/conv2d_121/BiasAdd/ReadVariableOp8CNN_aug_deep_drop_skip/conv2d_121/BiasAdd/ReadVariableOp2r
7CNN_aug_deep_drop_skip/conv2d_121/Conv2D/ReadVariableOp7CNN_aug_deep_drop_skip/conv2d_121/Conv2D/ReadVariableOp2t
8CNN_aug_deep_drop_skip/conv2d_122/BiasAdd/ReadVariableOp8CNN_aug_deep_drop_skip/conv2d_122/BiasAdd/ReadVariableOp2r
7CNN_aug_deep_drop_skip/conv2d_122/Conv2D/ReadVariableOp7CNN_aug_deep_drop_skip/conv2d_122/Conv2D/ReadVariableOp2t
8CNN_aug_deep_drop_skip/conv2d_123/BiasAdd/ReadVariableOp8CNN_aug_deep_drop_skip/conv2d_123/BiasAdd/ReadVariableOp2r
7CNN_aug_deep_drop_skip/conv2d_123/Conv2D/ReadVariableOp7CNN_aug_deep_drop_skip/conv2d_123/Conv2D/ReadVariableOp2t
8CNN_aug_deep_drop_skip/conv2d_124/BiasAdd/ReadVariableOp8CNN_aug_deep_drop_skip/conv2d_124/BiasAdd/ReadVariableOp2r
7CNN_aug_deep_drop_skip/conv2d_124/Conv2D/ReadVariableOp7CNN_aug_deep_drop_skip/conv2d_124/Conv2D/ReadVariableOp2t
8CNN_aug_deep_drop_skip/conv2d_125/BiasAdd/ReadVariableOp8CNN_aug_deep_drop_skip/conv2d_125/BiasAdd/ReadVariableOp2r
7CNN_aug_deep_drop_skip/conv2d_125/Conv2D/ReadVariableOp7CNN_aug_deep_drop_skip/conv2d_125/Conv2D/ReadVariableOp2t
8CNN_aug_deep_drop_skip/conv2d_126/BiasAdd/ReadVariableOp8CNN_aug_deep_drop_skip/conv2d_126/BiasAdd/ReadVariableOp2r
7CNN_aug_deep_drop_skip/conv2d_126/Conv2D/ReadVariableOp7CNN_aug_deep_drop_skip/conv2d_126/Conv2D/ReadVariableOp2t
8CNN_aug_deep_drop_skip/conv2d_127/BiasAdd/ReadVariableOp8CNN_aug_deep_drop_skip/conv2d_127/BiasAdd/ReadVariableOp2r
7CNN_aug_deep_drop_skip/conv2d_127/Conv2D/ReadVariableOp7CNN_aug_deep_drop_skip/conv2d_127/Conv2D/ReadVariableOp2t
8CNN_aug_deep_drop_skip/conv2d_128/BiasAdd/ReadVariableOp8CNN_aug_deep_drop_skip/conv2d_128/BiasAdd/ReadVariableOp2r
7CNN_aug_deep_drop_skip/conv2d_128/Conv2D/ReadVariableOp7CNN_aug_deep_drop_skip/conv2d_128/Conv2D/ReadVariableOp2t
8CNN_aug_deep_drop_skip/conv2d_129/BiasAdd/ReadVariableOp8CNN_aug_deep_drop_skip/conv2d_129/BiasAdd/ReadVariableOp2r
7CNN_aug_deep_drop_skip/conv2d_129/Conv2D/ReadVariableOp7CNN_aug_deep_drop_skip/conv2d_129/Conv2D/ReadVariableOp2p
6CNN_aug_deep_drop_skip/dense_54/BiasAdd/ReadVariableOp6CNN_aug_deep_drop_skip/dense_54/BiasAdd/ReadVariableOp2n
5CNN_aug_deep_drop_skip/dense_54/MatMul/ReadVariableOp5CNN_aug_deep_drop_skip/dense_54/MatMul/ReadVariableOp2p
6CNN_aug_deep_drop_skip/dense_55/BiasAdd/ReadVariableOp6CNN_aug_deep_drop_skip/dense_55/BiasAdd/ReadVariableOp2n
5CNN_aug_deep_drop_skip/dense_55/MatMul/ReadVariableOp5CNN_aug_deep_drop_skip/dense_55/MatMul/ReadVariableOp:[ W
1
_output_shapes
:???????????
"
_user_specified_name
input_28
?
e
G__inference_dropout_49_layer_call_and_return_conditional_losses_2126792

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
?
E__inference_dense_55_layer_call_and_return_conditional_losses_2127057

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
G__inference_dropout_47_layer_call_and_return_conditional_losses_2127857

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

?
G__inference_conv2d_126_layer_call_and_return_conditional_losses_2126816

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
?

?
G__inference_conv2d_125_layer_call_and_return_conditional_losses_2127962

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
G__inference_dropout_48_layer_call_and_return_conditional_losses_2127941

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
?
T
(__inference_add_25_layer_call_fn_2128156
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
C__inference_add_25_layer_call_and_return_conditional_losses_21269952
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
f
G__inference_dropout_47_layer_call_and_return_conditional_losses_2126630

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

?
G__inference_conv2d_126_layer_call_and_return_conditional_losses_2128009

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
?
e
,__inference_dropout_46_layer_call_fn_2127820

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
G__inference_dropout_46_layer_call_and_return_conditional_losses_21265732
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
?

?
G__inference_conv2d_121_layer_call_and_return_conditional_losses_2126545

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
?
H
,__inference_dropout_46_layer_call_fn_2127825

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
G__inference_dropout_46_layer_call_and_return_conditional_losses_21265782
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
?
?
,__inference_conv2d_126_layer_call_fn_2128018

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
G__inference_conv2d_126_layer_call_and_return_conditional_losses_21268162
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
?
N
2__inference_max_pooling2d_49_layer_call_fn_2126506

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
M__inference_max_pooling2d_49_layer_call_and_return_conditional_losses_21265002
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
?
e
G__inference_dropout_50_layer_call_and_return_conditional_losses_2126892

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
?
e
,__inference_dropout_51_layer_call_fn_2128119

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
G__inference_dropout_51_layer_call_and_return_conditional_losses_21269442
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
e
,__inference_dropout_50_layer_call_fn_2128072

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
G__inference_dropout_50_layer_call_and_return_conditional_losses_21268872
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
2__inference_max_pooling2d_51_layer_call_fn_2126530

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
M__inference_max_pooling2d_51_layer_call_and_return_conditional_losses_21265242
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
f
G__inference_dropout_46_layer_call_and_return_conditional_losses_2127810

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
?
?
,__inference_conv2d_123_layer_call_fn_2127892

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
G__inference_conv2d_123_layer_call_and_return_conditional_losses_21266592
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
?
?
%__inference_signature_wrapper_2127448
input_28
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

unknown_20
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_28unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__wrapped_model_21264942
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesw
u:???????????::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:???????????
"
_user_specified_name
input_28
?
?
8__inference_CNN_aug_deep_drop_skip_layer_call_fn_2127778

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

unknown_20
identity??StatefulPartitionedCall?
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
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_CNN_aug_deep_drop_skip_layer_call_and_return_conditional_losses_21273422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesw
u:???????????::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
e
G__inference_dropout_46_layer_call_and_return_conditional_losses_2126578

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
G__inference_conv2d_129_layer_call_and_return_conditional_losses_2126973

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
?
e
G__inference_dropout_51_layer_call_and_return_conditional_losses_2128114

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
?

?
G__inference_conv2d_122_layer_call_and_return_conditional_losses_2127836

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
?
?
,__inference_conv2d_124_layer_call_fn_2127924

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
G__inference_conv2d_124_layer_call_and_return_conditional_losses_21267022
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
?
o
C__inference_add_25_layer_call_and_return_conditional_losses_2128150
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
?d
?
S__inference_CNN_aug_deep_drop_skip_layer_call_and_return_conditional_losses_2127342

inputs
conv2d_121_2127273
conv2d_121_2127275
conv2d_122_2127279
conv2d_122_2127281
conv2d_123_2127285
conv2d_123_2127287
conv2d_124_2127292
conv2d_124_2127294
conv2d_125_2127298
conv2d_125_2127300
conv2d_126_2127304
conv2d_126_2127306
conv2d_127_2127311
conv2d_127_2127313
conv2d_128_2127317
conv2d_128_2127319
conv2d_129_2127323
conv2d_129_2127325
dense_54_2127331
dense_54_2127333
dense_55_2127336
dense_55_2127338
identity??"conv2d_121/StatefulPartitionedCall?"conv2d_122/StatefulPartitionedCall?"conv2d_123/StatefulPartitionedCall?"conv2d_124/StatefulPartitionedCall?"conv2d_125/StatefulPartitionedCall?"conv2d_126/StatefulPartitionedCall?"conv2d_127/StatefulPartitionedCall?"conv2d_128/StatefulPartitionedCall?"conv2d_129/StatefulPartitionedCall? dense_54/StatefulPartitionedCall? dense_55/StatefulPartitionedCall?
"conv2d_121/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_121_2127273conv2d_121_2127275*
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
G__inference_conv2d_121_layer_call_and_return_conditional_losses_21265452$
"conv2d_121/StatefulPartitionedCall?
dropout_46/PartitionedCallPartitionedCall+conv2d_121/StatefulPartitionedCall:output:0*
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
G__inference_dropout_46_layer_call_and_return_conditional_losses_21265782
dropout_46/PartitionedCall?
"conv2d_122/StatefulPartitionedCallStatefulPartitionedCall#dropout_46/PartitionedCall:output:0conv2d_122_2127279conv2d_122_2127281*
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
G__inference_conv2d_122_layer_call_and_return_conditional_losses_21266022$
"conv2d_122/StatefulPartitionedCall?
dropout_47/PartitionedCallPartitionedCall+conv2d_122/StatefulPartitionedCall:output:0*
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
G__inference_dropout_47_layer_call_and_return_conditional_losses_21266352
dropout_47/PartitionedCall?
"conv2d_123/StatefulPartitionedCallStatefulPartitionedCall#dropout_47/PartitionedCall:output:0conv2d_123_2127285conv2d_123_2127287*
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
G__inference_conv2d_123_layer_call_and_return_conditional_losses_21266592$
"conv2d_123/StatefulPartitionedCall?
add_23/PartitionedCallPartitionedCall+conv2d_123/StatefulPartitionedCall:output:0inputs*
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
C__inference_add_23_layer_call_and_return_conditional_losses_21266812
add_23/PartitionedCall?
 max_pooling2d_49/PartitionedCallPartitionedCalladd_23/PartitionedCall:output:0*
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
M__inference_max_pooling2d_49_layer_call_and_return_conditional_losses_21265002"
 max_pooling2d_49/PartitionedCall?
"conv2d_124/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_49/PartitionedCall:output:0conv2d_124_2127292conv2d_124_2127294*
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
G__inference_conv2d_124_layer_call_and_return_conditional_losses_21267022$
"conv2d_124/StatefulPartitionedCall?
dropout_48/PartitionedCallPartitionedCall+conv2d_124/StatefulPartitionedCall:output:0*
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
G__inference_dropout_48_layer_call_and_return_conditional_losses_21267352
dropout_48/PartitionedCall?
"conv2d_125/StatefulPartitionedCallStatefulPartitionedCall#dropout_48/PartitionedCall:output:0conv2d_125_2127298conv2d_125_2127300*
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
G__inference_conv2d_125_layer_call_and_return_conditional_losses_21267592$
"conv2d_125/StatefulPartitionedCall?
dropout_49/PartitionedCallPartitionedCall+conv2d_125/StatefulPartitionedCall:output:0*
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
G__inference_dropout_49_layer_call_and_return_conditional_losses_21267922
dropout_49/PartitionedCall?
"conv2d_126/StatefulPartitionedCallStatefulPartitionedCall#dropout_49/PartitionedCall:output:0conv2d_126_2127304conv2d_126_2127306*
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
G__inference_conv2d_126_layer_call_and_return_conditional_losses_21268162$
"conv2d_126/StatefulPartitionedCall?
add_24/PartitionedCallPartitionedCall+conv2d_126/StatefulPartitionedCall:output:0)max_pooling2d_49/PartitionedCall:output:0*
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
C__inference_add_24_layer_call_and_return_conditional_losses_21268382
add_24/PartitionedCall?
 max_pooling2d_50/PartitionedCallPartitionedCalladd_24/PartitionedCall:output:0*
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
M__inference_max_pooling2d_50_layer_call_and_return_conditional_losses_21265122"
 max_pooling2d_50/PartitionedCall?
"conv2d_127/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_50/PartitionedCall:output:0conv2d_127_2127311conv2d_127_2127313*
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
G__inference_conv2d_127_layer_call_and_return_conditional_losses_21268592$
"conv2d_127/StatefulPartitionedCall?
dropout_50/PartitionedCallPartitionedCall+conv2d_127/StatefulPartitionedCall:output:0*
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
G__inference_dropout_50_layer_call_and_return_conditional_losses_21268922
dropout_50/PartitionedCall?
"conv2d_128/StatefulPartitionedCallStatefulPartitionedCall#dropout_50/PartitionedCall:output:0conv2d_128_2127317conv2d_128_2127319*
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
G__inference_conv2d_128_layer_call_and_return_conditional_losses_21269162$
"conv2d_128/StatefulPartitionedCall?
dropout_51/PartitionedCallPartitionedCall+conv2d_128/StatefulPartitionedCall:output:0*
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
G__inference_dropout_51_layer_call_and_return_conditional_losses_21269492
dropout_51/PartitionedCall?
"conv2d_129/StatefulPartitionedCallStatefulPartitionedCall#dropout_51/PartitionedCall:output:0conv2d_129_2127323conv2d_129_2127325*
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
G__inference_conv2d_129_layer_call_and_return_conditional_losses_21269732$
"conv2d_129/StatefulPartitionedCall?
add_25/PartitionedCallPartitionedCall+conv2d_129/StatefulPartitionedCall:output:0)max_pooling2d_50/PartitionedCall:output:0*
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
C__inference_add_25_layer_call_and_return_conditional_losses_21269952
add_25/PartitionedCall?
 max_pooling2d_51/PartitionedCallPartitionedCalladd_25/PartitionedCall:output:0*
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
M__inference_max_pooling2d_51_layer_call_and_return_conditional_losses_21265242"
 max_pooling2d_51/PartitionedCall?
flatten_27/PartitionedCallPartitionedCall)max_pooling2d_51/PartitionedCall:output:0*
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
G__inference_flatten_27_layer_call_and_return_conditional_losses_21270112
flatten_27/PartitionedCall?
 dense_54/StatefulPartitionedCallStatefulPartitionedCall#flatten_27/PartitionedCall:output:0dense_54_2127331dense_54_2127333*
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
E__inference_dense_54_layer_call_and_return_conditional_losses_21270302"
 dense_54/StatefulPartitionedCall?
 dense_55/StatefulPartitionedCallStatefulPartitionedCall)dense_54/StatefulPartitionedCall:output:0dense_55_2127336dense_55_2127338*
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
E__inference_dense_55_layer_call_and_return_conditional_losses_21270572"
 dense_55/StatefulPartitionedCall?
IdentityIdentity)dense_55/StatefulPartitionedCall:output:0#^conv2d_121/StatefulPartitionedCall#^conv2d_122/StatefulPartitionedCall#^conv2d_123/StatefulPartitionedCall#^conv2d_124/StatefulPartitionedCall#^conv2d_125/StatefulPartitionedCall#^conv2d_126/StatefulPartitionedCall#^conv2d_127/StatefulPartitionedCall#^conv2d_128/StatefulPartitionedCall#^conv2d_129/StatefulPartitionedCall!^dense_54/StatefulPartitionedCall!^dense_55/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesw
u:???????????::::::::::::::::::::::2H
"conv2d_121/StatefulPartitionedCall"conv2d_121/StatefulPartitionedCall2H
"conv2d_122/StatefulPartitionedCall"conv2d_122/StatefulPartitionedCall2H
"conv2d_123/StatefulPartitionedCall"conv2d_123/StatefulPartitionedCall2H
"conv2d_124/StatefulPartitionedCall"conv2d_124/StatefulPartitionedCall2H
"conv2d_125/StatefulPartitionedCall"conv2d_125/StatefulPartitionedCall2H
"conv2d_126/StatefulPartitionedCall"conv2d_126/StatefulPartitionedCall2H
"conv2d_127/StatefulPartitionedCall"conv2d_127/StatefulPartitionedCall2H
"conv2d_128/StatefulPartitionedCall"conv2d_128/StatefulPartitionedCall2H
"conv2d_129/StatefulPartitionedCall"conv2d_129/StatefulPartitionedCall2D
 dense_54/StatefulPartitionedCall dense_54/StatefulPartitionedCall2D
 dense_55/StatefulPartitionedCall dense_55/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?d
?
S__inference_CNN_aug_deep_drop_skip_layer_call_and_return_conditional_losses_2127146
input_28
conv2d_121_2127077
conv2d_121_2127079
conv2d_122_2127083
conv2d_122_2127085
conv2d_123_2127089
conv2d_123_2127091
conv2d_124_2127096
conv2d_124_2127098
conv2d_125_2127102
conv2d_125_2127104
conv2d_126_2127108
conv2d_126_2127110
conv2d_127_2127115
conv2d_127_2127117
conv2d_128_2127121
conv2d_128_2127123
conv2d_129_2127127
conv2d_129_2127129
dense_54_2127135
dense_54_2127137
dense_55_2127140
dense_55_2127142
identity??"conv2d_121/StatefulPartitionedCall?"conv2d_122/StatefulPartitionedCall?"conv2d_123/StatefulPartitionedCall?"conv2d_124/StatefulPartitionedCall?"conv2d_125/StatefulPartitionedCall?"conv2d_126/StatefulPartitionedCall?"conv2d_127/StatefulPartitionedCall?"conv2d_128/StatefulPartitionedCall?"conv2d_129/StatefulPartitionedCall? dense_54/StatefulPartitionedCall? dense_55/StatefulPartitionedCall?
"conv2d_121/StatefulPartitionedCallStatefulPartitionedCallinput_28conv2d_121_2127077conv2d_121_2127079*
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
G__inference_conv2d_121_layer_call_and_return_conditional_losses_21265452$
"conv2d_121/StatefulPartitionedCall?
dropout_46/PartitionedCallPartitionedCall+conv2d_121/StatefulPartitionedCall:output:0*
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
G__inference_dropout_46_layer_call_and_return_conditional_losses_21265782
dropout_46/PartitionedCall?
"conv2d_122/StatefulPartitionedCallStatefulPartitionedCall#dropout_46/PartitionedCall:output:0conv2d_122_2127083conv2d_122_2127085*
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
G__inference_conv2d_122_layer_call_and_return_conditional_losses_21266022$
"conv2d_122/StatefulPartitionedCall?
dropout_47/PartitionedCallPartitionedCall+conv2d_122/StatefulPartitionedCall:output:0*
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
G__inference_dropout_47_layer_call_and_return_conditional_losses_21266352
dropout_47/PartitionedCall?
"conv2d_123/StatefulPartitionedCallStatefulPartitionedCall#dropout_47/PartitionedCall:output:0conv2d_123_2127089conv2d_123_2127091*
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
G__inference_conv2d_123_layer_call_and_return_conditional_losses_21266592$
"conv2d_123/StatefulPartitionedCall?
add_23/PartitionedCallPartitionedCall+conv2d_123/StatefulPartitionedCall:output:0input_28*
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
C__inference_add_23_layer_call_and_return_conditional_losses_21266812
add_23/PartitionedCall?
 max_pooling2d_49/PartitionedCallPartitionedCalladd_23/PartitionedCall:output:0*
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
M__inference_max_pooling2d_49_layer_call_and_return_conditional_losses_21265002"
 max_pooling2d_49/PartitionedCall?
"conv2d_124/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_49/PartitionedCall:output:0conv2d_124_2127096conv2d_124_2127098*
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
G__inference_conv2d_124_layer_call_and_return_conditional_losses_21267022$
"conv2d_124/StatefulPartitionedCall?
dropout_48/PartitionedCallPartitionedCall+conv2d_124/StatefulPartitionedCall:output:0*
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
G__inference_dropout_48_layer_call_and_return_conditional_losses_21267352
dropout_48/PartitionedCall?
"conv2d_125/StatefulPartitionedCallStatefulPartitionedCall#dropout_48/PartitionedCall:output:0conv2d_125_2127102conv2d_125_2127104*
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
G__inference_conv2d_125_layer_call_and_return_conditional_losses_21267592$
"conv2d_125/StatefulPartitionedCall?
dropout_49/PartitionedCallPartitionedCall+conv2d_125/StatefulPartitionedCall:output:0*
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
G__inference_dropout_49_layer_call_and_return_conditional_losses_21267922
dropout_49/PartitionedCall?
"conv2d_126/StatefulPartitionedCallStatefulPartitionedCall#dropout_49/PartitionedCall:output:0conv2d_126_2127108conv2d_126_2127110*
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
G__inference_conv2d_126_layer_call_and_return_conditional_losses_21268162$
"conv2d_126/StatefulPartitionedCall?
add_24/PartitionedCallPartitionedCall+conv2d_126/StatefulPartitionedCall:output:0)max_pooling2d_49/PartitionedCall:output:0*
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
C__inference_add_24_layer_call_and_return_conditional_losses_21268382
add_24/PartitionedCall?
 max_pooling2d_50/PartitionedCallPartitionedCalladd_24/PartitionedCall:output:0*
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
M__inference_max_pooling2d_50_layer_call_and_return_conditional_losses_21265122"
 max_pooling2d_50/PartitionedCall?
"conv2d_127/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_50/PartitionedCall:output:0conv2d_127_2127115conv2d_127_2127117*
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
G__inference_conv2d_127_layer_call_and_return_conditional_losses_21268592$
"conv2d_127/StatefulPartitionedCall?
dropout_50/PartitionedCallPartitionedCall+conv2d_127/StatefulPartitionedCall:output:0*
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
G__inference_dropout_50_layer_call_and_return_conditional_losses_21268922
dropout_50/PartitionedCall?
"conv2d_128/StatefulPartitionedCallStatefulPartitionedCall#dropout_50/PartitionedCall:output:0conv2d_128_2127121conv2d_128_2127123*
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
G__inference_conv2d_128_layer_call_and_return_conditional_losses_21269162$
"conv2d_128/StatefulPartitionedCall?
dropout_51/PartitionedCallPartitionedCall+conv2d_128/StatefulPartitionedCall:output:0*
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
G__inference_dropout_51_layer_call_and_return_conditional_losses_21269492
dropout_51/PartitionedCall?
"conv2d_129/StatefulPartitionedCallStatefulPartitionedCall#dropout_51/PartitionedCall:output:0conv2d_129_2127127conv2d_129_2127129*
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
G__inference_conv2d_129_layer_call_and_return_conditional_losses_21269732$
"conv2d_129/StatefulPartitionedCall?
add_25/PartitionedCallPartitionedCall+conv2d_129/StatefulPartitionedCall:output:0)max_pooling2d_50/PartitionedCall:output:0*
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
C__inference_add_25_layer_call_and_return_conditional_losses_21269952
add_25/PartitionedCall?
 max_pooling2d_51/PartitionedCallPartitionedCalladd_25/PartitionedCall:output:0*
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
M__inference_max_pooling2d_51_layer_call_and_return_conditional_losses_21265242"
 max_pooling2d_51/PartitionedCall?
flatten_27/PartitionedCallPartitionedCall)max_pooling2d_51/PartitionedCall:output:0*
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
G__inference_flatten_27_layer_call_and_return_conditional_losses_21270112
flatten_27/PartitionedCall?
 dense_54/StatefulPartitionedCallStatefulPartitionedCall#flatten_27/PartitionedCall:output:0dense_54_2127135dense_54_2127137*
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
E__inference_dense_54_layer_call_and_return_conditional_losses_21270302"
 dense_54/StatefulPartitionedCall?
 dense_55/StatefulPartitionedCallStatefulPartitionedCall)dense_54/StatefulPartitionedCall:output:0dense_55_2127140dense_55_2127142*
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
E__inference_dense_55_layer_call_and_return_conditional_losses_21270572"
 dense_55/StatefulPartitionedCall?
IdentityIdentity)dense_55/StatefulPartitionedCall:output:0#^conv2d_121/StatefulPartitionedCall#^conv2d_122/StatefulPartitionedCall#^conv2d_123/StatefulPartitionedCall#^conv2d_124/StatefulPartitionedCall#^conv2d_125/StatefulPartitionedCall#^conv2d_126/StatefulPartitionedCall#^conv2d_127/StatefulPartitionedCall#^conv2d_128/StatefulPartitionedCall#^conv2d_129/StatefulPartitionedCall!^dense_54/StatefulPartitionedCall!^dense_55/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesw
u:???????????::::::::::::::::::::::2H
"conv2d_121/StatefulPartitionedCall"conv2d_121/StatefulPartitionedCall2H
"conv2d_122/StatefulPartitionedCall"conv2d_122/StatefulPartitionedCall2H
"conv2d_123/StatefulPartitionedCall"conv2d_123/StatefulPartitionedCall2H
"conv2d_124/StatefulPartitionedCall"conv2d_124/StatefulPartitionedCall2H
"conv2d_125/StatefulPartitionedCall"conv2d_125/StatefulPartitionedCall2H
"conv2d_126/StatefulPartitionedCall"conv2d_126/StatefulPartitionedCall2H
"conv2d_127/StatefulPartitionedCall"conv2d_127/StatefulPartitionedCall2H
"conv2d_128/StatefulPartitionedCall"conv2d_128/StatefulPartitionedCall2H
"conv2d_129/StatefulPartitionedCall"conv2d_129/StatefulPartitionedCall2D
 dense_54/StatefulPartitionedCall dense_54/StatefulPartitionedCall2D
 dense_55/StatefulPartitionedCall dense_55/StatefulPartitionedCall:[ W
1
_output_shapes
:???????????
"
_user_specified_name
input_28
?
c
G__inference_flatten_27_layer_call_and_return_conditional_losses_2127011

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
f
G__inference_dropout_50_layer_call_and_return_conditional_losses_2126887

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
?
H
,__inference_dropout_47_layer_call_fn_2127872

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
G__inference_dropout_47_layer_call_and_return_conditional_losses_21266352
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
??
?
S__inference_CNN_aug_deep_drop_skip_layer_call_and_return_conditional_losses_2127585

inputs-
)conv2d_121_conv2d_readvariableop_resource.
*conv2d_121_biasadd_readvariableop_resource-
)conv2d_122_conv2d_readvariableop_resource.
*conv2d_122_biasadd_readvariableop_resource-
)conv2d_123_conv2d_readvariableop_resource.
*conv2d_123_biasadd_readvariableop_resource-
)conv2d_124_conv2d_readvariableop_resource.
*conv2d_124_biasadd_readvariableop_resource-
)conv2d_125_conv2d_readvariableop_resource.
*conv2d_125_biasadd_readvariableop_resource-
)conv2d_126_conv2d_readvariableop_resource.
*conv2d_126_biasadd_readvariableop_resource-
)conv2d_127_conv2d_readvariableop_resource.
*conv2d_127_biasadd_readvariableop_resource-
)conv2d_128_conv2d_readvariableop_resource.
*conv2d_128_biasadd_readvariableop_resource-
)conv2d_129_conv2d_readvariableop_resource.
*conv2d_129_biasadd_readvariableop_resource+
'dense_54_matmul_readvariableop_resource,
(dense_54_biasadd_readvariableop_resource+
'dense_55_matmul_readvariableop_resource,
(dense_55_biasadd_readvariableop_resource
identity??!conv2d_121/BiasAdd/ReadVariableOp? conv2d_121/Conv2D/ReadVariableOp?!conv2d_122/BiasAdd/ReadVariableOp? conv2d_122/Conv2D/ReadVariableOp?!conv2d_123/BiasAdd/ReadVariableOp? conv2d_123/Conv2D/ReadVariableOp?!conv2d_124/BiasAdd/ReadVariableOp? conv2d_124/Conv2D/ReadVariableOp?!conv2d_125/BiasAdd/ReadVariableOp? conv2d_125/Conv2D/ReadVariableOp?!conv2d_126/BiasAdd/ReadVariableOp? conv2d_126/Conv2D/ReadVariableOp?!conv2d_127/BiasAdd/ReadVariableOp? conv2d_127/Conv2D/ReadVariableOp?!conv2d_128/BiasAdd/ReadVariableOp? conv2d_128/Conv2D/ReadVariableOp?!conv2d_129/BiasAdd/ReadVariableOp? conv2d_129/Conv2D/ReadVariableOp?dense_54/BiasAdd/ReadVariableOp?dense_54/MatMul/ReadVariableOp?dense_55/BiasAdd/ReadVariableOp?dense_55/MatMul/ReadVariableOp?
 conv2d_121/Conv2D/ReadVariableOpReadVariableOp)conv2d_121_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_121/Conv2D/ReadVariableOp?
conv2d_121/Conv2DConv2Dinputs(conv2d_121/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_121/Conv2D?
!conv2d_121/BiasAdd/ReadVariableOpReadVariableOp*conv2d_121_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_121/BiasAdd/ReadVariableOp?
conv2d_121/BiasAddBiasAddconv2d_121/Conv2D:output:0)conv2d_121/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_121/BiasAdd?
conv2d_121/ReluReluconv2d_121/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_121/Reluy
dropout_46/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_46/dropout/Const?
dropout_46/dropout/MulMulconv2d_121/Relu:activations:0!dropout_46/dropout/Const:output:0*
T0*1
_output_shapes
:???????????2
dropout_46/dropout/Mul?
dropout_46/dropout/ShapeShapeconv2d_121/Relu:activations:0*
T0*
_output_shapes
:2
dropout_46/dropout/Shape?
/dropout_46/dropout/random_uniform/RandomUniformRandomUniform!dropout_46/dropout/Shape:output:0*
T0*1
_output_shapes
:???????????*
dtype021
/dropout_46/dropout/random_uniform/RandomUniform?
!dropout_46/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2#
!dropout_46/dropout/GreaterEqual/y?
dropout_46/dropout/GreaterEqualGreaterEqual8dropout_46/dropout/random_uniform/RandomUniform:output:0*dropout_46/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:???????????2!
dropout_46/dropout/GreaterEqual?
dropout_46/dropout/CastCast#dropout_46/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:???????????2
dropout_46/dropout/Cast?
dropout_46/dropout/Mul_1Muldropout_46/dropout/Mul:z:0dropout_46/dropout/Cast:y:0*
T0*1
_output_shapes
:???????????2
dropout_46/dropout/Mul_1?
 conv2d_122/Conv2D/ReadVariableOpReadVariableOp)conv2d_122_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_122/Conv2D/ReadVariableOp?
conv2d_122/Conv2DConv2Ddropout_46/dropout/Mul_1:z:0(conv2d_122/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_122/Conv2D?
!conv2d_122/BiasAdd/ReadVariableOpReadVariableOp*conv2d_122_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_122/BiasAdd/ReadVariableOp?
conv2d_122/BiasAddBiasAddconv2d_122/Conv2D:output:0)conv2d_122/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_122/BiasAdd?
conv2d_122/ReluReluconv2d_122/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_122/Reluy
dropout_47/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_47/dropout/Const?
dropout_47/dropout/MulMulconv2d_122/Relu:activations:0!dropout_47/dropout/Const:output:0*
T0*1
_output_shapes
:???????????2
dropout_47/dropout/Mul?
dropout_47/dropout/ShapeShapeconv2d_122/Relu:activations:0*
T0*
_output_shapes
:2
dropout_47/dropout/Shape?
/dropout_47/dropout/random_uniform/RandomUniformRandomUniform!dropout_47/dropout/Shape:output:0*
T0*1
_output_shapes
:???????????*
dtype021
/dropout_47/dropout/random_uniform/RandomUniform?
!dropout_47/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2#
!dropout_47/dropout/GreaterEqual/y?
dropout_47/dropout/GreaterEqualGreaterEqual8dropout_47/dropout/random_uniform/RandomUniform:output:0*dropout_47/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:???????????2!
dropout_47/dropout/GreaterEqual?
dropout_47/dropout/CastCast#dropout_47/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:???????????2
dropout_47/dropout/Cast?
dropout_47/dropout/Mul_1Muldropout_47/dropout/Mul:z:0dropout_47/dropout/Cast:y:0*
T0*1
_output_shapes
:???????????2
dropout_47/dropout/Mul_1?
 conv2d_123/Conv2D/ReadVariableOpReadVariableOp)conv2d_123_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_123/Conv2D/ReadVariableOp?
conv2d_123/Conv2DConv2Ddropout_47/dropout/Mul_1:z:0(conv2d_123/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_123/Conv2D?
!conv2d_123/BiasAdd/ReadVariableOpReadVariableOp*conv2d_123_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_123/BiasAdd/ReadVariableOp?
conv2d_123/BiasAddBiasAddconv2d_123/Conv2D:output:0)conv2d_123/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_123/BiasAdd?
conv2d_123/ReluReluconv2d_123/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_123/Relu?

add_23/addAddV2conv2d_123/Relu:activations:0inputs*
T0*1
_output_shapes
:???????????2

add_23/add?
max_pooling2d_49/MaxPoolMaxPooladd_23/add:z:0*/
_output_shapes
:?????????***
ksize
*
paddingVALID*
strides
2
max_pooling2d_49/MaxPool?
 conv2d_124/Conv2D/ReadVariableOpReadVariableOp)conv2d_124_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_124/Conv2D/ReadVariableOp?
conv2d_124/Conv2DConv2D!max_pooling2d_49/MaxPool:output:0(conv2d_124/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????***
paddingSAME*
strides
2
conv2d_124/Conv2D?
!conv2d_124/BiasAdd/ReadVariableOpReadVariableOp*conv2d_124_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_124/BiasAdd/ReadVariableOp?
conv2d_124/BiasAddBiasAddconv2d_124/Conv2D:output:0)conv2d_124/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**2
conv2d_124/BiasAdd?
conv2d_124/ReluReluconv2d_124/BiasAdd:output:0*
T0*/
_output_shapes
:?????????**2
conv2d_124/Reluy
dropout_48/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_48/dropout/Const?
dropout_48/dropout/MulMulconv2d_124/Relu:activations:0!dropout_48/dropout/Const:output:0*
T0*/
_output_shapes
:?????????**2
dropout_48/dropout/Mul?
dropout_48/dropout/ShapeShapeconv2d_124/Relu:activations:0*
T0*
_output_shapes
:2
dropout_48/dropout/Shape?
/dropout_48/dropout/random_uniform/RandomUniformRandomUniform!dropout_48/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????***
dtype021
/dropout_48/dropout/random_uniform/RandomUniform?
!dropout_48/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2#
!dropout_48/dropout/GreaterEqual/y?
dropout_48/dropout/GreaterEqualGreaterEqual8dropout_48/dropout/random_uniform/RandomUniform:output:0*dropout_48/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????**2!
dropout_48/dropout/GreaterEqual?
dropout_48/dropout/CastCast#dropout_48/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????**2
dropout_48/dropout/Cast?
dropout_48/dropout/Mul_1Muldropout_48/dropout/Mul:z:0dropout_48/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????**2
dropout_48/dropout/Mul_1?
 conv2d_125/Conv2D/ReadVariableOpReadVariableOp)conv2d_125_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_125/Conv2D/ReadVariableOp?
conv2d_125/Conv2DConv2Ddropout_48/dropout/Mul_1:z:0(conv2d_125/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????***
paddingSAME*
strides
2
conv2d_125/Conv2D?
!conv2d_125/BiasAdd/ReadVariableOpReadVariableOp*conv2d_125_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_125/BiasAdd/ReadVariableOp?
conv2d_125/BiasAddBiasAddconv2d_125/Conv2D:output:0)conv2d_125/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**2
conv2d_125/BiasAdd?
conv2d_125/ReluReluconv2d_125/BiasAdd:output:0*
T0*/
_output_shapes
:?????????**2
conv2d_125/Reluy
dropout_49/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_49/dropout/Const?
dropout_49/dropout/MulMulconv2d_125/Relu:activations:0!dropout_49/dropout/Const:output:0*
T0*/
_output_shapes
:?????????**2
dropout_49/dropout/Mul?
dropout_49/dropout/ShapeShapeconv2d_125/Relu:activations:0*
T0*
_output_shapes
:2
dropout_49/dropout/Shape?
/dropout_49/dropout/random_uniform/RandomUniformRandomUniform!dropout_49/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????***
dtype021
/dropout_49/dropout/random_uniform/RandomUniform?
!dropout_49/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2#
!dropout_49/dropout/GreaterEqual/y?
dropout_49/dropout/GreaterEqualGreaterEqual8dropout_49/dropout/random_uniform/RandomUniform:output:0*dropout_49/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????**2!
dropout_49/dropout/GreaterEqual?
dropout_49/dropout/CastCast#dropout_49/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????**2
dropout_49/dropout/Cast?
dropout_49/dropout/Mul_1Muldropout_49/dropout/Mul:z:0dropout_49/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????**2
dropout_49/dropout/Mul_1?
 conv2d_126/Conv2D/ReadVariableOpReadVariableOp)conv2d_126_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_126/Conv2D/ReadVariableOp?
conv2d_126/Conv2DConv2Ddropout_49/dropout/Mul_1:z:0(conv2d_126/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????***
paddingSAME*
strides
2
conv2d_126/Conv2D?
!conv2d_126/BiasAdd/ReadVariableOpReadVariableOp*conv2d_126_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_126/BiasAdd/ReadVariableOp?
conv2d_126/BiasAddBiasAddconv2d_126/Conv2D:output:0)conv2d_126/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**2
conv2d_126/BiasAdd?
conv2d_126/ReluReluconv2d_126/BiasAdd:output:0*
T0*/
_output_shapes
:?????????**2
conv2d_126/Relu?

add_24/addAddV2conv2d_126/Relu:activations:0!max_pooling2d_49/MaxPool:output:0*
T0*/
_output_shapes
:?????????**2

add_24/add?
max_pooling2d_50/MaxPoolMaxPooladd_24/add:z:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_50/MaxPool?
 conv2d_127/Conv2D/ReadVariableOpReadVariableOp)conv2d_127_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_127/Conv2D/ReadVariableOp?
conv2d_127/Conv2DConv2D!max_pooling2d_50/MaxPool:output:0(conv2d_127/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_127/Conv2D?
!conv2d_127/BiasAdd/ReadVariableOpReadVariableOp*conv2d_127_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_127/BiasAdd/ReadVariableOp?
conv2d_127/BiasAddBiasAddconv2d_127/Conv2D:output:0)conv2d_127/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_127/BiasAdd?
conv2d_127/ReluReluconv2d_127/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_127/Reluy
dropout_50/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_50/dropout/Const?
dropout_50/dropout/MulMulconv2d_127/Relu:activations:0!dropout_50/dropout/Const:output:0*
T0*/
_output_shapes
:?????????2
dropout_50/dropout/Mul?
dropout_50/dropout/ShapeShapeconv2d_127/Relu:activations:0*
T0*
_output_shapes
:2
dropout_50/dropout/Shape?
/dropout_50/dropout/random_uniform/RandomUniformRandomUniform!dropout_50/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype021
/dropout_50/dropout/random_uniform/RandomUniform?
!dropout_50/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2#
!dropout_50/dropout/GreaterEqual/y?
dropout_50/dropout/GreaterEqualGreaterEqual8dropout_50/dropout/random_uniform/RandomUniform:output:0*dropout_50/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????2!
dropout_50/dropout/GreaterEqual?
dropout_50/dropout/CastCast#dropout_50/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout_50/dropout/Cast?
dropout_50/dropout/Mul_1Muldropout_50/dropout/Mul:z:0dropout_50/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout_50/dropout/Mul_1?
 conv2d_128/Conv2D/ReadVariableOpReadVariableOp)conv2d_128_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_128/Conv2D/ReadVariableOp?
conv2d_128/Conv2DConv2Ddropout_50/dropout/Mul_1:z:0(conv2d_128/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_128/Conv2D?
!conv2d_128/BiasAdd/ReadVariableOpReadVariableOp*conv2d_128_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_128/BiasAdd/ReadVariableOp?
conv2d_128/BiasAddBiasAddconv2d_128/Conv2D:output:0)conv2d_128/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_128/BiasAdd?
conv2d_128/ReluReluconv2d_128/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_128/Reluy
dropout_51/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_51/dropout/Const?
dropout_51/dropout/MulMulconv2d_128/Relu:activations:0!dropout_51/dropout/Const:output:0*
T0*/
_output_shapes
:?????????2
dropout_51/dropout/Mul?
dropout_51/dropout/ShapeShapeconv2d_128/Relu:activations:0*
T0*
_output_shapes
:2
dropout_51/dropout/Shape?
/dropout_51/dropout/random_uniform/RandomUniformRandomUniform!dropout_51/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype021
/dropout_51/dropout/random_uniform/RandomUniform?
!dropout_51/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2#
!dropout_51/dropout/GreaterEqual/y?
dropout_51/dropout/GreaterEqualGreaterEqual8dropout_51/dropout/random_uniform/RandomUniform:output:0*dropout_51/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????2!
dropout_51/dropout/GreaterEqual?
dropout_51/dropout/CastCast#dropout_51/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout_51/dropout/Cast?
dropout_51/dropout/Mul_1Muldropout_51/dropout/Mul:z:0dropout_51/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout_51/dropout/Mul_1?
 conv2d_129/Conv2D/ReadVariableOpReadVariableOp)conv2d_129_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_129/Conv2D/ReadVariableOp?
conv2d_129/Conv2DConv2Ddropout_51/dropout/Mul_1:z:0(conv2d_129/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_129/Conv2D?
!conv2d_129/BiasAdd/ReadVariableOpReadVariableOp*conv2d_129_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_129/BiasAdd/ReadVariableOp?
conv2d_129/BiasAddBiasAddconv2d_129/Conv2D:output:0)conv2d_129/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_129/BiasAdd?
conv2d_129/ReluReluconv2d_129/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_129/Relu?

add_25/addAddV2conv2d_129/Relu:activations:0!max_pooling2d_50/MaxPool:output:0*
T0*/
_output_shapes
:?????????2

add_25/add?
max_pooling2d_51/MaxPoolMaxPooladd_25/add:z:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_51/MaxPoolu
flatten_27/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_27/Const?
flatten_27/ReshapeReshape!max_pooling2d_51/MaxPool:output:0flatten_27/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_27/Reshape?
dense_54/MatMul/ReadVariableOpReadVariableOp'dense_54_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_54/MatMul/ReadVariableOp?
dense_54/MatMulMatMulflatten_27/Reshape:output:0&dense_54/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_54/MatMul?
dense_54/BiasAdd/ReadVariableOpReadVariableOp(dense_54_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_54/BiasAdd/ReadVariableOp?
dense_54/BiasAddBiasAdddense_54/MatMul:product:0'dense_54/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_54/BiasAdds
dense_54/ReluReludense_54/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_54/Relu?
dense_55/MatMul/ReadVariableOpReadVariableOp'dense_55_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_55/MatMul/ReadVariableOp?
dense_55/MatMulMatMuldense_54/Relu:activations:0&dense_55/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_55/MatMul?
dense_55/BiasAdd/ReadVariableOpReadVariableOp(dense_55_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_55/BiasAdd/ReadVariableOp?
dense_55/BiasAddBiasAdddense_55/MatMul:product:0'dense_55/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_55/BiasAdd|
dense_55/SoftmaxSoftmaxdense_55/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_55/Softmax?
IdentityIdentitydense_55/Softmax:softmax:0"^conv2d_121/BiasAdd/ReadVariableOp!^conv2d_121/Conv2D/ReadVariableOp"^conv2d_122/BiasAdd/ReadVariableOp!^conv2d_122/Conv2D/ReadVariableOp"^conv2d_123/BiasAdd/ReadVariableOp!^conv2d_123/Conv2D/ReadVariableOp"^conv2d_124/BiasAdd/ReadVariableOp!^conv2d_124/Conv2D/ReadVariableOp"^conv2d_125/BiasAdd/ReadVariableOp!^conv2d_125/Conv2D/ReadVariableOp"^conv2d_126/BiasAdd/ReadVariableOp!^conv2d_126/Conv2D/ReadVariableOp"^conv2d_127/BiasAdd/ReadVariableOp!^conv2d_127/Conv2D/ReadVariableOp"^conv2d_128/BiasAdd/ReadVariableOp!^conv2d_128/Conv2D/ReadVariableOp"^conv2d_129/BiasAdd/ReadVariableOp!^conv2d_129/Conv2D/ReadVariableOp ^dense_54/BiasAdd/ReadVariableOp^dense_54/MatMul/ReadVariableOp ^dense_55/BiasAdd/ReadVariableOp^dense_55/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesw
u:???????????::::::::::::::::::::::2F
!conv2d_121/BiasAdd/ReadVariableOp!conv2d_121/BiasAdd/ReadVariableOp2D
 conv2d_121/Conv2D/ReadVariableOp conv2d_121/Conv2D/ReadVariableOp2F
!conv2d_122/BiasAdd/ReadVariableOp!conv2d_122/BiasAdd/ReadVariableOp2D
 conv2d_122/Conv2D/ReadVariableOp conv2d_122/Conv2D/ReadVariableOp2F
!conv2d_123/BiasAdd/ReadVariableOp!conv2d_123/BiasAdd/ReadVariableOp2D
 conv2d_123/Conv2D/ReadVariableOp conv2d_123/Conv2D/ReadVariableOp2F
!conv2d_124/BiasAdd/ReadVariableOp!conv2d_124/BiasAdd/ReadVariableOp2D
 conv2d_124/Conv2D/ReadVariableOp conv2d_124/Conv2D/ReadVariableOp2F
!conv2d_125/BiasAdd/ReadVariableOp!conv2d_125/BiasAdd/ReadVariableOp2D
 conv2d_125/Conv2D/ReadVariableOp conv2d_125/Conv2D/ReadVariableOp2F
!conv2d_126/BiasAdd/ReadVariableOp!conv2d_126/BiasAdd/ReadVariableOp2D
 conv2d_126/Conv2D/ReadVariableOp conv2d_126/Conv2D/ReadVariableOp2F
!conv2d_127/BiasAdd/ReadVariableOp!conv2d_127/BiasAdd/ReadVariableOp2D
 conv2d_127/Conv2D/ReadVariableOp conv2d_127/Conv2D/ReadVariableOp2F
!conv2d_128/BiasAdd/ReadVariableOp!conv2d_128/BiasAdd/ReadVariableOp2D
 conv2d_128/Conv2D/ReadVariableOp conv2d_128/Conv2D/ReadVariableOp2F
!conv2d_129/BiasAdd/ReadVariableOp!conv2d_129/BiasAdd/ReadVariableOp2D
 conv2d_129/Conv2D/ReadVariableOp conv2d_129/Conv2D/ReadVariableOp2B
dense_54/BiasAdd/ReadVariableOpdense_54/BiasAdd/ReadVariableOp2@
dense_54/MatMul/ReadVariableOpdense_54/MatMul/ReadVariableOp2B
dense_55/BiasAdd/ReadVariableOpdense_55/BiasAdd/ReadVariableOp2@
dense_55/MatMul/ReadVariableOpdense_55/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
??
?'
#__inference__traced_restore_2128690
file_prefix&
"assignvariableop_conv2d_121_kernel&
"assignvariableop_1_conv2d_121_bias(
$assignvariableop_2_conv2d_122_kernel&
"assignvariableop_3_conv2d_122_bias(
$assignvariableop_4_conv2d_123_kernel&
"assignvariableop_5_conv2d_123_bias(
$assignvariableop_6_conv2d_124_kernel&
"assignvariableop_7_conv2d_124_bias(
$assignvariableop_8_conv2d_125_kernel&
"assignvariableop_9_conv2d_125_bias)
%assignvariableop_10_conv2d_126_kernel'
#assignvariableop_11_conv2d_126_bias)
%assignvariableop_12_conv2d_127_kernel'
#assignvariableop_13_conv2d_127_bias)
%assignvariableop_14_conv2d_128_kernel'
#assignvariableop_15_conv2d_128_bias)
%assignvariableop_16_conv2d_129_kernel'
#assignvariableop_17_conv2d_129_bias'
#assignvariableop_18_dense_54_kernel%
!assignvariableop_19_dense_54_bias'
#assignvariableop_20_dense_55_kernel%
!assignvariableop_21_dense_55_bias!
assignvariableop_22_adam_iter#
assignvariableop_23_adam_beta_1#
assignvariableop_24_adam_beta_2"
assignvariableop_25_adam_decay*
&assignvariableop_26_adam_learning_rate
assignvariableop_27_total
assignvariableop_28_count
assignvariableop_29_total_1
assignvariableop_30_count_10
,assignvariableop_31_adam_conv2d_121_kernel_m.
*assignvariableop_32_adam_conv2d_121_bias_m0
,assignvariableop_33_adam_conv2d_122_kernel_m.
*assignvariableop_34_adam_conv2d_122_bias_m0
,assignvariableop_35_adam_conv2d_123_kernel_m.
*assignvariableop_36_adam_conv2d_123_bias_m0
,assignvariableop_37_adam_conv2d_124_kernel_m.
*assignvariableop_38_adam_conv2d_124_bias_m0
,assignvariableop_39_adam_conv2d_125_kernel_m.
*assignvariableop_40_adam_conv2d_125_bias_m0
,assignvariableop_41_adam_conv2d_126_kernel_m.
*assignvariableop_42_adam_conv2d_126_bias_m0
,assignvariableop_43_adam_conv2d_127_kernel_m.
*assignvariableop_44_adam_conv2d_127_bias_m0
,assignvariableop_45_adam_conv2d_128_kernel_m.
*assignvariableop_46_adam_conv2d_128_bias_m0
,assignvariableop_47_adam_conv2d_129_kernel_m.
*assignvariableop_48_adam_conv2d_129_bias_m.
*assignvariableop_49_adam_dense_54_kernel_m,
(assignvariableop_50_adam_dense_54_bias_m.
*assignvariableop_51_adam_dense_55_kernel_m,
(assignvariableop_52_adam_dense_55_bias_m0
,assignvariableop_53_adam_conv2d_121_kernel_v.
*assignvariableop_54_adam_conv2d_121_bias_v0
,assignvariableop_55_adam_conv2d_122_kernel_v.
*assignvariableop_56_adam_conv2d_122_bias_v0
,assignvariableop_57_adam_conv2d_123_kernel_v.
*assignvariableop_58_adam_conv2d_123_bias_v0
,assignvariableop_59_adam_conv2d_124_kernel_v.
*assignvariableop_60_adam_conv2d_124_bias_v0
,assignvariableop_61_adam_conv2d_125_kernel_v.
*assignvariableop_62_adam_conv2d_125_bias_v0
,assignvariableop_63_adam_conv2d_126_kernel_v.
*assignvariableop_64_adam_conv2d_126_bias_v0
,assignvariableop_65_adam_conv2d_127_kernel_v.
*assignvariableop_66_adam_conv2d_127_bias_v0
,assignvariableop_67_adam_conv2d_128_kernel_v.
*assignvariableop_68_adam_conv2d_128_bias_v0
,assignvariableop_69_adam_conv2d_129_kernel_v.
*assignvariableop_70_adam_conv2d_129_bias_v.
*assignvariableop_71_adam_dense_54_kernel_v,
(assignvariableop_72_adam_dense_54_bias_v.
*assignvariableop_73_adam_dense_55_kernel_v,
(assignvariableop_74_adam_dense_55_bias_v
identity_76??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_8?AssignVariableOp_9?*
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*?)
value?)B?)LB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*?
value?B?LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*Z
dtypesP
N2L	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp"assignvariableop_conv2d_121_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp"assignvariableop_1_conv2d_121_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp$assignvariableop_2_conv2d_122_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp"assignvariableop_3_conv2d_122_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp$assignvariableop_4_conv2d_123_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp"assignvariableop_5_conv2d_123_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp$assignvariableop_6_conv2d_124_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp"assignvariableop_7_conv2d_124_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp$assignvariableop_8_conv2d_125_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp"assignvariableop_9_conv2d_125_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp%assignvariableop_10_conv2d_126_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp#assignvariableop_11_conv2d_126_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp%assignvariableop_12_conv2d_127_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp#assignvariableop_13_conv2d_127_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp%assignvariableop_14_conv2d_128_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp#assignvariableop_15_conv2d_128_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp%assignvariableop_16_conv2d_129_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp#assignvariableop_17_conv2d_129_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp#assignvariableop_18_dense_54_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp!assignvariableop_19_dense_54_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp#assignvariableop_20_dense_55_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp!assignvariableop_21_dense_55_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpassignvariableop_22_adam_iterIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOpassignvariableop_23_adam_beta_1Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOpassignvariableop_24_adam_beta_2Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOpassignvariableop_25_adam_decayIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp&assignvariableop_26_adam_learning_rateIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOpassignvariableop_27_totalIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOpassignvariableop_28_countIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOpassignvariableop_29_total_1Identity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOpassignvariableop_30_count_1Identity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp,assignvariableop_31_adam_conv2d_121_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp*assignvariableop_32_adam_conv2d_121_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp,assignvariableop_33_adam_conv2d_122_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp*assignvariableop_34_adam_conv2d_122_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp,assignvariableop_35_adam_conv2d_123_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp*assignvariableop_36_adam_conv2d_123_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp,assignvariableop_37_adam_conv2d_124_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp*assignvariableop_38_adam_conv2d_124_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp,assignvariableop_39_adam_conv2d_125_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp*assignvariableop_40_adam_conv2d_125_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp,assignvariableop_41_adam_conv2d_126_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp*assignvariableop_42_adam_conv2d_126_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp,assignvariableop_43_adam_conv2d_127_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp*assignvariableop_44_adam_conv2d_127_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp,assignvariableop_45_adam_conv2d_128_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp*assignvariableop_46_adam_conv2d_128_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp,assignvariableop_47_adam_conv2d_129_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp*assignvariableop_48_adam_conv2d_129_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_dense_54_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp(assignvariableop_50_adam_dense_54_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp*assignvariableop_51_adam_dense_55_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp(assignvariableop_52_adam_dense_55_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp,assignvariableop_53_adam_conv2d_121_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp*assignvariableop_54_adam_conv2d_121_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp,assignvariableop_55_adam_conv2d_122_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp*assignvariableop_56_adam_conv2d_122_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp,assignvariableop_57_adam_conv2d_123_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp*assignvariableop_58_adam_conv2d_123_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp,assignvariableop_59_adam_conv2d_124_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp*assignvariableop_60_adam_conv2d_124_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp,assignvariableop_61_adam_conv2d_125_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp*assignvariableop_62_adam_conv2d_125_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp,assignvariableop_63_adam_conv2d_126_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp*assignvariableop_64_adam_conv2d_126_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp,assignvariableop_65_adam_conv2d_127_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp*assignvariableop_66_adam_conv2d_127_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOp,assignvariableop_67_adam_conv2d_128_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp*assignvariableop_68_adam_conv2d_128_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOp,assignvariableop_69_adam_conv2d_129_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp*assignvariableop_70_adam_conv2d_129_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOp*assignvariableop_71_adam_dense_54_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOp(assignvariableop_72_adam_dense_54_bias_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73?
AssignVariableOp_73AssignVariableOp*assignvariableop_73_adam_dense_55_kernel_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOp(assignvariableop_74_adam_dense_55_bias_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_749
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_75Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_75?
Identity_76IdentityIdentity_75:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_76"#
identity_76Identity_76:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
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
AssignVariableOp_74AssignVariableOp_742(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
f
G__inference_dropout_48_layer_call_and_return_conditional_losses_2126730

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
G__inference_conv2d_125_layer_call_and_return_conditional_losses_2126759

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
G__inference_conv2d_124_layer_call_and_return_conditional_losses_2127915

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
?
T
(__inference_add_24_layer_call_fn_2128030
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
C__inference_add_24_layer_call_and_return_conditional_losses_21268382
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
?
f
G__inference_dropout_50_layer_call_and_return_conditional_losses_2128062

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
G__inference_dropout_46_layer_call_and_return_conditional_losses_2126573

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
??
?
 __inference__traced_save_2128455
file_prefix0
,savev2_conv2d_121_kernel_read_readvariableop.
*savev2_conv2d_121_bias_read_readvariableop0
,savev2_conv2d_122_kernel_read_readvariableop.
*savev2_conv2d_122_bias_read_readvariableop0
,savev2_conv2d_123_kernel_read_readvariableop.
*savev2_conv2d_123_bias_read_readvariableop0
,savev2_conv2d_124_kernel_read_readvariableop.
*savev2_conv2d_124_bias_read_readvariableop0
,savev2_conv2d_125_kernel_read_readvariableop.
*savev2_conv2d_125_bias_read_readvariableop0
,savev2_conv2d_126_kernel_read_readvariableop.
*savev2_conv2d_126_bias_read_readvariableop0
,savev2_conv2d_127_kernel_read_readvariableop.
*savev2_conv2d_127_bias_read_readvariableop0
,savev2_conv2d_128_kernel_read_readvariableop.
*savev2_conv2d_128_bias_read_readvariableop0
,savev2_conv2d_129_kernel_read_readvariableop.
*savev2_conv2d_129_bias_read_readvariableop.
*savev2_dense_54_kernel_read_readvariableop,
(savev2_dense_54_bias_read_readvariableop.
*savev2_dense_55_kernel_read_readvariableop,
(savev2_dense_55_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop7
3savev2_adam_conv2d_121_kernel_m_read_readvariableop5
1savev2_adam_conv2d_121_bias_m_read_readvariableop7
3savev2_adam_conv2d_122_kernel_m_read_readvariableop5
1savev2_adam_conv2d_122_bias_m_read_readvariableop7
3savev2_adam_conv2d_123_kernel_m_read_readvariableop5
1savev2_adam_conv2d_123_bias_m_read_readvariableop7
3savev2_adam_conv2d_124_kernel_m_read_readvariableop5
1savev2_adam_conv2d_124_bias_m_read_readvariableop7
3savev2_adam_conv2d_125_kernel_m_read_readvariableop5
1savev2_adam_conv2d_125_bias_m_read_readvariableop7
3savev2_adam_conv2d_126_kernel_m_read_readvariableop5
1savev2_adam_conv2d_126_bias_m_read_readvariableop7
3savev2_adam_conv2d_127_kernel_m_read_readvariableop5
1savev2_adam_conv2d_127_bias_m_read_readvariableop7
3savev2_adam_conv2d_128_kernel_m_read_readvariableop5
1savev2_adam_conv2d_128_bias_m_read_readvariableop7
3savev2_adam_conv2d_129_kernel_m_read_readvariableop5
1savev2_adam_conv2d_129_bias_m_read_readvariableop5
1savev2_adam_dense_54_kernel_m_read_readvariableop3
/savev2_adam_dense_54_bias_m_read_readvariableop5
1savev2_adam_dense_55_kernel_m_read_readvariableop3
/savev2_adam_dense_55_bias_m_read_readvariableop7
3savev2_adam_conv2d_121_kernel_v_read_readvariableop5
1savev2_adam_conv2d_121_bias_v_read_readvariableop7
3savev2_adam_conv2d_122_kernel_v_read_readvariableop5
1savev2_adam_conv2d_122_bias_v_read_readvariableop7
3savev2_adam_conv2d_123_kernel_v_read_readvariableop5
1savev2_adam_conv2d_123_bias_v_read_readvariableop7
3savev2_adam_conv2d_124_kernel_v_read_readvariableop5
1savev2_adam_conv2d_124_bias_v_read_readvariableop7
3savev2_adam_conv2d_125_kernel_v_read_readvariableop5
1savev2_adam_conv2d_125_bias_v_read_readvariableop7
3savev2_adam_conv2d_126_kernel_v_read_readvariableop5
1savev2_adam_conv2d_126_bias_v_read_readvariableop7
3savev2_adam_conv2d_127_kernel_v_read_readvariableop5
1savev2_adam_conv2d_127_bias_v_read_readvariableop7
3savev2_adam_conv2d_128_kernel_v_read_readvariableop5
1savev2_adam_conv2d_128_bias_v_read_readvariableop7
3savev2_adam_conv2d_129_kernel_v_read_readvariableop5
1savev2_adam_conv2d_129_bias_v_read_readvariableop5
1savev2_adam_dense_54_kernel_v_read_readvariableop3
/savev2_adam_dense_54_bias_v_read_readvariableop5
1savev2_adam_dense_55_kernel_v_read_readvariableop3
/savev2_adam_dense_55_bias_v_read_readvariableop
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
ShardedFilename?*
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*?)
value?)B?)LB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*?
value?B?LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_conv2d_121_kernel_read_readvariableop*savev2_conv2d_121_bias_read_readvariableop,savev2_conv2d_122_kernel_read_readvariableop*savev2_conv2d_122_bias_read_readvariableop,savev2_conv2d_123_kernel_read_readvariableop*savev2_conv2d_123_bias_read_readvariableop,savev2_conv2d_124_kernel_read_readvariableop*savev2_conv2d_124_bias_read_readvariableop,savev2_conv2d_125_kernel_read_readvariableop*savev2_conv2d_125_bias_read_readvariableop,savev2_conv2d_126_kernel_read_readvariableop*savev2_conv2d_126_bias_read_readvariableop,savev2_conv2d_127_kernel_read_readvariableop*savev2_conv2d_127_bias_read_readvariableop,savev2_conv2d_128_kernel_read_readvariableop*savev2_conv2d_128_bias_read_readvariableop,savev2_conv2d_129_kernel_read_readvariableop*savev2_conv2d_129_bias_read_readvariableop*savev2_dense_54_kernel_read_readvariableop(savev2_dense_54_bias_read_readvariableop*savev2_dense_55_kernel_read_readvariableop(savev2_dense_55_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop3savev2_adam_conv2d_121_kernel_m_read_readvariableop1savev2_adam_conv2d_121_bias_m_read_readvariableop3savev2_adam_conv2d_122_kernel_m_read_readvariableop1savev2_adam_conv2d_122_bias_m_read_readvariableop3savev2_adam_conv2d_123_kernel_m_read_readvariableop1savev2_adam_conv2d_123_bias_m_read_readvariableop3savev2_adam_conv2d_124_kernel_m_read_readvariableop1savev2_adam_conv2d_124_bias_m_read_readvariableop3savev2_adam_conv2d_125_kernel_m_read_readvariableop1savev2_adam_conv2d_125_bias_m_read_readvariableop3savev2_adam_conv2d_126_kernel_m_read_readvariableop1savev2_adam_conv2d_126_bias_m_read_readvariableop3savev2_adam_conv2d_127_kernel_m_read_readvariableop1savev2_adam_conv2d_127_bias_m_read_readvariableop3savev2_adam_conv2d_128_kernel_m_read_readvariableop1savev2_adam_conv2d_128_bias_m_read_readvariableop3savev2_adam_conv2d_129_kernel_m_read_readvariableop1savev2_adam_conv2d_129_bias_m_read_readvariableop1savev2_adam_dense_54_kernel_m_read_readvariableop/savev2_adam_dense_54_bias_m_read_readvariableop1savev2_adam_dense_55_kernel_m_read_readvariableop/savev2_adam_dense_55_bias_m_read_readvariableop3savev2_adam_conv2d_121_kernel_v_read_readvariableop1savev2_adam_conv2d_121_bias_v_read_readvariableop3savev2_adam_conv2d_122_kernel_v_read_readvariableop1savev2_adam_conv2d_122_bias_v_read_readvariableop3savev2_adam_conv2d_123_kernel_v_read_readvariableop1savev2_adam_conv2d_123_bias_v_read_readvariableop3savev2_adam_conv2d_124_kernel_v_read_readvariableop1savev2_adam_conv2d_124_bias_v_read_readvariableop3savev2_adam_conv2d_125_kernel_v_read_readvariableop1savev2_adam_conv2d_125_bias_v_read_readvariableop3savev2_adam_conv2d_126_kernel_v_read_readvariableop1savev2_adam_conv2d_126_bias_v_read_readvariableop3savev2_adam_conv2d_127_kernel_v_read_readvariableop1savev2_adam_conv2d_127_bias_v_read_readvariableop3savev2_adam_conv2d_128_kernel_v_read_readvariableop1savev2_adam_conv2d_128_bias_v_read_readvariableop3savev2_adam_conv2d_129_kernel_v_read_readvariableop1savev2_adam_conv2d_129_bias_v_read_readvariableop1savev2_adam_dense_54_kernel_v_read_readvariableop/savev2_adam_dense_54_bias_v_read_readvariableop1savev2_adam_dense_55_kernel_v_read_readvariableop/savev2_adam_dense_55_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *Z
dtypesP
N2L	2
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
?: ::::::::::::::::::: : : :: : : : : : : : : ::::::::::::::::::: : : :::::::::::::::::::: : : :: 2(
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
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,	(
&
_output_shapes
:: 


_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :, (
&
_output_shapes
:: !

_output_shapes
::,"(
&
_output_shapes
:: #

_output_shapes
::,$(
&
_output_shapes
:: %

_output_shapes
::,&(
&
_output_shapes
:: '

_output_shapes
::,((
&
_output_shapes
:: )

_output_shapes
::,*(
&
_output_shapes
:: +

_output_shapes
::,,(
&
_output_shapes
:: -

_output_shapes
::,.(
&
_output_shapes
:: /

_output_shapes
::,0(
&
_output_shapes
:: 1

_output_shapes
::$2 

_output_shapes

: : 3

_output_shapes
: :$4 

_output_shapes

: : 5

_output_shapes
::,6(
&
_output_shapes
:: 7

_output_shapes
::,8(
&
_output_shapes
:: 9

_output_shapes
::,:(
&
_output_shapes
:: ;

_output_shapes
::,<(
&
_output_shapes
:: =

_output_shapes
::,>(
&
_output_shapes
:: ?
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
::,D(
&
_output_shapes
:: E

_output_shapes
::,F(
&
_output_shapes
:: G

_output_shapes
::$H 

_output_shapes

: : I

_output_shapes
: :$J 

_output_shapes

: : K

_output_shapes
::L

_output_shapes
: 
?
o
C__inference_add_23_layer_call_and_return_conditional_losses_2127898
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
?

?
G__inference_conv2d_121_layer_call_and_return_conditional_losses_2127789

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
E__inference_dense_55_layer_call_and_return_conditional_losses_2128198

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
G__inference_conv2d_128_layer_call_and_return_conditional_losses_2126916

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
?
8__inference_CNN_aug_deep_drop_skip_layer_call_fn_2127389
input_28
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

unknown_20
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_28unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_CNN_aug_deep_drop_skip_layer_call_and_return_conditional_losses_21273422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesw
u:???????????::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:???????????
"
_user_specified_name
input_28
?

*__inference_dense_54_layer_call_fn_2128187

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
E__inference_dense_54_layer_call_and_return_conditional_losses_21270302
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
??
?
S__inference_CNN_aug_deep_drop_skip_layer_call_and_return_conditional_losses_2127680

inputs-
)conv2d_121_conv2d_readvariableop_resource.
*conv2d_121_biasadd_readvariableop_resource-
)conv2d_122_conv2d_readvariableop_resource.
*conv2d_122_biasadd_readvariableop_resource-
)conv2d_123_conv2d_readvariableop_resource.
*conv2d_123_biasadd_readvariableop_resource-
)conv2d_124_conv2d_readvariableop_resource.
*conv2d_124_biasadd_readvariableop_resource-
)conv2d_125_conv2d_readvariableop_resource.
*conv2d_125_biasadd_readvariableop_resource-
)conv2d_126_conv2d_readvariableop_resource.
*conv2d_126_biasadd_readvariableop_resource-
)conv2d_127_conv2d_readvariableop_resource.
*conv2d_127_biasadd_readvariableop_resource-
)conv2d_128_conv2d_readvariableop_resource.
*conv2d_128_biasadd_readvariableop_resource-
)conv2d_129_conv2d_readvariableop_resource.
*conv2d_129_biasadd_readvariableop_resource+
'dense_54_matmul_readvariableop_resource,
(dense_54_biasadd_readvariableop_resource+
'dense_55_matmul_readvariableop_resource,
(dense_55_biasadd_readvariableop_resource
identity??!conv2d_121/BiasAdd/ReadVariableOp? conv2d_121/Conv2D/ReadVariableOp?!conv2d_122/BiasAdd/ReadVariableOp? conv2d_122/Conv2D/ReadVariableOp?!conv2d_123/BiasAdd/ReadVariableOp? conv2d_123/Conv2D/ReadVariableOp?!conv2d_124/BiasAdd/ReadVariableOp? conv2d_124/Conv2D/ReadVariableOp?!conv2d_125/BiasAdd/ReadVariableOp? conv2d_125/Conv2D/ReadVariableOp?!conv2d_126/BiasAdd/ReadVariableOp? conv2d_126/Conv2D/ReadVariableOp?!conv2d_127/BiasAdd/ReadVariableOp? conv2d_127/Conv2D/ReadVariableOp?!conv2d_128/BiasAdd/ReadVariableOp? conv2d_128/Conv2D/ReadVariableOp?!conv2d_129/BiasAdd/ReadVariableOp? conv2d_129/Conv2D/ReadVariableOp?dense_54/BiasAdd/ReadVariableOp?dense_54/MatMul/ReadVariableOp?dense_55/BiasAdd/ReadVariableOp?dense_55/MatMul/ReadVariableOp?
 conv2d_121/Conv2D/ReadVariableOpReadVariableOp)conv2d_121_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_121/Conv2D/ReadVariableOp?
conv2d_121/Conv2DConv2Dinputs(conv2d_121/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_121/Conv2D?
!conv2d_121/BiasAdd/ReadVariableOpReadVariableOp*conv2d_121_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_121/BiasAdd/ReadVariableOp?
conv2d_121/BiasAddBiasAddconv2d_121/Conv2D:output:0)conv2d_121/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_121/BiasAdd?
conv2d_121/ReluReluconv2d_121/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_121/Relu?
dropout_46/IdentityIdentityconv2d_121/Relu:activations:0*
T0*1
_output_shapes
:???????????2
dropout_46/Identity?
 conv2d_122/Conv2D/ReadVariableOpReadVariableOp)conv2d_122_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_122/Conv2D/ReadVariableOp?
conv2d_122/Conv2DConv2Ddropout_46/Identity:output:0(conv2d_122/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_122/Conv2D?
!conv2d_122/BiasAdd/ReadVariableOpReadVariableOp*conv2d_122_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_122/BiasAdd/ReadVariableOp?
conv2d_122/BiasAddBiasAddconv2d_122/Conv2D:output:0)conv2d_122/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_122/BiasAdd?
conv2d_122/ReluReluconv2d_122/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_122/Relu?
dropout_47/IdentityIdentityconv2d_122/Relu:activations:0*
T0*1
_output_shapes
:???????????2
dropout_47/Identity?
 conv2d_123/Conv2D/ReadVariableOpReadVariableOp)conv2d_123_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_123/Conv2D/ReadVariableOp?
conv2d_123/Conv2DConv2Ddropout_47/Identity:output:0(conv2d_123/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_123/Conv2D?
!conv2d_123/BiasAdd/ReadVariableOpReadVariableOp*conv2d_123_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_123/BiasAdd/ReadVariableOp?
conv2d_123/BiasAddBiasAddconv2d_123/Conv2D:output:0)conv2d_123/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_123/BiasAdd?
conv2d_123/ReluReluconv2d_123/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_123/Relu?

add_23/addAddV2conv2d_123/Relu:activations:0inputs*
T0*1
_output_shapes
:???????????2

add_23/add?
max_pooling2d_49/MaxPoolMaxPooladd_23/add:z:0*/
_output_shapes
:?????????***
ksize
*
paddingVALID*
strides
2
max_pooling2d_49/MaxPool?
 conv2d_124/Conv2D/ReadVariableOpReadVariableOp)conv2d_124_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_124/Conv2D/ReadVariableOp?
conv2d_124/Conv2DConv2D!max_pooling2d_49/MaxPool:output:0(conv2d_124/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????***
paddingSAME*
strides
2
conv2d_124/Conv2D?
!conv2d_124/BiasAdd/ReadVariableOpReadVariableOp*conv2d_124_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_124/BiasAdd/ReadVariableOp?
conv2d_124/BiasAddBiasAddconv2d_124/Conv2D:output:0)conv2d_124/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**2
conv2d_124/BiasAdd?
conv2d_124/ReluReluconv2d_124/BiasAdd:output:0*
T0*/
_output_shapes
:?????????**2
conv2d_124/Relu?
dropout_48/IdentityIdentityconv2d_124/Relu:activations:0*
T0*/
_output_shapes
:?????????**2
dropout_48/Identity?
 conv2d_125/Conv2D/ReadVariableOpReadVariableOp)conv2d_125_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_125/Conv2D/ReadVariableOp?
conv2d_125/Conv2DConv2Ddropout_48/Identity:output:0(conv2d_125/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????***
paddingSAME*
strides
2
conv2d_125/Conv2D?
!conv2d_125/BiasAdd/ReadVariableOpReadVariableOp*conv2d_125_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_125/BiasAdd/ReadVariableOp?
conv2d_125/BiasAddBiasAddconv2d_125/Conv2D:output:0)conv2d_125/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**2
conv2d_125/BiasAdd?
conv2d_125/ReluReluconv2d_125/BiasAdd:output:0*
T0*/
_output_shapes
:?????????**2
conv2d_125/Relu?
dropout_49/IdentityIdentityconv2d_125/Relu:activations:0*
T0*/
_output_shapes
:?????????**2
dropout_49/Identity?
 conv2d_126/Conv2D/ReadVariableOpReadVariableOp)conv2d_126_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_126/Conv2D/ReadVariableOp?
conv2d_126/Conv2DConv2Ddropout_49/Identity:output:0(conv2d_126/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????***
paddingSAME*
strides
2
conv2d_126/Conv2D?
!conv2d_126/BiasAdd/ReadVariableOpReadVariableOp*conv2d_126_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_126/BiasAdd/ReadVariableOp?
conv2d_126/BiasAddBiasAddconv2d_126/Conv2D:output:0)conv2d_126/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**2
conv2d_126/BiasAdd?
conv2d_126/ReluReluconv2d_126/BiasAdd:output:0*
T0*/
_output_shapes
:?????????**2
conv2d_126/Relu?

add_24/addAddV2conv2d_126/Relu:activations:0!max_pooling2d_49/MaxPool:output:0*
T0*/
_output_shapes
:?????????**2

add_24/add?
max_pooling2d_50/MaxPoolMaxPooladd_24/add:z:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_50/MaxPool?
 conv2d_127/Conv2D/ReadVariableOpReadVariableOp)conv2d_127_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_127/Conv2D/ReadVariableOp?
conv2d_127/Conv2DConv2D!max_pooling2d_50/MaxPool:output:0(conv2d_127/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_127/Conv2D?
!conv2d_127/BiasAdd/ReadVariableOpReadVariableOp*conv2d_127_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_127/BiasAdd/ReadVariableOp?
conv2d_127/BiasAddBiasAddconv2d_127/Conv2D:output:0)conv2d_127/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_127/BiasAdd?
conv2d_127/ReluReluconv2d_127/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_127/Relu?
dropout_50/IdentityIdentityconv2d_127/Relu:activations:0*
T0*/
_output_shapes
:?????????2
dropout_50/Identity?
 conv2d_128/Conv2D/ReadVariableOpReadVariableOp)conv2d_128_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_128/Conv2D/ReadVariableOp?
conv2d_128/Conv2DConv2Ddropout_50/Identity:output:0(conv2d_128/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_128/Conv2D?
!conv2d_128/BiasAdd/ReadVariableOpReadVariableOp*conv2d_128_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_128/BiasAdd/ReadVariableOp?
conv2d_128/BiasAddBiasAddconv2d_128/Conv2D:output:0)conv2d_128/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_128/BiasAdd?
conv2d_128/ReluReluconv2d_128/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_128/Relu?
dropout_51/IdentityIdentityconv2d_128/Relu:activations:0*
T0*/
_output_shapes
:?????????2
dropout_51/Identity?
 conv2d_129/Conv2D/ReadVariableOpReadVariableOp)conv2d_129_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_129/Conv2D/ReadVariableOp?
conv2d_129/Conv2DConv2Ddropout_51/Identity:output:0(conv2d_129/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_129/Conv2D?
!conv2d_129/BiasAdd/ReadVariableOpReadVariableOp*conv2d_129_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_129/BiasAdd/ReadVariableOp?
conv2d_129/BiasAddBiasAddconv2d_129/Conv2D:output:0)conv2d_129/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_129/BiasAdd?
conv2d_129/ReluReluconv2d_129/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_129/Relu?

add_25/addAddV2conv2d_129/Relu:activations:0!max_pooling2d_50/MaxPool:output:0*
T0*/
_output_shapes
:?????????2

add_25/add?
max_pooling2d_51/MaxPoolMaxPooladd_25/add:z:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_51/MaxPoolu
flatten_27/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_27/Const?
flatten_27/ReshapeReshape!max_pooling2d_51/MaxPool:output:0flatten_27/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_27/Reshape?
dense_54/MatMul/ReadVariableOpReadVariableOp'dense_54_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_54/MatMul/ReadVariableOp?
dense_54/MatMulMatMulflatten_27/Reshape:output:0&dense_54/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_54/MatMul?
dense_54/BiasAdd/ReadVariableOpReadVariableOp(dense_54_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_54/BiasAdd/ReadVariableOp?
dense_54/BiasAddBiasAdddense_54/MatMul:product:0'dense_54/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_54/BiasAdds
dense_54/ReluReludense_54/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_54/Relu?
dense_55/MatMul/ReadVariableOpReadVariableOp'dense_55_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_55/MatMul/ReadVariableOp?
dense_55/MatMulMatMuldense_54/Relu:activations:0&dense_55/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_55/MatMul?
dense_55/BiasAdd/ReadVariableOpReadVariableOp(dense_55_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_55/BiasAdd/ReadVariableOp?
dense_55/BiasAddBiasAdddense_55/MatMul:product:0'dense_55/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_55/BiasAdd|
dense_55/SoftmaxSoftmaxdense_55/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_55/Softmax?
IdentityIdentitydense_55/Softmax:softmax:0"^conv2d_121/BiasAdd/ReadVariableOp!^conv2d_121/Conv2D/ReadVariableOp"^conv2d_122/BiasAdd/ReadVariableOp!^conv2d_122/Conv2D/ReadVariableOp"^conv2d_123/BiasAdd/ReadVariableOp!^conv2d_123/Conv2D/ReadVariableOp"^conv2d_124/BiasAdd/ReadVariableOp!^conv2d_124/Conv2D/ReadVariableOp"^conv2d_125/BiasAdd/ReadVariableOp!^conv2d_125/Conv2D/ReadVariableOp"^conv2d_126/BiasAdd/ReadVariableOp!^conv2d_126/Conv2D/ReadVariableOp"^conv2d_127/BiasAdd/ReadVariableOp!^conv2d_127/Conv2D/ReadVariableOp"^conv2d_128/BiasAdd/ReadVariableOp!^conv2d_128/Conv2D/ReadVariableOp"^conv2d_129/BiasAdd/ReadVariableOp!^conv2d_129/Conv2D/ReadVariableOp ^dense_54/BiasAdd/ReadVariableOp^dense_54/MatMul/ReadVariableOp ^dense_55/BiasAdd/ReadVariableOp^dense_55/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesw
u:???????????::::::::::::::::::::::2F
!conv2d_121/BiasAdd/ReadVariableOp!conv2d_121/BiasAdd/ReadVariableOp2D
 conv2d_121/Conv2D/ReadVariableOp conv2d_121/Conv2D/ReadVariableOp2F
!conv2d_122/BiasAdd/ReadVariableOp!conv2d_122/BiasAdd/ReadVariableOp2D
 conv2d_122/Conv2D/ReadVariableOp conv2d_122/Conv2D/ReadVariableOp2F
!conv2d_123/BiasAdd/ReadVariableOp!conv2d_123/BiasAdd/ReadVariableOp2D
 conv2d_123/Conv2D/ReadVariableOp conv2d_123/Conv2D/ReadVariableOp2F
!conv2d_124/BiasAdd/ReadVariableOp!conv2d_124/BiasAdd/ReadVariableOp2D
 conv2d_124/Conv2D/ReadVariableOp conv2d_124/Conv2D/ReadVariableOp2F
!conv2d_125/BiasAdd/ReadVariableOp!conv2d_125/BiasAdd/ReadVariableOp2D
 conv2d_125/Conv2D/ReadVariableOp conv2d_125/Conv2D/ReadVariableOp2F
!conv2d_126/BiasAdd/ReadVariableOp!conv2d_126/BiasAdd/ReadVariableOp2D
 conv2d_126/Conv2D/ReadVariableOp conv2d_126/Conv2D/ReadVariableOp2F
!conv2d_127/BiasAdd/ReadVariableOp!conv2d_127/BiasAdd/ReadVariableOp2D
 conv2d_127/Conv2D/ReadVariableOp conv2d_127/Conv2D/ReadVariableOp2F
!conv2d_128/BiasAdd/ReadVariableOp!conv2d_128/BiasAdd/ReadVariableOp2D
 conv2d_128/Conv2D/ReadVariableOp conv2d_128/Conv2D/ReadVariableOp2F
!conv2d_129/BiasAdd/ReadVariableOp!conv2d_129/BiasAdd/ReadVariableOp2D
 conv2d_129/Conv2D/ReadVariableOp conv2d_129/Conv2D/ReadVariableOp2B
dense_54/BiasAdd/ReadVariableOpdense_54/BiasAdd/ReadVariableOp2@
dense_54/MatMul/ReadVariableOpdense_54/MatMul/ReadVariableOp2B
dense_55/BiasAdd/ReadVariableOpdense_55/BiasAdd/ReadVariableOp2@
dense_55/MatMul/ReadVariableOpdense_55/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
m
C__inference_add_25_layer_call_and_return_conditional_losses_2126995

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
?
e
,__inference_dropout_49_layer_call_fn_2127993

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
G__inference_dropout_49_layer_call_and_return_conditional_losses_21267872
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
?

*__inference_dense_55_layer_call_fn_2128207

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
E__inference_dense_55_layer_call_and_return_conditional_losses_21270572
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
?
e
G__inference_dropout_46_layer_call_and_return_conditional_losses_2127815

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
E__inference_dense_54_layer_call_and_return_conditional_losses_2128178

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
?
8__inference_CNN_aug_deep_drop_skip_layer_call_fn_2127268
input_28
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

unknown_20
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_28unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_CNN_aug_deep_drop_skip_layer_call_and_return_conditional_losses_21272212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesw
u:???????????::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:???????????
"
_user_specified_name
input_28
?
?
8__inference_CNN_aug_deep_drop_skip_layer_call_fn_2127729

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

unknown_20
identity??StatefulPartitionedCall?
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
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_CNN_aug_deep_drop_skip_layer_call_and_return_conditional_losses_21272212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesw
u:???????????::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
G__inference_conv2d_129_layer_call_and_return_conditional_losses_2128135

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
?
H
,__inference_dropout_51_layer_call_fn_2128124

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
G__inference_dropout_51_layer_call_and_return_conditional_losses_21269492
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
?
i
M__inference_max_pooling2d_49_layer_call_and_return_conditional_losses_2126500

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
?
o
C__inference_add_24_layer_call_and_return_conditional_losses_2128024
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
?

?
G__inference_conv2d_123_layer_call_and_return_conditional_losses_2127883

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
?
e
,__inference_dropout_48_layer_call_fn_2127946

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
G__inference_dropout_48_layer_call_and_return_conditional_losses_21267302
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
?
e
G__inference_dropout_49_layer_call_and_return_conditional_losses_2127988

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
G__inference_dropout_49_layer_call_and_return_conditional_losses_2127983

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
?
c
G__inference_flatten_27_layer_call_and_return_conditional_losses_2128162

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
G__inference_conv2d_128_layer_call_and_return_conditional_losses_2128088

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
G__inference_dropout_51_layer_call_and_return_conditional_losses_2128109

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
?
m
C__inference_add_24_layer_call_and_return_conditional_losses_2126838

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
?
H
,__inference_flatten_27_layer_call_fn_2128167

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
G__inference_flatten_27_layer_call_and_return_conditional_losses_21270112
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
,__inference_conv2d_129_layer_call_fn_2128144

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
G__inference_conv2d_129_layer_call_and_return_conditional_losses_21269732
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
?	
?
E__inference_dense_54_layer_call_and_return_conditional_losses_2127030

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
?
i
M__inference_max_pooling2d_51_layer_call_and_return_conditional_losses_2126524

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
H
,__inference_dropout_49_layer_call_fn_2127998

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
G__inference_dropout_49_layer_call_and_return_conditional_losses_21267922
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
?
e
G__inference_dropout_48_layer_call_and_return_conditional_losses_2126735

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
?
e
G__inference_dropout_51_layer_call_and_return_conditional_losses_2126949

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
?
H
,__inference_dropout_48_layer_call_fn_2127951

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
G__inference_dropout_48_layer_call_and_return_conditional_losses_21267352
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
?
T
(__inference_add_23_layer_call_fn_2127904
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
C__inference_add_23_layer_call_and_return_conditional_losses_21266812
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
?
m
C__inference_add_23_layer_call_and_return_conditional_losses_2126681

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
?

?
G__inference_conv2d_122_layer_call_and_return_conditional_losses_2126602

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
?
?
,__inference_conv2d_127_layer_call_fn_2128050

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
G__inference_conv2d_127_layer_call_and_return_conditional_losses_21268592
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
?

?
G__inference_conv2d_124_layer_call_and_return_conditional_losses_2126702

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
?

?
G__inference_conv2d_127_layer_call_and_return_conditional_losses_2128041

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
?
f
G__inference_dropout_49_layer_call_and_return_conditional_losses_2126787

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
?
?
,__inference_conv2d_128_layer_call_fn_2128097

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
G__inference_conv2d_128_layer_call_and_return_conditional_losses_21269162
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
,__inference_conv2d_121_layer_call_fn_2127798

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
G__inference_conv2d_121_layer_call_and_return_conditional_losses_21265452
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
,__inference_dropout_47_layer_call_fn_2127867

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
G__inference_dropout_47_layer_call_and_return_conditional_losses_21266302
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
?
f
G__inference_dropout_51_layer_call_and_return_conditional_losses_2126944

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
?
e
G__inference_dropout_47_layer_call_and_return_conditional_losses_2126635

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
?n
?	
S__inference_CNN_aug_deep_drop_skip_layer_call_and_return_conditional_losses_2127221

inputs
conv2d_121_2127152
conv2d_121_2127154
conv2d_122_2127158
conv2d_122_2127160
conv2d_123_2127164
conv2d_123_2127166
conv2d_124_2127171
conv2d_124_2127173
conv2d_125_2127177
conv2d_125_2127179
conv2d_126_2127183
conv2d_126_2127185
conv2d_127_2127190
conv2d_127_2127192
conv2d_128_2127196
conv2d_128_2127198
conv2d_129_2127202
conv2d_129_2127204
dense_54_2127210
dense_54_2127212
dense_55_2127215
dense_55_2127217
identity??"conv2d_121/StatefulPartitionedCall?"conv2d_122/StatefulPartitionedCall?"conv2d_123/StatefulPartitionedCall?"conv2d_124/StatefulPartitionedCall?"conv2d_125/StatefulPartitionedCall?"conv2d_126/StatefulPartitionedCall?"conv2d_127/StatefulPartitionedCall?"conv2d_128/StatefulPartitionedCall?"conv2d_129/StatefulPartitionedCall? dense_54/StatefulPartitionedCall? dense_55/StatefulPartitionedCall?"dropout_46/StatefulPartitionedCall?"dropout_47/StatefulPartitionedCall?"dropout_48/StatefulPartitionedCall?"dropout_49/StatefulPartitionedCall?"dropout_50/StatefulPartitionedCall?"dropout_51/StatefulPartitionedCall?
"conv2d_121/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_121_2127152conv2d_121_2127154*
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
G__inference_conv2d_121_layer_call_and_return_conditional_losses_21265452$
"conv2d_121/StatefulPartitionedCall?
"dropout_46/StatefulPartitionedCallStatefulPartitionedCall+conv2d_121/StatefulPartitionedCall:output:0*
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
G__inference_dropout_46_layer_call_and_return_conditional_losses_21265732$
"dropout_46/StatefulPartitionedCall?
"conv2d_122/StatefulPartitionedCallStatefulPartitionedCall+dropout_46/StatefulPartitionedCall:output:0conv2d_122_2127158conv2d_122_2127160*
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
G__inference_conv2d_122_layer_call_and_return_conditional_losses_21266022$
"conv2d_122/StatefulPartitionedCall?
"dropout_47/StatefulPartitionedCallStatefulPartitionedCall+conv2d_122/StatefulPartitionedCall:output:0#^dropout_46/StatefulPartitionedCall*
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
G__inference_dropout_47_layer_call_and_return_conditional_losses_21266302$
"dropout_47/StatefulPartitionedCall?
"conv2d_123/StatefulPartitionedCallStatefulPartitionedCall+dropout_47/StatefulPartitionedCall:output:0conv2d_123_2127164conv2d_123_2127166*
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
G__inference_conv2d_123_layer_call_and_return_conditional_losses_21266592$
"conv2d_123/StatefulPartitionedCall?
add_23/PartitionedCallPartitionedCall+conv2d_123/StatefulPartitionedCall:output:0inputs*
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
C__inference_add_23_layer_call_and_return_conditional_losses_21266812
add_23/PartitionedCall?
 max_pooling2d_49/PartitionedCallPartitionedCalladd_23/PartitionedCall:output:0*
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
M__inference_max_pooling2d_49_layer_call_and_return_conditional_losses_21265002"
 max_pooling2d_49/PartitionedCall?
"conv2d_124/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_49/PartitionedCall:output:0conv2d_124_2127171conv2d_124_2127173*
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
G__inference_conv2d_124_layer_call_and_return_conditional_losses_21267022$
"conv2d_124/StatefulPartitionedCall?
"dropout_48/StatefulPartitionedCallStatefulPartitionedCall+conv2d_124/StatefulPartitionedCall:output:0#^dropout_47/StatefulPartitionedCall*
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
G__inference_dropout_48_layer_call_and_return_conditional_losses_21267302$
"dropout_48/StatefulPartitionedCall?
"conv2d_125/StatefulPartitionedCallStatefulPartitionedCall+dropout_48/StatefulPartitionedCall:output:0conv2d_125_2127177conv2d_125_2127179*
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
G__inference_conv2d_125_layer_call_and_return_conditional_losses_21267592$
"conv2d_125/StatefulPartitionedCall?
"dropout_49/StatefulPartitionedCallStatefulPartitionedCall+conv2d_125/StatefulPartitionedCall:output:0#^dropout_48/StatefulPartitionedCall*
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
G__inference_dropout_49_layer_call_and_return_conditional_losses_21267872$
"dropout_49/StatefulPartitionedCall?
"conv2d_126/StatefulPartitionedCallStatefulPartitionedCall+dropout_49/StatefulPartitionedCall:output:0conv2d_126_2127183conv2d_126_2127185*
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
G__inference_conv2d_126_layer_call_and_return_conditional_losses_21268162$
"conv2d_126/StatefulPartitionedCall?
add_24/PartitionedCallPartitionedCall+conv2d_126/StatefulPartitionedCall:output:0)max_pooling2d_49/PartitionedCall:output:0*
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
C__inference_add_24_layer_call_and_return_conditional_losses_21268382
add_24/PartitionedCall?
 max_pooling2d_50/PartitionedCallPartitionedCalladd_24/PartitionedCall:output:0*
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
M__inference_max_pooling2d_50_layer_call_and_return_conditional_losses_21265122"
 max_pooling2d_50/PartitionedCall?
"conv2d_127/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_50/PartitionedCall:output:0conv2d_127_2127190conv2d_127_2127192*
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
G__inference_conv2d_127_layer_call_and_return_conditional_losses_21268592$
"conv2d_127/StatefulPartitionedCall?
"dropout_50/StatefulPartitionedCallStatefulPartitionedCall+conv2d_127/StatefulPartitionedCall:output:0#^dropout_49/StatefulPartitionedCall*
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
G__inference_dropout_50_layer_call_and_return_conditional_losses_21268872$
"dropout_50/StatefulPartitionedCall?
"conv2d_128/StatefulPartitionedCallStatefulPartitionedCall+dropout_50/StatefulPartitionedCall:output:0conv2d_128_2127196conv2d_128_2127198*
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
G__inference_conv2d_128_layer_call_and_return_conditional_losses_21269162$
"conv2d_128/StatefulPartitionedCall?
"dropout_51/StatefulPartitionedCallStatefulPartitionedCall+conv2d_128/StatefulPartitionedCall:output:0#^dropout_50/StatefulPartitionedCall*
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
G__inference_dropout_51_layer_call_and_return_conditional_losses_21269442$
"dropout_51/StatefulPartitionedCall?
"conv2d_129/StatefulPartitionedCallStatefulPartitionedCall+dropout_51/StatefulPartitionedCall:output:0conv2d_129_2127202conv2d_129_2127204*
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
G__inference_conv2d_129_layer_call_and_return_conditional_losses_21269732$
"conv2d_129/StatefulPartitionedCall?
add_25/PartitionedCallPartitionedCall+conv2d_129/StatefulPartitionedCall:output:0)max_pooling2d_50/PartitionedCall:output:0*
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
C__inference_add_25_layer_call_and_return_conditional_losses_21269952
add_25/PartitionedCall?
 max_pooling2d_51/PartitionedCallPartitionedCalladd_25/PartitionedCall:output:0*
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
M__inference_max_pooling2d_51_layer_call_and_return_conditional_losses_21265242"
 max_pooling2d_51/PartitionedCall?
flatten_27/PartitionedCallPartitionedCall)max_pooling2d_51/PartitionedCall:output:0*
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
G__inference_flatten_27_layer_call_and_return_conditional_losses_21270112
flatten_27/PartitionedCall?
 dense_54/StatefulPartitionedCallStatefulPartitionedCall#flatten_27/PartitionedCall:output:0dense_54_2127210dense_54_2127212*
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
E__inference_dense_54_layer_call_and_return_conditional_losses_21270302"
 dense_54/StatefulPartitionedCall?
 dense_55/StatefulPartitionedCallStatefulPartitionedCall)dense_54/StatefulPartitionedCall:output:0dense_55_2127215dense_55_2127217*
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
E__inference_dense_55_layer_call_and_return_conditional_losses_21270572"
 dense_55/StatefulPartitionedCall?
IdentityIdentity)dense_55/StatefulPartitionedCall:output:0#^conv2d_121/StatefulPartitionedCall#^conv2d_122/StatefulPartitionedCall#^conv2d_123/StatefulPartitionedCall#^conv2d_124/StatefulPartitionedCall#^conv2d_125/StatefulPartitionedCall#^conv2d_126/StatefulPartitionedCall#^conv2d_127/StatefulPartitionedCall#^conv2d_128/StatefulPartitionedCall#^conv2d_129/StatefulPartitionedCall!^dense_54/StatefulPartitionedCall!^dense_55/StatefulPartitionedCall#^dropout_46/StatefulPartitionedCall#^dropout_47/StatefulPartitionedCall#^dropout_48/StatefulPartitionedCall#^dropout_49/StatefulPartitionedCall#^dropout_50/StatefulPartitionedCall#^dropout_51/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesw
u:???????????::::::::::::::::::::::2H
"conv2d_121/StatefulPartitionedCall"conv2d_121/StatefulPartitionedCall2H
"conv2d_122/StatefulPartitionedCall"conv2d_122/StatefulPartitionedCall2H
"conv2d_123/StatefulPartitionedCall"conv2d_123/StatefulPartitionedCall2H
"conv2d_124/StatefulPartitionedCall"conv2d_124/StatefulPartitionedCall2H
"conv2d_125/StatefulPartitionedCall"conv2d_125/StatefulPartitionedCall2H
"conv2d_126/StatefulPartitionedCall"conv2d_126/StatefulPartitionedCall2H
"conv2d_127/StatefulPartitionedCall"conv2d_127/StatefulPartitionedCall2H
"conv2d_128/StatefulPartitionedCall"conv2d_128/StatefulPartitionedCall2H
"conv2d_129/StatefulPartitionedCall"conv2d_129/StatefulPartitionedCall2D
 dense_54/StatefulPartitionedCall dense_54/StatefulPartitionedCall2D
 dense_55/StatefulPartitionedCall dense_55/StatefulPartitionedCall2H
"dropout_46/StatefulPartitionedCall"dropout_46/StatefulPartitionedCall2H
"dropout_47/StatefulPartitionedCall"dropout_47/StatefulPartitionedCall2H
"dropout_48/StatefulPartitionedCall"dropout_48/StatefulPartitionedCall2H
"dropout_49/StatefulPartitionedCall"dropout_49/StatefulPartitionedCall2H
"dropout_50/StatefulPartitionedCall"dropout_50/StatefulPartitionedCall2H
"dropout_51/StatefulPartitionedCall"dropout_51/StatefulPartitionedCall:Y U
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
input_28;
serving_default_input_28:0???????????<
dense_550
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
ӽ
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer_with_weights-4
layer-10
layer-11
layer_with_weights-5
layer-12
layer-13
layer-14
layer_with_weights-6
layer-15
layer-16
layer_with_weights-7
layer-17
layer-18
layer_with_weights-8
layer-19
layer-20
layer-21
layer-22
layer_with_weights-9
layer-23
layer_with_weights-10
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
_tf_keras_network??{"class_name": "Functional", "name": "CNN_aug_deep_drop_skip", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "CNN_aug_deep_drop_skip", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_28"}, "name": "input_28", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_121", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_121", "inbound_nodes": [[["input_28", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_46", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "name": "dropout_46", "inbound_nodes": [[["conv2d_121", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_122", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_122", "inbound_nodes": [[["dropout_46", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_47", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "name": "dropout_47", "inbound_nodes": [[["conv2d_122", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_123", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_123", "inbound_nodes": [[["dropout_47", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_23", "trainable": true, "dtype": "float32"}, "name": "add_23", "inbound_nodes": [[["conv2d_123", 0, 0, {}], ["input_28", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_49", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "name": "max_pooling2d_49", "inbound_nodes": [[["add_23", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_124", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_124", "inbound_nodes": [[["max_pooling2d_49", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_48", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "name": "dropout_48", "inbound_nodes": [[["conv2d_124", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_125", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_125", "inbound_nodes": [[["dropout_48", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_49", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "name": "dropout_49", "inbound_nodes": [[["conv2d_125", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_126", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_126", "inbound_nodes": [[["dropout_49", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_24", "trainable": true, "dtype": "float32"}, "name": "add_24", "inbound_nodes": [[["conv2d_126", 0, 0, {}], ["max_pooling2d_49", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_50", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "name": "max_pooling2d_50", "inbound_nodes": [[["add_24", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_127", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_127", "inbound_nodes": [[["max_pooling2d_50", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_50", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "name": "dropout_50", "inbound_nodes": [[["conv2d_127", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_128", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_128", "inbound_nodes": [[["dropout_50", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_51", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "name": "dropout_51", "inbound_nodes": [[["conv2d_128", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_129", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_129", "inbound_nodes": [[["dropout_51", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_25", "trainable": true, "dtype": "float32"}, "name": "add_25", "inbound_nodes": [[["conv2d_129", 0, 0, {}], ["max_pooling2d_50", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_51", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "name": "max_pooling2d_51", "inbound_nodes": [[["add_25", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_27", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_27", "inbound_nodes": [[["max_pooling2d_51", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_54", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_54", "inbound_nodes": [[["flatten_27", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_55", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_55", "inbound_nodes": [[["dense_54", 0, 0, {}]]]}], "input_layers": [["input_28", 0, 0]], "output_layers": [["dense_55", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "CNN_aug_deep_drop_skip", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_28"}, "name": "input_28", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_121", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_121", "inbound_nodes": [[["input_28", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_46", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "name": "dropout_46", "inbound_nodes": [[["conv2d_121", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_122", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_122", "inbound_nodes": [[["dropout_46", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_47", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "name": "dropout_47", "inbound_nodes": [[["conv2d_122", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_123", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_123", "inbound_nodes": [[["dropout_47", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_23", "trainable": true, "dtype": "float32"}, "name": "add_23", "inbound_nodes": [[["conv2d_123", 0, 0, {}], ["input_28", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_49", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "name": "max_pooling2d_49", "inbound_nodes": [[["add_23", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_124", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_124", "inbound_nodes": [[["max_pooling2d_49", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_48", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "name": "dropout_48", "inbound_nodes": [[["conv2d_124", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_125", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_125", "inbound_nodes": [[["dropout_48", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_49", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "name": "dropout_49", "inbound_nodes": [[["conv2d_125", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_126", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_126", "inbound_nodes": [[["dropout_49", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_24", "trainable": true, "dtype": "float32"}, "name": "add_24", "inbound_nodes": [[["conv2d_126", 0, 0, {}], ["max_pooling2d_49", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_50", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "name": "max_pooling2d_50", "inbound_nodes": [[["add_24", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_127", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_127", "inbound_nodes": [[["max_pooling2d_50", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_50", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "name": "dropout_50", "inbound_nodes": [[["conv2d_127", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_128", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_128", "inbound_nodes": [[["dropout_50", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_51", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "name": "dropout_51", "inbound_nodes": [[["conv2d_128", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_129", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_129", "inbound_nodes": [[["dropout_51", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_25", "trainable": true, "dtype": "float32"}, "name": "add_25", "inbound_nodes": [[["conv2d_129", 0, 0, {}], ["max_pooling2d_50", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_51", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "name": "max_pooling2d_51", "inbound_nodes": [[["add_25", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_27", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_27", "inbound_nodes": [[["max_pooling2d_51", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_54", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_54", "inbound_nodes": [[["flatten_27", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_55", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_55", "inbound_nodes": [[["dense_54", 0, 0, {}]]]}], "input_layers": [["input_28", 0, 0]], "output_layers": [["dense_55", 0, 0]]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_28", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_28"}}
?	

 kernel
!bias
"regularization_losses
#trainable_variables
$	variables
%	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_121", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_121", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 1]}}
?
&regularization_losses
'trainable_variables
(	variables
)	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_46", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_46", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}
?	

*kernel
+bias
,regularization_losses
-trainable_variables
.	variables
/	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_122", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_122", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 8]}}
?
0regularization_losses
1trainable_variables
2	variables
3	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_47", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_47", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}
?	

4kernel
5bias
6regularization_losses
7trainable_variables
8	variables
9	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_123", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_123", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 8]}}
?
:regularization_losses
;trainable_variables
<	variables
=	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Add", "name": "add_23", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "add_23", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 128, 128, 1]}, {"class_name": "TensorShape", "items": [null, 128, 128, 1]}]}
?
>regularization_losses
?trainable_variables
@	variables
A	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_49", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_49", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

Bkernel
Cbias
Dregularization_losses
Etrainable_variables
F	variables
G	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_124", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_124", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 42, 42, 1]}}
?
Hregularization_losses
Itrainable_variables
J	variables
K	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_48", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_48", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}
?	

Lkernel
Mbias
Nregularization_losses
Otrainable_variables
P	variables
Q	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_125", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_125", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 42, 42, 8]}}
?
Rregularization_losses
Strainable_variables
T	variables
U	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_49", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_49", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}
?	

Vkernel
Wbias
Xregularization_losses
Ytrainable_variables
Z	variables
[	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_126", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_126", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 42, 42, 8]}}
?
\regularization_losses
]trainable_variables
^	variables
_	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Add", "name": "add_24", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "add_24", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 42, 42, 1]}, {"class_name": "TensorShape", "items": [null, 42, 42, 1]}]}
?
`regularization_losses
atrainable_variables
b	variables
c	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_50", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_50", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

dkernel
ebias
fregularization_losses
gtrainable_variables
h	variables
i	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_127", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_127", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 1]}}
?
jregularization_losses
ktrainable_variables
l	variables
m	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_50", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_50", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}
?	

nkernel
obias
pregularization_losses
qtrainable_variables
r	variables
s	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_128", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_128", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 8]}}
?
tregularization_losses
utrainable_variables
v	variables
w	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_51", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_51", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}
?	

xkernel
ybias
zregularization_losses
{trainable_variables
|	variables
}	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_129", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_129", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 8]}}
?
~regularization_losses
trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Add", "name": "add_25", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "add_25", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 14, 14, 1]}, {"class_name": "TensorShape", "items": [null, 14, 14, 1]}]}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_51", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_51", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_27", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_27", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_54", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_54", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
?
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_55", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_55", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
?
	?iter
?beta_1
?beta_2

?decay
?learning_rate m?!m?*m?+m?4m?5m?Bm?Cm?Lm?Mm?Vm?Wm?dm?em?nm?om?xm?ym?	?m?	?m?	?m?	?m? v?!v?*v?+v?4v?5v?Bv?Cv?Lv?Mv?Vv?Wv?dv?ev?nv?ov?xv?yv?	?v?	?v?	?v?	?v?"
	optimizer
 "
trackable_list_wrapper
?
 0
!1
*2
+3
44
55
B6
C7
L8
M9
V10
W11
d12
e13
n14
o15
x16
y17
?18
?19
?20
?21"
trackable_list_wrapper
?
 0
!1
*2
+3
44
55
B6
C7
L8
M9
V10
W11
d12
e13
n14
o15
x16
y17
?18
?19
?20
?21"
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
+:)2conv2d_121/kernel
:2conv2d_121/bias
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
&regularization_losses
?layers
?layer_metrics
?metrics
'trainable_variables
(	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)2conv2d_122/kernel
:2conv2d_122/bias
 "
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
?
,regularization_losses
?layers
?layer_metrics
?metrics
-trainable_variables
.	variables
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
0regularization_losses
?layers
?layer_metrics
?metrics
1trainable_variables
2	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)2conv2d_123/kernel
:2conv2d_123/bias
 "
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
?
6regularization_losses
?layers
?layer_metrics
?metrics
7trainable_variables
8	variables
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
:regularization_losses
?layers
?layer_metrics
?metrics
;trainable_variables
<	variables
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
+:)2conv2d_124/kernel
:2conv2d_124/bias
 "
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
.
B0
C1"
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
Hregularization_losses
?layers
?layer_metrics
?metrics
Itrainable_variables
J	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)2conv2d_125/kernel
:2conv2d_125/bias
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
Rregularization_losses
?layers
?layer_metrics
?metrics
Strainable_variables
T	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)2conv2d_126/kernel
:2conv2d_126/bias
 "
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
?
Xregularization_losses
?layers
?layer_metrics
?metrics
Ytrainable_variables
Z	variables
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
\regularization_losses
?layers
?layer_metrics
?metrics
]trainable_variables
^	variables
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
+:)2conv2d_127/kernel
:2conv2d_127/bias
 "
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
.
d0
e1"
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
jregularization_losses
?layers
?layer_metrics
?metrics
ktrainable_variables
l	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)2conv2d_128/kernel
:2conv2d_128/bias
 "
trackable_list_wrapper
.
n0
o1"
trackable_list_wrapper
.
n0
o1"
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
tregularization_losses
?layers
?layer_metrics
?metrics
utrainable_variables
v	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)2conv2d_129/kernel
:2conv2d_129/bias
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
?layers
?layer_metrics
?metrics
{trainable_variables
|	variables
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
~regularization_losses
?layers
?layer_metrics
?metrics
trainable_variables
?	variables
 ?layer_regularization_losses
?non_trainable_variables
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
?layers
?layer_metrics
?metrics
?trainable_variables
?	variables
 ?layer_regularization_losses
?non_trainable_variables
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
?layers
?layer_metrics
?metrics
?trainable_variables
?	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!: 2dense_54/kernel
: 2dense_54/bias
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
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!: 2dense_55/kernel
:2dense_55/bias
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
0:.2Adam/conv2d_121/kernel/m
": 2Adam/conv2d_121/bias/m
0:.2Adam/conv2d_122/kernel/m
": 2Adam/conv2d_122/bias/m
0:.2Adam/conv2d_123/kernel/m
": 2Adam/conv2d_123/bias/m
0:.2Adam/conv2d_124/kernel/m
": 2Adam/conv2d_124/bias/m
0:.2Adam/conv2d_125/kernel/m
": 2Adam/conv2d_125/bias/m
0:.2Adam/conv2d_126/kernel/m
": 2Adam/conv2d_126/bias/m
0:.2Adam/conv2d_127/kernel/m
": 2Adam/conv2d_127/bias/m
0:.2Adam/conv2d_128/kernel/m
": 2Adam/conv2d_128/bias/m
0:.2Adam/conv2d_129/kernel/m
": 2Adam/conv2d_129/bias/m
&:$ 2Adam/dense_54/kernel/m
 : 2Adam/dense_54/bias/m
&:$ 2Adam/dense_55/kernel/m
 :2Adam/dense_55/bias/m
0:.2Adam/conv2d_121/kernel/v
": 2Adam/conv2d_121/bias/v
0:.2Adam/conv2d_122/kernel/v
": 2Adam/conv2d_122/bias/v
0:.2Adam/conv2d_123/kernel/v
": 2Adam/conv2d_123/bias/v
0:.2Adam/conv2d_124/kernel/v
": 2Adam/conv2d_124/bias/v
0:.2Adam/conv2d_125/kernel/v
": 2Adam/conv2d_125/bias/v
0:.2Adam/conv2d_126/kernel/v
": 2Adam/conv2d_126/bias/v
0:.2Adam/conv2d_127/kernel/v
": 2Adam/conv2d_127/bias/v
0:.2Adam/conv2d_128/kernel/v
": 2Adam/conv2d_128/bias/v
0:.2Adam/conv2d_129/kernel/v
": 2Adam/conv2d_129/bias/v
&:$ 2Adam/dense_54/kernel/v
 : 2Adam/dense_54/bias/v
&:$ 2Adam/dense_55/kernel/v
 :2Adam/dense_55/bias/v
?2?
"__inference__wrapped_model_2126494?
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
input_28???????????
?2?
S__inference_CNN_aug_deep_drop_skip_layer_call_and_return_conditional_losses_2127585
S__inference_CNN_aug_deep_drop_skip_layer_call_and_return_conditional_losses_2127146
S__inference_CNN_aug_deep_drop_skip_layer_call_and_return_conditional_losses_2127680
S__inference_CNN_aug_deep_drop_skip_layer_call_and_return_conditional_losses_2127074?
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
8__inference_CNN_aug_deep_drop_skip_layer_call_fn_2127268
8__inference_CNN_aug_deep_drop_skip_layer_call_fn_2127389
8__inference_CNN_aug_deep_drop_skip_layer_call_fn_2127729
8__inference_CNN_aug_deep_drop_skip_layer_call_fn_2127778?
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
G__inference_conv2d_121_layer_call_and_return_conditional_losses_2127789?
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
,__inference_conv2d_121_layer_call_fn_2127798?
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
G__inference_dropout_46_layer_call_and_return_conditional_losses_2127810
G__inference_dropout_46_layer_call_and_return_conditional_losses_2127815?
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
,__inference_dropout_46_layer_call_fn_2127820
,__inference_dropout_46_layer_call_fn_2127825?
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
G__inference_conv2d_122_layer_call_and_return_conditional_losses_2127836?
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
,__inference_conv2d_122_layer_call_fn_2127845?
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
G__inference_dropout_47_layer_call_and_return_conditional_losses_2127862
G__inference_dropout_47_layer_call_and_return_conditional_losses_2127857?
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
,__inference_dropout_47_layer_call_fn_2127872
,__inference_dropout_47_layer_call_fn_2127867?
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
G__inference_conv2d_123_layer_call_and_return_conditional_losses_2127883?
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
,__inference_conv2d_123_layer_call_fn_2127892?
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
C__inference_add_23_layer_call_and_return_conditional_losses_2127898?
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
(__inference_add_23_layer_call_fn_2127904?
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
M__inference_max_pooling2d_49_layer_call_and_return_conditional_losses_2126500?
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
2__inference_max_pooling2d_49_layer_call_fn_2126506?
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
G__inference_conv2d_124_layer_call_and_return_conditional_losses_2127915?
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
,__inference_conv2d_124_layer_call_fn_2127924?
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
G__inference_dropout_48_layer_call_and_return_conditional_losses_2127936
G__inference_dropout_48_layer_call_and_return_conditional_losses_2127941?
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
,__inference_dropout_48_layer_call_fn_2127946
,__inference_dropout_48_layer_call_fn_2127951?
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
G__inference_conv2d_125_layer_call_and_return_conditional_losses_2127962?
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
,__inference_conv2d_125_layer_call_fn_2127971?
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
G__inference_dropout_49_layer_call_and_return_conditional_losses_2127988
G__inference_dropout_49_layer_call_and_return_conditional_losses_2127983?
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
,__inference_dropout_49_layer_call_fn_2127993
,__inference_dropout_49_layer_call_fn_2127998?
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
G__inference_conv2d_126_layer_call_and_return_conditional_losses_2128009?
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
,__inference_conv2d_126_layer_call_fn_2128018?
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
C__inference_add_24_layer_call_and_return_conditional_losses_2128024?
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
(__inference_add_24_layer_call_fn_2128030?
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
M__inference_max_pooling2d_50_layer_call_and_return_conditional_losses_2126512?
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
2__inference_max_pooling2d_50_layer_call_fn_2126518?
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
G__inference_conv2d_127_layer_call_and_return_conditional_losses_2128041?
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
,__inference_conv2d_127_layer_call_fn_2128050?
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
G__inference_dropout_50_layer_call_and_return_conditional_losses_2128067
G__inference_dropout_50_layer_call_and_return_conditional_losses_2128062?
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
,__inference_dropout_50_layer_call_fn_2128072
,__inference_dropout_50_layer_call_fn_2128077?
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
G__inference_conv2d_128_layer_call_and_return_conditional_losses_2128088?
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
,__inference_conv2d_128_layer_call_fn_2128097?
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
G__inference_dropout_51_layer_call_and_return_conditional_losses_2128114
G__inference_dropout_51_layer_call_and_return_conditional_losses_2128109?
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
,__inference_dropout_51_layer_call_fn_2128119
,__inference_dropout_51_layer_call_fn_2128124?
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
G__inference_conv2d_129_layer_call_and_return_conditional_losses_2128135?
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
,__inference_conv2d_129_layer_call_fn_2128144?
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
C__inference_add_25_layer_call_and_return_conditional_losses_2128150?
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
(__inference_add_25_layer_call_fn_2128156?
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
M__inference_max_pooling2d_51_layer_call_and_return_conditional_losses_2126524?
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
2__inference_max_pooling2d_51_layer_call_fn_2126530?
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
G__inference_flatten_27_layer_call_and_return_conditional_losses_2128162?
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
,__inference_flatten_27_layer_call_fn_2128167?
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
E__inference_dense_54_layer_call_and_return_conditional_losses_2128178?
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
*__inference_dense_54_layer_call_fn_2128187?
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
E__inference_dense_55_layer_call_and_return_conditional_losses_2128198?
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
*__inference_dense_55_layer_call_fn_2128207?
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
%__inference_signature_wrapper_2127448input_28"?
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
S__inference_CNN_aug_deep_drop_skip_layer_call_and_return_conditional_losses_2127074? !*+45BCLMVWdenoxy????C?@
9?6
,?)
input_28???????????
p

 
? "%?"
?
0?????????
? ?
S__inference_CNN_aug_deep_drop_skip_layer_call_and_return_conditional_losses_2127146? !*+45BCLMVWdenoxy????C?@
9?6
,?)
input_28???????????
p 

 
? "%?"
?
0?????????
? ?
S__inference_CNN_aug_deep_drop_skip_layer_call_and_return_conditional_losses_2127585? !*+45BCLMVWdenoxy????A?>
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
S__inference_CNN_aug_deep_drop_skip_layer_call_and_return_conditional_losses_2127680? !*+45BCLMVWdenoxy????A?>
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
8__inference_CNN_aug_deep_drop_skip_layer_call_fn_2127268{ !*+45BCLMVWdenoxy????C?@
9?6
,?)
input_28???????????
p

 
? "???????????
8__inference_CNN_aug_deep_drop_skip_layer_call_fn_2127389{ !*+45BCLMVWdenoxy????C?@
9?6
,?)
input_28???????????
p 

 
? "???????????
8__inference_CNN_aug_deep_drop_skip_layer_call_fn_2127729y !*+45BCLMVWdenoxy????A?>
7?4
*?'
inputs???????????
p

 
? "???????????
8__inference_CNN_aug_deep_drop_skip_layer_call_fn_2127778y !*+45BCLMVWdenoxy????A?>
7?4
*?'
inputs???????????
p 

 
? "???????????
"__inference__wrapped_model_2126494? !*+45BCLMVWdenoxy????;?8
1?.
,?)
input_28???????????
? "3?0
.
dense_55"?
dense_55??????????
C__inference_add_23_layer_call_and_return_conditional_losses_2127898?n?k
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
(__inference_add_23_layer_call_fn_2127904?n?k
d?a
_?\
,?)
inputs/0???????????
,?)
inputs/1???????????
? ""?????????????
C__inference_add_24_layer_call_and_return_conditional_losses_2128024?j?g
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
(__inference_add_24_layer_call_fn_2128030?j?g
`?]
[?X
*?'
inputs/0?????????**
*?'
inputs/1?????????**
? " ??????????**?
C__inference_add_25_layer_call_and_return_conditional_losses_2128150?j?g
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
(__inference_add_25_layer_call_fn_2128156?j?g
`?]
[?X
*?'
inputs/0?????????
*?'
inputs/1?????????
? " ???????????
G__inference_conv2d_121_layer_call_and_return_conditional_losses_2127789p !9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
,__inference_conv2d_121_layer_call_fn_2127798c !9?6
/?,
*?'
inputs???????????
? ""?????????????
G__inference_conv2d_122_layer_call_and_return_conditional_losses_2127836p*+9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
,__inference_conv2d_122_layer_call_fn_2127845c*+9?6
/?,
*?'
inputs???????????
? ""?????????????
G__inference_conv2d_123_layer_call_and_return_conditional_losses_2127883p459?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
,__inference_conv2d_123_layer_call_fn_2127892c459?6
/?,
*?'
inputs???????????
? ""?????????????
G__inference_conv2d_124_layer_call_and_return_conditional_losses_2127915lBC7?4
-?*
(?%
inputs?????????**
? "-?*
#? 
0?????????**
? ?
,__inference_conv2d_124_layer_call_fn_2127924_BC7?4
-?*
(?%
inputs?????????**
? " ??????????**?
G__inference_conv2d_125_layer_call_and_return_conditional_losses_2127962lLM7?4
-?*
(?%
inputs?????????**
? "-?*
#? 
0?????????**
? ?
,__inference_conv2d_125_layer_call_fn_2127971_LM7?4
-?*
(?%
inputs?????????**
? " ??????????**?
G__inference_conv2d_126_layer_call_and_return_conditional_losses_2128009lVW7?4
-?*
(?%
inputs?????????**
? "-?*
#? 
0?????????**
? ?
,__inference_conv2d_126_layer_call_fn_2128018_VW7?4
-?*
(?%
inputs?????????**
? " ??????????**?
G__inference_conv2d_127_layer_call_and_return_conditional_losses_2128041lde7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
,__inference_conv2d_127_layer_call_fn_2128050_de7?4
-?*
(?%
inputs?????????
? " ???????????
G__inference_conv2d_128_layer_call_and_return_conditional_losses_2128088lno7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
,__inference_conv2d_128_layer_call_fn_2128097_no7?4
-?*
(?%
inputs?????????
? " ???????????
G__inference_conv2d_129_layer_call_and_return_conditional_losses_2128135lxy7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
,__inference_conv2d_129_layer_call_fn_2128144_xy7?4
-?*
(?%
inputs?????????
? " ???????????
E__inference_dense_54_layer_call_and_return_conditional_losses_2128178^??/?,
%?"
 ?
inputs?????????
? "%?"
?
0????????? 
? 
*__inference_dense_54_layer_call_fn_2128187Q??/?,
%?"
 ?
inputs?????????
? "?????????? ?
E__inference_dense_55_layer_call_and_return_conditional_losses_2128198^??/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????
? 
*__inference_dense_55_layer_call_fn_2128207Q??/?,
%?"
 ?
inputs????????? 
? "???????????
G__inference_dropout_46_layer_call_and_return_conditional_losses_2127810p=?:
3?0
*?'
inputs???????????
p
? "/?,
%?"
0???????????
? ?
G__inference_dropout_46_layer_call_and_return_conditional_losses_2127815p=?:
3?0
*?'
inputs???????????
p 
? "/?,
%?"
0???????????
? ?
,__inference_dropout_46_layer_call_fn_2127820c=?:
3?0
*?'
inputs???????????
p
? ""?????????????
,__inference_dropout_46_layer_call_fn_2127825c=?:
3?0
*?'
inputs???????????
p 
? ""?????????????
G__inference_dropout_47_layer_call_and_return_conditional_losses_2127857p=?:
3?0
*?'
inputs???????????
p
? "/?,
%?"
0???????????
? ?
G__inference_dropout_47_layer_call_and_return_conditional_losses_2127862p=?:
3?0
*?'
inputs???????????
p 
? "/?,
%?"
0???????????
? ?
,__inference_dropout_47_layer_call_fn_2127867c=?:
3?0
*?'
inputs???????????
p
? ""?????????????
,__inference_dropout_47_layer_call_fn_2127872c=?:
3?0
*?'
inputs???????????
p 
? ""?????????????
G__inference_dropout_48_layer_call_and_return_conditional_losses_2127936l;?8
1?.
(?%
inputs?????????**
p
? "-?*
#? 
0?????????**
? ?
G__inference_dropout_48_layer_call_and_return_conditional_losses_2127941l;?8
1?.
(?%
inputs?????????**
p 
? "-?*
#? 
0?????????**
? ?
,__inference_dropout_48_layer_call_fn_2127946_;?8
1?.
(?%
inputs?????????**
p
? " ??????????**?
,__inference_dropout_48_layer_call_fn_2127951_;?8
1?.
(?%
inputs?????????**
p 
? " ??????????**?
G__inference_dropout_49_layer_call_and_return_conditional_losses_2127983l;?8
1?.
(?%
inputs?????????**
p
? "-?*
#? 
0?????????**
? ?
G__inference_dropout_49_layer_call_and_return_conditional_losses_2127988l;?8
1?.
(?%
inputs?????????**
p 
? "-?*
#? 
0?????????**
? ?
,__inference_dropout_49_layer_call_fn_2127993_;?8
1?.
(?%
inputs?????????**
p
? " ??????????**?
,__inference_dropout_49_layer_call_fn_2127998_;?8
1?.
(?%
inputs?????????**
p 
? " ??????????**?
G__inference_dropout_50_layer_call_and_return_conditional_losses_2128062l;?8
1?.
(?%
inputs?????????
p
? "-?*
#? 
0?????????
? ?
G__inference_dropout_50_layer_call_and_return_conditional_losses_2128067l;?8
1?.
(?%
inputs?????????
p 
? "-?*
#? 
0?????????
? ?
,__inference_dropout_50_layer_call_fn_2128072_;?8
1?.
(?%
inputs?????????
p
? " ???????????
,__inference_dropout_50_layer_call_fn_2128077_;?8
1?.
(?%
inputs?????????
p 
? " ???????????
G__inference_dropout_51_layer_call_and_return_conditional_losses_2128109l;?8
1?.
(?%
inputs?????????
p
? "-?*
#? 
0?????????
? ?
G__inference_dropout_51_layer_call_and_return_conditional_losses_2128114l;?8
1?.
(?%
inputs?????????
p 
? "-?*
#? 
0?????????
? ?
,__inference_dropout_51_layer_call_fn_2128119_;?8
1?.
(?%
inputs?????????
p
? " ???????????
,__inference_dropout_51_layer_call_fn_2128124_;?8
1?.
(?%
inputs?????????
p 
? " ???????????
G__inference_flatten_27_layer_call_and_return_conditional_losses_2128162`7?4
-?*
(?%
inputs?????????
? "%?"
?
0?????????
? ?
,__inference_flatten_27_layer_call_fn_2128167S7?4
-?*
(?%
inputs?????????
? "???????????
M__inference_max_pooling2d_49_layer_call_and_return_conditional_losses_2126500?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
2__inference_max_pooling2d_49_layer_call_fn_2126506?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
M__inference_max_pooling2d_50_layer_call_and_return_conditional_losses_2126512?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
2__inference_max_pooling2d_50_layer_call_fn_2126518?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
M__inference_max_pooling2d_51_layer_call_and_return_conditional_losses_2126524?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
2__inference_max_pooling2d_51_layer_call_fn_2126530?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
%__inference_signature_wrapper_2127448? !*+45BCLMVWdenoxy????G?D
? 
=?:
8
input_28,?)
input_28???????????"3?0
.
dense_55"?
dense_55?????????