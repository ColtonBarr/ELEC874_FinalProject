÷ø	
­
B
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
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

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

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
delete_old_dirsbool(
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
dtypetype
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
¾
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
executor_typestring 
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.4.12v2.4.0-49-g85c8b2a817f8ãÉ

conv2d_82/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_82/kernel
}
$conv2d_82/kernel/Read/ReadVariableOpReadVariableOpconv2d_82/kernel*&
_output_shapes
:*
dtype0
t
conv2d_82/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_82/bias
m
"conv2d_82/bias/Read/ReadVariableOpReadVariableOpconv2d_82/bias*
_output_shapes
:*
dtype0

conv2d_83/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_83/kernel
}
$conv2d_83/kernel/Read/ReadVariableOpReadVariableOpconv2d_83/kernel*&
_output_shapes
:*
dtype0
t
conv2d_83/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_83/bias
m
"conv2d_83/bias/Read/ReadVariableOpReadVariableOpconv2d_83/bias*
_output_shapes
:*
dtype0

conv2d_84/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_84/kernel
}
$conv2d_84/kernel/Read/ReadVariableOpReadVariableOpconv2d_84/kernel*&
_output_shapes
:*
dtype0
t
conv2d_84/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_84/bias
m
"conv2d_84/bias/Read/ReadVariableOpReadVariableOpconv2d_84/bias*
_output_shapes
:*
dtype0
{
dense_34/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ä * 
shared_namedense_34/kernel
t
#dense_34/kernel/Read/ReadVariableOpReadVariableOpdense_34/kernel*
_output_shapes
:	ä *
dtype0
r
dense_34/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_34/bias
k
!dense_34/bias/Read/ReadVariableOpReadVariableOpdense_34/bias*
_output_shapes
: *
dtype0
z
dense_35/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_35/kernel
s
#dense_35/kernel/Read/ReadVariableOpReadVariableOpdense_35/kernel*
_output_shapes

: *
dtype0
r
dense_35/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_35/bias
k
!dense_35/bias/Read/ReadVariableOpReadVariableOpdense_35/bias*
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

Adam/conv2d_82/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_82/kernel/m

+Adam/conv2d_82/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_82/kernel/m*&
_output_shapes
:*
dtype0

Adam/conv2d_82/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_82/bias/m
{
)Adam/conv2d_82/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_82/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_83/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_83/kernel/m

+Adam/conv2d_83/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_83/kernel/m*&
_output_shapes
:*
dtype0

Adam/conv2d_83/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_83/bias/m
{
)Adam/conv2d_83/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_83/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_84/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_84/kernel/m

+Adam/conv2d_84/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_84/kernel/m*&
_output_shapes
:*
dtype0

Adam/conv2d_84/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_84/bias/m
{
)Adam/conv2d_84/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_84/bias/m*
_output_shapes
:*
dtype0

Adam/dense_34/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ä *'
shared_nameAdam/dense_34/kernel/m

*Adam/dense_34/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_34/kernel/m*
_output_shapes
:	ä *
dtype0

Adam/dense_34/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_34/bias/m
y
(Adam/dense_34/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_34/bias/m*
_output_shapes
: *
dtype0

Adam/dense_35/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_35/kernel/m

*Adam/dense_35/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_35/kernel/m*
_output_shapes

: *
dtype0

Adam/dense_35/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_35/bias/m
y
(Adam/dense_35/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_35/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_82/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_82/kernel/v

+Adam/conv2d_82/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_82/kernel/v*&
_output_shapes
:*
dtype0

Adam/conv2d_82/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_82/bias/v
{
)Adam/conv2d_82/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_82/bias/v*
_output_shapes
:*
dtype0

Adam/conv2d_83/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_83/kernel/v

+Adam/conv2d_83/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_83/kernel/v*&
_output_shapes
:*
dtype0

Adam/conv2d_83/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_83/bias/v
{
)Adam/conv2d_83/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_83/bias/v*
_output_shapes
:*
dtype0

Adam/conv2d_84/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_84/kernel/v

+Adam/conv2d_84/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_84/kernel/v*&
_output_shapes
:*
dtype0

Adam/conv2d_84/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_84/bias/v
{
)Adam/conv2d_84/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_84/bias/v*
_output_shapes
:*
dtype0

Adam/dense_34/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ä *'
shared_nameAdam/dense_34/kernel/v

*Adam/dense_34/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_34/kernel/v*
_output_shapes
:	ä *
dtype0

Adam/dense_34/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_34/bias/v
y
(Adam/dense_34/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_34/bias/v*
_output_shapes
: *
dtype0

Adam/dense_35/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_35/kernel/v

*Adam/dense_35/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_35/kernel/v*
_output_shapes

: *
dtype0

Adam/dense_35/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_35/bias/v
y
(Adam/dense_35/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_35/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
>
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Æ=
value¼=B¹= B²=
è
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer-5
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
 
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
 	variables
!	keras_api
R
"regularization_losses
#trainable_variables
$	variables
%	keras_api
R
&regularization_losses
'trainable_variables
(	variables
)	keras_api
R
*regularization_losses
+trainable_variables
,	variables
-	keras_api
h

.kernel
/bias
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
ÿ
:iter

;beta_1

<beta_2
	=decay
>learning_ratemwmxmymzm{m|.m}/m~4m5mvvvvvv.v/v4v5v
 
F
0
1
2
3
4
5
.6
/7
48
59
F
0
1
2
3
4
5
.6
/7
48
59
­
regularization_losses

?layers
@layer_metrics
Ametrics
trainable_variables
	variables
Blayer_regularization_losses
Cnon_trainable_variables
 
\Z
VARIABLE_VALUEconv2d_82/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_82/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
regularization_losses

Dlayers
Elayer_metrics
Fmetrics
trainable_variables
	variables
Glayer_regularization_losses
Hnon_trainable_variables
\Z
VARIABLE_VALUEconv2d_83/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_83/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
regularization_losses

Ilayers
Jlayer_metrics
Kmetrics
trainable_variables
	variables
Llayer_regularization_losses
Mnon_trainable_variables
\Z
VARIABLE_VALUEconv2d_84/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_84/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
regularization_losses

Nlayers
Olayer_metrics
Pmetrics
trainable_variables
 	variables
Qlayer_regularization_losses
Rnon_trainable_variables
 
 
 
­
"regularization_losses

Slayers
Tlayer_metrics
Umetrics
#trainable_variables
$	variables
Vlayer_regularization_losses
Wnon_trainable_variables
 
 
 
­
&regularization_losses

Xlayers
Ylayer_metrics
Zmetrics
'trainable_variables
(	variables
[layer_regularization_losses
\non_trainable_variables
 
 
 
­
*regularization_losses

]layers
^layer_metrics
_metrics
+trainable_variables
,	variables
`layer_regularization_losses
anon_trainable_variables
[Y
VARIABLE_VALUEdense_34/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_34/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

.0
/1

.0
/1
­
0regularization_losses

blayers
clayer_metrics
dmetrics
1trainable_variables
2	variables
elayer_regularization_losses
fnon_trainable_variables
[Y
VARIABLE_VALUEdense_35/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_35/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

40
51

40
51
­
6regularization_losses

glayers
hlayer_metrics
imetrics
7trainable_variables
8	variables
jlayer_regularization_losses
knon_trainable_variables
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
?
0
1
2
3
4
5
6
7
	8
 

l0
m1
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
4
	ntotal
	ocount
p	variables
q	keras_api
D
	rtotal
	scount
t
_fn_kwargs
u	variables
v	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

n0
o1

p	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

r0
s1

u	variables
}
VARIABLE_VALUEAdam/conv2d_82/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_82/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_83/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_83/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_84/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_84/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_34/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_34/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_35/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_35/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_82/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_82/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_83/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_83/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_84/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_84/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_34/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_34/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_35/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_35/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_input_18Placeholder*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*&
shape:ÿÿÿÿÿÿÿÿÿ
í
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_18conv2d_82/kernelconv2d_82/biasconv2d_83/kernelconv2d_83/biasconv2d_84/kernelconv2d_84/biasdense_34/kerneldense_34/biasdense_35/kerneldense_35/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_869271
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
±
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_82/kernel/Read/ReadVariableOp"conv2d_82/bias/Read/ReadVariableOp$conv2d_83/kernel/Read/ReadVariableOp"conv2d_83/bias/Read/ReadVariableOp$conv2d_84/kernel/Read/ReadVariableOp"conv2d_84/bias/Read/ReadVariableOp#dense_34/kernel/Read/ReadVariableOp!dense_34/bias/Read/ReadVariableOp#dense_35/kernel/Read/ReadVariableOp!dense_35/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/conv2d_82/kernel/m/Read/ReadVariableOp)Adam/conv2d_82/bias/m/Read/ReadVariableOp+Adam/conv2d_83/kernel/m/Read/ReadVariableOp)Adam/conv2d_83/bias/m/Read/ReadVariableOp+Adam/conv2d_84/kernel/m/Read/ReadVariableOp)Adam/conv2d_84/bias/m/Read/ReadVariableOp*Adam/dense_34/kernel/m/Read/ReadVariableOp(Adam/dense_34/bias/m/Read/ReadVariableOp*Adam/dense_35/kernel/m/Read/ReadVariableOp(Adam/dense_35/bias/m/Read/ReadVariableOp+Adam/conv2d_82/kernel/v/Read/ReadVariableOp)Adam/conv2d_82/bias/v/Read/ReadVariableOp+Adam/conv2d_83/kernel/v/Read/ReadVariableOp)Adam/conv2d_83/bias/v/Read/ReadVariableOp+Adam/conv2d_84/kernel/v/Read/ReadVariableOp)Adam/conv2d_84/bias/v/Read/ReadVariableOp*Adam/dense_34/kernel/v/Read/ReadVariableOp(Adam/dense_34/bias/v/Read/ReadVariableOp*Adam/dense_35/kernel/v/Read/ReadVariableOp(Adam/dense_35/bias/v/Read/ReadVariableOpConst*4
Tin-
+2)	*
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
GPU 2J 8 *(
f#R!
__inference__traced_save_869670
 
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_82/kernelconv2d_82/biasconv2d_83/kernelconv2d_83/biasconv2d_84/kernelconv2d_84/biasdense_34/kerneldense_34/biasdense_35/kerneldense_35/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv2d_82/kernel/mAdam/conv2d_82/bias/mAdam/conv2d_83/kernel/mAdam/conv2d_83/bias/mAdam/conv2d_84/kernel/mAdam/conv2d_84/bias/mAdam/dense_34/kernel/mAdam/dense_34/bias/mAdam/dense_35/kernel/mAdam/dense_35/bias/mAdam/conv2d_82/kernel/vAdam/conv2d_82/bias/vAdam/conv2d_83/kernel/vAdam/conv2d_83/bias/vAdam/conv2d_84/kernel/vAdam/conv2d_84/bias/vAdam/dense_34/kernel/vAdam/dense_34/bias/vAdam/dense_35/kernel/vAdam/dense_35/bias/v*3
Tin,
*2(*
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
GPU 2J 8 *+
f&R$
"__inference__traced_restore_869797ü£
ªS

__inference__traced_save_869670
file_prefix/
+savev2_conv2d_82_kernel_read_readvariableop-
)savev2_conv2d_82_bias_read_readvariableop/
+savev2_conv2d_83_kernel_read_readvariableop-
)savev2_conv2d_83_bias_read_readvariableop/
+savev2_conv2d_84_kernel_read_readvariableop-
)savev2_conv2d_84_bias_read_readvariableop.
*savev2_dense_34_kernel_read_readvariableop,
(savev2_dense_34_bias_read_readvariableop.
*savev2_dense_35_kernel_read_readvariableop,
(savev2_dense_35_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_conv2d_82_kernel_m_read_readvariableop4
0savev2_adam_conv2d_82_bias_m_read_readvariableop6
2savev2_adam_conv2d_83_kernel_m_read_readvariableop4
0savev2_adam_conv2d_83_bias_m_read_readvariableop6
2savev2_adam_conv2d_84_kernel_m_read_readvariableop4
0savev2_adam_conv2d_84_bias_m_read_readvariableop5
1savev2_adam_dense_34_kernel_m_read_readvariableop3
/savev2_adam_dense_34_bias_m_read_readvariableop5
1savev2_adam_dense_35_kernel_m_read_readvariableop3
/savev2_adam_dense_35_bias_m_read_readvariableop6
2savev2_adam_conv2d_82_kernel_v_read_readvariableop4
0savev2_adam_conv2d_82_bias_v_read_readvariableop6
2savev2_adam_conv2d_83_kernel_v_read_readvariableop4
0savev2_adam_conv2d_83_bias_v_read_readvariableop6
2savev2_adam_conv2d_84_kernel_v_read_readvariableop4
0savev2_adam_conv2d_84_bias_v_read_readvariableop5
1savev2_adam_dense_34_kernel_v_read_readvariableop3
/savev2_adam_dense_34_bias_v_read_readvariableop5
1savev2_adam_dense_35_kernel_v_read_readvariableop3
/savev2_adam_dense_35_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
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
Const_1
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
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*
valueB(B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesØ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesá
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_82_kernel_read_readvariableop)savev2_conv2d_82_bias_read_readvariableop+savev2_conv2d_83_kernel_read_readvariableop)savev2_conv2d_83_bias_read_readvariableop+savev2_conv2d_84_kernel_read_readvariableop)savev2_conv2d_84_bias_read_readvariableop*savev2_dense_34_kernel_read_readvariableop(savev2_dense_34_bias_read_readvariableop*savev2_dense_35_kernel_read_readvariableop(savev2_dense_35_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_conv2d_82_kernel_m_read_readvariableop0savev2_adam_conv2d_82_bias_m_read_readvariableop2savev2_adam_conv2d_83_kernel_m_read_readvariableop0savev2_adam_conv2d_83_bias_m_read_readvariableop2savev2_adam_conv2d_84_kernel_m_read_readvariableop0savev2_adam_conv2d_84_bias_m_read_readvariableop1savev2_adam_dense_34_kernel_m_read_readvariableop/savev2_adam_dense_34_bias_m_read_readvariableop1savev2_adam_dense_35_kernel_m_read_readvariableop/savev2_adam_dense_35_bias_m_read_readvariableop2savev2_adam_conv2d_82_kernel_v_read_readvariableop0savev2_adam_conv2d_82_bias_v_read_readvariableop2savev2_adam_conv2d_83_kernel_v_read_readvariableop0savev2_adam_conv2d_83_bias_v_read_readvariableop2savev2_adam_conv2d_84_kernel_v_read_readvariableop0savev2_adam_conv2d_84_bias_v_read_readvariableop1savev2_adam_dense_34_kernel_v_read_readvariableop/savev2_adam_dense_34_bias_v_read_readvariableop1savev2_adam_dense_35_kernel_v_read_readvariableop/savev2_adam_dense_35_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *6
dtypes,
*2(	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
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

identity_1Identity_1:output:0*æ
_input_shapesÔ
Ñ: :::::::	ä : : :: : : : : : : : : :::::::	ä : : ::::::::	ä : : :: 2(
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
::%!

_output_shapes
:	ä : 

_output_shapes
: :$	 

_output_shapes

: : 


_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	ä : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::, (
&
_output_shapes
:: !

_output_shapes
::,"(
&
_output_shapes
:: #

_output_shapes
::%$!

_output_shapes
:	ä : %

_output_shapes
: :$& 

_output_shapes

: : '

_output_shapes
::(

_output_shapes
: 
Ï
ó
$__inference_signature_wrapper_869271
input_18
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity¢StatefulPartitionedCall»
StatefulPartitionedCallStatefulPartitionedCallinput_18unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_8689072
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_18
%
ñ
H__inference_CNN_aug_skip_layer_call_and_return_conditional_losses_869213

inputs
conv2d_82_869184
conv2d_82_869186
conv2d_83_869189
conv2d_83_869191
conv2d_84_869194
conv2d_84_869196
dense_34_869202
dense_34_869204
dense_35_869207
dense_35_869209
identity¢!conv2d_82/StatefulPartitionedCall¢!conv2d_83/StatefulPartitionedCall¢!conv2d_84/StatefulPartitionedCall¢ dense_34/StatefulPartitionedCall¢ dense_35/StatefulPartitionedCall£
!conv2d_82/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_82_869184conv2d_82_869186*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_82_layer_call_and_return_conditional_losses_8689342#
!conv2d_82/StatefulPartitionedCallÇ
!conv2d_83/StatefulPartitionedCallStatefulPartitionedCall*conv2d_82/StatefulPartitionedCall:output:0conv2d_83_869189conv2d_83_869191*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_83_layer_call_and_return_conditional_losses_8689612#
!conv2d_83/StatefulPartitionedCallÇ
!conv2d_84/StatefulPartitionedCallStatefulPartitionedCall*conv2d_83/StatefulPartitionedCall:output:0conv2d_84_869194conv2d_84_869196*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_84_layer_call_and_return_conditional_losses_8689882#
!conv2d_84/StatefulPartitionedCall
add_16/PartitionedCallPartitionedCall*conv2d_84/StatefulPartitionedCall:output:0inputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_add_16_layer_call_and_return_conditional_losses_8690102
add_16/PartitionedCall
 max_pooling2d_33/PartitionedCallPartitionedCalladd_16/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*** 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_8689132"
 max_pooling2d_33/PartitionedCallþ
flatten_17/PartitionedCallPartitionedCall)max_pooling2d_33/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿä* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_17_layer_call_and_return_conditional_losses_8690262
flatten_17/PartitionedCall±
 dense_34/StatefulPartitionedCallStatefulPartitionedCall#flatten_17/PartitionedCall:output:0dense_34_869202dense_34_869204*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_34_layer_call_and_return_conditional_losses_8690452"
 dense_34/StatefulPartitionedCall·
 dense_35/StatefulPartitionedCallStatefulPartitionedCall)dense_34/StatefulPartitionedCall:output:0dense_35_869207dense_35_869209*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_35_layer_call_and_return_conditional_losses_8690722"
 dense_35/StatefulPartitionedCall¯
IdentityIdentity)dense_35/StatefulPartitionedCall:output:0"^conv2d_82/StatefulPartitionedCall"^conv2d_83/StatefulPartitionedCall"^conv2d_84/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ::::::::::2F
!conv2d_82/StatefulPartitionedCall!conv2d_82/StatefulPartitionedCall2F
!conv2d_83/StatefulPartitionedCall!conv2d_83/StatefulPartitionedCall2F
!conv2d_84/StatefulPartitionedCall!conv2d_84/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


*__inference_conv2d_82_layer_call_fn_869427

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallÿ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_82_layer_call_and_return_conditional_losses_8689342
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü
~
)__inference_dense_35_layer_call_fn_869530

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_35_layer_call_and_return_conditional_losses_8690722
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
§%
ó
H__inference_CNN_aug_skip_layer_call_and_return_conditional_losses_869089
input_18
conv2d_82_868945
conv2d_82_868947
conv2d_83_868972
conv2d_83_868974
conv2d_84_868999
conv2d_84_869001
dense_34_869056
dense_34_869058
dense_35_869083
dense_35_869085
identity¢!conv2d_82/StatefulPartitionedCall¢!conv2d_83/StatefulPartitionedCall¢!conv2d_84/StatefulPartitionedCall¢ dense_34/StatefulPartitionedCall¢ dense_35/StatefulPartitionedCall¥
!conv2d_82/StatefulPartitionedCallStatefulPartitionedCallinput_18conv2d_82_868945conv2d_82_868947*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_82_layer_call_and_return_conditional_losses_8689342#
!conv2d_82/StatefulPartitionedCallÇ
!conv2d_83/StatefulPartitionedCallStatefulPartitionedCall*conv2d_82/StatefulPartitionedCall:output:0conv2d_83_868972conv2d_83_868974*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_83_layer_call_and_return_conditional_losses_8689612#
!conv2d_83/StatefulPartitionedCallÇ
!conv2d_84/StatefulPartitionedCallStatefulPartitionedCall*conv2d_83/StatefulPartitionedCall:output:0conv2d_84_868999conv2d_84_869001*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_84_layer_call_and_return_conditional_losses_8689882#
!conv2d_84/StatefulPartitionedCall
add_16/PartitionedCallPartitionedCall*conv2d_84/StatefulPartitionedCall:output:0input_18*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_add_16_layer_call_and_return_conditional_losses_8690102
add_16/PartitionedCall
 max_pooling2d_33/PartitionedCallPartitionedCalladd_16/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*** 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_8689132"
 max_pooling2d_33/PartitionedCallþ
flatten_17/PartitionedCallPartitionedCall)max_pooling2d_33/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿä* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_17_layer_call_and_return_conditional_losses_8690262
flatten_17/PartitionedCall±
 dense_34/StatefulPartitionedCallStatefulPartitionedCall#flatten_17/PartitionedCall:output:0dense_34_869056dense_34_869058*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_34_layer_call_and_return_conditional_losses_8690452"
 dense_34/StatefulPartitionedCall·
 dense_35/StatefulPartitionedCallStatefulPartitionedCall)dense_34/StatefulPartitionedCall:output:0dense_35_869083dense_35_869085*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_35_layer_call_and_return_conditional_losses_8690722"
 dense_35/StatefulPartitionedCall¯
IdentityIdentity)dense_35/StatefulPartitionedCall:output:0"^conv2d_82/StatefulPartitionedCall"^conv2d_83/StatefulPartitionedCall"^conv2d_84/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ::::::::::2F
!conv2d_82/StatefulPartitionedCall!conv2d_82/StatefulPartitionedCall2F
!conv2d_83/StatefulPartitionedCall!conv2d_83/StatefulPartitionedCall2F
!conv2d_84/StatefulPartitionedCall!conv2d_84/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall:[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_18
Ü
l
B__inference_add_16_layer_call_and_return_conditional_losses_869010

inputs
inputs_1
identitya
addAddV2inputsinputs_1*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
adde
IdentityIdentityadd:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*M
_input_shapes<
::ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:YU
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÿ
ü
-__inference_CNN_aug_skip_layer_call_fn_869179
input_18
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity¢StatefulPartitionedCallâ
StatefulPartitionedCallStatefulPartitionedCallinput_18unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_CNN_aug_skip_layer_call_and_return_conditional_losses_8691562
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_18
Û

Þ
E__inference_conv2d_82_layer_call_and_return_conditional_losses_869418

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¥
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu¡
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÿ
ü
-__inference_CNN_aug_skip_layer_call_fn_869236
input_18
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity¢StatefulPartitionedCallâ
StatefulPartitionedCallStatefulPartitionedCallinput_18unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_CNN_aug_skip_layer_call_and_return_conditional_losses_8692132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_18
ö	
Ý
D__inference_dense_35_layer_call_and_return_conditional_losses_869072

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Softmax
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


*__inference_conv2d_84_layer_call_fn_869467

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallÿ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_84_layer_call_and_return_conditional_losses_8689882
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Û

Þ
E__inference_conv2d_82_layer_call_and_return_conditional_losses_868934

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¥
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu¡
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ù
ú
-__inference_CNN_aug_skip_layer_call_fn_869407

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
	unknown_8
identity¢StatefulPartitionedCallà
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_CNN_aug_skip_layer_call_and_return_conditional_losses_8692132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ö	
Ý
D__inference_dense_35_layer_call_and_return_conditional_losses_869521

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Softmax
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¾
b
F__inference_flatten_17_layer_call_and_return_conditional_losses_869026

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿä  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿä2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿä2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ**:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
 
_user_specified_nameinputs
F
è
!__inference__wrapped_model_868907
input_189
5cnn_aug_skip_conv2d_82_conv2d_readvariableop_resource:
6cnn_aug_skip_conv2d_82_biasadd_readvariableop_resource9
5cnn_aug_skip_conv2d_83_conv2d_readvariableop_resource:
6cnn_aug_skip_conv2d_83_biasadd_readvariableop_resource9
5cnn_aug_skip_conv2d_84_conv2d_readvariableop_resource:
6cnn_aug_skip_conv2d_84_biasadd_readvariableop_resource8
4cnn_aug_skip_dense_34_matmul_readvariableop_resource9
5cnn_aug_skip_dense_34_biasadd_readvariableop_resource8
4cnn_aug_skip_dense_35_matmul_readvariableop_resource9
5cnn_aug_skip_dense_35_biasadd_readvariableop_resource
identity¢-CNN_aug_skip/conv2d_82/BiasAdd/ReadVariableOp¢,CNN_aug_skip/conv2d_82/Conv2D/ReadVariableOp¢-CNN_aug_skip/conv2d_83/BiasAdd/ReadVariableOp¢,CNN_aug_skip/conv2d_83/Conv2D/ReadVariableOp¢-CNN_aug_skip/conv2d_84/BiasAdd/ReadVariableOp¢,CNN_aug_skip/conv2d_84/Conv2D/ReadVariableOp¢,CNN_aug_skip/dense_34/BiasAdd/ReadVariableOp¢+CNN_aug_skip/dense_34/MatMul/ReadVariableOp¢,CNN_aug_skip/dense_35/BiasAdd/ReadVariableOp¢+CNN_aug_skip/dense_35/MatMul/ReadVariableOpÚ
,CNN_aug_skip/conv2d_82/Conv2D/ReadVariableOpReadVariableOp5cnn_aug_skip_conv2d_82_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02.
,CNN_aug_skip/conv2d_82/Conv2D/ReadVariableOpì
CNN_aug_skip/conv2d_82/Conv2DConv2Dinput_184CNN_aug_skip/conv2d_82/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
CNN_aug_skip/conv2d_82/Conv2DÑ
-CNN_aug_skip/conv2d_82/BiasAdd/ReadVariableOpReadVariableOp6cnn_aug_skip_conv2d_82_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-CNN_aug_skip/conv2d_82/BiasAdd/ReadVariableOpæ
CNN_aug_skip/conv2d_82/BiasAddBiasAdd&CNN_aug_skip/conv2d_82/Conv2D:output:05CNN_aug_skip/conv2d_82/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
CNN_aug_skip/conv2d_82/BiasAdd§
CNN_aug_skip/conv2d_82/ReluRelu'CNN_aug_skip/conv2d_82/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
CNN_aug_skip/conv2d_82/ReluÚ
,CNN_aug_skip/conv2d_83/Conv2D/ReadVariableOpReadVariableOp5cnn_aug_skip_conv2d_83_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02.
,CNN_aug_skip/conv2d_83/Conv2D/ReadVariableOp
CNN_aug_skip/conv2d_83/Conv2DConv2D)CNN_aug_skip/conv2d_82/Relu:activations:04CNN_aug_skip/conv2d_83/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
CNN_aug_skip/conv2d_83/Conv2DÑ
-CNN_aug_skip/conv2d_83/BiasAdd/ReadVariableOpReadVariableOp6cnn_aug_skip_conv2d_83_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-CNN_aug_skip/conv2d_83/BiasAdd/ReadVariableOpæ
CNN_aug_skip/conv2d_83/BiasAddBiasAdd&CNN_aug_skip/conv2d_83/Conv2D:output:05CNN_aug_skip/conv2d_83/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
CNN_aug_skip/conv2d_83/BiasAdd§
CNN_aug_skip/conv2d_83/ReluRelu'CNN_aug_skip/conv2d_83/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
CNN_aug_skip/conv2d_83/ReluÚ
,CNN_aug_skip/conv2d_84/Conv2D/ReadVariableOpReadVariableOp5cnn_aug_skip_conv2d_84_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02.
,CNN_aug_skip/conv2d_84/Conv2D/ReadVariableOp
CNN_aug_skip/conv2d_84/Conv2DConv2D)CNN_aug_skip/conv2d_83/Relu:activations:04CNN_aug_skip/conv2d_84/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
CNN_aug_skip/conv2d_84/Conv2DÑ
-CNN_aug_skip/conv2d_84/BiasAdd/ReadVariableOpReadVariableOp6cnn_aug_skip_conv2d_84_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-CNN_aug_skip/conv2d_84/BiasAdd/ReadVariableOpæ
CNN_aug_skip/conv2d_84/BiasAddBiasAdd&CNN_aug_skip/conv2d_84/Conv2D:output:05CNN_aug_skip/conv2d_84/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
CNN_aug_skip/conv2d_84/BiasAdd§
CNN_aug_skip/conv2d_84/ReluRelu'CNN_aug_skip/conv2d_84/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
CNN_aug_skip/conv2d_84/Relu¬
CNN_aug_skip/add_16/addAddV2)CNN_aug_skip/conv2d_84/Relu:activations:0input_18*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
CNN_aug_skip/add_16/addã
%CNN_aug_skip/max_pooling2d_33/MaxPoolMaxPoolCNN_aug_skip/add_16/add:z:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ***
ksize
*
paddingVALID*
strides
2'
%CNN_aug_skip/max_pooling2d_33/MaxPool
CNN_aug_skip/flatten_17/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿä  2
CNN_aug_skip/flatten_17/ConstØ
CNN_aug_skip/flatten_17/ReshapeReshape.CNN_aug_skip/max_pooling2d_33/MaxPool:output:0&CNN_aug_skip/flatten_17/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿä2!
CNN_aug_skip/flatten_17/ReshapeÐ
+CNN_aug_skip/dense_34/MatMul/ReadVariableOpReadVariableOp4cnn_aug_skip_dense_34_matmul_readvariableop_resource*
_output_shapes
:	ä *
dtype02-
+CNN_aug_skip/dense_34/MatMul/ReadVariableOp×
CNN_aug_skip/dense_34/MatMulMatMul(CNN_aug_skip/flatten_17/Reshape:output:03CNN_aug_skip/dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
CNN_aug_skip/dense_34/MatMulÎ
,CNN_aug_skip/dense_34/BiasAdd/ReadVariableOpReadVariableOp5cnn_aug_skip_dense_34_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,CNN_aug_skip/dense_34/BiasAdd/ReadVariableOpÙ
CNN_aug_skip/dense_34/BiasAddBiasAdd&CNN_aug_skip/dense_34/MatMul:product:04CNN_aug_skip/dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
CNN_aug_skip/dense_34/BiasAdd
CNN_aug_skip/dense_34/ReluRelu&CNN_aug_skip/dense_34/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
CNN_aug_skip/dense_34/ReluÏ
+CNN_aug_skip/dense_35/MatMul/ReadVariableOpReadVariableOp4cnn_aug_skip_dense_35_matmul_readvariableop_resource*
_output_shapes

: *
dtype02-
+CNN_aug_skip/dense_35/MatMul/ReadVariableOp×
CNN_aug_skip/dense_35/MatMulMatMul(CNN_aug_skip/dense_34/Relu:activations:03CNN_aug_skip/dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
CNN_aug_skip/dense_35/MatMulÎ
,CNN_aug_skip/dense_35/BiasAdd/ReadVariableOpReadVariableOp5cnn_aug_skip_dense_35_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,CNN_aug_skip/dense_35/BiasAdd/ReadVariableOpÙ
CNN_aug_skip/dense_35/BiasAddBiasAdd&CNN_aug_skip/dense_35/MatMul:product:04CNN_aug_skip/dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
CNN_aug_skip/dense_35/BiasAdd£
CNN_aug_skip/dense_35/SoftmaxSoftmax&CNN_aug_skip/dense_35/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
CNN_aug_skip/dense_35/SoftmaxÒ
IdentityIdentity'CNN_aug_skip/dense_35/Softmax:softmax:0.^CNN_aug_skip/conv2d_82/BiasAdd/ReadVariableOp-^CNN_aug_skip/conv2d_82/Conv2D/ReadVariableOp.^CNN_aug_skip/conv2d_83/BiasAdd/ReadVariableOp-^CNN_aug_skip/conv2d_83/Conv2D/ReadVariableOp.^CNN_aug_skip/conv2d_84/BiasAdd/ReadVariableOp-^CNN_aug_skip/conv2d_84/Conv2D/ReadVariableOp-^CNN_aug_skip/dense_34/BiasAdd/ReadVariableOp,^CNN_aug_skip/dense_34/MatMul/ReadVariableOp-^CNN_aug_skip/dense_35/BiasAdd/ReadVariableOp,^CNN_aug_skip/dense_35/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ::::::::::2^
-CNN_aug_skip/conv2d_82/BiasAdd/ReadVariableOp-CNN_aug_skip/conv2d_82/BiasAdd/ReadVariableOp2\
,CNN_aug_skip/conv2d_82/Conv2D/ReadVariableOp,CNN_aug_skip/conv2d_82/Conv2D/ReadVariableOp2^
-CNN_aug_skip/conv2d_83/BiasAdd/ReadVariableOp-CNN_aug_skip/conv2d_83/BiasAdd/ReadVariableOp2\
,CNN_aug_skip/conv2d_83/Conv2D/ReadVariableOp,CNN_aug_skip/conv2d_83/Conv2D/ReadVariableOp2^
-CNN_aug_skip/conv2d_84/BiasAdd/ReadVariableOp-CNN_aug_skip/conv2d_84/BiasAdd/ReadVariableOp2\
,CNN_aug_skip/conv2d_84/Conv2D/ReadVariableOp,CNN_aug_skip/conv2d_84/Conv2D/ReadVariableOp2\
,CNN_aug_skip/dense_34/BiasAdd/ReadVariableOp,CNN_aug_skip/dense_34/BiasAdd/ReadVariableOp2Z
+CNN_aug_skip/dense_34/MatMul/ReadVariableOp+CNN_aug_skip/dense_34/MatMul/ReadVariableOp2\
,CNN_aug_skip/dense_35/BiasAdd/ReadVariableOp,CNN_aug_skip/dense_35/BiasAdd/ReadVariableOp2Z
+CNN_aug_skip/dense_35/MatMul/ReadVariableOp+CNN_aug_skip/dense_35/MatMul/ReadVariableOp:[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_18


*__inference_conv2d_83_layer_call_fn_869447

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallÿ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_83_layer_call_and_return_conditional_losses_8689612
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ	
Ý
D__inference_dense_34_layer_call_and_return_conditional_losses_869045

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ä *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿä::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿä
 
_user_specified_nameinputs
Û

Þ
E__inference_conv2d_84_layer_call_and_return_conditional_losses_869458

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¥
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu¡
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¼7

H__inference_CNN_aug_skip_layer_call_and_return_conditional_losses_869357

inputs,
(conv2d_82_conv2d_readvariableop_resource-
)conv2d_82_biasadd_readvariableop_resource,
(conv2d_83_conv2d_readvariableop_resource-
)conv2d_83_biasadd_readvariableop_resource,
(conv2d_84_conv2d_readvariableop_resource-
)conv2d_84_biasadd_readvariableop_resource+
'dense_34_matmul_readvariableop_resource,
(dense_34_biasadd_readvariableop_resource+
'dense_35_matmul_readvariableop_resource,
(dense_35_biasadd_readvariableop_resource
identity¢ conv2d_82/BiasAdd/ReadVariableOp¢conv2d_82/Conv2D/ReadVariableOp¢ conv2d_83/BiasAdd/ReadVariableOp¢conv2d_83/Conv2D/ReadVariableOp¢ conv2d_84/BiasAdd/ReadVariableOp¢conv2d_84/Conv2D/ReadVariableOp¢dense_34/BiasAdd/ReadVariableOp¢dense_34/MatMul/ReadVariableOp¢dense_35/BiasAdd/ReadVariableOp¢dense_35/MatMul/ReadVariableOp³
conv2d_82/Conv2D/ReadVariableOpReadVariableOp(conv2d_82_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_82/Conv2D/ReadVariableOpÃ
conv2d_82/Conv2DConv2Dinputs'conv2d_82/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv2d_82/Conv2Dª
 conv2d_82/BiasAdd/ReadVariableOpReadVariableOp)conv2d_82_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_82/BiasAdd/ReadVariableOp²
conv2d_82/BiasAddBiasAddconv2d_82/Conv2D:output:0(conv2d_82/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_82/BiasAdd
conv2d_82/ReluReluconv2d_82/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_82/Relu³
conv2d_83/Conv2D/ReadVariableOpReadVariableOp(conv2d_83_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_83/Conv2D/ReadVariableOpÙ
conv2d_83/Conv2DConv2Dconv2d_82/Relu:activations:0'conv2d_83/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv2d_83/Conv2Dª
 conv2d_83/BiasAdd/ReadVariableOpReadVariableOp)conv2d_83_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_83/BiasAdd/ReadVariableOp²
conv2d_83/BiasAddBiasAddconv2d_83/Conv2D:output:0(conv2d_83/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_83/BiasAdd
conv2d_83/ReluReluconv2d_83/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_83/Relu³
conv2d_84/Conv2D/ReadVariableOpReadVariableOp(conv2d_84_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_84/Conv2D/ReadVariableOpÙ
conv2d_84/Conv2DConv2Dconv2d_83/Relu:activations:0'conv2d_84/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv2d_84/Conv2Dª
 conv2d_84/BiasAdd/ReadVariableOpReadVariableOp)conv2d_84_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_84/BiasAdd/ReadVariableOp²
conv2d_84/BiasAddBiasAddconv2d_84/Conv2D:output:0(conv2d_84/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_84/BiasAdd
conv2d_84/ReluReluconv2d_84/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_84/Relu

add_16/addAddV2conv2d_84/Relu:activations:0inputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

add_16/add¼
max_pooling2d_33/MaxPoolMaxPooladd_16/add:z:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ***
ksize
*
paddingVALID*
strides
2
max_pooling2d_33/MaxPoolu
flatten_17/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿä  2
flatten_17/Const¤
flatten_17/ReshapeReshape!max_pooling2d_33/MaxPool:output:0flatten_17/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿä2
flatten_17/Reshape©
dense_34/MatMul/ReadVariableOpReadVariableOp'dense_34_matmul_readvariableop_resource*
_output_shapes
:	ä *
dtype02 
dense_34/MatMul/ReadVariableOp£
dense_34/MatMulMatMulflatten_17/Reshape:output:0&dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_34/MatMul§
dense_34/BiasAdd/ReadVariableOpReadVariableOp(dense_34_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_34/BiasAdd/ReadVariableOp¥
dense_34/BiasAddBiasAdddense_34/MatMul:product:0'dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_34/BiasAdds
dense_34/ReluReludense_34/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_34/Relu¨
dense_35/MatMul/ReadVariableOpReadVariableOp'dense_35_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_35/MatMul/ReadVariableOp£
dense_35/MatMulMatMuldense_34/Relu:activations:0&dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_35/MatMul§
dense_35/BiasAdd/ReadVariableOpReadVariableOp(dense_35_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_35/BiasAdd/ReadVariableOp¥
dense_35/BiasAddBiasAdddense_35/MatMul:product:0'dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_35/BiasAdd|
dense_35/SoftmaxSoftmaxdense_35/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_35/SoftmaxÃ
IdentityIdentitydense_35/Softmax:softmax:0!^conv2d_82/BiasAdd/ReadVariableOp ^conv2d_82/Conv2D/ReadVariableOp!^conv2d_83/BiasAdd/ReadVariableOp ^conv2d_83/Conv2D/ReadVariableOp!^conv2d_84/BiasAdd/ReadVariableOp ^conv2d_84/Conv2D/ReadVariableOp ^dense_34/BiasAdd/ReadVariableOp^dense_34/MatMul/ReadVariableOp ^dense_35/BiasAdd/ReadVariableOp^dense_35/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ::::::::::2D
 conv2d_82/BiasAdd/ReadVariableOp conv2d_82/BiasAdd/ReadVariableOp2B
conv2d_82/Conv2D/ReadVariableOpconv2d_82/Conv2D/ReadVariableOp2D
 conv2d_83/BiasAdd/ReadVariableOp conv2d_83/BiasAdd/ReadVariableOp2B
conv2d_83/Conv2D/ReadVariableOpconv2d_83/Conv2D/ReadVariableOp2D
 conv2d_84/BiasAdd/ReadVariableOp conv2d_84/BiasAdd/ReadVariableOp2B
conv2d_84/Conv2D/ReadVariableOpconv2d_84/Conv2D/ReadVariableOp2B
dense_34/BiasAdd/ReadVariableOpdense_34/BiasAdd/ReadVariableOp2@
dense_34/MatMul/ReadVariableOpdense_34/MatMul/ReadVariableOp2B
dense_35/BiasAdd/ReadVariableOpdense_35/BiasAdd/ReadVariableOp2@
dense_35/MatMul/ReadVariableOpdense_35/MatMul/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
è¤

"__inference__traced_restore_869797
file_prefix%
!assignvariableop_conv2d_82_kernel%
!assignvariableop_1_conv2d_82_bias'
#assignvariableop_2_conv2d_83_kernel%
!assignvariableop_3_conv2d_83_bias'
#assignvariableop_4_conv2d_84_kernel%
!assignvariableop_5_conv2d_84_bias&
"assignvariableop_6_dense_34_kernel$
 assignvariableop_7_dense_34_bias&
"assignvariableop_8_dense_35_kernel$
 assignvariableop_9_dense_35_bias!
assignvariableop_10_adam_iter#
assignvariableop_11_adam_beta_1#
assignvariableop_12_adam_beta_2"
assignvariableop_13_adam_decay*
&assignvariableop_14_adam_learning_rate
assignvariableop_15_total
assignvariableop_16_count
assignvariableop_17_total_1
assignvariableop_18_count_1/
+assignvariableop_19_adam_conv2d_82_kernel_m-
)assignvariableop_20_adam_conv2d_82_bias_m/
+assignvariableop_21_adam_conv2d_83_kernel_m-
)assignvariableop_22_adam_conv2d_83_bias_m/
+assignvariableop_23_adam_conv2d_84_kernel_m-
)assignvariableop_24_adam_conv2d_84_bias_m.
*assignvariableop_25_adam_dense_34_kernel_m,
(assignvariableop_26_adam_dense_34_bias_m.
*assignvariableop_27_adam_dense_35_kernel_m,
(assignvariableop_28_adam_dense_35_bias_m/
+assignvariableop_29_adam_conv2d_82_kernel_v-
)assignvariableop_30_adam_conv2d_82_bias_v/
+assignvariableop_31_adam_conv2d_83_kernel_v-
)assignvariableop_32_adam_conv2d_83_bias_v/
+assignvariableop_33_adam_conv2d_84_kernel_v-
)assignvariableop_34_adam_conv2d_84_bias_v.
*assignvariableop_35_adam_dense_34_kernel_v,
(assignvariableop_36_adam_dense_34_bias_v.
*assignvariableop_37_adam_dense_35_kernel_v,
(assignvariableop_38_adam_dense_35_bias_v
identity_40¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*
valueB(B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesÞ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesö
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*¶
_output_shapes£
 ::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity 
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_82_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¦
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_82_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¨
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_83_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¦
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_83_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¨
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_84_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¦
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_84_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6§
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_34_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¥
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_34_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8§
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_35_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¥
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_35_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_10¥
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_iterIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11§
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12§
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_2Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13¦
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_decayIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14®
AssignVariableOp_14AssignVariableOp&assignvariableop_14_adam_learning_rateIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15¡
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16¡
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17£
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18£
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19³
AssignVariableOp_19AssignVariableOp+assignvariableop_19_adam_conv2d_82_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20±
AssignVariableOp_20AssignVariableOp)assignvariableop_20_adam_conv2d_82_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21³
AssignVariableOp_21AssignVariableOp+assignvariableop_21_adam_conv2d_83_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22±
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_conv2d_83_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23³
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_conv2d_84_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24±
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_conv2d_84_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25²
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_34_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26°
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_dense_34_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27²
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_dense_35_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28°
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_dense_35_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29³
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_conv2d_82_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30±
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_conv2d_82_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31³
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_conv2d_83_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32±
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_conv2d_83_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33³
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_conv2d_84_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34±
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_conv2d_84_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35²
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_dense_34_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36°
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_dense_34_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37²
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_dense_35_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38°
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_dense_35_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_389
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp¸
Identity_39Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_39«
Identity_40IdentityIdentity_39:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_40"#
identity_40Identity_40:output:0*³
_input_shapes¡
: :::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_38AssignVariableOp_382(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
¨
G
+__inference_flatten_17_layer_call_fn_869490

inputs
identityÅ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿä* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_17_layer_call_and_return_conditional_losses_8690262
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿä2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ**:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
 
_user_specified_nameinputs
¯
M
1__inference_max_pooling2d_33_layer_call_fn_868919

inputs
identityí
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_8689132
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾
b
F__inference_flatten_17_layer_call_and_return_conditional_losses_869485

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿä  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿä2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿä2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ**:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
 
_user_specified_nameinputs
Û

Þ
E__inference_conv2d_83_layer_call_and_return_conditional_losses_869438

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¥
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu¡
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ä
n
B__inference_add_16_layer_call_and_return_conditional_losses_869473
inputs_0
inputs_1
identityc
addAddV2inputs_0inputs_1*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
adde
IdentityIdentityadd:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*M
_input_shapes<
::ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
¼7

H__inference_CNN_aug_skip_layer_call_and_return_conditional_losses_869314

inputs,
(conv2d_82_conv2d_readvariableop_resource-
)conv2d_82_biasadd_readvariableop_resource,
(conv2d_83_conv2d_readvariableop_resource-
)conv2d_83_biasadd_readvariableop_resource,
(conv2d_84_conv2d_readvariableop_resource-
)conv2d_84_biasadd_readvariableop_resource+
'dense_34_matmul_readvariableop_resource,
(dense_34_biasadd_readvariableop_resource+
'dense_35_matmul_readvariableop_resource,
(dense_35_biasadd_readvariableop_resource
identity¢ conv2d_82/BiasAdd/ReadVariableOp¢conv2d_82/Conv2D/ReadVariableOp¢ conv2d_83/BiasAdd/ReadVariableOp¢conv2d_83/Conv2D/ReadVariableOp¢ conv2d_84/BiasAdd/ReadVariableOp¢conv2d_84/Conv2D/ReadVariableOp¢dense_34/BiasAdd/ReadVariableOp¢dense_34/MatMul/ReadVariableOp¢dense_35/BiasAdd/ReadVariableOp¢dense_35/MatMul/ReadVariableOp³
conv2d_82/Conv2D/ReadVariableOpReadVariableOp(conv2d_82_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_82/Conv2D/ReadVariableOpÃ
conv2d_82/Conv2DConv2Dinputs'conv2d_82/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv2d_82/Conv2Dª
 conv2d_82/BiasAdd/ReadVariableOpReadVariableOp)conv2d_82_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_82/BiasAdd/ReadVariableOp²
conv2d_82/BiasAddBiasAddconv2d_82/Conv2D:output:0(conv2d_82/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_82/BiasAdd
conv2d_82/ReluReluconv2d_82/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_82/Relu³
conv2d_83/Conv2D/ReadVariableOpReadVariableOp(conv2d_83_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_83/Conv2D/ReadVariableOpÙ
conv2d_83/Conv2DConv2Dconv2d_82/Relu:activations:0'conv2d_83/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv2d_83/Conv2Dª
 conv2d_83/BiasAdd/ReadVariableOpReadVariableOp)conv2d_83_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_83/BiasAdd/ReadVariableOp²
conv2d_83/BiasAddBiasAddconv2d_83/Conv2D:output:0(conv2d_83/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_83/BiasAdd
conv2d_83/ReluReluconv2d_83/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_83/Relu³
conv2d_84/Conv2D/ReadVariableOpReadVariableOp(conv2d_84_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_84/Conv2D/ReadVariableOpÙ
conv2d_84/Conv2DConv2Dconv2d_83/Relu:activations:0'conv2d_84/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv2d_84/Conv2Dª
 conv2d_84/BiasAdd/ReadVariableOpReadVariableOp)conv2d_84_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_84/BiasAdd/ReadVariableOp²
conv2d_84/BiasAddBiasAddconv2d_84/Conv2D:output:0(conv2d_84/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_84/BiasAdd
conv2d_84/ReluReluconv2d_84/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_84/Relu

add_16/addAddV2conv2d_84/Relu:activations:0inputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

add_16/add¼
max_pooling2d_33/MaxPoolMaxPooladd_16/add:z:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ***
ksize
*
paddingVALID*
strides
2
max_pooling2d_33/MaxPoolu
flatten_17/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿä  2
flatten_17/Const¤
flatten_17/ReshapeReshape!max_pooling2d_33/MaxPool:output:0flatten_17/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿä2
flatten_17/Reshape©
dense_34/MatMul/ReadVariableOpReadVariableOp'dense_34_matmul_readvariableop_resource*
_output_shapes
:	ä *
dtype02 
dense_34/MatMul/ReadVariableOp£
dense_34/MatMulMatMulflatten_17/Reshape:output:0&dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_34/MatMul§
dense_34/BiasAdd/ReadVariableOpReadVariableOp(dense_34_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_34/BiasAdd/ReadVariableOp¥
dense_34/BiasAddBiasAdddense_34/MatMul:product:0'dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_34/BiasAdds
dense_34/ReluReludense_34/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_34/Relu¨
dense_35/MatMul/ReadVariableOpReadVariableOp'dense_35_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_35/MatMul/ReadVariableOp£
dense_35/MatMulMatMuldense_34/Relu:activations:0&dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_35/MatMul§
dense_35/BiasAdd/ReadVariableOpReadVariableOp(dense_35_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_35/BiasAdd/ReadVariableOp¥
dense_35/BiasAddBiasAdddense_35/MatMul:product:0'dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_35/BiasAdd|
dense_35/SoftmaxSoftmaxdense_35/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_35/SoftmaxÃ
IdentityIdentitydense_35/Softmax:softmax:0!^conv2d_82/BiasAdd/ReadVariableOp ^conv2d_82/Conv2D/ReadVariableOp!^conv2d_83/BiasAdd/ReadVariableOp ^conv2d_83/Conv2D/ReadVariableOp!^conv2d_84/BiasAdd/ReadVariableOp ^conv2d_84/Conv2D/ReadVariableOp ^dense_34/BiasAdd/ReadVariableOp^dense_34/MatMul/ReadVariableOp ^dense_35/BiasAdd/ReadVariableOp^dense_35/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ::::::::::2D
 conv2d_82/BiasAdd/ReadVariableOp conv2d_82/BiasAdd/ReadVariableOp2B
conv2d_82/Conv2D/ReadVariableOpconv2d_82/Conv2D/ReadVariableOp2D
 conv2d_83/BiasAdd/ReadVariableOp conv2d_83/BiasAdd/ReadVariableOp2B
conv2d_83/Conv2D/ReadVariableOpconv2d_83/Conv2D/ReadVariableOp2D
 conv2d_84/BiasAdd/ReadVariableOp conv2d_84/BiasAdd/ReadVariableOp2B
conv2d_84/Conv2D/ReadVariableOpconv2d_84/Conv2D/ReadVariableOp2B
dense_34/BiasAdd/ReadVariableOpdense_34/BiasAdd/ReadVariableOp2@
dense_34/MatMul/ReadVariableOpdense_34/MatMul/ReadVariableOp2B
dense_35/BiasAdd/ReadVariableOpdense_35/BiasAdd/ReadVariableOp2@
dense_35/MatMul/ReadVariableOpdense_35/MatMul/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Û

Þ
E__inference_conv2d_83_layer_call_and_return_conditional_losses_868961

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¥
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu¡
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ	
Ý
D__inference_dense_34_layer_call_and_return_conditional_losses_869501

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ä *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿä::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿä
 
_user_specified_nameinputs
Ï
S
'__inference_add_16_layer_call_fn_869479
inputs_0
inputs_1
identity×
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_add_16_layer_call_and_return_conditional_losses_8690102
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*M
_input_shapes<
::ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
§%
ó
H__inference_CNN_aug_skip_layer_call_and_return_conditional_losses_869121
input_18
conv2d_82_869092
conv2d_82_869094
conv2d_83_869097
conv2d_83_869099
conv2d_84_869102
conv2d_84_869104
dense_34_869110
dense_34_869112
dense_35_869115
dense_35_869117
identity¢!conv2d_82/StatefulPartitionedCall¢!conv2d_83/StatefulPartitionedCall¢!conv2d_84/StatefulPartitionedCall¢ dense_34/StatefulPartitionedCall¢ dense_35/StatefulPartitionedCall¥
!conv2d_82/StatefulPartitionedCallStatefulPartitionedCallinput_18conv2d_82_869092conv2d_82_869094*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_82_layer_call_and_return_conditional_losses_8689342#
!conv2d_82/StatefulPartitionedCallÇ
!conv2d_83/StatefulPartitionedCallStatefulPartitionedCall*conv2d_82/StatefulPartitionedCall:output:0conv2d_83_869097conv2d_83_869099*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_83_layer_call_and_return_conditional_losses_8689612#
!conv2d_83/StatefulPartitionedCallÇ
!conv2d_84/StatefulPartitionedCallStatefulPartitionedCall*conv2d_83/StatefulPartitionedCall:output:0conv2d_84_869102conv2d_84_869104*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_84_layer_call_and_return_conditional_losses_8689882#
!conv2d_84/StatefulPartitionedCall
add_16/PartitionedCallPartitionedCall*conv2d_84/StatefulPartitionedCall:output:0input_18*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_add_16_layer_call_and_return_conditional_losses_8690102
add_16/PartitionedCall
 max_pooling2d_33/PartitionedCallPartitionedCalladd_16/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*** 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_8689132"
 max_pooling2d_33/PartitionedCallþ
flatten_17/PartitionedCallPartitionedCall)max_pooling2d_33/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿä* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_17_layer_call_and_return_conditional_losses_8690262
flatten_17/PartitionedCall±
 dense_34/StatefulPartitionedCallStatefulPartitionedCall#flatten_17/PartitionedCall:output:0dense_34_869110dense_34_869112*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_34_layer_call_and_return_conditional_losses_8690452"
 dense_34/StatefulPartitionedCall·
 dense_35/StatefulPartitionedCallStatefulPartitionedCall)dense_34/StatefulPartitionedCall:output:0dense_35_869115dense_35_869117*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_35_layer_call_and_return_conditional_losses_8690722"
 dense_35/StatefulPartitionedCall¯
IdentityIdentity)dense_35/StatefulPartitionedCall:output:0"^conv2d_82/StatefulPartitionedCall"^conv2d_83/StatefulPartitionedCall"^conv2d_84/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ::::::::::2F
!conv2d_82/StatefulPartitionedCall!conv2d_82/StatefulPartitionedCall2F
!conv2d_83/StatefulPartitionedCall!conv2d_83/StatefulPartitionedCall2F
!conv2d_84/StatefulPartitionedCall!conv2d_84/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall:[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_18

h
L__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_868913

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Þ
~
)__inference_dense_34_layer_call_fn_869510

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_34_layer_call_and_return_conditional_losses_8690452
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿä::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿä
 
_user_specified_nameinputs
%
ñ
H__inference_CNN_aug_skip_layer_call_and_return_conditional_losses_869156

inputs
conv2d_82_869127
conv2d_82_869129
conv2d_83_869132
conv2d_83_869134
conv2d_84_869137
conv2d_84_869139
dense_34_869145
dense_34_869147
dense_35_869150
dense_35_869152
identity¢!conv2d_82/StatefulPartitionedCall¢!conv2d_83/StatefulPartitionedCall¢!conv2d_84/StatefulPartitionedCall¢ dense_34/StatefulPartitionedCall¢ dense_35/StatefulPartitionedCall£
!conv2d_82/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_82_869127conv2d_82_869129*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_82_layer_call_and_return_conditional_losses_8689342#
!conv2d_82/StatefulPartitionedCallÇ
!conv2d_83/StatefulPartitionedCallStatefulPartitionedCall*conv2d_82/StatefulPartitionedCall:output:0conv2d_83_869132conv2d_83_869134*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_83_layer_call_and_return_conditional_losses_8689612#
!conv2d_83/StatefulPartitionedCallÇ
!conv2d_84/StatefulPartitionedCallStatefulPartitionedCall*conv2d_83/StatefulPartitionedCall:output:0conv2d_84_869137conv2d_84_869139*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_84_layer_call_and_return_conditional_losses_8689882#
!conv2d_84/StatefulPartitionedCall
add_16/PartitionedCallPartitionedCall*conv2d_84/StatefulPartitionedCall:output:0inputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_add_16_layer_call_and_return_conditional_losses_8690102
add_16/PartitionedCall
 max_pooling2d_33/PartitionedCallPartitionedCalladd_16/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*** 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_8689132"
 max_pooling2d_33/PartitionedCallþ
flatten_17/PartitionedCallPartitionedCall)max_pooling2d_33/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿä* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_17_layer_call_and_return_conditional_losses_8690262
flatten_17/PartitionedCall±
 dense_34/StatefulPartitionedCallStatefulPartitionedCall#flatten_17/PartitionedCall:output:0dense_34_869145dense_34_869147*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_34_layer_call_and_return_conditional_losses_8690452"
 dense_34/StatefulPartitionedCall·
 dense_35/StatefulPartitionedCallStatefulPartitionedCall)dense_34/StatefulPartitionedCall:output:0dense_35_869150dense_35_869152*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_35_layer_call_and_return_conditional_losses_8690722"
 dense_35/StatefulPartitionedCall¯
IdentityIdentity)dense_35/StatefulPartitionedCall:output:0"^conv2d_82/StatefulPartitionedCall"^conv2d_83/StatefulPartitionedCall"^conv2d_84/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ::::::::::2F
!conv2d_82/StatefulPartitionedCall!conv2d_82/StatefulPartitionedCall2F
!conv2d_83/StatefulPartitionedCall!conv2d_83/StatefulPartitionedCall2F
!conv2d_84/StatefulPartitionedCall!conv2d_84/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Û

Þ
E__inference_conv2d_84_layer_call_and_return_conditional_losses_868988

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¥
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu¡
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ù
ú
-__inference_CNN_aug_skip_layer_call_fn_869382

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
	unknown_8
identity¢StatefulPartitionedCallà
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_CNN_aug_skip_layer_call_and_return_conditional_losses_8691562
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*·
serving_default£
G
input_18;
serving_default_input_18:0ÿÿÿÿÿÿÿÿÿ<
dense_350
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:ó
O
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer-5
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
_default_save_signature
+&call_and_return_all_conditional_losses
__call__"ÐK
_tf_keras_network´K{"class_name": "Functional", "name": "CNN_aug_skip", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "CNN_aug_skip", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_18"}, "name": "input_18", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_82", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_82", "inbound_nodes": [[["input_18", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_83", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_83", "inbound_nodes": [[["conv2d_82", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_84", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_84", "inbound_nodes": [[["conv2d_83", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_16", "trainable": true, "dtype": "float32"}, "name": "add_16", "inbound_nodes": [[["conv2d_84", 0, 0, {}], ["input_18", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_33", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "name": "max_pooling2d_33", "inbound_nodes": [[["add_16", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_17", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_17", "inbound_nodes": [[["max_pooling2d_33", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_34", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_34", "inbound_nodes": [[["flatten_17", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_35", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_35", "inbound_nodes": [[["dense_34", 0, 0, {}]]]}], "input_layers": [["input_18", 0, 0]], "output_layers": [["dense_35", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "CNN_aug_skip", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_18"}, "name": "input_18", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_82", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_82", "inbound_nodes": [[["input_18", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_83", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_83", "inbound_nodes": [[["conv2d_82", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_84", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_84", "inbound_nodes": [[["conv2d_83", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_16", "trainable": true, "dtype": "float32"}, "name": "add_16", "inbound_nodes": [[["conv2d_84", 0, 0, {}], ["input_18", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_33", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "name": "max_pooling2d_33", "inbound_nodes": [[["add_16", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_17", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_17", "inbound_nodes": [[["max_pooling2d_33", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_34", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_34", "inbound_nodes": [[["flatten_17", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_35", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_35", "inbound_nodes": [[["dense_34", 0, 0, {}]]]}], "input_layers": [["input_18", 0, 0]], "output_layers": [["dense_35", 0, 0]]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ÿ"ü
_tf_keras_input_layerÜ{"class_name": "InputLayer", "name": "input_18", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_18"}}
õ	

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"Î
_tf_keras_layer´{"class_name": "Conv2D", "name": "conv2d_82", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_82", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 1]}}
õ	

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"Î
_tf_keras_layer´{"class_name": "Conv2D", "name": "conv2d_83", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_83", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 8]}}
õ	

kernel
bias
regularization_losses
trainable_variables
 	variables
!	keras_api
+&call_and_return_all_conditional_losses
__call__"Î
_tf_keras_layer´{"class_name": "Conv2D", "name": "conv2d_84", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_84", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 8]}}
¿
"regularization_losses
#trainable_variables
$	variables
%	keras_api
+&call_and_return_all_conditional_losses
__call__"®
_tf_keras_layer{"class_name": "Add", "name": "add_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "add_16", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 128, 128, 1]}, {"class_name": "TensorShape", "items": [null, 128, 128, 1]}]}

&regularization_losses
'trainable_variables
(	variables
)	keras_api
+&call_and_return_all_conditional_losses
__call__"ò
_tf_keras_layerØ{"class_name": "MaxPooling2D", "name": "max_pooling2d_33", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_33", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ê
*regularization_losses
+trainable_variables
,	variables
-	keras_api
+&call_and_return_all_conditional_losses
__call__"Ù
_tf_keras_layer¿{"class_name": "Flatten", "name": "flatten_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_17", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ø

.kernel
/bias
0regularization_losses
1trainable_variables
2	variables
3	keras_api
+&call_and_return_all_conditional_losses
__call__"Ñ
_tf_keras_layer·{"class_name": "Dense", "name": "dense_34", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_34", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1764}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1764]}}
ö

4kernel
5bias
6regularization_losses
7trainable_variables
8	variables
9	keras_api
+&call_and_return_all_conditional_losses
__call__"Ï
_tf_keras_layerµ{"class_name": "Dense", "name": "dense_35", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_35", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}

:iter

;beta_1

<beta_2
	=decay
>learning_ratemwmxmymzm{m|.m}/m~4m5mvvvvvv.v/v4v5v"
	optimizer
 "
trackable_list_wrapper
f
0
1
2
3
4
5
.6
/7
48
59"
trackable_list_wrapper
f
0
1
2
3
4
5
.6
/7
48
59"
trackable_list_wrapper
Î
regularization_losses

?layers
@layer_metrics
Ametrics
trainable_variables
	variables
Blayer_regularization_losses
Cnon_trainable_variables
__call__
_default_save_signature
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
*:(2conv2d_82/kernel
:2conv2d_82/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
°
regularization_losses

Dlayers
Elayer_metrics
Fmetrics
trainable_variables
	variables
Glayer_regularization_losses
Hnon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
*:(2conv2d_83/kernel
:2conv2d_83/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
°
regularization_losses

Ilayers
Jlayer_metrics
Kmetrics
trainable_variables
	variables
Llayer_regularization_losses
Mnon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
*:(2conv2d_84/kernel
:2conv2d_84/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
°
regularization_losses

Nlayers
Olayer_metrics
Pmetrics
trainable_variables
 	variables
Qlayer_regularization_losses
Rnon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
"regularization_losses

Slayers
Tlayer_metrics
Umetrics
#trainable_variables
$	variables
Vlayer_regularization_losses
Wnon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
&regularization_losses

Xlayers
Ylayer_metrics
Zmetrics
'trainable_variables
(	variables
[layer_regularization_losses
\non_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
*regularization_losses

]layers
^layer_metrics
_metrics
+trainable_variables
,	variables
`layer_regularization_losses
anon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
": 	ä 2dense_34/kernel
: 2dense_34/bias
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
°
0regularization_losses

blayers
clayer_metrics
dmetrics
1trainable_variables
2	variables
elayer_regularization_losses
fnon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
!: 2dense_35/kernel
:2dense_35/bias
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
°
6regularization_losses

glayers
hlayer_metrics
imetrics
7trainable_variables
8	variables
jlayer_regularization_losses
knon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
_
0
1
2
3
4
5
6
7
	8"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
l0
m1"
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
»
	ntotal
	ocount
p	variables
q	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
ÿ
	rtotal
	scount
t
_fn_kwargs
u	variables
v	keras_api"¸
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
:  (2total
:  (2count
.
n0
o1"
trackable_list_wrapper
-
p	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
r0
s1"
trackable_list_wrapper
-
u	variables"
_generic_user_object
/:-2Adam/conv2d_82/kernel/m
!:2Adam/conv2d_82/bias/m
/:-2Adam/conv2d_83/kernel/m
!:2Adam/conv2d_83/bias/m
/:-2Adam/conv2d_84/kernel/m
!:2Adam/conv2d_84/bias/m
':%	ä 2Adam/dense_34/kernel/m
 : 2Adam/dense_34/bias/m
&:$ 2Adam/dense_35/kernel/m
 :2Adam/dense_35/bias/m
/:-2Adam/conv2d_82/kernel/v
!:2Adam/conv2d_82/bias/v
/:-2Adam/conv2d_83/kernel/v
!:2Adam/conv2d_83/bias/v
/:-2Adam/conv2d_84/kernel/v
!:2Adam/conv2d_84/bias/v
':%	ä 2Adam/dense_34/kernel/v
 : 2Adam/dense_34/bias/v
&:$ 2Adam/dense_35/kernel/v
 :2Adam/dense_35/bias/v
ê2ç
!__inference__wrapped_model_868907Á
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *1¢.
,)
input_18ÿÿÿÿÿÿÿÿÿ
î2ë
H__inference_CNN_aug_skip_layer_call_and_return_conditional_losses_869357
H__inference_CNN_aug_skip_layer_call_and_return_conditional_losses_869314
H__inference_CNN_aug_skip_layer_call_and_return_conditional_losses_869089
H__inference_CNN_aug_skip_layer_call_and_return_conditional_losses_869121À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2ÿ
-__inference_CNN_aug_skip_layer_call_fn_869407
-__inference_CNN_aug_skip_layer_call_fn_869179
-__inference_CNN_aug_skip_layer_call_fn_869236
-__inference_CNN_aug_skip_layer_call_fn_869382À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ï2ì
E__inference_conv2d_82_layer_call_and_return_conditional_losses_869418¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_conv2d_82_layer_call_fn_869427¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_conv2d_83_layer_call_and_return_conditional_losses_869438¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_conv2d_83_layer_call_fn_869447¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_conv2d_84_layer_call_and_return_conditional_losses_869458¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_conv2d_84_layer_call_fn_869467¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_add_16_layer_call_and_return_conditional_losses_869473¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_add_16_layer_call_fn_869479¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
´2±
L__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_868913à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
1__inference_max_pooling2d_33_layer_call_fn_868919à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ð2í
F__inference_flatten_17_layer_call_and_return_conditional_losses_869485¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_flatten_17_layer_call_fn_869490¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_dense_34_layer_call_and_return_conditional_losses_869501¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_dense_34_layer_call_fn_869510¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_dense_35_layer_call_and_return_conditional_losses_869521¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_dense_35_layer_call_fn_869530¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÌBÉ
$__inference_signature_wrapper_869271input_18"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 Ä
H__inference_CNN_aug_skip_layer_call_and_return_conditional_losses_869089x
./45C¢@
9¢6
,)
input_18ÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ä
H__inference_CNN_aug_skip_layer_call_and_return_conditional_losses_869121x
./45C¢@
9¢6
,)
input_18ÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Â
H__inference_CNN_aug_skip_layer_call_and_return_conditional_losses_869314v
./45A¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Â
H__inference_CNN_aug_skip_layer_call_and_return_conditional_losses_869357v
./45A¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
-__inference_CNN_aug_skip_layer_call_fn_869179k
./45C¢@
9¢6
,)
input_18ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_CNN_aug_skip_layer_call_fn_869236k
./45C¢@
9¢6
,)
input_18ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_CNN_aug_skip_layer_call_fn_869382i
./45A¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_CNN_aug_skip_layer_call_fn_869407i
./45A¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ£
!__inference__wrapped_model_868907~
./45;¢8
1¢.
,)
input_18ÿÿÿÿÿÿÿÿÿ
ª "3ª0
.
dense_35"
dense_35ÿÿÿÿÿÿÿÿÿè
B__inference_add_16_layer_call_and_return_conditional_losses_869473¡n¢k
d¢a
_\
,)
inputs/0ÿÿÿÿÿÿÿÿÿ
,)
inputs/1ÿÿÿÿÿÿÿÿÿ
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 À
'__inference_add_16_layer_call_fn_869479n¢k
d¢a
_\
,)
inputs/0ÿÿÿÿÿÿÿÿÿ
,)
inputs/1ÿÿÿÿÿÿÿÿÿ
ª ""ÿÿÿÿÿÿÿÿÿ¹
E__inference_conv2d_82_layer_call_and_return_conditional_losses_869418p9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 
*__inference_conv2d_82_layer_call_fn_869427c9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ
ª ""ÿÿÿÿÿÿÿÿÿ¹
E__inference_conv2d_83_layer_call_and_return_conditional_losses_869438p9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 
*__inference_conv2d_83_layer_call_fn_869447c9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ
ª ""ÿÿÿÿÿÿÿÿÿ¹
E__inference_conv2d_84_layer_call_and_return_conditional_losses_869458p9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 
*__inference_conv2d_84_layer_call_fn_869467c9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ
ª ""ÿÿÿÿÿÿÿÿÿ¥
D__inference_dense_34_layer_call_and_return_conditional_losses_869501]./0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿä
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 }
)__inference_dense_34_layer_call_fn_869510P./0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿä
ª "ÿÿÿÿÿÿÿÿÿ ¤
D__inference_dense_35_layer_call_and_return_conditional_losses_869521\45/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 |
)__inference_dense_35_layer_call_fn_869530O45/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ«
F__inference_flatten_17_layer_call_and_return_conditional_losses_869485a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ**
ª "&¢#

0ÿÿÿÿÿÿÿÿÿä
 
+__inference_flatten_17_layer_call_fn_869490T7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ**
ª "ÿÿÿÿÿÿÿÿÿäï
L__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_868913R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ç
1__inference_max_pooling2d_33_layer_call_fn_868919R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ³
$__inference_signature_wrapper_869271
./45G¢D
¢ 
=ª:
8
input_18,)
input_18ÿÿÿÿÿÿÿÿÿ"3ª0
.
dense_35"
dense_35ÿÿÿÿÿÿÿÿÿ