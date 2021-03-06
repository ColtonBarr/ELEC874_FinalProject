߮
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
conv2d_95/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_95/kernel
}
$conv2d_95/kernel/Read/ReadVariableOpReadVariableOpconv2d_95/kernel*&
_output_shapes
:*
dtype0
t
conv2d_95/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_95/bias
m
"conv2d_95/bias/Read/ReadVariableOpReadVariableOpconv2d_95/bias*
_output_shapes
:*
dtype0
?
batch_normalization_36/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_36/gamma
?
0batch_normalization_36/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_36/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_36/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_36/beta
?
/batch_normalization_36/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_36/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_36/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_36/moving_mean
?
6batch_normalization_36/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_36/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_36/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_36/moving_variance
?
:batch_normalization_36/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_36/moving_variance*
_output_shapes
:*
dtype0
?
conv2d_96/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_96/kernel
}
$conv2d_96/kernel/Read/ReadVariableOpReadVariableOpconv2d_96/kernel*&
_output_shapes
:*
dtype0
t
conv2d_96/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_96/bias
m
"conv2d_96/bias/Read/ReadVariableOpReadVariableOpconv2d_96/bias*
_output_shapes
:*
dtype0
?
batch_normalization_37/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_37/gamma
?
0batch_normalization_37/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_37/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_37/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_37/beta
?
/batch_normalization_37/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_37/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_37/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_37/moving_mean
?
6batch_normalization_37/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_37/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_37/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_37/moving_variance
?
:batch_normalization_37/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_37/moving_variance*
_output_shapes
:*
dtype0
{
dense_44/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?n * 
shared_namedense_44/kernel
t
#dense_44/kernel/Read/ReadVariableOpReadVariableOpdense_44/kernel*
_output_shapes
:	?n *
dtype0
r
dense_44/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_44/bias
k
!dense_44/bias/Read/ReadVariableOpReadVariableOpdense_44/bias*
_output_shapes
: *
dtype0
z
dense_45/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_45/kernel
s
#dense_45/kernel/Read/ReadVariableOpReadVariableOpdense_45/kernel*
_output_shapes

: *
dtype0
r
dense_45/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_45/bias
k
!dense_45/bias/Read/ReadVariableOpReadVariableOpdense_45/bias*
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
Adam/conv2d_95/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_95/kernel/m
?
+Adam/conv2d_95/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_95/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_95/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_95/bias/m
{
)Adam/conv2d_95/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_95/bias/m*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_36/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_36/gamma/m
?
7Adam/batch_normalization_36/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_36/gamma/m*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_36/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_36/beta/m
?
6Adam/batch_normalization_36/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_36/beta/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_96/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_96/kernel/m
?
+Adam/conv2d_96/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_96/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_96/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_96/bias/m
{
)Adam/conv2d_96/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_96/bias/m*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_37/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_37/gamma/m
?
7Adam/batch_normalization_37/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_37/gamma/m*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_37/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_37/beta/m
?
6Adam/batch_normalization_37/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_37/beta/m*
_output_shapes
:*
dtype0
?
Adam/dense_44/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?n *'
shared_nameAdam/dense_44/kernel/m
?
*Adam/dense_44/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_44/kernel/m*
_output_shapes
:	?n *
dtype0
?
Adam/dense_44/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_44/bias/m
y
(Adam/dense_44/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_44/bias/m*
_output_shapes
: *
dtype0
?
Adam/dense_45/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_45/kernel/m
?
*Adam/dense_45/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_45/kernel/m*
_output_shapes

: *
dtype0
?
Adam/dense_45/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_45/bias/m
y
(Adam/dense_45/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_45/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_95/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_95/kernel/v
?
+Adam/conv2d_95/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_95/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_95/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_95/bias/v
{
)Adam/conv2d_95/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_95/bias/v*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_36/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_36/gamma/v
?
7Adam/batch_normalization_36/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_36/gamma/v*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_36/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_36/beta/v
?
6Adam/batch_normalization_36/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_36/beta/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_96/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_96/kernel/v
?
+Adam/conv2d_96/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_96/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_96/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_96/bias/v
{
)Adam/conv2d_96/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_96/bias/v*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_37/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_37/gamma/v
?
7Adam/batch_normalization_37/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_37/gamma/v*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_37/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_37/beta/v
?
6Adam/batch_normalization_37/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_37/beta/v*
_output_shapes
:*
dtype0
?
Adam/dense_44/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?n *'
shared_nameAdam/dense_44/kernel/v
?
*Adam/dense_44/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_44/kernel/v*
_output_shapes
:	?n *
dtype0
?
Adam/dense_44/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_44/bias/v
y
(Adam/dense_44/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_44/bias/v*
_output_shapes
: *
dtype0
?
Adam/dense_45/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_45/kernel/v
?
*Adam/dense_45/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_45/kernel/v*
_output_shapes

: *
dtype0
?
Adam/dense_45/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_45/bias/v
y
(Adam/dense_45/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_45/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?P
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?O
value?OB?O B?O
?
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
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
 
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
?
axis
	gamma
beta
moving_mean
moving_variance
regularization_losses
trainable_variables
	variables
 	keras_api
R
!regularization_losses
"trainable_variables
#	variables
$	keras_api
h

%kernel
&bias
'regularization_losses
(trainable_variables
)	variables
*	keras_api
?
+axis
	,gamma
-beta
.moving_mean
/moving_variance
0regularization_losses
1trainable_variables
2	variables
3	keras_api
R
4regularization_losses
5trainable_variables
6	variables
7	keras_api
R
8regularization_losses
9trainable_variables
:	variables
;	keras_api
R
<regularization_losses
=trainable_variables
>	variables
?	keras_api
h

@kernel
Abias
Bregularization_losses
Ctrainable_variables
D	variables
E	keras_api
h

Fkernel
Gbias
Hregularization_losses
Itrainable_variables
J	variables
K	keras_api
?
Liter

Mbeta_1

Nbeta_2
	Odecay
Plearning_ratem?m?m?m?%m?&m?,m?-m?@m?Am?Fm?Gm?v?v?v?v?%v?&v?,v?-v?@v?Av?Fv?Gv?
 
V
0
1
2
3
%4
&5
,6
-7
@8
A9
F10
G11
v
0
1
2
3
4
5
%6
&7
,8
-9
.10
/11
@12
A13
F14
G15
?
regularization_losses

Qlayers
Rlayer_metrics
Smetrics
trainable_variables
	variables
Tlayer_regularization_losses
Unon_trainable_variables
 
\Z
VARIABLE_VALUEconv2d_95/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_95/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
regularization_losses

Vlayers
Wlayer_metrics
Xmetrics
trainable_variables
	variables
Ylayer_regularization_losses
Znon_trainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_36/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_36/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_36/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_36/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
2
3
?
regularization_losses

[layers
\layer_metrics
]metrics
trainable_variables
	variables
^layer_regularization_losses
_non_trainable_variables
 
 
 
?
!regularization_losses

`layers
alayer_metrics
bmetrics
"trainable_variables
#	variables
clayer_regularization_losses
dnon_trainable_variables
\Z
VARIABLE_VALUEconv2d_96/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_96/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

%0
&1

%0
&1
?
'regularization_losses

elayers
flayer_metrics
gmetrics
(trainable_variables
)	variables
hlayer_regularization_losses
inon_trainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_37/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_37/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_37/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_37/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

,0
-1

,0
-1
.2
/3
?
0regularization_losses

jlayers
klayer_metrics
lmetrics
1trainable_variables
2	variables
mlayer_regularization_losses
nnon_trainable_variables
 
 
 
?
4regularization_losses

olayers
player_metrics
qmetrics
5trainable_variables
6	variables
rlayer_regularization_losses
snon_trainable_variables
 
 
 
?
8regularization_losses

tlayers
ulayer_metrics
vmetrics
9trainable_variables
:	variables
wlayer_regularization_losses
xnon_trainable_variables
 
 
 
?
<regularization_losses

ylayers
zlayer_metrics
{metrics
=trainable_variables
>	variables
|layer_regularization_losses
}non_trainable_variables
[Y
VARIABLE_VALUEdense_44/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_44/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

@0
A1

@0
A1
?
Bregularization_losses

~layers
layer_metrics
?metrics
Ctrainable_variables
D	variables
 ?layer_regularization_losses
?non_trainable_variables
[Y
VARIABLE_VALUEdense_45/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_45/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

F0
G1

F0
G1
?
Hregularization_losses
?layers
?layer_metrics
?metrics
Itrainable_variables
J	variables
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
N
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
 

?0
?1
 

0
1
.2
/3
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
0
1
 
 
 
 
 
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
.0
/1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
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
}
VARIABLE_VALUEAdam/conv2d_95/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_95/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_36/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_36/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_96/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_96/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_37/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_37/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_44/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_44/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_45/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_45/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_95/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_95/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_36/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_36/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_96/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_96/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_37/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_37/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_44/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_44/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_45/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_45/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_23Placeholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_23conv2d_95/kernelconv2d_95/biasbatch_normalization_36/gammabatch_normalization_36/beta"batch_normalization_36/moving_mean&batch_normalization_36/moving_varianceconv2d_96/kernelconv2d_96/biasbatch_normalization_37/gammabatch_normalization_37/beta"batch_normalization_37/moving_mean&batch_normalization_37/moving_variancedense_44/kerneldense_44/biasdense_45/kerneldense_45/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_1393265
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_95/kernel/Read/ReadVariableOp"conv2d_95/bias/Read/ReadVariableOp0batch_normalization_36/gamma/Read/ReadVariableOp/batch_normalization_36/beta/Read/ReadVariableOp6batch_normalization_36/moving_mean/Read/ReadVariableOp:batch_normalization_36/moving_variance/Read/ReadVariableOp$conv2d_96/kernel/Read/ReadVariableOp"conv2d_96/bias/Read/ReadVariableOp0batch_normalization_37/gamma/Read/ReadVariableOp/batch_normalization_37/beta/Read/ReadVariableOp6batch_normalization_37/moving_mean/Read/ReadVariableOp:batch_normalization_37/moving_variance/Read/ReadVariableOp#dense_44/kernel/Read/ReadVariableOp!dense_44/bias/Read/ReadVariableOp#dense_45/kernel/Read/ReadVariableOp!dense_45/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/conv2d_95/kernel/m/Read/ReadVariableOp)Adam/conv2d_95/bias/m/Read/ReadVariableOp7Adam/batch_normalization_36/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_36/beta/m/Read/ReadVariableOp+Adam/conv2d_96/kernel/m/Read/ReadVariableOp)Adam/conv2d_96/bias/m/Read/ReadVariableOp7Adam/batch_normalization_37/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_37/beta/m/Read/ReadVariableOp*Adam/dense_44/kernel/m/Read/ReadVariableOp(Adam/dense_44/bias/m/Read/ReadVariableOp*Adam/dense_45/kernel/m/Read/ReadVariableOp(Adam/dense_45/bias/m/Read/ReadVariableOp+Adam/conv2d_95/kernel/v/Read/ReadVariableOp)Adam/conv2d_95/bias/v/Read/ReadVariableOp7Adam/batch_normalization_36/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_36/beta/v/Read/ReadVariableOp+Adam/conv2d_96/kernel/v/Read/ReadVariableOp)Adam/conv2d_96/bias/v/Read/ReadVariableOp7Adam/batch_normalization_37/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_37/beta/v/Read/ReadVariableOp*Adam/dense_44/kernel/v/Read/ReadVariableOp(Adam/dense_44/bias/v/Read/ReadVariableOp*Adam/dense_45/kernel/v/Read/ReadVariableOp(Adam/dense_45/bias/v/Read/ReadVariableOpConst*>
Tin7
523	*
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
 __inference__traced_save_1394058
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_95/kernelconv2d_95/biasbatch_normalization_36/gammabatch_normalization_36/beta"batch_normalization_36/moving_mean&batch_normalization_36/moving_varianceconv2d_96/kernelconv2d_96/biasbatch_normalization_37/gammabatch_normalization_37/beta"batch_normalization_37/moving_mean&batch_normalization_37/moving_variancedense_44/kerneldense_44/biasdense_45/kerneldense_45/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv2d_95/kernel/mAdam/conv2d_95/bias/m#Adam/batch_normalization_36/gamma/m"Adam/batch_normalization_36/beta/mAdam/conv2d_96/kernel/mAdam/conv2d_96/bias/m#Adam/batch_normalization_37/gamma/m"Adam/batch_normalization_37/beta/mAdam/dense_44/kernel/mAdam/dense_44/bias/mAdam/dense_45/kernel/mAdam/dense_45/bias/mAdam/conv2d_95/kernel/vAdam/conv2d_95/bias/v#Adam/batch_normalization_36/gamma/v"Adam/batch_normalization_36/beta/vAdam/conv2d_96/kernel/vAdam/conv2d_96/bias/v#Adam/batch_normalization_37/gamma/v"Adam/batch_normalization_37/beta/vAdam/dense_44/kernel/vAdam/dense_44/bias/vAdam/dense_45/kernel/vAdam/dense_45/bias/v*=
Tin6
422*
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
#__inference__traced_restore_1394215??
?
e
G__inference_dropout_36_layer_call_and_return_conditional_losses_1392792

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
,__inference_dropout_36_layer_call_fn_1393662

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
G__inference_dropout_36_layer_call_and_return_conditional_losses_13927922
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
8__inference_batch_normalization_37_layer_call_fn_1393810

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
S__inference_batch_normalization_37_layer_call_and_return_conditional_losses_13928692
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
?
H
,__inference_dropout_37_layer_call_fn_1393837

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
G__inference_dropout_37_layer_call_and_return_conditional_losses_13929222
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
?f
?
 __inference__traced_save_1394058
file_prefix/
+savev2_conv2d_95_kernel_read_readvariableop-
)savev2_conv2d_95_bias_read_readvariableop;
7savev2_batch_normalization_36_gamma_read_readvariableop:
6savev2_batch_normalization_36_beta_read_readvariableopA
=savev2_batch_normalization_36_moving_mean_read_readvariableopE
Asavev2_batch_normalization_36_moving_variance_read_readvariableop/
+savev2_conv2d_96_kernel_read_readvariableop-
)savev2_conv2d_96_bias_read_readvariableop;
7savev2_batch_normalization_37_gamma_read_readvariableop:
6savev2_batch_normalization_37_beta_read_readvariableopA
=savev2_batch_normalization_37_moving_mean_read_readvariableopE
Asavev2_batch_normalization_37_moving_variance_read_readvariableop.
*savev2_dense_44_kernel_read_readvariableop,
(savev2_dense_44_bias_read_readvariableop.
*savev2_dense_45_kernel_read_readvariableop,
(savev2_dense_45_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_conv2d_95_kernel_m_read_readvariableop4
0savev2_adam_conv2d_95_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_36_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_36_beta_m_read_readvariableop6
2savev2_adam_conv2d_96_kernel_m_read_readvariableop4
0savev2_adam_conv2d_96_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_37_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_37_beta_m_read_readvariableop5
1savev2_adam_dense_44_kernel_m_read_readvariableop3
/savev2_adam_dense_44_bias_m_read_readvariableop5
1savev2_adam_dense_45_kernel_m_read_readvariableop3
/savev2_adam_dense_45_bias_m_read_readvariableop6
2savev2_adam_conv2d_95_kernel_v_read_readvariableop4
0savev2_adam_conv2d_95_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_36_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_36_beta_v_read_readvariableop6
2savev2_adam_conv2d_96_kernel_v_read_readvariableop4
0savev2_adam_conv2d_96_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_37_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_37_beta_v_read_readvariableop5
1savev2_adam_dense_44_kernel_v_read_readvariableop3
/savev2_adam_dense_44_bias_v_read_readvariableop5
1savev2_adam_dense_45_kernel_v_read_readvariableop3
/savev2_adam_dense_45_bias_v_read_readvariableop
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*?
value?B?2B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_95_kernel_read_readvariableop)savev2_conv2d_95_bias_read_readvariableop7savev2_batch_normalization_36_gamma_read_readvariableop6savev2_batch_normalization_36_beta_read_readvariableop=savev2_batch_normalization_36_moving_mean_read_readvariableopAsavev2_batch_normalization_36_moving_variance_read_readvariableop+savev2_conv2d_96_kernel_read_readvariableop)savev2_conv2d_96_bias_read_readvariableop7savev2_batch_normalization_37_gamma_read_readvariableop6savev2_batch_normalization_37_beta_read_readvariableop=savev2_batch_normalization_37_moving_mean_read_readvariableopAsavev2_batch_normalization_37_moving_variance_read_readvariableop*savev2_dense_44_kernel_read_readvariableop(savev2_dense_44_bias_read_readvariableop*savev2_dense_45_kernel_read_readvariableop(savev2_dense_45_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_conv2d_95_kernel_m_read_readvariableop0savev2_adam_conv2d_95_bias_m_read_readvariableop>savev2_adam_batch_normalization_36_gamma_m_read_readvariableop=savev2_adam_batch_normalization_36_beta_m_read_readvariableop2savev2_adam_conv2d_96_kernel_m_read_readvariableop0savev2_adam_conv2d_96_bias_m_read_readvariableop>savev2_adam_batch_normalization_37_gamma_m_read_readvariableop=savev2_adam_batch_normalization_37_beta_m_read_readvariableop1savev2_adam_dense_44_kernel_m_read_readvariableop/savev2_adam_dense_44_bias_m_read_readvariableop1savev2_adam_dense_45_kernel_m_read_readvariableop/savev2_adam_dense_45_bias_m_read_readvariableop2savev2_adam_conv2d_95_kernel_v_read_readvariableop0savev2_adam_conv2d_95_bias_v_read_readvariableop>savev2_adam_batch_normalization_36_gamma_v_read_readvariableop=savev2_adam_batch_normalization_36_beta_v_read_readvariableop2savev2_adam_conv2d_96_kernel_v_read_readvariableop0savev2_adam_conv2d_96_bias_v_read_readvariableop>savev2_adam_batch_normalization_37_gamma_v_read_readvariableop=savev2_adam_batch_normalization_37_beta_v_read_readvariableop1savev2_adam_dense_44_kernel_v_read_readvariableop/savev2_adam_dense_44_bias_v_read_readvariableop1savev2_adam_dense_45_kernel_v_read_readvariableop/savev2_adam_dense_45_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *@
dtypes6
422	2
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :::::::::::::	?n : : :: : : : : : : : : :::::::::	?n : : ::::::::::	?n : : :: 2(
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
::%!

_output_shapes
:	?n : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::  

_output_shapes
:: !

_output_shapes
::%"!

_output_shapes
:	?n : #

_output_shapes
: :$$ 

_output_shapes

: : %

_output_shapes
::,&(
&
_output_shapes
:: '

_output_shapes
:: (

_output_shapes
:: )

_output_shapes
::,*(
&
_output_shapes
:: +

_output_shapes
:: ,

_output_shapes
:: -

_output_shapes
::%.!

_output_shapes
:	?n : /

_output_shapes
: :$0 

_output_shapes

: : 1

_output_shapes
::2

_output_shapes
: 
?
?
S__inference_batch_normalization_37_layer_call_and_return_conditional_losses_1392617

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
G__inference_dropout_37_layer_call_and_return_conditional_losses_1393822

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
E__inference_dense_44_layer_call_and_return_conditional_losses_1393859

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?n *
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
:??????????n::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????n
 
_user_specified_nameinputs
?5
?
O__inference_CNN_aug_bnorm_drop_layer_call_and_return_conditional_losses_1393100

inputs
conv2d_95_1393057
conv2d_95_1393059"
batch_normalization_36_1393062"
batch_normalization_36_1393064"
batch_normalization_36_1393066"
batch_normalization_36_1393068
conv2d_96_1393072
conv2d_96_1393074"
batch_normalization_37_1393077"
batch_normalization_37_1393079"
batch_normalization_37_1393081"
batch_normalization_37_1393083
dense_44_1393089
dense_44_1393091
dense_45_1393094
dense_45_1393096
identity??.batch_normalization_36/StatefulPartitionedCall?.batch_normalization_37/StatefulPartitionedCall?!conv2d_95/StatefulPartitionedCall?!conv2d_96/StatefulPartitionedCall? dense_44/StatefulPartitionedCall? dense_45/StatefulPartitionedCall?"dropout_36/StatefulPartitionedCall?"dropout_37/StatefulPartitionedCall?
!conv2d_95/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_95_1393057conv2d_95_1393059*
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
GPU 2J 8? *O
fJRH
F__inference_conv2d_95_layer_call_and_return_conditional_losses_13926862#
!conv2d_95/StatefulPartitionedCall?
.batch_normalization_36/StatefulPartitionedCallStatefulPartitionedCall*conv2d_95/StatefulPartitionedCall:output:0batch_normalization_36_1393062batch_normalization_36_1393064batch_normalization_36_1393066batch_normalization_36_1393068*
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
S__inference_batch_normalization_36_layer_call_and_return_conditional_losses_139272120
.batch_normalization_36/StatefulPartitionedCall?
"dropout_36/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_36/StatefulPartitionedCall:output:0*
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
G__inference_dropout_36_layer_call_and_return_conditional_losses_13927872$
"dropout_36/StatefulPartitionedCall?
!conv2d_96/StatefulPartitionedCallStatefulPartitionedCall+dropout_36/StatefulPartitionedCall:output:0conv2d_96_1393072conv2d_96_1393074*
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
GPU 2J 8? *O
fJRH
F__inference_conv2d_96_layer_call_and_return_conditional_losses_13928162#
!conv2d_96/StatefulPartitionedCall?
.batch_normalization_37/StatefulPartitionedCallStatefulPartitionedCall*conv2d_96/StatefulPartitionedCall:output:0batch_normalization_37_1393077batch_normalization_37_1393079batch_normalization_37_1393081batch_normalization_37_1393083*
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
S__inference_batch_normalization_37_layer_call_and_return_conditional_losses_139285120
.batch_normalization_37/StatefulPartitionedCall?
"dropout_37/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_37/StatefulPartitionedCall:output:0#^dropout_36/StatefulPartitionedCall*
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
G__inference_dropout_37_layer_call_and_return_conditional_losses_13929172$
"dropout_37/StatefulPartitionedCall?
 max_pooling2d_38/PartitionedCallPartitionedCall+dropout_37/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_13926652"
 max_pooling2d_38/PartitionedCall?
flatten_22/PartitionedCallPartitionedCall)max_pooling2d_38/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????n* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_flatten_22_layer_call_and_return_conditional_losses_13929422
flatten_22/PartitionedCall?
 dense_44/StatefulPartitionedCallStatefulPartitionedCall#flatten_22/PartitionedCall:output:0dense_44_1393089dense_44_1393091*
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
E__inference_dense_44_layer_call_and_return_conditional_losses_13929612"
 dense_44/StatefulPartitionedCall?
 dense_45/StatefulPartitionedCallStatefulPartitionedCall)dense_44/StatefulPartitionedCall:output:0dense_45_1393094dense_45_1393096*
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
E__inference_dense_45_layer_call_and_return_conditional_losses_13929882"
 dense_45/StatefulPartitionedCall?
IdentityIdentity)dense_45/StatefulPartitionedCall:output:0/^batch_normalization_36/StatefulPartitionedCall/^batch_normalization_37/StatefulPartitionedCall"^conv2d_95/StatefulPartitionedCall"^conv2d_96/StatefulPartitionedCall!^dense_44/StatefulPartitionedCall!^dense_45/StatefulPartitionedCall#^dropout_36/StatefulPartitionedCall#^dropout_37/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:???????????::::::::::::::::2`
.batch_normalization_36/StatefulPartitionedCall.batch_normalization_36/StatefulPartitionedCall2`
.batch_normalization_37/StatefulPartitionedCall.batch_normalization_37/StatefulPartitionedCall2F
!conv2d_95/StatefulPartitionedCall!conv2d_95/StatefulPartitionedCall2F
!conv2d_96/StatefulPartitionedCall!conv2d_96/StatefulPartitionedCall2D
 dense_44/StatefulPartitionedCall dense_44/StatefulPartitionedCall2D
 dense_45/StatefulPartitionedCall dense_45/StatefulPartitionedCall2H
"dropout_36/StatefulPartitionedCall"dropout_36/StatefulPartitionedCall2H
"dropout_37/StatefulPartitionedCall"dropout_37/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_36_layer_call_fn_1393622

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
S__inference_batch_normalization_36_layer_call_and_return_conditional_losses_13927212
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
*__inference_dense_44_layer_call_fn_1393868

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
E__inference_dense_44_layer_call_and_return_conditional_losses_13929612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????n::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????n
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_36_layer_call_and_return_conditional_losses_1392513

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
G__inference_flatten_22_layer_call_and_return_conditional_losses_1392942

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 7  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????n2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????n2

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
S__inference_batch_normalization_37_layer_call_and_return_conditional_losses_1393766

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
8__inference_batch_normalization_36_layer_call_fn_1393571

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
S__inference_batch_normalization_36_layer_call_and_return_conditional_losses_13925442
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
?2
?
O__inference_CNN_aug_bnorm_drop_layer_call_and_return_conditional_losses_1393183

inputs
conv2d_95_1393140
conv2d_95_1393142"
batch_normalization_36_1393145"
batch_normalization_36_1393147"
batch_normalization_36_1393149"
batch_normalization_36_1393151
conv2d_96_1393155
conv2d_96_1393157"
batch_normalization_37_1393160"
batch_normalization_37_1393162"
batch_normalization_37_1393164"
batch_normalization_37_1393166
dense_44_1393172
dense_44_1393174
dense_45_1393177
dense_45_1393179
identity??.batch_normalization_36/StatefulPartitionedCall?.batch_normalization_37/StatefulPartitionedCall?!conv2d_95/StatefulPartitionedCall?!conv2d_96/StatefulPartitionedCall? dense_44/StatefulPartitionedCall? dense_45/StatefulPartitionedCall?
!conv2d_95/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_95_1393140conv2d_95_1393142*
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
GPU 2J 8? *O
fJRH
F__inference_conv2d_95_layer_call_and_return_conditional_losses_13926862#
!conv2d_95/StatefulPartitionedCall?
.batch_normalization_36/StatefulPartitionedCallStatefulPartitionedCall*conv2d_95/StatefulPartitionedCall:output:0batch_normalization_36_1393145batch_normalization_36_1393147batch_normalization_36_1393149batch_normalization_36_1393151*
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
S__inference_batch_normalization_36_layer_call_and_return_conditional_losses_139273920
.batch_normalization_36/StatefulPartitionedCall?
dropout_36/PartitionedCallPartitionedCall7batch_normalization_36/StatefulPartitionedCall:output:0*
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
G__inference_dropout_36_layer_call_and_return_conditional_losses_13927922
dropout_36/PartitionedCall?
!conv2d_96/StatefulPartitionedCallStatefulPartitionedCall#dropout_36/PartitionedCall:output:0conv2d_96_1393155conv2d_96_1393157*
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
GPU 2J 8? *O
fJRH
F__inference_conv2d_96_layer_call_and_return_conditional_losses_13928162#
!conv2d_96/StatefulPartitionedCall?
.batch_normalization_37/StatefulPartitionedCallStatefulPartitionedCall*conv2d_96/StatefulPartitionedCall:output:0batch_normalization_37_1393160batch_normalization_37_1393162batch_normalization_37_1393164batch_normalization_37_1393166*
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
S__inference_batch_normalization_37_layer_call_and_return_conditional_losses_139286920
.batch_normalization_37/StatefulPartitionedCall?
dropout_37/PartitionedCallPartitionedCall7batch_normalization_37/StatefulPartitionedCall:output:0*
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
G__inference_dropout_37_layer_call_and_return_conditional_losses_13929222
dropout_37/PartitionedCall?
 max_pooling2d_38/PartitionedCallPartitionedCall#dropout_37/PartitionedCall:output:0*
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
M__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_13926652"
 max_pooling2d_38/PartitionedCall?
flatten_22/PartitionedCallPartitionedCall)max_pooling2d_38/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????n* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_flatten_22_layer_call_and_return_conditional_losses_13929422
flatten_22/PartitionedCall?
 dense_44/StatefulPartitionedCallStatefulPartitionedCall#flatten_22/PartitionedCall:output:0dense_44_1393172dense_44_1393174*
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
E__inference_dense_44_layer_call_and_return_conditional_losses_13929612"
 dense_44/StatefulPartitionedCall?
 dense_45/StatefulPartitionedCallStatefulPartitionedCall)dense_44/StatefulPartitionedCall:output:0dense_45_1393177dense_45_1393179*
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
E__inference_dense_45_layer_call_and_return_conditional_losses_13929882"
 dense_45/StatefulPartitionedCall?
IdentityIdentity)dense_45/StatefulPartitionedCall:output:0/^batch_normalization_36/StatefulPartitionedCall/^batch_normalization_37/StatefulPartitionedCall"^conv2d_95/StatefulPartitionedCall"^conv2d_96/StatefulPartitionedCall!^dense_44/StatefulPartitionedCall!^dense_45/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:???????????::::::::::::::::2`
.batch_normalization_36/StatefulPartitionedCall.batch_normalization_36/StatefulPartitionedCall2`
.batch_normalization_37/StatefulPartitionedCall.batch_normalization_37/StatefulPartitionedCall2F
!conv2d_95/StatefulPartitionedCall!conv2d_95/StatefulPartitionedCall2F
!conv2d_96/StatefulPartitionedCall!conv2d_96/StatefulPartitionedCall2D
 dense_44/StatefulPartitionedCall dense_44/StatefulPartitionedCall2D
 dense_45/StatefulPartitionedCall dense_45/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
+__inference_conv2d_95_layer_call_fn_1393507

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
GPU 2J 8? *O
fJRH
F__inference_conv2d_95_layer_call_and_return_conditional_losses_13926862
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
?
H
,__inference_flatten_22_layer_call_fn_1393848

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
:??????????n* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_flatten_22_layer_call_and_return_conditional_losses_13929422
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????n2

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
S__inference_batch_normalization_37_layer_call_and_return_conditional_losses_1393702

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
S__inference_batch_normalization_36_layer_call_and_return_conditional_losses_1393609

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
8__inference_batch_normalization_37_layer_call_fn_1393797

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
S__inference_batch_normalization_37_layer_call_and_return_conditional_losses_13928512
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
?T
?
O__inference_CNN_aug_bnorm_drop_layer_call_and_return_conditional_losses_1393413

inputs,
(conv2d_95_conv2d_readvariableop_resource-
)conv2d_95_biasadd_readvariableop_resource2
.batch_normalization_36_readvariableop_resource4
0batch_normalization_36_readvariableop_1_resourceC
?batch_normalization_36_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_36_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_96_conv2d_readvariableop_resource-
)conv2d_96_biasadd_readvariableop_resource2
.batch_normalization_37_readvariableop_resource4
0batch_normalization_37_readvariableop_1_resourceC
?batch_normalization_37_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_37_fusedbatchnormv3_readvariableop_1_resource+
'dense_44_matmul_readvariableop_resource,
(dense_44_biasadd_readvariableop_resource+
'dense_45_matmul_readvariableop_resource,
(dense_45_biasadd_readvariableop_resource
identity??6batch_normalization_36/FusedBatchNormV3/ReadVariableOp?8batch_normalization_36/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_36/ReadVariableOp?'batch_normalization_36/ReadVariableOp_1?6batch_normalization_37/FusedBatchNormV3/ReadVariableOp?8batch_normalization_37/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_37/ReadVariableOp?'batch_normalization_37/ReadVariableOp_1? conv2d_95/BiasAdd/ReadVariableOp?conv2d_95/Conv2D/ReadVariableOp? conv2d_96/BiasAdd/ReadVariableOp?conv2d_96/Conv2D/ReadVariableOp?dense_44/BiasAdd/ReadVariableOp?dense_44/MatMul/ReadVariableOp?dense_45/BiasAdd/ReadVariableOp?dense_45/MatMul/ReadVariableOp?
conv2d_95/Conv2D/ReadVariableOpReadVariableOp(conv2d_95_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_95/Conv2D/ReadVariableOp?
conv2d_95/Conv2DConv2Dinputs'conv2d_95/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_95/Conv2D?
 conv2d_95/BiasAdd/ReadVariableOpReadVariableOp)conv2d_95_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_95/BiasAdd/ReadVariableOp?
conv2d_95/BiasAddBiasAddconv2d_95/Conv2D:output:0(conv2d_95/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_95/BiasAdd?
conv2d_95/ReluReluconv2d_95/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_95/Relu?
%batch_normalization_36/ReadVariableOpReadVariableOp.batch_normalization_36_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_36/ReadVariableOp?
'batch_normalization_36/ReadVariableOp_1ReadVariableOp0batch_normalization_36_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_36/ReadVariableOp_1?
6batch_normalization_36/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_36_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_36/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_36/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_36_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_36/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_36/FusedBatchNormV3FusedBatchNormV3conv2d_95/Relu:activations:0-batch_normalization_36/ReadVariableOp:value:0/batch_normalization_36/ReadVariableOp_1:value:0>batch_normalization_36/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_36/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2)
'batch_normalization_36/FusedBatchNormV3?
dropout_36/IdentityIdentity+batch_normalization_36/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
dropout_36/Identity?
conv2d_96/Conv2D/ReadVariableOpReadVariableOp(conv2d_96_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_96/Conv2D/ReadVariableOp?
conv2d_96/Conv2DConv2Ddropout_36/Identity:output:0'conv2d_96/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_96/Conv2D?
 conv2d_96/BiasAdd/ReadVariableOpReadVariableOp)conv2d_96_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_96/BiasAdd/ReadVariableOp?
conv2d_96/BiasAddBiasAddconv2d_96/Conv2D:output:0(conv2d_96/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_96/BiasAdd?
conv2d_96/ReluReluconv2d_96/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_96/Relu?
%batch_normalization_37/ReadVariableOpReadVariableOp.batch_normalization_37_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_37/ReadVariableOp?
'batch_normalization_37/ReadVariableOp_1ReadVariableOp0batch_normalization_37_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_37/ReadVariableOp_1?
6batch_normalization_37/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_37_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_37/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_37/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_37_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_37/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_37/FusedBatchNormV3FusedBatchNormV3conv2d_96/Relu:activations:0-batch_normalization_37/ReadVariableOp:value:0/batch_normalization_37/ReadVariableOp_1:value:0>batch_normalization_37/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_37/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2)
'batch_normalization_37/FusedBatchNormV3?
dropout_37/IdentityIdentity+batch_normalization_37/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
dropout_37/Identity?
max_pooling2d_38/MaxPoolMaxPooldropout_37/Identity:output:0*/
_output_shapes
:?????????***
ksize
*
paddingVALID*
strides
2
max_pooling2d_38/MaxPoolu
flatten_22/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 7  2
flatten_22/Const?
flatten_22/ReshapeReshape!max_pooling2d_38/MaxPool:output:0flatten_22/Const:output:0*
T0*(
_output_shapes
:??????????n2
flatten_22/Reshape?
dense_44/MatMul/ReadVariableOpReadVariableOp'dense_44_matmul_readvariableop_resource*
_output_shapes
:	?n *
dtype02 
dense_44/MatMul/ReadVariableOp?
dense_44/MatMulMatMulflatten_22/Reshape:output:0&dense_44/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_44/MatMul?
dense_44/BiasAdd/ReadVariableOpReadVariableOp(dense_44_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_44/BiasAdd/ReadVariableOp?
dense_44/BiasAddBiasAdddense_44/MatMul:product:0'dense_44/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_44/BiasAdds
dense_44/ReluReludense_44/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_44/Relu?
dense_45/MatMul/ReadVariableOpReadVariableOp'dense_45_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_45/MatMul/ReadVariableOp?
dense_45/MatMulMatMuldense_44/Relu:activations:0&dense_45/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_45/MatMul?
dense_45/BiasAdd/ReadVariableOpReadVariableOp(dense_45_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_45/BiasAdd/ReadVariableOp?
dense_45/BiasAddBiasAdddense_45/MatMul:product:0'dense_45/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_45/BiasAdd|
dense_45/SoftmaxSoftmaxdense_45/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_45/Softmax?
IdentityIdentitydense_45/Softmax:softmax:07^batch_normalization_36/FusedBatchNormV3/ReadVariableOp9^batch_normalization_36/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_36/ReadVariableOp(^batch_normalization_36/ReadVariableOp_17^batch_normalization_37/FusedBatchNormV3/ReadVariableOp9^batch_normalization_37/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_37/ReadVariableOp(^batch_normalization_37/ReadVariableOp_1!^conv2d_95/BiasAdd/ReadVariableOp ^conv2d_95/Conv2D/ReadVariableOp!^conv2d_96/BiasAdd/ReadVariableOp ^conv2d_96/Conv2D/ReadVariableOp ^dense_44/BiasAdd/ReadVariableOp^dense_44/MatMul/ReadVariableOp ^dense_45/BiasAdd/ReadVariableOp^dense_45/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:???????????::::::::::::::::2p
6batch_normalization_36/FusedBatchNormV3/ReadVariableOp6batch_normalization_36/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_36/FusedBatchNormV3/ReadVariableOp_18batch_normalization_36/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_36/ReadVariableOp%batch_normalization_36/ReadVariableOp2R
'batch_normalization_36/ReadVariableOp_1'batch_normalization_36/ReadVariableOp_12p
6batch_normalization_37/FusedBatchNormV3/ReadVariableOp6batch_normalization_37/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_37/FusedBatchNormV3/ReadVariableOp_18batch_normalization_37/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_37/ReadVariableOp%batch_normalization_37/ReadVariableOp2R
'batch_normalization_37/ReadVariableOp_1'batch_normalization_37/ReadVariableOp_12D
 conv2d_95/BiasAdd/ReadVariableOp conv2d_95/BiasAdd/ReadVariableOp2B
conv2d_95/Conv2D/ReadVariableOpconv2d_95/Conv2D/ReadVariableOp2D
 conv2d_96/BiasAdd/ReadVariableOp conv2d_96/BiasAdd/ReadVariableOp2B
conv2d_96/Conv2D/ReadVariableOpconv2d_96/Conv2D/ReadVariableOp2B
dense_44/BiasAdd/ReadVariableOpdense_44/BiasAdd/ReadVariableOp2@
dense_44/MatMul/ReadVariableOpdense_44/MatMul/ReadVariableOp2B
dense_45/BiasAdd/ReadVariableOpdense_45/BiasAdd/ReadVariableOp2@
dense_45/MatMul/ReadVariableOpdense_45/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
c
G__inference_flatten_22_layer_call_and_return_conditional_losses_1393843

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 7  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????n2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????n2

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
F__inference_conv2d_96_layer_call_and_return_conditional_losses_1393673

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
?

?
F__inference_conv2d_95_layer_call_and_return_conditional_losses_1392686

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
?
?
S__inference_batch_normalization_36_layer_call_and_return_conditional_losses_1393591

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
S__inference_batch_normalization_37_layer_call_and_return_conditional_losses_1392648

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
S__inference_batch_normalization_37_layer_call_and_return_conditional_losses_1392869

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
?
e
G__inference_dropout_36_layer_call_and_return_conditional_losses_1393652

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
?
?
8__inference_batch_normalization_37_layer_call_fn_1393746

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
S__inference_batch_normalization_37_layer_call_and_return_conditional_losses_13926482
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
M__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_1392665

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
?2
?
O__inference_CNN_aug_bnorm_drop_layer_call_and_return_conditional_losses_1393051
input_23
conv2d_95_1393008
conv2d_95_1393010"
batch_normalization_36_1393013"
batch_normalization_36_1393015"
batch_normalization_36_1393017"
batch_normalization_36_1393019
conv2d_96_1393023
conv2d_96_1393025"
batch_normalization_37_1393028"
batch_normalization_37_1393030"
batch_normalization_37_1393032"
batch_normalization_37_1393034
dense_44_1393040
dense_44_1393042
dense_45_1393045
dense_45_1393047
identity??.batch_normalization_36/StatefulPartitionedCall?.batch_normalization_37/StatefulPartitionedCall?!conv2d_95/StatefulPartitionedCall?!conv2d_96/StatefulPartitionedCall? dense_44/StatefulPartitionedCall? dense_45/StatefulPartitionedCall?
!conv2d_95/StatefulPartitionedCallStatefulPartitionedCallinput_23conv2d_95_1393008conv2d_95_1393010*
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
GPU 2J 8? *O
fJRH
F__inference_conv2d_95_layer_call_and_return_conditional_losses_13926862#
!conv2d_95/StatefulPartitionedCall?
.batch_normalization_36/StatefulPartitionedCallStatefulPartitionedCall*conv2d_95/StatefulPartitionedCall:output:0batch_normalization_36_1393013batch_normalization_36_1393015batch_normalization_36_1393017batch_normalization_36_1393019*
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
S__inference_batch_normalization_36_layer_call_and_return_conditional_losses_139273920
.batch_normalization_36/StatefulPartitionedCall?
dropout_36/PartitionedCallPartitionedCall7batch_normalization_36/StatefulPartitionedCall:output:0*
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
G__inference_dropout_36_layer_call_and_return_conditional_losses_13927922
dropout_36/PartitionedCall?
!conv2d_96/StatefulPartitionedCallStatefulPartitionedCall#dropout_36/PartitionedCall:output:0conv2d_96_1393023conv2d_96_1393025*
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
GPU 2J 8? *O
fJRH
F__inference_conv2d_96_layer_call_and_return_conditional_losses_13928162#
!conv2d_96/StatefulPartitionedCall?
.batch_normalization_37/StatefulPartitionedCallStatefulPartitionedCall*conv2d_96/StatefulPartitionedCall:output:0batch_normalization_37_1393028batch_normalization_37_1393030batch_normalization_37_1393032batch_normalization_37_1393034*
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
S__inference_batch_normalization_37_layer_call_and_return_conditional_losses_139286920
.batch_normalization_37/StatefulPartitionedCall?
dropout_37/PartitionedCallPartitionedCall7batch_normalization_37/StatefulPartitionedCall:output:0*
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
G__inference_dropout_37_layer_call_and_return_conditional_losses_13929222
dropout_37/PartitionedCall?
 max_pooling2d_38/PartitionedCallPartitionedCall#dropout_37/PartitionedCall:output:0*
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
M__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_13926652"
 max_pooling2d_38/PartitionedCall?
flatten_22/PartitionedCallPartitionedCall)max_pooling2d_38/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????n* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_flatten_22_layer_call_and_return_conditional_losses_13929422
flatten_22/PartitionedCall?
 dense_44/StatefulPartitionedCallStatefulPartitionedCall#flatten_22/PartitionedCall:output:0dense_44_1393040dense_44_1393042*
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
E__inference_dense_44_layer_call_and_return_conditional_losses_13929612"
 dense_44/StatefulPartitionedCall?
 dense_45/StatefulPartitionedCallStatefulPartitionedCall)dense_44/StatefulPartitionedCall:output:0dense_45_1393045dense_45_1393047*
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
E__inference_dense_45_layer_call_and_return_conditional_losses_13929882"
 dense_45/StatefulPartitionedCall?
IdentityIdentity)dense_45/StatefulPartitionedCall:output:0/^batch_normalization_36/StatefulPartitionedCall/^batch_normalization_37/StatefulPartitionedCall"^conv2d_95/StatefulPartitionedCall"^conv2d_96/StatefulPartitionedCall!^dense_44/StatefulPartitionedCall!^dense_45/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:???????????::::::::::::::::2`
.batch_normalization_36/StatefulPartitionedCall.batch_normalization_36/StatefulPartitionedCall2`
.batch_normalization_37/StatefulPartitionedCall.batch_normalization_37/StatefulPartitionedCall2F
!conv2d_95/StatefulPartitionedCall!conv2d_95/StatefulPartitionedCall2F
!conv2d_96/StatefulPartitionedCall!conv2d_96/StatefulPartitionedCall2D
 dense_44/StatefulPartitionedCall dense_44/StatefulPartitionedCall2D
 dense_45/StatefulPartitionedCall dense_45/StatefulPartitionedCall:[ W
1
_output_shapes
:???????????
"
_user_specified_name
input_23
?
?
S__inference_batch_normalization_36_layer_call_and_return_conditional_losses_1392721

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
8__inference_batch_normalization_36_layer_call_fn_1393558

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
S__inference_batch_normalization_36_layer_call_and_return_conditional_losses_13925132
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
S__inference_batch_normalization_36_layer_call_and_return_conditional_losses_1393545

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
F__inference_conv2d_95_layer_call_and_return_conditional_losses_1393498

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
?r
?
"__inference__wrapped_model_1392451
input_23?
;cnn_aug_bnorm_drop_conv2d_95_conv2d_readvariableop_resource@
<cnn_aug_bnorm_drop_conv2d_95_biasadd_readvariableop_resourceE
Acnn_aug_bnorm_drop_batch_normalization_36_readvariableop_resourceG
Ccnn_aug_bnorm_drop_batch_normalization_36_readvariableop_1_resourceV
Rcnn_aug_bnorm_drop_batch_normalization_36_fusedbatchnormv3_readvariableop_resourceX
Tcnn_aug_bnorm_drop_batch_normalization_36_fusedbatchnormv3_readvariableop_1_resource?
;cnn_aug_bnorm_drop_conv2d_96_conv2d_readvariableop_resource@
<cnn_aug_bnorm_drop_conv2d_96_biasadd_readvariableop_resourceE
Acnn_aug_bnorm_drop_batch_normalization_37_readvariableop_resourceG
Ccnn_aug_bnorm_drop_batch_normalization_37_readvariableop_1_resourceV
Rcnn_aug_bnorm_drop_batch_normalization_37_fusedbatchnormv3_readvariableop_resourceX
Tcnn_aug_bnorm_drop_batch_normalization_37_fusedbatchnormv3_readvariableop_1_resource>
:cnn_aug_bnorm_drop_dense_44_matmul_readvariableop_resource?
;cnn_aug_bnorm_drop_dense_44_biasadd_readvariableop_resource>
:cnn_aug_bnorm_drop_dense_45_matmul_readvariableop_resource?
;cnn_aug_bnorm_drop_dense_45_biasadd_readvariableop_resource
identity??ICNN_aug_bnorm_drop/batch_normalization_36/FusedBatchNormV3/ReadVariableOp?KCNN_aug_bnorm_drop/batch_normalization_36/FusedBatchNormV3/ReadVariableOp_1?8CNN_aug_bnorm_drop/batch_normalization_36/ReadVariableOp?:CNN_aug_bnorm_drop/batch_normalization_36/ReadVariableOp_1?ICNN_aug_bnorm_drop/batch_normalization_37/FusedBatchNormV3/ReadVariableOp?KCNN_aug_bnorm_drop/batch_normalization_37/FusedBatchNormV3/ReadVariableOp_1?8CNN_aug_bnorm_drop/batch_normalization_37/ReadVariableOp?:CNN_aug_bnorm_drop/batch_normalization_37/ReadVariableOp_1?3CNN_aug_bnorm_drop/conv2d_95/BiasAdd/ReadVariableOp?2CNN_aug_bnorm_drop/conv2d_95/Conv2D/ReadVariableOp?3CNN_aug_bnorm_drop/conv2d_96/BiasAdd/ReadVariableOp?2CNN_aug_bnorm_drop/conv2d_96/Conv2D/ReadVariableOp?2CNN_aug_bnorm_drop/dense_44/BiasAdd/ReadVariableOp?1CNN_aug_bnorm_drop/dense_44/MatMul/ReadVariableOp?2CNN_aug_bnorm_drop/dense_45/BiasAdd/ReadVariableOp?1CNN_aug_bnorm_drop/dense_45/MatMul/ReadVariableOp?
2CNN_aug_bnorm_drop/conv2d_95/Conv2D/ReadVariableOpReadVariableOp;cnn_aug_bnorm_drop_conv2d_95_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype024
2CNN_aug_bnorm_drop/conv2d_95/Conv2D/ReadVariableOp?
#CNN_aug_bnorm_drop/conv2d_95/Conv2DConv2Dinput_23:CNN_aug_bnorm_drop/conv2d_95/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2%
#CNN_aug_bnorm_drop/conv2d_95/Conv2D?
3CNN_aug_bnorm_drop/conv2d_95/BiasAdd/ReadVariableOpReadVariableOp<cnn_aug_bnorm_drop_conv2d_95_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3CNN_aug_bnorm_drop/conv2d_95/BiasAdd/ReadVariableOp?
$CNN_aug_bnorm_drop/conv2d_95/BiasAddBiasAdd,CNN_aug_bnorm_drop/conv2d_95/Conv2D:output:0;CNN_aug_bnorm_drop/conv2d_95/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2&
$CNN_aug_bnorm_drop/conv2d_95/BiasAdd?
!CNN_aug_bnorm_drop/conv2d_95/ReluRelu-CNN_aug_bnorm_drop/conv2d_95/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2#
!CNN_aug_bnorm_drop/conv2d_95/Relu?
8CNN_aug_bnorm_drop/batch_normalization_36/ReadVariableOpReadVariableOpAcnn_aug_bnorm_drop_batch_normalization_36_readvariableop_resource*
_output_shapes
:*
dtype02:
8CNN_aug_bnorm_drop/batch_normalization_36/ReadVariableOp?
:CNN_aug_bnorm_drop/batch_normalization_36/ReadVariableOp_1ReadVariableOpCcnn_aug_bnorm_drop_batch_normalization_36_readvariableop_1_resource*
_output_shapes
:*
dtype02<
:CNN_aug_bnorm_drop/batch_normalization_36/ReadVariableOp_1?
ICNN_aug_bnorm_drop/batch_normalization_36/FusedBatchNormV3/ReadVariableOpReadVariableOpRcnn_aug_bnorm_drop_batch_normalization_36_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02K
ICNN_aug_bnorm_drop/batch_normalization_36/FusedBatchNormV3/ReadVariableOp?
KCNN_aug_bnorm_drop/batch_normalization_36/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpTcnn_aug_bnorm_drop_batch_normalization_36_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02M
KCNN_aug_bnorm_drop/batch_normalization_36/FusedBatchNormV3/ReadVariableOp_1?
:CNN_aug_bnorm_drop/batch_normalization_36/FusedBatchNormV3FusedBatchNormV3/CNN_aug_bnorm_drop/conv2d_95/Relu:activations:0@CNN_aug_bnorm_drop/batch_normalization_36/ReadVariableOp:value:0BCNN_aug_bnorm_drop/batch_normalization_36/ReadVariableOp_1:value:0QCNN_aug_bnorm_drop/batch_normalization_36/FusedBatchNormV3/ReadVariableOp:value:0SCNN_aug_bnorm_drop/batch_normalization_36/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2<
:CNN_aug_bnorm_drop/batch_normalization_36/FusedBatchNormV3?
&CNN_aug_bnorm_drop/dropout_36/IdentityIdentity>CNN_aug_bnorm_drop/batch_normalization_36/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2(
&CNN_aug_bnorm_drop/dropout_36/Identity?
2CNN_aug_bnorm_drop/conv2d_96/Conv2D/ReadVariableOpReadVariableOp;cnn_aug_bnorm_drop_conv2d_96_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype024
2CNN_aug_bnorm_drop/conv2d_96/Conv2D/ReadVariableOp?
#CNN_aug_bnorm_drop/conv2d_96/Conv2DConv2D/CNN_aug_bnorm_drop/dropout_36/Identity:output:0:CNN_aug_bnorm_drop/conv2d_96/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2%
#CNN_aug_bnorm_drop/conv2d_96/Conv2D?
3CNN_aug_bnorm_drop/conv2d_96/BiasAdd/ReadVariableOpReadVariableOp<cnn_aug_bnorm_drop_conv2d_96_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3CNN_aug_bnorm_drop/conv2d_96/BiasAdd/ReadVariableOp?
$CNN_aug_bnorm_drop/conv2d_96/BiasAddBiasAdd,CNN_aug_bnorm_drop/conv2d_96/Conv2D:output:0;CNN_aug_bnorm_drop/conv2d_96/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2&
$CNN_aug_bnorm_drop/conv2d_96/BiasAdd?
!CNN_aug_bnorm_drop/conv2d_96/ReluRelu-CNN_aug_bnorm_drop/conv2d_96/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2#
!CNN_aug_bnorm_drop/conv2d_96/Relu?
8CNN_aug_bnorm_drop/batch_normalization_37/ReadVariableOpReadVariableOpAcnn_aug_bnorm_drop_batch_normalization_37_readvariableop_resource*
_output_shapes
:*
dtype02:
8CNN_aug_bnorm_drop/batch_normalization_37/ReadVariableOp?
:CNN_aug_bnorm_drop/batch_normalization_37/ReadVariableOp_1ReadVariableOpCcnn_aug_bnorm_drop_batch_normalization_37_readvariableop_1_resource*
_output_shapes
:*
dtype02<
:CNN_aug_bnorm_drop/batch_normalization_37/ReadVariableOp_1?
ICNN_aug_bnorm_drop/batch_normalization_37/FusedBatchNormV3/ReadVariableOpReadVariableOpRcnn_aug_bnorm_drop_batch_normalization_37_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02K
ICNN_aug_bnorm_drop/batch_normalization_37/FusedBatchNormV3/ReadVariableOp?
KCNN_aug_bnorm_drop/batch_normalization_37/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpTcnn_aug_bnorm_drop_batch_normalization_37_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02M
KCNN_aug_bnorm_drop/batch_normalization_37/FusedBatchNormV3/ReadVariableOp_1?
:CNN_aug_bnorm_drop/batch_normalization_37/FusedBatchNormV3FusedBatchNormV3/CNN_aug_bnorm_drop/conv2d_96/Relu:activations:0@CNN_aug_bnorm_drop/batch_normalization_37/ReadVariableOp:value:0BCNN_aug_bnorm_drop/batch_normalization_37/ReadVariableOp_1:value:0QCNN_aug_bnorm_drop/batch_normalization_37/FusedBatchNormV3/ReadVariableOp:value:0SCNN_aug_bnorm_drop/batch_normalization_37/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2<
:CNN_aug_bnorm_drop/batch_normalization_37/FusedBatchNormV3?
&CNN_aug_bnorm_drop/dropout_37/IdentityIdentity>CNN_aug_bnorm_drop/batch_normalization_37/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2(
&CNN_aug_bnorm_drop/dropout_37/Identity?
+CNN_aug_bnorm_drop/max_pooling2d_38/MaxPoolMaxPool/CNN_aug_bnorm_drop/dropout_37/Identity:output:0*/
_output_shapes
:?????????***
ksize
*
paddingVALID*
strides
2-
+CNN_aug_bnorm_drop/max_pooling2d_38/MaxPool?
#CNN_aug_bnorm_drop/flatten_22/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 7  2%
#CNN_aug_bnorm_drop/flatten_22/Const?
%CNN_aug_bnorm_drop/flatten_22/ReshapeReshape4CNN_aug_bnorm_drop/max_pooling2d_38/MaxPool:output:0,CNN_aug_bnorm_drop/flatten_22/Const:output:0*
T0*(
_output_shapes
:??????????n2'
%CNN_aug_bnorm_drop/flatten_22/Reshape?
1CNN_aug_bnorm_drop/dense_44/MatMul/ReadVariableOpReadVariableOp:cnn_aug_bnorm_drop_dense_44_matmul_readvariableop_resource*
_output_shapes
:	?n *
dtype023
1CNN_aug_bnorm_drop/dense_44/MatMul/ReadVariableOp?
"CNN_aug_bnorm_drop/dense_44/MatMulMatMul.CNN_aug_bnorm_drop/flatten_22/Reshape:output:09CNN_aug_bnorm_drop/dense_44/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2$
"CNN_aug_bnorm_drop/dense_44/MatMul?
2CNN_aug_bnorm_drop/dense_44/BiasAdd/ReadVariableOpReadVariableOp;cnn_aug_bnorm_drop_dense_44_biasadd_readvariableop_resource*
_output_shapes
: *
dtype024
2CNN_aug_bnorm_drop/dense_44/BiasAdd/ReadVariableOp?
#CNN_aug_bnorm_drop/dense_44/BiasAddBiasAdd,CNN_aug_bnorm_drop/dense_44/MatMul:product:0:CNN_aug_bnorm_drop/dense_44/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2%
#CNN_aug_bnorm_drop/dense_44/BiasAdd?
 CNN_aug_bnorm_drop/dense_44/ReluRelu,CNN_aug_bnorm_drop/dense_44/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2"
 CNN_aug_bnorm_drop/dense_44/Relu?
1CNN_aug_bnorm_drop/dense_45/MatMul/ReadVariableOpReadVariableOp:cnn_aug_bnorm_drop_dense_45_matmul_readvariableop_resource*
_output_shapes

: *
dtype023
1CNN_aug_bnorm_drop/dense_45/MatMul/ReadVariableOp?
"CNN_aug_bnorm_drop/dense_45/MatMulMatMul.CNN_aug_bnorm_drop/dense_44/Relu:activations:09CNN_aug_bnorm_drop/dense_45/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2$
"CNN_aug_bnorm_drop/dense_45/MatMul?
2CNN_aug_bnorm_drop/dense_45/BiasAdd/ReadVariableOpReadVariableOp;cnn_aug_bnorm_drop_dense_45_biasadd_readvariableop_resource*
_output_shapes
:*
dtype024
2CNN_aug_bnorm_drop/dense_45/BiasAdd/ReadVariableOp?
#CNN_aug_bnorm_drop/dense_45/BiasAddBiasAdd,CNN_aug_bnorm_drop/dense_45/MatMul:product:0:CNN_aug_bnorm_drop/dense_45/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2%
#CNN_aug_bnorm_drop/dense_45/BiasAdd?
#CNN_aug_bnorm_drop/dense_45/SoftmaxSoftmax,CNN_aug_bnorm_drop/dense_45/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2%
#CNN_aug_bnorm_drop/dense_45/Softmax?
IdentityIdentity-CNN_aug_bnorm_drop/dense_45/Softmax:softmax:0J^CNN_aug_bnorm_drop/batch_normalization_36/FusedBatchNormV3/ReadVariableOpL^CNN_aug_bnorm_drop/batch_normalization_36/FusedBatchNormV3/ReadVariableOp_19^CNN_aug_bnorm_drop/batch_normalization_36/ReadVariableOp;^CNN_aug_bnorm_drop/batch_normalization_36/ReadVariableOp_1J^CNN_aug_bnorm_drop/batch_normalization_37/FusedBatchNormV3/ReadVariableOpL^CNN_aug_bnorm_drop/batch_normalization_37/FusedBatchNormV3/ReadVariableOp_19^CNN_aug_bnorm_drop/batch_normalization_37/ReadVariableOp;^CNN_aug_bnorm_drop/batch_normalization_37/ReadVariableOp_14^CNN_aug_bnorm_drop/conv2d_95/BiasAdd/ReadVariableOp3^CNN_aug_bnorm_drop/conv2d_95/Conv2D/ReadVariableOp4^CNN_aug_bnorm_drop/conv2d_96/BiasAdd/ReadVariableOp3^CNN_aug_bnorm_drop/conv2d_96/Conv2D/ReadVariableOp3^CNN_aug_bnorm_drop/dense_44/BiasAdd/ReadVariableOp2^CNN_aug_bnorm_drop/dense_44/MatMul/ReadVariableOp3^CNN_aug_bnorm_drop/dense_45/BiasAdd/ReadVariableOp2^CNN_aug_bnorm_drop/dense_45/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:???????????::::::::::::::::2?
ICNN_aug_bnorm_drop/batch_normalization_36/FusedBatchNormV3/ReadVariableOpICNN_aug_bnorm_drop/batch_normalization_36/FusedBatchNormV3/ReadVariableOp2?
KCNN_aug_bnorm_drop/batch_normalization_36/FusedBatchNormV3/ReadVariableOp_1KCNN_aug_bnorm_drop/batch_normalization_36/FusedBatchNormV3/ReadVariableOp_12t
8CNN_aug_bnorm_drop/batch_normalization_36/ReadVariableOp8CNN_aug_bnorm_drop/batch_normalization_36/ReadVariableOp2x
:CNN_aug_bnorm_drop/batch_normalization_36/ReadVariableOp_1:CNN_aug_bnorm_drop/batch_normalization_36/ReadVariableOp_12?
ICNN_aug_bnorm_drop/batch_normalization_37/FusedBatchNormV3/ReadVariableOpICNN_aug_bnorm_drop/batch_normalization_37/FusedBatchNormV3/ReadVariableOp2?
KCNN_aug_bnorm_drop/batch_normalization_37/FusedBatchNormV3/ReadVariableOp_1KCNN_aug_bnorm_drop/batch_normalization_37/FusedBatchNormV3/ReadVariableOp_12t
8CNN_aug_bnorm_drop/batch_normalization_37/ReadVariableOp8CNN_aug_bnorm_drop/batch_normalization_37/ReadVariableOp2x
:CNN_aug_bnorm_drop/batch_normalization_37/ReadVariableOp_1:CNN_aug_bnorm_drop/batch_normalization_37/ReadVariableOp_12j
3CNN_aug_bnorm_drop/conv2d_95/BiasAdd/ReadVariableOp3CNN_aug_bnorm_drop/conv2d_95/BiasAdd/ReadVariableOp2h
2CNN_aug_bnorm_drop/conv2d_95/Conv2D/ReadVariableOp2CNN_aug_bnorm_drop/conv2d_95/Conv2D/ReadVariableOp2j
3CNN_aug_bnorm_drop/conv2d_96/BiasAdd/ReadVariableOp3CNN_aug_bnorm_drop/conv2d_96/BiasAdd/ReadVariableOp2h
2CNN_aug_bnorm_drop/conv2d_96/Conv2D/ReadVariableOp2CNN_aug_bnorm_drop/conv2d_96/Conv2D/ReadVariableOp2h
2CNN_aug_bnorm_drop/dense_44/BiasAdd/ReadVariableOp2CNN_aug_bnorm_drop/dense_44/BiasAdd/ReadVariableOp2f
1CNN_aug_bnorm_drop/dense_44/MatMul/ReadVariableOp1CNN_aug_bnorm_drop/dense_44/MatMul/ReadVariableOp2h
2CNN_aug_bnorm_drop/dense_45/BiasAdd/ReadVariableOp2CNN_aug_bnorm_drop/dense_45/BiasAdd/ReadVariableOp2f
1CNN_aug_bnorm_drop/dense_45/MatMul/ReadVariableOp1CNN_aug_bnorm_drop/dense_45/MatMul/ReadVariableOp:[ W
1
_output_shapes
:???????????
"
_user_specified_name
input_23
?
?
S__inference_batch_normalization_37_layer_call_and_return_conditional_losses_1392851

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
?
e
G__inference_dropout_37_layer_call_and_return_conditional_losses_1393827

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
??
?
#__inference__traced_restore_1394215
file_prefix%
!assignvariableop_conv2d_95_kernel%
!assignvariableop_1_conv2d_95_bias3
/assignvariableop_2_batch_normalization_36_gamma2
.assignvariableop_3_batch_normalization_36_beta9
5assignvariableop_4_batch_normalization_36_moving_mean=
9assignvariableop_5_batch_normalization_36_moving_variance'
#assignvariableop_6_conv2d_96_kernel%
!assignvariableop_7_conv2d_96_bias3
/assignvariableop_8_batch_normalization_37_gamma2
.assignvariableop_9_batch_normalization_37_beta:
6assignvariableop_10_batch_normalization_37_moving_mean>
:assignvariableop_11_batch_normalization_37_moving_variance'
#assignvariableop_12_dense_44_kernel%
!assignvariableop_13_dense_44_bias'
#assignvariableop_14_dense_45_kernel%
!assignvariableop_15_dense_45_bias!
assignvariableop_16_adam_iter#
assignvariableop_17_adam_beta_1#
assignvariableop_18_adam_beta_2"
assignvariableop_19_adam_decay*
&assignvariableop_20_adam_learning_rate
assignvariableop_21_total
assignvariableop_22_count
assignvariableop_23_total_1
assignvariableop_24_count_1/
+assignvariableop_25_adam_conv2d_95_kernel_m-
)assignvariableop_26_adam_conv2d_95_bias_m;
7assignvariableop_27_adam_batch_normalization_36_gamma_m:
6assignvariableop_28_adam_batch_normalization_36_beta_m/
+assignvariableop_29_adam_conv2d_96_kernel_m-
)assignvariableop_30_adam_conv2d_96_bias_m;
7assignvariableop_31_adam_batch_normalization_37_gamma_m:
6assignvariableop_32_adam_batch_normalization_37_beta_m.
*assignvariableop_33_adam_dense_44_kernel_m,
(assignvariableop_34_adam_dense_44_bias_m.
*assignvariableop_35_adam_dense_45_kernel_m,
(assignvariableop_36_adam_dense_45_bias_m/
+assignvariableop_37_adam_conv2d_95_kernel_v-
)assignvariableop_38_adam_conv2d_95_bias_v;
7assignvariableop_39_adam_batch_normalization_36_gamma_v:
6assignvariableop_40_adam_batch_normalization_36_beta_v/
+assignvariableop_41_adam_conv2d_96_kernel_v-
)assignvariableop_42_adam_conv2d_96_bias_v;
7assignvariableop_43_adam_batch_normalization_37_gamma_v:
6assignvariableop_44_adam_batch_normalization_37_beta_v.
*assignvariableop_45_adam_dense_44_kernel_v,
(assignvariableop_46_adam_dense_44_bias_v.
*assignvariableop_47_adam_dense_45_kernel_v,
(assignvariableop_48_adam_dense_45_bias_v
identity_50??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*?
value?B?2B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::*@
dtypes6
422	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_95_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_95_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp/assignvariableop_2_batch_normalization_36_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp.assignvariableop_3_batch_normalization_36_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp5assignvariableop_4_batch_normalization_36_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp9assignvariableop_5_batch_normalization_36_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_96_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_96_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp/assignvariableop_8_batch_normalization_37_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_37_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp6assignvariableop_10_batch_normalization_37_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_normalization_37_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_44_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_44_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp#assignvariableop_14_dense_45_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp!assignvariableop_15_dense_45_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_iterIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_beta_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_beta_2Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_adam_decayIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp&assignvariableop_20_adam_learning_rateIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpassignvariableop_21_totalIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpassignvariableop_22_countIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOpassignvariableop_23_total_1Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOpassignvariableop_24_count_1Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp+assignvariableop_25_adam_conv2d_95_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_conv2d_95_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp7assignvariableop_27_adam_batch_normalization_36_gamma_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp6assignvariableop_28_adam_batch_normalization_36_beta_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_conv2d_96_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_conv2d_96_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp7assignvariableop_31_adam_batch_normalization_37_gamma_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp6assignvariableop_32_adam_batch_normalization_37_beta_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_dense_44_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_dense_44_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_dense_45_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_dense_45_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp+assignvariableop_37_adam_conv2d_95_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_conv2d_95_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp7assignvariableop_39_adam_batch_normalization_36_gamma_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp6assignvariableop_40_adam_batch_normalization_36_beta_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp+assignvariableop_41_adam_conv2d_96_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp)assignvariableop_42_adam_conv2d_96_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp7assignvariableop_43_adam_batch_normalization_37_gamma_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp6assignvariableop_44_adam_batch_normalization_37_beta_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp*assignvariableop_45_adam_dense_44_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp(assignvariableop_46_adam_dense_44_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp*assignvariableop_47_adam_dense_45_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp(assignvariableop_48_adam_dense_45_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_489
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?	
Identity_49Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_49?	
Identity_50IdentityIdentity_49:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_50"#
identity_50Identity_50:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_48AssignVariableOp_482(
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
?
?
8__inference_batch_normalization_36_layer_call_fn_1393635

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
S__inference_batch_normalization_36_layer_call_and_return_conditional_losses_13927392
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
?
f
G__inference_dropout_36_layer_call_and_return_conditional_losses_1392787

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
?
e
,__inference_dropout_36_layer_call_fn_1393657

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
G__inference_dropout_36_layer_call_and_return_conditional_losses_13927872
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
N
2__inference_max_pooling2d_38_layer_call_fn_1392671

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
M__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_13926652
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
G__inference_dropout_36_layer_call_and_return_conditional_losses_1393647

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
E__inference_dense_44_layer_call_and_return_conditional_losses_1392961

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?n *
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
:??????????n::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????n
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_36_layer_call_and_return_conditional_losses_1393527

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

?
4__inference_CNN_aug_bnorm_drop_layer_call_fn_1393487

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

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_CNN_aug_bnorm_drop_layer_call_and_return_conditional_losses_13931832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:???????????::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?|
?
O__inference_CNN_aug_bnorm_drop_layer_call_and_return_conditional_losses_1393348

inputs,
(conv2d_95_conv2d_readvariableop_resource-
)conv2d_95_biasadd_readvariableop_resource2
.batch_normalization_36_readvariableop_resource4
0batch_normalization_36_readvariableop_1_resourceC
?batch_normalization_36_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_36_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_96_conv2d_readvariableop_resource-
)conv2d_96_biasadd_readvariableop_resource2
.batch_normalization_37_readvariableop_resource4
0batch_normalization_37_readvariableop_1_resourceC
?batch_normalization_37_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_37_fusedbatchnormv3_readvariableop_1_resource+
'dense_44_matmul_readvariableop_resource,
(dense_44_biasadd_readvariableop_resource+
'dense_45_matmul_readvariableop_resource,
(dense_45_biasadd_readvariableop_resource
identity??%batch_normalization_36/AssignNewValue?'batch_normalization_36/AssignNewValue_1?6batch_normalization_36/FusedBatchNormV3/ReadVariableOp?8batch_normalization_36/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_36/ReadVariableOp?'batch_normalization_36/ReadVariableOp_1?%batch_normalization_37/AssignNewValue?'batch_normalization_37/AssignNewValue_1?6batch_normalization_37/FusedBatchNormV3/ReadVariableOp?8batch_normalization_37/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_37/ReadVariableOp?'batch_normalization_37/ReadVariableOp_1? conv2d_95/BiasAdd/ReadVariableOp?conv2d_95/Conv2D/ReadVariableOp? conv2d_96/BiasAdd/ReadVariableOp?conv2d_96/Conv2D/ReadVariableOp?dense_44/BiasAdd/ReadVariableOp?dense_44/MatMul/ReadVariableOp?dense_45/BiasAdd/ReadVariableOp?dense_45/MatMul/ReadVariableOp?
conv2d_95/Conv2D/ReadVariableOpReadVariableOp(conv2d_95_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_95/Conv2D/ReadVariableOp?
conv2d_95/Conv2DConv2Dinputs'conv2d_95/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_95/Conv2D?
 conv2d_95/BiasAdd/ReadVariableOpReadVariableOp)conv2d_95_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_95/BiasAdd/ReadVariableOp?
conv2d_95/BiasAddBiasAddconv2d_95/Conv2D:output:0(conv2d_95/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_95/BiasAdd?
conv2d_95/ReluReluconv2d_95/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_95/Relu?
%batch_normalization_36/ReadVariableOpReadVariableOp.batch_normalization_36_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_36/ReadVariableOp?
'batch_normalization_36/ReadVariableOp_1ReadVariableOp0batch_normalization_36_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_36/ReadVariableOp_1?
6batch_normalization_36/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_36_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_36/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_36/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_36_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_36/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_36/FusedBatchNormV3FusedBatchNormV3conv2d_95/Relu:activations:0-batch_normalization_36/ReadVariableOp:value:0/batch_normalization_36/ReadVariableOp_1:value:0>batch_normalization_36/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_36/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_36/FusedBatchNormV3?
%batch_normalization_36/AssignNewValueAssignVariableOp?batch_normalization_36_fusedbatchnormv3_readvariableop_resource4batch_normalization_36/FusedBatchNormV3:batch_mean:07^batch_normalization_36/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*R
_classH
FDloc:@batch_normalization_36/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_36/AssignNewValue?
'batch_normalization_36/AssignNewValue_1AssignVariableOpAbatch_normalization_36_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_36/FusedBatchNormV3:batch_variance:09^batch_normalization_36/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_36/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_36/AssignNewValue_1y
dropout_36/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_36/dropout/Const?
dropout_36/dropout/MulMul+batch_normalization_36/FusedBatchNormV3:y:0!dropout_36/dropout/Const:output:0*
T0*1
_output_shapes
:???????????2
dropout_36/dropout/Mul?
dropout_36/dropout/ShapeShape+batch_normalization_36/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
dropout_36/dropout/Shape?
/dropout_36/dropout/random_uniform/RandomUniformRandomUniform!dropout_36/dropout/Shape:output:0*
T0*1
_output_shapes
:???????????*
dtype021
/dropout_36/dropout/random_uniform/RandomUniform?
!dropout_36/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2#
!dropout_36/dropout/GreaterEqual/y?
dropout_36/dropout/GreaterEqualGreaterEqual8dropout_36/dropout/random_uniform/RandomUniform:output:0*dropout_36/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:???????????2!
dropout_36/dropout/GreaterEqual?
dropout_36/dropout/CastCast#dropout_36/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:???????????2
dropout_36/dropout/Cast?
dropout_36/dropout/Mul_1Muldropout_36/dropout/Mul:z:0dropout_36/dropout/Cast:y:0*
T0*1
_output_shapes
:???????????2
dropout_36/dropout/Mul_1?
conv2d_96/Conv2D/ReadVariableOpReadVariableOp(conv2d_96_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_96/Conv2D/ReadVariableOp?
conv2d_96/Conv2DConv2Ddropout_36/dropout/Mul_1:z:0'conv2d_96/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_96/Conv2D?
 conv2d_96/BiasAdd/ReadVariableOpReadVariableOp)conv2d_96_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_96/BiasAdd/ReadVariableOp?
conv2d_96/BiasAddBiasAddconv2d_96/Conv2D:output:0(conv2d_96/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_96/BiasAdd?
conv2d_96/ReluReluconv2d_96/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_96/Relu?
%batch_normalization_37/ReadVariableOpReadVariableOp.batch_normalization_37_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_37/ReadVariableOp?
'batch_normalization_37/ReadVariableOp_1ReadVariableOp0batch_normalization_37_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_37/ReadVariableOp_1?
6batch_normalization_37/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_37_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_37/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_37/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_37_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_37/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_37/FusedBatchNormV3FusedBatchNormV3conv2d_96/Relu:activations:0-batch_normalization_37/ReadVariableOp:value:0/batch_normalization_37/ReadVariableOp_1:value:0>batch_normalization_37/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_37/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_37/FusedBatchNormV3?
%batch_normalization_37/AssignNewValueAssignVariableOp?batch_normalization_37_fusedbatchnormv3_readvariableop_resource4batch_normalization_37/FusedBatchNormV3:batch_mean:07^batch_normalization_37/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*R
_classH
FDloc:@batch_normalization_37/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_37/AssignNewValue?
'batch_normalization_37/AssignNewValue_1AssignVariableOpAbatch_normalization_37_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_37/FusedBatchNormV3:batch_variance:09^batch_normalization_37/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_37/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_37/AssignNewValue_1y
dropout_37/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_37/dropout/Const?
dropout_37/dropout/MulMul+batch_normalization_37/FusedBatchNormV3:y:0!dropout_37/dropout/Const:output:0*
T0*1
_output_shapes
:???????????2
dropout_37/dropout/Mul?
dropout_37/dropout/ShapeShape+batch_normalization_37/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
dropout_37/dropout/Shape?
/dropout_37/dropout/random_uniform/RandomUniformRandomUniform!dropout_37/dropout/Shape:output:0*
T0*1
_output_shapes
:???????????*
dtype021
/dropout_37/dropout/random_uniform/RandomUniform?
!dropout_37/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2#
!dropout_37/dropout/GreaterEqual/y?
dropout_37/dropout/GreaterEqualGreaterEqual8dropout_37/dropout/random_uniform/RandomUniform:output:0*dropout_37/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:???????????2!
dropout_37/dropout/GreaterEqual?
dropout_37/dropout/CastCast#dropout_37/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:???????????2
dropout_37/dropout/Cast?
dropout_37/dropout/Mul_1Muldropout_37/dropout/Mul:z:0dropout_37/dropout/Cast:y:0*
T0*1
_output_shapes
:???????????2
dropout_37/dropout/Mul_1?
max_pooling2d_38/MaxPoolMaxPooldropout_37/dropout/Mul_1:z:0*/
_output_shapes
:?????????***
ksize
*
paddingVALID*
strides
2
max_pooling2d_38/MaxPoolu
flatten_22/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 7  2
flatten_22/Const?
flatten_22/ReshapeReshape!max_pooling2d_38/MaxPool:output:0flatten_22/Const:output:0*
T0*(
_output_shapes
:??????????n2
flatten_22/Reshape?
dense_44/MatMul/ReadVariableOpReadVariableOp'dense_44_matmul_readvariableop_resource*
_output_shapes
:	?n *
dtype02 
dense_44/MatMul/ReadVariableOp?
dense_44/MatMulMatMulflatten_22/Reshape:output:0&dense_44/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_44/MatMul?
dense_44/BiasAdd/ReadVariableOpReadVariableOp(dense_44_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_44/BiasAdd/ReadVariableOp?
dense_44/BiasAddBiasAdddense_44/MatMul:product:0'dense_44/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_44/BiasAdds
dense_44/ReluReludense_44/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_44/Relu?
dense_45/MatMul/ReadVariableOpReadVariableOp'dense_45_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_45/MatMul/ReadVariableOp?
dense_45/MatMulMatMuldense_44/Relu:activations:0&dense_45/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_45/MatMul?
dense_45/BiasAdd/ReadVariableOpReadVariableOp(dense_45_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_45/BiasAdd/ReadVariableOp?
dense_45/BiasAddBiasAdddense_45/MatMul:product:0'dense_45/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_45/BiasAdd|
dense_45/SoftmaxSoftmaxdense_45/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_45/Softmax?
IdentityIdentitydense_45/Softmax:softmax:0&^batch_normalization_36/AssignNewValue(^batch_normalization_36/AssignNewValue_17^batch_normalization_36/FusedBatchNormV3/ReadVariableOp9^batch_normalization_36/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_36/ReadVariableOp(^batch_normalization_36/ReadVariableOp_1&^batch_normalization_37/AssignNewValue(^batch_normalization_37/AssignNewValue_17^batch_normalization_37/FusedBatchNormV3/ReadVariableOp9^batch_normalization_37/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_37/ReadVariableOp(^batch_normalization_37/ReadVariableOp_1!^conv2d_95/BiasAdd/ReadVariableOp ^conv2d_95/Conv2D/ReadVariableOp!^conv2d_96/BiasAdd/ReadVariableOp ^conv2d_96/Conv2D/ReadVariableOp ^dense_44/BiasAdd/ReadVariableOp^dense_44/MatMul/ReadVariableOp ^dense_45/BiasAdd/ReadVariableOp^dense_45/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:???????????::::::::::::::::2N
%batch_normalization_36/AssignNewValue%batch_normalization_36/AssignNewValue2R
'batch_normalization_36/AssignNewValue_1'batch_normalization_36/AssignNewValue_12p
6batch_normalization_36/FusedBatchNormV3/ReadVariableOp6batch_normalization_36/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_36/FusedBatchNormV3/ReadVariableOp_18batch_normalization_36/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_36/ReadVariableOp%batch_normalization_36/ReadVariableOp2R
'batch_normalization_36/ReadVariableOp_1'batch_normalization_36/ReadVariableOp_12N
%batch_normalization_37/AssignNewValue%batch_normalization_37/AssignNewValue2R
'batch_normalization_37/AssignNewValue_1'batch_normalization_37/AssignNewValue_12p
6batch_normalization_37/FusedBatchNormV3/ReadVariableOp6batch_normalization_37/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_37/FusedBatchNormV3/ReadVariableOp_18batch_normalization_37/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_37/ReadVariableOp%batch_normalization_37/ReadVariableOp2R
'batch_normalization_37/ReadVariableOp_1'batch_normalization_37/ReadVariableOp_12D
 conv2d_95/BiasAdd/ReadVariableOp conv2d_95/BiasAdd/ReadVariableOp2B
conv2d_95/Conv2D/ReadVariableOpconv2d_95/Conv2D/ReadVariableOp2D
 conv2d_96/BiasAdd/ReadVariableOp conv2d_96/BiasAdd/ReadVariableOp2B
conv2d_96/Conv2D/ReadVariableOpconv2d_96/Conv2D/ReadVariableOp2B
dense_44/BiasAdd/ReadVariableOpdense_44/BiasAdd/ReadVariableOp2@
dense_44/MatMul/ReadVariableOpdense_44/MatMul/ReadVariableOp2B
dense_45/BiasAdd/ReadVariableOpdense_45/BiasAdd/ReadVariableOp2@
dense_45/MatMul/ReadVariableOpdense_45/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?	
?
E__inference_dense_45_layer_call_and_return_conditional_losses_1392988

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
?
e
G__inference_dropout_37_layer_call_and_return_conditional_losses_1392922

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
?
?
+__inference_conv2d_96_layer_call_fn_1393682

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
GPU 2J 8? *O
fJRH
F__inference_conv2d_96_layer_call_and_return_conditional_losses_13928162
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

?
4__inference_CNN_aug_bnorm_drop_layer_call_fn_1393218
input_23
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

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_23unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_CNN_aug_bnorm_drop_layer_call_and_return_conditional_losses_13931832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:???????????::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:???????????
"
_user_specified_name
input_23
?
f
G__inference_dropout_37_layer_call_and_return_conditional_losses_1392917

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

?
%__inference_signature_wrapper_1393265
input_23
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

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_23unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__wrapped_model_13924512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:???????????::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:???????????
"
_user_specified_name
input_23
?
?
S__inference_batch_normalization_36_layer_call_and_return_conditional_losses_1392544

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
?5
?
O__inference_CNN_aug_bnorm_drop_layer_call_and_return_conditional_losses_1393005
input_23
conv2d_95_1392697
conv2d_95_1392699"
batch_normalization_36_1392766"
batch_normalization_36_1392768"
batch_normalization_36_1392770"
batch_normalization_36_1392772
conv2d_96_1392827
conv2d_96_1392829"
batch_normalization_37_1392896"
batch_normalization_37_1392898"
batch_normalization_37_1392900"
batch_normalization_37_1392902
dense_44_1392972
dense_44_1392974
dense_45_1392999
dense_45_1393001
identity??.batch_normalization_36/StatefulPartitionedCall?.batch_normalization_37/StatefulPartitionedCall?!conv2d_95/StatefulPartitionedCall?!conv2d_96/StatefulPartitionedCall? dense_44/StatefulPartitionedCall? dense_45/StatefulPartitionedCall?"dropout_36/StatefulPartitionedCall?"dropout_37/StatefulPartitionedCall?
!conv2d_95/StatefulPartitionedCallStatefulPartitionedCallinput_23conv2d_95_1392697conv2d_95_1392699*
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
GPU 2J 8? *O
fJRH
F__inference_conv2d_95_layer_call_and_return_conditional_losses_13926862#
!conv2d_95/StatefulPartitionedCall?
.batch_normalization_36/StatefulPartitionedCallStatefulPartitionedCall*conv2d_95/StatefulPartitionedCall:output:0batch_normalization_36_1392766batch_normalization_36_1392768batch_normalization_36_1392770batch_normalization_36_1392772*
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
S__inference_batch_normalization_36_layer_call_and_return_conditional_losses_139272120
.batch_normalization_36/StatefulPartitionedCall?
"dropout_36/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_36/StatefulPartitionedCall:output:0*
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
G__inference_dropout_36_layer_call_and_return_conditional_losses_13927872$
"dropout_36/StatefulPartitionedCall?
!conv2d_96/StatefulPartitionedCallStatefulPartitionedCall+dropout_36/StatefulPartitionedCall:output:0conv2d_96_1392827conv2d_96_1392829*
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
GPU 2J 8? *O
fJRH
F__inference_conv2d_96_layer_call_and_return_conditional_losses_13928162#
!conv2d_96/StatefulPartitionedCall?
.batch_normalization_37/StatefulPartitionedCallStatefulPartitionedCall*conv2d_96/StatefulPartitionedCall:output:0batch_normalization_37_1392896batch_normalization_37_1392898batch_normalization_37_1392900batch_normalization_37_1392902*
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
S__inference_batch_normalization_37_layer_call_and_return_conditional_losses_139285120
.batch_normalization_37/StatefulPartitionedCall?
"dropout_37/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_37/StatefulPartitionedCall:output:0#^dropout_36/StatefulPartitionedCall*
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
G__inference_dropout_37_layer_call_and_return_conditional_losses_13929172$
"dropout_37/StatefulPartitionedCall?
 max_pooling2d_38/PartitionedCallPartitionedCall+dropout_37/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_13926652"
 max_pooling2d_38/PartitionedCall?
flatten_22/PartitionedCallPartitionedCall)max_pooling2d_38/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????n* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_flatten_22_layer_call_and_return_conditional_losses_13929422
flatten_22/PartitionedCall?
 dense_44/StatefulPartitionedCallStatefulPartitionedCall#flatten_22/PartitionedCall:output:0dense_44_1392972dense_44_1392974*
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
E__inference_dense_44_layer_call_and_return_conditional_losses_13929612"
 dense_44/StatefulPartitionedCall?
 dense_45/StatefulPartitionedCallStatefulPartitionedCall)dense_44/StatefulPartitionedCall:output:0dense_45_1392999dense_45_1393001*
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
E__inference_dense_45_layer_call_and_return_conditional_losses_13929882"
 dense_45/StatefulPartitionedCall?
IdentityIdentity)dense_45/StatefulPartitionedCall:output:0/^batch_normalization_36/StatefulPartitionedCall/^batch_normalization_37/StatefulPartitionedCall"^conv2d_95/StatefulPartitionedCall"^conv2d_96/StatefulPartitionedCall!^dense_44/StatefulPartitionedCall!^dense_45/StatefulPartitionedCall#^dropout_36/StatefulPartitionedCall#^dropout_37/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:???????????::::::::::::::::2`
.batch_normalization_36/StatefulPartitionedCall.batch_normalization_36/StatefulPartitionedCall2`
.batch_normalization_37/StatefulPartitionedCall.batch_normalization_37/StatefulPartitionedCall2F
!conv2d_95/StatefulPartitionedCall!conv2d_95/StatefulPartitionedCall2F
!conv2d_96/StatefulPartitionedCall!conv2d_96/StatefulPartitionedCall2D
 dense_44/StatefulPartitionedCall dense_44/StatefulPartitionedCall2D
 dense_45/StatefulPartitionedCall dense_45/StatefulPartitionedCall2H
"dropout_36/StatefulPartitionedCall"dropout_36/StatefulPartitionedCall2H
"dropout_37/StatefulPartitionedCall"dropout_37/StatefulPartitionedCall:[ W
1
_output_shapes
:???????????
"
_user_specified_name
input_23
?
e
,__inference_dropout_37_layer_call_fn_1393832

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
G__inference_dropout_37_layer_call_and_return_conditional_losses_13929172
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
E__inference_dense_45_layer_call_and_return_conditional_losses_1393879

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

?
4__inference_CNN_aug_bnorm_drop_layer_call_fn_1393450

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

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_CNN_aug_bnorm_drop_layer_call_and_return_conditional_losses_13931002
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:???????????::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
F__inference_conv2d_96_layer_call_and_return_conditional_losses_1392816

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
8__inference_batch_normalization_37_layer_call_fn_1393733

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
S__inference_batch_normalization_37_layer_call_and_return_conditional_losses_13926172
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
S__inference_batch_normalization_37_layer_call_and_return_conditional_losses_1393720

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
S__inference_batch_normalization_36_layer_call_and_return_conditional_losses_1392739

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

?
4__inference_CNN_aug_bnorm_drop_layer_call_fn_1393135
input_23
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

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_23unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_CNN_aug_bnorm_drop_layer_call_and_return_conditional_losses_13931002
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:???????????::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:???????????
"
_user_specified_name
input_23
?
?
S__inference_batch_normalization_37_layer_call_and_return_conditional_losses_1393784

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
?

*__inference_dense_45_layer_call_fn_1393888

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
E__inference_dense_45_layer_call_and_return_conditional_losses_13929882
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
input_23;
serving_default_input_23:0???????????<
dense_450
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?\
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
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
?_default_save_signature
+?&call_and_return_all_conditional_losses
?__call__"?X
_tf_keras_network?X{"class_name": "Functional", "name": "CNN_aug_bnorm_drop", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "CNN_aug_bnorm_drop", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_23"}, "name": "input_23", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_95", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_95", "inbound_nodes": [[["input_23", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_36", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_36", "inbound_nodes": [[["conv2d_95", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_36", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "name": "dropout_36", "inbound_nodes": [[["batch_normalization_36", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_96", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_96", "inbound_nodes": [[["dropout_36", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_37", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_37", "inbound_nodes": [[["conv2d_96", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_37", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "name": "dropout_37", "inbound_nodes": [[["batch_normalization_37", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_38", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "name": "max_pooling2d_38", "inbound_nodes": [[["dropout_37", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_22", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_22", "inbound_nodes": [[["max_pooling2d_38", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_44", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_44", "inbound_nodes": [[["flatten_22", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_45", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_45", "inbound_nodes": [[["dense_44", 0, 0, {}]]]}], "input_layers": [["input_23", 0, 0]], "output_layers": [["dense_45", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "CNN_aug_bnorm_drop", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_23"}, "name": "input_23", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_95", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_95", "inbound_nodes": [[["input_23", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_36", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_36", "inbound_nodes": [[["conv2d_95", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_36", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "name": "dropout_36", "inbound_nodes": [[["batch_normalization_36", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_96", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_96", "inbound_nodes": [[["dropout_36", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_37", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_37", "inbound_nodes": [[["conv2d_96", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_37", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "name": "dropout_37", "inbound_nodes": [[["batch_normalization_37", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_38", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "name": "max_pooling2d_38", "inbound_nodes": [[["dropout_37", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_22", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_22", "inbound_nodes": [[["max_pooling2d_38", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_44", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_44", "inbound_nodes": [[["flatten_22", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_45", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_45", "inbound_nodes": [[["dense_44", 0, 0, {}]]]}], "input_layers": [["input_23", 0, 0]], "output_layers": [["dense_45", 0, 0]]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_23", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_23"}}
?	

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_95", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_95", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 1]}}
?	
axis
	gamma
beta
moving_mean
moving_variance
regularization_losses
trainable_variables
	variables
 	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_36", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_36", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 8]}}
?
!regularization_losses
"trainable_variables
#	variables
$	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_36", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_36", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}
?	

%kernel
&bias
'regularization_losses
(trainable_variables
)	variables
*	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_96", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_96", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 8]}}
?	
+axis
	,gamma
-beta
.moving_mean
/moving_variance
0regularization_losses
1trainable_variables
2	variables
3	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_37", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_37", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 8]}}
?
4regularization_losses
5trainable_variables
6	variables
7	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_37", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_37", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}
?
8regularization_losses
9trainable_variables
:	variables
;	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_38", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_38", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
<regularization_losses
=trainable_variables
>	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_22", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_22", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

@kernel
Abias
Bregularization_losses
Ctrainable_variables
D	variables
E	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_44", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_44", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 14112}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14112]}}
?

Fkernel
Gbias
Hregularization_losses
Itrainable_variables
J	variables
K	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_45", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_45", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
?
Liter

Mbeta_1

Nbeta_2
	Odecay
Plearning_ratem?m?m?m?%m?&m?,m?-m?@m?Am?Fm?Gm?v?v?v?v?%v?&v?,v?-v?@v?Av?Fv?Gv?"
	optimizer
 "
trackable_list_wrapper
v
0
1
2
3
%4
&5
,6
-7
@8
A9
F10
G11"
trackable_list_wrapper
?
0
1
2
3
4
5
%6
&7
,8
-9
.10
/11
@12
A13
F14
G15"
trackable_list_wrapper
?
regularization_losses

Qlayers
Rlayer_metrics
Smetrics
trainable_variables
	variables
Tlayer_regularization_losses
Unon_trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
*:(2conv2d_95/kernel
:2conv2d_95/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
regularization_losses

Vlayers
Wlayer_metrics
Xmetrics
trainable_variables
	variables
Ylayer_regularization_losses
Znon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_36/gamma
):'2batch_normalization_36/beta
2:0 (2"batch_normalization_36/moving_mean
6:4 (2&batch_normalization_36/moving_variance
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
?
regularization_losses

[layers
\layer_metrics
]metrics
trainable_variables
	variables
^layer_regularization_losses
_non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
!regularization_losses

`layers
alayer_metrics
bmetrics
"trainable_variables
#	variables
clayer_regularization_losses
dnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(2conv2d_96/kernel
:2conv2d_96/bias
 "
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
?
'regularization_losses

elayers
flayer_metrics
gmetrics
(trainable_variables
)	variables
hlayer_regularization_losses
inon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_37/gamma
):'2batch_normalization_37/beta
2:0 (2"batch_normalization_37/moving_mean
6:4 (2&batch_normalization_37/moving_variance
 "
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
<
,0
-1
.2
/3"
trackable_list_wrapper
?
0regularization_losses

jlayers
klayer_metrics
lmetrics
1trainable_variables
2	variables
mlayer_regularization_losses
nnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
4regularization_losses

olayers
player_metrics
qmetrics
5trainable_variables
6	variables
rlayer_regularization_losses
snon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
8regularization_losses

tlayers
ulayer_metrics
vmetrics
9trainable_variables
:	variables
wlayer_regularization_losses
xnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
<regularization_losses

ylayers
zlayer_metrics
{metrics
=trainable_variables
>	variables
|layer_regularization_losses
}non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	?n 2dense_44/kernel
: 2dense_44/bias
 "
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
?
Bregularization_losses

~layers
layer_metrics
?metrics
Ctrainable_variables
D	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!: 2dense_45/kernel
:2dense_45/bias
 "
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
.
F0
G1"
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
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
n
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
10"
trackable_list_wrapper
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
.2
/3"
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
0
1"
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
.0
/1"
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
/:-2Adam/conv2d_95/kernel/m
!:2Adam/conv2d_95/bias/m
/:-2#Adam/batch_normalization_36/gamma/m
.:,2"Adam/batch_normalization_36/beta/m
/:-2Adam/conv2d_96/kernel/m
!:2Adam/conv2d_96/bias/m
/:-2#Adam/batch_normalization_37/gamma/m
.:,2"Adam/batch_normalization_37/beta/m
':%	?n 2Adam/dense_44/kernel/m
 : 2Adam/dense_44/bias/m
&:$ 2Adam/dense_45/kernel/m
 :2Adam/dense_45/bias/m
/:-2Adam/conv2d_95/kernel/v
!:2Adam/conv2d_95/bias/v
/:-2#Adam/batch_normalization_36/gamma/v
.:,2"Adam/batch_normalization_36/beta/v
/:-2Adam/conv2d_96/kernel/v
!:2Adam/conv2d_96/bias/v
/:-2#Adam/batch_normalization_37/gamma/v
.:,2"Adam/batch_normalization_37/beta/v
':%	?n 2Adam/dense_44/kernel/v
 : 2Adam/dense_44/bias/v
&:$ 2Adam/dense_45/kernel/v
 :2Adam/dense_45/bias/v
?2?
"__inference__wrapped_model_1392451?
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
input_23???????????
?2?
O__inference_CNN_aug_bnorm_drop_layer_call_and_return_conditional_losses_1393051
O__inference_CNN_aug_bnorm_drop_layer_call_and_return_conditional_losses_1393348
O__inference_CNN_aug_bnorm_drop_layer_call_and_return_conditional_losses_1393413
O__inference_CNN_aug_bnorm_drop_layer_call_and_return_conditional_losses_1393005?
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
4__inference_CNN_aug_bnorm_drop_layer_call_fn_1393218
4__inference_CNN_aug_bnorm_drop_layer_call_fn_1393450
4__inference_CNN_aug_bnorm_drop_layer_call_fn_1393135
4__inference_CNN_aug_bnorm_drop_layer_call_fn_1393487?
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
F__inference_conv2d_95_layer_call_and_return_conditional_losses_1393498?
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
+__inference_conv2d_95_layer_call_fn_1393507?
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
S__inference_batch_normalization_36_layer_call_and_return_conditional_losses_1393545
S__inference_batch_normalization_36_layer_call_and_return_conditional_losses_1393609
S__inference_batch_normalization_36_layer_call_and_return_conditional_losses_1393527
S__inference_batch_normalization_36_layer_call_and_return_conditional_losses_1393591?
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
8__inference_batch_normalization_36_layer_call_fn_1393558
8__inference_batch_normalization_36_layer_call_fn_1393571
8__inference_batch_normalization_36_layer_call_fn_1393635
8__inference_batch_normalization_36_layer_call_fn_1393622?
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
G__inference_dropout_36_layer_call_and_return_conditional_losses_1393647
G__inference_dropout_36_layer_call_and_return_conditional_losses_1393652?
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
,__inference_dropout_36_layer_call_fn_1393657
,__inference_dropout_36_layer_call_fn_1393662?
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
F__inference_conv2d_96_layer_call_and_return_conditional_losses_1393673?
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
+__inference_conv2d_96_layer_call_fn_1393682?
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
S__inference_batch_normalization_37_layer_call_and_return_conditional_losses_1393702
S__inference_batch_normalization_37_layer_call_and_return_conditional_losses_1393720
S__inference_batch_normalization_37_layer_call_and_return_conditional_losses_1393766
S__inference_batch_normalization_37_layer_call_and_return_conditional_losses_1393784?
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
8__inference_batch_normalization_37_layer_call_fn_1393746
8__inference_batch_normalization_37_layer_call_fn_1393733
8__inference_batch_normalization_37_layer_call_fn_1393797
8__inference_batch_normalization_37_layer_call_fn_1393810?
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
G__inference_dropout_37_layer_call_and_return_conditional_losses_1393822
G__inference_dropout_37_layer_call_and_return_conditional_losses_1393827?
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
,__inference_dropout_37_layer_call_fn_1393837
,__inference_dropout_37_layer_call_fn_1393832?
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
M__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_1392665?
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
2__inference_max_pooling2d_38_layer_call_fn_1392671?
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
G__inference_flatten_22_layer_call_and_return_conditional_losses_1393843?
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
,__inference_flatten_22_layer_call_fn_1393848?
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
E__inference_dense_44_layer_call_and_return_conditional_losses_1393859?
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
*__inference_dense_44_layer_call_fn_1393868?
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
E__inference_dense_45_layer_call_and_return_conditional_losses_1393879?
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
*__inference_dense_45_layer_call_fn_1393888?
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
%__inference_signature_wrapper_1393265input_23"?
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
O__inference_CNN_aug_bnorm_drop_layer_call_and_return_conditional_losses_1393005~%&,-./@AFGC?@
9?6
,?)
input_23???????????
p

 
? "%?"
?
0?????????
? ?
O__inference_CNN_aug_bnorm_drop_layer_call_and_return_conditional_losses_1393051~%&,-./@AFGC?@
9?6
,?)
input_23???????????
p 

 
? "%?"
?
0?????????
? ?
O__inference_CNN_aug_bnorm_drop_layer_call_and_return_conditional_losses_1393348|%&,-./@AFGA?>
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
O__inference_CNN_aug_bnorm_drop_layer_call_and_return_conditional_losses_1393413|%&,-./@AFGA?>
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
4__inference_CNN_aug_bnorm_drop_layer_call_fn_1393135q%&,-./@AFGC?@
9?6
,?)
input_23???????????
p

 
? "???????????
4__inference_CNN_aug_bnorm_drop_layer_call_fn_1393218q%&,-./@AFGC?@
9?6
,?)
input_23???????????
p 

 
? "???????????
4__inference_CNN_aug_bnorm_drop_layer_call_fn_1393450o%&,-./@AFGA?>
7?4
*?'
inputs???????????
p

 
? "???????????
4__inference_CNN_aug_bnorm_drop_layer_call_fn_1393487o%&,-./@AFGA?>
7?4
*?'
inputs???????????
p 

 
? "???????????
"__inference__wrapped_model_1392451?%&,-./@AFG;?8
1?.
,?)
input_23???????????
? "3?0
.
dense_45"?
dense_45??????????
S__inference_batch_normalization_36_layer_call_and_return_conditional_losses_1393527?M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_36_layer_call_and_return_conditional_losses_1393545?M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_36_layer_call_and_return_conditional_losses_1393591v=?:
3?0
*?'
inputs???????????
p
? "/?,
%?"
0???????????
? ?
S__inference_batch_normalization_36_layer_call_and_return_conditional_losses_1393609v=?:
3?0
*?'
inputs???????????
p 
? "/?,
%?"
0???????????
? ?
8__inference_batch_normalization_36_layer_call_fn_1393558?M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
8__inference_batch_normalization_36_layer_call_fn_1393571?M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
8__inference_batch_normalization_36_layer_call_fn_1393622i=?:
3?0
*?'
inputs???????????
p
? ""?????????????
8__inference_batch_normalization_36_layer_call_fn_1393635i=?:
3?0
*?'
inputs???????????
p 
? ""?????????????
S__inference_batch_normalization_37_layer_call_and_return_conditional_losses_1393702?,-./M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_37_layer_call_and_return_conditional_losses_1393720?,-./M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_37_layer_call_and_return_conditional_losses_1393766v,-./=?:
3?0
*?'
inputs???????????
p
? "/?,
%?"
0???????????
? ?
S__inference_batch_normalization_37_layer_call_and_return_conditional_losses_1393784v,-./=?:
3?0
*?'
inputs???????????
p 
? "/?,
%?"
0???????????
? ?
8__inference_batch_normalization_37_layer_call_fn_1393733?,-./M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
8__inference_batch_normalization_37_layer_call_fn_1393746?,-./M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
8__inference_batch_normalization_37_layer_call_fn_1393797i,-./=?:
3?0
*?'
inputs???????????
p
? ""?????????????
8__inference_batch_normalization_37_layer_call_fn_1393810i,-./=?:
3?0
*?'
inputs???????????
p 
? ""?????????????
F__inference_conv2d_95_layer_call_and_return_conditional_losses_1393498p9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
+__inference_conv2d_95_layer_call_fn_1393507c9?6
/?,
*?'
inputs???????????
? ""?????????????
F__inference_conv2d_96_layer_call_and_return_conditional_losses_1393673p%&9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
+__inference_conv2d_96_layer_call_fn_1393682c%&9?6
/?,
*?'
inputs???????????
? ""?????????????
E__inference_dense_44_layer_call_and_return_conditional_losses_1393859]@A0?-
&?#
!?
inputs??????????n
? "%?"
?
0????????? 
? ~
*__inference_dense_44_layer_call_fn_1393868P@A0?-
&?#
!?
inputs??????????n
? "?????????? ?
E__inference_dense_45_layer_call_and_return_conditional_losses_1393879\FG/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????
? }
*__inference_dense_45_layer_call_fn_1393888OFG/?,
%?"
 ?
inputs????????? 
? "???????????
G__inference_dropout_36_layer_call_and_return_conditional_losses_1393647p=?:
3?0
*?'
inputs???????????
p
? "/?,
%?"
0???????????
? ?
G__inference_dropout_36_layer_call_and_return_conditional_losses_1393652p=?:
3?0
*?'
inputs???????????
p 
? "/?,
%?"
0???????????
? ?
,__inference_dropout_36_layer_call_fn_1393657c=?:
3?0
*?'
inputs???????????
p
? ""?????????????
,__inference_dropout_36_layer_call_fn_1393662c=?:
3?0
*?'
inputs???????????
p 
? ""?????????????
G__inference_dropout_37_layer_call_and_return_conditional_losses_1393822p=?:
3?0
*?'
inputs???????????
p
? "/?,
%?"
0???????????
? ?
G__inference_dropout_37_layer_call_and_return_conditional_losses_1393827p=?:
3?0
*?'
inputs???????????
p 
? "/?,
%?"
0???????????
? ?
,__inference_dropout_37_layer_call_fn_1393832c=?:
3?0
*?'
inputs???????????
p
? ""?????????????
,__inference_dropout_37_layer_call_fn_1393837c=?:
3?0
*?'
inputs???????????
p 
? ""?????????????
G__inference_flatten_22_layer_call_and_return_conditional_losses_1393843a7?4
-?*
(?%
inputs?????????**
? "&?#
?
0??????????n
? ?
,__inference_flatten_22_layer_call_fn_1393848T7?4
-?*
(?%
inputs?????????**
? "???????????n?
M__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_1392665?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
2__inference_max_pooling2d_38_layer_call_fn_1392671?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
%__inference_signature_wrapper_1393265?%&,-./@AFGG?D
? 
=?:
8
input_23,?)
input_23???????????"3?0
.
dense_45"?
dense_45?????????