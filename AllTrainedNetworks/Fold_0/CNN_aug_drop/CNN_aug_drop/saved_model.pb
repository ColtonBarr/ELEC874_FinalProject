??	
??
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
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
?
conv2d_85/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_85/kernel
}
$conv2d_85/kernel/Read/ReadVariableOpReadVariableOpconv2d_85/kernel*&
_output_shapes
:*
dtype0
t
conv2d_85/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_85/bias
m
"conv2d_85/bias/Read/ReadVariableOpReadVariableOpconv2d_85/bias*
_output_shapes
:*
dtype0
?
conv2d_86/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_86/kernel
}
$conv2d_86/kernel/Read/ReadVariableOpReadVariableOpconv2d_86/kernel*&
_output_shapes
:*
dtype0
t
conv2d_86/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_86/bias
m
"conv2d_86/bias/Read/ReadVariableOpReadVariableOpconv2d_86/bias*
_output_shapes
:*
dtype0
{
dense_36/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?n * 
shared_namedense_36/kernel
t
#dense_36/kernel/Read/ReadVariableOpReadVariableOpdense_36/kernel*
_output_shapes
:	?n *
dtype0
r
dense_36/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_36/bias
k
!dense_36/bias/Read/ReadVariableOpReadVariableOpdense_36/bias*
_output_shapes
: *
dtype0
z
dense_37/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_37/kernel
s
#dense_37/kernel/Read/ReadVariableOpReadVariableOpdense_37/kernel*
_output_shapes

: *
dtype0
r
dense_37/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_37/bias
k
!dense_37/bias/Read/ReadVariableOpReadVariableOpdense_37/bias*
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
Adam/conv2d_85/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_85/kernel/m
?
+Adam/conv2d_85/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_85/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_85/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_85/bias/m
{
)Adam/conv2d_85/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_85/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_86/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_86/kernel/m
?
+Adam/conv2d_86/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_86/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_86/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_86/bias/m
{
)Adam/conv2d_86/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_86/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_36/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?n *'
shared_nameAdam/dense_36/kernel/m
?
*Adam/dense_36/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_36/kernel/m*
_output_shapes
:	?n *
dtype0
?
Adam/dense_36/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_36/bias/m
y
(Adam/dense_36/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_36/bias/m*
_output_shapes
: *
dtype0
?
Adam/dense_37/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_37/kernel/m
?
*Adam/dense_37/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_37/kernel/m*
_output_shapes

: *
dtype0
?
Adam/dense_37/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_37/bias/m
y
(Adam/dense_37/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_37/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_85/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_85/kernel/v
?
+Adam/conv2d_85/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_85/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_85/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_85/bias/v
{
)Adam/conv2d_85/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_85/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_86/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_86/kernel/v
?
+Adam/conv2d_86/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_86/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_86/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_86/bias/v
{
)Adam/conv2d_86/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_86/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_36/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?n *'
shared_nameAdam/dense_36/kernel/v
?
*Adam/dense_36/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_36/kernel/v*
_output_shapes
:	?n *
dtype0
?
Adam/dense_36/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_36/bias/v
y
(Adam/dense_36/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_36/bias/v*
_output_shapes
: *
dtype0
?
Adam/dense_37/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_37/kernel/v
?
*Adam/dense_37/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_37/kernel/v*
_output_shapes

: *
dtype0
?
Adam/dense_37/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_37/bias/v
y
(Adam/dense_37/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_37/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?7
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?7
value?6B?6 B?6
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer_with_weights-3
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
R
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
R
 regularization_losses
!trainable_variables
"	variables
#	keras_api
R
$regularization_losses
%trainable_variables
&	variables
'	keras_api
R
(regularization_losses
)trainable_variables
*	variables
+	keras_api
h

,kernel
-bias
.regularization_losses
/trainable_variables
0	variables
1	keras_api
h

2kernel
3bias
4regularization_losses
5trainable_variables
6	variables
7	keras_api
?
8iter

9beta_1

:beta_2
	;decay
<learning_ratemumvmwmx,my-mz2m{3m|v}v~vv?,v?-v?2v?3v?
 
8
0
1
2
3
,4
-5
26
37
8
0
1
2
3
,4
-5
26
37
?
regularization_losses

=layers
>layer_metrics
?metrics
trainable_variables
	variables
@layer_regularization_losses
Anon_trainable_variables
 
\Z
VARIABLE_VALUEconv2d_85/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_85/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
regularization_losses

Blayers
Clayer_metrics
Dmetrics
trainable_variables
	variables
Elayer_regularization_losses
Fnon_trainable_variables
 
 
 
?
regularization_losses

Glayers
Hlayer_metrics
Imetrics
trainable_variables
	variables
Jlayer_regularization_losses
Knon_trainable_variables
\Z
VARIABLE_VALUEconv2d_86/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_86/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
regularization_losses

Llayers
Mlayer_metrics
Nmetrics
trainable_variables
	variables
Olayer_regularization_losses
Pnon_trainable_variables
 
 
 
?
 regularization_losses

Qlayers
Rlayer_metrics
Smetrics
!trainable_variables
"	variables
Tlayer_regularization_losses
Unon_trainable_variables
 
 
 
?
$regularization_losses

Vlayers
Wlayer_metrics
Xmetrics
%trainable_variables
&	variables
Ylayer_regularization_losses
Znon_trainable_variables
 
 
 
?
(regularization_losses

[layers
\layer_metrics
]metrics
)trainable_variables
*	variables
^layer_regularization_losses
_non_trainable_variables
[Y
VARIABLE_VALUEdense_36/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_36/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

,0
-1

,0
-1
?
.regularization_losses

`layers
alayer_metrics
bmetrics
/trainable_variables
0	variables
clayer_regularization_losses
dnon_trainable_variables
[Y
VARIABLE_VALUEdense_37/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_37/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

20
31

20
31
?
4regularization_losses

elayers
flayer_metrics
gmetrics
5trainable_variables
6	variables
hlayer_regularization_losses
inon_trainable_variables
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
j0
k1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
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
	ltotal
	mcount
n	variables
o	keras_api
D
	ptotal
	qcount
r
_fn_kwargs
s	variables
t	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

l0
m1

n	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

p0
q1

s	variables
}
VARIABLE_VALUEAdam/conv2d_85/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_85/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_86/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_86/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_36/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_36/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_37/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_37/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_85/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_85/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_86/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_86/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_36/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_36/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_37/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_37/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_19Placeholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_19conv2d_85/kernelconv2d_85/biasconv2d_86/kernelconv2d_86/biasdense_36/kerneldense_36/biasdense_37/kerneldense_37/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_996243
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_85/kernel/Read/ReadVariableOp"conv2d_85/bias/Read/ReadVariableOp$conv2d_86/kernel/Read/ReadVariableOp"conv2d_86/bias/Read/ReadVariableOp#dense_36/kernel/Read/ReadVariableOp!dense_36/bias/Read/ReadVariableOp#dense_37/kernel/Read/ReadVariableOp!dense_37/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/conv2d_85/kernel/m/Read/ReadVariableOp)Adam/conv2d_85/bias/m/Read/ReadVariableOp+Adam/conv2d_86/kernel/m/Read/ReadVariableOp)Adam/conv2d_86/bias/m/Read/ReadVariableOp*Adam/dense_36/kernel/m/Read/ReadVariableOp(Adam/dense_36/bias/m/Read/ReadVariableOp*Adam/dense_37/kernel/m/Read/ReadVariableOp(Adam/dense_37/bias/m/Read/ReadVariableOp+Adam/conv2d_85/kernel/v/Read/ReadVariableOp)Adam/conv2d_85/bias/v/Read/ReadVariableOp+Adam/conv2d_86/kernel/v/Read/ReadVariableOp)Adam/conv2d_86/bias/v/Read/ReadVariableOp*Adam/dense_36/kernel/v/Read/ReadVariableOp(Adam/dense_36/bias/v/Read/ReadVariableOp*Adam/dense_37/kernel/v/Read/ReadVariableOp(Adam/dense_37/bias/v/Read/ReadVariableOpConst*.
Tin'
%2#	*
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
GPU 2J 8? *(
f#R!
__inference__traced_save_996640
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_85/kernelconv2d_85/biasconv2d_86/kernelconv2d_86/biasdense_36/kerneldense_36/biasdense_37/kerneldense_37/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv2d_85/kernel/mAdam/conv2d_85/bias/mAdam/conv2d_86/kernel/mAdam/conv2d_86/bias/mAdam/dense_36/kernel/mAdam/dense_36/bias/mAdam/dense_37/kernel/mAdam/dense_37/bias/mAdam/conv2d_85/kernel/vAdam/conv2d_85/bias/vAdam/conv2d_86/kernel/vAdam/conv2d_86/bias/vAdam/dense_36/kernel/vAdam/dense_36/bias/vAdam/dense_37/kernel/vAdam/dense_37/bias/v*-
Tin&
$2"*
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
GPU 2J 8? *+
f&R$
"__inference__traced_restore_996749??
?
d
F__inference_dropout_33_layer_call_and_return_conditional_losses_996457

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
M
1__inference_max_pooling2d_34_layer_call_fn_995897

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
GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_9958912
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
?
~
)__inference_dense_36_layer_call_fn_996498

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
GPU 2J 8? *M
fHRF
D__inference_dense_36_layer_call_and_return_conditional_losses_9960412
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
?
b
F__inference_flatten_18_layer_call_and_return_conditional_losses_996473

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
?.
?
H__inference_CNN_aug_drop_layer_call_and_return_conditional_losses_996331

inputs,
(conv2d_85_conv2d_readvariableop_resource-
)conv2d_85_biasadd_readvariableop_resource,
(conv2d_86_conv2d_readvariableop_resource-
)conv2d_86_biasadd_readvariableop_resource+
'dense_36_matmul_readvariableop_resource,
(dense_36_biasadd_readvariableop_resource+
'dense_37_matmul_readvariableop_resource,
(dense_37_biasadd_readvariableop_resource
identity?? conv2d_85/BiasAdd/ReadVariableOp?conv2d_85/Conv2D/ReadVariableOp? conv2d_86/BiasAdd/ReadVariableOp?conv2d_86/Conv2D/ReadVariableOp?dense_36/BiasAdd/ReadVariableOp?dense_36/MatMul/ReadVariableOp?dense_37/BiasAdd/ReadVariableOp?dense_37/MatMul/ReadVariableOp?
conv2d_85/Conv2D/ReadVariableOpReadVariableOp(conv2d_85_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_85/Conv2D/ReadVariableOp?
conv2d_85/Conv2DConv2Dinputs'conv2d_85/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_85/Conv2D?
 conv2d_85/BiasAdd/ReadVariableOpReadVariableOp)conv2d_85_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_85/BiasAdd/ReadVariableOp?
conv2d_85/BiasAddBiasAddconv2d_85/Conv2D:output:0(conv2d_85/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_85/BiasAdd?
conv2d_85/ReluReluconv2d_85/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_85/Relu?
dropout_32/IdentityIdentityconv2d_85/Relu:activations:0*
T0*1
_output_shapes
:???????????2
dropout_32/Identity?
conv2d_86/Conv2D/ReadVariableOpReadVariableOp(conv2d_86_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_86/Conv2D/ReadVariableOp?
conv2d_86/Conv2DConv2Ddropout_32/Identity:output:0'conv2d_86/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_86/Conv2D?
 conv2d_86/BiasAdd/ReadVariableOpReadVariableOp)conv2d_86_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_86/BiasAdd/ReadVariableOp?
conv2d_86/BiasAddBiasAddconv2d_86/Conv2D:output:0(conv2d_86/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_86/BiasAdd?
conv2d_86/ReluReluconv2d_86/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_86/Relu?
dropout_33/IdentityIdentityconv2d_86/Relu:activations:0*
T0*1
_output_shapes
:???????????2
dropout_33/Identity?
max_pooling2d_34/MaxPoolMaxPooldropout_33/Identity:output:0*/
_output_shapes
:?????????***
ksize
*
paddingVALID*
strides
2
max_pooling2d_34/MaxPoolu
flatten_18/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 7  2
flatten_18/Const?
flatten_18/ReshapeReshape!max_pooling2d_34/MaxPool:output:0flatten_18/Const:output:0*
T0*(
_output_shapes
:??????????n2
flatten_18/Reshape?
dense_36/MatMul/ReadVariableOpReadVariableOp'dense_36_matmul_readvariableop_resource*
_output_shapes
:	?n *
dtype02 
dense_36/MatMul/ReadVariableOp?
dense_36/MatMulMatMulflatten_18/Reshape:output:0&dense_36/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_36/MatMul?
dense_36/BiasAdd/ReadVariableOpReadVariableOp(dense_36_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_36/BiasAdd/ReadVariableOp?
dense_36/BiasAddBiasAdddense_36/MatMul:product:0'dense_36/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_36/BiasAdds
dense_36/ReluReludense_36/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_36/Relu?
dense_37/MatMul/ReadVariableOpReadVariableOp'dense_37_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_37/MatMul/ReadVariableOp?
dense_37/MatMulMatMuldense_36/Relu:activations:0&dense_37/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_37/MatMul?
dense_37/BiasAdd/ReadVariableOpReadVariableOp(dense_37_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_37/BiasAdd/ReadVariableOp?
dense_37/BiasAddBiasAdddense_37/MatMul:product:0'dense_37/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_37/BiasAdd|
dense_37/SoftmaxSoftmaxdense_37/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_37/Softmax?
IdentityIdentitydense_37/Softmax:softmax:0!^conv2d_85/BiasAdd/ReadVariableOp ^conv2d_85/Conv2D/ReadVariableOp!^conv2d_86/BiasAdd/ReadVariableOp ^conv2d_86/Conv2D/ReadVariableOp ^dense_36/BiasAdd/ReadVariableOp^dense_36/MatMul/ReadVariableOp ^dense_37/BiasAdd/ReadVariableOp^dense_37/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:???????????::::::::2D
 conv2d_85/BiasAdd/ReadVariableOp conv2d_85/BiasAdd/ReadVariableOp2B
conv2d_85/Conv2D/ReadVariableOpconv2d_85/Conv2D/ReadVariableOp2D
 conv2d_86/BiasAdd/ReadVariableOp conv2d_86/BiasAdd/ReadVariableOp2B
conv2d_86/Conv2D/ReadVariableOpconv2d_86/Conv2D/ReadVariableOp2B
dense_36/BiasAdd/ReadVariableOpdense_36/BiasAdd/ReadVariableOp2@
dense_36/MatMul/ReadVariableOpdense_36/MatMul/ReadVariableOp2B
dense_37/BiasAdd/ReadVariableOpdense_37/BiasAdd/ReadVariableOp2@
dense_37/MatMul/ReadVariableOpdense_37/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?	
?
D__inference_dense_36_layer_call_and_return_conditional_losses_996041

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
?
G
+__inference_dropout_32_layer_call_fn_996420

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
GPU 2J 8? *O
fJRH
F__inference_dropout_32_layer_call_and_return_conditional_losses_9959452
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
?#
?
H__inference_CNN_aug_drop_layer_call_and_return_conditional_losses_996193

inputs
conv2d_85_996168
conv2d_85_996170
conv2d_86_996174
conv2d_86_996176
dense_36_996182
dense_36_996184
dense_37_996187
dense_37_996189
identity??!conv2d_85/StatefulPartitionedCall?!conv2d_86/StatefulPartitionedCall? dense_36/StatefulPartitionedCall? dense_37/StatefulPartitionedCall?
!conv2d_85/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_85_996168conv2d_85_996170*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_85_layer_call_and_return_conditional_losses_9959122#
!conv2d_85/StatefulPartitionedCall?
dropout_32/PartitionedCallPartitionedCall*conv2d_85/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *O
fJRH
F__inference_dropout_32_layer_call_and_return_conditional_losses_9959452
dropout_32/PartitionedCall?
!conv2d_86/StatefulPartitionedCallStatefulPartitionedCall#dropout_32/PartitionedCall:output:0conv2d_86_996174conv2d_86_996176*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_86_layer_call_and_return_conditional_losses_9959692#
!conv2d_86/StatefulPartitionedCall?
dropout_33/PartitionedCallPartitionedCall*conv2d_86/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *O
fJRH
F__inference_dropout_33_layer_call_and_return_conditional_losses_9960022
dropout_33/PartitionedCall?
 max_pooling2d_34/PartitionedCallPartitionedCall#dropout_33/PartitionedCall:output:0*
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
GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_9958912"
 max_pooling2d_34/PartitionedCall?
flatten_18/PartitionedCallPartitionedCall)max_pooling2d_34/PartitionedCall:output:0*
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
GPU 2J 8? *O
fJRH
F__inference_flatten_18_layer_call_and_return_conditional_losses_9960222
flatten_18/PartitionedCall?
 dense_36/StatefulPartitionedCallStatefulPartitionedCall#flatten_18/PartitionedCall:output:0dense_36_996182dense_36_996184*
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
GPU 2J 8? *M
fHRF
D__inference_dense_36_layer_call_and_return_conditional_losses_9960412"
 dense_36/StatefulPartitionedCall?
 dense_37/StatefulPartitionedCallStatefulPartitionedCall)dense_36/StatefulPartitionedCall:output:0dense_37_996187dense_37_996189*
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
GPU 2J 8? *M
fHRF
D__inference_dense_37_layer_call_and_return_conditional_losses_9960682"
 dense_37/StatefulPartitionedCall?
IdentityIdentity)dense_37/StatefulPartitionedCall:output:0"^conv2d_85/StatefulPartitionedCall"^conv2d_86/StatefulPartitionedCall!^dense_36/StatefulPartitionedCall!^dense_37/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:???????????::::::::2F
!conv2d_85/StatefulPartitionedCall!conv2d_85/StatefulPartitionedCall2F
!conv2d_86/StatefulPartitionedCall!conv2d_86/StatefulPartitionedCall2D
 dense_36/StatefulPartitionedCall dense_36/StatefulPartitionedCall2D
 dense_37/StatefulPartitionedCall dense_37/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
h
L__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_995891

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
?B
?
H__inference_CNN_aug_drop_layer_call_and_return_conditional_losses_996294

inputs,
(conv2d_85_conv2d_readvariableop_resource-
)conv2d_85_biasadd_readvariableop_resource,
(conv2d_86_conv2d_readvariableop_resource-
)conv2d_86_biasadd_readvariableop_resource+
'dense_36_matmul_readvariableop_resource,
(dense_36_biasadd_readvariableop_resource+
'dense_37_matmul_readvariableop_resource,
(dense_37_biasadd_readvariableop_resource
identity?? conv2d_85/BiasAdd/ReadVariableOp?conv2d_85/Conv2D/ReadVariableOp? conv2d_86/BiasAdd/ReadVariableOp?conv2d_86/Conv2D/ReadVariableOp?dense_36/BiasAdd/ReadVariableOp?dense_36/MatMul/ReadVariableOp?dense_37/BiasAdd/ReadVariableOp?dense_37/MatMul/ReadVariableOp?
conv2d_85/Conv2D/ReadVariableOpReadVariableOp(conv2d_85_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_85/Conv2D/ReadVariableOp?
conv2d_85/Conv2DConv2Dinputs'conv2d_85/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_85/Conv2D?
 conv2d_85/BiasAdd/ReadVariableOpReadVariableOp)conv2d_85_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_85/BiasAdd/ReadVariableOp?
conv2d_85/BiasAddBiasAddconv2d_85/Conv2D:output:0(conv2d_85/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_85/BiasAdd?
conv2d_85/ReluReluconv2d_85/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_85/Reluy
dropout_32/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_32/dropout/Const?
dropout_32/dropout/MulMulconv2d_85/Relu:activations:0!dropout_32/dropout/Const:output:0*
T0*1
_output_shapes
:???????????2
dropout_32/dropout/Mul?
dropout_32/dropout/ShapeShapeconv2d_85/Relu:activations:0*
T0*
_output_shapes
:2
dropout_32/dropout/Shape?
/dropout_32/dropout/random_uniform/RandomUniformRandomUniform!dropout_32/dropout/Shape:output:0*
T0*1
_output_shapes
:???????????*
dtype021
/dropout_32/dropout/random_uniform/RandomUniform?
!dropout_32/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2#
!dropout_32/dropout/GreaterEqual/y?
dropout_32/dropout/GreaterEqualGreaterEqual8dropout_32/dropout/random_uniform/RandomUniform:output:0*dropout_32/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:???????????2!
dropout_32/dropout/GreaterEqual?
dropout_32/dropout/CastCast#dropout_32/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:???????????2
dropout_32/dropout/Cast?
dropout_32/dropout/Mul_1Muldropout_32/dropout/Mul:z:0dropout_32/dropout/Cast:y:0*
T0*1
_output_shapes
:???????????2
dropout_32/dropout/Mul_1?
conv2d_86/Conv2D/ReadVariableOpReadVariableOp(conv2d_86_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_86/Conv2D/ReadVariableOp?
conv2d_86/Conv2DConv2Ddropout_32/dropout/Mul_1:z:0'conv2d_86/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_86/Conv2D?
 conv2d_86/BiasAdd/ReadVariableOpReadVariableOp)conv2d_86_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_86/BiasAdd/ReadVariableOp?
conv2d_86/BiasAddBiasAddconv2d_86/Conv2D:output:0(conv2d_86/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_86/BiasAdd?
conv2d_86/ReluReluconv2d_86/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_86/Reluy
dropout_33/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_33/dropout/Const?
dropout_33/dropout/MulMulconv2d_86/Relu:activations:0!dropout_33/dropout/Const:output:0*
T0*1
_output_shapes
:???????????2
dropout_33/dropout/Mul?
dropout_33/dropout/ShapeShapeconv2d_86/Relu:activations:0*
T0*
_output_shapes
:2
dropout_33/dropout/Shape?
/dropout_33/dropout/random_uniform/RandomUniformRandomUniform!dropout_33/dropout/Shape:output:0*
T0*1
_output_shapes
:???????????*
dtype021
/dropout_33/dropout/random_uniform/RandomUniform?
!dropout_33/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2#
!dropout_33/dropout/GreaterEqual/y?
dropout_33/dropout/GreaterEqualGreaterEqual8dropout_33/dropout/random_uniform/RandomUniform:output:0*dropout_33/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:???????????2!
dropout_33/dropout/GreaterEqual?
dropout_33/dropout/CastCast#dropout_33/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:???????????2
dropout_33/dropout/Cast?
dropout_33/dropout/Mul_1Muldropout_33/dropout/Mul:z:0dropout_33/dropout/Cast:y:0*
T0*1
_output_shapes
:???????????2
dropout_33/dropout/Mul_1?
max_pooling2d_34/MaxPoolMaxPooldropout_33/dropout/Mul_1:z:0*/
_output_shapes
:?????????***
ksize
*
paddingVALID*
strides
2
max_pooling2d_34/MaxPoolu
flatten_18/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 7  2
flatten_18/Const?
flatten_18/ReshapeReshape!max_pooling2d_34/MaxPool:output:0flatten_18/Const:output:0*
T0*(
_output_shapes
:??????????n2
flatten_18/Reshape?
dense_36/MatMul/ReadVariableOpReadVariableOp'dense_36_matmul_readvariableop_resource*
_output_shapes
:	?n *
dtype02 
dense_36/MatMul/ReadVariableOp?
dense_36/MatMulMatMulflatten_18/Reshape:output:0&dense_36/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_36/MatMul?
dense_36/BiasAdd/ReadVariableOpReadVariableOp(dense_36_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_36/BiasAdd/ReadVariableOp?
dense_36/BiasAddBiasAdddense_36/MatMul:product:0'dense_36/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_36/BiasAdds
dense_36/ReluReludense_36/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_36/Relu?
dense_37/MatMul/ReadVariableOpReadVariableOp'dense_37_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_37/MatMul/ReadVariableOp?
dense_37/MatMulMatMuldense_36/Relu:activations:0&dense_37/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_37/MatMul?
dense_37/BiasAdd/ReadVariableOpReadVariableOp(dense_37_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_37/BiasAdd/ReadVariableOp?
dense_37/BiasAddBiasAdddense_37/MatMul:product:0'dense_37/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_37/BiasAdd|
dense_37/SoftmaxSoftmaxdense_37/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_37/Softmax?
IdentityIdentitydense_37/Softmax:softmax:0!^conv2d_85/BiasAdd/ReadVariableOp ^conv2d_85/Conv2D/ReadVariableOp!^conv2d_86/BiasAdd/ReadVariableOp ^conv2d_86/Conv2D/ReadVariableOp ^dense_36/BiasAdd/ReadVariableOp^dense_36/MatMul/ReadVariableOp ^dense_37/BiasAdd/ReadVariableOp^dense_37/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:???????????::::::::2D
 conv2d_85/BiasAdd/ReadVariableOp conv2d_85/BiasAdd/ReadVariableOp2B
conv2d_85/Conv2D/ReadVariableOpconv2d_85/Conv2D/ReadVariableOp2D
 conv2d_86/BiasAdd/ReadVariableOp conv2d_86/BiasAdd/ReadVariableOp2B
conv2d_86/Conv2D/ReadVariableOpconv2d_86/Conv2D/ReadVariableOp2B
dense_36/BiasAdd/ReadVariableOpdense_36/BiasAdd/ReadVariableOp2@
dense_36/MatMul/ReadVariableOpdense_36/MatMul/ReadVariableOp2B
dense_37/BiasAdd/ReadVariableOpdense_37/BiasAdd/ReadVariableOp2@
dense_37/MatMul/ReadVariableOpdense_37/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
E__inference_conv2d_85_layer_call_and_return_conditional_losses_996384

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
E__inference_conv2d_85_layer_call_and_return_conditional_losses_995912

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
-__inference_CNN_aug_drop_layer_call_fn_996352

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_CNN_aug_drop_layer_call_and_return_conditional_losses_9961442
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:???????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
e
F__inference_dropout_32_layer_call_and_return_conditional_losses_996405

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
D__inference_dense_37_layer_call_and_return_conditional_losses_996509

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
?
?
-__inference_CNN_aug_drop_layer_call_fn_996373

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_CNN_aug_drop_layer_call_and_return_conditional_losses_9961932
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:???????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?&
?
H__inference_CNN_aug_drop_layer_call_and_return_conditional_losses_996085
input_19
conv2d_85_995923
conv2d_85_995925
conv2d_86_995980
conv2d_86_995982
dense_36_996052
dense_36_996054
dense_37_996079
dense_37_996081
identity??!conv2d_85/StatefulPartitionedCall?!conv2d_86/StatefulPartitionedCall? dense_36/StatefulPartitionedCall? dense_37/StatefulPartitionedCall?"dropout_32/StatefulPartitionedCall?"dropout_33/StatefulPartitionedCall?
!conv2d_85/StatefulPartitionedCallStatefulPartitionedCallinput_19conv2d_85_995923conv2d_85_995925*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_85_layer_call_and_return_conditional_losses_9959122#
!conv2d_85/StatefulPartitionedCall?
"dropout_32/StatefulPartitionedCallStatefulPartitionedCall*conv2d_85/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *O
fJRH
F__inference_dropout_32_layer_call_and_return_conditional_losses_9959402$
"dropout_32/StatefulPartitionedCall?
!conv2d_86/StatefulPartitionedCallStatefulPartitionedCall+dropout_32/StatefulPartitionedCall:output:0conv2d_86_995980conv2d_86_995982*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_86_layer_call_and_return_conditional_losses_9959692#
!conv2d_86/StatefulPartitionedCall?
"dropout_33/StatefulPartitionedCallStatefulPartitionedCall*conv2d_86/StatefulPartitionedCall:output:0#^dropout_32/StatefulPartitionedCall*
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
GPU 2J 8? *O
fJRH
F__inference_dropout_33_layer_call_and_return_conditional_losses_9959972$
"dropout_33/StatefulPartitionedCall?
 max_pooling2d_34/PartitionedCallPartitionedCall+dropout_33/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_9958912"
 max_pooling2d_34/PartitionedCall?
flatten_18/PartitionedCallPartitionedCall)max_pooling2d_34/PartitionedCall:output:0*
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
GPU 2J 8? *O
fJRH
F__inference_flatten_18_layer_call_and_return_conditional_losses_9960222
flatten_18/PartitionedCall?
 dense_36/StatefulPartitionedCallStatefulPartitionedCall#flatten_18/PartitionedCall:output:0dense_36_996052dense_36_996054*
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
GPU 2J 8? *M
fHRF
D__inference_dense_36_layer_call_and_return_conditional_losses_9960412"
 dense_36/StatefulPartitionedCall?
 dense_37/StatefulPartitionedCallStatefulPartitionedCall)dense_36/StatefulPartitionedCall:output:0dense_37_996079dense_37_996081*
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
GPU 2J 8? *M
fHRF
D__inference_dense_37_layer_call_and_return_conditional_losses_9960682"
 dense_37/StatefulPartitionedCall?
IdentityIdentity)dense_37/StatefulPartitionedCall:output:0"^conv2d_85/StatefulPartitionedCall"^conv2d_86/StatefulPartitionedCall!^dense_36/StatefulPartitionedCall!^dense_37/StatefulPartitionedCall#^dropout_32/StatefulPartitionedCall#^dropout_33/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:???????????::::::::2F
!conv2d_85/StatefulPartitionedCall!conv2d_85/StatefulPartitionedCall2F
!conv2d_86/StatefulPartitionedCall!conv2d_86/StatefulPartitionedCall2D
 dense_36/StatefulPartitionedCall dense_36/StatefulPartitionedCall2D
 dense_37/StatefulPartitionedCall dense_37/StatefulPartitionedCall2H
"dropout_32/StatefulPartitionedCall"dropout_32/StatefulPartitionedCall2H
"dropout_33/StatefulPartitionedCall"dropout_33/StatefulPartitionedCall:[ W
1
_output_shapes
:???????????
"
_user_specified_name
input_19
?

?
E__inference_conv2d_86_layer_call_and_return_conditional_losses_995969

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
-__inference_CNN_aug_drop_layer_call_fn_996163
input_19
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_19unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_CNN_aug_drop_layer_call_and_return_conditional_losses_9961442
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:???????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:???????????
"
_user_specified_name
input_19
?
G
+__inference_dropout_33_layer_call_fn_996467

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
GPU 2J 8? *O
fJRH
F__inference_dropout_33_layer_call_and_return_conditional_losses_9960022
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
?
d
+__inference_dropout_32_layer_call_fn_996415

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
GPU 2J 8? *O
fJRH
F__inference_dropout_32_layer_call_and_return_conditional_losses_9959402
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
??
?
"__inference__traced_restore_996749
file_prefix%
!assignvariableop_conv2d_85_kernel%
!assignvariableop_1_conv2d_85_bias'
#assignvariableop_2_conv2d_86_kernel%
!assignvariableop_3_conv2d_86_bias&
"assignvariableop_4_dense_36_kernel$
 assignvariableop_5_dense_36_bias&
"assignvariableop_6_dense_37_kernel$
 assignvariableop_7_dense_37_bias 
assignvariableop_8_adam_iter"
assignvariableop_9_adam_beta_1#
assignvariableop_10_adam_beta_2"
assignvariableop_11_adam_decay*
&assignvariableop_12_adam_learning_rate
assignvariableop_13_total
assignvariableop_14_count
assignvariableop_15_total_1
assignvariableop_16_count_1/
+assignvariableop_17_adam_conv2d_85_kernel_m-
)assignvariableop_18_adam_conv2d_85_bias_m/
+assignvariableop_19_adam_conv2d_86_kernel_m-
)assignvariableop_20_adam_conv2d_86_bias_m.
*assignvariableop_21_adam_dense_36_kernel_m,
(assignvariableop_22_adam_dense_36_bias_m.
*assignvariableop_23_adam_dense_37_kernel_m,
(assignvariableop_24_adam_dense_37_bias_m/
+assignvariableop_25_adam_conv2d_85_kernel_v-
)assignvariableop_26_adam_conv2d_85_bias_v/
+assignvariableop_27_adam_conv2d_86_kernel_v-
)assignvariableop_28_adam_conv2d_86_bias_v.
*assignvariableop_29_adam_dense_36_kernel_v,
(assignvariableop_30_adam_dense_36_bias_v.
*assignvariableop_31_adam_dense_37_kernel_v,
(assignvariableop_32_adam_dense_37_bias_v
identity_34??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*?
value?B?"B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::*0
dtypes&
$2"	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_85_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_85_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_86_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_86_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_36_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_36_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_37_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_37_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp+assignvariableop_17_adam_conv2d_85_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp)assignvariableop_18_adam_conv2d_85_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp+assignvariableop_19_adam_conv2d_86_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp)assignvariableop_20_adam_conv2d_86_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_dense_36_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_dense_36_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_dense_37_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_dense_37_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp+assignvariableop_25_adam_conv2d_85_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_conv2d_85_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_conv2d_86_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_conv2d_86_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_dense_36_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_dense_36_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_dense_37_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_dense_37_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_329
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_33Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_33?
Identity_34IdentityIdentity_33:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_34"#
identity_34Identity_34:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_32AssignVariableOp_322(
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
?
d
+__inference_dropout_33_layer_call_fn_996462

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
GPU 2J 8? *O
fJRH
F__inference_dropout_33_layer_call_and_return_conditional_losses_9959972
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
?
?
$__inference_signature_wrapper_996243
input_19
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_19unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_9958852
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:???????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:???????????
"
_user_specified_name
input_19
?
e
F__inference_dropout_32_layer_call_and_return_conditional_losses_995940

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
~
)__inference_dense_37_layer_call_fn_996518

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
GPU 2J 8? *M
fHRF
D__inference_dense_37_layer_call_and_return_conditional_losses_9960682
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
?
b
F__inference_flatten_18_layer_call_and_return_conditional_losses_996022

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
?
d
F__inference_dropout_32_layer_call_and_return_conditional_losses_996410

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
?H
?
__inference__traced_save_996640
file_prefix/
+savev2_conv2d_85_kernel_read_readvariableop-
)savev2_conv2d_85_bias_read_readvariableop/
+savev2_conv2d_86_kernel_read_readvariableop-
)savev2_conv2d_86_bias_read_readvariableop.
*savev2_dense_36_kernel_read_readvariableop,
(savev2_dense_36_bias_read_readvariableop.
*savev2_dense_37_kernel_read_readvariableop,
(savev2_dense_37_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_conv2d_85_kernel_m_read_readvariableop4
0savev2_adam_conv2d_85_bias_m_read_readvariableop6
2savev2_adam_conv2d_86_kernel_m_read_readvariableop4
0savev2_adam_conv2d_86_bias_m_read_readvariableop5
1savev2_adam_dense_36_kernel_m_read_readvariableop3
/savev2_adam_dense_36_bias_m_read_readvariableop5
1savev2_adam_dense_37_kernel_m_read_readvariableop3
/savev2_adam_dense_37_bias_m_read_readvariableop6
2savev2_adam_conv2d_85_kernel_v_read_readvariableop4
0savev2_adam_conv2d_85_bias_v_read_readvariableop6
2savev2_adam_conv2d_86_kernel_v_read_readvariableop4
0savev2_adam_conv2d_86_bias_v_read_readvariableop5
1savev2_adam_dense_36_kernel_v_read_readvariableop3
/savev2_adam_dense_36_bias_v_read_readvariableop5
1savev2_adam_dense_37_kernel_v_read_readvariableop3
/savev2_adam_dense_37_bias_v_read_readvariableop
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*?
value?B?"B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_85_kernel_read_readvariableop)savev2_conv2d_85_bias_read_readvariableop+savev2_conv2d_86_kernel_read_readvariableop)savev2_conv2d_86_bias_read_readvariableop*savev2_dense_36_kernel_read_readvariableop(savev2_dense_36_bias_read_readvariableop*savev2_dense_37_kernel_read_readvariableop(savev2_dense_37_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_conv2d_85_kernel_m_read_readvariableop0savev2_adam_conv2d_85_bias_m_read_readvariableop2savev2_adam_conv2d_86_kernel_m_read_readvariableop0savev2_adam_conv2d_86_bias_m_read_readvariableop1savev2_adam_dense_36_kernel_m_read_readvariableop/savev2_adam_dense_36_bias_m_read_readvariableop1savev2_adam_dense_37_kernel_m_read_readvariableop/savev2_adam_dense_37_bias_m_read_readvariableop2savev2_adam_conv2d_85_kernel_v_read_readvariableop0savev2_adam_conv2d_85_bias_v_read_readvariableop2savev2_adam_conv2d_86_kernel_v_read_readvariableop0savev2_adam_conv2d_86_bias_v_read_readvariableop1savev2_adam_dense_36_kernel_v_read_readvariableop/savev2_adam_dense_36_bias_v_read_readvariableop1savev2_adam_dense_37_kernel_v_read_readvariableop/savev2_adam_dense_37_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *0
dtypes&
$2"	2
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
?: :::::	?n : : :: : : : : : : : : :::::	?n : : ::::::	?n : : :: 2(
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
::%!

_output_shapes
:	?n : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::	

_output_shapes
: :


_output_shapes
: :
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
: :,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	?n : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	?n : 

_output_shapes
: :$  

_output_shapes

: : !

_output_shapes
::"

_output_shapes
: 
?
G
+__inference_flatten_18_layer_call_fn_996478

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
GPU 2J 8? *O
fJRH
F__inference_flatten_18_layer_call_and_return_conditional_losses_9960222
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
?
e
F__inference_dropout_33_layer_call_and_return_conditional_losses_996452

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
?
d
F__inference_dropout_32_layer_call_and_return_conditional_losses_995945

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
D__inference_dense_37_layer_call_and_return_conditional_losses_996068

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
?#
?
H__inference_CNN_aug_drop_layer_call_and_return_conditional_losses_996113
input_19
conv2d_85_996088
conv2d_85_996090
conv2d_86_996094
conv2d_86_996096
dense_36_996102
dense_36_996104
dense_37_996107
dense_37_996109
identity??!conv2d_85/StatefulPartitionedCall?!conv2d_86/StatefulPartitionedCall? dense_36/StatefulPartitionedCall? dense_37/StatefulPartitionedCall?
!conv2d_85/StatefulPartitionedCallStatefulPartitionedCallinput_19conv2d_85_996088conv2d_85_996090*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_85_layer_call_and_return_conditional_losses_9959122#
!conv2d_85/StatefulPartitionedCall?
dropout_32/PartitionedCallPartitionedCall*conv2d_85/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *O
fJRH
F__inference_dropout_32_layer_call_and_return_conditional_losses_9959452
dropout_32/PartitionedCall?
!conv2d_86/StatefulPartitionedCallStatefulPartitionedCall#dropout_32/PartitionedCall:output:0conv2d_86_996094conv2d_86_996096*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_86_layer_call_and_return_conditional_losses_9959692#
!conv2d_86/StatefulPartitionedCall?
dropout_33/PartitionedCallPartitionedCall*conv2d_86/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *O
fJRH
F__inference_dropout_33_layer_call_and_return_conditional_losses_9960022
dropout_33/PartitionedCall?
 max_pooling2d_34/PartitionedCallPartitionedCall#dropout_33/PartitionedCall:output:0*
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
GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_9958912"
 max_pooling2d_34/PartitionedCall?
flatten_18/PartitionedCallPartitionedCall)max_pooling2d_34/PartitionedCall:output:0*
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
GPU 2J 8? *O
fJRH
F__inference_flatten_18_layer_call_and_return_conditional_losses_9960222
flatten_18/PartitionedCall?
 dense_36/StatefulPartitionedCallStatefulPartitionedCall#flatten_18/PartitionedCall:output:0dense_36_996102dense_36_996104*
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
GPU 2J 8? *M
fHRF
D__inference_dense_36_layer_call_and_return_conditional_losses_9960412"
 dense_36/StatefulPartitionedCall?
 dense_37/StatefulPartitionedCallStatefulPartitionedCall)dense_36/StatefulPartitionedCall:output:0dense_37_996107dense_37_996109*
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
GPU 2J 8? *M
fHRF
D__inference_dense_37_layer_call_and_return_conditional_losses_9960682"
 dense_37/StatefulPartitionedCall?
IdentityIdentity)dense_37/StatefulPartitionedCall:output:0"^conv2d_85/StatefulPartitionedCall"^conv2d_86/StatefulPartitionedCall!^dense_36/StatefulPartitionedCall!^dense_37/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:???????????::::::::2F
!conv2d_85/StatefulPartitionedCall!conv2d_85/StatefulPartitionedCall2F
!conv2d_86/StatefulPartitionedCall!conv2d_86/StatefulPartitionedCall2D
 dense_36/StatefulPartitionedCall dense_36/StatefulPartitionedCall2D
 dense_37/StatefulPartitionedCall dense_37/StatefulPartitionedCall:[ W
1
_output_shapes
:???????????
"
_user_specified_name
input_19
?&
?
H__inference_CNN_aug_drop_layer_call_and_return_conditional_losses_996144

inputs
conv2d_85_996119
conv2d_85_996121
conv2d_86_996125
conv2d_86_996127
dense_36_996133
dense_36_996135
dense_37_996138
dense_37_996140
identity??!conv2d_85/StatefulPartitionedCall?!conv2d_86/StatefulPartitionedCall? dense_36/StatefulPartitionedCall? dense_37/StatefulPartitionedCall?"dropout_32/StatefulPartitionedCall?"dropout_33/StatefulPartitionedCall?
!conv2d_85/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_85_996119conv2d_85_996121*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_85_layer_call_and_return_conditional_losses_9959122#
!conv2d_85/StatefulPartitionedCall?
"dropout_32/StatefulPartitionedCallStatefulPartitionedCall*conv2d_85/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *O
fJRH
F__inference_dropout_32_layer_call_and_return_conditional_losses_9959402$
"dropout_32/StatefulPartitionedCall?
!conv2d_86/StatefulPartitionedCallStatefulPartitionedCall+dropout_32/StatefulPartitionedCall:output:0conv2d_86_996125conv2d_86_996127*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_86_layer_call_and_return_conditional_losses_9959692#
!conv2d_86/StatefulPartitionedCall?
"dropout_33/StatefulPartitionedCallStatefulPartitionedCall*conv2d_86/StatefulPartitionedCall:output:0#^dropout_32/StatefulPartitionedCall*
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
GPU 2J 8? *O
fJRH
F__inference_dropout_33_layer_call_and_return_conditional_losses_9959972$
"dropout_33/StatefulPartitionedCall?
 max_pooling2d_34/PartitionedCallPartitionedCall+dropout_33/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_9958912"
 max_pooling2d_34/PartitionedCall?
flatten_18/PartitionedCallPartitionedCall)max_pooling2d_34/PartitionedCall:output:0*
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
GPU 2J 8? *O
fJRH
F__inference_flatten_18_layer_call_and_return_conditional_losses_9960222
flatten_18/PartitionedCall?
 dense_36/StatefulPartitionedCallStatefulPartitionedCall#flatten_18/PartitionedCall:output:0dense_36_996133dense_36_996135*
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
GPU 2J 8? *M
fHRF
D__inference_dense_36_layer_call_and_return_conditional_losses_9960412"
 dense_36/StatefulPartitionedCall?
 dense_37/StatefulPartitionedCallStatefulPartitionedCall)dense_36/StatefulPartitionedCall:output:0dense_37_996138dense_37_996140*
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
GPU 2J 8? *M
fHRF
D__inference_dense_37_layer_call_and_return_conditional_losses_9960682"
 dense_37/StatefulPartitionedCall?
IdentityIdentity)dense_37/StatefulPartitionedCall:output:0"^conv2d_85/StatefulPartitionedCall"^conv2d_86/StatefulPartitionedCall!^dense_36/StatefulPartitionedCall!^dense_37/StatefulPartitionedCall#^dropout_32/StatefulPartitionedCall#^dropout_33/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:???????????::::::::2F
!conv2d_85/StatefulPartitionedCall!conv2d_85/StatefulPartitionedCall2F
!conv2d_86/StatefulPartitionedCall!conv2d_86/StatefulPartitionedCall2D
 dense_36/StatefulPartitionedCall dense_36/StatefulPartitionedCall2D
 dense_37/StatefulPartitionedCall dense_37/StatefulPartitionedCall2H
"dropout_32/StatefulPartitionedCall"dropout_32/StatefulPartitionedCall2H
"dropout_33/StatefulPartitionedCall"dropout_33/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?;
?
!__inference__wrapped_model_995885
input_199
5cnn_aug_drop_conv2d_85_conv2d_readvariableop_resource:
6cnn_aug_drop_conv2d_85_biasadd_readvariableop_resource9
5cnn_aug_drop_conv2d_86_conv2d_readvariableop_resource:
6cnn_aug_drop_conv2d_86_biasadd_readvariableop_resource8
4cnn_aug_drop_dense_36_matmul_readvariableop_resource9
5cnn_aug_drop_dense_36_biasadd_readvariableop_resource8
4cnn_aug_drop_dense_37_matmul_readvariableop_resource9
5cnn_aug_drop_dense_37_biasadd_readvariableop_resource
identity??-CNN_aug_drop/conv2d_85/BiasAdd/ReadVariableOp?,CNN_aug_drop/conv2d_85/Conv2D/ReadVariableOp?-CNN_aug_drop/conv2d_86/BiasAdd/ReadVariableOp?,CNN_aug_drop/conv2d_86/Conv2D/ReadVariableOp?,CNN_aug_drop/dense_36/BiasAdd/ReadVariableOp?+CNN_aug_drop/dense_36/MatMul/ReadVariableOp?,CNN_aug_drop/dense_37/BiasAdd/ReadVariableOp?+CNN_aug_drop/dense_37/MatMul/ReadVariableOp?
,CNN_aug_drop/conv2d_85/Conv2D/ReadVariableOpReadVariableOp5cnn_aug_drop_conv2d_85_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02.
,CNN_aug_drop/conv2d_85/Conv2D/ReadVariableOp?
CNN_aug_drop/conv2d_85/Conv2DConv2Dinput_194CNN_aug_drop/conv2d_85/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
CNN_aug_drop/conv2d_85/Conv2D?
-CNN_aug_drop/conv2d_85/BiasAdd/ReadVariableOpReadVariableOp6cnn_aug_drop_conv2d_85_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-CNN_aug_drop/conv2d_85/BiasAdd/ReadVariableOp?
CNN_aug_drop/conv2d_85/BiasAddBiasAdd&CNN_aug_drop/conv2d_85/Conv2D:output:05CNN_aug_drop/conv2d_85/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2 
CNN_aug_drop/conv2d_85/BiasAdd?
CNN_aug_drop/conv2d_85/ReluRelu'CNN_aug_drop/conv2d_85/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
CNN_aug_drop/conv2d_85/Relu?
 CNN_aug_drop/dropout_32/IdentityIdentity)CNN_aug_drop/conv2d_85/Relu:activations:0*
T0*1
_output_shapes
:???????????2"
 CNN_aug_drop/dropout_32/Identity?
,CNN_aug_drop/conv2d_86/Conv2D/ReadVariableOpReadVariableOp5cnn_aug_drop_conv2d_86_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02.
,CNN_aug_drop/conv2d_86/Conv2D/ReadVariableOp?
CNN_aug_drop/conv2d_86/Conv2DConv2D)CNN_aug_drop/dropout_32/Identity:output:04CNN_aug_drop/conv2d_86/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
CNN_aug_drop/conv2d_86/Conv2D?
-CNN_aug_drop/conv2d_86/BiasAdd/ReadVariableOpReadVariableOp6cnn_aug_drop_conv2d_86_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-CNN_aug_drop/conv2d_86/BiasAdd/ReadVariableOp?
CNN_aug_drop/conv2d_86/BiasAddBiasAdd&CNN_aug_drop/conv2d_86/Conv2D:output:05CNN_aug_drop/conv2d_86/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2 
CNN_aug_drop/conv2d_86/BiasAdd?
CNN_aug_drop/conv2d_86/ReluRelu'CNN_aug_drop/conv2d_86/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
CNN_aug_drop/conv2d_86/Relu?
 CNN_aug_drop/dropout_33/IdentityIdentity)CNN_aug_drop/conv2d_86/Relu:activations:0*
T0*1
_output_shapes
:???????????2"
 CNN_aug_drop/dropout_33/Identity?
%CNN_aug_drop/max_pooling2d_34/MaxPoolMaxPool)CNN_aug_drop/dropout_33/Identity:output:0*/
_output_shapes
:?????????***
ksize
*
paddingVALID*
strides
2'
%CNN_aug_drop/max_pooling2d_34/MaxPool?
CNN_aug_drop/flatten_18/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 7  2
CNN_aug_drop/flatten_18/Const?
CNN_aug_drop/flatten_18/ReshapeReshape.CNN_aug_drop/max_pooling2d_34/MaxPool:output:0&CNN_aug_drop/flatten_18/Const:output:0*
T0*(
_output_shapes
:??????????n2!
CNN_aug_drop/flatten_18/Reshape?
+CNN_aug_drop/dense_36/MatMul/ReadVariableOpReadVariableOp4cnn_aug_drop_dense_36_matmul_readvariableop_resource*
_output_shapes
:	?n *
dtype02-
+CNN_aug_drop/dense_36/MatMul/ReadVariableOp?
CNN_aug_drop/dense_36/MatMulMatMul(CNN_aug_drop/flatten_18/Reshape:output:03CNN_aug_drop/dense_36/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
CNN_aug_drop/dense_36/MatMul?
,CNN_aug_drop/dense_36/BiasAdd/ReadVariableOpReadVariableOp5cnn_aug_drop_dense_36_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,CNN_aug_drop/dense_36/BiasAdd/ReadVariableOp?
CNN_aug_drop/dense_36/BiasAddBiasAdd&CNN_aug_drop/dense_36/MatMul:product:04CNN_aug_drop/dense_36/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
CNN_aug_drop/dense_36/BiasAdd?
CNN_aug_drop/dense_36/ReluRelu&CNN_aug_drop/dense_36/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
CNN_aug_drop/dense_36/Relu?
+CNN_aug_drop/dense_37/MatMul/ReadVariableOpReadVariableOp4cnn_aug_drop_dense_37_matmul_readvariableop_resource*
_output_shapes

: *
dtype02-
+CNN_aug_drop/dense_37/MatMul/ReadVariableOp?
CNN_aug_drop/dense_37/MatMulMatMul(CNN_aug_drop/dense_36/Relu:activations:03CNN_aug_drop/dense_37/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
CNN_aug_drop/dense_37/MatMul?
,CNN_aug_drop/dense_37/BiasAdd/ReadVariableOpReadVariableOp5cnn_aug_drop_dense_37_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,CNN_aug_drop/dense_37/BiasAdd/ReadVariableOp?
CNN_aug_drop/dense_37/BiasAddBiasAdd&CNN_aug_drop/dense_37/MatMul:product:04CNN_aug_drop/dense_37/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
CNN_aug_drop/dense_37/BiasAdd?
CNN_aug_drop/dense_37/SoftmaxSoftmax&CNN_aug_drop/dense_37/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
CNN_aug_drop/dense_37/Softmax?
IdentityIdentity'CNN_aug_drop/dense_37/Softmax:softmax:0.^CNN_aug_drop/conv2d_85/BiasAdd/ReadVariableOp-^CNN_aug_drop/conv2d_85/Conv2D/ReadVariableOp.^CNN_aug_drop/conv2d_86/BiasAdd/ReadVariableOp-^CNN_aug_drop/conv2d_86/Conv2D/ReadVariableOp-^CNN_aug_drop/dense_36/BiasAdd/ReadVariableOp,^CNN_aug_drop/dense_36/MatMul/ReadVariableOp-^CNN_aug_drop/dense_37/BiasAdd/ReadVariableOp,^CNN_aug_drop/dense_37/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:???????????::::::::2^
-CNN_aug_drop/conv2d_85/BiasAdd/ReadVariableOp-CNN_aug_drop/conv2d_85/BiasAdd/ReadVariableOp2\
,CNN_aug_drop/conv2d_85/Conv2D/ReadVariableOp,CNN_aug_drop/conv2d_85/Conv2D/ReadVariableOp2^
-CNN_aug_drop/conv2d_86/BiasAdd/ReadVariableOp-CNN_aug_drop/conv2d_86/BiasAdd/ReadVariableOp2\
,CNN_aug_drop/conv2d_86/Conv2D/ReadVariableOp,CNN_aug_drop/conv2d_86/Conv2D/ReadVariableOp2\
,CNN_aug_drop/dense_36/BiasAdd/ReadVariableOp,CNN_aug_drop/dense_36/BiasAdd/ReadVariableOp2Z
+CNN_aug_drop/dense_36/MatMul/ReadVariableOp+CNN_aug_drop/dense_36/MatMul/ReadVariableOp2\
,CNN_aug_drop/dense_37/BiasAdd/ReadVariableOp,CNN_aug_drop/dense_37/BiasAdd/ReadVariableOp2Z
+CNN_aug_drop/dense_37/MatMul/ReadVariableOp+CNN_aug_drop/dense_37/MatMul/ReadVariableOp:[ W
1
_output_shapes
:???????????
"
_user_specified_name
input_19
?	
?
D__inference_dense_36_layer_call_and_return_conditional_losses_996489

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
?

?
E__inference_conv2d_86_layer_call_and_return_conditional_losses_996431

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
e
F__inference_dropout_33_layer_call_and_return_conditional_losses_995997

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

*__inference_conv2d_86_layer_call_fn_996440

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
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_86_layer_call_and_return_conditional_losses_9959692
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
?
?
-__inference_CNN_aug_drop_layer_call_fn_996212
input_19
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_19unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_CNN_aug_drop_layer_call_and_return_conditional_losses_9961932
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:???????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:???????????
"
_user_specified_name
input_19
?
d
F__inference_dropout_33_layer_call_and_return_conditional_losses_996002

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

*__inference_conv2d_85_layer_call_fn_996393

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
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_85_layer_call_and_return_conditional_losses_9959122
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
input_19;
serving_default_input_19:0???????????<
dense_370
StatefulPartitionedCall:0?????????tensorflow/serving/predict:׎
?G
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer_with_weights-3
	layer-8

	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
?_default_save_signature
+?&call_and_return_all_conditional_losses
?__call__"?D
_tf_keras_network?C{"class_name": "Functional", "name": "CNN_aug_drop", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "CNN_aug_drop", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_19"}, "name": "input_19", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_85", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_85", "inbound_nodes": [[["input_19", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_32", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "name": "dropout_32", "inbound_nodes": [[["conv2d_85", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_86", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_86", "inbound_nodes": [[["dropout_32", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_33", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "name": "dropout_33", "inbound_nodes": [[["conv2d_86", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_34", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "name": "max_pooling2d_34", "inbound_nodes": [[["dropout_33", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_18", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_18", "inbound_nodes": [[["max_pooling2d_34", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_36", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_36", "inbound_nodes": [[["flatten_18", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_37", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_37", "inbound_nodes": [[["dense_36", 0, 0, {}]]]}], "input_layers": [["input_19", 0, 0]], "output_layers": [["dense_37", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "CNN_aug_drop", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_19"}, "name": "input_19", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_85", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_85", "inbound_nodes": [[["input_19", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_32", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "name": "dropout_32", "inbound_nodes": [[["conv2d_85", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_86", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_86", "inbound_nodes": [[["dropout_32", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_33", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "name": "dropout_33", "inbound_nodes": [[["conv2d_86", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_34", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "name": "max_pooling2d_34", "inbound_nodes": [[["dropout_33", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_18", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_18", "inbound_nodes": [[["max_pooling2d_34", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_36", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_36", "inbound_nodes": [[["flatten_18", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_37", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_37", "inbound_nodes": [[["dense_36", 0, 0, {}]]]}], "input_layers": [["input_19", 0, 0]], "output_layers": [["dense_37", 0, 0]]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_19", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_19"}}
?	

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_85", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_85", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 1]}}
?
regularization_losses
trainable_variables
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_32", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_32", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}
?	

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_86", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_86", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 8]}}
?
 regularization_losses
!trainable_variables
"	variables
#	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_33", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_33", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}
?
$regularization_losses
%trainable_variables
&	variables
'	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_34", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_34", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
(regularization_losses
)trainable_variables
*	variables
+	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_18", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

,kernel
-bias
.regularization_losses
/trainable_variables
0	variables
1	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_36", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_36", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 14112}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14112]}}
?

2kernel
3bias
4regularization_losses
5trainable_variables
6	variables
7	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_37", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_37", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
?
8iter

9beta_1

:beta_2
	;decay
<learning_ratemumvmwmx,my-mz2m{3m|v}v~vv?,v?-v?2v?3v?"
	optimizer
 "
trackable_list_wrapper
X
0
1
2
3
,4
-5
26
37"
trackable_list_wrapper
X
0
1
2
3
,4
-5
26
37"
trackable_list_wrapper
?
regularization_losses

=layers
>layer_metrics
?metrics
trainable_variables
	variables
@layer_regularization_losses
Anon_trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
*:(2conv2d_85/kernel
:2conv2d_85/bias
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
?
regularization_losses

Blayers
Clayer_metrics
Dmetrics
trainable_variables
	variables
Elayer_regularization_losses
Fnon_trainable_variables
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
regularization_losses

Glayers
Hlayer_metrics
Imetrics
trainable_variables
	variables
Jlayer_regularization_losses
Knon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(2conv2d_86/kernel
:2conv2d_86/bias
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

Llayers
Mlayer_metrics
Nmetrics
trainable_variables
	variables
Olayer_regularization_losses
Pnon_trainable_variables
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
 regularization_losses

Qlayers
Rlayer_metrics
Smetrics
!trainable_variables
"	variables
Tlayer_regularization_losses
Unon_trainable_variables
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
$regularization_losses

Vlayers
Wlayer_metrics
Xmetrics
%trainable_variables
&	variables
Ylayer_regularization_losses
Znon_trainable_variables
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
(regularization_losses

[layers
\layer_metrics
]metrics
)trainable_variables
*	variables
^layer_regularization_losses
_non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	?n 2dense_36/kernel
: 2dense_36/bias
 "
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
?
.regularization_losses

`layers
alayer_metrics
bmetrics
/trainable_variables
0	variables
clayer_regularization_losses
dnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!: 2dense_37/kernel
:2dense_37/bias
 "
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
?
4regularization_losses

elayers
flayer_metrics
gmetrics
5trainable_variables
6	variables
hlayer_regularization_losses
inon_trainable_variables
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
j0
k1"
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
	ltotal
	mcount
n	variables
o	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?
	ptotal
	qcount
r
_fn_kwargs
s	variables
t	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
:  (2total
:  (2count
.
l0
m1"
trackable_list_wrapper
-
n	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
p0
q1"
trackable_list_wrapper
-
s	variables"
_generic_user_object
/:-2Adam/conv2d_85/kernel/m
!:2Adam/conv2d_85/bias/m
/:-2Adam/conv2d_86/kernel/m
!:2Adam/conv2d_86/bias/m
':%	?n 2Adam/dense_36/kernel/m
 : 2Adam/dense_36/bias/m
&:$ 2Adam/dense_37/kernel/m
 :2Adam/dense_37/bias/m
/:-2Adam/conv2d_85/kernel/v
!:2Adam/conv2d_85/bias/v
/:-2Adam/conv2d_86/kernel/v
!:2Adam/conv2d_86/bias/v
':%	?n 2Adam/dense_36/kernel/v
 : 2Adam/dense_36/bias/v
&:$ 2Adam/dense_37/kernel/v
 :2Adam/dense_37/bias/v
?2?
!__inference__wrapped_model_995885?
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
input_19???????????
?2?
H__inference_CNN_aug_drop_layer_call_and_return_conditional_losses_996294
H__inference_CNN_aug_drop_layer_call_and_return_conditional_losses_996331
H__inference_CNN_aug_drop_layer_call_and_return_conditional_losses_996113
H__inference_CNN_aug_drop_layer_call_and_return_conditional_losses_996085?
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
?2?
-__inference_CNN_aug_drop_layer_call_fn_996163
-__inference_CNN_aug_drop_layer_call_fn_996212
-__inference_CNN_aug_drop_layer_call_fn_996373
-__inference_CNN_aug_drop_layer_call_fn_996352?
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
E__inference_conv2d_85_layer_call_and_return_conditional_losses_996384?
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
*__inference_conv2d_85_layer_call_fn_996393?
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
F__inference_dropout_32_layer_call_and_return_conditional_losses_996405
F__inference_dropout_32_layer_call_and_return_conditional_losses_996410?
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
+__inference_dropout_32_layer_call_fn_996420
+__inference_dropout_32_layer_call_fn_996415?
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
E__inference_conv2d_86_layer_call_and_return_conditional_losses_996431?
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
*__inference_conv2d_86_layer_call_fn_996440?
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
F__inference_dropout_33_layer_call_and_return_conditional_losses_996457
F__inference_dropout_33_layer_call_and_return_conditional_losses_996452?
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
+__inference_dropout_33_layer_call_fn_996467
+__inference_dropout_33_layer_call_fn_996462?
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
L__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_995891?
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
1__inference_max_pooling2d_34_layer_call_fn_995897?
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
F__inference_flatten_18_layer_call_and_return_conditional_losses_996473?
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
+__inference_flatten_18_layer_call_fn_996478?
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
D__inference_dense_36_layer_call_and_return_conditional_losses_996489?
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
)__inference_dense_36_layer_call_fn_996498?
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
D__inference_dense_37_layer_call_and_return_conditional_losses_996509?
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
)__inference_dense_37_layer_call_fn_996518?
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
$__inference_signature_wrapper_996243input_19"?
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
H__inference_CNN_aug_drop_layer_call_and_return_conditional_losses_996085v,-23C?@
9?6
,?)
input_19???????????
p

 
? "%?"
?
0?????????
? ?
H__inference_CNN_aug_drop_layer_call_and_return_conditional_losses_996113v,-23C?@
9?6
,?)
input_19???????????
p 

 
? "%?"
?
0?????????
? ?
H__inference_CNN_aug_drop_layer_call_and_return_conditional_losses_996294t,-23A?>
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
H__inference_CNN_aug_drop_layer_call_and_return_conditional_losses_996331t,-23A?>
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
-__inference_CNN_aug_drop_layer_call_fn_996163i,-23C?@
9?6
,?)
input_19???????????
p

 
? "???????????
-__inference_CNN_aug_drop_layer_call_fn_996212i,-23C?@
9?6
,?)
input_19???????????
p 

 
? "???????????
-__inference_CNN_aug_drop_layer_call_fn_996352g,-23A?>
7?4
*?'
inputs???????????
p

 
? "???????????
-__inference_CNN_aug_drop_layer_call_fn_996373g,-23A?>
7?4
*?'
inputs???????????
p 

 
? "???????????
!__inference__wrapped_model_995885|,-23;?8
1?.
,?)
input_19???????????
? "3?0
.
dense_37"?
dense_37??????????
E__inference_conv2d_85_layer_call_and_return_conditional_losses_996384p9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
*__inference_conv2d_85_layer_call_fn_996393c9?6
/?,
*?'
inputs???????????
? ""?????????????
E__inference_conv2d_86_layer_call_and_return_conditional_losses_996431p9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
*__inference_conv2d_86_layer_call_fn_996440c9?6
/?,
*?'
inputs???????????
? ""?????????????
D__inference_dense_36_layer_call_and_return_conditional_losses_996489],-0?-
&?#
!?
inputs??????????n
? "%?"
?
0????????? 
? }
)__inference_dense_36_layer_call_fn_996498P,-0?-
&?#
!?
inputs??????????n
? "?????????? ?
D__inference_dense_37_layer_call_and_return_conditional_losses_996509\23/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????
? |
)__inference_dense_37_layer_call_fn_996518O23/?,
%?"
 ?
inputs????????? 
? "???????????
F__inference_dropout_32_layer_call_and_return_conditional_losses_996405p=?:
3?0
*?'
inputs???????????
p
? "/?,
%?"
0???????????
? ?
F__inference_dropout_32_layer_call_and_return_conditional_losses_996410p=?:
3?0
*?'
inputs???????????
p 
? "/?,
%?"
0???????????
? ?
+__inference_dropout_32_layer_call_fn_996415c=?:
3?0
*?'
inputs???????????
p
? ""?????????????
+__inference_dropout_32_layer_call_fn_996420c=?:
3?0
*?'
inputs???????????
p 
? ""?????????????
F__inference_dropout_33_layer_call_and_return_conditional_losses_996452p=?:
3?0
*?'
inputs???????????
p
? "/?,
%?"
0???????????
? ?
F__inference_dropout_33_layer_call_and_return_conditional_losses_996457p=?:
3?0
*?'
inputs???????????
p 
? "/?,
%?"
0???????????
? ?
+__inference_dropout_33_layer_call_fn_996462c=?:
3?0
*?'
inputs???????????
p
? ""?????????????
+__inference_dropout_33_layer_call_fn_996467c=?:
3?0
*?'
inputs???????????
p 
? ""?????????????
F__inference_flatten_18_layer_call_and_return_conditional_losses_996473a7?4
-?*
(?%
inputs?????????**
? "&?#
?
0??????????n
? ?
+__inference_flatten_18_layer_call_fn_996478T7?4
-?*
(?%
inputs?????????**
? "???????????n?
L__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_995891?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
1__inference_max_pooling2d_34_layer_call_fn_995897?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
$__inference_signature_wrapper_996243?,-23G?D
? 
=?:
8
input_19,?)
input_19???????????"3?0
.
dense_37"?
dense_37?????????