??
??
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
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
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
-
Sqrt
x"T
y"T"
Ttype:

2
3
Square
x"T
y"T"
Ttype:
2
	
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
<
Sub
x"T
y"T
z"T"
Ttype:
2	
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02v2.5.0-0-ga4dfb8d1a718??
l
RMSprop/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameRMSprop/iter
e
 RMSprop/iter/Read/ReadVariableOpReadVariableOpRMSprop/iter*
_output_shapes
: *
dtype0	
n
RMSprop/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/decay
g
!RMSprop/decay/Read/ReadVariableOpReadVariableOpRMSprop/decay*
_output_shapes
: *
dtype0
~
RMSprop/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameRMSprop/learning_rate
w
)RMSprop/learning_rate/Read/ReadVariableOpReadVariableOpRMSprop/learning_rate*
_output_shapes
: *
dtype0
t
RMSprop/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameRMSprop/momentum
m
$RMSprop/momentum/Read/ReadVariableOpReadVariableOpRMSprop/momentum*
_output_shapes
: *
dtype0
j
RMSprop/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/rho
c
RMSprop/rho/Read/ReadVariableOpReadVariableOpRMSprop/rho*
_output_shapes
: *
dtype0
?
conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_4/kernel
{
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*&
_output_shapes
:*
dtype0
r
conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_4/bias
k
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes
:*
dtype0
?
conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_5/kernel
{
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*&
_output_shapes
:*
dtype0
r
conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_5/bias
k
!conv2d_5/bias/Read/ReadVariableOpReadVariableOpconv2d_5/bias*
_output_shapes
:*
dtype0
z
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense_4/kernel
s
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel* 
_output_shapes
:
??*
dtype0
q
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_4/bias
j
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes	
:?*
dtype0
y
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?2*
shared_namedense_5/kernel
r
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes
:	?2*
dtype0
p
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
:2*
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
?
RMSprop/conv2d_4/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameRMSprop/conv2d_4/kernel/rms
?
/RMSprop/conv2d_4/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_4/kernel/rms*&
_output_shapes
:*
dtype0
?
RMSprop/conv2d_4/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameRMSprop/conv2d_4/bias/rms
?
-RMSprop/conv2d_4/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_4/bias/rms*
_output_shapes
:*
dtype0
?
RMSprop/conv2d_5/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameRMSprop/conv2d_5/kernel/rms
?
/RMSprop/conv2d_5/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_5/kernel/rms*&
_output_shapes
:*
dtype0
?
RMSprop/conv2d_5/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameRMSprop/conv2d_5/bias/rms
?
-RMSprop/conv2d_5/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_5/bias/rms*
_output_shapes
:*
dtype0
?
RMSprop/dense_4/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*+
shared_nameRMSprop/dense_4/kernel/rms
?
.RMSprop/dense_4/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_4/kernel/rms* 
_output_shapes
:
??*
dtype0
?
RMSprop/dense_4/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_nameRMSprop/dense_4/bias/rms
?
,RMSprop/dense_4/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_4/bias/rms*
_output_shapes	
:?*
dtype0
?
RMSprop/dense_5/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?2*+
shared_nameRMSprop/dense_5/kernel/rms
?
.RMSprop/dense_5/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_5/kernel/rms*
_output_shapes
:	?2*
dtype0
?
RMSprop/dense_5/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*)
shared_nameRMSprop/dense_5/bias/rms
?
,RMSprop/dense_5/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_5/bias/rms*
_output_shapes
:2*
dtype0

NoOpNoOp
?<
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?;
value?;B?; B?;
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
	optimizer
trainable_variables
	variables
regularization_losses
		keras_api


signatures
 
 
?
layer_with_weights-0
layer-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer-6
layer-7
layer-8
layer_with_weights-2
layer-9
layer-10
layer_with_weights-3
layer-11
trainable_variables
	variables
regularization_losses
	keras_api
R
trainable_variables
	variables
regularization_losses
	keras_api
?
iter
	 decay
!learning_rate
"momentum
#rho
$rms?
%rms?
&rms?
'rms?
(rms?
)rms?
*rms?
+rms?
8
$0
%1
&2
'3
(4
)5
*6
+7
8
$0
%1
&2
'3
(4
)5
*6
+7
 
?
trainable_variables
,layer_metrics
-metrics
	variables
.layer_regularization_losses
regularization_losses

/layers
0non_trainable_variables
 
h

$kernel
%bias
1trainable_variables
2	variables
3regularization_losses
4	keras_api
R
5trainable_variables
6	variables
7regularization_losses
8	keras_api
R
9trainable_variables
:	variables
;regularization_losses
<	keras_api
R
=trainable_variables
>	variables
?regularization_losses
@	keras_api
h

&kernel
'bias
Atrainable_variables
B	variables
Cregularization_losses
D	keras_api
R
Etrainable_variables
F	variables
Gregularization_losses
H	keras_api
R
Itrainable_variables
J	variables
Kregularization_losses
L	keras_api
R
Mtrainable_variables
N	variables
Oregularization_losses
P	keras_api
R
Qtrainable_variables
R	variables
Sregularization_losses
T	keras_api
h

(kernel
)bias
Utrainable_variables
V	variables
Wregularization_losses
X	keras_api
R
Ytrainable_variables
Z	variables
[regularization_losses
\	keras_api
h

*kernel
+bias
]trainable_variables
^	variables
_regularization_losses
`	keras_api
8
$0
%1
&2
'3
(4
)5
*6
+7
8
$0
%1
&2
'3
(4
)5
*6
+7
 
?
trainable_variables
alayer_metrics
bmetrics
	variables
clayer_regularization_losses
regularization_losses

dlayers
enon_trainable_variables
 
 
 
?
trainable_variables
flayer_metrics
gmetrics
	variables
hlayer_regularization_losses
regularization_losses

ilayers
jnon_trainable_variables
KI
VARIABLE_VALUERMSprop/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUERMSprop/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUERMSprop/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUERMSprop/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUERMSprop/rho(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d_4/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEconv2d_4/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d_5/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEconv2d_5/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_4/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEdense_4/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_5/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEdense_5/bias0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
 

k0
 

0
1
2
3
 

$0
%1

$0
%1
 
?
1trainable_variables
llayer_metrics
mmetrics
2	variables
nlayer_regularization_losses
3regularization_losses

olayers
pnon_trainable_variables
 
 
 
?
5trainable_variables
qlayer_metrics
rmetrics
6	variables
slayer_regularization_losses
7regularization_losses

tlayers
unon_trainable_variables
 
 
 
?
9trainable_variables
vlayer_metrics
wmetrics
:	variables
xlayer_regularization_losses
;regularization_losses

ylayers
znon_trainable_variables
 
 
 
?
=trainable_variables
{layer_metrics
|metrics
>	variables
}layer_regularization_losses
?regularization_losses

~layers
non_trainable_variables

&0
'1

&0
'1
 
?
Atrainable_variables
?layer_metrics
?metrics
B	variables
 ?layer_regularization_losses
Cregularization_losses
?layers
?non_trainable_variables
 
 
 
?
Etrainable_variables
?layer_metrics
?metrics
F	variables
 ?layer_regularization_losses
Gregularization_losses
?layers
?non_trainable_variables
 
 
 
?
Itrainable_variables
?layer_metrics
?metrics
J	variables
 ?layer_regularization_losses
Kregularization_losses
?layers
?non_trainable_variables
 
 
 
?
Mtrainable_variables
?layer_metrics
?metrics
N	variables
 ?layer_regularization_losses
Oregularization_losses
?layers
?non_trainable_variables
 
 
 
?
Qtrainable_variables
?layer_metrics
?metrics
R	variables
 ?layer_regularization_losses
Sregularization_losses
?layers
?non_trainable_variables

(0
)1

(0
)1
 
?
Utrainable_variables
?layer_metrics
?metrics
V	variables
 ?layer_regularization_losses
Wregularization_losses
?layers
?non_trainable_variables
 
 
 
?
Ytrainable_variables
?layer_metrics
?metrics
Z	variables
 ?layer_regularization_losses
[regularization_losses
?layers
?non_trainable_variables

*0
+1

*0
+1
 
?
]trainable_variables
?layer_metrics
?metrics
^	variables
 ?layer_regularization_losses
_regularization_losses
?layers
?non_trainable_variables
 
 
 
V
0
1
2
3
4
5
6
7
8
9
10
11
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
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
}
VARIABLE_VALUERMSprop/conv2d_4/kernel/rmsNtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUERMSprop/conv2d_4/bias/rmsNtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUERMSprop/conv2d_5/kernel/rmsNtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUERMSprop/conv2d_5/bias/rmsNtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUERMSprop/dense_4/kernel/rmsNtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUERMSprop/dense_4/bias/rmsNtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUERMSprop/dense_5/kernel/rmsNtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUERMSprop/dense_5/bias/rmsNtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_5Placeholder*/
_output_shapes
:?????????8.*
dtype0*$
shape:?????????8.
?
serving_default_input_6Placeholder*/
_output_shapes
:?????????8.*
dtype0*$
shape:?????????8.
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_5serving_default_input_6conv2d_4/kernelconv2d_4/biasconv2d_5/kernelconv2d_5/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/bias*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference_signature_wrapper_15567
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOp!conv2d_4/bias/Read/ReadVariableOp#conv2d_5/kernel/Read/ReadVariableOp!conv2d_5/bias/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp/RMSprop/conv2d_4/kernel/rms/Read/ReadVariableOp-RMSprop/conv2d_4/bias/rms/Read/ReadVariableOp/RMSprop/conv2d_5/kernel/rms/Read/ReadVariableOp-RMSprop/conv2d_5/bias/rms/Read/ReadVariableOp.RMSprop/dense_4/kernel/rms/Read/ReadVariableOp,RMSprop/dense_4/bias/rms/Read/ReadVariableOp.RMSprop/dense_5/kernel/rms/Read/ReadVariableOp,RMSprop/dense_5/bias/rms/Read/ReadVariableOpConst*$
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference__traced_save_16261
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rhoconv2d_4/kernelconv2d_4/biasconv2d_5/kernelconv2d_5/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/biastotalcountRMSprop/conv2d_4/kernel/rmsRMSprop/conv2d_4/bias/rmsRMSprop/conv2d_5/kernel/rmsRMSprop/conv2d_5/bias/rmsRMSprop/dense_4/kernel/rmsRMSprop/dense_4/bias/rmsRMSprop/dense_5/kernel/rmsRMSprop/dense_5/bias/rms*#
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__traced_restore_16340??
?
c
D__inference_dropout_8_layer_call_and_return_conditional_losses_15025

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
b
D__inference_dropout_8_layer_call_and_return_conditional_losses_14956

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
'__inference_model_3_layer_call_fn_15473
input_5
input_6!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?2
	unknown_6:2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_5input_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_model_3_layer_call_and_return_conditional_losses_154322
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:?????????8.:?????????8.: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????8.
!
_user_specified_name	input_5:XT
/
_output_shapes
:?????????8.
!
_user_specified_name	input_6
??
?
B__inference_model_3_layer_call_and_return_conditional_losses_15685
inputs_0
inputs_1N
4sequential_2_conv2d_4_conv2d_readvariableop_resource:C
5sequential_2_conv2d_4_biasadd_readvariableop_resource:N
4sequential_2_conv2d_5_conv2d_readvariableop_resource:C
5sequential_2_conv2d_5_biasadd_readvariableop_resource:G
3sequential_2_dense_4_matmul_readvariableop_resource:
??C
4sequential_2_dense_4_biasadd_readvariableop_resource:	?F
3sequential_2_dense_5_matmul_readvariableop_resource:	?2B
4sequential_2_dense_5_biasadd_readvariableop_resource:2
identity??,sequential_2/conv2d_4/BiasAdd/ReadVariableOp?.sequential_2/conv2d_4/BiasAdd_1/ReadVariableOp?+sequential_2/conv2d_4/Conv2D/ReadVariableOp?-sequential_2/conv2d_4/Conv2D_1/ReadVariableOp?,sequential_2/conv2d_5/BiasAdd/ReadVariableOp?.sequential_2/conv2d_5/BiasAdd_1/ReadVariableOp?+sequential_2/conv2d_5/Conv2D/ReadVariableOp?-sequential_2/conv2d_5/Conv2D_1/ReadVariableOp?+sequential_2/dense_4/BiasAdd/ReadVariableOp?-sequential_2/dense_4/BiasAdd_1/ReadVariableOp?*sequential_2/dense_4/MatMul/ReadVariableOp?,sequential_2/dense_4/MatMul_1/ReadVariableOp?+sequential_2/dense_5/BiasAdd/ReadVariableOp?-sequential_2/dense_5/BiasAdd_1/ReadVariableOp?*sequential_2/dense_5/MatMul/ReadVariableOp?,sequential_2/dense_5/MatMul_1/ReadVariableOp?
+sequential_2/conv2d_4/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02-
+sequential_2/conv2d_4/Conv2D/ReadVariableOp?
sequential_2/conv2d_4/Conv2DConv2Dinputs_03sequential_2/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????6,*
data_formatNCHW*
paddingVALID*
strides
2
sequential_2/conv2d_4/Conv2D?
,sequential_2/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_2/conv2d_4/BiasAdd/ReadVariableOp?
sequential_2/conv2d_4/BiasAddBiasAdd%sequential_2/conv2d_4/Conv2D:output:04sequential_2/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????6,*
data_formatNCHW2
sequential_2/conv2d_4/BiasAdd?
sequential_2/activation_4/ReluRelu&sequential_2/conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????6,2 
sequential_2/activation_4/Relu?
$sequential_2/max_pooling2d_4/MaxPoolMaxPool,sequential_2/activation_4/Relu:activations:0*/
_output_shapes
:?????????,*
ksize
*
paddingVALID*
strides
2&
$sequential_2/max_pooling2d_4/MaxPool?
sequential_2/dropout_6/IdentityIdentity-sequential_2/max_pooling2d_4/MaxPool:output:0*
T0*/
_output_shapes
:?????????,2!
sequential_2/dropout_6/Identity?
+sequential_2/conv2d_5/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02-
+sequential_2/conv2d_5/Conv2D/ReadVariableOp?
sequential_2/conv2d_5/Conv2DConv2D(sequential_2/dropout_6/Identity:output:03sequential_2/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**
data_formatNCHW*
paddingVALID*
strides
2
sequential_2/conv2d_5/Conv2D?
,sequential_2/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_2/conv2d_5/BiasAdd/ReadVariableOp?
sequential_2/conv2d_5/BiasAddBiasAdd%sequential_2/conv2d_5/Conv2D:output:04sequential_2/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**
data_formatNCHW2
sequential_2/conv2d_5/BiasAdd?
sequential_2/activation_5/ReluRelu&sequential_2/conv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:?????????*2 
sequential_2/activation_5/Relu?
$sequential_2/max_pooling2d_5/MaxPoolMaxPool,sequential_2/activation_5/Relu:activations:0*/
_output_shapes
:?????????**
ksize
*
paddingVALID*
strides
2&
$sequential_2/max_pooling2d_5/MaxPool?
sequential_2/dropout_7/IdentityIdentity-sequential_2/max_pooling2d_5/MaxPool:output:0*
T0*/
_output_shapes
:?????????*2!
sequential_2/dropout_7/Identity?
sequential_2/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
sequential_2/flatten_2/Const?
sequential_2/flatten_2/ReshapeReshape(sequential_2/dropout_7/Identity:output:0%sequential_2/flatten_2/Const:output:0*
T0*(
_output_shapes
:??????????2 
sequential_2/flatten_2/Reshape?
*sequential_2/dense_4/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*sequential_2/dense_4/MatMul/ReadVariableOp?
sequential_2/dense_4/MatMulMatMul'sequential_2/flatten_2/Reshape:output:02sequential_2/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_2/dense_4/MatMul?
+sequential_2/dense_4/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential_2/dense_4/BiasAdd/ReadVariableOp?
sequential_2/dense_4/BiasAddBiasAdd%sequential_2/dense_4/MatMul:product:03sequential_2/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_2/dense_4/BiasAdd?
sequential_2/dense_4/ReluRelu%sequential_2/dense_4/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_2/dense_4/Relu?
sequential_2/dropout_8/IdentityIdentity'sequential_2/dense_4/Relu:activations:0*
T0*(
_output_shapes
:??????????2!
sequential_2/dropout_8/Identity?
*sequential_2/dense_5/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_5_matmul_readvariableop_resource*
_output_shapes
:	?2*
dtype02,
*sequential_2/dense_5/MatMul/ReadVariableOp?
sequential_2/dense_5/MatMulMatMul(sequential_2/dropout_8/Identity:output:02sequential_2/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
sequential_2/dense_5/MatMul?
+sequential_2/dense_5/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_5_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02-
+sequential_2/dense_5/BiasAdd/ReadVariableOp?
sequential_2/dense_5/BiasAddBiasAdd%sequential_2/dense_5/MatMul:product:03sequential_2/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
sequential_2/dense_5/BiasAdd?
sequential_2/dense_5/ReluRelu%sequential_2/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
sequential_2/dense_5/Relu?
-sequential_2/conv2d_4/Conv2D_1/ReadVariableOpReadVariableOp4sequential_2_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02/
-sequential_2/conv2d_4/Conv2D_1/ReadVariableOp?
sequential_2/conv2d_4/Conv2D_1Conv2Dinputs_15sequential_2/conv2d_4/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????6,*
data_formatNCHW*
paddingVALID*
strides
2 
sequential_2/conv2d_4/Conv2D_1?
.sequential_2/conv2d_4/BiasAdd_1/ReadVariableOpReadVariableOp5sequential_2_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.sequential_2/conv2d_4/BiasAdd_1/ReadVariableOp?
sequential_2/conv2d_4/BiasAdd_1BiasAdd'sequential_2/conv2d_4/Conv2D_1:output:06sequential_2/conv2d_4/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????6,*
data_formatNCHW2!
sequential_2/conv2d_4/BiasAdd_1?
 sequential_2/activation_4/Relu_1Relu(sequential_2/conv2d_4/BiasAdd_1:output:0*
T0*/
_output_shapes
:?????????6,2"
 sequential_2/activation_4/Relu_1?
&sequential_2/max_pooling2d_4/MaxPool_1MaxPool.sequential_2/activation_4/Relu_1:activations:0*/
_output_shapes
:?????????,*
ksize
*
paddingVALID*
strides
2(
&sequential_2/max_pooling2d_4/MaxPool_1?
!sequential_2/dropout_6/Identity_1Identity/sequential_2/max_pooling2d_4/MaxPool_1:output:0*
T0*/
_output_shapes
:?????????,2#
!sequential_2/dropout_6/Identity_1?
-sequential_2/conv2d_5/Conv2D_1/ReadVariableOpReadVariableOp4sequential_2_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02/
-sequential_2/conv2d_5/Conv2D_1/ReadVariableOp?
sequential_2/conv2d_5/Conv2D_1Conv2D*sequential_2/dropout_6/Identity_1:output:05sequential_2/conv2d_5/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**
data_formatNCHW*
paddingVALID*
strides
2 
sequential_2/conv2d_5/Conv2D_1?
.sequential_2/conv2d_5/BiasAdd_1/ReadVariableOpReadVariableOp5sequential_2_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.sequential_2/conv2d_5/BiasAdd_1/ReadVariableOp?
sequential_2/conv2d_5/BiasAdd_1BiasAdd'sequential_2/conv2d_5/Conv2D_1:output:06sequential_2/conv2d_5/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**
data_formatNCHW2!
sequential_2/conv2d_5/BiasAdd_1?
 sequential_2/activation_5/Relu_1Relu(sequential_2/conv2d_5/BiasAdd_1:output:0*
T0*/
_output_shapes
:?????????*2"
 sequential_2/activation_5/Relu_1?
&sequential_2/max_pooling2d_5/MaxPool_1MaxPool.sequential_2/activation_5/Relu_1:activations:0*/
_output_shapes
:?????????**
ksize
*
paddingVALID*
strides
2(
&sequential_2/max_pooling2d_5/MaxPool_1?
!sequential_2/dropout_7/Identity_1Identity/sequential_2/max_pooling2d_5/MaxPool_1:output:0*
T0*/
_output_shapes
:?????????*2#
!sequential_2/dropout_7/Identity_1?
sequential_2/flatten_2/Const_1Const*
_output_shapes
:*
dtype0*
valueB"?????  2 
sequential_2/flatten_2/Const_1?
 sequential_2/flatten_2/Reshape_1Reshape*sequential_2/dropout_7/Identity_1:output:0'sequential_2/flatten_2/Const_1:output:0*
T0*(
_output_shapes
:??????????2"
 sequential_2/flatten_2/Reshape_1?
,sequential_2/dense_4/MatMul_1/ReadVariableOpReadVariableOp3sequential_2_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02.
,sequential_2/dense_4/MatMul_1/ReadVariableOp?
sequential_2/dense_4/MatMul_1MatMul)sequential_2/flatten_2/Reshape_1:output:04sequential_2/dense_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_2/dense_4/MatMul_1?
-sequential_2/dense_4/BiasAdd_1/ReadVariableOpReadVariableOp4sequential_2_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-sequential_2/dense_4/BiasAdd_1/ReadVariableOp?
sequential_2/dense_4/BiasAdd_1BiasAdd'sequential_2/dense_4/MatMul_1:product:05sequential_2/dense_4/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
sequential_2/dense_4/BiasAdd_1?
sequential_2/dense_4/Relu_1Relu'sequential_2/dense_4/BiasAdd_1:output:0*
T0*(
_output_shapes
:??????????2
sequential_2/dense_4/Relu_1?
!sequential_2/dropout_8/Identity_1Identity)sequential_2/dense_4/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2#
!sequential_2/dropout_8/Identity_1?
,sequential_2/dense_5/MatMul_1/ReadVariableOpReadVariableOp3sequential_2_dense_5_matmul_readvariableop_resource*
_output_shapes
:	?2*
dtype02.
,sequential_2/dense_5/MatMul_1/ReadVariableOp?
sequential_2/dense_5/MatMul_1MatMul*sequential_2/dropout_8/Identity_1:output:04sequential_2/dense_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
sequential_2/dense_5/MatMul_1?
-sequential_2/dense_5/BiasAdd_1/ReadVariableOpReadVariableOp4sequential_2_dense_5_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02/
-sequential_2/dense_5/BiasAdd_1/ReadVariableOp?
sequential_2/dense_5/BiasAdd_1BiasAdd'sequential_2/dense_5/MatMul_1:product:05sequential_2/dense_5/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22 
sequential_2/dense_5/BiasAdd_1?
sequential_2/dense_5/Relu_1Relu'sequential_2/dense_5/BiasAdd_1:output:0*
T0*'
_output_shapes
:?????????22
sequential_2/dense_5/Relu_1?
lambda_3/subSub'sequential_2/dense_5/Relu:activations:0)sequential_2/dense_5/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
lambda_3/subp
lambda_3/SquareSquarelambda_3/sub:z:0*
T0*'
_output_shapes
:?????????22
lambda_3/Square?
lambda_3/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2 
lambda_3/Sum/reduction_indices?
lambda_3/SumSumlambda_3/Square:y:0'lambda_3/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2
lambda_3/Sume
lambda_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lambda_3/Const?
lambda_3/MaximumMaximumlambda_3/Sum:output:0lambda_3/Const:output:0*
T0*'
_output_shapes
:?????????2
lambda_3/Maximumn
lambda_3/SqrtSqrtlambda_3/Maximum:z:0*
T0*'
_output_shapes
:?????????2
lambda_3/Sqrt?
IdentityIdentitylambda_3/Sqrt:y:0-^sequential_2/conv2d_4/BiasAdd/ReadVariableOp/^sequential_2/conv2d_4/BiasAdd_1/ReadVariableOp,^sequential_2/conv2d_4/Conv2D/ReadVariableOp.^sequential_2/conv2d_4/Conv2D_1/ReadVariableOp-^sequential_2/conv2d_5/BiasAdd/ReadVariableOp/^sequential_2/conv2d_5/BiasAdd_1/ReadVariableOp,^sequential_2/conv2d_5/Conv2D/ReadVariableOp.^sequential_2/conv2d_5/Conv2D_1/ReadVariableOp,^sequential_2/dense_4/BiasAdd/ReadVariableOp.^sequential_2/dense_4/BiasAdd_1/ReadVariableOp+^sequential_2/dense_4/MatMul/ReadVariableOp-^sequential_2/dense_4/MatMul_1/ReadVariableOp,^sequential_2/dense_5/BiasAdd/ReadVariableOp.^sequential_2/dense_5/BiasAdd_1/ReadVariableOp+^sequential_2/dense_5/MatMul/ReadVariableOp-^sequential_2/dense_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:?????????8.:?????????8.: : : : : : : : 2\
,sequential_2/conv2d_4/BiasAdd/ReadVariableOp,sequential_2/conv2d_4/BiasAdd/ReadVariableOp2`
.sequential_2/conv2d_4/BiasAdd_1/ReadVariableOp.sequential_2/conv2d_4/BiasAdd_1/ReadVariableOp2Z
+sequential_2/conv2d_4/Conv2D/ReadVariableOp+sequential_2/conv2d_4/Conv2D/ReadVariableOp2^
-sequential_2/conv2d_4/Conv2D_1/ReadVariableOp-sequential_2/conv2d_4/Conv2D_1/ReadVariableOp2\
,sequential_2/conv2d_5/BiasAdd/ReadVariableOp,sequential_2/conv2d_5/BiasAdd/ReadVariableOp2`
.sequential_2/conv2d_5/BiasAdd_1/ReadVariableOp.sequential_2/conv2d_5/BiasAdd_1/ReadVariableOp2Z
+sequential_2/conv2d_5/Conv2D/ReadVariableOp+sequential_2/conv2d_5/Conv2D/ReadVariableOp2^
-sequential_2/conv2d_5/Conv2D_1/ReadVariableOp-sequential_2/conv2d_5/Conv2D_1/ReadVariableOp2Z
+sequential_2/dense_4/BiasAdd/ReadVariableOp+sequential_2/dense_4/BiasAdd/ReadVariableOp2^
-sequential_2/dense_4/BiasAdd_1/ReadVariableOp-sequential_2/dense_4/BiasAdd_1/ReadVariableOp2X
*sequential_2/dense_4/MatMul/ReadVariableOp*sequential_2/dense_4/MatMul/ReadVariableOp2\
,sequential_2/dense_4/MatMul_1/ReadVariableOp,sequential_2/dense_4/MatMul_1/ReadVariableOp2Z
+sequential_2/dense_5/BiasAdd/ReadVariableOp+sequential_2/dense_5/BiasAdd/ReadVariableOp2^
-sequential_2/dense_5/BiasAdd_1/ReadVariableOp-sequential_2/dense_5/BiasAdd_1/ReadVariableOp2X
*sequential_2/dense_5/MatMul/ReadVariableOp*sequential_2/dense_5/MatMul/ReadVariableOp2\
,sequential_2/dense_5/MatMul_1/ReadVariableOp,sequential_2/dense_5/MatMul_1/ReadVariableOp:Y U
/
_output_shapes
:?????????8.
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????8.
"
_user_specified_name
inputs/1
?
b
D__inference_dropout_6_layer_call_and_return_conditional_losses_16022

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????,2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????,2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????,:W S
/
_output_shapes
:?????????,
 
_user_specified_nameinputs
?
H
,__inference_activation_4_layer_call_fn_16002

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
:?????????6,* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_148852
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????6,2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????6,:W S
/
_output_shapes
:?????????6,
 
_user_specified_nameinputs
?
b
D__inference_dropout_7_layer_call_and_return_conditional_losses_14924

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????*2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????*2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????*:W S
/
_output_shapes
:?????????*
 
_user_specified_nameinputs
?
c
G__inference_activation_4_layer_call_and_return_conditional_losses_14885

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????6,2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????6,2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????6,:W S
/
_output_shapes
:?????????6,
 
_user_specified_nameinputs
?
?
B__inference_model_3_layer_call_and_return_conditional_losses_15505
input_5
input_6,
sequential_2_15477: 
sequential_2_15479:,
sequential_2_15481: 
sequential_2_15483:&
sequential_2_15485:
??!
sequential_2_15487:	?%
sequential_2_15489:	?2 
sequential_2_15491:2
identity??$sequential_2/StatefulPartitionedCall?&sequential_2/StatefulPartitionedCall_1?
$sequential_2/StatefulPartitionedCallStatefulPartitionedCallinput_5sequential_2_15477sequential_2_15479sequential_2_15481sequential_2_15483sequential_2_15485sequential_2_15487sequential_2_15489sequential_2_15491*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_149762&
$sequential_2/StatefulPartitionedCall?
&sequential_2/StatefulPartitionedCall_1StatefulPartitionedCallinput_6sequential_2_15477sequential_2_15479sequential_2_15481sequential_2_15483sequential_2_15485sequential_2_15487sequential_2_15489sequential_2_15491*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_149762(
&sequential_2/StatefulPartitionedCall_1?
lambda_3/PartitionedCallPartitionedCall-sequential_2/StatefulPartitionedCall:output:0/sequential_2/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_lambda_3_layer_call_and_return_conditional_losses_153282
lambda_3/PartitionedCall?
IdentityIdentity!lambda_3/PartitionedCall:output:0%^sequential_2/StatefulPartitionedCall'^sequential_2/StatefulPartitionedCall_1*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:?????????8.:?????????8.: : : : : : : : 2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall2P
&sequential_2/StatefulPartitionedCall_1&sequential_2/StatefulPartitionedCall_1:X T
/
_output_shapes
:?????????8.
!
_user_specified_name	input_5:XT
/
_output_shapes
:?????????8.
!
_user_specified_name	input_6
?
c
D__inference_dropout_6_layer_call_and_return_conditional_losses_16034

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
:?????????,2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????,*
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
:?????????,2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????,2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????,2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????,2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????,:W S
/
_output_shapes
:?????????,
 
_user_specified_nameinputs
?
b
D__inference_dropout_7_layer_call_and_return_conditional_losses_16078

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????*2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????*2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????*:W S
/
_output_shapes
:?????????*
 
_user_specified_nameinputs
?	
o
C__inference_lambda_3_layer_call_and_return_conditional_losses_15978
inputs_0
inputs_1
identityW
subSubinputs_0inputs_1*
T0*'
_output_shapes
:?????????22
subU
SquareSquaresub:z:0*
T0*'
_output_shapes
:?????????22
Squarep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices?
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2
SumS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Constm
MaximumMaximumSum:output:0Const:output:0*
T0*'
_output_shapes
:?????????2	
MaximumS
SqrtSqrtMaximum:z:0*
T0*'
_output_shapes
:?????????2
Sqrt\
IdentityIdentitySqrt:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????2:?????????2:Q M
'
_output_shapes
:?????????2
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????2
"
_user_specified_name
inputs/1
?0
?
G__inference_sequential_2_layer_call_and_return_conditional_losses_14976

inputs(
conv2d_4_14875:
conv2d_4_14877:(
conv2d_5_14906:
conv2d_5_14908:!
dense_4_14946:
??
dense_4_14948:	? 
dense_5_14970:	?2
dense_5_14972:2
identity?? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_4_14875conv2d_4_14877*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????6,*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_148742"
 conv2d_4/StatefulPartitionedCall?
activation_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????6,* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_148852
activation_4/PartitionedCall?
max_pooling2d_4/PartitionedCallPartitionedCall%activation_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????,* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_148392!
max_pooling2d_4/PartitionedCall?
dropout_6/PartitionedCallPartitionedCall(max_pooling2d_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????,* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_148932
dropout_6/PartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall"dropout_6/PartitionedCall:output:0conv2d_5_14906conv2d_5_14908*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????**$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_149052"
 conv2d_5/StatefulPartitionedCall?
activation_5/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????** 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_activation_5_layer_call_and_return_conditional_losses_149162
activation_5/PartitionedCall?
max_pooling2d_5/PartitionedCallPartitionedCall%activation_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????** 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_148512!
max_pooling2d_5/PartitionedCall?
dropout_7/PartitionedCallPartitionedCall(max_pooling2d_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????** 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_149242
dropout_7/PartitionedCall?
flatten_2/PartitionedCallPartitionedCall"dropout_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_149322
flatten_2/PartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_4_14946dense_4_14948*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_149452!
dense_4/StatefulPartitionedCall?
dropout_8/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_149562
dropout_8/PartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall"dropout_8/PartitionedCall:output:0dense_5_14970dense_5_14972*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_149692!
dense_5/StatefulPartitionedCall?
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????8.: : : : : : : : 2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:W S
/
_output_shapes
:?????????8.
 
_user_specified_nameinputs
?
`
D__inference_flatten_2_layer_call_and_return_conditional_losses_16101

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????*:W S
/
_output_shapes
:?????????*
 
_user_specified_nameinputs
?4
?
G__inference_sequential_2_layer_call_and_return_conditional_losses_15177

inputs(
conv2d_4_15148:
conv2d_4_15150:(
conv2d_5_15156:
conv2d_5_15158:!
dense_4_15165:
??
dense_4_15167:	? 
dense_5_15171:	?2
dense_5_15173:2
identity?? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?!dropout_6/StatefulPartitionedCall?!dropout_7/StatefulPartitionedCall?!dropout_8/StatefulPartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_4_15148conv2d_4_15150*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????6,*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_148742"
 conv2d_4/StatefulPartitionedCall?
activation_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????6,* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_148852
activation_4/PartitionedCall?
max_pooling2d_4/PartitionedCallPartitionedCall%activation_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????,* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_148392!
max_pooling2d_4/PartitionedCall?
!dropout_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????,* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_151032#
!dropout_6/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall*dropout_6/StatefulPartitionedCall:output:0conv2d_5_15156conv2d_5_15158*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????**$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_149052"
 conv2d_5/StatefulPartitionedCall?
activation_5/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????** 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_activation_5_layer_call_and_return_conditional_losses_149162
activation_5/PartitionedCall?
max_pooling2d_5/PartitionedCallPartitionedCall%activation_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????** 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_148512!
max_pooling2d_5/PartitionedCall?
!dropout_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_5/PartitionedCall:output:0"^dropout_6/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????** 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_150642#
!dropout_7/StatefulPartitionedCall?
flatten_2/PartitionedCallPartitionedCall*dropout_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_149322
flatten_2/PartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_4_15165dense_4_15167*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_149452!
dense_4/StatefulPartitionedCall?
!dropout_8/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0"^dropout_7/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_150252#
!dropout_8/StatefulPartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall*dropout_8/StatefulPartitionedCall:output:0dense_5_15171dense_5_15173*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_149692!
dense_5/StatefulPartitionedCall?
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall"^dropout_6/StatefulPartitionedCall"^dropout_7/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????8.: : : : : : : : 2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2F
!dropout_6/StatefulPartitionedCall!dropout_6/StatefulPartitionedCall2F
!dropout_7/StatefulPartitionedCall!dropout_7/StatefulPartitionedCall2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall:W S
/
_output_shapes
:?????????8.
 
_user_specified_nameinputs
?

?
C__inference_conv2d_4_layer_call_and_return_conditional_losses_15997

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????6,*
data_formatNCHW*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????6,*
data_formatNCHW2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????6,2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????8.: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????8.
 
_user_specified_nameinputs
?
c
G__inference_activation_5_layer_call_and_return_conditional_losses_16063

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????*2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????*2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????*:W S
/
_output_shapes
:?????????*
 
_user_specified_nameinputs
?
c
D__inference_dropout_7_layer_call_and_return_conditional_losses_15064

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
:?????????*2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????**
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
:?????????*2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????*2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????*2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????*2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????*:W S
/
_output_shapes
:?????????*
 
_user_specified_nameinputs
?	
?
,__inference_sequential_2_layer_call_fn_15217
conv2d_4_input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?2
	unknown_6:2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_4_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_151772
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????8.: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:?????????8.
(
_user_specified_nameconv2d_4_input
?
c
G__inference_activation_5_layer_call_and_return_conditional_losses_14916

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????*2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????*2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????*:W S
/
_output_shapes
:?????????*
 
_user_specified_nameinputs
?

?
'__inference_model_3_layer_call_fn_15589
inputs_0
inputs_1!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?2
	unknown_6:2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_model_3_layer_call_and_return_conditional_losses_153312
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:?????????8.:?????????8.: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:?????????8.
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????8.
"
_user_specified_name
inputs/1
?O
?
G__inference_sequential_2_layer_call_and_return_conditional_losses_15942

inputsA
'conv2d_4_conv2d_readvariableop_resource:6
(conv2d_4_biasadd_readvariableop_resource:A
'conv2d_5_conv2d_readvariableop_resource:6
(conv2d_5_biasadd_readvariableop_resource::
&dense_4_matmul_readvariableop_resource:
??6
'dense_4_biasadd_readvariableop_resource:	?9
&dense_5_matmul_readvariableop_resource:	?25
'dense_5_biasadd_readvariableop_resource:2
identity??conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?conv2d_5/BiasAdd/ReadVariableOp?conv2d_5/Conv2D/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_4/Conv2D/ReadVariableOp?
conv2d_4/Conv2DConv2Dinputs&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????6,*
data_formatNCHW*
paddingVALID*
strides
2
conv2d_4/Conv2D?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????6,*
data_formatNCHW2
conv2d_4/BiasAdd?
activation_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????6,2
activation_4/Relu?
max_pooling2d_4/MaxPoolMaxPoolactivation_4/Relu:activations:0*/
_output_shapes
:?????????,*
ksize
*
paddingVALID*
strides
2
max_pooling2d_4/MaxPoolw
dropout_6/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_6/dropout/Const?
dropout_6/dropout/MulMul max_pooling2d_4/MaxPool:output:0 dropout_6/dropout/Const:output:0*
T0*/
_output_shapes
:?????????,2
dropout_6/dropout/Mul?
dropout_6/dropout/ShapeShape max_pooling2d_4/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_6/dropout/Shape?
.dropout_6/dropout/random_uniform/RandomUniformRandomUniform dropout_6/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????,*
dtype020
.dropout_6/dropout/random_uniform/RandomUniform?
 dropout_6/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2"
 dropout_6/dropout/GreaterEqual/y?
dropout_6/dropout/GreaterEqualGreaterEqual7dropout_6/dropout/random_uniform/RandomUniform:output:0)dropout_6/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????,2 
dropout_6/dropout/GreaterEqual?
dropout_6/dropout/CastCast"dropout_6/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????,2
dropout_6/dropout/Cast?
dropout_6/dropout/Mul_1Muldropout_6/dropout/Mul:z:0dropout_6/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????,2
dropout_6/dropout/Mul_1?
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_5/Conv2D/ReadVariableOp?
conv2d_5/Conv2DConv2Ddropout_6/dropout/Mul_1:z:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**
data_formatNCHW*
paddingVALID*
strides
2
conv2d_5/Conv2D?
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp?
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**
data_formatNCHW2
conv2d_5/BiasAdd?
activation_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:?????????*2
activation_5/Relu?
max_pooling2d_5/MaxPoolMaxPoolactivation_5/Relu:activations:0*/
_output_shapes
:?????????**
ksize
*
paddingVALID*
strides
2
max_pooling2d_5/MaxPoolw
dropout_7/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_7/dropout/Const?
dropout_7/dropout/MulMul max_pooling2d_5/MaxPool:output:0 dropout_7/dropout/Const:output:0*
T0*/
_output_shapes
:?????????*2
dropout_7/dropout/Mul?
dropout_7/dropout/ShapeShape max_pooling2d_5/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_7/dropout/Shape?
.dropout_7/dropout/random_uniform/RandomUniformRandomUniform dropout_7/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????**
dtype020
.dropout_7/dropout/random_uniform/RandomUniform?
 dropout_7/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2"
 dropout_7/dropout/GreaterEqual/y?
dropout_7/dropout/GreaterEqualGreaterEqual7dropout_7/dropout/random_uniform/RandomUniform:output:0)dropout_7/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????*2 
dropout_7/dropout/GreaterEqual?
dropout_7/dropout/CastCast"dropout_7/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????*2
dropout_7/dropout/Cast?
dropout_7/dropout/Mul_1Muldropout_7/dropout/Mul:z:0dropout_7/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????*2
dropout_7/dropout/Mul_1s
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_2/Const?
flatten_2/ReshapeReshapedropout_7/dropout/Mul_1:z:0flatten_2/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_2/Reshape?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMulflatten_2/Reshape:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_4/BiasAddq
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_4/Reluw
dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_8/dropout/Const?
dropout_8/dropout/MulMuldense_4/Relu:activations:0 dropout_8/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_8/dropout/Mul|
dropout_8/dropout/ShapeShapedense_4/Relu:activations:0*
T0*
_output_shapes
:2
dropout_8/dropout/Shape?
.dropout_8/dropout/random_uniform/RandomUniformRandomUniform dropout_8/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype020
.dropout_8/dropout/random_uniform/RandomUniform?
 dropout_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2"
 dropout_8/dropout/GreaterEqual/y?
dropout_8/dropout/GreaterEqualGreaterEqual7dropout_8/dropout/random_uniform/RandomUniform:output:0)dropout_8/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2 
dropout_8/dropout/GreaterEqual?
dropout_8/dropout/CastCast"dropout_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_8/dropout/Cast?
dropout_8/dropout/Mul_1Muldropout_8/dropout/Mul:z:0dropout_8/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_8/dropout/Mul_1?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	?2*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMuldropout_8/dropout/Mul_1:z:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
dense_5/BiasAddp
dense_5/ReluReludense_5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
dense_5/Relu?
IdentityIdentitydense_5/Relu:activations:0 ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????8.: : : : : : : : 2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????8.
 
_user_specified_nameinputs
?
?
'__inference_dense_4_layer_call_fn_16110

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_149452
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
D__inference_dropout_6_layer_call_and_return_conditional_losses_15103

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
:?????????,2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????,*
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
:?????????,2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????,2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????,2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????,2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????,:W S
/
_output_shapes
:?????????,
 
_user_specified_nameinputs
?

?
B__inference_dense_4_layer_call_and_return_conditional_losses_16121

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
D__inference_dropout_8_layer_call_and_return_conditional_losses_16148

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
 __inference__wrapped_model_14833
input_5
input_6V
<model_3_sequential_2_conv2d_4_conv2d_readvariableop_resource:K
=model_3_sequential_2_conv2d_4_biasadd_readvariableop_resource:V
<model_3_sequential_2_conv2d_5_conv2d_readvariableop_resource:K
=model_3_sequential_2_conv2d_5_biasadd_readvariableop_resource:O
;model_3_sequential_2_dense_4_matmul_readvariableop_resource:
??K
<model_3_sequential_2_dense_4_biasadd_readvariableop_resource:	?N
;model_3_sequential_2_dense_5_matmul_readvariableop_resource:	?2J
<model_3_sequential_2_dense_5_biasadd_readvariableop_resource:2
identity??4model_3/sequential_2/conv2d_4/BiasAdd/ReadVariableOp?6model_3/sequential_2/conv2d_4/BiasAdd_1/ReadVariableOp?3model_3/sequential_2/conv2d_4/Conv2D/ReadVariableOp?5model_3/sequential_2/conv2d_4/Conv2D_1/ReadVariableOp?4model_3/sequential_2/conv2d_5/BiasAdd/ReadVariableOp?6model_3/sequential_2/conv2d_5/BiasAdd_1/ReadVariableOp?3model_3/sequential_2/conv2d_5/Conv2D/ReadVariableOp?5model_3/sequential_2/conv2d_5/Conv2D_1/ReadVariableOp?3model_3/sequential_2/dense_4/BiasAdd/ReadVariableOp?5model_3/sequential_2/dense_4/BiasAdd_1/ReadVariableOp?2model_3/sequential_2/dense_4/MatMul/ReadVariableOp?4model_3/sequential_2/dense_4/MatMul_1/ReadVariableOp?3model_3/sequential_2/dense_5/BiasAdd/ReadVariableOp?5model_3/sequential_2/dense_5/BiasAdd_1/ReadVariableOp?2model_3/sequential_2/dense_5/MatMul/ReadVariableOp?4model_3/sequential_2/dense_5/MatMul_1/ReadVariableOp?
3model_3/sequential_2/conv2d_4/Conv2D/ReadVariableOpReadVariableOp<model_3_sequential_2_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype025
3model_3/sequential_2/conv2d_4/Conv2D/ReadVariableOp?
$model_3/sequential_2/conv2d_4/Conv2DConv2Dinput_5;model_3/sequential_2/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????6,*
data_formatNCHW*
paddingVALID*
strides
2&
$model_3/sequential_2/conv2d_4/Conv2D?
4model_3/sequential_2/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp=model_3_sequential_2_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype026
4model_3/sequential_2/conv2d_4/BiasAdd/ReadVariableOp?
%model_3/sequential_2/conv2d_4/BiasAddBiasAdd-model_3/sequential_2/conv2d_4/Conv2D:output:0<model_3/sequential_2/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????6,*
data_formatNCHW2'
%model_3/sequential_2/conv2d_4/BiasAdd?
&model_3/sequential_2/activation_4/ReluRelu.model_3/sequential_2/conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????6,2(
&model_3/sequential_2/activation_4/Relu?
,model_3/sequential_2/max_pooling2d_4/MaxPoolMaxPool4model_3/sequential_2/activation_4/Relu:activations:0*/
_output_shapes
:?????????,*
ksize
*
paddingVALID*
strides
2.
,model_3/sequential_2/max_pooling2d_4/MaxPool?
'model_3/sequential_2/dropout_6/IdentityIdentity5model_3/sequential_2/max_pooling2d_4/MaxPool:output:0*
T0*/
_output_shapes
:?????????,2)
'model_3/sequential_2/dropout_6/Identity?
3model_3/sequential_2/conv2d_5/Conv2D/ReadVariableOpReadVariableOp<model_3_sequential_2_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype025
3model_3/sequential_2/conv2d_5/Conv2D/ReadVariableOp?
$model_3/sequential_2/conv2d_5/Conv2DConv2D0model_3/sequential_2/dropout_6/Identity:output:0;model_3/sequential_2/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**
data_formatNCHW*
paddingVALID*
strides
2&
$model_3/sequential_2/conv2d_5/Conv2D?
4model_3/sequential_2/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp=model_3_sequential_2_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype026
4model_3/sequential_2/conv2d_5/BiasAdd/ReadVariableOp?
%model_3/sequential_2/conv2d_5/BiasAddBiasAdd-model_3/sequential_2/conv2d_5/Conv2D:output:0<model_3/sequential_2/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**
data_formatNCHW2'
%model_3/sequential_2/conv2d_5/BiasAdd?
&model_3/sequential_2/activation_5/ReluRelu.model_3/sequential_2/conv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:?????????*2(
&model_3/sequential_2/activation_5/Relu?
,model_3/sequential_2/max_pooling2d_5/MaxPoolMaxPool4model_3/sequential_2/activation_5/Relu:activations:0*/
_output_shapes
:?????????**
ksize
*
paddingVALID*
strides
2.
,model_3/sequential_2/max_pooling2d_5/MaxPool?
'model_3/sequential_2/dropout_7/IdentityIdentity5model_3/sequential_2/max_pooling2d_5/MaxPool:output:0*
T0*/
_output_shapes
:?????????*2)
'model_3/sequential_2/dropout_7/Identity?
$model_3/sequential_2/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2&
$model_3/sequential_2/flatten_2/Const?
&model_3/sequential_2/flatten_2/ReshapeReshape0model_3/sequential_2/dropout_7/Identity:output:0-model_3/sequential_2/flatten_2/Const:output:0*
T0*(
_output_shapes
:??????????2(
&model_3/sequential_2/flatten_2/Reshape?
2model_3/sequential_2/dense_4/MatMul/ReadVariableOpReadVariableOp;model_3_sequential_2_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype024
2model_3/sequential_2/dense_4/MatMul/ReadVariableOp?
#model_3/sequential_2/dense_4/MatMulMatMul/model_3/sequential_2/flatten_2/Reshape:output:0:model_3/sequential_2/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2%
#model_3/sequential_2/dense_4/MatMul?
3model_3/sequential_2/dense_4/BiasAdd/ReadVariableOpReadVariableOp<model_3_sequential_2_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype025
3model_3/sequential_2/dense_4/BiasAdd/ReadVariableOp?
$model_3/sequential_2/dense_4/BiasAddBiasAdd-model_3/sequential_2/dense_4/MatMul:product:0;model_3/sequential_2/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2&
$model_3/sequential_2/dense_4/BiasAdd?
!model_3/sequential_2/dense_4/ReluRelu-model_3/sequential_2/dense_4/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2#
!model_3/sequential_2/dense_4/Relu?
'model_3/sequential_2/dropout_8/IdentityIdentity/model_3/sequential_2/dense_4/Relu:activations:0*
T0*(
_output_shapes
:??????????2)
'model_3/sequential_2/dropout_8/Identity?
2model_3/sequential_2/dense_5/MatMul/ReadVariableOpReadVariableOp;model_3_sequential_2_dense_5_matmul_readvariableop_resource*
_output_shapes
:	?2*
dtype024
2model_3/sequential_2/dense_5/MatMul/ReadVariableOp?
#model_3/sequential_2/dense_5/MatMulMatMul0model_3/sequential_2/dropout_8/Identity:output:0:model_3/sequential_2/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22%
#model_3/sequential_2/dense_5/MatMul?
3model_3/sequential_2/dense_5/BiasAdd/ReadVariableOpReadVariableOp<model_3_sequential_2_dense_5_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype025
3model_3/sequential_2/dense_5/BiasAdd/ReadVariableOp?
$model_3/sequential_2/dense_5/BiasAddBiasAdd-model_3/sequential_2/dense_5/MatMul:product:0;model_3/sequential_2/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22&
$model_3/sequential_2/dense_5/BiasAdd?
!model_3/sequential_2/dense_5/ReluRelu-model_3/sequential_2/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22#
!model_3/sequential_2/dense_5/Relu?
5model_3/sequential_2/conv2d_4/Conv2D_1/ReadVariableOpReadVariableOp<model_3_sequential_2_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype027
5model_3/sequential_2/conv2d_4/Conv2D_1/ReadVariableOp?
&model_3/sequential_2/conv2d_4/Conv2D_1Conv2Dinput_6=model_3/sequential_2/conv2d_4/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????6,*
data_formatNCHW*
paddingVALID*
strides
2(
&model_3/sequential_2/conv2d_4/Conv2D_1?
6model_3/sequential_2/conv2d_4/BiasAdd_1/ReadVariableOpReadVariableOp=model_3_sequential_2_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
6model_3/sequential_2/conv2d_4/BiasAdd_1/ReadVariableOp?
'model_3/sequential_2/conv2d_4/BiasAdd_1BiasAdd/model_3/sequential_2/conv2d_4/Conv2D_1:output:0>model_3/sequential_2/conv2d_4/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????6,*
data_formatNCHW2)
'model_3/sequential_2/conv2d_4/BiasAdd_1?
(model_3/sequential_2/activation_4/Relu_1Relu0model_3/sequential_2/conv2d_4/BiasAdd_1:output:0*
T0*/
_output_shapes
:?????????6,2*
(model_3/sequential_2/activation_4/Relu_1?
.model_3/sequential_2/max_pooling2d_4/MaxPool_1MaxPool6model_3/sequential_2/activation_4/Relu_1:activations:0*/
_output_shapes
:?????????,*
ksize
*
paddingVALID*
strides
20
.model_3/sequential_2/max_pooling2d_4/MaxPool_1?
)model_3/sequential_2/dropout_6/Identity_1Identity7model_3/sequential_2/max_pooling2d_4/MaxPool_1:output:0*
T0*/
_output_shapes
:?????????,2+
)model_3/sequential_2/dropout_6/Identity_1?
5model_3/sequential_2/conv2d_5/Conv2D_1/ReadVariableOpReadVariableOp<model_3_sequential_2_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype027
5model_3/sequential_2/conv2d_5/Conv2D_1/ReadVariableOp?
&model_3/sequential_2/conv2d_5/Conv2D_1Conv2D2model_3/sequential_2/dropout_6/Identity_1:output:0=model_3/sequential_2/conv2d_5/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**
data_formatNCHW*
paddingVALID*
strides
2(
&model_3/sequential_2/conv2d_5/Conv2D_1?
6model_3/sequential_2/conv2d_5/BiasAdd_1/ReadVariableOpReadVariableOp=model_3_sequential_2_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
6model_3/sequential_2/conv2d_5/BiasAdd_1/ReadVariableOp?
'model_3/sequential_2/conv2d_5/BiasAdd_1BiasAdd/model_3/sequential_2/conv2d_5/Conv2D_1:output:0>model_3/sequential_2/conv2d_5/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**
data_formatNCHW2)
'model_3/sequential_2/conv2d_5/BiasAdd_1?
(model_3/sequential_2/activation_5/Relu_1Relu0model_3/sequential_2/conv2d_5/BiasAdd_1:output:0*
T0*/
_output_shapes
:?????????*2*
(model_3/sequential_2/activation_5/Relu_1?
.model_3/sequential_2/max_pooling2d_5/MaxPool_1MaxPool6model_3/sequential_2/activation_5/Relu_1:activations:0*/
_output_shapes
:?????????**
ksize
*
paddingVALID*
strides
20
.model_3/sequential_2/max_pooling2d_5/MaxPool_1?
)model_3/sequential_2/dropout_7/Identity_1Identity7model_3/sequential_2/max_pooling2d_5/MaxPool_1:output:0*
T0*/
_output_shapes
:?????????*2+
)model_3/sequential_2/dropout_7/Identity_1?
&model_3/sequential_2/flatten_2/Const_1Const*
_output_shapes
:*
dtype0*
valueB"?????  2(
&model_3/sequential_2/flatten_2/Const_1?
(model_3/sequential_2/flatten_2/Reshape_1Reshape2model_3/sequential_2/dropout_7/Identity_1:output:0/model_3/sequential_2/flatten_2/Const_1:output:0*
T0*(
_output_shapes
:??????????2*
(model_3/sequential_2/flatten_2/Reshape_1?
4model_3/sequential_2/dense_4/MatMul_1/ReadVariableOpReadVariableOp;model_3_sequential_2_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype026
4model_3/sequential_2/dense_4/MatMul_1/ReadVariableOp?
%model_3/sequential_2/dense_4/MatMul_1MatMul1model_3/sequential_2/flatten_2/Reshape_1:output:0<model_3/sequential_2/dense_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2'
%model_3/sequential_2/dense_4/MatMul_1?
5model_3/sequential_2/dense_4/BiasAdd_1/ReadVariableOpReadVariableOp<model_3_sequential_2_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype027
5model_3/sequential_2/dense_4/BiasAdd_1/ReadVariableOp?
&model_3/sequential_2/dense_4/BiasAdd_1BiasAdd/model_3/sequential_2/dense_4/MatMul_1:product:0=model_3/sequential_2/dense_4/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2(
&model_3/sequential_2/dense_4/BiasAdd_1?
#model_3/sequential_2/dense_4/Relu_1Relu/model_3/sequential_2/dense_4/BiasAdd_1:output:0*
T0*(
_output_shapes
:??????????2%
#model_3/sequential_2/dense_4/Relu_1?
)model_3/sequential_2/dropout_8/Identity_1Identity1model_3/sequential_2/dense_4/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2+
)model_3/sequential_2/dropout_8/Identity_1?
4model_3/sequential_2/dense_5/MatMul_1/ReadVariableOpReadVariableOp;model_3_sequential_2_dense_5_matmul_readvariableop_resource*
_output_shapes
:	?2*
dtype026
4model_3/sequential_2/dense_5/MatMul_1/ReadVariableOp?
%model_3/sequential_2/dense_5/MatMul_1MatMul2model_3/sequential_2/dropout_8/Identity_1:output:0<model_3/sequential_2/dense_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22'
%model_3/sequential_2/dense_5/MatMul_1?
5model_3/sequential_2/dense_5/BiasAdd_1/ReadVariableOpReadVariableOp<model_3_sequential_2_dense_5_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype027
5model_3/sequential_2/dense_5/BiasAdd_1/ReadVariableOp?
&model_3/sequential_2/dense_5/BiasAdd_1BiasAdd/model_3/sequential_2/dense_5/MatMul_1:product:0=model_3/sequential_2/dense_5/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22(
&model_3/sequential_2/dense_5/BiasAdd_1?
#model_3/sequential_2/dense_5/Relu_1Relu/model_3/sequential_2/dense_5/BiasAdd_1:output:0*
T0*'
_output_shapes
:?????????22%
#model_3/sequential_2/dense_5/Relu_1?
model_3/lambda_3/subSub/model_3/sequential_2/dense_5/Relu:activations:01model_3/sequential_2/dense_5/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
model_3/lambda_3/sub?
model_3/lambda_3/SquareSquaremodel_3/lambda_3/sub:z:0*
T0*'
_output_shapes
:?????????22
model_3/lambda_3/Square?
&model_3/lambda_3/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2(
&model_3/lambda_3/Sum/reduction_indices?
model_3/lambda_3/SumSummodel_3/lambda_3/Square:y:0/model_3/lambda_3/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2
model_3/lambda_3/Sumu
model_3/lambda_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model_3/lambda_3/Const?
model_3/lambda_3/MaximumMaximummodel_3/lambda_3/Sum:output:0model_3/lambda_3/Const:output:0*
T0*'
_output_shapes
:?????????2
model_3/lambda_3/Maximum?
model_3/lambda_3/SqrtSqrtmodel_3/lambda_3/Maximum:z:0*
T0*'
_output_shapes
:?????????2
model_3/lambda_3/Sqrt?
IdentityIdentitymodel_3/lambda_3/Sqrt:y:05^model_3/sequential_2/conv2d_4/BiasAdd/ReadVariableOp7^model_3/sequential_2/conv2d_4/BiasAdd_1/ReadVariableOp4^model_3/sequential_2/conv2d_4/Conv2D/ReadVariableOp6^model_3/sequential_2/conv2d_4/Conv2D_1/ReadVariableOp5^model_3/sequential_2/conv2d_5/BiasAdd/ReadVariableOp7^model_3/sequential_2/conv2d_5/BiasAdd_1/ReadVariableOp4^model_3/sequential_2/conv2d_5/Conv2D/ReadVariableOp6^model_3/sequential_2/conv2d_5/Conv2D_1/ReadVariableOp4^model_3/sequential_2/dense_4/BiasAdd/ReadVariableOp6^model_3/sequential_2/dense_4/BiasAdd_1/ReadVariableOp3^model_3/sequential_2/dense_4/MatMul/ReadVariableOp5^model_3/sequential_2/dense_4/MatMul_1/ReadVariableOp4^model_3/sequential_2/dense_5/BiasAdd/ReadVariableOp6^model_3/sequential_2/dense_5/BiasAdd_1/ReadVariableOp3^model_3/sequential_2/dense_5/MatMul/ReadVariableOp5^model_3/sequential_2/dense_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:?????????8.:?????????8.: : : : : : : : 2l
4model_3/sequential_2/conv2d_4/BiasAdd/ReadVariableOp4model_3/sequential_2/conv2d_4/BiasAdd/ReadVariableOp2p
6model_3/sequential_2/conv2d_4/BiasAdd_1/ReadVariableOp6model_3/sequential_2/conv2d_4/BiasAdd_1/ReadVariableOp2j
3model_3/sequential_2/conv2d_4/Conv2D/ReadVariableOp3model_3/sequential_2/conv2d_4/Conv2D/ReadVariableOp2n
5model_3/sequential_2/conv2d_4/Conv2D_1/ReadVariableOp5model_3/sequential_2/conv2d_4/Conv2D_1/ReadVariableOp2l
4model_3/sequential_2/conv2d_5/BiasAdd/ReadVariableOp4model_3/sequential_2/conv2d_5/BiasAdd/ReadVariableOp2p
6model_3/sequential_2/conv2d_5/BiasAdd_1/ReadVariableOp6model_3/sequential_2/conv2d_5/BiasAdd_1/ReadVariableOp2j
3model_3/sequential_2/conv2d_5/Conv2D/ReadVariableOp3model_3/sequential_2/conv2d_5/Conv2D/ReadVariableOp2n
5model_3/sequential_2/conv2d_5/Conv2D_1/ReadVariableOp5model_3/sequential_2/conv2d_5/Conv2D_1/ReadVariableOp2j
3model_3/sequential_2/dense_4/BiasAdd/ReadVariableOp3model_3/sequential_2/dense_4/BiasAdd/ReadVariableOp2n
5model_3/sequential_2/dense_4/BiasAdd_1/ReadVariableOp5model_3/sequential_2/dense_4/BiasAdd_1/ReadVariableOp2h
2model_3/sequential_2/dense_4/MatMul/ReadVariableOp2model_3/sequential_2/dense_4/MatMul/ReadVariableOp2l
4model_3/sequential_2/dense_4/MatMul_1/ReadVariableOp4model_3/sequential_2/dense_4/MatMul_1/ReadVariableOp2j
3model_3/sequential_2/dense_5/BiasAdd/ReadVariableOp3model_3/sequential_2/dense_5/BiasAdd/ReadVariableOp2n
5model_3/sequential_2/dense_5/BiasAdd_1/ReadVariableOp5model_3/sequential_2/dense_5/BiasAdd_1/ReadVariableOp2h
2model_3/sequential_2/dense_5/MatMul/ReadVariableOp2model_3/sequential_2/dense_5/MatMul/ReadVariableOp2l
4model_3/sequential_2/dense_5/MatMul_1/ReadVariableOp4model_3/sequential_2/dense_5/MatMul_1/ReadVariableOp:X T
/
_output_shapes
:?????????8.
!
_user_specified_name	input_5:XT
/
_output_shapes
:?????????8.
!
_user_specified_name	input_6
?0
?
G__inference_sequential_2_layer_call_and_return_conditional_losses_15249
conv2d_4_input(
conv2d_4_15220:
conv2d_4_15222:(
conv2d_5_15228:
conv2d_5_15230:!
dense_4_15237:
??
dense_4_15239:	? 
dense_5_15243:	?2
dense_5_15245:2
identity?? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallconv2d_4_inputconv2d_4_15220conv2d_4_15222*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????6,*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_148742"
 conv2d_4/StatefulPartitionedCall?
activation_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????6,* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_148852
activation_4/PartitionedCall?
max_pooling2d_4/PartitionedCallPartitionedCall%activation_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????,* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_148392!
max_pooling2d_4/PartitionedCall?
dropout_6/PartitionedCallPartitionedCall(max_pooling2d_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????,* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_148932
dropout_6/PartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall"dropout_6/PartitionedCall:output:0conv2d_5_15228conv2d_5_15230*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????**$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_149052"
 conv2d_5/StatefulPartitionedCall?
activation_5/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????** 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_activation_5_layer_call_and_return_conditional_losses_149162
activation_5/PartitionedCall?
max_pooling2d_5/PartitionedCallPartitionedCall%activation_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????** 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_148512!
max_pooling2d_5/PartitionedCall?
dropout_7/PartitionedCallPartitionedCall(max_pooling2d_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????** 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_149242
dropout_7/PartitionedCall?
flatten_2/PartitionedCallPartitionedCall"dropout_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_149322
flatten_2/PartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_4_15237dense_4_15239*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_149452!
dense_4/StatefulPartitionedCall?
dropout_8/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_149562
dropout_8/PartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall"dropout_8/PartitionedCall:output:0dense_5_15243dense_5_15245*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_149692!
dense_5/StatefulPartitionedCall?
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????8.: : : : : : : : 2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:_ [
/
_output_shapes
:?????????8.
(
_user_specified_nameconv2d_4_input
?d
?
!__inference__traced_restore_16340
file_prefix'
assignvariableop_rmsprop_iter:	 *
 assignvariableop_1_rmsprop_decay: 2
(assignvariableop_2_rmsprop_learning_rate: -
#assignvariableop_3_rmsprop_momentum: (
assignvariableop_4_rmsprop_rho: <
"assignvariableop_5_conv2d_4_kernel:.
 assignvariableop_6_conv2d_4_bias:<
"assignvariableop_7_conv2d_5_kernel:.
 assignvariableop_8_conv2d_5_bias:5
!assignvariableop_9_dense_4_kernel:
??/
 assignvariableop_10_dense_4_bias:	?5
"assignvariableop_11_dense_5_kernel:	?2.
 assignvariableop_12_dense_5_bias:2#
assignvariableop_13_total: #
assignvariableop_14_count: I
/assignvariableop_15_rmsprop_conv2d_4_kernel_rms:;
-assignvariableop_16_rmsprop_conv2d_4_bias_rms:I
/assignvariableop_17_rmsprop_conv2d_5_kernel_rms:;
-assignvariableop_18_rmsprop_conv2d_5_bias_rms:B
.assignvariableop_19_rmsprop_dense_4_kernel_rms:
??;
,assignvariableop_20_rmsprop_dense_4_bias_rms:	?A
.assignvariableop_21_rmsprop_dense_5_kernel_rms:	?2:
,assignvariableop_22_rmsprop_dense_5_bias_rms:2
identity_24??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*C
value:B8B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*t
_output_shapesb
`::::::::::::::::::::::::*&
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_rmsprop_iterIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_rmsprop_decayIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp(assignvariableop_2_rmsprop_learning_rateIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp#assignvariableop_3_rmsprop_momentumIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_rmsprop_rhoIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp"assignvariableop_5_conv2d_4_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp assignvariableop_6_conv2d_4_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp"assignvariableop_7_conv2d_5_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp assignvariableop_8_conv2d_5_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp!assignvariableop_9_dense_4_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp assignvariableop_10_dense_4_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp"assignvariableop_11_dense_5_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp assignvariableop_12_dense_5_biasIdentity_12:output:0"/device:CPU:0*
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
AssignVariableOp_15AssignVariableOp/assignvariableop_15_rmsprop_conv2d_4_kernel_rmsIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp-assignvariableop_16_rmsprop_conv2d_4_bias_rmsIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp/assignvariableop_17_rmsprop_conv2d_5_kernel_rmsIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp-assignvariableop_18_rmsprop_conv2d_5_bias_rmsIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp.assignvariableop_19_rmsprop_dense_4_kernel_rmsIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp,assignvariableop_20_rmsprop_dense_4_bias_rmsIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp.assignvariableop_21_rmsprop_dense_5_kernel_rmsIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp,assignvariableop_22_rmsprop_dense_5_bias_rmsIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_229
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_23Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_23?
Identity_24IdentityIdentity_23:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_24"#
identity_24Identity_24:output:0*C
_input_shapes2
0: : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_22AssignVariableOp_222(
AssignVariableOp_3AssignVariableOp_32(
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
??
?
B__inference_model_3_layer_call_and_return_conditional_losses_15801
inputs_0
inputs_1N
4sequential_2_conv2d_4_conv2d_readvariableop_resource:C
5sequential_2_conv2d_4_biasadd_readvariableop_resource:N
4sequential_2_conv2d_5_conv2d_readvariableop_resource:C
5sequential_2_conv2d_5_biasadd_readvariableop_resource:G
3sequential_2_dense_4_matmul_readvariableop_resource:
??C
4sequential_2_dense_4_biasadd_readvariableop_resource:	?F
3sequential_2_dense_5_matmul_readvariableop_resource:	?2B
4sequential_2_dense_5_biasadd_readvariableop_resource:2
identity??,sequential_2/conv2d_4/BiasAdd/ReadVariableOp?.sequential_2/conv2d_4/BiasAdd_1/ReadVariableOp?+sequential_2/conv2d_4/Conv2D/ReadVariableOp?-sequential_2/conv2d_4/Conv2D_1/ReadVariableOp?,sequential_2/conv2d_5/BiasAdd/ReadVariableOp?.sequential_2/conv2d_5/BiasAdd_1/ReadVariableOp?+sequential_2/conv2d_5/Conv2D/ReadVariableOp?-sequential_2/conv2d_5/Conv2D_1/ReadVariableOp?+sequential_2/dense_4/BiasAdd/ReadVariableOp?-sequential_2/dense_4/BiasAdd_1/ReadVariableOp?*sequential_2/dense_4/MatMul/ReadVariableOp?,sequential_2/dense_4/MatMul_1/ReadVariableOp?+sequential_2/dense_5/BiasAdd/ReadVariableOp?-sequential_2/dense_5/BiasAdd_1/ReadVariableOp?*sequential_2/dense_5/MatMul/ReadVariableOp?,sequential_2/dense_5/MatMul_1/ReadVariableOp?
+sequential_2/conv2d_4/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02-
+sequential_2/conv2d_4/Conv2D/ReadVariableOp?
sequential_2/conv2d_4/Conv2DConv2Dinputs_03sequential_2/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????6,*
data_formatNCHW*
paddingVALID*
strides
2
sequential_2/conv2d_4/Conv2D?
,sequential_2/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_2/conv2d_4/BiasAdd/ReadVariableOp?
sequential_2/conv2d_4/BiasAddBiasAdd%sequential_2/conv2d_4/Conv2D:output:04sequential_2/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????6,*
data_formatNCHW2
sequential_2/conv2d_4/BiasAdd?
sequential_2/activation_4/ReluRelu&sequential_2/conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????6,2 
sequential_2/activation_4/Relu?
$sequential_2/max_pooling2d_4/MaxPoolMaxPool,sequential_2/activation_4/Relu:activations:0*/
_output_shapes
:?????????,*
ksize
*
paddingVALID*
strides
2&
$sequential_2/max_pooling2d_4/MaxPool?
$sequential_2/dropout_6/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2&
$sequential_2/dropout_6/dropout/Const?
"sequential_2/dropout_6/dropout/MulMul-sequential_2/max_pooling2d_4/MaxPool:output:0-sequential_2/dropout_6/dropout/Const:output:0*
T0*/
_output_shapes
:?????????,2$
"sequential_2/dropout_6/dropout/Mul?
$sequential_2/dropout_6/dropout/ShapeShape-sequential_2/max_pooling2d_4/MaxPool:output:0*
T0*
_output_shapes
:2&
$sequential_2/dropout_6/dropout/Shape?
;sequential_2/dropout_6/dropout/random_uniform/RandomUniformRandomUniform-sequential_2/dropout_6/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????,*
dtype02=
;sequential_2/dropout_6/dropout/random_uniform/RandomUniform?
-sequential_2/dropout_6/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2/
-sequential_2/dropout_6/dropout/GreaterEqual/y?
+sequential_2/dropout_6/dropout/GreaterEqualGreaterEqualDsequential_2/dropout_6/dropout/random_uniform/RandomUniform:output:06sequential_2/dropout_6/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????,2-
+sequential_2/dropout_6/dropout/GreaterEqual?
#sequential_2/dropout_6/dropout/CastCast/sequential_2/dropout_6/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????,2%
#sequential_2/dropout_6/dropout/Cast?
$sequential_2/dropout_6/dropout/Mul_1Mul&sequential_2/dropout_6/dropout/Mul:z:0'sequential_2/dropout_6/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????,2&
$sequential_2/dropout_6/dropout/Mul_1?
+sequential_2/conv2d_5/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02-
+sequential_2/conv2d_5/Conv2D/ReadVariableOp?
sequential_2/conv2d_5/Conv2DConv2D(sequential_2/dropout_6/dropout/Mul_1:z:03sequential_2/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**
data_formatNCHW*
paddingVALID*
strides
2
sequential_2/conv2d_5/Conv2D?
,sequential_2/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_2/conv2d_5/BiasAdd/ReadVariableOp?
sequential_2/conv2d_5/BiasAddBiasAdd%sequential_2/conv2d_5/Conv2D:output:04sequential_2/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**
data_formatNCHW2
sequential_2/conv2d_5/BiasAdd?
sequential_2/activation_5/ReluRelu&sequential_2/conv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:?????????*2 
sequential_2/activation_5/Relu?
$sequential_2/max_pooling2d_5/MaxPoolMaxPool,sequential_2/activation_5/Relu:activations:0*/
_output_shapes
:?????????**
ksize
*
paddingVALID*
strides
2&
$sequential_2/max_pooling2d_5/MaxPool?
$sequential_2/dropout_7/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2&
$sequential_2/dropout_7/dropout/Const?
"sequential_2/dropout_7/dropout/MulMul-sequential_2/max_pooling2d_5/MaxPool:output:0-sequential_2/dropout_7/dropout/Const:output:0*
T0*/
_output_shapes
:?????????*2$
"sequential_2/dropout_7/dropout/Mul?
$sequential_2/dropout_7/dropout/ShapeShape-sequential_2/max_pooling2d_5/MaxPool:output:0*
T0*
_output_shapes
:2&
$sequential_2/dropout_7/dropout/Shape?
;sequential_2/dropout_7/dropout/random_uniform/RandomUniformRandomUniform-sequential_2/dropout_7/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????**
dtype02=
;sequential_2/dropout_7/dropout/random_uniform/RandomUniform?
-sequential_2/dropout_7/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2/
-sequential_2/dropout_7/dropout/GreaterEqual/y?
+sequential_2/dropout_7/dropout/GreaterEqualGreaterEqualDsequential_2/dropout_7/dropout/random_uniform/RandomUniform:output:06sequential_2/dropout_7/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????*2-
+sequential_2/dropout_7/dropout/GreaterEqual?
#sequential_2/dropout_7/dropout/CastCast/sequential_2/dropout_7/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????*2%
#sequential_2/dropout_7/dropout/Cast?
$sequential_2/dropout_7/dropout/Mul_1Mul&sequential_2/dropout_7/dropout/Mul:z:0'sequential_2/dropout_7/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????*2&
$sequential_2/dropout_7/dropout/Mul_1?
sequential_2/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
sequential_2/flatten_2/Const?
sequential_2/flatten_2/ReshapeReshape(sequential_2/dropout_7/dropout/Mul_1:z:0%sequential_2/flatten_2/Const:output:0*
T0*(
_output_shapes
:??????????2 
sequential_2/flatten_2/Reshape?
*sequential_2/dense_4/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*sequential_2/dense_4/MatMul/ReadVariableOp?
sequential_2/dense_4/MatMulMatMul'sequential_2/flatten_2/Reshape:output:02sequential_2/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_2/dense_4/MatMul?
+sequential_2/dense_4/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential_2/dense_4/BiasAdd/ReadVariableOp?
sequential_2/dense_4/BiasAddBiasAdd%sequential_2/dense_4/MatMul:product:03sequential_2/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_2/dense_4/BiasAdd?
sequential_2/dense_4/ReluRelu%sequential_2/dense_4/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_2/dense_4/Relu?
$sequential_2/dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2&
$sequential_2/dropout_8/dropout/Const?
"sequential_2/dropout_8/dropout/MulMul'sequential_2/dense_4/Relu:activations:0-sequential_2/dropout_8/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2$
"sequential_2/dropout_8/dropout/Mul?
$sequential_2/dropout_8/dropout/ShapeShape'sequential_2/dense_4/Relu:activations:0*
T0*
_output_shapes
:2&
$sequential_2/dropout_8/dropout/Shape?
;sequential_2/dropout_8/dropout/random_uniform/RandomUniformRandomUniform-sequential_2/dropout_8/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02=
;sequential_2/dropout_8/dropout/random_uniform/RandomUniform?
-sequential_2/dropout_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2/
-sequential_2/dropout_8/dropout/GreaterEqual/y?
+sequential_2/dropout_8/dropout/GreaterEqualGreaterEqualDsequential_2/dropout_8/dropout/random_uniform/RandomUniform:output:06sequential_2/dropout_8/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2-
+sequential_2/dropout_8/dropout/GreaterEqual?
#sequential_2/dropout_8/dropout/CastCast/sequential_2/dropout_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2%
#sequential_2/dropout_8/dropout/Cast?
$sequential_2/dropout_8/dropout/Mul_1Mul&sequential_2/dropout_8/dropout/Mul:z:0'sequential_2/dropout_8/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2&
$sequential_2/dropout_8/dropout/Mul_1?
*sequential_2/dense_5/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_5_matmul_readvariableop_resource*
_output_shapes
:	?2*
dtype02,
*sequential_2/dense_5/MatMul/ReadVariableOp?
sequential_2/dense_5/MatMulMatMul(sequential_2/dropout_8/dropout/Mul_1:z:02sequential_2/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
sequential_2/dense_5/MatMul?
+sequential_2/dense_5/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_5_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02-
+sequential_2/dense_5/BiasAdd/ReadVariableOp?
sequential_2/dense_5/BiasAddBiasAdd%sequential_2/dense_5/MatMul:product:03sequential_2/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
sequential_2/dense_5/BiasAdd?
sequential_2/dense_5/ReluRelu%sequential_2/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
sequential_2/dense_5/Relu?
-sequential_2/conv2d_4/Conv2D_1/ReadVariableOpReadVariableOp4sequential_2_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02/
-sequential_2/conv2d_4/Conv2D_1/ReadVariableOp?
sequential_2/conv2d_4/Conv2D_1Conv2Dinputs_15sequential_2/conv2d_4/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????6,*
data_formatNCHW*
paddingVALID*
strides
2 
sequential_2/conv2d_4/Conv2D_1?
.sequential_2/conv2d_4/BiasAdd_1/ReadVariableOpReadVariableOp5sequential_2_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.sequential_2/conv2d_4/BiasAdd_1/ReadVariableOp?
sequential_2/conv2d_4/BiasAdd_1BiasAdd'sequential_2/conv2d_4/Conv2D_1:output:06sequential_2/conv2d_4/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????6,*
data_formatNCHW2!
sequential_2/conv2d_4/BiasAdd_1?
 sequential_2/activation_4/Relu_1Relu(sequential_2/conv2d_4/BiasAdd_1:output:0*
T0*/
_output_shapes
:?????????6,2"
 sequential_2/activation_4/Relu_1?
&sequential_2/max_pooling2d_4/MaxPool_1MaxPool.sequential_2/activation_4/Relu_1:activations:0*/
_output_shapes
:?????????,*
ksize
*
paddingVALID*
strides
2(
&sequential_2/max_pooling2d_4/MaxPool_1?
&sequential_2/dropout_6/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2(
&sequential_2/dropout_6/dropout_1/Const?
$sequential_2/dropout_6/dropout_1/MulMul/sequential_2/max_pooling2d_4/MaxPool_1:output:0/sequential_2/dropout_6/dropout_1/Const:output:0*
T0*/
_output_shapes
:?????????,2&
$sequential_2/dropout_6/dropout_1/Mul?
&sequential_2/dropout_6/dropout_1/ShapeShape/sequential_2/max_pooling2d_4/MaxPool_1:output:0*
T0*
_output_shapes
:2(
&sequential_2/dropout_6/dropout_1/Shape?
=sequential_2/dropout_6/dropout_1/random_uniform/RandomUniformRandomUniform/sequential_2/dropout_6/dropout_1/Shape:output:0*
T0*/
_output_shapes
:?????????,*
dtype02?
=sequential_2/dropout_6/dropout_1/random_uniform/RandomUniform?
/sequential_2/dropout_6/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>21
/sequential_2/dropout_6/dropout_1/GreaterEqual/y?
-sequential_2/dropout_6/dropout_1/GreaterEqualGreaterEqualFsequential_2/dropout_6/dropout_1/random_uniform/RandomUniform:output:08sequential_2/dropout_6/dropout_1/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????,2/
-sequential_2/dropout_6/dropout_1/GreaterEqual?
%sequential_2/dropout_6/dropout_1/CastCast1sequential_2/dropout_6/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????,2'
%sequential_2/dropout_6/dropout_1/Cast?
&sequential_2/dropout_6/dropout_1/Mul_1Mul(sequential_2/dropout_6/dropout_1/Mul:z:0)sequential_2/dropout_6/dropout_1/Cast:y:0*
T0*/
_output_shapes
:?????????,2(
&sequential_2/dropout_6/dropout_1/Mul_1?
-sequential_2/conv2d_5/Conv2D_1/ReadVariableOpReadVariableOp4sequential_2_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02/
-sequential_2/conv2d_5/Conv2D_1/ReadVariableOp?
sequential_2/conv2d_5/Conv2D_1Conv2D*sequential_2/dropout_6/dropout_1/Mul_1:z:05sequential_2/conv2d_5/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**
data_formatNCHW*
paddingVALID*
strides
2 
sequential_2/conv2d_5/Conv2D_1?
.sequential_2/conv2d_5/BiasAdd_1/ReadVariableOpReadVariableOp5sequential_2_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.sequential_2/conv2d_5/BiasAdd_1/ReadVariableOp?
sequential_2/conv2d_5/BiasAdd_1BiasAdd'sequential_2/conv2d_5/Conv2D_1:output:06sequential_2/conv2d_5/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**
data_formatNCHW2!
sequential_2/conv2d_5/BiasAdd_1?
 sequential_2/activation_5/Relu_1Relu(sequential_2/conv2d_5/BiasAdd_1:output:0*
T0*/
_output_shapes
:?????????*2"
 sequential_2/activation_5/Relu_1?
&sequential_2/max_pooling2d_5/MaxPool_1MaxPool.sequential_2/activation_5/Relu_1:activations:0*/
_output_shapes
:?????????**
ksize
*
paddingVALID*
strides
2(
&sequential_2/max_pooling2d_5/MaxPool_1?
&sequential_2/dropout_7/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2(
&sequential_2/dropout_7/dropout_1/Const?
$sequential_2/dropout_7/dropout_1/MulMul/sequential_2/max_pooling2d_5/MaxPool_1:output:0/sequential_2/dropout_7/dropout_1/Const:output:0*
T0*/
_output_shapes
:?????????*2&
$sequential_2/dropout_7/dropout_1/Mul?
&sequential_2/dropout_7/dropout_1/ShapeShape/sequential_2/max_pooling2d_5/MaxPool_1:output:0*
T0*
_output_shapes
:2(
&sequential_2/dropout_7/dropout_1/Shape?
=sequential_2/dropout_7/dropout_1/random_uniform/RandomUniformRandomUniform/sequential_2/dropout_7/dropout_1/Shape:output:0*
T0*/
_output_shapes
:?????????**
dtype02?
=sequential_2/dropout_7/dropout_1/random_uniform/RandomUniform?
/sequential_2/dropout_7/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>21
/sequential_2/dropout_7/dropout_1/GreaterEqual/y?
-sequential_2/dropout_7/dropout_1/GreaterEqualGreaterEqualFsequential_2/dropout_7/dropout_1/random_uniform/RandomUniform:output:08sequential_2/dropout_7/dropout_1/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????*2/
-sequential_2/dropout_7/dropout_1/GreaterEqual?
%sequential_2/dropout_7/dropout_1/CastCast1sequential_2/dropout_7/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????*2'
%sequential_2/dropout_7/dropout_1/Cast?
&sequential_2/dropout_7/dropout_1/Mul_1Mul(sequential_2/dropout_7/dropout_1/Mul:z:0)sequential_2/dropout_7/dropout_1/Cast:y:0*
T0*/
_output_shapes
:?????????*2(
&sequential_2/dropout_7/dropout_1/Mul_1?
sequential_2/flatten_2/Const_1Const*
_output_shapes
:*
dtype0*
valueB"?????  2 
sequential_2/flatten_2/Const_1?
 sequential_2/flatten_2/Reshape_1Reshape*sequential_2/dropout_7/dropout_1/Mul_1:z:0'sequential_2/flatten_2/Const_1:output:0*
T0*(
_output_shapes
:??????????2"
 sequential_2/flatten_2/Reshape_1?
,sequential_2/dense_4/MatMul_1/ReadVariableOpReadVariableOp3sequential_2_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02.
,sequential_2/dense_4/MatMul_1/ReadVariableOp?
sequential_2/dense_4/MatMul_1MatMul)sequential_2/flatten_2/Reshape_1:output:04sequential_2/dense_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_2/dense_4/MatMul_1?
-sequential_2/dense_4/BiasAdd_1/ReadVariableOpReadVariableOp4sequential_2_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-sequential_2/dense_4/BiasAdd_1/ReadVariableOp?
sequential_2/dense_4/BiasAdd_1BiasAdd'sequential_2/dense_4/MatMul_1:product:05sequential_2/dense_4/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
sequential_2/dense_4/BiasAdd_1?
sequential_2/dense_4/Relu_1Relu'sequential_2/dense_4/BiasAdd_1:output:0*
T0*(
_output_shapes
:??????????2
sequential_2/dense_4/Relu_1?
&sequential_2/dropout_8/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2(
&sequential_2/dropout_8/dropout_1/Const?
$sequential_2/dropout_8/dropout_1/MulMul)sequential_2/dense_4/Relu_1:activations:0/sequential_2/dropout_8/dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????2&
$sequential_2/dropout_8/dropout_1/Mul?
&sequential_2/dropout_8/dropout_1/ShapeShape)sequential_2/dense_4/Relu_1:activations:0*
T0*
_output_shapes
:2(
&sequential_2/dropout_8/dropout_1/Shape?
=sequential_2/dropout_8/dropout_1/random_uniform/RandomUniformRandomUniform/sequential_2/dropout_8/dropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02?
=sequential_2/dropout_8/dropout_1/random_uniform/RandomUniform?
/sequential_2/dropout_8/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=21
/sequential_2/dropout_8/dropout_1/GreaterEqual/y?
-sequential_2/dropout_8/dropout_1/GreaterEqualGreaterEqualFsequential_2/dropout_8/dropout_1/random_uniform/RandomUniform:output:08sequential_2/dropout_8/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2/
-sequential_2/dropout_8/dropout_1/GreaterEqual?
%sequential_2/dropout_8/dropout_1/CastCast1sequential_2/dropout_8/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2'
%sequential_2/dropout_8/dropout_1/Cast?
&sequential_2/dropout_8/dropout_1/Mul_1Mul(sequential_2/dropout_8/dropout_1/Mul:z:0)sequential_2/dropout_8/dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????2(
&sequential_2/dropout_8/dropout_1/Mul_1?
,sequential_2/dense_5/MatMul_1/ReadVariableOpReadVariableOp3sequential_2_dense_5_matmul_readvariableop_resource*
_output_shapes
:	?2*
dtype02.
,sequential_2/dense_5/MatMul_1/ReadVariableOp?
sequential_2/dense_5/MatMul_1MatMul*sequential_2/dropout_8/dropout_1/Mul_1:z:04sequential_2/dense_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
sequential_2/dense_5/MatMul_1?
-sequential_2/dense_5/BiasAdd_1/ReadVariableOpReadVariableOp4sequential_2_dense_5_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02/
-sequential_2/dense_5/BiasAdd_1/ReadVariableOp?
sequential_2/dense_5/BiasAdd_1BiasAdd'sequential_2/dense_5/MatMul_1:product:05sequential_2/dense_5/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22 
sequential_2/dense_5/BiasAdd_1?
sequential_2/dense_5/Relu_1Relu'sequential_2/dense_5/BiasAdd_1:output:0*
T0*'
_output_shapes
:?????????22
sequential_2/dense_5/Relu_1?
lambda_3/subSub'sequential_2/dense_5/Relu:activations:0)sequential_2/dense_5/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
lambda_3/subp
lambda_3/SquareSquarelambda_3/sub:z:0*
T0*'
_output_shapes
:?????????22
lambda_3/Square?
lambda_3/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2 
lambda_3/Sum/reduction_indices?
lambda_3/SumSumlambda_3/Square:y:0'lambda_3/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2
lambda_3/Sume
lambda_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lambda_3/Const?
lambda_3/MaximumMaximumlambda_3/Sum:output:0lambda_3/Const:output:0*
T0*'
_output_shapes
:?????????2
lambda_3/Maximumn
lambda_3/SqrtSqrtlambda_3/Maximum:z:0*
T0*'
_output_shapes
:?????????2
lambda_3/Sqrt?
IdentityIdentitylambda_3/Sqrt:y:0-^sequential_2/conv2d_4/BiasAdd/ReadVariableOp/^sequential_2/conv2d_4/BiasAdd_1/ReadVariableOp,^sequential_2/conv2d_4/Conv2D/ReadVariableOp.^sequential_2/conv2d_4/Conv2D_1/ReadVariableOp-^sequential_2/conv2d_5/BiasAdd/ReadVariableOp/^sequential_2/conv2d_5/BiasAdd_1/ReadVariableOp,^sequential_2/conv2d_5/Conv2D/ReadVariableOp.^sequential_2/conv2d_5/Conv2D_1/ReadVariableOp,^sequential_2/dense_4/BiasAdd/ReadVariableOp.^sequential_2/dense_4/BiasAdd_1/ReadVariableOp+^sequential_2/dense_4/MatMul/ReadVariableOp-^sequential_2/dense_4/MatMul_1/ReadVariableOp,^sequential_2/dense_5/BiasAdd/ReadVariableOp.^sequential_2/dense_5/BiasAdd_1/ReadVariableOp+^sequential_2/dense_5/MatMul/ReadVariableOp-^sequential_2/dense_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:?????????8.:?????????8.: : : : : : : : 2\
,sequential_2/conv2d_4/BiasAdd/ReadVariableOp,sequential_2/conv2d_4/BiasAdd/ReadVariableOp2`
.sequential_2/conv2d_4/BiasAdd_1/ReadVariableOp.sequential_2/conv2d_4/BiasAdd_1/ReadVariableOp2Z
+sequential_2/conv2d_4/Conv2D/ReadVariableOp+sequential_2/conv2d_4/Conv2D/ReadVariableOp2^
-sequential_2/conv2d_4/Conv2D_1/ReadVariableOp-sequential_2/conv2d_4/Conv2D_1/ReadVariableOp2\
,sequential_2/conv2d_5/BiasAdd/ReadVariableOp,sequential_2/conv2d_5/BiasAdd/ReadVariableOp2`
.sequential_2/conv2d_5/BiasAdd_1/ReadVariableOp.sequential_2/conv2d_5/BiasAdd_1/ReadVariableOp2Z
+sequential_2/conv2d_5/Conv2D/ReadVariableOp+sequential_2/conv2d_5/Conv2D/ReadVariableOp2^
-sequential_2/conv2d_5/Conv2D_1/ReadVariableOp-sequential_2/conv2d_5/Conv2D_1/ReadVariableOp2Z
+sequential_2/dense_4/BiasAdd/ReadVariableOp+sequential_2/dense_4/BiasAdd/ReadVariableOp2^
-sequential_2/dense_4/BiasAdd_1/ReadVariableOp-sequential_2/dense_4/BiasAdd_1/ReadVariableOp2X
*sequential_2/dense_4/MatMul/ReadVariableOp*sequential_2/dense_4/MatMul/ReadVariableOp2\
,sequential_2/dense_4/MatMul_1/ReadVariableOp,sequential_2/dense_4/MatMul_1/ReadVariableOp2Z
+sequential_2/dense_5/BiasAdd/ReadVariableOp+sequential_2/dense_5/BiasAdd/ReadVariableOp2^
-sequential_2/dense_5/BiasAdd_1/ReadVariableOp-sequential_2/dense_5/BiasAdd_1/ReadVariableOp2X
*sequential_2/dense_5/MatMul/ReadVariableOp*sequential_2/dense_5/MatMul/ReadVariableOp2\
,sequential_2/dense_5/MatMul_1/ReadVariableOp,sequential_2/dense_5/MatMul_1/ReadVariableOp:Y U
/
_output_shapes
:?????????8.
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????8.
"
_user_specified_name
inputs/1
?4
?
G__inference_sequential_2_layer_call_and_return_conditional_losses_15281
conv2d_4_input(
conv2d_4_15252:
conv2d_4_15254:(
conv2d_5_15260:
conv2d_5_15262:!
dense_4_15269:
??
dense_4_15271:	? 
dense_5_15275:	?2
dense_5_15277:2
identity?? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?!dropout_6/StatefulPartitionedCall?!dropout_7/StatefulPartitionedCall?!dropout_8/StatefulPartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallconv2d_4_inputconv2d_4_15252conv2d_4_15254*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????6,*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_148742"
 conv2d_4/StatefulPartitionedCall?
activation_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????6,* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_148852
activation_4/PartitionedCall?
max_pooling2d_4/PartitionedCallPartitionedCall%activation_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????,* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_148392!
max_pooling2d_4/PartitionedCall?
!dropout_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????,* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_151032#
!dropout_6/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall*dropout_6/StatefulPartitionedCall:output:0conv2d_5_15260conv2d_5_15262*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????**$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_149052"
 conv2d_5/StatefulPartitionedCall?
activation_5/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????** 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_activation_5_layer_call_and_return_conditional_losses_149162
activation_5/PartitionedCall?
max_pooling2d_5/PartitionedCallPartitionedCall%activation_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????** 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_148512!
max_pooling2d_5/PartitionedCall?
!dropout_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_5/PartitionedCall:output:0"^dropout_6/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????** 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_150642#
!dropout_7/StatefulPartitionedCall?
flatten_2/PartitionedCallPartitionedCall*dropout_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_149322
flatten_2/PartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_4_15269dense_4_15271*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_149452!
dense_4/StatefulPartitionedCall?
!dropout_8/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0"^dropout_7/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_150252#
!dropout_8/StatefulPartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall*dropout_8/StatefulPartitionedCall:output:0dense_5_15275dense_5_15277*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_149692!
dense_5/StatefulPartitionedCall?
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall"^dropout_6/StatefulPartitionedCall"^dropout_7/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????8.: : : : : : : : 2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2F
!dropout_6/StatefulPartitionedCall!dropout_6/StatefulPartitionedCall2F
!dropout_7/StatefulPartitionedCall!dropout_7/StatefulPartitionedCall2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall:_ [
/
_output_shapes
:?????????8.
(
_user_specified_nameconv2d_4_input
?	
o
C__inference_lambda_3_layer_call_and_return_conditional_losses_15966
inputs_0
inputs_1
identityW
subSubinputs_0inputs_1*
T0*'
_output_shapes
:?????????22
subU
SquareSquaresub:z:0*
T0*'
_output_shapes
:?????????22
Squarep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices?
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2
SumS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Constm
MaximumMaximumSum:output:0Const:output:0*
T0*'
_output_shapes
:?????????2	
MaximumS
SqrtSqrtMaximum:z:0*
T0*'
_output_shapes
:?????????2
Sqrt\
IdentityIdentitySqrt:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????2:?????????2:Q M
'
_output_shapes
:?????????2
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????2
"
_user_specified_name
inputs/1
?	
?
,__inference_sequential_2_layer_call_fn_15843

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?2
	unknown_6:2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_151772
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????8.: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????8.
 
_user_specified_nameinputs
?
E
)__inference_flatten_2_layer_call_fn_16095

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
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_149322
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????*:W S
/
_output_shapes
:?????????*
 
_user_specified_nameinputs
?
E
)__inference_dropout_6_layer_call_fn_16012

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
:?????????,* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_148932
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????,2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????,:W S
/
_output_shapes
:?????????,
 
_user_specified_nameinputs
?	
?
,__inference_sequential_2_layer_call_fn_14995
conv2d_4_input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?2
	unknown_6:2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_4_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_149762
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????8.: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:?????????8.
(
_user_specified_nameconv2d_4_input
?
K
/__inference_max_pooling2d_5_layer_call_fn_14857

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
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_148512
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_14839

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?	
m
C__inference_lambda_3_layer_call_and_return_conditional_losses_15372

inputs
inputs_1
identityU
subSubinputsinputs_1*
T0*'
_output_shapes
:?????????22
subU
SquareSquaresub:z:0*
T0*'
_output_shapes
:?????????22
Squarep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices?
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2
SumS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Constm
MaximumMaximumSum:output:0Const:output:0*
T0*'
_output_shapes
:?????????2	
MaximumS
SqrtSqrtMaximum:z:0*
T0*'
_output_shapes
:?????????2
Sqrt\
IdentityIdentitySqrt:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????2:?????????2:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
(__inference_conv2d_5_layer_call_fn_16043

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????**$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_149052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????*2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????,: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????,
 
_user_specified_nameinputs
?
T
(__inference_lambda_3_layer_call_fn_15954
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_lambda_3_layer_call_and_return_conditional_losses_153722
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????2:?????????2:Q M
'
_output_shapes
:?????????2
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????2
"
_user_specified_name
inputs/1
?
b
)__inference_dropout_8_layer_call_fn_16131

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_150252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
E
)__inference_dropout_8_layer_call_fn_16126

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
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_149562
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
b
)__inference_dropout_6_layer_call_fn_16017

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
:?????????,* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_151032
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????,2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????,22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????,
 
_user_specified_nameinputs
?

?
'__inference_model_3_layer_call_fn_15350
input_5
input_6!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?2
	unknown_6:2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_5input_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_model_3_layer_call_and_return_conditional_losses_153312
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:?????????8.:?????????8.: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????8.
!
_user_specified_name	input_5:XT
/
_output_shapes
:?????????8.
!
_user_specified_name	input_6
?
?
'__inference_dense_5_layer_call_fn_16157

inputs
unknown:	?2
	unknown_0:2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_149692
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
E
)__inference_dropout_7_layer_call_fn_16068

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
:?????????** 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_149242
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????*2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????*:W S
/
_output_shapes
:?????????*
 
_user_specified_nameinputs
?

?
B__inference_dense_4_layer_call_and_return_conditional_losses_14945

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
B__inference_dense_5_layer_call_and_return_conditional_losses_14969

inputs1
matmul_readvariableop_resource:	?2-
biasadd_readvariableop_resource:2
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????22
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
b
)__inference_dropout_7_layer_call_fn_16073

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
:?????????** 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_150642
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????*2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????*22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????*
 
_user_specified_nameinputs
?
K
/__inference_max_pooling2d_4_layer_call_fn_14845

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
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_148392
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
'__inference_model_3_layer_call_fn_15611
inputs_0
inputs_1!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?2
	unknown_6:2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_model_3_layer_call_and_return_conditional_losses_154322
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:?????????8.:?????????8.: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:?????????8.
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????8.
"
_user_specified_name
inputs/1
?
H
,__inference_activation_5_layer_call_fn_16058

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
:?????????** 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_activation_5_layer_call_and_return_conditional_losses_149162
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????*2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????*:W S
/
_output_shapes
:?????????*
 
_user_specified_nameinputs
?

?
C__inference_conv2d_4_layer_call_and_return_conditional_losses_14874

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????6,*
data_formatNCHW*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????6,*
data_formatNCHW2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????6,2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????8.: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????8.
 
_user_specified_nameinputs
?7
?	
__inference__traced_save_16261
file_prefix+
'savev2_rmsprop_iter_read_readvariableop	,
(savev2_rmsprop_decay_read_readvariableop4
0savev2_rmsprop_learning_rate_read_readvariableop/
+savev2_rmsprop_momentum_read_readvariableop*
&savev2_rmsprop_rho_read_readvariableop.
*savev2_conv2d_4_kernel_read_readvariableop,
(savev2_conv2d_4_bias_read_readvariableop.
*savev2_conv2d_5_kernel_read_readvariableop,
(savev2_conv2d_5_bias_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop:
6savev2_rmsprop_conv2d_4_kernel_rms_read_readvariableop8
4savev2_rmsprop_conv2d_4_bias_rms_read_readvariableop:
6savev2_rmsprop_conv2d_5_kernel_rms_read_readvariableop8
4savev2_rmsprop_conv2d_5_bias_rms_read_readvariableop9
5savev2_rmsprop_dense_4_kernel_rms_read_readvariableop7
3savev2_rmsprop_dense_4_bias_rms_read_readvariableop9
5savev2_rmsprop_dense_5_kernel_rms_read_readvariableop7
3savev2_rmsprop_dense_5_bias_rms_read_readvariableop
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*C
value:B8B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop*savev2_conv2d_4_kernel_read_readvariableop(savev2_conv2d_4_bias_read_readvariableop*savev2_conv2d_5_kernel_read_readvariableop(savev2_conv2d_5_bias_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop6savev2_rmsprop_conv2d_4_kernel_rms_read_readvariableop4savev2_rmsprop_conv2d_4_bias_rms_read_readvariableop6savev2_rmsprop_conv2d_5_kernel_rms_read_readvariableop4savev2_rmsprop_conv2d_5_bias_rms_read_readvariableop5savev2_rmsprop_dense_4_kernel_rms_read_readvariableop3savev2_rmsprop_dense_4_bias_rms_read_readvariableop5savev2_rmsprop_dense_5_kernel_rms_read_readvariableop3savev2_rmsprop_dense_5_bias_rms_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *&
dtypes
2	2
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

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : : :::::
??:?:	?2:2: : :::::
??:?:	?2:2: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 	

_output_shapes
::&
"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?2: 

_output_shapes
:2:

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?2: 

_output_shapes
:2:

_output_shapes
: 
?	
?
,__inference_sequential_2_layer_call_fn_15822

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?2
	unknown_6:2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_149762
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????8.: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????8.
 
_user_specified_nameinputs
?
c
G__inference_activation_4_layer_call_and_return_conditional_losses_16007

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????6,2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????6,2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????6,:W S
/
_output_shapes
:?????????6,
 
_user_specified_nameinputs
?
?
B__inference_model_3_layer_call_and_return_conditional_losses_15331

inputs
inputs_1,
sequential_2_15290: 
sequential_2_15292:,
sequential_2_15294: 
sequential_2_15296:&
sequential_2_15298:
??!
sequential_2_15300:	?%
sequential_2_15302:	?2 
sequential_2_15304:2
identity??$sequential_2/StatefulPartitionedCall?&sequential_2/StatefulPartitionedCall_1?
$sequential_2/StatefulPartitionedCallStatefulPartitionedCallinputssequential_2_15290sequential_2_15292sequential_2_15294sequential_2_15296sequential_2_15298sequential_2_15300sequential_2_15302sequential_2_15304*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_149762&
$sequential_2/StatefulPartitionedCall?
&sequential_2/StatefulPartitionedCall_1StatefulPartitionedCallinputs_1sequential_2_15290sequential_2_15292sequential_2_15294sequential_2_15296sequential_2_15298sequential_2_15300sequential_2_15302sequential_2_15304*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_149762(
&sequential_2/StatefulPartitionedCall_1?
lambda_3/PartitionedCallPartitionedCall-sequential_2/StatefulPartitionedCall:output:0/sequential_2/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_lambda_3_layer_call_and_return_conditional_losses_153282
lambda_3/PartitionedCall?
IdentityIdentity!lambda_3/PartitionedCall:output:0%^sequential_2/StatefulPartitionedCall'^sequential_2/StatefulPartitionedCall_1*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:?????????8.:?????????8.: : : : : : : : 2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall2P
&sequential_2/StatefulPartitionedCall_1&sequential_2/StatefulPartitionedCall_1:W S
/
_output_shapes
:?????????8.
 
_user_specified_nameinputs:WS
/
_output_shapes
:?????????8.
 
_user_specified_nameinputs
?2
?
G__inference_sequential_2_layer_call_and_return_conditional_losses_15882

inputsA
'conv2d_4_conv2d_readvariableop_resource:6
(conv2d_4_biasadd_readvariableop_resource:A
'conv2d_5_conv2d_readvariableop_resource:6
(conv2d_5_biasadd_readvariableop_resource::
&dense_4_matmul_readvariableop_resource:
??6
'dense_4_biasadd_readvariableop_resource:	?9
&dense_5_matmul_readvariableop_resource:	?25
'dense_5_biasadd_readvariableop_resource:2
identity??conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?conv2d_5/BiasAdd/ReadVariableOp?conv2d_5/Conv2D/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_4/Conv2D/ReadVariableOp?
conv2d_4/Conv2DConv2Dinputs&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????6,*
data_formatNCHW*
paddingVALID*
strides
2
conv2d_4/Conv2D?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????6,*
data_formatNCHW2
conv2d_4/BiasAdd?
activation_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????6,2
activation_4/Relu?
max_pooling2d_4/MaxPoolMaxPoolactivation_4/Relu:activations:0*/
_output_shapes
:?????????,*
ksize
*
paddingVALID*
strides
2
max_pooling2d_4/MaxPool?
dropout_6/IdentityIdentity max_pooling2d_4/MaxPool:output:0*
T0*/
_output_shapes
:?????????,2
dropout_6/Identity?
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_5/Conv2D/ReadVariableOp?
conv2d_5/Conv2DConv2Ddropout_6/Identity:output:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**
data_formatNCHW*
paddingVALID*
strides
2
conv2d_5/Conv2D?
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp?
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**
data_formatNCHW2
conv2d_5/BiasAdd?
activation_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:?????????*2
activation_5/Relu?
max_pooling2d_5/MaxPoolMaxPoolactivation_5/Relu:activations:0*/
_output_shapes
:?????????**
ksize
*
paddingVALID*
strides
2
max_pooling2d_5/MaxPool?
dropout_7/IdentityIdentity max_pooling2d_5/MaxPool:output:0*
T0*/
_output_shapes
:?????????*2
dropout_7/Identitys
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_2/Const?
flatten_2/ReshapeReshapedropout_7/Identity:output:0flatten_2/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_2/Reshape?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMulflatten_2/Reshape:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_4/BiasAddq
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_4/Relu?
dropout_8/IdentityIdentitydense_4/Relu:activations:0*
T0*(
_output_shapes
:??????????2
dropout_8/Identity?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	?2*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMuldropout_8/Identity:output:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
dense_5/BiasAddp
dense_5/ReluReludense_5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
dense_5/Relu?
IdentityIdentitydense_5/Relu:activations:0 ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????8.: : : : : : : : 2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????8.
 
_user_specified_nameinputs
?

?
B__inference_dense_5_layer_call_and_return_conditional_losses_16168

inputs1
matmul_readvariableop_resource:	?2-
biasadd_readvariableop_resource:2
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????22
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
D__inference_dropout_7_layer_call_and_return_conditional_losses_16090

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
:?????????*2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????**
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
:?????????*2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????*2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????*2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????*2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????*:W S
/
_output_shapes
:?????????*
 
_user_specified_nameinputs
?
?
B__inference_model_3_layer_call_and_return_conditional_losses_15537
input_5
input_6,
sequential_2_15509: 
sequential_2_15511:,
sequential_2_15513: 
sequential_2_15515:&
sequential_2_15517:
??!
sequential_2_15519:	?%
sequential_2_15521:	?2 
sequential_2_15523:2
identity??$sequential_2/StatefulPartitionedCall?&sequential_2/StatefulPartitionedCall_1?
$sequential_2/StatefulPartitionedCallStatefulPartitionedCallinput_5sequential_2_15509sequential_2_15511sequential_2_15513sequential_2_15515sequential_2_15517sequential_2_15519sequential_2_15521sequential_2_15523*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_151772&
$sequential_2/StatefulPartitionedCall?
&sequential_2/StatefulPartitionedCall_1StatefulPartitionedCallinput_6sequential_2_15509sequential_2_15511sequential_2_15513sequential_2_15515sequential_2_15517sequential_2_15519sequential_2_15521sequential_2_15523*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_151772(
&sequential_2/StatefulPartitionedCall_1?
lambda_3/PartitionedCallPartitionedCall-sequential_2/StatefulPartitionedCall:output:0/sequential_2/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_lambda_3_layer_call_and_return_conditional_losses_153722
lambda_3/PartitionedCall?
IdentityIdentity!lambda_3/PartitionedCall:output:0%^sequential_2/StatefulPartitionedCall'^sequential_2/StatefulPartitionedCall_1*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:?????????8.:?????????8.: : : : : : : : 2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall2P
&sequential_2/StatefulPartitionedCall_1&sequential_2/StatefulPartitionedCall_1:X T
/
_output_shapes
:?????????8.
!
_user_specified_name	input_5:XT
/
_output_shapes
:?????????8.
!
_user_specified_name	input_6
?
`
D__inference_flatten_2_layer_call_and_return_conditional_losses_14932

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????*:W S
/
_output_shapes
:?????????*
 
_user_specified_nameinputs
?

?
C__inference_conv2d_5_layer_call_and_return_conditional_losses_14905

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**
data_formatNCHW*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**
data_formatNCHW2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????*2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????,: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????,
 
_user_specified_nameinputs
?
?
B__inference_model_3_layer_call_and_return_conditional_losses_15432

inputs
inputs_1,
sequential_2_15404: 
sequential_2_15406:,
sequential_2_15408: 
sequential_2_15410:&
sequential_2_15412:
??!
sequential_2_15414:	?%
sequential_2_15416:	?2 
sequential_2_15418:2
identity??$sequential_2/StatefulPartitionedCall?&sequential_2/StatefulPartitionedCall_1?
$sequential_2/StatefulPartitionedCallStatefulPartitionedCallinputssequential_2_15404sequential_2_15406sequential_2_15408sequential_2_15410sequential_2_15412sequential_2_15414sequential_2_15416sequential_2_15418*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_151772&
$sequential_2/StatefulPartitionedCall?
&sequential_2/StatefulPartitionedCall_1StatefulPartitionedCallinputs_1sequential_2_15404sequential_2_15406sequential_2_15408sequential_2_15410sequential_2_15412sequential_2_15414sequential_2_15416sequential_2_15418*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_151772(
&sequential_2/StatefulPartitionedCall_1?
lambda_3/PartitionedCallPartitionedCall-sequential_2/StatefulPartitionedCall:output:0/sequential_2/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_lambda_3_layer_call_and_return_conditional_losses_153722
lambda_3/PartitionedCall?
IdentityIdentity!lambda_3/PartitionedCall:output:0%^sequential_2/StatefulPartitionedCall'^sequential_2/StatefulPartitionedCall_1*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:?????????8.:?????????8.: : : : : : : : 2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall2P
&sequential_2/StatefulPartitionedCall_1&sequential_2/StatefulPartitionedCall_1:W S
/
_output_shapes
:?????????8.
 
_user_specified_nameinputs:WS
/
_output_shapes
:?????????8.
 
_user_specified_nameinputs
?

?
#__inference_signature_wrapper_15567
input_5
input_6!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?2
	unknown_6:2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_5input_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__wrapped_model_148332
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:?????????8.:?????????8.: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????8.
!
_user_specified_name	input_5:XT
/
_output_shapes
:?????????8.
!
_user_specified_name	input_6
?

?
C__inference_conv2d_5_layer_call_and_return_conditional_losses_16053

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**
data_formatNCHW*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**
data_formatNCHW2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????*2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????,: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????,
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_14851

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?	
m
C__inference_lambda_3_layer_call_and_return_conditional_losses_15328

inputs
inputs_1
identityU
subSubinputsinputs_1*
T0*'
_output_shapes
:?????????22
subU
SquareSquaresub:z:0*
T0*'
_output_shapes
:?????????22
Squarep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices?
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2
SumS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Constm
MaximumMaximumSum:output:0Const:output:0*
T0*'
_output_shapes
:?????????2	
MaximumS
SqrtSqrtMaximum:z:0*
T0*'
_output_shapes
:?????????2
Sqrt\
IdentityIdentitySqrt:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????2:?????????2:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
b
D__inference_dropout_6_layer_call_and_return_conditional_losses_14893

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????,2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????,2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????,:W S
/
_output_shapes
:?????????,
 
_user_specified_nameinputs
?
T
(__inference_lambda_3_layer_call_fn_15948
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_lambda_3_layer_call_and_return_conditional_losses_153282
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????2:?????????2:Q M
'
_output_shapes
:?????????2
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????2
"
_user_specified_name
inputs/1
?
?
(__inference_conv2d_4_layer_call_fn_15987

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????6,*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_148742
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????6,2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????8.: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????8.
 
_user_specified_nameinputs
?
b
D__inference_dropout_8_layer_call_and_return_conditional_losses_16136

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
C
input_58
serving_default_input_5:0?????????8.
C
input_68
serving_default_input_6:0?????????8.<
lambda_30
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?f
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
	optimizer
trainable_variables
	variables
regularization_losses
		keras_api


signatures
?_default_save_signature
?__call__
+?&call_and_return_all_conditional_losses"?d
_tf_keras_network?d{"name": "model_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 56, 46]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}, "name": "input_5", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 56, 46]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_6"}, "name": "input_6", "inbound_nodes": []}, {"class_name": "Sequential", "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 56, 46]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_4_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 56, 46]}, "dtype": "float32", "filters": 6, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_6", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_5", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_7", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_8", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "name": "sequential_2", "inbound_nodes": [[["input_5", 0, 0, {}]], [["input_6", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "lambda_3", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAwAAAAcAAABDAAAAcygAAAB8AFwCfQF9AnQAoAF0AGoCdACgA3wBfAIYAKEBZAFk\nAmQDjQOhAVMAKQRO6QEAAABUKQLaBGF4aXPaCGtlZXBkaW1zKQTaAUvaBHNxcnTaA3N1bdoGc3F1\nYXJlKQPaBXZlY3Rz2gF42gF5qQByCwAAAPofPGlweXRob24taW5wdXQtMzMtNDQ3ZmY1OTVhYjIz\nPtoSZXVjbGlkZWFuX2Rpc3RhbmNlAQAAAHMEAAAAAAEIAQ==\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAwAAAAIAAABDAAAAcxQAAAB8AFwCfQF9AnwBZAEZAGQCZgJTACkDTukAAAAA6QEA\nAACpACkD2gZzaGFwZXPaBnNoYXBlMdoGc2hhcGUycgMAAAByAwAAAPofPGlweXRob24taW5wdXQt\nMzMtNDQ3ZmY1OTVhYjIzPtoWZXVjbF9kaXN0X291dHB1dF9zaGFwZQYAAABzBAAAAAABCAE=\n", null, null]}, "output_shape_type": "lambda", "output_shape_module": "__main__", "arguments": {}}, "name": "lambda_3", "inbound_nodes": [[["sequential_2", 1, 0, {}], ["sequential_2", 2, 0, {}]]]}], "input_layers": [["input_5", 0, 0], ["input_6", 0, 0]], "output_layers": [["lambda_3", 0, 0]]}, "shared_object_id": 25, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1, 56, 46]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1, 56, 46]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1, 56, 46]}, {"class_name": "TensorShape", "items": [null, 1, 56, 46]}], "is_graph_network": true, "save_spec": [{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1, 56, 46]}, "float32", "input_5"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1, 56, 46]}, "float32", "input_6"]}], "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 56, 46]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}, "name": "input_5", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 56, 46]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_6"}, "name": "input_6", "inbound_nodes": [], "shared_object_id": 1}, {"class_name": "Sequential", "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 56, 46]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_4_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 56, 46]}, "dtype": "float32", "filters": 6, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_6", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_5", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_7", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_8", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "name": "sequential_2", "inbound_nodes": [[["input_5", 0, 0, {}]], [["input_6", 0, 0, {}]]], "shared_object_id": 23}, {"class_name": "Lambda", "config": {"name": "lambda_3", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAwAAAAcAAABDAAAAcygAAAB8AFwCfQF9AnQAoAF0AGoCdACgA3wBfAIYAKEBZAFk\nAmQDjQOhAVMAKQRO6QEAAABUKQLaBGF4aXPaCGtlZXBkaW1zKQTaAUvaBHNxcnTaA3N1bdoGc3F1\nYXJlKQPaBXZlY3Rz2gF42gF5qQByCwAAAPofPGlweXRob24taW5wdXQtMzMtNDQ3ZmY1OTVhYjIz\nPtoSZXVjbGlkZWFuX2Rpc3RhbmNlAQAAAHMEAAAAAAEIAQ==\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAwAAAAIAAABDAAAAcxQAAAB8AFwCfQF9AnwBZAEZAGQCZgJTACkDTukAAAAA6QEA\nAACpACkD2gZzaGFwZXPaBnNoYXBlMdoGc2hhcGUycgMAAAByAwAAAPofPGlweXRob24taW5wdXQt\nMzMtNDQ3ZmY1OTVhYjIzPtoWZXVjbF9kaXN0X291dHB1dF9zaGFwZQYAAABzBAAAAAABCAE=\n", null, null]}, "output_shape_type": "lambda", "output_shape_module": "__main__", "arguments": {}}, "name": "lambda_3", "inbound_nodes": [[["sequential_2", 1, 0, {}], ["sequential_2", 2, 0, {}]]], "shared_object_id": 24}], "input_layers": [["input_5", 0, 0], ["input_6", 0, 0]], "output_layers": [["lambda_3", 0, 0]]}}, "training_config": {"loss": "contrastive_loss", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "RMSprop", "config": {"name": "RMSprop", "learning_rate": 0.0010000000474974513, "decay": 0.0, "rho": 0.8999999761581421, "momentum": 0.0, "epsilon": 1e-07, "centered": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_5", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 56, 46]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 56, 46]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_6", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 56, 46]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 56, 46]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_6"}}
?J
layer_with_weights-0
layer-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer-6
layer-7
layer-8
layer_with_weights-2
layer-9
layer-10
layer_with_weights-3
layer-11
trainable_variables
	variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?G
_tf_keras_sequential?G{"name": "sequential_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 56, 46]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_4_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 56, 46]}, "dtype": "float32", "filters": 6, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_6", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_5", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_7", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_8", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "inbound_nodes": [[["input_5", 0, 0, {}]], [["input_6", 0, 0, {}]]], "shared_object_id": 23, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 1}}, "shared_object_id": 28}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 56, 46]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1, 56, 46]}, "float32", "conv2d_4_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 56, 46]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_4_input"}, "shared_object_id": 2}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 56, 46]}, "dtype": "float32", "filters": 6, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 3}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 5}, {"class_name": "Activation", "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "relu"}, "shared_object_id": 6}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 7}, {"class_name": "Dropout", "config": {"name": "dropout_6", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "shared_object_id": 8}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 11}, {"class_name": "Activation", "config": {"name": "activation_5", "trainable": true, "dtype": "float32", "activation": "relu"}, "shared_object_id": 12}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 13}, {"class_name": "Dropout", "config": {"name": "dropout_7", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "shared_object_id": 14}, {"class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 15}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 16}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 17}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 18}, {"class_name": "Dropout", "config": {"name": "dropout_8", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "shared_object_id": 19}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 20}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 22}]}}}
?

trainable_variables
	variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "lambda_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Lambda", "config": {"name": "lambda_3", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAwAAAAcAAABDAAAAcygAAAB8AFwCfQF9AnQAoAF0AGoCdACgA3wBfAIYAKEBZAFk\nAmQDjQOhAVMAKQRO6QEAAABUKQLaBGF4aXPaCGtlZXBkaW1zKQTaAUvaBHNxcnTaA3N1bdoGc3F1\nYXJlKQPaBXZlY3Rz2gF42gF5qQByCwAAAPofPGlweXRob24taW5wdXQtMzMtNDQ3ZmY1OTVhYjIz\nPtoSZXVjbGlkZWFuX2Rpc3RhbmNlAQAAAHMEAAAAAAEIAQ==\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAwAAAAIAAABDAAAAcxQAAAB8AFwCfQF9AnwBZAEZAGQCZgJTACkDTukAAAAA6QEA\nAACpACkD2gZzaGFwZXPaBnNoYXBlMdoGc2hhcGUycgMAAAByAwAAAPofPGlweXRob24taW5wdXQt\nMzMtNDQ3ZmY1OTVhYjIzPtoWZXVjbF9kaXN0X291dHB1dF9zaGFwZQYAAABzBAAAAAABCAE=\n", null, null]}, "output_shape_type": "lambda", "output_shape_module": "__main__", "arguments": {}}, "inbound_nodes": [[["sequential_2", 1, 0, {}], ["sequential_2", 2, 0, {}]]], "shared_object_id": 24}
?
iter
	 decay
!learning_rate
"momentum
#rho
$rms?
%rms?
&rms?
'rms?
(rms?
)rms?
*rms?
+rms?"
	optimizer
X
$0
%1
&2
'3
(4
)5
*6
+7"
trackable_list_wrapper
X
$0
%1
&2
'3
(4
)5
*6
+7"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
,layer_metrics
-metrics
	variables
.layer_regularization_losses
regularization_losses

/layers
0non_trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?

$kernel
%bias
1trainable_variables
2	variables
3regularization_losses
4	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?

_tf_keras_layer?
{"name": "conv2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 56, 46]}, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 56, 46]}, "dtype": "float32", "filters": 6, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 3}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 5, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 1}}, "shared_object_id": 28}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 56, 46]}}
?
5trainable_variables
6	variables
7regularization_losses
8	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "activation_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "relu"}, "shared_object_id": 6}
?
9trainable_variables
:	variables
;regularization_losses
<	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "max_pooling2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 7, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 29}}
?
=trainable_variables
>	variables
?regularization_losses
@	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dropout_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_6", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "shared_object_id": 8}
?


&kernel
'bias
Atrainable_variables
B	variables
Cregularization_losses
D	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "conv2d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 11, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 3}}, "shared_object_id": 30}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 27, 44]}}
?
Etrainable_variables
F	variables
Gregularization_losses
H	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "activation_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_5", "trainable": true, "dtype": "float32", "activation": "relu"}, "shared_object_id": 12}
?
Itrainable_variables
J	variables
Kregularization_losses
L	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "max_pooling2d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 31}}
?
Mtrainable_variables
N	variables
Oregularization_losses
P	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dropout_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_7", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "shared_object_id": 14}
?
Qtrainable_variables
R	variables
Sregularization_losses
T	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "flatten_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 15, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 32}}
?

(kernel
)bias
Utrainable_variables
V	variables
Wregularization_losses
X	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 16}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 17}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 18, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 3024}}, "shared_object_id": 33}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3024]}}
?
Ytrainable_variables
Z	variables
[regularization_losses
\	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dropout_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_8", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "shared_object_id": 19}
?

*kernel
+bias
]trainable_variables
^	variables
_regularization_losses
`	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 20}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 22, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}, "shared_object_id": 34}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
X
$0
%1
&2
'3
(4
)5
*6
+7"
trackable_list_wrapper
X
$0
%1
&2
'3
(4
)5
*6
+7"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
alayer_metrics
bmetrics
	variables
clayer_regularization_losses
regularization_losses

dlayers
enon_trainable_variables
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
trainable_variables
flayer_metrics
gmetrics
	variables
hlayer_regularization_losses
regularization_losses

ilayers
jnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2RMSprop/iter
: (2RMSprop/decay
: (2RMSprop/learning_rate
: (2RMSprop/momentum
: (2RMSprop/rho
):'2conv2d_4/kernel
:2conv2d_4/bias
):'2conv2d_5/kernel
:2conv2d_5/bias
": 
??2dense_4/kernel
:?2dense_4/bias
!:	?22dense_5/kernel
:22dense_5/bias
 "
trackable_dict_wrapper
'
k0"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
1trainable_variables
llayer_metrics
mmetrics
2	variables
nlayer_regularization_losses
3regularization_losses

olayers
pnon_trainable_variables
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
5trainable_variables
qlayer_metrics
rmetrics
6	variables
slayer_regularization_losses
7regularization_losses

tlayers
unon_trainable_variables
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
9trainable_variables
vlayer_metrics
wmetrics
:	variables
xlayer_regularization_losses
;regularization_losses

ylayers
znon_trainable_variables
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
=trainable_variables
{layer_metrics
|metrics
>	variables
}layer_regularization_losses
?regularization_losses

~layers
non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Atrainable_variables
?layer_metrics
?metrics
B	variables
 ?layer_regularization_losses
Cregularization_losses
?layers
?non_trainable_variables
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
Etrainable_variables
?layer_metrics
?metrics
F	variables
 ?layer_regularization_losses
Gregularization_losses
?layers
?non_trainable_variables
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
Itrainable_variables
?layer_metrics
?metrics
J	variables
 ?layer_regularization_losses
Kregularization_losses
?layers
?non_trainable_variables
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
Mtrainable_variables
?layer_metrics
?metrics
N	variables
 ?layer_regularization_losses
Oregularization_losses
?layers
?non_trainable_variables
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
Qtrainable_variables
?layer_metrics
?metrics
R	variables
 ?layer_regularization_losses
Sregularization_losses
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Utrainable_variables
?layer_metrics
?metrics
V	variables
 ?layer_regularization_losses
Wregularization_losses
?layers
?non_trainable_variables
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
Ytrainable_variables
?layer_metrics
?metrics
Z	variables
 ?layer_regularization_losses
[regularization_losses
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
]trainable_variables
?layer_metrics
?metrics
^	variables
 ?layer_regularization_losses
_regularization_losses
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
v
0
1
2
3
4
5
6
7
8
9
10
11"
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
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 35}
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
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
3:12RMSprop/conv2d_4/kernel/rms
%:#2RMSprop/conv2d_4/bias/rms
3:12RMSprop/conv2d_5/kernel/rms
%:#2RMSprop/conv2d_5/bias/rms
,:*
??2RMSprop/dense_4/kernel/rms
%:#?2RMSprop/dense_4/bias/rms
+:)	?22RMSprop/dense_5/kernel/rms
$:"22RMSprop/dense_5/bias/rms
?2?
 __inference__wrapped_model_14833?
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
annotations? *^?[
Y?V
)?&
input_5?????????8.
)?&
input_6?????????8.
?2?
'__inference_model_3_layer_call_fn_15350
'__inference_model_3_layer_call_fn_15589
'__inference_model_3_layer_call_fn_15611
'__inference_model_3_layer_call_fn_15473?
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
B__inference_model_3_layer_call_and_return_conditional_losses_15685
B__inference_model_3_layer_call_and_return_conditional_losses_15801
B__inference_model_3_layer_call_and_return_conditional_losses_15505
B__inference_model_3_layer_call_and_return_conditional_losses_15537?
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
?2?
,__inference_sequential_2_layer_call_fn_14995
,__inference_sequential_2_layer_call_fn_15822
,__inference_sequential_2_layer_call_fn_15843
,__inference_sequential_2_layer_call_fn_15217?
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
G__inference_sequential_2_layer_call_and_return_conditional_losses_15882
G__inference_sequential_2_layer_call_and_return_conditional_losses_15942
G__inference_sequential_2_layer_call_and_return_conditional_losses_15249
G__inference_sequential_2_layer_call_and_return_conditional_losses_15281?
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
?2?
(__inference_lambda_3_layer_call_fn_15948
(__inference_lambda_3_layer_call_fn_15954?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_lambda_3_layer_call_and_return_conditional_losses_15966
C__inference_lambda_3_layer_call_and_return_conditional_losses_15978?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
#__inference_signature_wrapper_15567input_5input_6"?
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
 
?2?
(__inference_conv2d_4_layer_call_fn_15987?
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
C__inference_conv2d_4_layer_call_and_return_conditional_losses_15997?
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
,__inference_activation_4_layer_call_fn_16002?
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
G__inference_activation_4_layer_call_and_return_conditional_losses_16007?
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
/__inference_max_pooling2d_4_layer_call_fn_14845?
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
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_14839?
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
)__inference_dropout_6_layer_call_fn_16012
)__inference_dropout_6_layer_call_fn_16017?
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
D__inference_dropout_6_layer_call_and_return_conditional_losses_16022
D__inference_dropout_6_layer_call_and_return_conditional_losses_16034?
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
(__inference_conv2d_5_layer_call_fn_16043?
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
C__inference_conv2d_5_layer_call_and_return_conditional_losses_16053?
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
,__inference_activation_5_layer_call_fn_16058?
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
G__inference_activation_5_layer_call_and_return_conditional_losses_16063?
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
/__inference_max_pooling2d_5_layer_call_fn_14857?
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
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_14851?
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
)__inference_dropout_7_layer_call_fn_16068
)__inference_dropout_7_layer_call_fn_16073?
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
D__inference_dropout_7_layer_call_and_return_conditional_losses_16078
D__inference_dropout_7_layer_call_and_return_conditional_losses_16090?
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
)__inference_flatten_2_layer_call_fn_16095?
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
D__inference_flatten_2_layer_call_and_return_conditional_losses_16101?
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
'__inference_dense_4_layer_call_fn_16110?
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
B__inference_dense_4_layer_call_and_return_conditional_losses_16121?
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
)__inference_dropout_8_layer_call_fn_16126
)__inference_dropout_8_layer_call_fn_16131?
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
D__inference_dropout_8_layer_call_and_return_conditional_losses_16136
D__inference_dropout_8_layer_call_and_return_conditional_losses_16148?
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
'__inference_dense_5_layer_call_fn_16157?
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
B__inference_dense_5_layer_call_and_return_conditional_losses_16168?
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
 ?
 __inference__wrapped_model_14833?$%&'()*+h?e
^?[
Y?V
)?&
input_5?????????8.
)?&
input_6?????????8.
? "3?0
.
lambda_3"?
lambda_3??????????
G__inference_activation_4_layer_call_and_return_conditional_losses_16007h7?4
-?*
(?%
inputs?????????6,
? "-?*
#? 
0?????????6,
? ?
,__inference_activation_4_layer_call_fn_16002[7?4
-?*
(?%
inputs?????????6,
? " ??????????6,?
G__inference_activation_5_layer_call_and_return_conditional_losses_16063h7?4
-?*
(?%
inputs?????????*
? "-?*
#? 
0?????????*
? ?
,__inference_activation_5_layer_call_fn_16058[7?4
-?*
(?%
inputs?????????*
? " ??????????*?
C__inference_conv2d_4_layer_call_and_return_conditional_losses_15997l$%7?4
-?*
(?%
inputs?????????8.
? "-?*
#? 
0?????????6,
? ?
(__inference_conv2d_4_layer_call_fn_15987_$%7?4
-?*
(?%
inputs?????????8.
? " ??????????6,?
C__inference_conv2d_5_layer_call_and_return_conditional_losses_16053l&'7?4
-?*
(?%
inputs?????????,
? "-?*
#? 
0?????????*
? ?
(__inference_conv2d_5_layer_call_fn_16043_&'7?4
-?*
(?%
inputs?????????,
? " ??????????*?
B__inference_dense_4_layer_call_and_return_conditional_losses_16121^()0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? |
'__inference_dense_4_layer_call_fn_16110Q()0?-
&?#
!?
inputs??????????
? "????????????
B__inference_dense_5_layer_call_and_return_conditional_losses_16168]*+0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????2
? {
'__inference_dense_5_layer_call_fn_16157P*+0?-
&?#
!?
inputs??????????
? "??????????2?
D__inference_dropout_6_layer_call_and_return_conditional_losses_16022l;?8
1?.
(?%
inputs?????????,
p 
? "-?*
#? 
0?????????,
? ?
D__inference_dropout_6_layer_call_and_return_conditional_losses_16034l;?8
1?.
(?%
inputs?????????,
p
? "-?*
#? 
0?????????,
? ?
)__inference_dropout_6_layer_call_fn_16012_;?8
1?.
(?%
inputs?????????,
p 
? " ??????????,?
)__inference_dropout_6_layer_call_fn_16017_;?8
1?.
(?%
inputs?????????,
p
? " ??????????,?
D__inference_dropout_7_layer_call_and_return_conditional_losses_16078l;?8
1?.
(?%
inputs?????????*
p 
? "-?*
#? 
0?????????*
? ?
D__inference_dropout_7_layer_call_and_return_conditional_losses_16090l;?8
1?.
(?%
inputs?????????*
p
? "-?*
#? 
0?????????*
? ?
)__inference_dropout_7_layer_call_fn_16068_;?8
1?.
(?%
inputs?????????*
p 
? " ??????????*?
)__inference_dropout_7_layer_call_fn_16073_;?8
1?.
(?%
inputs?????????*
p
? " ??????????*?
D__inference_dropout_8_layer_call_and_return_conditional_losses_16136^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
D__inference_dropout_8_layer_call_and_return_conditional_losses_16148^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ~
)__inference_dropout_8_layer_call_fn_16126Q4?1
*?'
!?
inputs??????????
p 
? "???????????~
)__inference_dropout_8_layer_call_fn_16131Q4?1
*?'
!?
inputs??????????
p
? "????????????
D__inference_flatten_2_layer_call_and_return_conditional_losses_16101a7?4
-?*
(?%
inputs?????????*
? "&?#
?
0??????????
? ?
)__inference_flatten_2_layer_call_fn_16095T7?4
-?*
(?%
inputs?????????*
? "????????????
C__inference_lambda_3_layer_call_and_return_conditional_losses_15966?b?_
X?U
K?H
"?
inputs/0?????????2
"?
inputs/1?????????2

 
p 
? "%?"
?
0?????????
? ?
C__inference_lambda_3_layer_call_and_return_conditional_losses_15978?b?_
X?U
K?H
"?
inputs/0?????????2
"?
inputs/1?????????2

 
p
? "%?"
?
0?????????
? ?
(__inference_lambda_3_layer_call_fn_15948~b?_
X?U
K?H
"?
inputs/0?????????2
"?
inputs/1?????????2

 
p 
? "???????????
(__inference_lambda_3_layer_call_fn_15954~b?_
X?U
K?H
"?
inputs/0?????????2
"?
inputs/1?????????2

 
p
? "???????????
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_14839?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_max_pooling2d_4_layer_call_fn_14845?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_14851?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_max_pooling2d_5_layer_call_fn_14857?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
B__inference_model_3_layer_call_and_return_conditional_losses_15505?$%&'()*+p?m
f?c
Y?V
)?&
input_5?????????8.
)?&
input_6?????????8.
p 

 
? "%?"
?
0?????????
? ?
B__inference_model_3_layer_call_and_return_conditional_losses_15537?$%&'()*+p?m
f?c
Y?V
)?&
input_5?????????8.
)?&
input_6?????????8.
p

 
? "%?"
?
0?????????
? ?
B__inference_model_3_layer_call_and_return_conditional_losses_15685?$%&'()*+r?o
h?e
[?X
*?'
inputs/0?????????8.
*?'
inputs/1?????????8.
p 

 
? "%?"
?
0?????????
? ?
B__inference_model_3_layer_call_and_return_conditional_losses_15801?$%&'()*+r?o
h?e
[?X
*?'
inputs/0?????????8.
*?'
inputs/1?????????8.
p

 
? "%?"
?
0?????????
? ?
'__inference_model_3_layer_call_fn_15350?$%&'()*+p?m
f?c
Y?V
)?&
input_5?????????8.
)?&
input_6?????????8.
p 

 
? "???????????
'__inference_model_3_layer_call_fn_15473?$%&'()*+p?m
f?c
Y?V
)?&
input_5?????????8.
)?&
input_6?????????8.
p

 
? "???????????
'__inference_model_3_layer_call_fn_15589?$%&'()*+r?o
h?e
[?X
*?'
inputs/0?????????8.
*?'
inputs/1?????????8.
p 

 
? "???????????
'__inference_model_3_layer_call_fn_15611?$%&'()*+r?o
h?e
[?X
*?'
inputs/0?????????8.
*?'
inputs/1?????????8.
p

 
? "???????????
G__inference_sequential_2_layer_call_and_return_conditional_losses_15249z$%&'()*+G?D
=?:
0?-
conv2d_4_input?????????8.
p 

 
? "%?"
?
0?????????2
? ?
G__inference_sequential_2_layer_call_and_return_conditional_losses_15281z$%&'()*+G?D
=?:
0?-
conv2d_4_input?????????8.
p

 
? "%?"
?
0?????????2
? ?
G__inference_sequential_2_layer_call_and_return_conditional_losses_15882r$%&'()*+??<
5?2
(?%
inputs?????????8.
p 

 
? "%?"
?
0?????????2
? ?
G__inference_sequential_2_layer_call_and_return_conditional_losses_15942r$%&'()*+??<
5?2
(?%
inputs?????????8.
p

 
? "%?"
?
0?????????2
? ?
,__inference_sequential_2_layer_call_fn_14995m$%&'()*+G?D
=?:
0?-
conv2d_4_input?????????8.
p 

 
? "??????????2?
,__inference_sequential_2_layer_call_fn_15217m$%&'()*+G?D
=?:
0?-
conv2d_4_input?????????8.
p

 
? "??????????2?
,__inference_sequential_2_layer_call_fn_15822e$%&'()*+??<
5?2
(?%
inputs?????????8.
p 

 
? "??????????2?
,__inference_sequential_2_layer_call_fn_15843e$%&'()*+??<
5?2
(?%
inputs?????????8.
p

 
? "??????????2?
#__inference_signature_wrapper_15567?$%&'()*+y?v
? 
o?l
4
input_5)?&
input_5?????????8.
4
input_6)?&
input_6?????????8."3?0
.
lambda_3"?
lambda_3?????????