��
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
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
executor_typestring �
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�"serve*2.0.02v2.0.0-rc2-26-g64c3d382ca8��
�
conv2d_16/kernelVarHandleOp*
shape: *!
shared_nameconv2d_16/kernel*
dtype0*
_output_shapes
: 
}
$conv2d_16/kernel/Read/ReadVariableOpReadVariableOpconv2d_16/kernel*
dtype0*&
_output_shapes
: 
t
conv2d_16/biasVarHandleOp*
shape: *
shared_nameconv2d_16/bias*
dtype0*
_output_shapes
: 
m
"conv2d_16/bias/Read/ReadVariableOpReadVariableOpconv2d_16/bias*
dtype0*
_output_shapes
: 
�
conv2d_17/kernelVarHandleOp*
shape:  *!
shared_nameconv2d_17/kernel*
dtype0*
_output_shapes
: 
}
$conv2d_17/kernel/Read/ReadVariableOpReadVariableOpconv2d_17/kernel*
dtype0*&
_output_shapes
:  
t
conv2d_17/biasVarHandleOp*
shape: *
shared_nameconv2d_17/bias*
dtype0*
_output_shapes
: 
m
"conv2d_17/bias/Read/ReadVariableOpReadVariableOpconv2d_17/bias*
dtype0*
_output_shapes
: 
�
conv2d_18/kernelVarHandleOp*
shape: @*!
shared_nameconv2d_18/kernel*
dtype0*
_output_shapes
: 
}
$conv2d_18/kernel/Read/ReadVariableOpReadVariableOpconv2d_18/kernel*
dtype0*&
_output_shapes
: @
t
conv2d_18/biasVarHandleOp*
shape:@*
shared_nameconv2d_18/bias*
dtype0*
_output_shapes
: 
m
"conv2d_18/bias/Read/ReadVariableOpReadVariableOpconv2d_18/bias*
dtype0*
_output_shapes
:@
�
conv2d_19/kernelVarHandleOp*
shape:@@*!
shared_nameconv2d_19/kernel*
dtype0*
_output_shapes
: 
}
$conv2d_19/kernel/Read/ReadVariableOpReadVariableOpconv2d_19/kernel*
dtype0*&
_output_shapes
:@@
t
conv2d_19/biasVarHandleOp*
shape:@*
shared_nameconv2d_19/bias*
dtype0*
_output_shapes
: 
m
"conv2d_19/bias/Read/ReadVariableOpReadVariableOpconv2d_19/bias*
dtype0*
_output_shapes
:@
�
conv2d_20/kernelVarHandleOp*
shape:@@*!
shared_nameconv2d_20/kernel*
dtype0*
_output_shapes
: 
}
$conv2d_20/kernel/Read/ReadVariableOpReadVariableOpconv2d_20/kernel*
dtype0*&
_output_shapes
:@@
t
conv2d_20/biasVarHandleOp*
shape:@*
shared_nameconv2d_20/bias*
dtype0*
_output_shapes
: 
m
"conv2d_20/bias/Read/ReadVariableOpReadVariableOpconv2d_20/bias*
dtype0*
_output_shapes
:@
y
dense_4/kernelVarHandleOp*
shape:	@�*
shared_namedense_4/kernel*
dtype0*
_output_shapes
: 
r
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
dtype0*
_output_shapes
:	@�
q
dense_4/biasVarHandleOp*
shape:�*
shared_namedense_4/bias*
dtype0*
_output_shapes
: 
j
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
dtype0*
_output_shapes	
:�
y
dense_5/kernelVarHandleOp*
shape:	�&*
shared_namedense_5/kernel*
dtype0*
_output_shapes
: 
r
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
dtype0*
_output_shapes
:	�&
p
dense_5/biasVarHandleOp*
shape:&*
shared_namedense_5/bias*
dtype0*
_output_shapes
: 
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
dtype0*
_output_shapes
:&
f
	Adam/iterVarHandleOp*
shape: *
shared_name	Adam/iter*
dtype0	*
_output_shapes
: 
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
dtype0	*
_output_shapes
: 
j
Adam/beta_1VarHandleOp*
shape: *
shared_nameAdam/beta_1*
dtype0*
_output_shapes
: 
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
j
Adam/beta_2VarHandleOp*
shape: *
shared_nameAdam/beta_2*
dtype0*
_output_shapes
: 
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
h

Adam/decayVarHandleOp*
shape: *
shared_name
Adam/decay*
dtype0*
_output_shapes
: 
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
dtype0*
_output_shapes
: 
x
Adam/learning_rateVarHandleOp*
shape: *#
shared_nameAdam/learning_rate*
dtype0*
_output_shapes
: 
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
dtype0*
_output_shapes
: 
^
totalVarHandleOp*
shape: *
shared_nametotal*
dtype0*
_output_shapes
: 
W
total/Read/ReadVariableOpReadVariableOptotal*
dtype0*
_output_shapes
: 
^
countVarHandleOp*
shape: *
shared_namecount*
dtype0*
_output_shapes
: 
W
count/Read/ReadVariableOpReadVariableOpcount*
dtype0*
_output_shapes
: 
�
Adam/conv2d_16/kernel/mVarHandleOp*
shape: *(
shared_nameAdam/conv2d_16/kernel/m*
dtype0*
_output_shapes
: 
�
+Adam/conv2d_16/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_16/kernel/m*
dtype0*&
_output_shapes
: 
�
Adam/conv2d_16/bias/mVarHandleOp*
shape: *&
shared_nameAdam/conv2d_16/bias/m*
dtype0*
_output_shapes
: 
{
)Adam/conv2d_16/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_16/bias/m*
dtype0*
_output_shapes
: 
�
Adam/conv2d_17/kernel/mVarHandleOp*
shape:  *(
shared_nameAdam/conv2d_17/kernel/m*
dtype0*
_output_shapes
: 
�
+Adam/conv2d_17/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_17/kernel/m*
dtype0*&
_output_shapes
:  
�
Adam/conv2d_17/bias/mVarHandleOp*
shape: *&
shared_nameAdam/conv2d_17/bias/m*
dtype0*
_output_shapes
: 
{
)Adam/conv2d_17/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_17/bias/m*
dtype0*
_output_shapes
: 
�
Adam/conv2d_18/kernel/mVarHandleOp*
shape: @*(
shared_nameAdam/conv2d_18/kernel/m*
dtype0*
_output_shapes
: 
�
+Adam/conv2d_18/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_18/kernel/m*
dtype0*&
_output_shapes
: @
�
Adam/conv2d_18/bias/mVarHandleOp*
shape:@*&
shared_nameAdam/conv2d_18/bias/m*
dtype0*
_output_shapes
: 
{
)Adam/conv2d_18/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_18/bias/m*
dtype0*
_output_shapes
:@
�
Adam/conv2d_19/kernel/mVarHandleOp*
shape:@@*(
shared_nameAdam/conv2d_19/kernel/m*
dtype0*
_output_shapes
: 
�
+Adam/conv2d_19/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_19/kernel/m*
dtype0*&
_output_shapes
:@@
�
Adam/conv2d_19/bias/mVarHandleOp*
shape:@*&
shared_nameAdam/conv2d_19/bias/m*
dtype0*
_output_shapes
: 
{
)Adam/conv2d_19/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_19/bias/m*
dtype0*
_output_shapes
:@
�
Adam/conv2d_20/kernel/mVarHandleOp*
shape:@@*(
shared_nameAdam/conv2d_20/kernel/m*
dtype0*
_output_shapes
: 
�
+Adam/conv2d_20/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_20/kernel/m*
dtype0*&
_output_shapes
:@@
�
Adam/conv2d_20/bias/mVarHandleOp*
shape:@*&
shared_nameAdam/conv2d_20/bias/m*
dtype0*
_output_shapes
: 
{
)Adam/conv2d_20/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_20/bias/m*
dtype0*
_output_shapes
:@
�
Adam/dense_4/kernel/mVarHandleOp*
shape:	@�*&
shared_nameAdam/dense_4/kernel/m*
dtype0*
_output_shapes
: 
�
)Adam/dense_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/m*
dtype0*
_output_shapes
:	@�

Adam/dense_4/bias/mVarHandleOp*
shape:�*$
shared_nameAdam/dense_4/bias/m*
dtype0*
_output_shapes
: 
x
'Adam/dense_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/m*
dtype0*
_output_shapes	
:�
�
Adam/dense_5/kernel/mVarHandleOp*
shape:	�&*&
shared_nameAdam/dense_5/kernel/m*
dtype0*
_output_shapes
: 
�
)Adam/dense_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/m*
dtype0*
_output_shapes
:	�&
~
Adam/dense_5/bias/mVarHandleOp*
shape:&*$
shared_nameAdam/dense_5/bias/m*
dtype0*
_output_shapes
: 
w
'Adam/dense_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/m*
dtype0*
_output_shapes
:&
�
Adam/conv2d_16/kernel/vVarHandleOp*
shape: *(
shared_nameAdam/conv2d_16/kernel/v*
dtype0*
_output_shapes
: 
�
+Adam/conv2d_16/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_16/kernel/v*
dtype0*&
_output_shapes
: 
�
Adam/conv2d_16/bias/vVarHandleOp*
shape: *&
shared_nameAdam/conv2d_16/bias/v*
dtype0*
_output_shapes
: 
{
)Adam/conv2d_16/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_16/bias/v*
dtype0*
_output_shapes
: 
�
Adam/conv2d_17/kernel/vVarHandleOp*
shape:  *(
shared_nameAdam/conv2d_17/kernel/v*
dtype0*
_output_shapes
: 
�
+Adam/conv2d_17/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_17/kernel/v*
dtype0*&
_output_shapes
:  
�
Adam/conv2d_17/bias/vVarHandleOp*
shape: *&
shared_nameAdam/conv2d_17/bias/v*
dtype0*
_output_shapes
: 
{
)Adam/conv2d_17/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_17/bias/v*
dtype0*
_output_shapes
: 
�
Adam/conv2d_18/kernel/vVarHandleOp*
shape: @*(
shared_nameAdam/conv2d_18/kernel/v*
dtype0*
_output_shapes
: 
�
+Adam/conv2d_18/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_18/kernel/v*
dtype0*&
_output_shapes
: @
�
Adam/conv2d_18/bias/vVarHandleOp*
shape:@*&
shared_nameAdam/conv2d_18/bias/v*
dtype0*
_output_shapes
: 
{
)Adam/conv2d_18/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_18/bias/v*
dtype0*
_output_shapes
:@
�
Adam/conv2d_19/kernel/vVarHandleOp*
shape:@@*(
shared_nameAdam/conv2d_19/kernel/v*
dtype0*
_output_shapes
: 
�
+Adam/conv2d_19/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_19/kernel/v*
dtype0*&
_output_shapes
:@@
�
Adam/conv2d_19/bias/vVarHandleOp*
shape:@*&
shared_nameAdam/conv2d_19/bias/v*
dtype0*
_output_shapes
: 
{
)Adam/conv2d_19/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_19/bias/v*
dtype0*
_output_shapes
:@
�
Adam/conv2d_20/kernel/vVarHandleOp*
shape:@@*(
shared_nameAdam/conv2d_20/kernel/v*
dtype0*
_output_shapes
: 
�
+Adam/conv2d_20/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_20/kernel/v*
dtype0*&
_output_shapes
:@@
�
Adam/conv2d_20/bias/vVarHandleOp*
shape:@*&
shared_nameAdam/conv2d_20/bias/v*
dtype0*
_output_shapes
: 
{
)Adam/conv2d_20/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_20/bias/v*
dtype0*
_output_shapes
:@
�
Adam/dense_4/kernel/vVarHandleOp*
shape:	@�*&
shared_nameAdam/dense_4/kernel/v*
dtype0*
_output_shapes
: 
�
)Adam/dense_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/v*
dtype0*
_output_shapes
:	@�

Adam/dense_4/bias/vVarHandleOp*
shape:�*$
shared_nameAdam/dense_4/bias/v*
dtype0*
_output_shapes
: 
x
'Adam/dense_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/v*
dtype0*
_output_shapes	
:�
�
Adam/dense_5/kernel/vVarHandleOp*
shape:	�&*&
shared_nameAdam/dense_5/kernel/v*
dtype0*
_output_shapes
: 
�
)Adam/dense_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/v*
dtype0*
_output_shapes
:	�&
~
Adam/dense_5/bias/vVarHandleOp*
shape:&*$
shared_nameAdam/dense_5/bias/v*
dtype0*
_output_shapes
: 
w
'Adam/dense_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/v*
dtype0*
_output_shapes
:&

NoOpNoOp
�l
ConstConst"/device:CPU:0*�l
value�lB�l B�l
�
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer-8

layer-9
layer_with_weights-3
layer-10
layer-11
layer-12
layer-13
layer_with_weights-4
layer-14
layer-15
layer-16
layer-17
layer-18
layer_with_weights-5
layer-19
layer-20
layer-21
layer_with_weights-6
layer-22
layer-23
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
R
	variables
 regularization_losses
!trainable_variables
"	keras_api
h

#kernel
$bias
%	variables
&regularization_losses
'trainable_variables
(	keras_api
R
)	variables
*regularization_losses
+trainable_variables
,	keras_api
h

-kernel
.bias
/	variables
0regularization_losses
1trainable_variables
2	keras_api
R
3	variables
4regularization_losses
5trainable_variables
6	keras_api
R
7	variables
8regularization_losses
9trainable_variables
:	keras_api
h

;kernel
<bias
=	variables
>regularization_losses
?trainable_variables
@	keras_api
R
A	variables
Bregularization_losses
Ctrainable_variables
D	keras_api
R
E	variables
Fregularization_losses
Gtrainable_variables
H	keras_api
R
I	variables
Jregularization_losses
Ktrainable_variables
L	keras_api
h

Mkernel
Nbias
O	variables
Pregularization_losses
Qtrainable_variables
R	keras_api
R
S	variables
Tregularization_losses
Utrainable_variables
V	keras_api
R
W	variables
Xregularization_losses
Ytrainable_variables
Z	keras_api
R
[	variables
\regularization_losses
]trainable_variables
^	keras_api
h

_kernel
`bias
a	variables
bregularization_losses
ctrainable_variables
d	keras_api
R
e	variables
fregularization_losses
gtrainable_variables
h	keras_api
R
i	variables
jregularization_losses
ktrainable_variables
l	keras_api
R
m	variables
nregularization_losses
otrainable_variables
p	keras_api
R
q	variables
rregularization_losses
strainable_variables
t	keras_api
h

ukernel
vbias
w	variables
xregularization_losses
ytrainable_variables
z	keras_api
R
{	variables
|regularization_losses
}trainable_variables
~	keras_api
U
	variables
�regularization_losses
�trainable_variables
�	keras_api
n
�kernel
	�bias
�	variables
�regularization_losses
�trainable_variables
�	keras_api
V
�	variables
�regularization_losses
�trainable_variables
�	keras_api
�
	�iter
�beta_1
�beta_2

�decay
�learning_rate#m�$m�-m�.m�;m�<m�Mm�Nm�_m�`m�um�vm�	�m�	�m�#v�$v�-v�.v�;v�<v�Mv�Nv�_v�`v�uv�vv�	�v�	�v�
h
#0
$1
-2
.3
;4
<5
M6
N7
_8
`9
u10
v11
�12
�13
 
h
#0
$1
-2
.3
;4
<5
M6
N7
_8
`9
u10
v11
�12
�13
�
�metrics
	variables
regularization_losses
 �layer_regularization_losses
�non_trainable_variables
�layers
trainable_variables
 
 
 
 
�
�metrics
	variables
 regularization_losses
 �layer_regularization_losses
�non_trainable_variables
�layers
!trainable_variables
\Z
VARIABLE_VALUEconv2d_16/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_16/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

#0
$1
 

#0
$1
�
�metrics
%	variables
&regularization_losses
 �layer_regularization_losses
�non_trainable_variables
�layers
'trainable_variables
 
 
 
�
�metrics
)	variables
*regularization_losses
 �layer_regularization_losses
�non_trainable_variables
�layers
+trainable_variables
\Z
VARIABLE_VALUEconv2d_17/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_17/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

-0
.1
 

-0
.1
�
�metrics
/	variables
0regularization_losses
 �layer_regularization_losses
�non_trainable_variables
�layers
1trainable_variables
 
 
 
�
�metrics
3	variables
4regularization_losses
 �layer_regularization_losses
�non_trainable_variables
�layers
5trainable_variables
 
 
 
�
�metrics
7	variables
8regularization_losses
 �layer_regularization_losses
�non_trainable_variables
�layers
9trainable_variables
\Z
VARIABLE_VALUEconv2d_18/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_18/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

;0
<1
 

;0
<1
�
�metrics
=	variables
>regularization_losses
 �layer_regularization_losses
�non_trainable_variables
�layers
?trainable_variables
 
 
 
�
�metrics
A	variables
Bregularization_losses
 �layer_regularization_losses
�non_trainable_variables
�layers
Ctrainable_variables
 
 
 
�
�metrics
E	variables
Fregularization_losses
 �layer_regularization_losses
�non_trainable_variables
�layers
Gtrainable_variables
 
 
 
�
�metrics
I	variables
Jregularization_losses
 �layer_regularization_losses
�non_trainable_variables
�layers
Ktrainable_variables
\Z
VARIABLE_VALUEconv2d_19/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_19/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

M0
N1
 

M0
N1
�
�metrics
O	variables
Pregularization_losses
 �layer_regularization_losses
�non_trainable_variables
�layers
Qtrainable_variables
 
 
 
�
�metrics
S	variables
Tregularization_losses
 �layer_regularization_losses
�non_trainable_variables
�layers
Utrainable_variables
 
 
 
�
�metrics
W	variables
Xregularization_losses
 �layer_regularization_losses
�non_trainable_variables
�layers
Ytrainable_variables
 
 
 
�
�metrics
[	variables
\regularization_losses
 �layer_regularization_losses
�non_trainable_variables
�layers
]trainable_variables
\Z
VARIABLE_VALUEconv2d_20/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_20/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

_0
`1
 

_0
`1
�
�metrics
a	variables
bregularization_losses
 �layer_regularization_losses
�non_trainable_variables
�layers
ctrainable_variables
 
 
 
�
�metrics
e	variables
fregularization_losses
 �layer_regularization_losses
�non_trainable_variables
�layers
gtrainable_variables
 
 
 
�
�metrics
i	variables
jregularization_losses
 �layer_regularization_losses
�non_trainable_variables
�layers
ktrainable_variables
 
 
 
�
�metrics
m	variables
nregularization_losses
 �layer_regularization_losses
�non_trainable_variables
�layers
otrainable_variables
 
 
 
�
�metrics
q	variables
rregularization_losses
 �layer_regularization_losses
�non_trainable_variables
�layers
strainable_variables
ZX
VARIABLE_VALUEdense_4/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_4/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

u0
v1
 

u0
v1
�
�metrics
w	variables
xregularization_losses
 �layer_regularization_losses
�non_trainable_variables
�layers
ytrainable_variables
 
 
 
�
�metrics
{	variables
|regularization_losses
 �layer_regularization_losses
�non_trainable_variables
�layers
}trainable_variables
 
 
 
�
�metrics
	variables
�regularization_losses
 �layer_regularization_losses
�non_trainable_variables
�layers
�trainable_variables
ZX
VARIABLE_VALUEdense_5/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_5/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

�0
�1
 

�0
�1
�
�metrics
�	variables
�regularization_losses
 �layer_regularization_losses
�non_trainable_variables
�layers
�trainable_variables
 
 
 
�
�metrics
�	variables
�regularization_losses
 �layer_regularization_losses
�non_trainable_variables
�layers
�trainable_variables
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

�0
 
 
�
0
1
2
3
4
5
6
	7

8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 


�total

�count
�
_fn_kwargs
�	variables
�regularization_losses
�trainable_variables
�	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1
 
 
�
�metrics
�	variables
�regularization_losses
 �layer_regularization_losses
�non_trainable_variables
�layers
�trainable_variables
 
 

�0
�1
 
}
VARIABLE_VALUEAdam/conv2d_16/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_16/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_17/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_17/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_18/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_18/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_19/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_19/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_20/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_20/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_4/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_4/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_5/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_5/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_16/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_16/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_17/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_17/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_18/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_18/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_19/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_19/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_20/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_20/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_4/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_4/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_5/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_5/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
: 
�
serving_default_conv2d_16_inputPlaceholder*$
shape:���������22*
dtype0*/
_output_shapes
:���������22
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_16_inputconv2d_16/kernelconv2d_16/biasconv2d_17/kernelconv2d_17/biasconv2d_18/kernelconv2d_18/biasconv2d_19/kernelconv2d_19/biasconv2d_20/kernelconv2d_20/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/bias*-
_gradient_op_typePartitionedCall-123580*-
f(R&
$__inference_signature_wrapper_122991*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:���������&
O
saver_filenamePlaceholder*
shape: *
dtype0*
_output_shapes
: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_16/kernel/Read/ReadVariableOp"conv2d_16/bias/Read/ReadVariableOp$conv2d_17/kernel/Read/ReadVariableOp"conv2d_17/bias/Read/ReadVariableOp$conv2d_18/kernel/Read/ReadVariableOp"conv2d_18/bias/Read/ReadVariableOp$conv2d_19/kernel/Read/ReadVariableOp"conv2d_19/bias/Read/ReadVariableOp$conv2d_20/kernel/Read/ReadVariableOp"conv2d_20/bias/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/conv2d_16/kernel/m/Read/ReadVariableOp)Adam/conv2d_16/bias/m/Read/ReadVariableOp+Adam/conv2d_17/kernel/m/Read/ReadVariableOp)Adam/conv2d_17/bias/m/Read/ReadVariableOp+Adam/conv2d_18/kernel/m/Read/ReadVariableOp)Adam/conv2d_18/bias/m/Read/ReadVariableOp+Adam/conv2d_19/kernel/m/Read/ReadVariableOp)Adam/conv2d_19/bias/m/Read/ReadVariableOp+Adam/conv2d_20/kernel/m/Read/ReadVariableOp)Adam/conv2d_20/bias/m/Read/ReadVariableOp)Adam/dense_4/kernel/m/Read/ReadVariableOp'Adam/dense_4/bias/m/Read/ReadVariableOp)Adam/dense_5/kernel/m/Read/ReadVariableOp'Adam/dense_5/bias/m/Read/ReadVariableOp+Adam/conv2d_16/kernel/v/Read/ReadVariableOp)Adam/conv2d_16/bias/v/Read/ReadVariableOp+Adam/conv2d_17/kernel/v/Read/ReadVariableOp)Adam/conv2d_17/bias/v/Read/ReadVariableOp+Adam/conv2d_18/kernel/v/Read/ReadVariableOp)Adam/conv2d_18/bias/v/Read/ReadVariableOp+Adam/conv2d_19/kernel/v/Read/ReadVariableOp)Adam/conv2d_19/bias/v/Read/ReadVariableOp+Adam/conv2d_20/kernel/v/Read/ReadVariableOp)Adam/conv2d_20/bias/v/Read/ReadVariableOp)Adam/dense_4/kernel/v/Read/ReadVariableOp'Adam/dense_4/bias/v/Read/ReadVariableOp)Adam/dense_5/kernel/v/Read/ReadVariableOp'Adam/dense_5/bias/v/Read/ReadVariableOpConst*-
_gradient_op_typePartitionedCall-123651*(
f#R!
__inference__traced_save_123650*
Tout
2**
config_proto

CPU

GPU 2J 8*>
Tin7
523	*
_output_shapes
: 
�	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_16/kernelconv2d_16/biasconv2d_17/kernelconv2d_17/biasconv2d_18/kernelconv2d_18/biasconv2d_19/kernelconv2d_19/biasconv2d_20/kernelconv2d_20/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/conv2d_16/kernel/mAdam/conv2d_16/bias/mAdam/conv2d_17/kernel/mAdam/conv2d_17/bias/mAdam/conv2d_18/kernel/mAdam/conv2d_18/bias/mAdam/conv2d_19/kernel/mAdam/conv2d_19/bias/mAdam/conv2d_20/kernel/mAdam/conv2d_20/bias/mAdam/dense_4/kernel/mAdam/dense_4/bias/mAdam/dense_5/kernel/mAdam/dense_5/bias/mAdam/conv2d_16/kernel/vAdam/conv2d_16/bias/vAdam/conv2d_17/kernel/vAdam/conv2d_17/bias/vAdam/conv2d_18/kernel/vAdam/conv2d_18/bias/vAdam/conv2d_19/kernel/vAdam/conv2d_19/bias/vAdam/conv2d_20/kernel/vAdam/conv2d_20/bias/vAdam/dense_4/kernel/vAdam/dense_4/bias/vAdam/dense_5/kernel/vAdam/dense_5/bias/v*-
_gradient_op_typePartitionedCall-123811*+
f&R$
"__inference__traced_restore_123810*
Tout
2**
config_proto

CPU

GPU 2J 8*=
Tin6
422*
_output_shapes
: ��
�
e
F__inference_dropout_13_layer_call_and_return_conditional_losses_123363

inputs
identity�Q
dropout/rateConst*
valueB
 *  �>*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: _
dropout/random_uniform/maxConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*/
_output_shapes
:���������@�
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: �
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:���������@�
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:���������@R
dropout/sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: �
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*/
_output_shapes
:���������@i
dropout/mulMulinputsdropout/truediv:z:0*
T0*/
_output_shapes
:���������@w
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*/
_output_shapes
:���������@q
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:���������@a
IdentityIdentitydropout/mul_1:z:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�
e
I__inference_activation_21_layer_call_and_return_conditional_losses_122454

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:���������@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�	
�
C__inference_dense_4_layer_call_and_return_conditional_losses_123399

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@�j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:�w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*.
_input_shapes
:���������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
�
G
+__inference_dropout_14_layer_call_fn_123451

inputs
identity�
PartitionedCallPartitionedCallinputs*-
_gradient_op_typePartitionedCall-122747*O
fJRH
F__inference_dropout_14_layer_call_and_return_conditional_losses_122735*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*(
_output_shapes
:����������a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*'
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�
a
E__inference_flatten_2_layer_call_and_return_conditional_losses_122650

inputs
identity^
Reshape/shapeConst*
valueB"����@   *
dtype0*
_output_shapes
:d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:���������@X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�[
�
__inference__traced_save_123650
file_prefix/
+savev2_conv2d_16_kernel_read_readvariableop-
)savev2_conv2d_16_bias_read_readvariableop/
+savev2_conv2d_17_kernel_read_readvariableop-
)savev2_conv2d_17_bias_read_readvariableop/
+savev2_conv2d_18_kernel_read_readvariableop-
)savev2_conv2d_18_bias_read_readvariableop/
+savev2_conv2d_19_kernel_read_readvariableop-
)savev2_conv2d_19_bias_read_readvariableop/
+savev2_conv2d_20_kernel_read_readvariableop-
)savev2_conv2d_20_bias_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_conv2d_16_kernel_m_read_readvariableop4
0savev2_adam_conv2d_16_bias_m_read_readvariableop6
2savev2_adam_conv2d_17_kernel_m_read_readvariableop4
0savev2_adam_conv2d_17_bias_m_read_readvariableop6
2savev2_adam_conv2d_18_kernel_m_read_readvariableop4
0savev2_adam_conv2d_18_bias_m_read_readvariableop6
2savev2_adam_conv2d_19_kernel_m_read_readvariableop4
0savev2_adam_conv2d_19_bias_m_read_readvariableop6
2savev2_adam_conv2d_20_kernel_m_read_readvariableop4
0savev2_adam_conv2d_20_bias_m_read_readvariableop4
0savev2_adam_dense_4_kernel_m_read_readvariableop2
.savev2_adam_dense_4_bias_m_read_readvariableop4
0savev2_adam_dense_5_kernel_m_read_readvariableop2
.savev2_adam_dense_5_bias_m_read_readvariableop6
2savev2_adam_conv2d_16_kernel_v_read_readvariableop4
0savev2_adam_conv2d_16_bias_v_read_readvariableop6
2savev2_adam_conv2d_17_kernel_v_read_readvariableop4
0savev2_adam_conv2d_17_bias_v_read_readvariableop6
2savev2_adam_conv2d_18_kernel_v_read_readvariableop4
0savev2_adam_conv2d_18_bias_v_read_readvariableop6
2savev2_adam_conv2d_19_kernel_v_read_readvariableop4
0savev2_adam_conv2d_19_bias_v_read_readvariableop6
2savev2_adam_conv2d_20_kernel_v_read_readvariableop4
0savev2_adam_conv2d_20_bias_v_read_readvariableop4
0savev2_adam_dense_4_kernel_v_read_readvariableop2
.savev2_adam_dense_4_bias_v_read_readvariableop4
0savev2_adam_dense_5_kernel_v_read_readvariableop2
.savev2_adam_dense_5_bias_v_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2�SaveV2_1�
StringJoin/inputs_1Const"/device:CPU:0*<
value3B1 B+_temp_b020f91165274bb58fe6ae4c46fd872b/part*
dtype0*
_output_shapes
: s

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: L

num_shardsConst*
value	B :*
dtype0*
_output_shapes
: f
ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*�
value�B�1B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:1�
SaveV2/shape_and_slicesConst"/device:CPU:0*u
valuelBj1B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:1�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_16_kernel_read_readvariableop)savev2_conv2d_16_bias_read_readvariableop+savev2_conv2d_17_kernel_read_readvariableop)savev2_conv2d_17_bias_read_readvariableop+savev2_conv2d_18_kernel_read_readvariableop)savev2_conv2d_18_bias_read_readvariableop+savev2_conv2d_19_kernel_read_readvariableop)savev2_conv2d_19_bias_read_readvariableop+savev2_conv2d_20_kernel_read_readvariableop)savev2_conv2d_20_bias_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_conv2d_16_kernel_m_read_readvariableop0savev2_adam_conv2d_16_bias_m_read_readvariableop2savev2_adam_conv2d_17_kernel_m_read_readvariableop0savev2_adam_conv2d_17_bias_m_read_readvariableop2savev2_adam_conv2d_18_kernel_m_read_readvariableop0savev2_adam_conv2d_18_bias_m_read_readvariableop2savev2_adam_conv2d_19_kernel_m_read_readvariableop0savev2_adam_conv2d_19_bias_m_read_readvariableop2savev2_adam_conv2d_20_kernel_m_read_readvariableop0savev2_adam_conv2d_20_bias_m_read_readvariableop0savev2_adam_dense_4_kernel_m_read_readvariableop.savev2_adam_dense_4_bias_m_read_readvariableop0savev2_adam_dense_5_kernel_m_read_readvariableop.savev2_adam_dense_5_bias_m_read_readvariableop2savev2_adam_conv2d_16_kernel_v_read_readvariableop0savev2_adam_conv2d_16_bias_v_read_readvariableop2savev2_adam_conv2d_17_kernel_v_read_readvariableop0savev2_adam_conv2d_17_bias_v_read_readvariableop2savev2_adam_conv2d_18_kernel_v_read_readvariableop0savev2_adam_conv2d_18_bias_v_read_readvariableop2savev2_adam_conv2d_19_kernel_v_read_readvariableop0savev2_adam_conv2d_19_bias_v_read_readvariableop2savev2_adam_conv2d_20_kernel_v_read_readvariableop0savev2_adam_conv2d_20_bias_v_read_readvariableop0savev2_adam_dense_4_kernel_v_read_readvariableop.savev2_adam_dense_4_bias_v_read_readvariableop0savev2_adam_dense_5_kernel_v_read_readvariableop.savev2_adam_dense_5_bias_v_read_readvariableop"/device:CPU:0*?
dtypes5
321	*
_output_shapes
 h
ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: �
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:q
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:�
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
dtypes
2*
_output_shapes
 �
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
T0*
N*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*�
_input_shapes�
�: : : :  : : @:@:@@:@:@@:@:	@�:�:	�&:&: : : : : : : : : :  : : @:@:@@:@:@@:@:	@�:�:	�&:&: : :  : : @:@:@@:@:@@:@:	@�:�:	�&:&: 2
SaveV2SaveV22(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2_1SaveV2_1: : :0 :# : :	 :+ : :+ '
%
_user_specified_namefile_prefix:" : : :* :% : : :2 :- : : :$ : : :, : :
 : :' : : :/ : : : :& : : :. : : :! : : :) : : :1 :  : : :( 
�
�
(__inference_dense_4_layer_call_fn_123406

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-122679*L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_122673*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*(
_output_shapes
:�����������
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*.
_input_shapes
:���������@::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
�
e
F__inference_dropout_11_layer_call_and_return_conditional_losses_122488

inputs
identity�Q
dropout/rateConst*
valueB
 *  �>*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: _
dropout/random_uniform/maxConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*/
_output_shapes
:���������

@�
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: �
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:���������

@�
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:���������

@R
dropout/sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: �
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*/
_output_shapes
:���������

@i
dropout/mulMulinputsdropout/truediv:z:0*
T0*/
_output_shapes
:���������

@w
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*/
_output_shapes
:���������

@q
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:���������

@a
IdentityIdentitydropout/mul_1:z:0*
T0*/
_output_shapes
:���������

@"
identityIdentity:output:0*.
_input_shapes
:���������

@:& "
 
_user_specified_nameinputs
�
e
F__inference_dropout_14_layer_call_and_return_conditional_losses_122728

inputs
identity�Q
dropout/rateConst*
valueB
 *   ?*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: _
dropout/random_uniform/maxConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*(
_output_shapes
:�����������
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: �
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:�����������
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:����������R
dropout/sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: �
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*(
_output_shapes
:����������b
dropout/mulMulinputsdropout/truediv:z:0*
T0*(
_output_shapes
:����������p
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*(
_output_shapes
:����������j
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������Z
IdentityIdentitydropout/mul_1:z:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*'
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�
�
(__inference_dense_5_layer_call_fn_123468

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-122768*L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_122762*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:���������&�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������&"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
��
�
"__inference__traced_restore_123810
file_prefix%
!assignvariableop_conv2d_16_kernel%
!assignvariableop_1_conv2d_16_bias'
#assignvariableop_2_conv2d_17_kernel%
!assignvariableop_3_conv2d_17_bias'
#assignvariableop_4_conv2d_18_kernel%
!assignvariableop_5_conv2d_18_bias'
#assignvariableop_6_conv2d_19_kernel%
!assignvariableop_7_conv2d_19_bias'
#assignvariableop_8_conv2d_20_kernel%
!assignvariableop_9_conv2d_20_bias&
"assignvariableop_10_dense_4_kernel$
 assignvariableop_11_dense_4_bias&
"assignvariableop_12_dense_5_kernel$
 assignvariableop_13_dense_5_bias!
assignvariableop_14_adam_iter#
assignvariableop_15_adam_beta_1#
assignvariableop_16_adam_beta_2"
assignvariableop_17_adam_decay*
&assignvariableop_18_adam_learning_rate
assignvariableop_19_total
assignvariableop_20_count/
+assignvariableop_21_adam_conv2d_16_kernel_m-
)assignvariableop_22_adam_conv2d_16_bias_m/
+assignvariableop_23_adam_conv2d_17_kernel_m-
)assignvariableop_24_adam_conv2d_17_bias_m/
+assignvariableop_25_adam_conv2d_18_kernel_m-
)assignvariableop_26_adam_conv2d_18_bias_m/
+assignvariableop_27_adam_conv2d_19_kernel_m-
)assignvariableop_28_adam_conv2d_19_bias_m/
+assignvariableop_29_adam_conv2d_20_kernel_m-
)assignvariableop_30_adam_conv2d_20_bias_m-
)assignvariableop_31_adam_dense_4_kernel_m+
'assignvariableop_32_adam_dense_4_bias_m-
)assignvariableop_33_adam_dense_5_kernel_m+
'assignvariableop_34_adam_dense_5_bias_m/
+assignvariableop_35_adam_conv2d_16_kernel_v-
)assignvariableop_36_adam_conv2d_16_bias_v/
+assignvariableop_37_adam_conv2d_17_kernel_v-
)assignvariableop_38_adam_conv2d_17_bias_v/
+assignvariableop_39_adam_conv2d_18_kernel_v-
)assignvariableop_40_adam_conv2d_18_bias_v/
+assignvariableop_41_adam_conv2d_19_kernel_v-
)assignvariableop_42_adam_conv2d_19_bias_v/
+assignvariableop_43_adam_conv2d_20_kernel_v-
)assignvariableop_44_adam_conv2d_20_bias_v-
)assignvariableop_45_adam_dense_4_kernel_v+
'assignvariableop_46_adam_dense_4_bias_v-
)assignvariableop_47_adam_dense_5_kernel_v+
'assignvariableop_48_adam_dense_5_bias_v
identity_50��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�	RestoreV2�RestoreV2_1�
RestoreV2/tensor_namesConst"/device:CPU:0*�
value�B�1B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:1�
RestoreV2/shape_and_slicesConst"/device:CPU:0*u
valuelBj1B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:1�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
dtypes5
321	*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::L
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:}
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_16_kernelIdentity:output:0*
dtype0*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_16_biasIdentity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_17_kernelIdentity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_17_biasIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_18_kernelIdentity_4:output:0*
dtype0*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_18_biasIdentity_5:output:0*
dtype0*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_19_kernelIdentity_6:output:0*
dtype0*
_output_shapes
 N

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_19_biasIdentity_7:output:0*
dtype0*
_output_shapes
 N

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp#assignvariableop_8_conv2d_20_kernelIdentity_8:output:0*
dtype0*
_output_shapes
 N

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp!assignvariableop_9_conv2d_20_biasIdentity_9:output:0*
dtype0*
_output_shapes
 P
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_4_kernelIdentity_10:output:0*
dtype0*
_output_shapes
 P
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_4_biasIdentity_11:output:0*
dtype0*
_output_shapes
 P
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_5_kernelIdentity_12:output:0*
dtype0*
_output_shapes
 P
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_5_biasIdentity_13:output:0*
dtype0*
_output_shapes
 P
Identity_14IdentityRestoreV2:tensors:14*
T0	*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_iterIdentity_14:output:0*
dtype0	*
_output_shapes
 P
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_beta_1Identity_15:output:0*
dtype0*
_output_shapes
 P
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_beta_2Identity_16:output:0*
dtype0*
_output_shapes
 P
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_decayIdentity_17:output:0*
dtype0*
_output_shapes
 P
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp&assignvariableop_18_adam_learning_rateIdentity_18:output:0*
dtype0*
_output_shapes
 P
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:{
AssignVariableOp_19AssignVariableOpassignvariableop_19_totalIdentity_19:output:0*
dtype0*
_output_shapes
 P
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:{
AssignVariableOp_20AssignVariableOpassignvariableop_20_countIdentity_20:output:0*
dtype0*
_output_shapes
 P
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp+assignvariableop_21_adam_conv2d_16_kernel_mIdentity_21:output:0*
dtype0*
_output_shapes
 P
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_conv2d_16_bias_mIdentity_22:output:0*
dtype0*
_output_shapes
 P
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_conv2d_17_kernel_mIdentity_23:output:0*
dtype0*
_output_shapes
 P
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_conv2d_17_bias_mIdentity_24:output:0*
dtype0*
_output_shapes
 P
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp+assignvariableop_25_adam_conv2d_18_kernel_mIdentity_25:output:0*
dtype0*
_output_shapes
 P
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_conv2d_18_bias_mIdentity_26:output:0*
dtype0*
_output_shapes
 P
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_conv2d_19_kernel_mIdentity_27:output:0*
dtype0*
_output_shapes
 P
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_conv2d_19_bias_mIdentity_28:output:0*
dtype0*
_output_shapes
 P
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_conv2d_20_kernel_mIdentity_29:output:0*
dtype0*
_output_shapes
 P
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_conv2d_20_bias_mIdentity_30:output:0*
dtype0*
_output_shapes
 P
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_dense_4_kernel_mIdentity_31:output:0*
dtype0*
_output_shapes
 P
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_dense_4_bias_mIdentity_32:output:0*
dtype0*
_output_shapes
 P
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp)assignvariableop_33_adam_dense_5_kernel_mIdentity_33:output:0*
dtype0*
_output_shapes
 P
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp'assignvariableop_34_adam_dense_5_bias_mIdentity_34:output:0*
dtype0*
_output_shapes
 P
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp+assignvariableop_35_adam_conv2d_16_kernel_vIdentity_35:output:0*
dtype0*
_output_shapes
 P
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_conv2d_16_bias_vIdentity_36:output:0*
dtype0*
_output_shapes
 P
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp+assignvariableop_37_adam_conv2d_17_kernel_vIdentity_37:output:0*
dtype0*
_output_shapes
 P
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_conv2d_17_bias_vIdentity_38:output:0*
dtype0*
_output_shapes
 P
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp+assignvariableop_39_adam_conv2d_18_kernel_vIdentity_39:output:0*
dtype0*
_output_shapes
 P
Identity_40IdentityRestoreV2:tensors:40*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp)assignvariableop_40_adam_conv2d_18_bias_vIdentity_40:output:0*
dtype0*
_output_shapes
 P
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp+assignvariableop_41_adam_conv2d_19_kernel_vIdentity_41:output:0*
dtype0*
_output_shapes
 P
Identity_42IdentityRestoreV2:tensors:42*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp)assignvariableop_42_adam_conv2d_19_bias_vIdentity_42:output:0*
dtype0*
_output_shapes
 P
Identity_43IdentityRestoreV2:tensors:43*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp+assignvariableop_43_adam_conv2d_20_kernel_vIdentity_43:output:0*
dtype0*
_output_shapes
 P
Identity_44IdentityRestoreV2:tensors:44*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp)assignvariableop_44_adam_conv2d_20_bias_vIdentity_44:output:0*
dtype0*
_output_shapes
 P
Identity_45IdentityRestoreV2:tensors:45*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp)assignvariableop_45_adam_dense_4_kernel_vIdentity_45:output:0*
dtype0*
_output_shapes
 P
Identity_46IdentityRestoreV2:tensors:46*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp'assignvariableop_46_adam_dense_4_bias_vIdentity_46:output:0*
dtype0*
_output_shapes
 P
Identity_47IdentityRestoreV2:tensors:47*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp)assignvariableop_47_adam_dense_5_kernel_vIdentity_47:output:0*
dtype0*
_output_shapes
 P
Identity_48IdentityRestoreV2:tensors:48*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp'assignvariableop_48_adam_dense_5_bias_vIdentity_48:output:0*
dtype0*
_output_shapes
 �
RestoreV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:�
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
dtypes
2*
_output_shapes
:1
NoOpNoOp"/device:CPU:0*
_output_shapes
 �	
Identity_49Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: �	
Identity_50IdentityIdentity_49:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: "#
identity_50Identity_50:output:0*�
_input_shapes�
�: :::::::::::::::::::::::::::::::::::::::::::::::::2*
AssignVariableOp_29AssignVariableOp_292*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_39AssignVariableOp_392*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482
RestoreV2_1RestoreV2_12(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_19AssignVariableOp_192*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_31AssignVariableOp_312$
AssignVariableOpAssignVariableOp2*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_28AssignVariableOp_28: : :0 :# : :	 :+ : :+ '
%
_user_specified_namefile_prefix:" : : :* :% : : :- : : :$ : : :, : :
 : :' : : :/ : : : :& : : :. : : :! : : :) : : :1 :  : : :( 
�
J
.__inference_activation_21_layer_call_fn_123253

inputs
identity�
PartitionedCallPartitionedCallinputs*-
_gradient_op_typePartitionedCall-122460*R
fMRK
I__inference_activation_21_layer_call_and_return_conditional_losses_122454*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������@h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�
e
F__inference_dropout_12_layer_call_and_return_conditional_losses_122554

inputs
identity�Q
dropout/rateConst*
valueB
 *  �>*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: _
dropout/random_uniform/maxConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*/
_output_shapes
:���������@�
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: �
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:���������@�
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:���������@R
dropout/sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: �
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*/
_output_shapes
:���������@i
dropout/mulMulinputsdropout/truediv:z:0*
T0*/
_output_shapes
:���������@w
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*/
_output_shapes
:���������@q
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:���������@a
IdentityIdentitydropout/mul_1:z:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�
�
*__inference_conv2d_18_layer_call_fn_122296

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-122291*N
fIRG
E__inference_conv2d_18_layer_call_and_return_conditional_losses_122285*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*A
_output_shapes/
-:+���������������������������@�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@"
identityIdentity:output:0*H
_input_shapes7
5:+��������������������������� ::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
�
e
I__inference_activation_21_layer_call_and_return_conditional_losses_123248

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:���������@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�
h
L__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_122304

inputs
identity�
MaxPoolMaxPoolinputs*
strides
*
ksize
*
paddingVALID*J
_output_shapes8
6:4������������������������������������{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:& "
 
_user_specified_nameinputs
�R
�
H__inference_sequential_3_layer_call_and_return_conditional_losses_122841
conv2d_16_input,
(conv2d_16_statefulpartitionedcall_args_1,
(conv2d_16_statefulpartitionedcall_args_2,
(conv2d_17_statefulpartitionedcall_args_1,
(conv2d_17_statefulpartitionedcall_args_2,
(conv2d_18_statefulpartitionedcall_args_1,
(conv2d_18_statefulpartitionedcall_args_2,
(conv2d_19_statefulpartitionedcall_args_1,
(conv2d_19_statefulpartitionedcall_args_2,
(conv2d_20_statefulpartitionedcall_args_1,
(conv2d_20_statefulpartitionedcall_args_2*
&dense_4_statefulpartitionedcall_args_1*
&dense_4_statefulpartitionedcall_args_2*
&dense_5_statefulpartitionedcall_args_1*
&dense_5_statefulpartitionedcall_args_2
identity��!conv2d_16/StatefulPartitionedCall�!conv2d_17/StatefulPartitionedCall�!conv2d_18/StatefulPartitionedCall�!conv2d_19/StatefulPartitionedCall�!conv2d_20/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCallconv2d_16_input(conv2d_16_statefulpartitionedcall_args_1(conv2d_16_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-122225*N
fIRG
E__inference_conv2d_16_layer_call_and_return_conditional_losses_122219*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������00 �
activation_19/PartitionedCallPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-122417*R
fMRK
I__inference_activation_19_layer_call_and_return_conditional_losses_122411*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������00 �
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall&activation_19/PartitionedCall:output:0(conv2d_17_statefulpartitionedcall_args_1(conv2d_17_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-122249*N
fIRG
E__inference_conv2d_17_layer_call_and_return_conditional_losses_122243*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������.. �
activation_20/PartitionedCallPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-122438*R
fMRK
I__inference_activation_20_layer_call_and_return_conditional_losses_122432*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������.. �
 max_pooling2d_12/PartitionedCallPartitionedCall&activation_20/PartitionedCall:output:0*-
_gradient_op_typePartitionedCall-122268*U
fPRN
L__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_122262*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:��������� �
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_12/PartitionedCall:output:0(conv2d_18_statefulpartitionedcall_args_1(conv2d_18_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-122291*N
fIRG
E__inference_conv2d_18_layer_call_and_return_conditional_losses_122285*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������@�
activation_21/PartitionedCallPartitionedCall*conv2d_18/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-122460*R
fMRK
I__inference_activation_21_layer_call_and_return_conditional_losses_122454*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������@�
 max_pooling2d_13/PartitionedCallPartitionedCall&activation_21/PartitionedCall:output:0*-
_gradient_op_typePartitionedCall-122310*U
fPRN
L__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_122304*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������

@�
dropout_11/PartitionedCallPartitionedCall)max_pooling2d_13/PartitionedCall:output:0*-
_gradient_op_typePartitionedCall-122507*O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_122495*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������

@�
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCall#dropout_11/PartitionedCall:output:0(conv2d_19_statefulpartitionedcall_args_1(conv2d_19_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-122333*N
fIRG
E__inference_conv2d_19_layer_call_and_return_conditional_losses_122327*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������@�
activation_22/PartitionedCallPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-122526*R
fMRK
I__inference_activation_22_layer_call_and_return_conditional_losses_122520*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������@�
 max_pooling2d_14/PartitionedCallPartitionedCall&activation_22/PartitionedCall:output:0*-
_gradient_op_typePartitionedCall-122352*U
fPRN
L__inference_max_pooling2d_14_layer_call_and_return_conditional_losses_122346*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������@�
dropout_12/PartitionedCallPartitionedCall)max_pooling2d_14/PartitionedCall:output:0*-
_gradient_op_typePartitionedCall-122573*O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_122561*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������@�
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall#dropout_12/PartitionedCall:output:0(conv2d_20_statefulpartitionedcall_args_1(conv2d_20_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-122375*N
fIRG
E__inference_conv2d_20_layer_call_and_return_conditional_losses_122369*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������@�
activation_23/PartitionedCallPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-122592*R
fMRK
I__inference_activation_23_layer_call_and_return_conditional_losses_122586*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������@�
 max_pooling2d_15/PartitionedCallPartitionedCall&activation_23/PartitionedCall:output:0*-
_gradient_op_typePartitionedCall-122394*U
fPRN
L__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_122388*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������@�
dropout_13/PartitionedCallPartitionedCall)max_pooling2d_15/PartitionedCall:output:0*-
_gradient_op_typePartitionedCall-122639*O
fJRH
F__inference_dropout_13_layer_call_and_return_conditional_losses_122627*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������@�
flatten_2/PartitionedCallPartitionedCall#dropout_13/PartitionedCall:output:0*-
_gradient_op_typePartitionedCall-122656*N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_122650*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:���������@�
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0&dense_4_statefulpartitionedcall_args_1&dense_4_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-122679*L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_122673*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*(
_output_shapes
:�����������
activation_24/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-122701*R
fMRK
I__inference_activation_24_layer_call_and_return_conditional_losses_122695*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*(
_output_shapes
:�����������
dropout_14/PartitionedCallPartitionedCall&activation_24/PartitionedCall:output:0*-
_gradient_op_typePartitionedCall-122747*O
fJRH
F__inference_dropout_14_layer_call_and_return_conditional_losses_122735*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*(
_output_shapes
:�����������
dense_5/StatefulPartitionedCallStatefulPartitionedCall#dropout_14/PartitionedCall:output:0&dense_5_statefulpartitionedcall_args_1&dense_5_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-122768*L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_122762*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:���������&�
activation_25/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-122790*R
fMRK
I__inference_activation_25_layer_call_and_return_conditional_losses_122784*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:���������&�
IdentityIdentity&activation_25/PartitionedCall:output:0"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall"^conv2d_18/StatefulPartitionedCall"^conv2d_19/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*
T0*'
_output_shapes
:���������&"
identityIdentity:output:0*f
_input_shapesU
S:���������22::::::::::::::2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall: : : : : :	 : : : :/ +
)
_user_specified_nameconv2d_16_input: : : : :
 
�d
�
!__inference__wrapped_model_122206
conv2d_16_input9
5sequential_3_conv2d_16_conv2d_readvariableop_resource:
6sequential_3_conv2d_16_biasadd_readvariableop_resource9
5sequential_3_conv2d_17_conv2d_readvariableop_resource:
6sequential_3_conv2d_17_biasadd_readvariableop_resource9
5sequential_3_conv2d_18_conv2d_readvariableop_resource:
6sequential_3_conv2d_18_biasadd_readvariableop_resource9
5sequential_3_conv2d_19_conv2d_readvariableop_resource:
6sequential_3_conv2d_19_biasadd_readvariableop_resource9
5sequential_3_conv2d_20_conv2d_readvariableop_resource:
6sequential_3_conv2d_20_biasadd_readvariableop_resource7
3sequential_3_dense_4_matmul_readvariableop_resource8
4sequential_3_dense_4_biasadd_readvariableop_resource7
3sequential_3_dense_5_matmul_readvariableop_resource8
4sequential_3_dense_5_biasadd_readvariableop_resource
identity��-sequential_3/conv2d_16/BiasAdd/ReadVariableOp�,sequential_3/conv2d_16/Conv2D/ReadVariableOp�-sequential_3/conv2d_17/BiasAdd/ReadVariableOp�,sequential_3/conv2d_17/Conv2D/ReadVariableOp�-sequential_3/conv2d_18/BiasAdd/ReadVariableOp�,sequential_3/conv2d_18/Conv2D/ReadVariableOp�-sequential_3/conv2d_19/BiasAdd/ReadVariableOp�,sequential_3/conv2d_19/Conv2D/ReadVariableOp�-sequential_3/conv2d_20/BiasAdd/ReadVariableOp�,sequential_3/conv2d_20/Conv2D/ReadVariableOp�+sequential_3/dense_4/BiasAdd/ReadVariableOp�*sequential_3/dense_4/MatMul/ReadVariableOp�+sequential_3/dense_5/BiasAdd/ReadVariableOp�*sequential_3/dense_5/MatMul/ReadVariableOp�
,sequential_3/conv2d_16/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_16_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
: �
sequential_3/conv2d_16/Conv2DConv2Dconv2d_16_input4sequential_3/conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*/
_output_shapes
:���������00 �
-sequential_3/conv2d_16/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_16_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
: �
sequential_3/conv2d_16/BiasAddBiasAdd&sequential_3/conv2d_16/Conv2D:output:05sequential_3/conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������00 �
sequential_3/activation_19/ReluRelu'sequential_3/conv2d_16/BiasAdd:output:0*
T0*/
_output_shapes
:���������00 �
,sequential_3/conv2d_17/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_17_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:  �
sequential_3/conv2d_17/Conv2DConv2D-sequential_3/activation_19/Relu:activations:04sequential_3/conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*/
_output_shapes
:���������.. �
-sequential_3/conv2d_17/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_17_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
: �
sequential_3/conv2d_17/BiasAddBiasAdd&sequential_3/conv2d_17/Conv2D:output:05sequential_3/conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������.. �
sequential_3/activation_20/ReluRelu'sequential_3/conv2d_17/BiasAdd:output:0*
T0*/
_output_shapes
:���������.. �
%sequential_3/max_pooling2d_12/MaxPoolMaxPool-sequential_3/activation_20/Relu:activations:0*
strides
*
ksize
*
paddingVALID*/
_output_shapes
:��������� �
,sequential_3/conv2d_18/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_18_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
: @�
sequential_3/conv2d_18/Conv2DConv2D.sequential_3/max_pooling2d_12/MaxPool:output:04sequential_3/conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*/
_output_shapes
:���������@�
-sequential_3/conv2d_18/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_18_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@�
sequential_3/conv2d_18/BiasAddBiasAdd&sequential_3/conv2d_18/Conv2D:output:05sequential_3/conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
sequential_3/conv2d_18/ReluRelu'sequential_3/conv2d_18/BiasAdd:output:0*
T0*/
_output_shapes
:���������@�
sequential_3/activation_21/ReluRelu)sequential_3/conv2d_18/Relu:activations:0*
T0*/
_output_shapes
:���������@�
%sequential_3/max_pooling2d_13/MaxPoolMaxPool-sequential_3/activation_21/Relu:activations:0*
strides
*
ksize
*
paddingVALID*/
_output_shapes
:���������

@�
 sequential_3/dropout_11/IdentityIdentity.sequential_3/max_pooling2d_13/MaxPool:output:0*
T0*/
_output_shapes
:���������

@�
,sequential_3/conv2d_19/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_19_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:@@�
sequential_3/conv2d_19/Conv2DConv2D)sequential_3/dropout_11/Identity:output:04sequential_3/conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*/
_output_shapes
:���������@�
-sequential_3/conv2d_19/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_19_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@�
sequential_3/conv2d_19/BiasAddBiasAdd&sequential_3/conv2d_19/Conv2D:output:05sequential_3/conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
sequential_3/conv2d_19/ReluRelu'sequential_3/conv2d_19/BiasAdd:output:0*
T0*/
_output_shapes
:���������@�
sequential_3/activation_22/ReluRelu)sequential_3/conv2d_19/Relu:activations:0*
T0*/
_output_shapes
:���������@�
%sequential_3/max_pooling2d_14/MaxPoolMaxPool-sequential_3/activation_22/Relu:activations:0*
strides
*
ksize
*
paddingVALID*/
_output_shapes
:���������@�
 sequential_3/dropout_12/IdentityIdentity.sequential_3/max_pooling2d_14/MaxPool:output:0*
T0*/
_output_shapes
:���������@�
,sequential_3/conv2d_20/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_20_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:@@�
sequential_3/conv2d_20/Conv2DConv2D)sequential_3/dropout_12/Identity:output:04sequential_3/conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*/
_output_shapes
:���������@�
-sequential_3/conv2d_20/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_20_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@�
sequential_3/conv2d_20/BiasAddBiasAdd&sequential_3/conv2d_20/Conv2D:output:05sequential_3/conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
sequential_3/conv2d_20/ReluRelu'sequential_3/conv2d_20/BiasAdd:output:0*
T0*/
_output_shapes
:���������@�
sequential_3/activation_23/ReluRelu)sequential_3/conv2d_20/Relu:activations:0*
T0*/
_output_shapes
:���������@�
%sequential_3/max_pooling2d_15/MaxPoolMaxPool-sequential_3/activation_23/Relu:activations:0*
strides
*
ksize
*
paddingVALID*/
_output_shapes
:���������@�
 sequential_3/dropout_13/IdentityIdentity.sequential_3/max_pooling2d_15/MaxPool:output:0*
T0*/
_output_shapes
:���������@u
$sequential_3/flatten_2/Reshape/shapeConst*
valueB"����@   *
dtype0*
_output_shapes
:�
sequential_3/flatten_2/ReshapeReshape)sequential_3/dropout_13/Identity:output:0-sequential_3/flatten_2/Reshape/shape:output:0*
T0*'
_output_shapes
:���������@�
*sequential_3/dense_4/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_4_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@��
sequential_3/dense_4/MatMulMatMul'sequential_3/flatten_2/Reshape:output:02sequential_3/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
+sequential_3/dense_4/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:��
sequential_3/dense_4/BiasAddBiasAdd%sequential_3/dense_4/MatMul:product:03sequential_3/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
sequential_3/activation_24/ReluRelu%sequential_3/dense_4/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
 sequential_3/dropout_14/IdentityIdentity-sequential_3/activation_24/Relu:activations:0*
T0*(
_output_shapes
:�����������
*sequential_3/dense_5/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_5_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	�&�
sequential_3/dense_5/MatMulMatMul)sequential_3/dropout_14/Identity:output:02sequential_3/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������&�
+sequential_3/dense_5/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_5_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:&�
sequential_3/dense_5/BiasAddBiasAdd%sequential_3/dense_5/MatMul:product:03sequential_3/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������&�
"sequential_3/activation_25/SoftmaxSoftmax%sequential_3/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:���������&�
IdentityIdentity,sequential_3/activation_25/Softmax:softmax:0.^sequential_3/conv2d_16/BiasAdd/ReadVariableOp-^sequential_3/conv2d_16/Conv2D/ReadVariableOp.^sequential_3/conv2d_17/BiasAdd/ReadVariableOp-^sequential_3/conv2d_17/Conv2D/ReadVariableOp.^sequential_3/conv2d_18/BiasAdd/ReadVariableOp-^sequential_3/conv2d_18/Conv2D/ReadVariableOp.^sequential_3/conv2d_19/BiasAdd/ReadVariableOp-^sequential_3/conv2d_19/Conv2D/ReadVariableOp.^sequential_3/conv2d_20/BiasAdd/ReadVariableOp-^sequential_3/conv2d_20/Conv2D/ReadVariableOp,^sequential_3/dense_4/BiasAdd/ReadVariableOp+^sequential_3/dense_4/MatMul/ReadVariableOp,^sequential_3/dense_5/BiasAdd/ReadVariableOp+^sequential_3/dense_5/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������&"
identityIdentity:output:0*f
_input_shapesU
S:���������22::::::::::::::2^
-sequential_3/conv2d_16/BiasAdd/ReadVariableOp-sequential_3/conv2d_16/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_5/BiasAdd/ReadVariableOp+sequential_3/dense_5/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_17/Conv2D/ReadVariableOp,sequential_3/conv2d_17/Conv2D/ReadVariableOp2^
-sequential_3/conv2d_19/BiasAdd/ReadVariableOp-sequential_3/conv2d_19/BiasAdd/ReadVariableOp2X
*sequential_3/dense_4/MatMul/ReadVariableOp*sequential_3/dense_4/MatMul/ReadVariableOp2^
-sequential_3/conv2d_17/BiasAdd/ReadVariableOp-sequential_3/conv2d_17/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_18/Conv2D/ReadVariableOp,sequential_3/conv2d_18/Conv2D/ReadVariableOp2X
*sequential_3/dense_5/MatMul/ReadVariableOp*sequential_3/dense_5/MatMul/ReadVariableOp2\
,sequential_3/conv2d_20/Conv2D/ReadVariableOp,sequential_3/conv2d_20/Conv2D/ReadVariableOp2^
-sequential_3/conv2d_20/BiasAdd/ReadVariableOp-sequential_3/conv2d_20/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_4/BiasAdd/ReadVariableOp+sequential_3/dense_4/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_19/Conv2D/ReadVariableOp,sequential_3/conv2d_19/Conv2D/ReadVariableOp2^
-sequential_3/conv2d_18/BiasAdd/ReadVariableOp-sequential_3/conv2d_18/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_16/Conv2D/ReadVariableOp,sequential_3/conv2d_16/Conv2D/ReadVariableOp: : : : : :	 : : : :/ +
)
_user_specified_nameconv2d_16_input: : : : :
 
�

�
E__inference_conv2d_17_layer_call_and_return_conditional_losses_122243

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:  �
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*A
_output_shapes/
-:+��������������������������� �
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
: �
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� �
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+��������������������������� "
identityIdentity:output:0*H
_input_shapes7
5:+��������������������������� ::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
�
J
.__inference_activation_22_layer_call_fn_123298

inputs
identity�
PartitionedCallPartitionedCallinputs*-
_gradient_op_typePartitionedCall-122526*R
fMRK
I__inference_activation_22_layer_call_and_return_conditional_losses_122520*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������@h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�
J
.__inference_activation_20_layer_call_fn_123243

inputs
identity�
PartitionedCallPartitionedCallinputs*-
_gradient_op_typePartitionedCall-122438*R
fMRK
I__inference_activation_20_layer_call_and_return_conditional_losses_122432*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������.. h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������.. "
identityIdentity:output:0*.
_input_shapes
:���������.. :& "
 
_user_specified_nameinputs
�
d
F__inference_dropout_13_layer_call_and_return_conditional_losses_122627

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:���������@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������@"!

identity_1Identity_1:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�
e
I__inference_activation_22_layer_call_and_return_conditional_losses_123293

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:���������@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�
�
E__inference_conv2d_19_layer_call_and_return_conditional_losses_122327

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:@@�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*A
_output_shapes/
-:+���������������������������@�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������@�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+���������������������������@"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������@::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
�
e
I__inference_activation_20_layer_call_and_return_conditional_losses_123238

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:���������.. b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������.. "
identityIdentity:output:0*.
_input_shapes
:���������.. :& "
 
_user_specified_nameinputs
�
M
1__inference_max_pooling2d_13_layer_call_fn_122313

inputs
identity�
PartitionedCallPartitionedCallinputs*-
_gradient_op_typePartitionedCall-122310*U
fPRN
L__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_122304*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*J
_output_shapes8
6:4�������������������������������������
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:& "
 
_user_specified_nameinputs
�
�
C__inference_dense_5_layer_call_and_return_conditional_losses_122762

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	�&i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������&�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:&v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������&�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������&"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
�
J
.__inference_activation_25_layer_call_fn_123478

inputs
identity�
PartitionedCallPartitionedCallinputs*-
_gradient_op_typePartitionedCall-122790*R
fMRK
I__inference_activation_25_layer_call_and_return_conditional_losses_122784*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:���������&`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������&"
identityIdentity:output:0*&
_input_shapes
:���������&:& "
 
_user_specified_nameinputs
�
�
*__inference_conv2d_17_layer_call_fn_122254

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-122249*N
fIRG
E__inference_conv2d_17_layer_call_and_return_conditional_losses_122243*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*A
_output_shapes/
-:+��������������������������� �
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� "
identityIdentity:output:0*H
_input_shapes7
5:+��������������������������� ::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
�
F
*__inference_flatten_2_layer_call_fn_123389

inputs
identity�
PartitionedCallPartitionedCallinputs*-
_gradient_op_typePartitionedCall-122656*N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_122650*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:���������@`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�
e
F__inference_dropout_14_layer_call_and_return_conditional_losses_123436

inputs
identity�Q
dropout/rateConst*
valueB
 *   ?*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: _
dropout/random_uniform/maxConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*(
_output_shapes
:�����������
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: �
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:�����������
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:����������R
dropout/sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: �
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*(
_output_shapes
:����������b
dropout/mulMulinputsdropout/truediv:z:0*
T0*(
_output_shapes
:����������p
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*(
_output_shapes
:����������j
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������Z
IdentityIdentitydropout/mul_1:z:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*'
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�
d
F__inference_dropout_12_layer_call_and_return_conditional_losses_122561

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:���������@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������@"!

identity_1Identity_1:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�Y
�
H__inference_sequential_3_layer_call_and_return_conditional_losses_122798
conv2d_16_input,
(conv2d_16_statefulpartitionedcall_args_1,
(conv2d_16_statefulpartitionedcall_args_2,
(conv2d_17_statefulpartitionedcall_args_1,
(conv2d_17_statefulpartitionedcall_args_2,
(conv2d_18_statefulpartitionedcall_args_1,
(conv2d_18_statefulpartitionedcall_args_2,
(conv2d_19_statefulpartitionedcall_args_1,
(conv2d_19_statefulpartitionedcall_args_2,
(conv2d_20_statefulpartitionedcall_args_1,
(conv2d_20_statefulpartitionedcall_args_2*
&dense_4_statefulpartitionedcall_args_1*
&dense_4_statefulpartitionedcall_args_2*
&dense_5_statefulpartitionedcall_args_1*
&dense_5_statefulpartitionedcall_args_2
identity��!conv2d_16/StatefulPartitionedCall�!conv2d_17/StatefulPartitionedCall�!conv2d_18/StatefulPartitionedCall�!conv2d_19/StatefulPartitionedCall�!conv2d_20/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�"dropout_11/StatefulPartitionedCall�"dropout_12/StatefulPartitionedCall�"dropout_13/StatefulPartitionedCall�"dropout_14/StatefulPartitionedCall�
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCallconv2d_16_input(conv2d_16_statefulpartitionedcall_args_1(conv2d_16_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-122225*N
fIRG
E__inference_conv2d_16_layer_call_and_return_conditional_losses_122219*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������00 �
activation_19/PartitionedCallPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-122417*R
fMRK
I__inference_activation_19_layer_call_and_return_conditional_losses_122411*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������00 �
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall&activation_19/PartitionedCall:output:0(conv2d_17_statefulpartitionedcall_args_1(conv2d_17_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-122249*N
fIRG
E__inference_conv2d_17_layer_call_and_return_conditional_losses_122243*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������.. �
activation_20/PartitionedCallPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-122438*R
fMRK
I__inference_activation_20_layer_call_and_return_conditional_losses_122432*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������.. �
 max_pooling2d_12/PartitionedCallPartitionedCall&activation_20/PartitionedCall:output:0*-
_gradient_op_typePartitionedCall-122268*U
fPRN
L__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_122262*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:��������� �
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_12/PartitionedCall:output:0(conv2d_18_statefulpartitionedcall_args_1(conv2d_18_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-122291*N
fIRG
E__inference_conv2d_18_layer_call_and_return_conditional_losses_122285*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������@�
activation_21/PartitionedCallPartitionedCall*conv2d_18/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-122460*R
fMRK
I__inference_activation_21_layer_call_and_return_conditional_losses_122454*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������@�
 max_pooling2d_13/PartitionedCallPartitionedCall&activation_21/PartitionedCall:output:0*-
_gradient_op_typePartitionedCall-122310*U
fPRN
L__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_122304*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������

@�
"dropout_11/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_13/PartitionedCall:output:0*-
_gradient_op_typePartitionedCall-122499*O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_122488*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������

@�
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCall+dropout_11/StatefulPartitionedCall:output:0(conv2d_19_statefulpartitionedcall_args_1(conv2d_19_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-122333*N
fIRG
E__inference_conv2d_19_layer_call_and_return_conditional_losses_122327*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������@�
activation_22/PartitionedCallPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-122526*R
fMRK
I__inference_activation_22_layer_call_and_return_conditional_losses_122520*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������@�
 max_pooling2d_14/PartitionedCallPartitionedCall&activation_22/PartitionedCall:output:0*-
_gradient_op_typePartitionedCall-122352*U
fPRN
L__inference_max_pooling2d_14_layer_call_and_return_conditional_losses_122346*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������@�
"dropout_12/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_14/PartitionedCall:output:0#^dropout_11/StatefulPartitionedCall*-
_gradient_op_typePartitionedCall-122565*O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_122554*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������@�
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall+dropout_12/StatefulPartitionedCall:output:0(conv2d_20_statefulpartitionedcall_args_1(conv2d_20_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-122375*N
fIRG
E__inference_conv2d_20_layer_call_and_return_conditional_losses_122369*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������@�
activation_23/PartitionedCallPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-122592*R
fMRK
I__inference_activation_23_layer_call_and_return_conditional_losses_122586*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������@�
 max_pooling2d_15/PartitionedCallPartitionedCall&activation_23/PartitionedCall:output:0*-
_gradient_op_typePartitionedCall-122394*U
fPRN
L__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_122388*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������@�
"dropout_13/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_15/PartitionedCall:output:0#^dropout_12/StatefulPartitionedCall*-
_gradient_op_typePartitionedCall-122631*O
fJRH
F__inference_dropout_13_layer_call_and_return_conditional_losses_122620*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������@�
flatten_2/PartitionedCallPartitionedCall+dropout_13/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-122656*N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_122650*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:���������@�
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0&dense_4_statefulpartitionedcall_args_1&dense_4_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-122679*L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_122673*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*(
_output_shapes
:�����������
activation_24/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-122701*R
fMRK
I__inference_activation_24_layer_call_and_return_conditional_losses_122695*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*(
_output_shapes
:�����������
"dropout_14/StatefulPartitionedCallStatefulPartitionedCall&activation_24/PartitionedCall:output:0#^dropout_13/StatefulPartitionedCall*-
_gradient_op_typePartitionedCall-122739*O
fJRH
F__inference_dropout_14_layer_call_and_return_conditional_losses_122728*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*(
_output_shapes
:�����������
dense_5/StatefulPartitionedCallStatefulPartitionedCall+dropout_14/StatefulPartitionedCall:output:0&dense_5_statefulpartitionedcall_args_1&dense_5_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-122768*L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_122762*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:���������&�
activation_25/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-122790*R
fMRK
I__inference_activation_25_layer_call_and_return_conditional_losses_122784*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:���������&�
IdentityIdentity&activation_25/PartitionedCall:output:0"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall"^conv2d_18/StatefulPartitionedCall"^conv2d_19/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall#^dropout_11/StatefulPartitionedCall#^dropout_12/StatefulPartitionedCall#^dropout_13/StatefulPartitionedCall#^dropout_14/StatefulPartitionedCall*
T0*'
_output_shapes
:���������&"
identityIdentity:output:0*f
_input_shapesU
S:���������22::::::::::::::2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2H
"dropout_11/StatefulPartitionedCall"dropout_11/StatefulPartitionedCall2H
"dropout_12/StatefulPartitionedCall"dropout_12/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall2H
"dropout_13/StatefulPartitionedCall"dropout_13/StatefulPartitionedCall2H
"dropout_14/StatefulPartitionedCall"dropout_14/StatefulPartitionedCall2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall: : : : : :	 : : : :/ +
)
_user_specified_nameconv2d_16_input: : : : :
 
�
�
*__inference_conv2d_16_layer_call_fn_122230

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-122225*N
fIRG
E__inference_conv2d_16_layer_call_and_return_conditional_losses_122219*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*A
_output_shapes/
-:+��������������������������� �
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� "
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
�
e
I__inference_activation_24_layer_call_and_return_conditional_losses_122695

inputs
identityG
ReluReluinputs*
T0*(
_output_shapes
:����������[
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*'
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�

�
E__inference_conv2d_16_layer_call_and_return_conditional_losses_122219

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
: �
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*A
_output_shapes/
-:+��������������������������� �
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
: �
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� �
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+��������������������������� "
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
�
J
.__inference_activation_19_layer_call_fn_123233

inputs
identity�
PartitionedCallPartitionedCallinputs*-
_gradient_op_typePartitionedCall-122417*R
fMRK
I__inference_activation_19_layer_call_and_return_conditional_losses_122411*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������00 h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������00 "
identityIdentity:output:0*.
_input_shapes
:���������00 :& "
 
_user_specified_nameinputs
�
�
C__inference_dense_5_layer_call_and_return_conditional_losses_123461

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	�&i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������&�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:&v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������&�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������&"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
�
e
I__inference_activation_19_layer_call_and_return_conditional_losses_123228

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:���������00 b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������00 "
identityIdentity:output:0*.
_input_shapes
:���������00 :& "
 
_user_specified_nameinputs
�
d
F__inference_dropout_14_layer_call_and_return_conditional_losses_123441

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*'
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�
�
-__inference_sequential_3_layer_call_fn_122966
conv2d_16_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv2d_16_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14*-
_gradient_op_typePartitionedCall-122949*Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_122948*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:���������&�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������&"
identityIdentity:output:0*f
_input_shapesU
S:���������22::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : :	 : : : :/ +
)
_user_specified_nameconv2d_16_input: : : : :
 
�
d
+__inference_dropout_12_layer_call_fn_123328

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*-
_gradient_op_typePartitionedCall-122565*O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_122554*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������@�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*.
_input_shapes
:���������@22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
e
I__inference_activation_23_layer_call_and_return_conditional_losses_123338

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:���������@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�
e
I__inference_activation_25_layer_call_and_return_conditional_losses_123473

inputs
identityL
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:���������&Y
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:���������&"
identityIdentity:output:0*&
_input_shapes
:���������&:& "
 
_user_specified_nameinputs
�
�
$__inference_signature_wrapper_122991
conv2d_16_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv2d_16_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14*-
_gradient_op_typePartitionedCall-122974**
f%R#
!__inference__wrapped_model_122206*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:���������&�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������&"
identityIdentity:output:0*f
_input_shapesU
S:���������22::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : :	 : : : :/ +
)
_user_specified_nameconv2d_16_input: : : : :
 
�
h
L__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_122262

inputs
identity�
MaxPoolMaxPoolinputs*
strides
*
ksize
*
paddingVALID*J
_output_shapes8
6:4������������������������������������{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:& "
 
_user_specified_nameinputs
�
h
L__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_122388

inputs
identity�
MaxPoolMaxPoolinputs*
strides
*
ksize
*
paddingVALID*J
_output_shapes8
6:4������������������������������������{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:& "
 
_user_specified_nameinputs
�
J
.__inference_activation_23_layer_call_fn_123343

inputs
identity�
PartitionedCallPartitionedCallinputs*-
_gradient_op_typePartitionedCall-122592*R
fMRK
I__inference_activation_23_layer_call_and_return_conditional_losses_122586*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������@h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�
d
F__inference_dropout_11_layer_call_and_return_conditional_losses_122495

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:���������

@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������

@"!

identity_1Identity_1:output:0*.
_input_shapes
:���������

@:& "
 
_user_specified_nameinputs
�
�
-__inference_sequential_3_layer_call_fn_122903
conv2d_16_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv2d_16_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14*-
_gradient_op_typePartitionedCall-122886*Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_122885*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:���������&�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������&"
identityIdentity:output:0*f
_input_shapesU
S:���������22::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : :	 : : : :/ +
)
_user_specified_nameconv2d_16_input: : : : :
 
�
M
1__inference_max_pooling2d_14_layer_call_fn_122355

inputs
identity�
PartitionedCallPartitionedCallinputs*-
_gradient_op_typePartitionedCall-122352*U
fPRN
L__inference_max_pooling2d_14_layer_call_and_return_conditional_losses_122346*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*J
_output_shapes8
6:4�������������������������������������
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:& "
 
_user_specified_nameinputs
�
M
1__inference_max_pooling2d_15_layer_call_fn_122397

inputs
identity�
PartitionedCallPartitionedCallinputs*-
_gradient_op_typePartitionedCall-122394*U
fPRN
L__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_122388*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*J
_output_shapes8
6:4�������������������������������������
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:& "
 
_user_specified_nameinputs
�
h
L__inference_max_pooling2d_14_layer_call_and_return_conditional_losses_122346

inputs
identity�
MaxPoolMaxPoolinputs*
strides
*
ksize
*
paddingVALID*J
_output_shapes8
6:4������������������������������������{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:& "
 
_user_specified_nameinputs
�
e
I__inference_activation_25_layer_call_and_return_conditional_losses_122784

inputs
identityL
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:���������&Y
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:���������&"
identityIdentity:output:0*&
_input_shapes
:���������&:& "
 
_user_specified_nameinputs
�Y
�
H__inference_sequential_3_layer_call_and_return_conditional_losses_122885

inputs,
(conv2d_16_statefulpartitionedcall_args_1,
(conv2d_16_statefulpartitionedcall_args_2,
(conv2d_17_statefulpartitionedcall_args_1,
(conv2d_17_statefulpartitionedcall_args_2,
(conv2d_18_statefulpartitionedcall_args_1,
(conv2d_18_statefulpartitionedcall_args_2,
(conv2d_19_statefulpartitionedcall_args_1,
(conv2d_19_statefulpartitionedcall_args_2,
(conv2d_20_statefulpartitionedcall_args_1,
(conv2d_20_statefulpartitionedcall_args_2*
&dense_4_statefulpartitionedcall_args_1*
&dense_4_statefulpartitionedcall_args_2*
&dense_5_statefulpartitionedcall_args_1*
&dense_5_statefulpartitionedcall_args_2
identity��!conv2d_16/StatefulPartitionedCall�!conv2d_17/StatefulPartitionedCall�!conv2d_18/StatefulPartitionedCall�!conv2d_19/StatefulPartitionedCall�!conv2d_20/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�"dropout_11/StatefulPartitionedCall�"dropout_12/StatefulPartitionedCall�"dropout_13/StatefulPartitionedCall�"dropout_14/StatefulPartitionedCall�
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCallinputs(conv2d_16_statefulpartitionedcall_args_1(conv2d_16_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-122225*N
fIRG
E__inference_conv2d_16_layer_call_and_return_conditional_losses_122219*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������00 �
activation_19/PartitionedCallPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-122417*R
fMRK
I__inference_activation_19_layer_call_and_return_conditional_losses_122411*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������00 �
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall&activation_19/PartitionedCall:output:0(conv2d_17_statefulpartitionedcall_args_1(conv2d_17_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-122249*N
fIRG
E__inference_conv2d_17_layer_call_and_return_conditional_losses_122243*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������.. �
activation_20/PartitionedCallPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-122438*R
fMRK
I__inference_activation_20_layer_call_and_return_conditional_losses_122432*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������.. �
 max_pooling2d_12/PartitionedCallPartitionedCall&activation_20/PartitionedCall:output:0*-
_gradient_op_typePartitionedCall-122268*U
fPRN
L__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_122262*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:��������� �
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_12/PartitionedCall:output:0(conv2d_18_statefulpartitionedcall_args_1(conv2d_18_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-122291*N
fIRG
E__inference_conv2d_18_layer_call_and_return_conditional_losses_122285*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������@�
activation_21/PartitionedCallPartitionedCall*conv2d_18/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-122460*R
fMRK
I__inference_activation_21_layer_call_and_return_conditional_losses_122454*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������@�
 max_pooling2d_13/PartitionedCallPartitionedCall&activation_21/PartitionedCall:output:0*-
_gradient_op_typePartitionedCall-122310*U
fPRN
L__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_122304*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������

@�
"dropout_11/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_13/PartitionedCall:output:0*-
_gradient_op_typePartitionedCall-122499*O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_122488*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������

@�
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCall+dropout_11/StatefulPartitionedCall:output:0(conv2d_19_statefulpartitionedcall_args_1(conv2d_19_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-122333*N
fIRG
E__inference_conv2d_19_layer_call_and_return_conditional_losses_122327*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������@�
activation_22/PartitionedCallPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-122526*R
fMRK
I__inference_activation_22_layer_call_and_return_conditional_losses_122520*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������@�
 max_pooling2d_14/PartitionedCallPartitionedCall&activation_22/PartitionedCall:output:0*-
_gradient_op_typePartitionedCall-122352*U
fPRN
L__inference_max_pooling2d_14_layer_call_and_return_conditional_losses_122346*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������@�
"dropout_12/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_14/PartitionedCall:output:0#^dropout_11/StatefulPartitionedCall*-
_gradient_op_typePartitionedCall-122565*O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_122554*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������@�
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall+dropout_12/StatefulPartitionedCall:output:0(conv2d_20_statefulpartitionedcall_args_1(conv2d_20_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-122375*N
fIRG
E__inference_conv2d_20_layer_call_and_return_conditional_losses_122369*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������@�
activation_23/PartitionedCallPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-122592*R
fMRK
I__inference_activation_23_layer_call_and_return_conditional_losses_122586*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������@�
 max_pooling2d_15/PartitionedCallPartitionedCall&activation_23/PartitionedCall:output:0*-
_gradient_op_typePartitionedCall-122394*U
fPRN
L__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_122388*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������@�
"dropout_13/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_15/PartitionedCall:output:0#^dropout_12/StatefulPartitionedCall*-
_gradient_op_typePartitionedCall-122631*O
fJRH
F__inference_dropout_13_layer_call_and_return_conditional_losses_122620*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������@�
flatten_2/PartitionedCallPartitionedCall+dropout_13/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-122656*N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_122650*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:���������@�
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0&dense_4_statefulpartitionedcall_args_1&dense_4_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-122679*L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_122673*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*(
_output_shapes
:�����������
activation_24/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-122701*R
fMRK
I__inference_activation_24_layer_call_and_return_conditional_losses_122695*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*(
_output_shapes
:�����������
"dropout_14/StatefulPartitionedCallStatefulPartitionedCall&activation_24/PartitionedCall:output:0#^dropout_13/StatefulPartitionedCall*-
_gradient_op_typePartitionedCall-122739*O
fJRH
F__inference_dropout_14_layer_call_and_return_conditional_losses_122728*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*(
_output_shapes
:�����������
dense_5/StatefulPartitionedCallStatefulPartitionedCall+dropout_14/StatefulPartitionedCall:output:0&dense_5_statefulpartitionedcall_args_1&dense_5_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-122768*L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_122762*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:���������&�
activation_25/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-122790*R
fMRK
I__inference_activation_25_layer_call_and_return_conditional_losses_122784*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:���������&�
IdentityIdentity&activation_25/PartitionedCall:output:0"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall"^conv2d_18/StatefulPartitionedCall"^conv2d_19/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall#^dropout_11/StatefulPartitionedCall#^dropout_12/StatefulPartitionedCall#^dropout_13/StatefulPartitionedCall#^dropout_14/StatefulPartitionedCall*
T0*'
_output_shapes
:���������&"
identityIdentity:output:0*f
_input_shapesU
S:���������22::::::::::::::2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2H
"dropout_11/StatefulPartitionedCall"dropout_11/StatefulPartitionedCall2H
"dropout_12/StatefulPartitionedCall"dropout_12/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2H
"dropout_13/StatefulPartitionedCall"dropout_13/StatefulPartitionedCall2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall2H
"dropout_14/StatefulPartitionedCall"dropout_14/StatefulPartitionedCall2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall: : : : : :	 : : : :& "
 
_user_specified_nameinputs: : : : :
 
�
�
-__inference_sequential_3_layer_call_fn_123204

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14*-
_gradient_op_typePartitionedCall-122886*Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_122885*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:���������&�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������&"
identityIdentity:output:0*f
_input_shapesU
S:���������22::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : :& "
 
_user_specified_nameinputs: : : : :
 : : : : : :	 : 
�
e
I__inference_activation_23_layer_call_and_return_conditional_losses_122586

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:���������@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�
d
F__inference_dropout_14_layer_call_and_return_conditional_losses_122735

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*'
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�
e
I__inference_activation_24_layer_call_and_return_conditional_losses_123411

inputs
identityG
ReluReluinputs*
T0*(
_output_shapes
:����������[
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*'
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�
e
F__inference_dropout_11_layer_call_and_return_conditional_losses_123273

inputs
identity�Q
dropout/rateConst*
valueB
 *  �>*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: _
dropout/random_uniform/maxConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*/
_output_shapes
:���������

@�
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: �
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:���������

@�
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:���������

@R
dropout/sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: �
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*/
_output_shapes
:���������

@i
dropout/mulMulinputsdropout/truediv:z:0*
T0*/
_output_shapes
:���������

@w
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*/
_output_shapes
:���������

@q
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:���������

@a
IdentityIdentitydropout/mul_1:z:0*
T0*/
_output_shapes
:���������

@"
identityIdentity:output:0*.
_input_shapes
:���������

@:& "
 
_user_specified_nameinputs
�
d
F__inference_dropout_12_layer_call_and_return_conditional_losses_123323

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:���������@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������@"!

identity_1Identity_1:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�
d
+__inference_dropout_14_layer_call_fn_123446

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*-
_gradient_op_typePartitionedCall-122739*O
fJRH
F__inference_dropout_14_layer_call_and_return_conditional_losses_122728*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*(
_output_shapes
:�����������
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
*__inference_conv2d_19_layer_call_fn_122338

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-122333*N
fIRG
E__inference_conv2d_19_layer_call_and_return_conditional_losses_122327*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*A
_output_shapes/
-:+���������������������������@�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������@::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
�
�
-__inference_sequential_3_layer_call_fn_123223

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14*-
_gradient_op_typePartitionedCall-122949*Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_122948*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:���������&�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������&"
identityIdentity:output:0*f
_input_shapesU
S:���������22::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : :& "
 
_user_specified_nameinputs: : : : :
 : : : : : :	 : 
�
G
+__inference_dropout_12_layer_call_fn_123333

inputs
identity�
PartitionedCallPartitionedCallinputs*-
_gradient_op_typePartitionedCall-122573*O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_122561*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������@h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�
�
*__inference_conv2d_20_layer_call_fn_122380

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-122375*N
fIRG
E__inference_conv2d_20_layer_call_and_return_conditional_losses_122369*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*A
_output_shapes/
-:+���������������������������@�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������@::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
�
e
I__inference_activation_22_layer_call_and_return_conditional_losses_122520

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:���������@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�R
�
H__inference_sequential_3_layer_call_and_return_conditional_losses_122948

inputs,
(conv2d_16_statefulpartitionedcall_args_1,
(conv2d_16_statefulpartitionedcall_args_2,
(conv2d_17_statefulpartitionedcall_args_1,
(conv2d_17_statefulpartitionedcall_args_2,
(conv2d_18_statefulpartitionedcall_args_1,
(conv2d_18_statefulpartitionedcall_args_2,
(conv2d_19_statefulpartitionedcall_args_1,
(conv2d_19_statefulpartitionedcall_args_2,
(conv2d_20_statefulpartitionedcall_args_1,
(conv2d_20_statefulpartitionedcall_args_2*
&dense_4_statefulpartitionedcall_args_1*
&dense_4_statefulpartitionedcall_args_2*
&dense_5_statefulpartitionedcall_args_1*
&dense_5_statefulpartitionedcall_args_2
identity��!conv2d_16/StatefulPartitionedCall�!conv2d_17/StatefulPartitionedCall�!conv2d_18/StatefulPartitionedCall�!conv2d_19/StatefulPartitionedCall�!conv2d_20/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCallinputs(conv2d_16_statefulpartitionedcall_args_1(conv2d_16_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-122225*N
fIRG
E__inference_conv2d_16_layer_call_and_return_conditional_losses_122219*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������00 �
activation_19/PartitionedCallPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-122417*R
fMRK
I__inference_activation_19_layer_call_and_return_conditional_losses_122411*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������00 �
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall&activation_19/PartitionedCall:output:0(conv2d_17_statefulpartitionedcall_args_1(conv2d_17_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-122249*N
fIRG
E__inference_conv2d_17_layer_call_and_return_conditional_losses_122243*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������.. �
activation_20/PartitionedCallPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-122438*R
fMRK
I__inference_activation_20_layer_call_and_return_conditional_losses_122432*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������.. �
 max_pooling2d_12/PartitionedCallPartitionedCall&activation_20/PartitionedCall:output:0*-
_gradient_op_typePartitionedCall-122268*U
fPRN
L__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_122262*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:��������� �
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_12/PartitionedCall:output:0(conv2d_18_statefulpartitionedcall_args_1(conv2d_18_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-122291*N
fIRG
E__inference_conv2d_18_layer_call_and_return_conditional_losses_122285*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������@�
activation_21/PartitionedCallPartitionedCall*conv2d_18/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-122460*R
fMRK
I__inference_activation_21_layer_call_and_return_conditional_losses_122454*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������@�
 max_pooling2d_13/PartitionedCallPartitionedCall&activation_21/PartitionedCall:output:0*-
_gradient_op_typePartitionedCall-122310*U
fPRN
L__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_122304*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������

@�
dropout_11/PartitionedCallPartitionedCall)max_pooling2d_13/PartitionedCall:output:0*-
_gradient_op_typePartitionedCall-122507*O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_122495*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������

@�
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCall#dropout_11/PartitionedCall:output:0(conv2d_19_statefulpartitionedcall_args_1(conv2d_19_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-122333*N
fIRG
E__inference_conv2d_19_layer_call_and_return_conditional_losses_122327*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������@�
activation_22/PartitionedCallPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-122526*R
fMRK
I__inference_activation_22_layer_call_and_return_conditional_losses_122520*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������@�
 max_pooling2d_14/PartitionedCallPartitionedCall&activation_22/PartitionedCall:output:0*-
_gradient_op_typePartitionedCall-122352*U
fPRN
L__inference_max_pooling2d_14_layer_call_and_return_conditional_losses_122346*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������@�
dropout_12/PartitionedCallPartitionedCall)max_pooling2d_14/PartitionedCall:output:0*-
_gradient_op_typePartitionedCall-122573*O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_122561*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������@�
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall#dropout_12/PartitionedCall:output:0(conv2d_20_statefulpartitionedcall_args_1(conv2d_20_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-122375*N
fIRG
E__inference_conv2d_20_layer_call_and_return_conditional_losses_122369*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������@�
activation_23/PartitionedCallPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-122592*R
fMRK
I__inference_activation_23_layer_call_and_return_conditional_losses_122586*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������@�
 max_pooling2d_15/PartitionedCallPartitionedCall&activation_23/PartitionedCall:output:0*-
_gradient_op_typePartitionedCall-122394*U
fPRN
L__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_122388*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������@�
dropout_13/PartitionedCallPartitionedCall)max_pooling2d_15/PartitionedCall:output:0*-
_gradient_op_typePartitionedCall-122639*O
fJRH
F__inference_dropout_13_layer_call_and_return_conditional_losses_122627*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������@�
flatten_2/PartitionedCallPartitionedCall#dropout_13/PartitionedCall:output:0*-
_gradient_op_typePartitionedCall-122656*N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_122650*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:���������@�
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0&dense_4_statefulpartitionedcall_args_1&dense_4_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-122679*L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_122673*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*(
_output_shapes
:�����������
activation_24/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-122701*R
fMRK
I__inference_activation_24_layer_call_and_return_conditional_losses_122695*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*(
_output_shapes
:�����������
dropout_14/PartitionedCallPartitionedCall&activation_24/PartitionedCall:output:0*-
_gradient_op_typePartitionedCall-122747*O
fJRH
F__inference_dropout_14_layer_call_and_return_conditional_losses_122735*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*(
_output_shapes
:�����������
dense_5/StatefulPartitionedCallStatefulPartitionedCall#dropout_14/PartitionedCall:output:0&dense_5_statefulpartitionedcall_args_1&dense_5_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-122768*L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_122762*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:���������&�
activation_25/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-122790*R
fMRK
I__inference_activation_25_layer_call_and_return_conditional_losses_122784*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:���������&�
IdentityIdentity&activation_25/PartitionedCall:output:0"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall"^conv2d_18/StatefulPartitionedCall"^conv2d_19/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*
T0*'
_output_shapes
:���������&"
identityIdentity:output:0*f
_input_shapesU
S:���������22::::::::::::::2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall: : :& "
 
_user_specified_nameinputs: : : : :
 : : : : : :	 : 
�
G
+__inference_dropout_11_layer_call_fn_123288

inputs
identity�
PartitionedCallPartitionedCallinputs*-
_gradient_op_typePartitionedCall-122507*O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_122495*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������

@h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������

@"
identityIdentity:output:0*.
_input_shapes
:���������

@:& "
 
_user_specified_nameinputs
�
d
+__inference_dropout_11_layer_call_fn_123283

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*-
_gradient_op_typePartitionedCall-122499*O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_122488*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������

@�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������

@"
identityIdentity:output:0*.
_input_shapes
:���������

@22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
e
F__inference_dropout_13_layer_call_and_return_conditional_losses_122620

inputs
identity�Q
dropout/rateConst*
valueB
 *  �>*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: _
dropout/random_uniform/maxConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*/
_output_shapes
:���������@�
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: �
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:���������@�
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:���������@R
dropout/sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: �
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*/
_output_shapes
:���������@i
dropout/mulMulinputsdropout/truediv:z:0*
T0*/
_output_shapes
:���������@w
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*/
_output_shapes
:���������@q
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:���������@a
IdentityIdentitydropout/mul_1:z:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�R
�	
H__inference_sequential_3_layer_call_and_return_conditional_losses_123185

inputs,
(conv2d_16_conv2d_readvariableop_resource-
)conv2d_16_biasadd_readvariableop_resource,
(conv2d_17_conv2d_readvariableop_resource-
)conv2d_17_biasadd_readvariableop_resource,
(conv2d_18_conv2d_readvariableop_resource-
)conv2d_18_biasadd_readvariableop_resource,
(conv2d_19_conv2d_readvariableop_resource-
)conv2d_19_biasadd_readvariableop_resource,
(conv2d_20_conv2d_readvariableop_resource-
)conv2d_20_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource
identity�� conv2d_16/BiasAdd/ReadVariableOp�conv2d_16/Conv2D/ReadVariableOp� conv2d_17/BiasAdd/ReadVariableOp�conv2d_17/Conv2D/ReadVariableOp� conv2d_18/BiasAdd/ReadVariableOp�conv2d_18/Conv2D/ReadVariableOp� conv2d_19/BiasAdd/ReadVariableOp�conv2d_19/Conv2D/ReadVariableOp� conv2d_20/BiasAdd/ReadVariableOp�conv2d_20/Conv2D/ReadVariableOp�dense_4/BiasAdd/ReadVariableOp�dense_4/MatMul/ReadVariableOp�dense_5/BiasAdd/ReadVariableOp�dense_5/MatMul/ReadVariableOp�
conv2d_16/Conv2D/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
: �
conv2d_16/Conv2DConv2Dinputs'conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*/
_output_shapes
:���������00 �
 conv2d_16/BiasAdd/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
: �
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0(conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������00 p
activation_19/ReluReluconv2d_16/BiasAdd:output:0*
T0*/
_output_shapes
:���������00 �
conv2d_17/Conv2D/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:  �
conv2d_17/Conv2DConv2D activation_19/Relu:activations:0'conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*/
_output_shapes
:���������.. �
 conv2d_17/BiasAdd/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
: �
conv2d_17/BiasAddBiasAddconv2d_17/Conv2D:output:0(conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������.. p
activation_20/ReluReluconv2d_17/BiasAdd:output:0*
T0*/
_output_shapes
:���������.. �
max_pooling2d_12/MaxPoolMaxPool activation_20/Relu:activations:0*
strides
*
ksize
*
paddingVALID*/
_output_shapes
:��������� �
conv2d_18/Conv2D/ReadVariableOpReadVariableOp(conv2d_18_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
: @�
conv2d_18/Conv2DConv2D!max_pooling2d_12/MaxPool:output:0'conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*/
_output_shapes
:���������@�
 conv2d_18/BiasAdd/ReadVariableOpReadVariableOp)conv2d_18_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@�
conv2d_18/BiasAddBiasAddconv2d_18/Conv2D:output:0(conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@l
conv2d_18/ReluReluconv2d_18/BiasAdd:output:0*
T0*/
_output_shapes
:���������@r
activation_21/ReluReluconv2d_18/Relu:activations:0*
T0*/
_output_shapes
:���������@�
max_pooling2d_13/MaxPoolMaxPool activation_21/Relu:activations:0*
strides
*
ksize
*
paddingVALID*/
_output_shapes
:���������

@|
dropout_11/IdentityIdentity!max_pooling2d_13/MaxPool:output:0*
T0*/
_output_shapes
:���������

@�
conv2d_19/Conv2D/ReadVariableOpReadVariableOp(conv2d_19_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:@@�
conv2d_19/Conv2DConv2Ddropout_11/Identity:output:0'conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*/
_output_shapes
:���������@�
 conv2d_19/BiasAdd/ReadVariableOpReadVariableOp)conv2d_19_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@�
conv2d_19/BiasAddBiasAddconv2d_19/Conv2D:output:0(conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@l
conv2d_19/ReluReluconv2d_19/BiasAdd:output:0*
T0*/
_output_shapes
:���������@r
activation_22/ReluReluconv2d_19/Relu:activations:0*
T0*/
_output_shapes
:���������@�
max_pooling2d_14/MaxPoolMaxPool activation_22/Relu:activations:0*
strides
*
ksize
*
paddingVALID*/
_output_shapes
:���������@|
dropout_12/IdentityIdentity!max_pooling2d_14/MaxPool:output:0*
T0*/
_output_shapes
:���������@�
conv2d_20/Conv2D/ReadVariableOpReadVariableOp(conv2d_20_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:@@�
conv2d_20/Conv2DConv2Ddropout_12/Identity:output:0'conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*/
_output_shapes
:���������@�
 conv2d_20/BiasAdd/ReadVariableOpReadVariableOp)conv2d_20_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@�
conv2d_20/BiasAddBiasAddconv2d_20/Conv2D:output:0(conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@l
conv2d_20/ReluReluconv2d_20/BiasAdd:output:0*
T0*/
_output_shapes
:���������@r
activation_23/ReluReluconv2d_20/Relu:activations:0*
T0*/
_output_shapes
:���������@�
max_pooling2d_15/MaxPoolMaxPool activation_23/Relu:activations:0*
strides
*
ksize
*
paddingVALID*/
_output_shapes
:���������@|
dropout_13/IdentityIdentity!max_pooling2d_15/MaxPool:output:0*
T0*/
_output_shapes
:���������@h
flatten_2/Reshape/shapeConst*
valueB"����@   *
dtype0*
_output_shapes
:�
flatten_2/ReshapeReshapedropout_13/Identity:output:0 flatten_2/Reshape/shape:output:0*
T0*'
_output_shapes
:���������@�
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@��
dense_4/MatMulMatMulflatten_2/Reshape:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:��
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������g
activation_24/ReluReludense_4/BiasAdd:output:0*
T0*(
_output_shapes
:����������t
dropout_14/IdentityIdentity activation_24/Relu:activations:0*
T0*(
_output_shapes
:�����������
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	�&�
dense_5/MatMulMatMuldropout_14/Identity:output:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������&�
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:&�
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������&l
activation_25/SoftmaxSoftmaxdense_5/BiasAdd:output:0*
T0*'
_output_shapes
:���������&�
IdentityIdentityactivation_25/Softmax:softmax:0!^conv2d_16/BiasAdd/ReadVariableOp ^conv2d_16/Conv2D/ReadVariableOp!^conv2d_17/BiasAdd/ReadVariableOp ^conv2d_17/Conv2D/ReadVariableOp!^conv2d_18/BiasAdd/ReadVariableOp ^conv2d_18/Conv2D/ReadVariableOp!^conv2d_19/BiasAdd/ReadVariableOp ^conv2d_19/Conv2D/ReadVariableOp!^conv2d_20/BiasAdd/ReadVariableOp ^conv2d_20/Conv2D/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������&"
identityIdentity:output:0*f
_input_shapesU
S:���������22::::::::::::::2B
conv2d_20/Conv2D/ReadVariableOpconv2d_20/Conv2D/ReadVariableOp2D
 conv2d_19/BiasAdd/ReadVariableOp conv2d_19/BiasAdd/ReadVariableOp2B
conv2d_19/Conv2D/ReadVariableOpconv2d_19/Conv2D/ReadVariableOp2B
conv2d_16/Conv2D/ReadVariableOpconv2d_16/Conv2D/ReadVariableOp2D
 conv2d_17/BiasAdd/ReadVariableOp conv2d_17/BiasAdd/ReadVariableOp2D
 conv2d_20/BiasAdd/ReadVariableOp conv2d_20/BiasAdd/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2B
conv2d_17/Conv2D/ReadVariableOpconv2d_17/Conv2D/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2D
 conv2d_18/BiasAdd/ReadVariableOp conv2d_18/BiasAdd/ReadVariableOp2B
conv2d_18/Conv2D/ReadVariableOpconv2d_18/Conv2D/ReadVariableOp2D
 conv2d_16/BiasAdd/ReadVariableOp conv2d_16/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp: : :& "
 
_user_specified_nameinputs: : : : :
 : : : : : :	 : 
�
�
E__inference_conv2d_18_layer_call_and_return_conditional_losses_122285

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
: @�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*A
_output_shapes/
-:+���������������������������@�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������@�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+���������������������������@"
identityIdentity:output:0*H
_input_shapes7
5:+��������������������������� ::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
�	
�
C__inference_dense_4_layer_call_and_return_conditional_losses_122673

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@�j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:�w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*.
_input_shapes
:���������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
�
e
I__inference_activation_20_layer_call_and_return_conditional_losses_122432

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:���������.. b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������.. "
identityIdentity:output:0*.
_input_shapes
:���������.. :& "
 
_user_specified_nameinputs
��
�	
H__inference_sequential_3_layer_call_and_return_conditional_losses_123119

inputs,
(conv2d_16_conv2d_readvariableop_resource-
)conv2d_16_biasadd_readvariableop_resource,
(conv2d_17_conv2d_readvariableop_resource-
)conv2d_17_biasadd_readvariableop_resource,
(conv2d_18_conv2d_readvariableop_resource-
)conv2d_18_biasadd_readvariableop_resource,
(conv2d_19_conv2d_readvariableop_resource-
)conv2d_19_biasadd_readvariableop_resource,
(conv2d_20_conv2d_readvariableop_resource-
)conv2d_20_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource
identity�� conv2d_16/BiasAdd/ReadVariableOp�conv2d_16/Conv2D/ReadVariableOp� conv2d_17/BiasAdd/ReadVariableOp�conv2d_17/Conv2D/ReadVariableOp� conv2d_18/BiasAdd/ReadVariableOp�conv2d_18/Conv2D/ReadVariableOp� conv2d_19/BiasAdd/ReadVariableOp�conv2d_19/Conv2D/ReadVariableOp� conv2d_20/BiasAdd/ReadVariableOp�conv2d_20/Conv2D/ReadVariableOp�dense_4/BiasAdd/ReadVariableOp�dense_4/MatMul/ReadVariableOp�dense_5/BiasAdd/ReadVariableOp�dense_5/MatMul/ReadVariableOp�
conv2d_16/Conv2D/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
: �
conv2d_16/Conv2DConv2Dinputs'conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*/
_output_shapes
:���������00 �
 conv2d_16/BiasAdd/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
: �
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0(conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������00 p
activation_19/ReluReluconv2d_16/BiasAdd:output:0*
T0*/
_output_shapes
:���������00 �
conv2d_17/Conv2D/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:  �
conv2d_17/Conv2DConv2D activation_19/Relu:activations:0'conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*/
_output_shapes
:���������.. �
 conv2d_17/BiasAdd/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
: �
conv2d_17/BiasAddBiasAddconv2d_17/Conv2D:output:0(conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������.. p
activation_20/ReluReluconv2d_17/BiasAdd:output:0*
T0*/
_output_shapes
:���������.. �
max_pooling2d_12/MaxPoolMaxPool activation_20/Relu:activations:0*
strides
*
ksize
*
paddingVALID*/
_output_shapes
:��������� �
conv2d_18/Conv2D/ReadVariableOpReadVariableOp(conv2d_18_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
: @�
conv2d_18/Conv2DConv2D!max_pooling2d_12/MaxPool:output:0'conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*/
_output_shapes
:���������@�
 conv2d_18/BiasAdd/ReadVariableOpReadVariableOp)conv2d_18_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@�
conv2d_18/BiasAddBiasAddconv2d_18/Conv2D:output:0(conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@l
conv2d_18/ReluReluconv2d_18/BiasAdd:output:0*
T0*/
_output_shapes
:���������@r
activation_21/ReluReluconv2d_18/Relu:activations:0*
T0*/
_output_shapes
:���������@�
max_pooling2d_13/MaxPoolMaxPool activation_21/Relu:activations:0*
strides
*
ksize
*
paddingVALID*/
_output_shapes
:���������

@\
dropout_11/dropout/rateConst*
valueB
 *  �>*
dtype0*
_output_shapes
: i
dropout_11/dropout/ShapeShape!max_pooling2d_13/MaxPool:output:0*
T0*
_output_shapes
:j
%dropout_11/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: j
%dropout_11/dropout/random_uniform/maxConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
/dropout_11/dropout/random_uniform/RandomUniformRandomUniform!dropout_11/dropout/Shape:output:0*
T0*
dtype0*/
_output_shapes
:���������

@�
%dropout_11/dropout/random_uniform/subSub.dropout_11/dropout/random_uniform/max:output:0.dropout_11/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: �
%dropout_11/dropout/random_uniform/mulMul8dropout_11/dropout/random_uniform/RandomUniform:output:0)dropout_11/dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:���������

@�
!dropout_11/dropout/random_uniformAdd)dropout_11/dropout/random_uniform/mul:z:0.dropout_11/dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:���������

@]
dropout_11/dropout/sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
dropout_11/dropout/subSub!dropout_11/dropout/sub/x:output:0 dropout_11/dropout/rate:output:0*
T0*
_output_shapes
: a
dropout_11/dropout/truediv/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
dropout_11/dropout/truedivRealDiv%dropout_11/dropout/truediv/x:output:0dropout_11/dropout/sub:z:0*
T0*
_output_shapes
: �
dropout_11/dropout/GreaterEqualGreaterEqual%dropout_11/dropout/random_uniform:z:0 dropout_11/dropout/rate:output:0*
T0*/
_output_shapes
:���������

@�
dropout_11/dropout/mulMul!max_pooling2d_13/MaxPool:output:0dropout_11/dropout/truediv:z:0*
T0*/
_output_shapes
:���������

@�
dropout_11/dropout/CastCast#dropout_11/dropout/GreaterEqual:z:0*

SrcT0
*

DstT0*/
_output_shapes
:���������

@�
dropout_11/dropout/mul_1Muldropout_11/dropout/mul:z:0dropout_11/dropout/Cast:y:0*
T0*/
_output_shapes
:���������

@�
conv2d_19/Conv2D/ReadVariableOpReadVariableOp(conv2d_19_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:@@�
conv2d_19/Conv2DConv2Ddropout_11/dropout/mul_1:z:0'conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*/
_output_shapes
:���������@�
 conv2d_19/BiasAdd/ReadVariableOpReadVariableOp)conv2d_19_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@�
conv2d_19/BiasAddBiasAddconv2d_19/Conv2D:output:0(conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@l
conv2d_19/ReluReluconv2d_19/BiasAdd:output:0*
T0*/
_output_shapes
:���������@r
activation_22/ReluReluconv2d_19/Relu:activations:0*
T0*/
_output_shapes
:���������@�
max_pooling2d_14/MaxPoolMaxPool activation_22/Relu:activations:0*
strides
*
ksize
*
paddingVALID*/
_output_shapes
:���������@\
dropout_12/dropout/rateConst*
valueB
 *  �>*
dtype0*
_output_shapes
: i
dropout_12/dropout/ShapeShape!max_pooling2d_14/MaxPool:output:0*
T0*
_output_shapes
:j
%dropout_12/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: j
%dropout_12/dropout/random_uniform/maxConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
/dropout_12/dropout/random_uniform/RandomUniformRandomUniform!dropout_12/dropout/Shape:output:0*
T0*
dtype0*/
_output_shapes
:���������@�
%dropout_12/dropout/random_uniform/subSub.dropout_12/dropout/random_uniform/max:output:0.dropout_12/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: �
%dropout_12/dropout/random_uniform/mulMul8dropout_12/dropout/random_uniform/RandomUniform:output:0)dropout_12/dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:���������@�
!dropout_12/dropout/random_uniformAdd)dropout_12/dropout/random_uniform/mul:z:0.dropout_12/dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:���������@]
dropout_12/dropout/sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
dropout_12/dropout/subSub!dropout_12/dropout/sub/x:output:0 dropout_12/dropout/rate:output:0*
T0*
_output_shapes
: a
dropout_12/dropout/truediv/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
dropout_12/dropout/truedivRealDiv%dropout_12/dropout/truediv/x:output:0dropout_12/dropout/sub:z:0*
T0*
_output_shapes
: �
dropout_12/dropout/GreaterEqualGreaterEqual%dropout_12/dropout/random_uniform:z:0 dropout_12/dropout/rate:output:0*
T0*/
_output_shapes
:���������@�
dropout_12/dropout/mulMul!max_pooling2d_14/MaxPool:output:0dropout_12/dropout/truediv:z:0*
T0*/
_output_shapes
:���������@�
dropout_12/dropout/CastCast#dropout_12/dropout/GreaterEqual:z:0*

SrcT0
*

DstT0*/
_output_shapes
:���������@�
dropout_12/dropout/mul_1Muldropout_12/dropout/mul:z:0dropout_12/dropout/Cast:y:0*
T0*/
_output_shapes
:���������@�
conv2d_20/Conv2D/ReadVariableOpReadVariableOp(conv2d_20_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:@@�
conv2d_20/Conv2DConv2Ddropout_12/dropout/mul_1:z:0'conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*/
_output_shapes
:���������@�
 conv2d_20/BiasAdd/ReadVariableOpReadVariableOp)conv2d_20_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@�
conv2d_20/BiasAddBiasAddconv2d_20/Conv2D:output:0(conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@l
conv2d_20/ReluReluconv2d_20/BiasAdd:output:0*
T0*/
_output_shapes
:���������@r
activation_23/ReluReluconv2d_20/Relu:activations:0*
T0*/
_output_shapes
:���������@�
max_pooling2d_15/MaxPoolMaxPool activation_23/Relu:activations:0*
strides
*
ksize
*
paddingVALID*/
_output_shapes
:���������@\
dropout_13/dropout/rateConst*
valueB
 *  �>*
dtype0*
_output_shapes
: i
dropout_13/dropout/ShapeShape!max_pooling2d_15/MaxPool:output:0*
T0*
_output_shapes
:j
%dropout_13/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: j
%dropout_13/dropout/random_uniform/maxConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
/dropout_13/dropout/random_uniform/RandomUniformRandomUniform!dropout_13/dropout/Shape:output:0*
T0*
dtype0*/
_output_shapes
:���������@�
%dropout_13/dropout/random_uniform/subSub.dropout_13/dropout/random_uniform/max:output:0.dropout_13/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: �
%dropout_13/dropout/random_uniform/mulMul8dropout_13/dropout/random_uniform/RandomUniform:output:0)dropout_13/dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:���������@�
!dropout_13/dropout/random_uniformAdd)dropout_13/dropout/random_uniform/mul:z:0.dropout_13/dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:���������@]
dropout_13/dropout/sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
dropout_13/dropout/subSub!dropout_13/dropout/sub/x:output:0 dropout_13/dropout/rate:output:0*
T0*
_output_shapes
: a
dropout_13/dropout/truediv/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
dropout_13/dropout/truedivRealDiv%dropout_13/dropout/truediv/x:output:0dropout_13/dropout/sub:z:0*
T0*
_output_shapes
: �
dropout_13/dropout/GreaterEqualGreaterEqual%dropout_13/dropout/random_uniform:z:0 dropout_13/dropout/rate:output:0*
T0*/
_output_shapes
:���������@�
dropout_13/dropout/mulMul!max_pooling2d_15/MaxPool:output:0dropout_13/dropout/truediv:z:0*
T0*/
_output_shapes
:���������@�
dropout_13/dropout/CastCast#dropout_13/dropout/GreaterEqual:z:0*

SrcT0
*

DstT0*/
_output_shapes
:���������@�
dropout_13/dropout/mul_1Muldropout_13/dropout/mul:z:0dropout_13/dropout/Cast:y:0*
T0*/
_output_shapes
:���������@h
flatten_2/Reshape/shapeConst*
valueB"����@   *
dtype0*
_output_shapes
:�
flatten_2/ReshapeReshapedropout_13/dropout/mul_1:z:0 flatten_2/Reshape/shape:output:0*
T0*'
_output_shapes
:���������@�
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@��
dense_4/MatMulMatMulflatten_2/Reshape:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:��
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������g
activation_24/ReluReludense_4/BiasAdd:output:0*
T0*(
_output_shapes
:����������\
dropout_14/dropout/rateConst*
valueB
 *   ?*
dtype0*
_output_shapes
: h
dropout_14/dropout/ShapeShape activation_24/Relu:activations:0*
T0*
_output_shapes
:j
%dropout_14/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: j
%dropout_14/dropout/random_uniform/maxConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
/dropout_14/dropout/random_uniform/RandomUniformRandomUniform!dropout_14/dropout/Shape:output:0*
T0*
dtype0*(
_output_shapes
:�����������
%dropout_14/dropout/random_uniform/subSub.dropout_14/dropout/random_uniform/max:output:0.dropout_14/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: �
%dropout_14/dropout/random_uniform/mulMul8dropout_14/dropout/random_uniform/RandomUniform:output:0)dropout_14/dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:�����������
!dropout_14/dropout/random_uniformAdd)dropout_14/dropout/random_uniform/mul:z:0.dropout_14/dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:����������]
dropout_14/dropout/sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
dropout_14/dropout/subSub!dropout_14/dropout/sub/x:output:0 dropout_14/dropout/rate:output:0*
T0*
_output_shapes
: a
dropout_14/dropout/truediv/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
dropout_14/dropout/truedivRealDiv%dropout_14/dropout/truediv/x:output:0dropout_14/dropout/sub:z:0*
T0*
_output_shapes
: �
dropout_14/dropout/GreaterEqualGreaterEqual%dropout_14/dropout/random_uniform:z:0 dropout_14/dropout/rate:output:0*
T0*(
_output_shapes
:�����������
dropout_14/dropout/mulMul activation_24/Relu:activations:0dropout_14/dropout/truediv:z:0*
T0*(
_output_shapes
:�����������
dropout_14/dropout/CastCast#dropout_14/dropout/GreaterEqual:z:0*

SrcT0
*

DstT0*(
_output_shapes
:�����������
dropout_14/dropout/mul_1Muldropout_14/dropout/mul:z:0dropout_14/dropout/Cast:y:0*
T0*(
_output_shapes
:�����������
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	�&�
dense_5/MatMulMatMuldropout_14/dropout/mul_1:z:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������&�
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:&�
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������&l
activation_25/SoftmaxSoftmaxdense_5/BiasAdd:output:0*
T0*'
_output_shapes
:���������&�
IdentityIdentityactivation_25/Softmax:softmax:0!^conv2d_16/BiasAdd/ReadVariableOp ^conv2d_16/Conv2D/ReadVariableOp!^conv2d_17/BiasAdd/ReadVariableOp ^conv2d_17/Conv2D/ReadVariableOp!^conv2d_18/BiasAdd/ReadVariableOp ^conv2d_18/Conv2D/ReadVariableOp!^conv2d_19/BiasAdd/ReadVariableOp ^conv2d_19/Conv2D/ReadVariableOp!^conv2d_20/BiasAdd/ReadVariableOp ^conv2d_20/Conv2D/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������&"
identityIdentity:output:0*f
_input_shapesU
S:���������22::::::::::::::2B
conv2d_20/Conv2D/ReadVariableOpconv2d_20/Conv2D/ReadVariableOp2B
conv2d_19/Conv2D/ReadVariableOpconv2d_19/Conv2D/ReadVariableOp2D
 conv2d_19/BiasAdd/ReadVariableOp conv2d_19/BiasAdd/ReadVariableOp2B
conv2d_16/Conv2D/ReadVariableOpconv2d_16/Conv2D/ReadVariableOp2D
 conv2d_17/BiasAdd/ReadVariableOp conv2d_17/BiasAdd/ReadVariableOp2D
 conv2d_20/BiasAdd/ReadVariableOp conv2d_20/BiasAdd/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2B
conv2d_17/Conv2D/ReadVariableOpconv2d_17/Conv2D/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2D
 conv2d_18/BiasAdd/ReadVariableOp conv2d_18/BiasAdd/ReadVariableOp2B
conv2d_18/Conv2D/ReadVariableOpconv2d_18/Conv2D/ReadVariableOp2D
 conv2d_16/BiasAdd/ReadVariableOp conv2d_16/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp: : :& "
 
_user_specified_nameinputs: : : : :
 : : : : : :	 : 
�
d
F__inference_dropout_11_layer_call_and_return_conditional_losses_123278

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:���������

@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������

@"!

identity_1Identity_1:output:0*.
_input_shapes
:���������

@:& "
 
_user_specified_nameinputs
�
M
1__inference_max_pooling2d_12_layer_call_fn_122271

inputs
identity�
PartitionedCallPartitionedCallinputs*-
_gradient_op_typePartitionedCall-122268*U
fPRN
L__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_122262*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*J
_output_shapes8
6:4�������������������������������������
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:& "
 
_user_specified_nameinputs
�
e
I__inference_activation_19_layer_call_and_return_conditional_losses_122411

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:���������00 b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������00 "
identityIdentity:output:0*.
_input_shapes
:���������00 :& "
 
_user_specified_nameinputs
�
�
E__inference_conv2d_20_layer_call_and_return_conditional_losses_122369

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:@@�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*A
_output_shapes/
-:+���������������������������@�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������@�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+���������������������������@"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������@::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
�
G
+__inference_dropout_13_layer_call_fn_123378

inputs
identity�
PartitionedCallPartitionedCallinputs*-
_gradient_op_typePartitionedCall-122639*O
fJRH
F__inference_dropout_13_layer_call_and_return_conditional_losses_122627*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������@h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�
d
F__inference_dropout_13_layer_call_and_return_conditional_losses_123368

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:���������@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������@"!

identity_1Identity_1:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�
a
E__inference_flatten_2_layer_call_and_return_conditional_losses_123384

inputs
identity^
Reshape/shapeConst*
valueB"����@   *
dtype0*
_output_shapes
:d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:���������@X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�
e
F__inference_dropout_12_layer_call_and_return_conditional_losses_123318

inputs
identity�Q
dropout/rateConst*
valueB
 *  �>*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: _
dropout/random_uniform/maxConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*/
_output_shapes
:���������@�
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: �
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:���������@�
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:���������@R
dropout/sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: �
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*/
_output_shapes
:���������@i
dropout/mulMulinputsdropout/truediv:z:0*
T0*/
_output_shapes
:���������@w
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*/
_output_shapes
:���������@q
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:���������@a
IdentityIdentitydropout/mul_1:z:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�
J
.__inference_activation_24_layer_call_fn_123416

inputs
identity�
PartitionedCallPartitionedCallinputs*-
_gradient_op_typePartitionedCall-122701*R
fMRK
I__inference_activation_24_layer_call_and_return_conditional_losses_122695*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*(
_output_shapes
:����������a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*'
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�
d
+__inference_dropout_13_layer_call_fn_123373

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*-
_gradient_op_typePartitionedCall-122631*O
fJRH
F__inference_dropout_13_layer_call_and_return_conditional_losses_122620*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������@�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*.
_input_shapes
:���������@22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs"wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*�
serving_default�
S
conv2d_16_input@
!serving_default_conv2d_16_input:0���������22A
activation_250
StatefulPartitionedCall:0���������&tensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:Ͱ
�k
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer-8

layer-9
layer_with_weights-3
layer-10
layer-11
layer-12
layer-13
layer_with_weights-4
layer-14
layer-15
layer-16
layer-17
layer-18
layer_with_weights-5
layer-19
layer-20
layer-21
layer_with_weights-6
layer-22
layer-23
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
�__call__
+�&call_and_return_all_conditional_losses
�_default_save_signature"�e
_tf_keras_sequential�e{"class_name": "Sequential", "name": "sequential_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential_3", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d_16", "trainable": true, "batch_input_shape": [null, 50, 50, 1], "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_19", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_17", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_20", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_12", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_18", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_21", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_13", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_11", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_19", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_22", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_14", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_12", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_20", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_23", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_15", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_13", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_24", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Dropout", "config": {"name": "dropout_14", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 38, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_25", "trainable": true, "dtype": "float32", "activation": "softmax"}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_3", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d_16", "trainable": true, "batch_input_shape": [null, 50, 50, 1], "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_19", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_17", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_20", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_12", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_18", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_21", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_13", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_11", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_19", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_22", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_14", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_12", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_20", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_23", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_15", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_13", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_24", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Dropout", "config": {"name": "dropout_14", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 38, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_25", "trainable": true, "dtype": "float32", "activation": "softmax"}}]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�
	variables
 regularization_losses
!trainable_variables
"	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "InputLayer", "name": "conv2d_16_input", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 50, 50, 1], "config": {"batch_input_shape": [null, 50, 50, 1], "dtype": "float32", "sparse": false, "name": "conv2d_16_input"}}
�

#kernel
$bias
%	variables
&regularization_losses
'trainable_variables
(	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 50, 50, 1], "config": {"name": "conv2d_16", "trainable": true, "batch_input_shape": [null, 50, 50, 1], "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}}
�
)	variables
*regularization_losses
+trainable_variables
,	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Activation", "name": "activation_19", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "activation_19", "trainable": true, "dtype": "float32", "activation": "relu"}}
�

-kernel
.bias
/	variables
0regularization_losses
1trainable_variables
2	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_17", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}}
�
3	variables
4regularization_losses
5trainable_variables
6	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Activation", "name": "activation_20", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "activation_20", "trainable": true, "dtype": "float32", "activation": "relu"}}
�
7	variables
8regularization_losses
9trainable_variables
:	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "max_pooling2d_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d_12", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�

;kernel
<bias
=	variables
>regularization_losses
?trainable_variables
@	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_18", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}}
�
A	variables
Bregularization_losses
Ctrainable_variables
D	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Activation", "name": "activation_21", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "activation_21", "trainable": true, "dtype": "float32", "activation": "relu"}}
�
E	variables
Fregularization_losses
Gtrainable_variables
H	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "max_pooling2d_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d_13", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
I	variables
Jregularization_losses
Ktrainable_variables
L	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dropout", "name": "dropout_11", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_11", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}
�

Mkernel
Nbias
O	variables
Pregularization_losses
Qtrainable_variables
R	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_19", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_19", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
�
S	variables
Tregularization_losses
Utrainable_variables
V	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Activation", "name": "activation_22", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "activation_22", "trainable": true, "dtype": "float32", "activation": "relu"}}
�
W	variables
Xregularization_losses
Ytrainable_variables
Z	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "max_pooling2d_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d_14", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
[	variables
\regularization_losses
]trainable_variables
^	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dropout", "name": "dropout_12", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_12", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}
�

_kernel
`bias
a	variables
bregularization_losses
ctrainable_variables
d	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_20", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_20", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
�
e	variables
fregularization_losses
gtrainable_variables
h	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Activation", "name": "activation_23", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "activation_23", "trainable": true, "dtype": "float32", "activation": "relu"}}
�
i	variables
jregularization_losses
ktrainable_variables
l	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "max_pooling2d_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d_15", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
m	variables
nregularization_losses
otrainable_variables
p	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dropout", "name": "dropout_13", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_13", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}
�
q	variables
rregularization_losses
strainable_variables
t	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Flatten", "name": "flatten_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
�

ukernel
vbias
w	variables
xregularization_losses
ytrainable_variables
z	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}}
�
{	variables
|regularization_losses
}trainable_variables
~	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Activation", "name": "activation_24", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "activation_24", "trainable": true, "dtype": "float32", "activation": "relu"}}
�
	variables
�regularization_losses
�trainable_variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dropout", "name": "dropout_14", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_14", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
�
�kernel
	�bias
�	variables
�regularization_losses
�trainable_variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 38, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}
�
�	variables
�regularization_losses
�trainable_variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Activation", "name": "activation_25", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "activation_25", "trainable": true, "dtype": "float32", "activation": "softmax"}}
�
	�iter
�beta_1
�beta_2

�decay
�learning_rate#m�$m�-m�.m�;m�<m�Mm�Nm�_m�`m�um�vm�	�m�	�m�#v�$v�-v�.v�;v�<v�Mv�Nv�_v�`v�uv�vv�	�v�	�v�"
	optimizer
�
#0
$1
-2
.3
;4
<5
M6
N7
_8
`9
u10
v11
�12
�13"
trackable_list_wrapper
 "
trackable_list_wrapper
�
#0
$1
-2
.3
;4
<5
M6
N7
_8
`9
u10
v11
�12
�13"
trackable_list_wrapper
�
�metrics
	variables
regularization_losses
 �layer_regularization_losses
�non_trainable_variables
�layers
trainable_variables
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
	variables
 regularization_losses
 �layer_regularization_losses
�non_trainable_variables
�layers
!trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
*:( 2conv2d_16/kernel
: 2conv2d_16/bias
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
�
�metrics
%	variables
&regularization_losses
 �layer_regularization_losses
�non_trainable_variables
�layers
'trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
)	variables
*regularization_losses
 �layer_regularization_losses
�non_trainable_variables
�layers
+trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
*:(  2conv2d_17/kernel
: 2conv2d_17/bias
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
�
�metrics
/	variables
0regularization_losses
 �layer_regularization_losses
�non_trainable_variables
�layers
1trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
3	variables
4regularization_losses
 �layer_regularization_losses
�non_trainable_variables
�layers
5trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
7	variables
8regularization_losses
 �layer_regularization_losses
�non_trainable_variables
�layers
9trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
*:( @2conv2d_18/kernel
:@2conv2d_18/bias
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
�
�metrics
=	variables
>regularization_losses
 �layer_regularization_losses
�non_trainable_variables
�layers
?trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
A	variables
Bregularization_losses
 �layer_regularization_losses
�non_trainable_variables
�layers
Ctrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
E	variables
Fregularization_losses
 �layer_regularization_losses
�non_trainable_variables
�layers
Gtrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
I	variables
Jregularization_losses
 �layer_regularization_losses
�non_trainable_variables
�layers
Ktrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
*:(@@2conv2d_19/kernel
:@2conv2d_19/bias
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
�
�metrics
O	variables
Pregularization_losses
 �layer_regularization_losses
�non_trainable_variables
�layers
Qtrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
S	variables
Tregularization_losses
 �layer_regularization_losses
�non_trainable_variables
�layers
Utrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
W	variables
Xregularization_losses
 �layer_regularization_losses
�non_trainable_variables
�layers
Ytrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
[	variables
\regularization_losses
 �layer_regularization_losses
�non_trainable_variables
�layers
]trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
*:(@@2conv2d_20/kernel
:@2conv2d_20/bias
.
_0
`1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
�
�metrics
a	variables
bregularization_losses
 �layer_regularization_losses
�non_trainable_variables
�layers
ctrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
e	variables
fregularization_losses
 �layer_regularization_losses
�non_trainable_variables
�layers
gtrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
i	variables
jregularization_losses
 �layer_regularization_losses
�non_trainable_variables
�layers
ktrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
m	variables
nregularization_losses
 �layer_regularization_losses
�non_trainable_variables
�layers
otrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
q	variables
rregularization_losses
 �layer_regularization_losses
�non_trainable_variables
�layers
strainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
!:	@�2dense_4/kernel
:�2dense_4/bias
.
u0
v1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
u0
v1"
trackable_list_wrapper
�
�metrics
w	variables
xregularization_losses
 �layer_regularization_losses
�non_trainable_variables
�layers
ytrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
{	variables
|regularization_losses
 �layer_regularization_losses
�non_trainable_variables
�layers
}trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
	variables
�regularization_losses
 �layer_regularization_losses
�non_trainable_variables
�layers
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
!:	�&2dense_5/kernel
:&2dense_5/bias
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�metrics
�	variables
�regularization_losses
 �layer_regularization_losses
�non_trainable_variables
�layers
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
�	variables
�regularization_losses
 �layer_regularization_losses
�non_trainable_variables
�layers
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
0
1
2
3
4
5
6
	7

8
9
10
11
12
13
14
15
16
17
18
19
20
21
22"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�

�total

�count
�
_fn_kwargs
�	variables
�regularization_losses
�trainable_variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "MeanMetricWrapper", "name": "accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "accuracy", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
�	variables
�regularization_losses
 �layer_regularization_losses
�non_trainable_variables
�layers
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
/:- 2Adam/conv2d_16/kernel/m
!: 2Adam/conv2d_16/bias/m
/:-  2Adam/conv2d_17/kernel/m
!: 2Adam/conv2d_17/bias/m
/:- @2Adam/conv2d_18/kernel/m
!:@2Adam/conv2d_18/bias/m
/:-@@2Adam/conv2d_19/kernel/m
!:@2Adam/conv2d_19/bias/m
/:-@@2Adam/conv2d_20/kernel/m
!:@2Adam/conv2d_20/bias/m
&:$	@�2Adam/dense_4/kernel/m
 :�2Adam/dense_4/bias/m
&:$	�&2Adam/dense_5/kernel/m
:&2Adam/dense_5/bias/m
/:- 2Adam/conv2d_16/kernel/v
!: 2Adam/conv2d_16/bias/v
/:-  2Adam/conv2d_17/kernel/v
!: 2Adam/conv2d_17/bias/v
/:- @2Adam/conv2d_18/kernel/v
!:@2Adam/conv2d_18/bias/v
/:-@@2Adam/conv2d_19/kernel/v
!:@2Adam/conv2d_19/bias/v
/:-@@2Adam/conv2d_20/kernel/v
!:@2Adam/conv2d_20/bias/v
&:$	@�2Adam/dense_4/kernel/v
 :�2Adam/dense_4/bias/v
&:$	�&2Adam/dense_5/kernel/v
:&2Adam/dense_5/bias/v
�2�
-__inference_sequential_3_layer_call_fn_123223
-__inference_sequential_3_layer_call_fn_122966
-__inference_sequential_3_layer_call_fn_122903
-__inference_sequential_3_layer_call_fn_123204�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
H__inference_sequential_3_layer_call_and_return_conditional_losses_122798
H__inference_sequential_3_layer_call_and_return_conditional_losses_123119
H__inference_sequential_3_layer_call_and_return_conditional_losses_123185
H__inference_sequential_3_layer_call_and_return_conditional_losses_122841�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
!__inference__wrapped_model_122206�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *6�3
1�.
conv2d_16_input���������22
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2�
*__inference_conv2d_16_layer_call_fn_122230�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������
�2�
E__inference_conv2d_16_layer_call_and_return_conditional_losses_122219�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������
�2�
.__inference_activation_19_layer_call_fn_123233�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
I__inference_activation_19_layer_call_and_return_conditional_losses_123228�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_conv2d_17_layer_call_fn_122254�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+��������������������������� 
�2�
E__inference_conv2d_17_layer_call_and_return_conditional_losses_122243�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+��������������������������� 
�2�
.__inference_activation_20_layer_call_fn_123243�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
I__inference_activation_20_layer_call_and_return_conditional_losses_123238�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
1__inference_max_pooling2d_12_layer_call_fn_122271�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
L__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_122262�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
*__inference_conv2d_18_layer_call_fn_122296�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+��������������������������� 
�2�
E__inference_conv2d_18_layer_call_and_return_conditional_losses_122285�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+��������������������������� 
�2�
.__inference_activation_21_layer_call_fn_123253�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
I__inference_activation_21_layer_call_and_return_conditional_losses_123248�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
1__inference_max_pooling2d_13_layer_call_fn_122313�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
L__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_122304�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
+__inference_dropout_11_layer_call_fn_123283
+__inference_dropout_11_layer_call_fn_123288�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
F__inference_dropout_11_layer_call_and_return_conditional_losses_123273
F__inference_dropout_11_layer_call_and_return_conditional_losses_123278�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
*__inference_conv2d_19_layer_call_fn_122338�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������@
�2�
E__inference_conv2d_19_layer_call_and_return_conditional_losses_122327�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������@
�2�
.__inference_activation_22_layer_call_fn_123298�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
I__inference_activation_22_layer_call_and_return_conditional_losses_123293�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
1__inference_max_pooling2d_14_layer_call_fn_122355�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
L__inference_max_pooling2d_14_layer_call_and_return_conditional_losses_122346�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
+__inference_dropout_12_layer_call_fn_123328
+__inference_dropout_12_layer_call_fn_123333�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
F__inference_dropout_12_layer_call_and_return_conditional_losses_123323
F__inference_dropout_12_layer_call_and_return_conditional_losses_123318�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
*__inference_conv2d_20_layer_call_fn_122380�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������@
�2�
E__inference_conv2d_20_layer_call_and_return_conditional_losses_122369�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������@
�2�
.__inference_activation_23_layer_call_fn_123343�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
I__inference_activation_23_layer_call_and_return_conditional_losses_123338�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
1__inference_max_pooling2d_15_layer_call_fn_122397�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
L__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_122388�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
+__inference_dropout_13_layer_call_fn_123378
+__inference_dropout_13_layer_call_fn_123373�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
F__inference_dropout_13_layer_call_and_return_conditional_losses_123368
F__inference_dropout_13_layer_call_and_return_conditional_losses_123363�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
*__inference_flatten_2_layer_call_fn_123389�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_flatten_2_layer_call_and_return_conditional_losses_123384�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_dense_4_layer_call_fn_123406�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_dense_4_layer_call_and_return_conditional_losses_123399�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
.__inference_activation_24_layer_call_fn_123416�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
I__inference_activation_24_layer_call_and_return_conditional_losses_123411�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_dropout_14_layer_call_fn_123446
+__inference_dropout_14_layer_call_fn_123451�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
F__inference_dropout_14_layer_call_and_return_conditional_losses_123441
F__inference_dropout_14_layer_call_and_return_conditional_losses_123436�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
(__inference_dense_5_layer_call_fn_123468�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_dense_5_layer_call_and_return_conditional_losses_123461�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
.__inference_activation_25_layer_call_fn_123478�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
I__inference_activation_25_layer_call_and_return_conditional_losses_123473�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
;B9
$__inference_signature_wrapper_122991conv2d_16_input
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 �
H__inference_sequential_3_layer_call_and_return_conditional_losses_123119z#$-.;<MN_`uv��?�<
5�2
(�%
inputs���������22
p

 
� "%�"
�
0���������&
� �
I__inference_activation_19_layer_call_and_return_conditional_losses_123228h7�4
-�*
(�%
inputs���������00 
� "-�*
#� 
0���������00 
� �
I__inference_activation_20_layer_call_and_return_conditional_losses_123238h7�4
-�*
(�%
inputs���������.. 
� "-�*
#� 
0���������.. 
� ~
(__inference_dense_5_layer_call_fn_123468R��0�-
&�#
!�
inputs����������
� "����������&�
I__inference_activation_25_layer_call_and_return_conditional_losses_123473X/�,
%�"
 �
inputs���������&
� "%�"
�
0���������&
� �
$__inference_signature_wrapper_122991�#$-.;<MN_`uv��S�P
� 
I�F
D
conv2d_16_input1�.
conv2d_16_input���������22"=�:
8
activation_25'�$
activation_25���������&�
E__inference_conv2d_17_layer_call_and_return_conditional_losses_122243�-.I�F
?�<
:�7
inputs+��������������������������� 
� "?�<
5�2
0+��������������������������� 
� �
*__inference_conv2d_20_layer_call_fn_122380�_`I�F
?�<
:�7
inputs+���������������������������@
� "2�/+���������������������������@�
F__inference_dropout_11_layer_call_and_return_conditional_losses_123273l;�8
1�.
(�%
inputs���������

@
p
� "-�*
#� 
0���������

@
� �
L__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_122304�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
F__inference_dropout_11_layer_call_and_return_conditional_losses_123278l;�8
1�.
(�%
inputs���������

@
p 
� "-�*
#� 
0���������

@
� �
H__inference_sequential_3_layer_call_and_return_conditional_losses_123185z#$-.;<MN_`uv��?�<
5�2
(�%
inputs���������22
p 

 
� "%�"
�
0���������&
� �
I__inference_activation_22_layer_call_and_return_conditional_losses_123293h7�4
-�*
(�%
inputs���������@
� "-�*
#� 
0���������@
� �
-__inference_sequential_3_layer_call_fn_122966v#$-.;<MN_`uv��H�E
>�;
1�.
conv2d_16_input���������22
p 

 
� "����������&�
.__inference_activation_23_layer_call_fn_123343[7�4
-�*
(�%
inputs���������@
� " ����������@�
E__inference_conv2d_19_layer_call_and_return_conditional_losses_122327�MNI�F
?�<
:�7
inputs+���������������������������@
� "?�<
5�2
0+���������������������������@
� 
.__inference_activation_24_layer_call_fn_123416M0�-
&�#
!�
inputs����������
� "������������
.__inference_activation_21_layer_call_fn_123253[7�4
-�*
(�%
inputs���������@
� " ����������@�
+__inference_dropout_12_layer_call_fn_123333_;�8
1�.
(�%
inputs���������@
p 
� " ����������@�
+__inference_dropout_12_layer_call_fn_123328_;�8
1�.
(�%
inputs���������@
p
� " ����������@�
C__inference_dense_4_layer_call_and_return_conditional_losses_123399]uv/�,
%�"
 �
inputs���������@
� "&�#
�
0����������
� �
I__inference_activation_24_layer_call_and_return_conditional_losses_123411Z0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� �
.__inference_activation_20_layer_call_fn_123243[7�4
-�*
(�%
inputs���������.. 
� " ����������.. �
!__inference__wrapped_model_122206�#$-.;<MN_`uv��@�=
6�3
1�.
conv2d_16_input���������22
� "=�:
8
activation_25'�$
activation_25���������&�
+__inference_dropout_13_layer_call_fn_123373_;�8
1�.
(�%
inputs���������@
p
� " ����������@�
F__inference_dropout_13_layer_call_and_return_conditional_losses_123363l;�8
1�.
(�%
inputs���������@
p
� "-�*
#� 
0���������@
� �
+__inference_dropout_11_layer_call_fn_123283_;�8
1�.
(�%
inputs���������

@
p
� " ����������

@�
+__inference_dropout_14_layer_call_fn_123446Q4�1
*�'
!�
inputs����������
p
� "������������
+__inference_dropout_14_layer_call_fn_123451Q4�1
*�'
!�
inputs����������
p 
� "������������
.__inference_activation_22_layer_call_fn_123298[7�4
-�*
(�%
inputs���������@
� " ����������@�
F__inference_dropout_13_layer_call_and_return_conditional_losses_123368l;�8
1�.
(�%
inputs���������@
p 
� "-�*
#� 
0���������@
� �
+__inference_dropout_13_layer_call_fn_123378_;�8
1�.
(�%
inputs���������@
p 
� " ����������@�
E__inference_conv2d_20_layer_call_and_return_conditional_losses_122369�_`I�F
?�<
:�7
inputs+���������������������������@
� "?�<
5�2
0+���������������������������@
� �
+__inference_dropout_11_layer_call_fn_123288_;�8
1�.
(�%
inputs���������

@
p 
� " ����������

@�
H__inference_sequential_3_layer_call_and_return_conditional_losses_122841�#$-.;<MN_`uv��H�E
>�;
1�.
conv2d_16_input���������22
p 

 
� "%�"
�
0���������&
� }
.__inference_activation_25_layer_call_fn_123478K/�,
%�"
 �
inputs���������&
� "����������&�
H__inference_sequential_3_layer_call_and_return_conditional_losses_122798�#$-.;<MN_`uv��H�E
>�;
1�.
conv2d_16_input���������22
p

 
� "%�"
�
0���������&
� �
E__inference_conv2d_16_layer_call_and_return_conditional_losses_122219�#$I�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+��������������������������� 
� �
*__inference_flatten_2_layer_call_fn_123389S7�4
-�*
(�%
inputs���������@
� "����������@�
I__inference_activation_21_layer_call_and_return_conditional_losses_123248h7�4
-�*
(�%
inputs���������@
� "-�*
#� 
0���������@
� �
L__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_122388�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
-__inference_sequential_3_layer_call_fn_123204m#$-.;<MN_`uv��?�<
5�2
(�%
inputs���������22
p

 
� "����������&�
1__inference_max_pooling2d_13_layer_call_fn_122313�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
-__inference_sequential_3_layer_call_fn_123223m#$-.;<MN_`uv��?�<
5�2
(�%
inputs���������22
p 

 
� "����������&�
*__inference_conv2d_16_layer_call_fn_122230�#$I�F
?�<
:�7
inputs+���������������������������
� "2�/+��������������������������� �
*__inference_conv2d_19_layer_call_fn_122338�MNI�F
?�<
:�7
inputs+���������������������������@
� "2�/+���������������������������@�
F__inference_dropout_12_layer_call_and_return_conditional_losses_123323l;�8
1�.
(�%
inputs���������@
p 
� "-�*
#� 
0���������@
� �
F__inference_dropout_12_layer_call_and_return_conditional_losses_123318l;�8
1�.
(�%
inputs���������@
p
� "-�*
#� 
0���������@
� �
*__inference_conv2d_17_layer_call_fn_122254�-.I�F
?�<
:�7
inputs+��������������������������� 
� "2�/+��������������������������� �
L__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_122262�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
1__inference_max_pooling2d_14_layer_call_fn_122355�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
I__inference_activation_23_layer_call_and_return_conditional_losses_123338h7�4
-�*
(�%
inputs���������@
� "-�*
#� 
0���������@
� �
1__inference_max_pooling2d_12_layer_call_fn_122271�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
E__inference_conv2d_18_layer_call_and_return_conditional_losses_122285�;<I�F
?�<
:�7
inputs+��������������������������� 
� "?�<
5�2
0+���������������������������@
� �
*__inference_conv2d_18_layer_call_fn_122296�;<I�F
?�<
:�7
inputs+��������������������������� 
� "2�/+���������������������������@�
1__inference_max_pooling2d_15_layer_call_fn_122397�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
C__inference_dense_5_layer_call_and_return_conditional_losses_123461_��0�-
&�#
!�
inputs����������
� "%�"
�
0���������&
� �
E__inference_flatten_2_layer_call_and_return_conditional_losses_123384`7�4
-�*
(�%
inputs���������@
� "%�"
�
0���������@
� �
L__inference_max_pooling2d_14_layer_call_and_return_conditional_losses_122346�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� |
(__inference_dense_4_layer_call_fn_123406Puv/�,
%�"
 �
inputs���������@
� "������������
.__inference_activation_19_layer_call_fn_123233[7�4
-�*
(�%
inputs���������00 
� " ����������00 �
F__inference_dropout_14_layer_call_and_return_conditional_losses_123441^4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
F__inference_dropout_14_layer_call_and_return_conditional_losses_123436^4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
-__inference_sequential_3_layer_call_fn_122903v#$-.;<MN_`uv��H�E
>�;
1�.
conv2d_16_input���������22
p

 
� "����������&