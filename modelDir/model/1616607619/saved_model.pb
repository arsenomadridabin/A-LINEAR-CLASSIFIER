źŽ
˝
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	

ArgMax

input"T
	dimension"Tidx
output"output_type"!
Ttype:
2	
"
Tidxtype0:
2	"
output_typetype0	:
2	
¸
AsString

input"T

output"
Ttype:
2		
"
	precisionint˙˙˙˙˙˙˙˙˙"

scientificbool( "
shortestbool( "
widthint˙˙˙˙˙˙˙˙˙"
fillstring 
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
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
ş
If
cond"Tcond
input2Tin
output2Tout"
Tcondtype"
Tin
list(type)("
Tout
list(type)("
then_branchfunc"
else_branchfunc" 
output_shapeslist(shape)
 
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
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

ParseExampleV2

serialized	
names
sparse_keys

dense_keys
ragged_keys
dense_defaults2Tdense
sparse_indices	*
num_sparse
sparse_values2sparse_types
sparse_shapes	*
num_sparse
dense_values2Tdense#
ragged_values2ragged_value_types'
ragged_row_splits2ragged_split_types"
Tdense
list(type)(:
2	"

num_sparseint("%
sparse_types
list(type)(:
2	"+
ragged_value_types
list(type)(:
2	"*
ragged_split_types
list(type)(:
2	"
dense_shapeslist(shape)(
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
b
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:

2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
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
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
9
Softmax
logits"T
softmax"T"
Ttype:
2
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 
9
VarIsInitializedOp
resource
is_initialized

&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.4.12v2.4.0-49-g85c8b2a817f8ě­

global_step/Initializer/zerosConst*
_class
loc:@global_step*
_output_shapes
: *
dtype0	*
value	B	 R 

global_stepVarHandleOp*
_class
loc:@global_step*
_output_shapes
: *
dtype0	*
shape: *
shared_nameglobal_step
g
,global_step/IsInitialized/VarIsInitializedOpVarIsInitializedOpglobal_step*
_output_shapes
: 
_
global_step/AssignAssignVariableOpglobal_stepglobal_step/Initializer/zeros*
dtype0	
c
global_step/Read/ReadVariableOpReadVariableOpglobal_step*
_output_shapes
: *
dtype0	
o
input_example_tensorPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
U
ParseExample/ConstConst*
_output_shapes
: *
dtype0*
valueB 
W
ParseExample/Const_1Const*
_output_shapes
: *
dtype0*
valueB 
W
ParseExample/Const_2Const*
_output_shapes
: *
dtype0*
valueB 
W
ParseExample/Const_3Const*
_output_shapes
: *
dtype0*
valueB 
W
ParseExample/Const_4Const*
_output_shapes
: *
dtype0*
valueB 
W
ParseExample/Const_5Const*
_output_shapes
: *
dtype0*
valueB 
W
ParseExample/Const_6Const*
_output_shapes
: *
dtype0*
valueB 
W
ParseExample/Const_7Const*
_output_shapes
: *
dtype0*
valueB 
W
ParseExample/Const_8Const*
_output_shapes
: *
dtype0*
valueB 
W
ParseExample/Const_9Const*
_output_shapes
: *
dtype0*
valueB 
X
ParseExample/Const_10Const*
_output_shapes
: *
dtype0*
valueB 
X
ParseExample/Const_11Const*
_output_shapes
: *
dtype0*
valueB 
X
ParseExample/Const_12Const*
_output_shapes
: *
dtype0*
valueB 
X
ParseExample/Const_13Const*
_output_shapes
: *
dtype0*
valueB 
X
ParseExample/Const_14Const*
_output_shapes
: *
dtype0*
valueB 
X
ParseExample/Const_15Const*
_output_shapes
: *
dtype0*
valueB 
X
ParseExample/Const_16Const*
_output_shapes
: *
dtype0*
valueB 
X
ParseExample/Const_17Const*
_output_shapes
: *
dtype0*
valueB 
X
ParseExample/Const_18Const*
_output_shapes
: *
dtype0*
valueB 
X
ParseExample/Const_19Const*
_output_shapes
: *
dtype0*
valueB 
X
ParseExample/Const_20Const*
_output_shapes
: *
dtype0*
valueB 
X
ParseExample/Const_21Const*
_output_shapes
: *
dtype0*
valueB 
X
ParseExample/Const_22Const*
_output_shapes
: *
dtype0*
valueB 
X
ParseExample/Const_23Const*
_output_shapes
: *
dtype0*
valueB 
d
!ParseExample/ParseExampleV2/namesConst*
_output_shapes
: *
dtype0*
valueB 
j
'ParseExample/ParseExampleV2/sparse_keysConst*
_output_shapes
: *
dtype0*
valueB 

&ParseExample/ParseExampleV2/dense_keysConst*
_output_shapes
:*
dtype0*Ź
value˘BB	Feature 1B
Feature 10B
Feature 11B
Feature 12B
Feature 13B
Feature 14B
Feature 15B
Feature 16B
Feature 17B
Feature 18B
Feature 19B	Feature 2B
Feature 20B
Feature 21B
Feature 22B
Feature 23B
Feature 24B	Feature 3B	Feature 4B	Feature 5B	Feature 6B	Feature 7B	Feature 8B	Feature 9
j
'ParseExample/ParseExampleV2/ragged_keysConst*
_output_shapes
: *
dtype0*
valueB 

ParseExample/ParseExampleV2ParseExampleV2input_example_tensor!ParseExample/ParseExampleV2/names'ParseExample/ParseExampleV2/sparse_keys&ParseExample/ParseExampleV2/dense_keys'ParseExample/ParseExampleV2/ragged_keysParseExample/ConstParseExample/Const_1ParseExample/Const_2ParseExample/Const_3ParseExample/Const_4ParseExample/Const_5ParseExample/Const_6ParseExample/Const_7ParseExample/Const_8ParseExample/Const_9ParseExample/Const_10ParseExample/Const_11ParseExample/Const_12ParseExample/Const_13ParseExample/Const_14ParseExample/Const_15ParseExample/Const_16ParseExample/Const_17ParseExample/Const_18ParseExample/Const_19ParseExample/Const_20ParseExample/Const_21ParseExample/Const_22ParseExample/Const_23*&
Tdense
2*Ţ
_output_shapesË
Č:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*¤
dense_shapes
::::::::::::::::::::::::*

num_sparse *
ragged_split_types
 *
ragged_value_types
 *
sparse_types
 
Ć
7linear/linear_model/Feature_1/weights/Initializer/zerosConst*8
_class.
,*loc:@linear/linear_model/Feature_1/weights*
_output_shapes

:*
dtype0*
valueB*    
ŕ
%linear/linear_model/Feature_1/weightsVarHandleOp*8
_class.
,*loc:@linear/linear_model/Feature_1/weights*
_output_shapes
: *
dtype0*
shape
:*6
shared_name'%linear/linear_model/Feature_1/weights

Flinear/linear_model/Feature_1/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp%linear/linear_model/Feature_1/weights*
_output_shapes
: 
­
,linear/linear_model/Feature_1/weights/AssignAssignVariableOp%linear/linear_model/Feature_1/weights7linear/linear_model/Feature_1/weights/Initializer/zeros*
dtype0

9linear/linear_model/Feature_1/weights/Read/ReadVariableOpReadVariableOp%linear/linear_model/Feature_1/weights*
_output_shapes

:*
dtype0
Č
8linear/linear_model/Feature_10/weights/Initializer/zerosConst*9
_class/
-+loc:@linear/linear_model/Feature_10/weights*
_output_shapes

:*
dtype0*
valueB*    
ă
&linear/linear_model/Feature_10/weightsVarHandleOp*9
_class/
-+loc:@linear/linear_model/Feature_10/weights*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&linear/linear_model/Feature_10/weights

Glinear/linear_model/Feature_10/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp&linear/linear_model/Feature_10/weights*
_output_shapes
: 
°
-linear/linear_model/Feature_10/weights/AssignAssignVariableOp&linear/linear_model/Feature_10/weights8linear/linear_model/Feature_10/weights/Initializer/zeros*
dtype0
Ą
:linear/linear_model/Feature_10/weights/Read/ReadVariableOpReadVariableOp&linear/linear_model/Feature_10/weights*
_output_shapes

:*
dtype0
Č
8linear/linear_model/Feature_11/weights/Initializer/zerosConst*9
_class/
-+loc:@linear/linear_model/Feature_11/weights*
_output_shapes

:*
dtype0*
valueB*    
ă
&linear/linear_model/Feature_11/weightsVarHandleOp*9
_class/
-+loc:@linear/linear_model/Feature_11/weights*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&linear/linear_model/Feature_11/weights

Glinear/linear_model/Feature_11/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp&linear/linear_model/Feature_11/weights*
_output_shapes
: 
°
-linear/linear_model/Feature_11/weights/AssignAssignVariableOp&linear/linear_model/Feature_11/weights8linear/linear_model/Feature_11/weights/Initializer/zeros*
dtype0
Ą
:linear/linear_model/Feature_11/weights/Read/ReadVariableOpReadVariableOp&linear/linear_model/Feature_11/weights*
_output_shapes

:*
dtype0
Č
8linear/linear_model/Feature_12/weights/Initializer/zerosConst*9
_class/
-+loc:@linear/linear_model/Feature_12/weights*
_output_shapes

:*
dtype0*
valueB*    
ă
&linear/linear_model/Feature_12/weightsVarHandleOp*9
_class/
-+loc:@linear/linear_model/Feature_12/weights*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&linear/linear_model/Feature_12/weights

Glinear/linear_model/Feature_12/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp&linear/linear_model/Feature_12/weights*
_output_shapes
: 
°
-linear/linear_model/Feature_12/weights/AssignAssignVariableOp&linear/linear_model/Feature_12/weights8linear/linear_model/Feature_12/weights/Initializer/zeros*
dtype0
Ą
:linear/linear_model/Feature_12/weights/Read/ReadVariableOpReadVariableOp&linear/linear_model/Feature_12/weights*
_output_shapes

:*
dtype0
Č
8linear/linear_model/Feature_13/weights/Initializer/zerosConst*9
_class/
-+loc:@linear/linear_model/Feature_13/weights*
_output_shapes

:*
dtype0*
valueB*    
ă
&linear/linear_model/Feature_13/weightsVarHandleOp*9
_class/
-+loc:@linear/linear_model/Feature_13/weights*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&linear/linear_model/Feature_13/weights

Glinear/linear_model/Feature_13/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp&linear/linear_model/Feature_13/weights*
_output_shapes
: 
°
-linear/linear_model/Feature_13/weights/AssignAssignVariableOp&linear/linear_model/Feature_13/weights8linear/linear_model/Feature_13/weights/Initializer/zeros*
dtype0
Ą
:linear/linear_model/Feature_13/weights/Read/ReadVariableOpReadVariableOp&linear/linear_model/Feature_13/weights*
_output_shapes

:*
dtype0
Č
8linear/linear_model/Feature_14/weights/Initializer/zerosConst*9
_class/
-+loc:@linear/linear_model/Feature_14/weights*
_output_shapes

:*
dtype0*
valueB*    
ă
&linear/linear_model/Feature_14/weightsVarHandleOp*9
_class/
-+loc:@linear/linear_model/Feature_14/weights*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&linear/linear_model/Feature_14/weights

Glinear/linear_model/Feature_14/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp&linear/linear_model/Feature_14/weights*
_output_shapes
: 
°
-linear/linear_model/Feature_14/weights/AssignAssignVariableOp&linear/linear_model/Feature_14/weights8linear/linear_model/Feature_14/weights/Initializer/zeros*
dtype0
Ą
:linear/linear_model/Feature_14/weights/Read/ReadVariableOpReadVariableOp&linear/linear_model/Feature_14/weights*
_output_shapes

:*
dtype0
Č
8linear/linear_model/Feature_15/weights/Initializer/zerosConst*9
_class/
-+loc:@linear/linear_model/Feature_15/weights*
_output_shapes

:*
dtype0*
valueB*    
ă
&linear/linear_model/Feature_15/weightsVarHandleOp*9
_class/
-+loc:@linear/linear_model/Feature_15/weights*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&linear/linear_model/Feature_15/weights

Glinear/linear_model/Feature_15/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp&linear/linear_model/Feature_15/weights*
_output_shapes
: 
°
-linear/linear_model/Feature_15/weights/AssignAssignVariableOp&linear/linear_model/Feature_15/weights8linear/linear_model/Feature_15/weights/Initializer/zeros*
dtype0
Ą
:linear/linear_model/Feature_15/weights/Read/ReadVariableOpReadVariableOp&linear/linear_model/Feature_15/weights*
_output_shapes

:*
dtype0
Č
8linear/linear_model/Feature_16/weights/Initializer/zerosConst*9
_class/
-+loc:@linear/linear_model/Feature_16/weights*
_output_shapes

:*
dtype0*
valueB*    
ă
&linear/linear_model/Feature_16/weightsVarHandleOp*9
_class/
-+loc:@linear/linear_model/Feature_16/weights*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&linear/linear_model/Feature_16/weights

Glinear/linear_model/Feature_16/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp&linear/linear_model/Feature_16/weights*
_output_shapes
: 
°
-linear/linear_model/Feature_16/weights/AssignAssignVariableOp&linear/linear_model/Feature_16/weights8linear/linear_model/Feature_16/weights/Initializer/zeros*
dtype0
Ą
:linear/linear_model/Feature_16/weights/Read/ReadVariableOpReadVariableOp&linear/linear_model/Feature_16/weights*
_output_shapes

:*
dtype0
Č
8linear/linear_model/Feature_17/weights/Initializer/zerosConst*9
_class/
-+loc:@linear/linear_model/Feature_17/weights*
_output_shapes

:*
dtype0*
valueB*    
ă
&linear/linear_model/Feature_17/weightsVarHandleOp*9
_class/
-+loc:@linear/linear_model/Feature_17/weights*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&linear/linear_model/Feature_17/weights

Glinear/linear_model/Feature_17/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp&linear/linear_model/Feature_17/weights*
_output_shapes
: 
°
-linear/linear_model/Feature_17/weights/AssignAssignVariableOp&linear/linear_model/Feature_17/weights8linear/linear_model/Feature_17/weights/Initializer/zeros*
dtype0
Ą
:linear/linear_model/Feature_17/weights/Read/ReadVariableOpReadVariableOp&linear/linear_model/Feature_17/weights*
_output_shapes

:*
dtype0
Č
8linear/linear_model/Feature_18/weights/Initializer/zerosConst*9
_class/
-+loc:@linear/linear_model/Feature_18/weights*
_output_shapes

:*
dtype0*
valueB*    
ă
&linear/linear_model/Feature_18/weightsVarHandleOp*9
_class/
-+loc:@linear/linear_model/Feature_18/weights*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&linear/linear_model/Feature_18/weights

Glinear/linear_model/Feature_18/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp&linear/linear_model/Feature_18/weights*
_output_shapes
: 
°
-linear/linear_model/Feature_18/weights/AssignAssignVariableOp&linear/linear_model/Feature_18/weights8linear/linear_model/Feature_18/weights/Initializer/zeros*
dtype0
Ą
:linear/linear_model/Feature_18/weights/Read/ReadVariableOpReadVariableOp&linear/linear_model/Feature_18/weights*
_output_shapes

:*
dtype0
Č
8linear/linear_model/Feature_19/weights/Initializer/zerosConst*9
_class/
-+loc:@linear/linear_model/Feature_19/weights*
_output_shapes

:*
dtype0*
valueB*    
ă
&linear/linear_model/Feature_19/weightsVarHandleOp*9
_class/
-+loc:@linear/linear_model/Feature_19/weights*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&linear/linear_model/Feature_19/weights

Glinear/linear_model/Feature_19/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp&linear/linear_model/Feature_19/weights*
_output_shapes
: 
°
-linear/linear_model/Feature_19/weights/AssignAssignVariableOp&linear/linear_model/Feature_19/weights8linear/linear_model/Feature_19/weights/Initializer/zeros*
dtype0
Ą
:linear/linear_model/Feature_19/weights/Read/ReadVariableOpReadVariableOp&linear/linear_model/Feature_19/weights*
_output_shapes

:*
dtype0
Ć
7linear/linear_model/Feature_2/weights/Initializer/zerosConst*8
_class.
,*loc:@linear/linear_model/Feature_2/weights*
_output_shapes

:*
dtype0*
valueB*    
ŕ
%linear/linear_model/Feature_2/weightsVarHandleOp*8
_class.
,*loc:@linear/linear_model/Feature_2/weights*
_output_shapes
: *
dtype0*
shape
:*6
shared_name'%linear/linear_model/Feature_2/weights

Flinear/linear_model/Feature_2/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp%linear/linear_model/Feature_2/weights*
_output_shapes
: 
­
,linear/linear_model/Feature_2/weights/AssignAssignVariableOp%linear/linear_model/Feature_2/weights7linear/linear_model/Feature_2/weights/Initializer/zeros*
dtype0

9linear/linear_model/Feature_2/weights/Read/ReadVariableOpReadVariableOp%linear/linear_model/Feature_2/weights*
_output_shapes

:*
dtype0
Č
8linear/linear_model/Feature_20/weights/Initializer/zerosConst*9
_class/
-+loc:@linear/linear_model/Feature_20/weights*
_output_shapes

:*
dtype0*
valueB*    
ă
&linear/linear_model/Feature_20/weightsVarHandleOp*9
_class/
-+loc:@linear/linear_model/Feature_20/weights*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&linear/linear_model/Feature_20/weights

Glinear/linear_model/Feature_20/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp&linear/linear_model/Feature_20/weights*
_output_shapes
: 
°
-linear/linear_model/Feature_20/weights/AssignAssignVariableOp&linear/linear_model/Feature_20/weights8linear/linear_model/Feature_20/weights/Initializer/zeros*
dtype0
Ą
:linear/linear_model/Feature_20/weights/Read/ReadVariableOpReadVariableOp&linear/linear_model/Feature_20/weights*
_output_shapes

:*
dtype0
Č
8linear/linear_model/Feature_21/weights/Initializer/zerosConst*9
_class/
-+loc:@linear/linear_model/Feature_21/weights*
_output_shapes

:*
dtype0*
valueB*    
ă
&linear/linear_model/Feature_21/weightsVarHandleOp*9
_class/
-+loc:@linear/linear_model/Feature_21/weights*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&linear/linear_model/Feature_21/weights

Glinear/linear_model/Feature_21/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp&linear/linear_model/Feature_21/weights*
_output_shapes
: 
°
-linear/linear_model/Feature_21/weights/AssignAssignVariableOp&linear/linear_model/Feature_21/weights8linear/linear_model/Feature_21/weights/Initializer/zeros*
dtype0
Ą
:linear/linear_model/Feature_21/weights/Read/ReadVariableOpReadVariableOp&linear/linear_model/Feature_21/weights*
_output_shapes

:*
dtype0
Č
8linear/linear_model/Feature_22/weights/Initializer/zerosConst*9
_class/
-+loc:@linear/linear_model/Feature_22/weights*
_output_shapes

:*
dtype0*
valueB*    
ă
&linear/linear_model/Feature_22/weightsVarHandleOp*9
_class/
-+loc:@linear/linear_model/Feature_22/weights*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&linear/linear_model/Feature_22/weights

Glinear/linear_model/Feature_22/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp&linear/linear_model/Feature_22/weights*
_output_shapes
: 
°
-linear/linear_model/Feature_22/weights/AssignAssignVariableOp&linear/linear_model/Feature_22/weights8linear/linear_model/Feature_22/weights/Initializer/zeros*
dtype0
Ą
:linear/linear_model/Feature_22/weights/Read/ReadVariableOpReadVariableOp&linear/linear_model/Feature_22/weights*
_output_shapes

:*
dtype0
Č
8linear/linear_model/Feature_23/weights/Initializer/zerosConst*9
_class/
-+loc:@linear/linear_model/Feature_23/weights*
_output_shapes

:*
dtype0*
valueB*    
ă
&linear/linear_model/Feature_23/weightsVarHandleOp*9
_class/
-+loc:@linear/linear_model/Feature_23/weights*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&linear/linear_model/Feature_23/weights

Glinear/linear_model/Feature_23/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp&linear/linear_model/Feature_23/weights*
_output_shapes
: 
°
-linear/linear_model/Feature_23/weights/AssignAssignVariableOp&linear/linear_model/Feature_23/weights8linear/linear_model/Feature_23/weights/Initializer/zeros*
dtype0
Ą
:linear/linear_model/Feature_23/weights/Read/ReadVariableOpReadVariableOp&linear/linear_model/Feature_23/weights*
_output_shapes

:*
dtype0
Č
8linear/linear_model/Feature_24/weights/Initializer/zerosConst*9
_class/
-+loc:@linear/linear_model/Feature_24/weights*
_output_shapes

:*
dtype0*
valueB*    
ă
&linear/linear_model/Feature_24/weightsVarHandleOp*9
_class/
-+loc:@linear/linear_model/Feature_24/weights*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&linear/linear_model/Feature_24/weights

Glinear/linear_model/Feature_24/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp&linear/linear_model/Feature_24/weights*
_output_shapes
: 
°
-linear/linear_model/Feature_24/weights/AssignAssignVariableOp&linear/linear_model/Feature_24/weights8linear/linear_model/Feature_24/weights/Initializer/zeros*
dtype0
Ą
:linear/linear_model/Feature_24/weights/Read/ReadVariableOpReadVariableOp&linear/linear_model/Feature_24/weights*
_output_shapes

:*
dtype0
Ć
7linear/linear_model/Feature_3/weights/Initializer/zerosConst*8
_class.
,*loc:@linear/linear_model/Feature_3/weights*
_output_shapes

:*
dtype0*
valueB*    
ŕ
%linear/linear_model/Feature_3/weightsVarHandleOp*8
_class.
,*loc:@linear/linear_model/Feature_3/weights*
_output_shapes
: *
dtype0*
shape
:*6
shared_name'%linear/linear_model/Feature_3/weights

Flinear/linear_model/Feature_3/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp%linear/linear_model/Feature_3/weights*
_output_shapes
: 
­
,linear/linear_model/Feature_3/weights/AssignAssignVariableOp%linear/linear_model/Feature_3/weights7linear/linear_model/Feature_3/weights/Initializer/zeros*
dtype0

9linear/linear_model/Feature_3/weights/Read/ReadVariableOpReadVariableOp%linear/linear_model/Feature_3/weights*
_output_shapes

:*
dtype0
Ć
7linear/linear_model/Feature_4/weights/Initializer/zerosConst*8
_class.
,*loc:@linear/linear_model/Feature_4/weights*
_output_shapes

:*
dtype0*
valueB*    
ŕ
%linear/linear_model/Feature_4/weightsVarHandleOp*8
_class.
,*loc:@linear/linear_model/Feature_4/weights*
_output_shapes
: *
dtype0*
shape
:*6
shared_name'%linear/linear_model/Feature_4/weights

Flinear/linear_model/Feature_4/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp%linear/linear_model/Feature_4/weights*
_output_shapes
: 
­
,linear/linear_model/Feature_4/weights/AssignAssignVariableOp%linear/linear_model/Feature_4/weights7linear/linear_model/Feature_4/weights/Initializer/zeros*
dtype0

9linear/linear_model/Feature_4/weights/Read/ReadVariableOpReadVariableOp%linear/linear_model/Feature_4/weights*
_output_shapes

:*
dtype0
Ć
7linear/linear_model/Feature_5/weights/Initializer/zerosConst*8
_class.
,*loc:@linear/linear_model/Feature_5/weights*
_output_shapes

:*
dtype0*
valueB*    
ŕ
%linear/linear_model/Feature_5/weightsVarHandleOp*8
_class.
,*loc:@linear/linear_model/Feature_5/weights*
_output_shapes
: *
dtype0*
shape
:*6
shared_name'%linear/linear_model/Feature_5/weights

Flinear/linear_model/Feature_5/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp%linear/linear_model/Feature_5/weights*
_output_shapes
: 
­
,linear/linear_model/Feature_5/weights/AssignAssignVariableOp%linear/linear_model/Feature_5/weights7linear/linear_model/Feature_5/weights/Initializer/zeros*
dtype0

9linear/linear_model/Feature_5/weights/Read/ReadVariableOpReadVariableOp%linear/linear_model/Feature_5/weights*
_output_shapes

:*
dtype0
Ć
7linear/linear_model/Feature_6/weights/Initializer/zerosConst*8
_class.
,*loc:@linear/linear_model/Feature_6/weights*
_output_shapes

:*
dtype0*
valueB*    
ŕ
%linear/linear_model/Feature_6/weightsVarHandleOp*8
_class.
,*loc:@linear/linear_model/Feature_6/weights*
_output_shapes
: *
dtype0*
shape
:*6
shared_name'%linear/linear_model/Feature_6/weights

Flinear/linear_model/Feature_6/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp%linear/linear_model/Feature_6/weights*
_output_shapes
: 
­
,linear/linear_model/Feature_6/weights/AssignAssignVariableOp%linear/linear_model/Feature_6/weights7linear/linear_model/Feature_6/weights/Initializer/zeros*
dtype0

9linear/linear_model/Feature_6/weights/Read/ReadVariableOpReadVariableOp%linear/linear_model/Feature_6/weights*
_output_shapes

:*
dtype0
Ć
7linear/linear_model/Feature_7/weights/Initializer/zerosConst*8
_class.
,*loc:@linear/linear_model/Feature_7/weights*
_output_shapes

:*
dtype0*
valueB*    
ŕ
%linear/linear_model/Feature_7/weightsVarHandleOp*8
_class.
,*loc:@linear/linear_model/Feature_7/weights*
_output_shapes
: *
dtype0*
shape
:*6
shared_name'%linear/linear_model/Feature_7/weights

Flinear/linear_model/Feature_7/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp%linear/linear_model/Feature_7/weights*
_output_shapes
: 
­
,linear/linear_model/Feature_7/weights/AssignAssignVariableOp%linear/linear_model/Feature_7/weights7linear/linear_model/Feature_7/weights/Initializer/zeros*
dtype0

9linear/linear_model/Feature_7/weights/Read/ReadVariableOpReadVariableOp%linear/linear_model/Feature_7/weights*
_output_shapes

:*
dtype0
Ć
7linear/linear_model/Feature_8/weights/Initializer/zerosConst*8
_class.
,*loc:@linear/linear_model/Feature_8/weights*
_output_shapes

:*
dtype0*
valueB*    
ŕ
%linear/linear_model/Feature_8/weightsVarHandleOp*8
_class.
,*loc:@linear/linear_model/Feature_8/weights*
_output_shapes
: *
dtype0*
shape
:*6
shared_name'%linear/linear_model/Feature_8/weights

Flinear/linear_model/Feature_8/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp%linear/linear_model/Feature_8/weights*
_output_shapes
: 
­
,linear/linear_model/Feature_8/weights/AssignAssignVariableOp%linear/linear_model/Feature_8/weights7linear/linear_model/Feature_8/weights/Initializer/zeros*
dtype0

9linear/linear_model/Feature_8/weights/Read/ReadVariableOpReadVariableOp%linear/linear_model/Feature_8/weights*
_output_shapes

:*
dtype0
Ć
7linear/linear_model/Feature_9/weights/Initializer/zerosConst*8
_class.
,*loc:@linear/linear_model/Feature_9/weights*
_output_shapes

:*
dtype0*
valueB*    
ŕ
%linear/linear_model/Feature_9/weightsVarHandleOp*8
_class.
,*loc:@linear/linear_model/Feature_9/weights*
_output_shapes
: *
dtype0*
shape
:*6
shared_name'%linear/linear_model/Feature_9/weights

Flinear/linear_model/Feature_9/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp%linear/linear_model/Feature_9/weights*
_output_shapes
: 
­
,linear/linear_model/Feature_9/weights/AssignAssignVariableOp%linear/linear_model/Feature_9/weights7linear/linear_model/Feature_9/weights/Initializer/zeros*
dtype0

9linear/linear_model/Feature_9/weights/Read/ReadVariableOpReadVariableOp%linear/linear_model/Feature_9/weights*
_output_shapes

:*
dtype0
´
2linear/linear_model/bias_weights/Initializer/zerosConst*3
_class)
'%loc:@linear/linear_model/bias_weights*
_output_shapes
:*
dtype0*
valueB*    
Í
 linear/linear_model/bias_weightsVarHandleOp*3
_class)
'%loc:@linear/linear_model/bias_weights*
_output_shapes
: *
dtype0*
shape:*1
shared_name" linear/linear_model/bias_weights

Alinear/linear_model/bias_weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp linear/linear_model/bias_weights*
_output_shapes
: 

'linear/linear_model/bias_weights/AssignAssignVariableOp linear/linear_model/bias_weights2linear/linear_model/bias_weights/Initializer/zeros*
dtype0

4linear/linear_model/bias_weights/Read/ReadVariableOpReadVariableOp linear/linear_model/bias_weights*
_output_shapes
:*
dtype0

Klinear/linear_model/linear/linear_model/linear/linear_model/Feature_1/ShapeShapeParseExample/ParseExampleV2*
T0*
_output_shapes
:
Ł
Ylinear/linear_model/linear/linear_model/linear/linear_model/Feature_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Ľ
[linear/linear_model/linear/linear_model/linear/linear_model/Feature_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Ľ
[linear/linear_model/linear/linear_model/linear/linear_model/Feature_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:

Slinear/linear_model/linear/linear_model/linear/linear_model/Feature_1/strided_sliceStridedSliceKlinear/linear_model/linear/linear_model/linear/linear_model/Feature_1/ShapeYlinear/linear_model/linear/linear_model/linear/linear_model/Feature_1/strided_slice/stack[linear/linear_model/linear/linear_model/linear/linear_model/Feature_1/strided_slice/stack_1[linear/linear_model/linear/linear_model/linear/linear_model/Feature_1/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask

Ulinear/linear_model/linear/linear_model/linear/linear_model/Feature_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
ľ
Slinear/linear_model/linear/linear_model/linear/linear_model/Feature_1/Reshape/shapePackSlinear/linear_model/linear/linear_model/linear/linear_model/Feature_1/strided_sliceUlinear/linear_model/linear/linear_model/linear/linear_model/Feature_1/Reshape/shape/1*
N*
T0*
_output_shapes
:
ü
Mlinear/linear_model/linear/linear_model/linear/linear_model/Feature_1/ReshapeReshapeParseExample/ParseExampleV2Slinear/linear_model/linear/linear_model/linear/linear_model/Feature_1/Reshape/shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ç
alinear/linear_model/linear/linear_model/linear/linear_model/Feature_1/weighted_sum/ReadVariableOpReadVariableOp%linear/linear_model/Feature_1/weights*
_output_shapes

:*
dtype0
Ŕ
Rlinear/linear_model/linear/linear_model/linear/linear_model/Feature_1/weighted_sumMatMulMlinear/linear_model/linear/linear_model/linear/linear_model/Feature_1/Reshapealinear/linear_model/linear/linear_model/linear/linear_model/Feature_1/weighted_sum/ReadVariableOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Llinear/linear_model/linear/linear_model/linear/linear_model/Feature_10/ShapeShapeParseExample/ParseExampleV2:1*
T0*
_output_shapes
:
¤
Zlinear/linear_model/linear/linear_model/linear/linear_model/Feature_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Ś
\linear/linear_model/linear/linear_model/linear/linear_model/Feature_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Ś
\linear/linear_model/linear/linear_model/linear/linear_model/Feature_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:

Tlinear/linear_model/linear/linear_model/linear/linear_model/Feature_10/strided_sliceStridedSliceLlinear/linear_model/linear/linear_model/linear/linear_model/Feature_10/ShapeZlinear/linear_model/linear/linear_model/linear/linear_model/Feature_10/strided_slice/stack\linear/linear_model/linear/linear_model/linear/linear_model/Feature_10/strided_slice/stack_1\linear/linear_model/linear/linear_model/linear/linear_model/Feature_10/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask

Vlinear/linear_model/linear/linear_model/linear/linear_model/Feature_10/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
¸
Tlinear/linear_model/linear/linear_model/linear/linear_model/Feature_10/Reshape/shapePackTlinear/linear_model/linear/linear_model/linear/linear_model/Feature_10/strided_sliceVlinear/linear_model/linear/linear_model/linear/linear_model/Feature_10/Reshape/shape/1*
N*
T0*
_output_shapes
:

Nlinear/linear_model/linear/linear_model/linear/linear_model/Feature_10/ReshapeReshapeParseExample/ParseExampleV2:1Tlinear/linear_model/linear/linear_model/linear/linear_model/Feature_10/Reshape/shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
É
blinear/linear_model/linear/linear_model/linear/linear_model/Feature_10/weighted_sum/ReadVariableOpReadVariableOp&linear/linear_model/Feature_10/weights*
_output_shapes

:*
dtype0
Ă
Slinear/linear_model/linear/linear_model/linear/linear_model/Feature_10/weighted_sumMatMulNlinear/linear_model/linear/linear_model/linear/linear_model/Feature_10/Reshapeblinear/linear_model/linear/linear_model/linear/linear_model/Feature_10/weighted_sum/ReadVariableOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Llinear/linear_model/linear/linear_model/linear/linear_model/Feature_11/ShapeShapeParseExample/ParseExampleV2:2*
T0*
_output_shapes
:
¤
Zlinear/linear_model/linear/linear_model/linear/linear_model/Feature_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Ś
\linear/linear_model/linear/linear_model/linear/linear_model/Feature_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Ś
\linear/linear_model/linear/linear_model/linear/linear_model/Feature_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:

Tlinear/linear_model/linear/linear_model/linear/linear_model/Feature_11/strided_sliceStridedSliceLlinear/linear_model/linear/linear_model/linear/linear_model/Feature_11/ShapeZlinear/linear_model/linear/linear_model/linear/linear_model/Feature_11/strided_slice/stack\linear/linear_model/linear/linear_model/linear/linear_model/Feature_11/strided_slice/stack_1\linear/linear_model/linear/linear_model/linear/linear_model/Feature_11/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask

Vlinear/linear_model/linear/linear_model/linear/linear_model/Feature_11/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
¸
Tlinear/linear_model/linear/linear_model/linear/linear_model/Feature_11/Reshape/shapePackTlinear/linear_model/linear/linear_model/linear/linear_model/Feature_11/strided_sliceVlinear/linear_model/linear/linear_model/linear/linear_model/Feature_11/Reshape/shape/1*
N*
T0*
_output_shapes
:

Nlinear/linear_model/linear/linear_model/linear/linear_model/Feature_11/ReshapeReshapeParseExample/ParseExampleV2:2Tlinear/linear_model/linear/linear_model/linear/linear_model/Feature_11/Reshape/shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
É
blinear/linear_model/linear/linear_model/linear/linear_model/Feature_11/weighted_sum/ReadVariableOpReadVariableOp&linear/linear_model/Feature_11/weights*
_output_shapes

:*
dtype0
Ă
Slinear/linear_model/linear/linear_model/linear/linear_model/Feature_11/weighted_sumMatMulNlinear/linear_model/linear/linear_model/linear/linear_model/Feature_11/Reshapeblinear/linear_model/linear/linear_model/linear/linear_model/Feature_11/weighted_sum/ReadVariableOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Llinear/linear_model/linear/linear_model/linear/linear_model/Feature_12/ShapeShapeParseExample/ParseExampleV2:3*
T0*
_output_shapes
:
¤
Zlinear/linear_model/linear/linear_model/linear/linear_model/Feature_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Ś
\linear/linear_model/linear/linear_model/linear/linear_model/Feature_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Ś
\linear/linear_model/linear/linear_model/linear/linear_model/Feature_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:

Tlinear/linear_model/linear/linear_model/linear/linear_model/Feature_12/strided_sliceStridedSliceLlinear/linear_model/linear/linear_model/linear/linear_model/Feature_12/ShapeZlinear/linear_model/linear/linear_model/linear/linear_model/Feature_12/strided_slice/stack\linear/linear_model/linear/linear_model/linear/linear_model/Feature_12/strided_slice/stack_1\linear/linear_model/linear/linear_model/linear/linear_model/Feature_12/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask

Vlinear/linear_model/linear/linear_model/linear/linear_model/Feature_12/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
¸
Tlinear/linear_model/linear/linear_model/linear/linear_model/Feature_12/Reshape/shapePackTlinear/linear_model/linear/linear_model/linear/linear_model/Feature_12/strided_sliceVlinear/linear_model/linear/linear_model/linear/linear_model/Feature_12/Reshape/shape/1*
N*
T0*
_output_shapes
:

Nlinear/linear_model/linear/linear_model/linear/linear_model/Feature_12/ReshapeReshapeParseExample/ParseExampleV2:3Tlinear/linear_model/linear/linear_model/linear/linear_model/Feature_12/Reshape/shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
É
blinear/linear_model/linear/linear_model/linear/linear_model/Feature_12/weighted_sum/ReadVariableOpReadVariableOp&linear/linear_model/Feature_12/weights*
_output_shapes

:*
dtype0
Ă
Slinear/linear_model/linear/linear_model/linear/linear_model/Feature_12/weighted_sumMatMulNlinear/linear_model/linear/linear_model/linear/linear_model/Feature_12/Reshapeblinear/linear_model/linear/linear_model/linear/linear_model/Feature_12/weighted_sum/ReadVariableOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Llinear/linear_model/linear/linear_model/linear/linear_model/Feature_13/ShapeShapeParseExample/ParseExampleV2:4*
T0*
_output_shapes
:
¤
Zlinear/linear_model/linear/linear_model/linear/linear_model/Feature_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Ś
\linear/linear_model/linear/linear_model/linear/linear_model/Feature_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Ś
\linear/linear_model/linear/linear_model/linear/linear_model/Feature_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:

Tlinear/linear_model/linear/linear_model/linear/linear_model/Feature_13/strided_sliceStridedSliceLlinear/linear_model/linear/linear_model/linear/linear_model/Feature_13/ShapeZlinear/linear_model/linear/linear_model/linear/linear_model/Feature_13/strided_slice/stack\linear/linear_model/linear/linear_model/linear/linear_model/Feature_13/strided_slice/stack_1\linear/linear_model/linear/linear_model/linear/linear_model/Feature_13/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask

Vlinear/linear_model/linear/linear_model/linear/linear_model/Feature_13/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
¸
Tlinear/linear_model/linear/linear_model/linear/linear_model/Feature_13/Reshape/shapePackTlinear/linear_model/linear/linear_model/linear/linear_model/Feature_13/strided_sliceVlinear/linear_model/linear/linear_model/linear/linear_model/Feature_13/Reshape/shape/1*
N*
T0*
_output_shapes
:

Nlinear/linear_model/linear/linear_model/linear/linear_model/Feature_13/ReshapeReshapeParseExample/ParseExampleV2:4Tlinear/linear_model/linear/linear_model/linear/linear_model/Feature_13/Reshape/shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
É
blinear/linear_model/linear/linear_model/linear/linear_model/Feature_13/weighted_sum/ReadVariableOpReadVariableOp&linear/linear_model/Feature_13/weights*
_output_shapes

:*
dtype0
Ă
Slinear/linear_model/linear/linear_model/linear/linear_model/Feature_13/weighted_sumMatMulNlinear/linear_model/linear/linear_model/linear/linear_model/Feature_13/Reshapeblinear/linear_model/linear/linear_model/linear/linear_model/Feature_13/weighted_sum/ReadVariableOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Llinear/linear_model/linear/linear_model/linear/linear_model/Feature_14/ShapeShapeParseExample/ParseExampleV2:5*
T0*
_output_shapes
:
¤
Zlinear/linear_model/linear/linear_model/linear/linear_model/Feature_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Ś
\linear/linear_model/linear/linear_model/linear/linear_model/Feature_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Ś
\linear/linear_model/linear/linear_model/linear/linear_model/Feature_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:

Tlinear/linear_model/linear/linear_model/linear/linear_model/Feature_14/strided_sliceStridedSliceLlinear/linear_model/linear/linear_model/linear/linear_model/Feature_14/ShapeZlinear/linear_model/linear/linear_model/linear/linear_model/Feature_14/strided_slice/stack\linear/linear_model/linear/linear_model/linear/linear_model/Feature_14/strided_slice/stack_1\linear/linear_model/linear/linear_model/linear/linear_model/Feature_14/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask

Vlinear/linear_model/linear/linear_model/linear/linear_model/Feature_14/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
¸
Tlinear/linear_model/linear/linear_model/linear/linear_model/Feature_14/Reshape/shapePackTlinear/linear_model/linear/linear_model/linear/linear_model/Feature_14/strided_sliceVlinear/linear_model/linear/linear_model/linear/linear_model/Feature_14/Reshape/shape/1*
N*
T0*
_output_shapes
:

Nlinear/linear_model/linear/linear_model/linear/linear_model/Feature_14/ReshapeReshapeParseExample/ParseExampleV2:5Tlinear/linear_model/linear/linear_model/linear/linear_model/Feature_14/Reshape/shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
É
blinear/linear_model/linear/linear_model/linear/linear_model/Feature_14/weighted_sum/ReadVariableOpReadVariableOp&linear/linear_model/Feature_14/weights*
_output_shapes

:*
dtype0
Ă
Slinear/linear_model/linear/linear_model/linear/linear_model/Feature_14/weighted_sumMatMulNlinear/linear_model/linear/linear_model/linear/linear_model/Feature_14/Reshapeblinear/linear_model/linear/linear_model/linear/linear_model/Feature_14/weighted_sum/ReadVariableOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Llinear/linear_model/linear/linear_model/linear/linear_model/Feature_15/ShapeShapeParseExample/ParseExampleV2:6*
T0*
_output_shapes
:
¤
Zlinear/linear_model/linear/linear_model/linear/linear_model/Feature_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Ś
\linear/linear_model/linear/linear_model/linear/linear_model/Feature_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Ś
\linear/linear_model/linear/linear_model/linear/linear_model/Feature_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:

Tlinear/linear_model/linear/linear_model/linear/linear_model/Feature_15/strided_sliceStridedSliceLlinear/linear_model/linear/linear_model/linear/linear_model/Feature_15/ShapeZlinear/linear_model/linear/linear_model/linear/linear_model/Feature_15/strided_slice/stack\linear/linear_model/linear/linear_model/linear/linear_model/Feature_15/strided_slice/stack_1\linear/linear_model/linear/linear_model/linear/linear_model/Feature_15/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask

Vlinear/linear_model/linear/linear_model/linear/linear_model/Feature_15/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
¸
Tlinear/linear_model/linear/linear_model/linear/linear_model/Feature_15/Reshape/shapePackTlinear/linear_model/linear/linear_model/linear/linear_model/Feature_15/strided_sliceVlinear/linear_model/linear/linear_model/linear/linear_model/Feature_15/Reshape/shape/1*
N*
T0*
_output_shapes
:

Nlinear/linear_model/linear/linear_model/linear/linear_model/Feature_15/ReshapeReshapeParseExample/ParseExampleV2:6Tlinear/linear_model/linear/linear_model/linear/linear_model/Feature_15/Reshape/shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
É
blinear/linear_model/linear/linear_model/linear/linear_model/Feature_15/weighted_sum/ReadVariableOpReadVariableOp&linear/linear_model/Feature_15/weights*
_output_shapes

:*
dtype0
Ă
Slinear/linear_model/linear/linear_model/linear/linear_model/Feature_15/weighted_sumMatMulNlinear/linear_model/linear/linear_model/linear/linear_model/Feature_15/Reshapeblinear/linear_model/linear/linear_model/linear/linear_model/Feature_15/weighted_sum/ReadVariableOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Llinear/linear_model/linear/linear_model/linear/linear_model/Feature_16/ShapeShapeParseExample/ParseExampleV2:7*
T0*
_output_shapes
:
¤
Zlinear/linear_model/linear/linear_model/linear/linear_model/Feature_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Ś
\linear/linear_model/linear/linear_model/linear/linear_model/Feature_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Ś
\linear/linear_model/linear/linear_model/linear/linear_model/Feature_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:

Tlinear/linear_model/linear/linear_model/linear/linear_model/Feature_16/strided_sliceStridedSliceLlinear/linear_model/linear/linear_model/linear/linear_model/Feature_16/ShapeZlinear/linear_model/linear/linear_model/linear/linear_model/Feature_16/strided_slice/stack\linear/linear_model/linear/linear_model/linear/linear_model/Feature_16/strided_slice/stack_1\linear/linear_model/linear/linear_model/linear/linear_model/Feature_16/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask

Vlinear/linear_model/linear/linear_model/linear/linear_model/Feature_16/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
¸
Tlinear/linear_model/linear/linear_model/linear/linear_model/Feature_16/Reshape/shapePackTlinear/linear_model/linear/linear_model/linear/linear_model/Feature_16/strided_sliceVlinear/linear_model/linear/linear_model/linear/linear_model/Feature_16/Reshape/shape/1*
N*
T0*
_output_shapes
:

Nlinear/linear_model/linear/linear_model/linear/linear_model/Feature_16/ReshapeReshapeParseExample/ParseExampleV2:7Tlinear/linear_model/linear/linear_model/linear/linear_model/Feature_16/Reshape/shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
É
blinear/linear_model/linear/linear_model/linear/linear_model/Feature_16/weighted_sum/ReadVariableOpReadVariableOp&linear/linear_model/Feature_16/weights*
_output_shapes

:*
dtype0
Ă
Slinear/linear_model/linear/linear_model/linear/linear_model/Feature_16/weighted_sumMatMulNlinear/linear_model/linear/linear_model/linear/linear_model/Feature_16/Reshapeblinear/linear_model/linear/linear_model/linear/linear_model/Feature_16/weighted_sum/ReadVariableOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Llinear/linear_model/linear/linear_model/linear/linear_model/Feature_17/ShapeShapeParseExample/ParseExampleV2:8*
T0*
_output_shapes
:
¤
Zlinear/linear_model/linear/linear_model/linear/linear_model/Feature_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Ś
\linear/linear_model/linear/linear_model/linear/linear_model/Feature_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Ś
\linear/linear_model/linear/linear_model/linear/linear_model/Feature_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:

Tlinear/linear_model/linear/linear_model/linear/linear_model/Feature_17/strided_sliceStridedSliceLlinear/linear_model/linear/linear_model/linear/linear_model/Feature_17/ShapeZlinear/linear_model/linear/linear_model/linear/linear_model/Feature_17/strided_slice/stack\linear/linear_model/linear/linear_model/linear/linear_model/Feature_17/strided_slice/stack_1\linear/linear_model/linear/linear_model/linear/linear_model/Feature_17/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask

Vlinear/linear_model/linear/linear_model/linear/linear_model/Feature_17/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
¸
Tlinear/linear_model/linear/linear_model/linear/linear_model/Feature_17/Reshape/shapePackTlinear/linear_model/linear/linear_model/linear/linear_model/Feature_17/strided_sliceVlinear/linear_model/linear/linear_model/linear/linear_model/Feature_17/Reshape/shape/1*
N*
T0*
_output_shapes
:

Nlinear/linear_model/linear/linear_model/linear/linear_model/Feature_17/ReshapeReshapeParseExample/ParseExampleV2:8Tlinear/linear_model/linear/linear_model/linear/linear_model/Feature_17/Reshape/shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
É
blinear/linear_model/linear/linear_model/linear/linear_model/Feature_17/weighted_sum/ReadVariableOpReadVariableOp&linear/linear_model/Feature_17/weights*
_output_shapes

:*
dtype0
Ă
Slinear/linear_model/linear/linear_model/linear/linear_model/Feature_17/weighted_sumMatMulNlinear/linear_model/linear/linear_model/linear/linear_model/Feature_17/Reshapeblinear/linear_model/linear/linear_model/linear/linear_model/Feature_17/weighted_sum/ReadVariableOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Llinear/linear_model/linear/linear_model/linear/linear_model/Feature_18/ShapeShapeParseExample/ParseExampleV2:9*
T0*
_output_shapes
:
¤
Zlinear/linear_model/linear/linear_model/linear/linear_model/Feature_18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Ś
\linear/linear_model/linear/linear_model/linear/linear_model/Feature_18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Ś
\linear/linear_model/linear/linear_model/linear/linear_model/Feature_18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:

Tlinear/linear_model/linear/linear_model/linear/linear_model/Feature_18/strided_sliceStridedSliceLlinear/linear_model/linear/linear_model/linear/linear_model/Feature_18/ShapeZlinear/linear_model/linear/linear_model/linear/linear_model/Feature_18/strided_slice/stack\linear/linear_model/linear/linear_model/linear/linear_model/Feature_18/strided_slice/stack_1\linear/linear_model/linear/linear_model/linear/linear_model/Feature_18/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask

Vlinear/linear_model/linear/linear_model/linear/linear_model/Feature_18/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
¸
Tlinear/linear_model/linear/linear_model/linear/linear_model/Feature_18/Reshape/shapePackTlinear/linear_model/linear/linear_model/linear/linear_model/Feature_18/strided_sliceVlinear/linear_model/linear/linear_model/linear/linear_model/Feature_18/Reshape/shape/1*
N*
T0*
_output_shapes
:

Nlinear/linear_model/linear/linear_model/linear/linear_model/Feature_18/ReshapeReshapeParseExample/ParseExampleV2:9Tlinear/linear_model/linear/linear_model/linear/linear_model/Feature_18/Reshape/shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
É
blinear/linear_model/linear/linear_model/linear/linear_model/Feature_18/weighted_sum/ReadVariableOpReadVariableOp&linear/linear_model/Feature_18/weights*
_output_shapes

:*
dtype0
Ă
Slinear/linear_model/linear/linear_model/linear/linear_model/Feature_18/weighted_sumMatMulNlinear/linear_model/linear/linear_model/linear/linear_model/Feature_18/Reshapeblinear/linear_model/linear/linear_model/linear/linear_model/Feature_18/weighted_sum/ReadVariableOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Llinear/linear_model/linear/linear_model/linear/linear_model/Feature_19/ShapeShapeParseExample/ParseExampleV2:10*
T0*
_output_shapes
:
¤
Zlinear/linear_model/linear/linear_model/linear/linear_model/Feature_19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Ś
\linear/linear_model/linear/linear_model/linear/linear_model/Feature_19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Ś
\linear/linear_model/linear/linear_model/linear/linear_model/Feature_19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:

Tlinear/linear_model/linear/linear_model/linear/linear_model/Feature_19/strided_sliceStridedSliceLlinear/linear_model/linear/linear_model/linear/linear_model/Feature_19/ShapeZlinear/linear_model/linear/linear_model/linear/linear_model/Feature_19/strided_slice/stack\linear/linear_model/linear/linear_model/linear/linear_model/Feature_19/strided_slice/stack_1\linear/linear_model/linear/linear_model/linear/linear_model/Feature_19/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask

Vlinear/linear_model/linear/linear_model/linear/linear_model/Feature_19/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
¸
Tlinear/linear_model/linear/linear_model/linear/linear_model/Feature_19/Reshape/shapePackTlinear/linear_model/linear/linear_model/linear/linear_model/Feature_19/strided_sliceVlinear/linear_model/linear/linear_model/linear/linear_model/Feature_19/Reshape/shape/1*
N*
T0*
_output_shapes
:

Nlinear/linear_model/linear/linear_model/linear/linear_model/Feature_19/ReshapeReshapeParseExample/ParseExampleV2:10Tlinear/linear_model/linear/linear_model/linear/linear_model/Feature_19/Reshape/shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
É
blinear/linear_model/linear/linear_model/linear/linear_model/Feature_19/weighted_sum/ReadVariableOpReadVariableOp&linear/linear_model/Feature_19/weights*
_output_shapes

:*
dtype0
Ă
Slinear/linear_model/linear/linear_model/linear/linear_model/Feature_19/weighted_sumMatMulNlinear/linear_model/linear/linear_model/linear/linear_model/Feature_19/Reshapeblinear/linear_model/linear/linear_model/linear/linear_model/Feature_19/weighted_sum/ReadVariableOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Klinear/linear_model/linear/linear_model/linear/linear_model/Feature_2/ShapeShapeParseExample/ParseExampleV2:11*
T0*
_output_shapes
:
Ł
Ylinear/linear_model/linear/linear_model/linear/linear_model/Feature_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Ľ
[linear/linear_model/linear/linear_model/linear/linear_model/Feature_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Ľ
[linear/linear_model/linear/linear_model/linear/linear_model/Feature_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:

Slinear/linear_model/linear/linear_model/linear/linear_model/Feature_2/strided_sliceStridedSliceKlinear/linear_model/linear/linear_model/linear/linear_model/Feature_2/ShapeYlinear/linear_model/linear/linear_model/linear/linear_model/Feature_2/strided_slice/stack[linear/linear_model/linear/linear_model/linear/linear_model/Feature_2/strided_slice/stack_1[linear/linear_model/linear/linear_model/linear/linear_model/Feature_2/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask

Ulinear/linear_model/linear/linear_model/linear/linear_model/Feature_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
ľ
Slinear/linear_model/linear/linear_model/linear/linear_model/Feature_2/Reshape/shapePackSlinear/linear_model/linear/linear_model/linear/linear_model/Feature_2/strided_sliceUlinear/linear_model/linear/linear_model/linear/linear_model/Feature_2/Reshape/shape/1*
N*
T0*
_output_shapes
:
˙
Mlinear/linear_model/linear/linear_model/linear/linear_model/Feature_2/ReshapeReshapeParseExample/ParseExampleV2:11Slinear/linear_model/linear/linear_model/linear/linear_model/Feature_2/Reshape/shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ç
alinear/linear_model/linear/linear_model/linear/linear_model/Feature_2/weighted_sum/ReadVariableOpReadVariableOp%linear/linear_model/Feature_2/weights*
_output_shapes

:*
dtype0
Ŕ
Rlinear/linear_model/linear/linear_model/linear/linear_model/Feature_2/weighted_sumMatMulMlinear/linear_model/linear/linear_model/linear/linear_model/Feature_2/Reshapealinear/linear_model/linear/linear_model/linear/linear_model/Feature_2/weighted_sum/ReadVariableOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Llinear/linear_model/linear/linear_model/linear/linear_model/Feature_20/ShapeShapeParseExample/ParseExampleV2:12*
T0*
_output_shapes
:
¤
Zlinear/linear_model/linear/linear_model/linear/linear_model/Feature_20/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Ś
\linear/linear_model/linear/linear_model/linear/linear_model/Feature_20/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Ś
\linear/linear_model/linear/linear_model/linear/linear_model/Feature_20/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:

Tlinear/linear_model/linear/linear_model/linear/linear_model/Feature_20/strided_sliceStridedSliceLlinear/linear_model/linear/linear_model/linear/linear_model/Feature_20/ShapeZlinear/linear_model/linear/linear_model/linear/linear_model/Feature_20/strided_slice/stack\linear/linear_model/linear/linear_model/linear/linear_model/Feature_20/strided_slice/stack_1\linear/linear_model/linear/linear_model/linear/linear_model/Feature_20/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask

Vlinear/linear_model/linear/linear_model/linear/linear_model/Feature_20/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
¸
Tlinear/linear_model/linear/linear_model/linear/linear_model/Feature_20/Reshape/shapePackTlinear/linear_model/linear/linear_model/linear/linear_model/Feature_20/strided_sliceVlinear/linear_model/linear/linear_model/linear/linear_model/Feature_20/Reshape/shape/1*
N*
T0*
_output_shapes
:

Nlinear/linear_model/linear/linear_model/linear/linear_model/Feature_20/ReshapeReshapeParseExample/ParseExampleV2:12Tlinear/linear_model/linear/linear_model/linear/linear_model/Feature_20/Reshape/shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
É
blinear/linear_model/linear/linear_model/linear/linear_model/Feature_20/weighted_sum/ReadVariableOpReadVariableOp&linear/linear_model/Feature_20/weights*
_output_shapes

:*
dtype0
Ă
Slinear/linear_model/linear/linear_model/linear/linear_model/Feature_20/weighted_sumMatMulNlinear/linear_model/linear/linear_model/linear/linear_model/Feature_20/Reshapeblinear/linear_model/linear/linear_model/linear/linear_model/Feature_20/weighted_sum/ReadVariableOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Llinear/linear_model/linear/linear_model/linear/linear_model/Feature_21/ShapeShapeParseExample/ParseExampleV2:13*
T0*
_output_shapes
:
¤
Zlinear/linear_model/linear/linear_model/linear/linear_model/Feature_21/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Ś
\linear/linear_model/linear/linear_model/linear/linear_model/Feature_21/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Ś
\linear/linear_model/linear/linear_model/linear/linear_model/Feature_21/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:

Tlinear/linear_model/linear/linear_model/linear/linear_model/Feature_21/strided_sliceStridedSliceLlinear/linear_model/linear/linear_model/linear/linear_model/Feature_21/ShapeZlinear/linear_model/linear/linear_model/linear/linear_model/Feature_21/strided_slice/stack\linear/linear_model/linear/linear_model/linear/linear_model/Feature_21/strided_slice/stack_1\linear/linear_model/linear/linear_model/linear/linear_model/Feature_21/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask

Vlinear/linear_model/linear/linear_model/linear/linear_model/Feature_21/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
¸
Tlinear/linear_model/linear/linear_model/linear/linear_model/Feature_21/Reshape/shapePackTlinear/linear_model/linear/linear_model/linear/linear_model/Feature_21/strided_sliceVlinear/linear_model/linear/linear_model/linear/linear_model/Feature_21/Reshape/shape/1*
N*
T0*
_output_shapes
:

Nlinear/linear_model/linear/linear_model/linear/linear_model/Feature_21/ReshapeReshapeParseExample/ParseExampleV2:13Tlinear/linear_model/linear/linear_model/linear/linear_model/Feature_21/Reshape/shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
É
blinear/linear_model/linear/linear_model/linear/linear_model/Feature_21/weighted_sum/ReadVariableOpReadVariableOp&linear/linear_model/Feature_21/weights*
_output_shapes

:*
dtype0
Ă
Slinear/linear_model/linear/linear_model/linear/linear_model/Feature_21/weighted_sumMatMulNlinear/linear_model/linear/linear_model/linear/linear_model/Feature_21/Reshapeblinear/linear_model/linear/linear_model/linear/linear_model/Feature_21/weighted_sum/ReadVariableOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Llinear/linear_model/linear/linear_model/linear/linear_model/Feature_22/ShapeShapeParseExample/ParseExampleV2:14*
T0*
_output_shapes
:
¤
Zlinear/linear_model/linear/linear_model/linear/linear_model/Feature_22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Ś
\linear/linear_model/linear/linear_model/linear/linear_model/Feature_22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Ś
\linear/linear_model/linear/linear_model/linear/linear_model/Feature_22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:

Tlinear/linear_model/linear/linear_model/linear/linear_model/Feature_22/strided_sliceStridedSliceLlinear/linear_model/linear/linear_model/linear/linear_model/Feature_22/ShapeZlinear/linear_model/linear/linear_model/linear/linear_model/Feature_22/strided_slice/stack\linear/linear_model/linear/linear_model/linear/linear_model/Feature_22/strided_slice/stack_1\linear/linear_model/linear/linear_model/linear/linear_model/Feature_22/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask

Vlinear/linear_model/linear/linear_model/linear/linear_model/Feature_22/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
¸
Tlinear/linear_model/linear/linear_model/linear/linear_model/Feature_22/Reshape/shapePackTlinear/linear_model/linear/linear_model/linear/linear_model/Feature_22/strided_sliceVlinear/linear_model/linear/linear_model/linear/linear_model/Feature_22/Reshape/shape/1*
N*
T0*
_output_shapes
:

Nlinear/linear_model/linear/linear_model/linear/linear_model/Feature_22/ReshapeReshapeParseExample/ParseExampleV2:14Tlinear/linear_model/linear/linear_model/linear/linear_model/Feature_22/Reshape/shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
É
blinear/linear_model/linear/linear_model/linear/linear_model/Feature_22/weighted_sum/ReadVariableOpReadVariableOp&linear/linear_model/Feature_22/weights*
_output_shapes

:*
dtype0
Ă
Slinear/linear_model/linear/linear_model/linear/linear_model/Feature_22/weighted_sumMatMulNlinear/linear_model/linear/linear_model/linear/linear_model/Feature_22/Reshapeblinear/linear_model/linear/linear_model/linear/linear_model/Feature_22/weighted_sum/ReadVariableOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Llinear/linear_model/linear/linear_model/linear/linear_model/Feature_23/ShapeShapeParseExample/ParseExampleV2:15*
T0*
_output_shapes
:
¤
Zlinear/linear_model/linear/linear_model/linear/linear_model/Feature_23/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Ś
\linear/linear_model/linear/linear_model/linear/linear_model/Feature_23/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Ś
\linear/linear_model/linear/linear_model/linear/linear_model/Feature_23/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:

Tlinear/linear_model/linear/linear_model/linear/linear_model/Feature_23/strided_sliceStridedSliceLlinear/linear_model/linear/linear_model/linear/linear_model/Feature_23/ShapeZlinear/linear_model/linear/linear_model/linear/linear_model/Feature_23/strided_slice/stack\linear/linear_model/linear/linear_model/linear/linear_model/Feature_23/strided_slice/stack_1\linear/linear_model/linear/linear_model/linear/linear_model/Feature_23/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask

Vlinear/linear_model/linear/linear_model/linear/linear_model/Feature_23/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
¸
Tlinear/linear_model/linear/linear_model/linear/linear_model/Feature_23/Reshape/shapePackTlinear/linear_model/linear/linear_model/linear/linear_model/Feature_23/strided_sliceVlinear/linear_model/linear/linear_model/linear/linear_model/Feature_23/Reshape/shape/1*
N*
T0*
_output_shapes
:

Nlinear/linear_model/linear/linear_model/linear/linear_model/Feature_23/ReshapeReshapeParseExample/ParseExampleV2:15Tlinear/linear_model/linear/linear_model/linear/linear_model/Feature_23/Reshape/shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
É
blinear/linear_model/linear/linear_model/linear/linear_model/Feature_23/weighted_sum/ReadVariableOpReadVariableOp&linear/linear_model/Feature_23/weights*
_output_shapes

:*
dtype0
Ă
Slinear/linear_model/linear/linear_model/linear/linear_model/Feature_23/weighted_sumMatMulNlinear/linear_model/linear/linear_model/linear/linear_model/Feature_23/Reshapeblinear/linear_model/linear/linear_model/linear/linear_model/Feature_23/weighted_sum/ReadVariableOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Llinear/linear_model/linear/linear_model/linear/linear_model/Feature_24/ShapeShapeParseExample/ParseExampleV2:16*
T0*
_output_shapes
:
¤
Zlinear/linear_model/linear/linear_model/linear/linear_model/Feature_24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Ś
\linear/linear_model/linear/linear_model/linear/linear_model/Feature_24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Ś
\linear/linear_model/linear/linear_model/linear/linear_model/Feature_24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:

Tlinear/linear_model/linear/linear_model/linear/linear_model/Feature_24/strided_sliceStridedSliceLlinear/linear_model/linear/linear_model/linear/linear_model/Feature_24/ShapeZlinear/linear_model/linear/linear_model/linear/linear_model/Feature_24/strided_slice/stack\linear/linear_model/linear/linear_model/linear/linear_model/Feature_24/strided_slice/stack_1\linear/linear_model/linear/linear_model/linear/linear_model/Feature_24/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask

Vlinear/linear_model/linear/linear_model/linear/linear_model/Feature_24/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
¸
Tlinear/linear_model/linear/linear_model/linear/linear_model/Feature_24/Reshape/shapePackTlinear/linear_model/linear/linear_model/linear/linear_model/Feature_24/strided_sliceVlinear/linear_model/linear/linear_model/linear/linear_model/Feature_24/Reshape/shape/1*
N*
T0*
_output_shapes
:

Nlinear/linear_model/linear/linear_model/linear/linear_model/Feature_24/ReshapeReshapeParseExample/ParseExampleV2:16Tlinear/linear_model/linear/linear_model/linear/linear_model/Feature_24/Reshape/shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
É
blinear/linear_model/linear/linear_model/linear/linear_model/Feature_24/weighted_sum/ReadVariableOpReadVariableOp&linear/linear_model/Feature_24/weights*
_output_shapes

:*
dtype0
Ă
Slinear/linear_model/linear/linear_model/linear/linear_model/Feature_24/weighted_sumMatMulNlinear/linear_model/linear/linear_model/linear/linear_model/Feature_24/Reshapeblinear/linear_model/linear/linear_model/linear/linear_model/Feature_24/weighted_sum/ReadVariableOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Klinear/linear_model/linear/linear_model/linear/linear_model/Feature_3/ShapeShapeParseExample/ParseExampleV2:17*
T0*
_output_shapes
:
Ł
Ylinear/linear_model/linear/linear_model/linear/linear_model/Feature_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Ľ
[linear/linear_model/linear/linear_model/linear/linear_model/Feature_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Ľ
[linear/linear_model/linear/linear_model/linear/linear_model/Feature_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:

Slinear/linear_model/linear/linear_model/linear/linear_model/Feature_3/strided_sliceStridedSliceKlinear/linear_model/linear/linear_model/linear/linear_model/Feature_3/ShapeYlinear/linear_model/linear/linear_model/linear/linear_model/Feature_3/strided_slice/stack[linear/linear_model/linear/linear_model/linear/linear_model/Feature_3/strided_slice/stack_1[linear/linear_model/linear/linear_model/linear/linear_model/Feature_3/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask

Ulinear/linear_model/linear/linear_model/linear/linear_model/Feature_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
ľ
Slinear/linear_model/linear/linear_model/linear/linear_model/Feature_3/Reshape/shapePackSlinear/linear_model/linear/linear_model/linear/linear_model/Feature_3/strided_sliceUlinear/linear_model/linear/linear_model/linear/linear_model/Feature_3/Reshape/shape/1*
N*
T0*
_output_shapes
:
˙
Mlinear/linear_model/linear/linear_model/linear/linear_model/Feature_3/ReshapeReshapeParseExample/ParseExampleV2:17Slinear/linear_model/linear/linear_model/linear/linear_model/Feature_3/Reshape/shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ç
alinear/linear_model/linear/linear_model/linear/linear_model/Feature_3/weighted_sum/ReadVariableOpReadVariableOp%linear/linear_model/Feature_3/weights*
_output_shapes

:*
dtype0
Ŕ
Rlinear/linear_model/linear/linear_model/linear/linear_model/Feature_3/weighted_sumMatMulMlinear/linear_model/linear/linear_model/linear/linear_model/Feature_3/Reshapealinear/linear_model/linear/linear_model/linear/linear_model/Feature_3/weighted_sum/ReadVariableOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Klinear/linear_model/linear/linear_model/linear/linear_model/Feature_4/ShapeShapeParseExample/ParseExampleV2:18*
T0*
_output_shapes
:
Ł
Ylinear/linear_model/linear/linear_model/linear/linear_model/Feature_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Ľ
[linear/linear_model/linear/linear_model/linear/linear_model/Feature_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Ľ
[linear/linear_model/linear/linear_model/linear/linear_model/Feature_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:

Slinear/linear_model/linear/linear_model/linear/linear_model/Feature_4/strided_sliceStridedSliceKlinear/linear_model/linear/linear_model/linear/linear_model/Feature_4/ShapeYlinear/linear_model/linear/linear_model/linear/linear_model/Feature_4/strided_slice/stack[linear/linear_model/linear/linear_model/linear/linear_model/Feature_4/strided_slice/stack_1[linear/linear_model/linear/linear_model/linear/linear_model/Feature_4/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask

Ulinear/linear_model/linear/linear_model/linear/linear_model/Feature_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
ľ
Slinear/linear_model/linear/linear_model/linear/linear_model/Feature_4/Reshape/shapePackSlinear/linear_model/linear/linear_model/linear/linear_model/Feature_4/strided_sliceUlinear/linear_model/linear/linear_model/linear/linear_model/Feature_4/Reshape/shape/1*
N*
T0*
_output_shapes
:
˙
Mlinear/linear_model/linear/linear_model/linear/linear_model/Feature_4/ReshapeReshapeParseExample/ParseExampleV2:18Slinear/linear_model/linear/linear_model/linear/linear_model/Feature_4/Reshape/shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ç
alinear/linear_model/linear/linear_model/linear/linear_model/Feature_4/weighted_sum/ReadVariableOpReadVariableOp%linear/linear_model/Feature_4/weights*
_output_shapes

:*
dtype0
Ŕ
Rlinear/linear_model/linear/linear_model/linear/linear_model/Feature_4/weighted_sumMatMulMlinear/linear_model/linear/linear_model/linear/linear_model/Feature_4/Reshapealinear/linear_model/linear/linear_model/linear/linear_model/Feature_4/weighted_sum/ReadVariableOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Klinear/linear_model/linear/linear_model/linear/linear_model/Feature_5/ShapeShapeParseExample/ParseExampleV2:19*
T0*
_output_shapes
:
Ł
Ylinear/linear_model/linear/linear_model/linear/linear_model/Feature_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Ľ
[linear/linear_model/linear/linear_model/linear/linear_model/Feature_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Ľ
[linear/linear_model/linear/linear_model/linear/linear_model/Feature_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:

Slinear/linear_model/linear/linear_model/linear/linear_model/Feature_5/strided_sliceStridedSliceKlinear/linear_model/linear/linear_model/linear/linear_model/Feature_5/ShapeYlinear/linear_model/linear/linear_model/linear/linear_model/Feature_5/strided_slice/stack[linear/linear_model/linear/linear_model/linear/linear_model/Feature_5/strided_slice/stack_1[linear/linear_model/linear/linear_model/linear/linear_model/Feature_5/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask

Ulinear/linear_model/linear/linear_model/linear/linear_model/Feature_5/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
ľ
Slinear/linear_model/linear/linear_model/linear/linear_model/Feature_5/Reshape/shapePackSlinear/linear_model/linear/linear_model/linear/linear_model/Feature_5/strided_sliceUlinear/linear_model/linear/linear_model/linear/linear_model/Feature_5/Reshape/shape/1*
N*
T0*
_output_shapes
:
˙
Mlinear/linear_model/linear/linear_model/linear/linear_model/Feature_5/ReshapeReshapeParseExample/ParseExampleV2:19Slinear/linear_model/linear/linear_model/linear/linear_model/Feature_5/Reshape/shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ç
alinear/linear_model/linear/linear_model/linear/linear_model/Feature_5/weighted_sum/ReadVariableOpReadVariableOp%linear/linear_model/Feature_5/weights*
_output_shapes

:*
dtype0
Ŕ
Rlinear/linear_model/linear/linear_model/linear/linear_model/Feature_5/weighted_sumMatMulMlinear/linear_model/linear/linear_model/linear/linear_model/Feature_5/Reshapealinear/linear_model/linear/linear_model/linear/linear_model/Feature_5/weighted_sum/ReadVariableOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Klinear/linear_model/linear/linear_model/linear/linear_model/Feature_6/ShapeShapeParseExample/ParseExampleV2:20*
T0*
_output_shapes
:
Ł
Ylinear/linear_model/linear/linear_model/linear/linear_model/Feature_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Ľ
[linear/linear_model/linear/linear_model/linear/linear_model/Feature_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Ľ
[linear/linear_model/linear/linear_model/linear/linear_model/Feature_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:

Slinear/linear_model/linear/linear_model/linear/linear_model/Feature_6/strided_sliceStridedSliceKlinear/linear_model/linear/linear_model/linear/linear_model/Feature_6/ShapeYlinear/linear_model/linear/linear_model/linear/linear_model/Feature_6/strided_slice/stack[linear/linear_model/linear/linear_model/linear/linear_model/Feature_6/strided_slice/stack_1[linear/linear_model/linear/linear_model/linear/linear_model/Feature_6/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask

Ulinear/linear_model/linear/linear_model/linear/linear_model/Feature_6/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
ľ
Slinear/linear_model/linear/linear_model/linear/linear_model/Feature_6/Reshape/shapePackSlinear/linear_model/linear/linear_model/linear/linear_model/Feature_6/strided_sliceUlinear/linear_model/linear/linear_model/linear/linear_model/Feature_6/Reshape/shape/1*
N*
T0*
_output_shapes
:
˙
Mlinear/linear_model/linear/linear_model/linear/linear_model/Feature_6/ReshapeReshapeParseExample/ParseExampleV2:20Slinear/linear_model/linear/linear_model/linear/linear_model/Feature_6/Reshape/shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ç
alinear/linear_model/linear/linear_model/linear/linear_model/Feature_6/weighted_sum/ReadVariableOpReadVariableOp%linear/linear_model/Feature_6/weights*
_output_shapes

:*
dtype0
Ŕ
Rlinear/linear_model/linear/linear_model/linear/linear_model/Feature_6/weighted_sumMatMulMlinear/linear_model/linear/linear_model/linear/linear_model/Feature_6/Reshapealinear/linear_model/linear/linear_model/linear/linear_model/Feature_6/weighted_sum/ReadVariableOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Klinear/linear_model/linear/linear_model/linear/linear_model/Feature_7/ShapeShapeParseExample/ParseExampleV2:21*
T0*
_output_shapes
:
Ł
Ylinear/linear_model/linear/linear_model/linear/linear_model/Feature_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Ľ
[linear/linear_model/linear/linear_model/linear/linear_model/Feature_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Ľ
[linear/linear_model/linear/linear_model/linear/linear_model/Feature_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:

Slinear/linear_model/linear/linear_model/linear/linear_model/Feature_7/strided_sliceStridedSliceKlinear/linear_model/linear/linear_model/linear/linear_model/Feature_7/ShapeYlinear/linear_model/linear/linear_model/linear/linear_model/Feature_7/strided_slice/stack[linear/linear_model/linear/linear_model/linear/linear_model/Feature_7/strided_slice/stack_1[linear/linear_model/linear/linear_model/linear/linear_model/Feature_7/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask

Ulinear/linear_model/linear/linear_model/linear/linear_model/Feature_7/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
ľ
Slinear/linear_model/linear/linear_model/linear/linear_model/Feature_7/Reshape/shapePackSlinear/linear_model/linear/linear_model/linear/linear_model/Feature_7/strided_sliceUlinear/linear_model/linear/linear_model/linear/linear_model/Feature_7/Reshape/shape/1*
N*
T0*
_output_shapes
:
˙
Mlinear/linear_model/linear/linear_model/linear/linear_model/Feature_7/ReshapeReshapeParseExample/ParseExampleV2:21Slinear/linear_model/linear/linear_model/linear/linear_model/Feature_7/Reshape/shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ç
alinear/linear_model/linear/linear_model/linear/linear_model/Feature_7/weighted_sum/ReadVariableOpReadVariableOp%linear/linear_model/Feature_7/weights*
_output_shapes

:*
dtype0
Ŕ
Rlinear/linear_model/linear/linear_model/linear/linear_model/Feature_7/weighted_sumMatMulMlinear/linear_model/linear/linear_model/linear/linear_model/Feature_7/Reshapealinear/linear_model/linear/linear_model/linear/linear_model/Feature_7/weighted_sum/ReadVariableOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Klinear/linear_model/linear/linear_model/linear/linear_model/Feature_8/ShapeShapeParseExample/ParseExampleV2:22*
T0*
_output_shapes
:
Ł
Ylinear/linear_model/linear/linear_model/linear/linear_model/Feature_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Ľ
[linear/linear_model/linear/linear_model/linear/linear_model/Feature_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Ľ
[linear/linear_model/linear/linear_model/linear/linear_model/Feature_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:

Slinear/linear_model/linear/linear_model/linear/linear_model/Feature_8/strided_sliceStridedSliceKlinear/linear_model/linear/linear_model/linear/linear_model/Feature_8/ShapeYlinear/linear_model/linear/linear_model/linear/linear_model/Feature_8/strided_slice/stack[linear/linear_model/linear/linear_model/linear/linear_model/Feature_8/strided_slice/stack_1[linear/linear_model/linear/linear_model/linear/linear_model/Feature_8/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask

Ulinear/linear_model/linear/linear_model/linear/linear_model/Feature_8/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
ľ
Slinear/linear_model/linear/linear_model/linear/linear_model/Feature_8/Reshape/shapePackSlinear/linear_model/linear/linear_model/linear/linear_model/Feature_8/strided_sliceUlinear/linear_model/linear/linear_model/linear/linear_model/Feature_8/Reshape/shape/1*
N*
T0*
_output_shapes
:
˙
Mlinear/linear_model/linear/linear_model/linear/linear_model/Feature_8/ReshapeReshapeParseExample/ParseExampleV2:22Slinear/linear_model/linear/linear_model/linear/linear_model/Feature_8/Reshape/shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ç
alinear/linear_model/linear/linear_model/linear/linear_model/Feature_8/weighted_sum/ReadVariableOpReadVariableOp%linear/linear_model/Feature_8/weights*
_output_shapes

:*
dtype0
Ŕ
Rlinear/linear_model/linear/linear_model/linear/linear_model/Feature_8/weighted_sumMatMulMlinear/linear_model/linear/linear_model/linear/linear_model/Feature_8/Reshapealinear/linear_model/linear/linear_model/linear/linear_model/Feature_8/weighted_sum/ReadVariableOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Klinear/linear_model/linear/linear_model/linear/linear_model/Feature_9/ShapeShapeParseExample/ParseExampleV2:23*
T0*
_output_shapes
:
Ł
Ylinear/linear_model/linear/linear_model/linear/linear_model/Feature_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Ľ
[linear/linear_model/linear/linear_model/linear/linear_model/Feature_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Ľ
[linear/linear_model/linear/linear_model/linear/linear_model/Feature_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:

Slinear/linear_model/linear/linear_model/linear/linear_model/Feature_9/strided_sliceStridedSliceKlinear/linear_model/linear/linear_model/linear/linear_model/Feature_9/ShapeYlinear/linear_model/linear/linear_model/linear/linear_model/Feature_9/strided_slice/stack[linear/linear_model/linear/linear_model/linear/linear_model/Feature_9/strided_slice/stack_1[linear/linear_model/linear/linear_model/linear/linear_model/Feature_9/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask

Ulinear/linear_model/linear/linear_model/linear/linear_model/Feature_9/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
ľ
Slinear/linear_model/linear/linear_model/linear/linear_model/Feature_9/Reshape/shapePackSlinear/linear_model/linear/linear_model/linear/linear_model/Feature_9/strided_sliceUlinear/linear_model/linear/linear_model/linear/linear_model/Feature_9/Reshape/shape/1*
N*
T0*
_output_shapes
:
˙
Mlinear/linear_model/linear/linear_model/linear/linear_model/Feature_9/ReshapeReshapeParseExample/ParseExampleV2:23Slinear/linear_model/linear/linear_model/linear/linear_model/Feature_9/Reshape/shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ç
alinear/linear_model/linear/linear_model/linear/linear_model/Feature_9/weighted_sum/ReadVariableOpReadVariableOp%linear/linear_model/Feature_9/weights*
_output_shapes

:*
dtype0
Ŕ
Rlinear/linear_model/linear/linear_model/linear/linear_model/Feature_9/weighted_sumMatMulMlinear/linear_model/linear/linear_model/linear/linear_model/Feature_9/Reshapealinear/linear_model/linear/linear_model/linear/linear_model/Feature_9/weighted_sum/ReadVariableOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Plinear/linear_model/linear/linear_model/linear/linear_model/weighted_sum_no_biasAddNRlinear/linear_model/linear/linear_model/linear/linear_model/Feature_1/weighted_sumSlinear/linear_model/linear/linear_model/linear/linear_model/Feature_10/weighted_sumSlinear/linear_model/linear/linear_model/linear/linear_model/Feature_11/weighted_sumSlinear/linear_model/linear/linear_model/linear/linear_model/Feature_12/weighted_sumSlinear/linear_model/linear/linear_model/linear/linear_model/Feature_13/weighted_sumSlinear/linear_model/linear/linear_model/linear/linear_model/Feature_14/weighted_sumSlinear/linear_model/linear/linear_model/linear/linear_model/Feature_15/weighted_sumSlinear/linear_model/linear/linear_model/linear/linear_model/Feature_16/weighted_sumSlinear/linear_model/linear/linear_model/linear/linear_model/Feature_17/weighted_sumSlinear/linear_model/linear/linear_model/linear/linear_model/Feature_18/weighted_sumSlinear/linear_model/linear/linear_model/linear/linear_model/Feature_19/weighted_sumRlinear/linear_model/linear/linear_model/linear/linear_model/Feature_2/weighted_sumSlinear/linear_model/linear/linear_model/linear/linear_model/Feature_20/weighted_sumSlinear/linear_model/linear/linear_model/linear/linear_model/Feature_21/weighted_sumSlinear/linear_model/linear/linear_model/linear/linear_model/Feature_22/weighted_sumSlinear/linear_model/linear/linear_model/linear/linear_model/Feature_23/weighted_sumSlinear/linear_model/linear/linear_model/linear/linear_model/Feature_24/weighted_sumRlinear/linear_model/linear/linear_model/linear/linear_model/Feature_3/weighted_sumRlinear/linear_model/linear/linear_model/linear/linear_model/Feature_4/weighted_sumRlinear/linear_model/linear/linear_model/linear/linear_model/Feature_5/weighted_sumRlinear/linear_model/linear/linear_model/linear/linear_model/Feature_6/weighted_sumRlinear/linear_model/linear/linear_model/linear/linear_model/Feature_7/weighted_sumRlinear/linear_model/linear/linear_model/linear/linear_model/Feature_8/weighted_sumRlinear/linear_model/linear/linear_model/linear/linear_model/Feature_9/weighted_sum*
N*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
´
Wlinear/linear_model/linear/linear_model/linear/linear_model/weighted_sum/ReadVariableOpReadVariableOp linear/linear_model/bias_weights*
_output_shapes
:*
dtype0
°
Hlinear/linear_model/linear/linear_model/linear/linear_model/weighted_sumBiasAddPlinear/linear_model/linear/linear_model/linear/linear_model/weighted_sum_no_biasWlinear/linear_model/linear/linear_model/linear/linear_model/weighted_sum/ReadVariableOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
k
ReadVariableOpReadVariableOp linear/linear_model/bias_weights*
_output_shapes
:*
dtype0
]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
ś
strided_sliceStridedSliceReadVariableOpstrided_slice/stackstrided_slice/stack_1strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
N
	bias/tagsConst*
_output_shapes
: *
dtype0*
valueB
 Bbias
P
biasScalarSummary	bias/tagsstrided_slice*
T0*
_output_shapes
: 

,zero_fraction/total_size/Size/ReadVariableOpReadVariableOp%linear/linear_model/Feature_1/weights*
_output_shapes

:*
dtype0
_
zero_fraction/total_size/SizeConst*
_output_shapes
: *
dtype0	*
value	B	 R

.zero_fraction/total_size/Size_1/ReadVariableOpReadVariableOp&linear/linear_model/Feature_10/weights*
_output_shapes

:*
dtype0
a
zero_fraction/total_size/Size_1Const*
_output_shapes
: *
dtype0	*
value	B	 R

.zero_fraction/total_size/Size_2/ReadVariableOpReadVariableOp&linear/linear_model/Feature_11/weights*
_output_shapes

:*
dtype0
a
zero_fraction/total_size/Size_2Const*
_output_shapes
: *
dtype0	*
value	B	 R

.zero_fraction/total_size/Size_3/ReadVariableOpReadVariableOp&linear/linear_model/Feature_12/weights*
_output_shapes

:*
dtype0
a
zero_fraction/total_size/Size_3Const*
_output_shapes
: *
dtype0	*
value	B	 R

.zero_fraction/total_size/Size_4/ReadVariableOpReadVariableOp&linear/linear_model/Feature_13/weights*
_output_shapes

:*
dtype0
a
zero_fraction/total_size/Size_4Const*
_output_shapes
: *
dtype0	*
value	B	 R

.zero_fraction/total_size/Size_5/ReadVariableOpReadVariableOp&linear/linear_model/Feature_14/weights*
_output_shapes

:*
dtype0
a
zero_fraction/total_size/Size_5Const*
_output_shapes
: *
dtype0	*
value	B	 R

.zero_fraction/total_size/Size_6/ReadVariableOpReadVariableOp&linear/linear_model/Feature_15/weights*
_output_shapes

:*
dtype0
a
zero_fraction/total_size/Size_6Const*
_output_shapes
: *
dtype0	*
value	B	 R

.zero_fraction/total_size/Size_7/ReadVariableOpReadVariableOp&linear/linear_model/Feature_16/weights*
_output_shapes

:*
dtype0
a
zero_fraction/total_size/Size_7Const*
_output_shapes
: *
dtype0	*
value	B	 R

.zero_fraction/total_size/Size_8/ReadVariableOpReadVariableOp&linear/linear_model/Feature_17/weights*
_output_shapes

:*
dtype0
a
zero_fraction/total_size/Size_8Const*
_output_shapes
: *
dtype0	*
value	B	 R

.zero_fraction/total_size/Size_9/ReadVariableOpReadVariableOp&linear/linear_model/Feature_18/weights*
_output_shapes

:*
dtype0
a
zero_fraction/total_size/Size_9Const*
_output_shapes
: *
dtype0	*
value	B	 R

/zero_fraction/total_size/Size_10/ReadVariableOpReadVariableOp&linear/linear_model/Feature_19/weights*
_output_shapes

:*
dtype0
b
 zero_fraction/total_size/Size_10Const*
_output_shapes
: *
dtype0	*
value	B	 R

/zero_fraction/total_size/Size_11/ReadVariableOpReadVariableOp%linear/linear_model/Feature_2/weights*
_output_shapes

:*
dtype0
b
 zero_fraction/total_size/Size_11Const*
_output_shapes
: *
dtype0	*
value	B	 R

/zero_fraction/total_size/Size_12/ReadVariableOpReadVariableOp&linear/linear_model/Feature_20/weights*
_output_shapes

:*
dtype0
b
 zero_fraction/total_size/Size_12Const*
_output_shapes
: *
dtype0	*
value	B	 R

/zero_fraction/total_size/Size_13/ReadVariableOpReadVariableOp&linear/linear_model/Feature_21/weights*
_output_shapes

:*
dtype0
b
 zero_fraction/total_size/Size_13Const*
_output_shapes
: *
dtype0	*
value	B	 R

/zero_fraction/total_size/Size_14/ReadVariableOpReadVariableOp&linear/linear_model/Feature_22/weights*
_output_shapes

:*
dtype0
b
 zero_fraction/total_size/Size_14Const*
_output_shapes
: *
dtype0	*
value	B	 R

/zero_fraction/total_size/Size_15/ReadVariableOpReadVariableOp&linear/linear_model/Feature_23/weights*
_output_shapes

:*
dtype0
b
 zero_fraction/total_size/Size_15Const*
_output_shapes
: *
dtype0	*
value	B	 R

/zero_fraction/total_size/Size_16/ReadVariableOpReadVariableOp&linear/linear_model/Feature_24/weights*
_output_shapes

:*
dtype0
b
 zero_fraction/total_size/Size_16Const*
_output_shapes
: *
dtype0	*
value	B	 R

/zero_fraction/total_size/Size_17/ReadVariableOpReadVariableOp%linear/linear_model/Feature_3/weights*
_output_shapes

:*
dtype0
b
 zero_fraction/total_size/Size_17Const*
_output_shapes
: *
dtype0	*
value	B	 R

/zero_fraction/total_size/Size_18/ReadVariableOpReadVariableOp%linear/linear_model/Feature_4/weights*
_output_shapes

:*
dtype0
b
 zero_fraction/total_size/Size_18Const*
_output_shapes
: *
dtype0	*
value	B	 R

/zero_fraction/total_size/Size_19/ReadVariableOpReadVariableOp%linear/linear_model/Feature_5/weights*
_output_shapes

:*
dtype0
b
 zero_fraction/total_size/Size_19Const*
_output_shapes
: *
dtype0	*
value	B	 R

/zero_fraction/total_size/Size_20/ReadVariableOpReadVariableOp%linear/linear_model/Feature_6/weights*
_output_shapes

:*
dtype0
b
 zero_fraction/total_size/Size_20Const*
_output_shapes
: *
dtype0	*
value	B	 R

/zero_fraction/total_size/Size_21/ReadVariableOpReadVariableOp%linear/linear_model/Feature_7/weights*
_output_shapes

:*
dtype0
b
 zero_fraction/total_size/Size_21Const*
_output_shapes
: *
dtype0	*
value	B	 R

/zero_fraction/total_size/Size_22/ReadVariableOpReadVariableOp%linear/linear_model/Feature_8/weights*
_output_shapes

:*
dtype0
b
 zero_fraction/total_size/Size_22Const*
_output_shapes
: *
dtype0	*
value	B	 R

/zero_fraction/total_size/Size_23/ReadVariableOpReadVariableOp%linear/linear_model/Feature_9/weights*
_output_shapes

:*
dtype0
b
 zero_fraction/total_size/Size_23Const*
_output_shapes
: *
dtype0	*
value	B	 R
ó
zero_fraction/total_size/AddNAddNzero_fraction/total_size/Sizezero_fraction/total_size/Size_1zero_fraction/total_size/Size_2zero_fraction/total_size/Size_3zero_fraction/total_size/Size_4zero_fraction/total_size/Size_5zero_fraction/total_size/Size_6zero_fraction/total_size/Size_7zero_fraction/total_size/Size_8zero_fraction/total_size/Size_9 zero_fraction/total_size/Size_10 zero_fraction/total_size/Size_11 zero_fraction/total_size/Size_12 zero_fraction/total_size/Size_13 zero_fraction/total_size/Size_14 zero_fraction/total_size/Size_15 zero_fraction/total_size/Size_16 zero_fraction/total_size/Size_17 zero_fraction/total_size/Size_18 zero_fraction/total_size/Size_19 zero_fraction/total_size/Size_20 zero_fraction/total_size/Size_21 zero_fraction/total_size/Size_22 zero_fraction/total_size/Size_23*
N*
T0	*
_output_shapes
: 
`
zero_fraction/total_zero/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 

zero_fraction/total_zero/EqualEqualzero_fraction/total_size/Sizezero_fraction/total_zero/Const*
T0	*
_output_shapes
: 
´
#zero_fraction/total_zero/zero_countIfzero_fraction/total_zero/Equal%linear/linear_model/Feature_1/weightszero_fraction/total_size/Size*
Tcond0
*
Tin
2	*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: *#
_read_only_resource_inputs
*A
else_branch2R0
.zero_fraction_total_zero_zero_count_false_3142*
output_shapes
: *@
then_branch1R/
-zero_fraction_total_zero_zero_count_true_3141
~
,zero_fraction/total_zero/zero_count/IdentityIdentity#zero_fraction/total_zero/zero_count*
T0*
_output_shapes
: 
b
 zero_fraction/total_zero/Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 R 

 zero_fraction/total_zero/Equal_1Equalzero_fraction/total_size/Size_1 zero_fraction/total_zero/Const_1*
T0	*
_output_shapes
: 
ż
%zero_fraction/total_zero/zero_count_1If zero_fraction/total_zero/Equal_1&linear/linear_model/Feature_10/weightszero_fraction/total_size/Size_1*
Tcond0
*
Tin
2	*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: *#
_read_only_resource_inputs
*C
else_branch4R2
0zero_fraction_total_zero_zero_count_1_false_3185*
output_shapes
: *B
then_branch3R1
/zero_fraction_total_zero_zero_count_1_true_3184

.zero_fraction/total_zero/zero_count_1/IdentityIdentity%zero_fraction/total_zero/zero_count_1*
T0*
_output_shapes
: 
b
 zero_fraction/total_zero/Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R 

 zero_fraction/total_zero/Equal_2Equalzero_fraction/total_size/Size_2 zero_fraction/total_zero/Const_2*
T0	*
_output_shapes
: 
ż
%zero_fraction/total_zero/zero_count_2If zero_fraction/total_zero/Equal_2&linear/linear_model/Feature_11/weightszero_fraction/total_size/Size_2*
Tcond0
*
Tin
2	*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: *#
_read_only_resource_inputs
*C
else_branch4R2
0zero_fraction_total_zero_zero_count_2_false_3228*
output_shapes
: *B
then_branch3R1
/zero_fraction_total_zero_zero_count_2_true_3227

.zero_fraction/total_zero/zero_count_2/IdentityIdentity%zero_fraction/total_zero/zero_count_2*
T0*
_output_shapes
: 
b
 zero_fraction/total_zero/Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 R 

 zero_fraction/total_zero/Equal_3Equalzero_fraction/total_size/Size_3 zero_fraction/total_zero/Const_3*
T0	*
_output_shapes
: 
ż
%zero_fraction/total_zero/zero_count_3If zero_fraction/total_zero/Equal_3&linear/linear_model/Feature_12/weightszero_fraction/total_size/Size_3*
Tcond0
*
Tin
2	*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: *#
_read_only_resource_inputs
*C
else_branch4R2
0zero_fraction_total_zero_zero_count_3_false_3271*
output_shapes
: *B
then_branch3R1
/zero_fraction_total_zero_zero_count_3_true_3270

.zero_fraction/total_zero/zero_count_3/IdentityIdentity%zero_fraction/total_zero/zero_count_3*
T0*
_output_shapes
: 
b
 zero_fraction/total_zero/Const_4Const*
_output_shapes
: *
dtype0	*
value	B	 R 

 zero_fraction/total_zero/Equal_4Equalzero_fraction/total_size/Size_4 zero_fraction/total_zero/Const_4*
T0	*
_output_shapes
: 
ż
%zero_fraction/total_zero/zero_count_4If zero_fraction/total_zero/Equal_4&linear/linear_model/Feature_13/weightszero_fraction/total_size/Size_4*
Tcond0
*
Tin
2	*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: *#
_read_only_resource_inputs
*C
else_branch4R2
0zero_fraction_total_zero_zero_count_4_false_3314*
output_shapes
: *B
then_branch3R1
/zero_fraction_total_zero_zero_count_4_true_3313

.zero_fraction/total_zero/zero_count_4/IdentityIdentity%zero_fraction/total_zero/zero_count_4*
T0*
_output_shapes
: 
b
 zero_fraction/total_zero/Const_5Const*
_output_shapes
: *
dtype0	*
value	B	 R 

 zero_fraction/total_zero/Equal_5Equalzero_fraction/total_size/Size_5 zero_fraction/total_zero/Const_5*
T0	*
_output_shapes
: 
ż
%zero_fraction/total_zero/zero_count_5If zero_fraction/total_zero/Equal_5&linear/linear_model/Feature_14/weightszero_fraction/total_size/Size_5*
Tcond0
*
Tin
2	*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: *#
_read_only_resource_inputs
*C
else_branch4R2
0zero_fraction_total_zero_zero_count_5_false_3357*
output_shapes
: *B
then_branch3R1
/zero_fraction_total_zero_zero_count_5_true_3356

.zero_fraction/total_zero/zero_count_5/IdentityIdentity%zero_fraction/total_zero/zero_count_5*
T0*
_output_shapes
: 
b
 zero_fraction/total_zero/Const_6Const*
_output_shapes
: *
dtype0	*
value	B	 R 

 zero_fraction/total_zero/Equal_6Equalzero_fraction/total_size/Size_6 zero_fraction/total_zero/Const_6*
T0	*
_output_shapes
: 
ż
%zero_fraction/total_zero/zero_count_6If zero_fraction/total_zero/Equal_6&linear/linear_model/Feature_15/weightszero_fraction/total_size/Size_6*
Tcond0
*
Tin
2	*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: *#
_read_only_resource_inputs
*C
else_branch4R2
0zero_fraction_total_zero_zero_count_6_false_3400*
output_shapes
: *B
then_branch3R1
/zero_fraction_total_zero_zero_count_6_true_3399

.zero_fraction/total_zero/zero_count_6/IdentityIdentity%zero_fraction/total_zero/zero_count_6*
T0*
_output_shapes
: 
b
 zero_fraction/total_zero/Const_7Const*
_output_shapes
: *
dtype0	*
value	B	 R 

 zero_fraction/total_zero/Equal_7Equalzero_fraction/total_size/Size_7 zero_fraction/total_zero/Const_7*
T0	*
_output_shapes
: 
ż
%zero_fraction/total_zero/zero_count_7If zero_fraction/total_zero/Equal_7&linear/linear_model/Feature_16/weightszero_fraction/total_size/Size_7*
Tcond0
*
Tin
2	*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: *#
_read_only_resource_inputs
*C
else_branch4R2
0zero_fraction_total_zero_zero_count_7_false_3443*
output_shapes
: *B
then_branch3R1
/zero_fraction_total_zero_zero_count_7_true_3442

.zero_fraction/total_zero/zero_count_7/IdentityIdentity%zero_fraction/total_zero/zero_count_7*
T0*
_output_shapes
: 
b
 zero_fraction/total_zero/Const_8Const*
_output_shapes
: *
dtype0	*
value	B	 R 

 zero_fraction/total_zero/Equal_8Equalzero_fraction/total_size/Size_8 zero_fraction/total_zero/Const_8*
T0	*
_output_shapes
: 
ż
%zero_fraction/total_zero/zero_count_8If zero_fraction/total_zero/Equal_8&linear/linear_model/Feature_17/weightszero_fraction/total_size/Size_8*
Tcond0
*
Tin
2	*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: *#
_read_only_resource_inputs
*C
else_branch4R2
0zero_fraction_total_zero_zero_count_8_false_3486*
output_shapes
: *B
then_branch3R1
/zero_fraction_total_zero_zero_count_8_true_3485

.zero_fraction/total_zero/zero_count_8/IdentityIdentity%zero_fraction/total_zero/zero_count_8*
T0*
_output_shapes
: 
b
 zero_fraction/total_zero/Const_9Const*
_output_shapes
: *
dtype0	*
value	B	 R 

 zero_fraction/total_zero/Equal_9Equalzero_fraction/total_size/Size_9 zero_fraction/total_zero/Const_9*
T0	*
_output_shapes
: 
ż
%zero_fraction/total_zero/zero_count_9If zero_fraction/total_zero/Equal_9&linear/linear_model/Feature_18/weightszero_fraction/total_size/Size_9*
Tcond0
*
Tin
2	*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: *#
_read_only_resource_inputs
*C
else_branch4R2
0zero_fraction_total_zero_zero_count_9_false_3529*
output_shapes
: *B
then_branch3R1
/zero_fraction_total_zero_zero_count_9_true_3528

.zero_fraction/total_zero/zero_count_9/IdentityIdentity%zero_fraction/total_zero/zero_count_9*
T0*
_output_shapes
: 
c
!zero_fraction/total_zero/Const_10Const*
_output_shapes
: *
dtype0	*
value	B	 R 

!zero_fraction/total_zero/Equal_10Equal zero_fraction/total_size/Size_10!zero_fraction/total_zero/Const_10*
T0	*
_output_shapes
: 
Ä
&zero_fraction/total_zero/zero_count_10If!zero_fraction/total_zero/Equal_10&linear/linear_model/Feature_19/weights zero_fraction/total_size/Size_10*
Tcond0
*
Tin
2	*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: *#
_read_only_resource_inputs
*D
else_branch5R3
1zero_fraction_total_zero_zero_count_10_false_3572*
output_shapes
: *C
then_branch4R2
0zero_fraction_total_zero_zero_count_10_true_3571

/zero_fraction/total_zero/zero_count_10/IdentityIdentity&zero_fraction/total_zero/zero_count_10*
T0*
_output_shapes
: 
c
!zero_fraction/total_zero/Const_11Const*
_output_shapes
: *
dtype0	*
value	B	 R 

!zero_fraction/total_zero/Equal_11Equal zero_fraction/total_size/Size_11!zero_fraction/total_zero/Const_11*
T0	*
_output_shapes
: 
Ă
&zero_fraction/total_zero/zero_count_11If!zero_fraction/total_zero/Equal_11%linear/linear_model/Feature_2/weights zero_fraction/total_size/Size_11*
Tcond0
*
Tin
2	*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: *#
_read_only_resource_inputs
*D
else_branch5R3
1zero_fraction_total_zero_zero_count_11_false_3615*
output_shapes
: *C
then_branch4R2
0zero_fraction_total_zero_zero_count_11_true_3614

/zero_fraction/total_zero/zero_count_11/IdentityIdentity&zero_fraction/total_zero/zero_count_11*
T0*
_output_shapes
: 
c
!zero_fraction/total_zero/Const_12Const*
_output_shapes
: *
dtype0	*
value	B	 R 

!zero_fraction/total_zero/Equal_12Equal zero_fraction/total_size/Size_12!zero_fraction/total_zero/Const_12*
T0	*
_output_shapes
: 
Ä
&zero_fraction/total_zero/zero_count_12If!zero_fraction/total_zero/Equal_12&linear/linear_model/Feature_20/weights zero_fraction/total_size/Size_12*
Tcond0
*
Tin
2	*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: *#
_read_only_resource_inputs
*D
else_branch5R3
1zero_fraction_total_zero_zero_count_12_false_3658*
output_shapes
: *C
then_branch4R2
0zero_fraction_total_zero_zero_count_12_true_3657

/zero_fraction/total_zero/zero_count_12/IdentityIdentity&zero_fraction/total_zero/zero_count_12*
T0*
_output_shapes
: 
c
!zero_fraction/total_zero/Const_13Const*
_output_shapes
: *
dtype0	*
value	B	 R 

!zero_fraction/total_zero/Equal_13Equal zero_fraction/total_size/Size_13!zero_fraction/total_zero/Const_13*
T0	*
_output_shapes
: 
Ä
&zero_fraction/total_zero/zero_count_13If!zero_fraction/total_zero/Equal_13&linear/linear_model/Feature_21/weights zero_fraction/total_size/Size_13*
Tcond0
*
Tin
2	*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: *#
_read_only_resource_inputs
*D
else_branch5R3
1zero_fraction_total_zero_zero_count_13_false_3701*
output_shapes
: *C
then_branch4R2
0zero_fraction_total_zero_zero_count_13_true_3700

/zero_fraction/total_zero/zero_count_13/IdentityIdentity&zero_fraction/total_zero/zero_count_13*
T0*
_output_shapes
: 
c
!zero_fraction/total_zero/Const_14Const*
_output_shapes
: *
dtype0	*
value	B	 R 

!zero_fraction/total_zero/Equal_14Equal zero_fraction/total_size/Size_14!zero_fraction/total_zero/Const_14*
T0	*
_output_shapes
: 
Ä
&zero_fraction/total_zero/zero_count_14If!zero_fraction/total_zero/Equal_14&linear/linear_model/Feature_22/weights zero_fraction/total_size/Size_14*
Tcond0
*
Tin
2	*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: *#
_read_only_resource_inputs
*D
else_branch5R3
1zero_fraction_total_zero_zero_count_14_false_3744*
output_shapes
: *C
then_branch4R2
0zero_fraction_total_zero_zero_count_14_true_3743

/zero_fraction/total_zero/zero_count_14/IdentityIdentity&zero_fraction/total_zero/zero_count_14*
T0*
_output_shapes
: 
c
!zero_fraction/total_zero/Const_15Const*
_output_shapes
: *
dtype0	*
value	B	 R 

!zero_fraction/total_zero/Equal_15Equal zero_fraction/total_size/Size_15!zero_fraction/total_zero/Const_15*
T0	*
_output_shapes
: 
Ä
&zero_fraction/total_zero/zero_count_15If!zero_fraction/total_zero/Equal_15&linear/linear_model/Feature_23/weights zero_fraction/total_size/Size_15*
Tcond0
*
Tin
2	*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: *#
_read_only_resource_inputs
*D
else_branch5R3
1zero_fraction_total_zero_zero_count_15_false_3787*
output_shapes
: *C
then_branch4R2
0zero_fraction_total_zero_zero_count_15_true_3786

/zero_fraction/total_zero/zero_count_15/IdentityIdentity&zero_fraction/total_zero/zero_count_15*
T0*
_output_shapes
: 
c
!zero_fraction/total_zero/Const_16Const*
_output_shapes
: *
dtype0	*
value	B	 R 

!zero_fraction/total_zero/Equal_16Equal zero_fraction/total_size/Size_16!zero_fraction/total_zero/Const_16*
T0	*
_output_shapes
: 
Ä
&zero_fraction/total_zero/zero_count_16If!zero_fraction/total_zero/Equal_16&linear/linear_model/Feature_24/weights zero_fraction/total_size/Size_16*
Tcond0
*
Tin
2	*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: *#
_read_only_resource_inputs
*D
else_branch5R3
1zero_fraction_total_zero_zero_count_16_false_3830*
output_shapes
: *C
then_branch4R2
0zero_fraction_total_zero_zero_count_16_true_3829

/zero_fraction/total_zero/zero_count_16/IdentityIdentity&zero_fraction/total_zero/zero_count_16*
T0*
_output_shapes
: 
c
!zero_fraction/total_zero/Const_17Const*
_output_shapes
: *
dtype0	*
value	B	 R 

!zero_fraction/total_zero/Equal_17Equal zero_fraction/total_size/Size_17!zero_fraction/total_zero/Const_17*
T0	*
_output_shapes
: 
Ă
&zero_fraction/total_zero/zero_count_17If!zero_fraction/total_zero/Equal_17%linear/linear_model/Feature_3/weights zero_fraction/total_size/Size_17*
Tcond0
*
Tin
2	*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: *#
_read_only_resource_inputs
*D
else_branch5R3
1zero_fraction_total_zero_zero_count_17_false_3873*
output_shapes
: *C
then_branch4R2
0zero_fraction_total_zero_zero_count_17_true_3872

/zero_fraction/total_zero/zero_count_17/IdentityIdentity&zero_fraction/total_zero/zero_count_17*
T0*
_output_shapes
: 
c
!zero_fraction/total_zero/Const_18Const*
_output_shapes
: *
dtype0	*
value	B	 R 

!zero_fraction/total_zero/Equal_18Equal zero_fraction/total_size/Size_18!zero_fraction/total_zero/Const_18*
T0	*
_output_shapes
: 
Ă
&zero_fraction/total_zero/zero_count_18If!zero_fraction/total_zero/Equal_18%linear/linear_model/Feature_4/weights zero_fraction/total_size/Size_18*
Tcond0
*
Tin
2	*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: *#
_read_only_resource_inputs
*D
else_branch5R3
1zero_fraction_total_zero_zero_count_18_false_3916*
output_shapes
: *C
then_branch4R2
0zero_fraction_total_zero_zero_count_18_true_3915

/zero_fraction/total_zero/zero_count_18/IdentityIdentity&zero_fraction/total_zero/zero_count_18*
T0*
_output_shapes
: 
c
!zero_fraction/total_zero/Const_19Const*
_output_shapes
: *
dtype0	*
value	B	 R 

!zero_fraction/total_zero/Equal_19Equal zero_fraction/total_size/Size_19!zero_fraction/total_zero/Const_19*
T0	*
_output_shapes
: 
Ă
&zero_fraction/total_zero/zero_count_19If!zero_fraction/total_zero/Equal_19%linear/linear_model/Feature_5/weights zero_fraction/total_size/Size_19*
Tcond0
*
Tin
2	*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: *#
_read_only_resource_inputs
*D
else_branch5R3
1zero_fraction_total_zero_zero_count_19_false_3959*
output_shapes
: *C
then_branch4R2
0zero_fraction_total_zero_zero_count_19_true_3958

/zero_fraction/total_zero/zero_count_19/IdentityIdentity&zero_fraction/total_zero/zero_count_19*
T0*
_output_shapes
: 
c
!zero_fraction/total_zero/Const_20Const*
_output_shapes
: *
dtype0	*
value	B	 R 

!zero_fraction/total_zero/Equal_20Equal zero_fraction/total_size/Size_20!zero_fraction/total_zero/Const_20*
T0	*
_output_shapes
: 
Ă
&zero_fraction/total_zero/zero_count_20If!zero_fraction/total_zero/Equal_20%linear/linear_model/Feature_6/weights zero_fraction/total_size/Size_20*
Tcond0
*
Tin
2	*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: *#
_read_only_resource_inputs
*D
else_branch5R3
1zero_fraction_total_zero_zero_count_20_false_4002*
output_shapes
: *C
then_branch4R2
0zero_fraction_total_zero_zero_count_20_true_4001

/zero_fraction/total_zero/zero_count_20/IdentityIdentity&zero_fraction/total_zero/zero_count_20*
T0*
_output_shapes
: 
c
!zero_fraction/total_zero/Const_21Const*
_output_shapes
: *
dtype0	*
value	B	 R 

!zero_fraction/total_zero/Equal_21Equal zero_fraction/total_size/Size_21!zero_fraction/total_zero/Const_21*
T0	*
_output_shapes
: 
Ă
&zero_fraction/total_zero/zero_count_21If!zero_fraction/total_zero/Equal_21%linear/linear_model/Feature_7/weights zero_fraction/total_size/Size_21*
Tcond0
*
Tin
2	*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: *#
_read_only_resource_inputs
*D
else_branch5R3
1zero_fraction_total_zero_zero_count_21_false_4045*
output_shapes
: *C
then_branch4R2
0zero_fraction_total_zero_zero_count_21_true_4044

/zero_fraction/total_zero/zero_count_21/IdentityIdentity&zero_fraction/total_zero/zero_count_21*
T0*
_output_shapes
: 
c
!zero_fraction/total_zero/Const_22Const*
_output_shapes
: *
dtype0	*
value	B	 R 

!zero_fraction/total_zero/Equal_22Equal zero_fraction/total_size/Size_22!zero_fraction/total_zero/Const_22*
T0	*
_output_shapes
: 
Ă
&zero_fraction/total_zero/zero_count_22If!zero_fraction/total_zero/Equal_22%linear/linear_model/Feature_8/weights zero_fraction/total_size/Size_22*
Tcond0
*
Tin
2	*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: *#
_read_only_resource_inputs
*D
else_branch5R3
1zero_fraction_total_zero_zero_count_22_false_4088*
output_shapes
: *C
then_branch4R2
0zero_fraction_total_zero_zero_count_22_true_4087

/zero_fraction/total_zero/zero_count_22/IdentityIdentity&zero_fraction/total_zero/zero_count_22*
T0*
_output_shapes
: 
c
!zero_fraction/total_zero/Const_23Const*
_output_shapes
: *
dtype0	*
value	B	 R 

!zero_fraction/total_zero/Equal_23Equal zero_fraction/total_size/Size_23!zero_fraction/total_zero/Const_23*
T0	*
_output_shapes
: 
Ă
&zero_fraction/total_zero/zero_count_23If!zero_fraction/total_zero/Equal_23%linear/linear_model/Feature_9/weights zero_fraction/total_size/Size_23*
Tcond0
*
Tin
2	*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: *#
_read_only_resource_inputs
*D
else_branch5R3
1zero_fraction_total_zero_zero_count_23_false_4131*
output_shapes
: *C
then_branch4R2
0zero_fraction_total_zero_zero_count_23_true_4130

/zero_fraction/total_zero/zero_count_23/IdentityIdentity&zero_fraction/total_zero/zero_count_23*
T0*
_output_shapes
: 
Ű	
zero_fraction/total_zero/AddNAddN,zero_fraction/total_zero/zero_count/Identity.zero_fraction/total_zero/zero_count_1/Identity.zero_fraction/total_zero/zero_count_2/Identity.zero_fraction/total_zero/zero_count_3/Identity.zero_fraction/total_zero/zero_count_4/Identity.zero_fraction/total_zero/zero_count_5/Identity.zero_fraction/total_zero/zero_count_6/Identity.zero_fraction/total_zero/zero_count_7/Identity.zero_fraction/total_zero/zero_count_8/Identity.zero_fraction/total_zero/zero_count_9/Identity/zero_fraction/total_zero/zero_count_10/Identity/zero_fraction/total_zero/zero_count_11/Identity/zero_fraction/total_zero/zero_count_12/Identity/zero_fraction/total_zero/zero_count_13/Identity/zero_fraction/total_zero/zero_count_14/Identity/zero_fraction/total_zero/zero_count_15/Identity/zero_fraction/total_zero/zero_count_16/Identity/zero_fraction/total_zero/zero_count_17/Identity/zero_fraction/total_zero/zero_count_18/Identity/zero_fraction/total_zero/zero_count_19/Identity/zero_fraction/total_zero/zero_count_20/Identity/zero_fraction/total_zero/zero_count_21/Identity/zero_fraction/total_zero/zero_count_22/Identity/zero_fraction/total_zero/zero_count_23/Identity*
N*
T0*
_output_shapes
: 
y
"zero_fraction/compute/float32_sizeCastzero_fraction/total_size/AddN*

DstT0*

SrcT0	*
_output_shapes
: 

zero_fraction/compute/truedivRealDivzero_fraction/total_zero/AddN"zero_fraction/compute/float32_size*
T0*
_output_shapes
: 
n
"zero_fraction/zero_fraction_or_nanIdentityzero_fraction/compute/truediv*
T0*
_output_shapes
: 
v
fraction_of_zero_weights/tagsConst*
_output_shapes
: *
dtype0*)
value B Bfraction_of_zero_weights

fraction_of_zero_weightsScalarSummaryfraction_of_zero_weights/tags"zero_fraction/zero_fraction_or_nan*
T0*
_output_shapes
: 

head/logits/ShapeShapeHlinear/linear_model/linear/linear_model/linear/linear_model/weighted_sum*
T0*
_output_shapes
:
g
%head/logits/assert_rank_at_least/rankConst*
_output_shapes
: *
dtype0*
value	B :
W
Ohead/logits/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp
H
@head/logits/assert_rank_at_least/static_checks_determined_all_okNoOp
 
head/predictions/logisticSigmoidHlinear/linear_model/linear/linear_model/linear/linear_model/weighted_sum*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
¤
head/predictions/zeros_like	ZerosLikeHlinear/linear_model/linear/linear_model/linear/linear_model/weighted_sum*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
q
&head/predictions/two_class_logits/axisConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
÷
!head/predictions/two_class_logitsConcatV2head/predictions/zeros_likeHlinear/linear_model/linear/linear_model/linear/linear_model/weighted_sum&head/predictions/two_class_logits/axis*
N*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
~
head/predictions/probabilitiesSoftmax!head/predictions/two_class_logits*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
o
$head/predictions/class_ids/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙

head/predictions/class_idsArgMax!head/predictions/two_class_logits$head/predictions/class_ids/dimension*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
j
head/predictions/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙

head/predictions/ExpandDims
ExpandDimshead/predictions/class_idshead/predictions/ExpandDims/dim*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
w
head/predictions/str_classesAsStringhead/predictions/ExpandDims*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

head/predictions/ShapeShapeHlinear/linear_model/linear/linear_model/linear/linear_model/weighted_sum*
T0*
_output_shapes
:
n
$head/predictions/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
p
&head/predictions/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
p
&head/predictions/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:

head/predictions/strided_sliceStridedSlicehead/predictions/Shape$head/predictions/strided_slice/stack&head/predictions/strided_slice/stack_1&head/predictions/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
^
head/predictions/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
^
head/predictions/range/limitConst*
_output_shapes
: *
dtype0*
value	B :
^
head/predictions/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :

head/predictions/rangeRangehead/predictions/range/starthead/predictions/range/limithead/predictions/range/delta*
_output_shapes
:
c
!head/predictions/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 

head/predictions/ExpandDims_1
ExpandDimshead/predictions/range!head/predictions/ExpandDims_1/dim*
T0*
_output_shapes

:
c
!head/predictions/Tile/multiples/1Const*
_output_shapes
: *
dtype0*
value	B :

head/predictions/Tile/multiplesPackhead/predictions/strided_slice!head/predictions/Tile/multiples/1*
N*
T0*
_output_shapes
:

head/predictions/TileTilehead/predictions/ExpandDims_1head/predictions/Tile/multiples*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

head/predictions/Shape_1ShapeHlinear/linear_model/linear/linear_model/linear/linear_model/weighted_sum*
T0*
_output_shapes
:
p
&head/predictions/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
r
(head/predictions/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
r
(head/predictions/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:

 head/predictions/strided_slice_1StridedSlicehead/predictions/Shape_1&head/predictions/strided_slice_1/stack(head/predictions/strided_slice_1/stack_1(head/predictions/strided_slice_1/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
`
head/predictions/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 
`
head/predictions/range_1/limitConst*
_output_shapes
: *
dtype0*
value	B :
`
head/predictions/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :

head/predictions/range_1Rangehead/predictions/range_1/starthead/predictions/range_1/limithead/predictions/range_1/delta*
_output_shapes
:
d
head/predictions/AsStringAsStringhead/predictions/range_1*
T0*
_output_shapes
:
c
!head/predictions/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B : 

head/predictions/ExpandDims_2
ExpandDimshead/predictions/AsString!head/predictions/ExpandDims_2/dim*
T0*
_output_shapes

:
e
#head/predictions/Tile_1/multiples/1Const*
_output_shapes
: *
dtype0*
value	B :

!head/predictions/Tile_1/multiplesPack head/predictions/strided_slice_1#head/predictions/Tile_1/multiples/1*
N*
T0*
_output_shapes
:

head/predictions/Tile_1Tilehead/predictions/ExpandDims_2!head/predictions/Tile_1/multiples*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
X

head/ShapeShapehead/predictions/probabilities*
T0*
_output_shapes
:
b
head/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
d
head/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
d
head/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Ć
head/strided_sliceStridedSlice
head/Shapehead/strided_slice/stackhead/strided_slice/stack_1head/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
R
head/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
R
head/range/limitConst*
_output_shapes
: *
dtype0*
value	B :
R
head/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
e

head/rangeRangehead/range/starthead/range/limithead/range/delta*
_output_shapes
:
J
head/AsStringAsString
head/range*
T0*
_output_shapes
:
U
head/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
j
head/ExpandDims
ExpandDimshead/AsStringhead/ExpandDims/dim*
T0*
_output_shapes

:
W
head/Tile/multiples/1Const*
_output_shapes
: *
dtype0*
value	B :
t
head/Tile/multiplesPackhead/strided_slicehead/Tile/multiples/1*
N*
T0*
_output_shapes
:
i
	head/TileTilehead/ExpandDimshead/Tile/multiples*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

initNoOp

init_all_tablesNoOp

init_1NoOp
4

group_depsNoOp^init^init_1^init_all_tables
Y
save/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
n
save/filenamePlaceholderWithDefaultsave/filename/input*
_output_shapes
: *
dtype0*
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
_output_shapes
: *
dtype0*
shape: 
{
save/StaticRegexFullMatchStaticRegexFullMatch
save/Const"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*
a
save/Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part
f
save/Const_2Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
|
save/SelectSelectsave/StaticRegexFullMatchsave/Const_1save/Const_2"/device:CPU:**
T0*
_output_shapes
: 
f
save/StringJoin
StringJoin
save/Constsave/Select"/device:CPU:**
N*
_output_shapes
: 
Q
save/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
k
save/ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 

save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
×
save/SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ű
valueńBîBglobal_stepB%linear/linear_model/Feature_1/weightsB&linear/linear_model/Feature_10/weightsB&linear/linear_model/Feature_11/weightsB&linear/linear_model/Feature_12/weightsB&linear/linear_model/Feature_13/weightsB&linear/linear_model/Feature_14/weightsB&linear/linear_model/Feature_15/weightsB&linear/linear_model/Feature_16/weightsB&linear/linear_model/Feature_17/weightsB&linear/linear_model/Feature_18/weightsB&linear/linear_model/Feature_19/weightsB%linear/linear_model/Feature_2/weightsB&linear/linear_model/Feature_20/weightsB&linear/linear_model/Feature_21/weightsB&linear/linear_model/Feature_22/weightsB&linear/linear_model/Feature_23/weightsB&linear/linear_model/Feature_24/weightsB%linear/linear_model/Feature_3/weightsB%linear/linear_model/Feature_4/weightsB%linear/linear_model/Feature_5/weightsB%linear/linear_model/Feature_6/weightsB%linear/linear_model/Feature_7/weightsB%linear/linear_model/Feature_8/weightsB%linear/linear_model/Feature_9/weightsB linear/linear_model/bias_weights
Ś
save/SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 

save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesglobal_step/Read/ReadVariableOp9linear/linear_model/Feature_1/weights/Read/ReadVariableOp:linear/linear_model/Feature_10/weights/Read/ReadVariableOp:linear/linear_model/Feature_11/weights/Read/ReadVariableOp:linear/linear_model/Feature_12/weights/Read/ReadVariableOp:linear/linear_model/Feature_13/weights/Read/ReadVariableOp:linear/linear_model/Feature_14/weights/Read/ReadVariableOp:linear/linear_model/Feature_15/weights/Read/ReadVariableOp:linear/linear_model/Feature_16/weights/Read/ReadVariableOp:linear/linear_model/Feature_17/weights/Read/ReadVariableOp:linear/linear_model/Feature_18/weights/Read/ReadVariableOp:linear/linear_model/Feature_19/weights/Read/ReadVariableOp9linear/linear_model/Feature_2/weights/Read/ReadVariableOp:linear/linear_model/Feature_20/weights/Read/ReadVariableOp:linear/linear_model/Feature_21/weights/Read/ReadVariableOp:linear/linear_model/Feature_22/weights/Read/ReadVariableOp:linear/linear_model/Feature_23/weights/Read/ReadVariableOp:linear/linear_model/Feature_24/weights/Read/ReadVariableOp9linear/linear_model/Feature_3/weights/Read/ReadVariableOp9linear/linear_model/Feature_4/weights/Read/ReadVariableOp9linear/linear_model/Feature_5/weights/Read/ReadVariableOp9linear/linear_model/Feature_6/weights/Read/ReadVariableOp9linear/linear_model/Feature_7/weights/Read/ReadVariableOp9linear/linear_model/Feature_8/weights/Read/ReadVariableOp9linear/linear_model/Feature_9/weights/Read/ReadVariableOp4linear/linear_model/bias_weights/Read/ReadVariableOp"/device:CPU:0*(
dtypes
2	
 
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
 
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
N*
T0*
_output_shapes
:
u
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0

save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
T0*
_output_shapes
: 
Ú
save/RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ű
valueńBîBglobal_stepB%linear/linear_model/Feature_1/weightsB&linear/linear_model/Feature_10/weightsB&linear/linear_model/Feature_11/weightsB&linear/linear_model/Feature_12/weightsB&linear/linear_model/Feature_13/weightsB&linear/linear_model/Feature_14/weightsB&linear/linear_model/Feature_15/weightsB&linear/linear_model/Feature_16/weightsB&linear/linear_model/Feature_17/weightsB&linear/linear_model/Feature_18/weightsB&linear/linear_model/Feature_19/weightsB%linear/linear_model/Feature_2/weightsB&linear/linear_model/Feature_20/weightsB&linear/linear_model/Feature_21/weightsB&linear/linear_model/Feature_22/weightsB&linear/linear_model/Feature_23/weightsB&linear/linear_model/Feature_24/weightsB%linear/linear_model/Feature_3/weightsB%linear/linear_model/Feature_4/weightsB%linear/linear_model/Feature_5/weightsB%linear/linear_model/Feature_6/weightsB%linear/linear_model/Feature_7/weightsB%linear/linear_model/Feature_8/weightsB%linear/linear_model/Feature_9/weightsB linear/linear_model/bias_weights
Š
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*|
_output_shapesj
h::::::::::::::::::::::::::*(
dtypes
2	
N
save/Identity_1Identitysave/RestoreV2*
T0	*
_output_shapes
:
T
save/AssignVariableOpAssignVariableOpglobal_stepsave/Identity_1*
dtype0	
P
save/Identity_2Identitysave/RestoreV2:1*
T0*
_output_shapes
:
p
save/AssignVariableOp_1AssignVariableOp%linear/linear_model/Feature_1/weightssave/Identity_2*
dtype0
P
save/Identity_3Identitysave/RestoreV2:2*
T0*
_output_shapes
:
q
save/AssignVariableOp_2AssignVariableOp&linear/linear_model/Feature_10/weightssave/Identity_3*
dtype0
P
save/Identity_4Identitysave/RestoreV2:3*
T0*
_output_shapes
:
q
save/AssignVariableOp_3AssignVariableOp&linear/linear_model/Feature_11/weightssave/Identity_4*
dtype0
P
save/Identity_5Identitysave/RestoreV2:4*
T0*
_output_shapes
:
q
save/AssignVariableOp_4AssignVariableOp&linear/linear_model/Feature_12/weightssave/Identity_5*
dtype0
P
save/Identity_6Identitysave/RestoreV2:5*
T0*
_output_shapes
:
q
save/AssignVariableOp_5AssignVariableOp&linear/linear_model/Feature_13/weightssave/Identity_6*
dtype0
P
save/Identity_7Identitysave/RestoreV2:6*
T0*
_output_shapes
:
q
save/AssignVariableOp_6AssignVariableOp&linear/linear_model/Feature_14/weightssave/Identity_7*
dtype0
P
save/Identity_8Identitysave/RestoreV2:7*
T0*
_output_shapes
:
q
save/AssignVariableOp_7AssignVariableOp&linear/linear_model/Feature_15/weightssave/Identity_8*
dtype0
P
save/Identity_9Identitysave/RestoreV2:8*
T0*
_output_shapes
:
q
save/AssignVariableOp_8AssignVariableOp&linear/linear_model/Feature_16/weightssave/Identity_9*
dtype0
Q
save/Identity_10Identitysave/RestoreV2:9*
T0*
_output_shapes
:
r
save/AssignVariableOp_9AssignVariableOp&linear/linear_model/Feature_17/weightssave/Identity_10*
dtype0
R
save/Identity_11Identitysave/RestoreV2:10*
T0*
_output_shapes
:
s
save/AssignVariableOp_10AssignVariableOp&linear/linear_model/Feature_18/weightssave/Identity_11*
dtype0
R
save/Identity_12Identitysave/RestoreV2:11*
T0*
_output_shapes
:
s
save/AssignVariableOp_11AssignVariableOp&linear/linear_model/Feature_19/weightssave/Identity_12*
dtype0
R
save/Identity_13Identitysave/RestoreV2:12*
T0*
_output_shapes
:
r
save/AssignVariableOp_12AssignVariableOp%linear/linear_model/Feature_2/weightssave/Identity_13*
dtype0
R
save/Identity_14Identitysave/RestoreV2:13*
T0*
_output_shapes
:
s
save/AssignVariableOp_13AssignVariableOp&linear/linear_model/Feature_20/weightssave/Identity_14*
dtype0
R
save/Identity_15Identitysave/RestoreV2:14*
T0*
_output_shapes
:
s
save/AssignVariableOp_14AssignVariableOp&linear/linear_model/Feature_21/weightssave/Identity_15*
dtype0
R
save/Identity_16Identitysave/RestoreV2:15*
T0*
_output_shapes
:
s
save/AssignVariableOp_15AssignVariableOp&linear/linear_model/Feature_22/weightssave/Identity_16*
dtype0
R
save/Identity_17Identitysave/RestoreV2:16*
T0*
_output_shapes
:
s
save/AssignVariableOp_16AssignVariableOp&linear/linear_model/Feature_23/weightssave/Identity_17*
dtype0
R
save/Identity_18Identitysave/RestoreV2:17*
T0*
_output_shapes
:
s
save/AssignVariableOp_17AssignVariableOp&linear/linear_model/Feature_24/weightssave/Identity_18*
dtype0
R
save/Identity_19Identitysave/RestoreV2:18*
T0*
_output_shapes
:
r
save/AssignVariableOp_18AssignVariableOp%linear/linear_model/Feature_3/weightssave/Identity_19*
dtype0
R
save/Identity_20Identitysave/RestoreV2:19*
T0*
_output_shapes
:
r
save/AssignVariableOp_19AssignVariableOp%linear/linear_model/Feature_4/weightssave/Identity_20*
dtype0
R
save/Identity_21Identitysave/RestoreV2:20*
T0*
_output_shapes
:
r
save/AssignVariableOp_20AssignVariableOp%linear/linear_model/Feature_5/weightssave/Identity_21*
dtype0
R
save/Identity_22Identitysave/RestoreV2:21*
T0*
_output_shapes
:
r
save/AssignVariableOp_21AssignVariableOp%linear/linear_model/Feature_6/weightssave/Identity_22*
dtype0
R
save/Identity_23Identitysave/RestoreV2:22*
T0*
_output_shapes
:
r
save/AssignVariableOp_22AssignVariableOp%linear/linear_model/Feature_7/weightssave/Identity_23*
dtype0
R
save/Identity_24Identitysave/RestoreV2:23*
T0*
_output_shapes
:
r
save/AssignVariableOp_23AssignVariableOp%linear/linear_model/Feature_8/weightssave/Identity_24*
dtype0
R
save/Identity_25Identitysave/RestoreV2:24*
T0*
_output_shapes
:
r
save/AssignVariableOp_24AssignVariableOp%linear/linear_model/Feature_9/weightssave/Identity_25*
dtype0
R
save/Identity_26Identitysave/RestoreV2:25*
T0*
_output_shapes
:
m
save/AssignVariableOp_25AssignVariableOp linear/linear_model/bias_weightssave/Identity_26*
dtype0
Ě
save/restore_shardNoOp^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_10^save/AssignVariableOp_11^save/AssignVariableOp_12^save/AssignVariableOp_13^save/AssignVariableOp_14^save/AssignVariableOp_15^save/AssignVariableOp_16^save/AssignVariableOp_17^save/AssignVariableOp_18^save/AssignVariableOp_19^save/AssignVariableOp_2^save/AssignVariableOp_20^save/AssignVariableOp_21^save/AssignVariableOp_22^save/AssignVariableOp_23^save/AssignVariableOp_24^save/AssignVariableOp_25^save/AssignVariableOp_3^save/AssignVariableOp_4^save/AssignVariableOp_5^save/AssignVariableOp_6^save/AssignVariableOp_7^save/AssignVariableOp_8^save/AssignVariableOp_9
-
save/restore_allNoOp^save/restore_shardîń
Đ
y
zero_fraction_cond_false_35397
3count_nonzero_notequal_zero_fraction_readvariableop
count_nonzero_nonzero_count	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0	*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0	*
_output_shapes
: 2
count_nonzero/nonzero_count"C
count_nonzero_nonzero_count$count_nonzero/nonzero_count:output:0*
_input_shapes

::$  

_output_shapes

:

ł
1zero_fraction_total_zero_zero_count_12_false_3658G
Czero_fraction_readvariableop_linear_linear_model_feature_20_weights)
%cast_zero_fraction_total_size_size_12	
mulŔ
zero_fraction/ReadVariableOpReadVariableOpCzero_fraction_readvariableop_linear_linear_model_feature_20_weights*
_output_shapes

:*
dtype02
zero_fraction/ReadVariableOpj
zero_fraction/SizeConst*
_output_shapes
: *
dtype0	*
value	B	 R2
zero_fraction/Size|
zero_fraction/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R˙˙˙˙2
zero_fraction/LessEqual/yĄ
zero_fraction/LessEqual	LessEqualzero_fraction/Size:output:0"zero_fraction/LessEqual/y:output:0*
T0	*
_output_shapes
: 2
zero_fraction/LessEqualů
zero_fraction/condStatelessIfzero_fraction/LessEqual:z:0$zero_fraction/ReadVariableOp:value:0*
Tcond0
*
Tin
2*
Tout
2	*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
else_branch!R
zero_fraction_cond_false_3668*
output_shapes
: */
then_branch R
zero_fraction_cond_true_36672
zero_fraction/cond
zero_fraction/cond/IdentityIdentityzero_fraction/cond:output:0*
T0	*
_output_shapes
: 2
zero_fraction/cond/Identityˇ
$zero_fraction/counts_to_fraction/subSubzero_fraction/Size:output:0$zero_fraction/cond/Identity:output:0*
T0	*
_output_shapes
: 2&
$zero_fraction/counts_to_fraction/sub°
%zero_fraction/counts_to_fraction/CastCast(zero_fraction/counts_to_fraction/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2'
%zero_fraction/counts_to_fraction/Cast§
'zero_fraction/counts_to_fraction/Cast_1Castzero_fraction/Size:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2)
'zero_fraction/counts_to_fraction/Cast_1Ř
(zero_fraction/counts_to_fraction/truedivRealDiv)zero_fraction/counts_to_fraction/Cast:y:0+zero_fraction/counts_to_fraction/Cast_1:y:0*
T0*
_output_shapes
: 2*
(zero_fraction/counts_to_fraction/truediv
zero_fraction/fractionIdentity,zero_fraction/counts_to_fraction/truediv:z:0*
T0*
_output_shapes
: 2
zero_fraction/fractionk
CastCast%cast_zero_fraction_total_size_size_12*

DstT0*

SrcT0	*
_output_shapes
: 2
CastG
mul_0Mulzero_fraction/fraction:output:0Cast:y:0*
T02
mul"
mul	mul_0:z:0*
_input_shapes
:: :

_output_shapes
: 

ł
1zero_fraction_total_zero_zero_count_16_false_3830G
Czero_fraction_readvariableop_linear_linear_model_feature_24_weights)
%cast_zero_fraction_total_size_size_16	
mulŔ
zero_fraction/ReadVariableOpReadVariableOpCzero_fraction_readvariableop_linear_linear_model_feature_24_weights*
_output_shapes

:*
dtype02
zero_fraction/ReadVariableOpj
zero_fraction/SizeConst*
_output_shapes
: *
dtype0	*
value	B	 R2
zero_fraction/Size|
zero_fraction/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R˙˙˙˙2
zero_fraction/LessEqual/yĄ
zero_fraction/LessEqual	LessEqualzero_fraction/Size:output:0"zero_fraction/LessEqual/y:output:0*
T0	*
_output_shapes
: 2
zero_fraction/LessEqualů
zero_fraction/condStatelessIfzero_fraction/LessEqual:z:0$zero_fraction/ReadVariableOp:value:0*
Tcond0
*
Tin
2*
Tout
2	*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
else_branch!R
zero_fraction_cond_false_3840*
output_shapes
: */
then_branch R
zero_fraction_cond_true_38392
zero_fraction/cond
zero_fraction/cond/IdentityIdentityzero_fraction/cond:output:0*
T0	*
_output_shapes
: 2
zero_fraction/cond/Identityˇ
$zero_fraction/counts_to_fraction/subSubzero_fraction/Size:output:0$zero_fraction/cond/Identity:output:0*
T0	*
_output_shapes
: 2&
$zero_fraction/counts_to_fraction/sub°
%zero_fraction/counts_to_fraction/CastCast(zero_fraction/counts_to_fraction/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2'
%zero_fraction/counts_to_fraction/Cast§
'zero_fraction/counts_to_fraction/Cast_1Castzero_fraction/Size:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2)
'zero_fraction/counts_to_fraction/Cast_1Ř
(zero_fraction/counts_to_fraction/truedivRealDiv)zero_fraction/counts_to_fraction/Cast:y:0+zero_fraction/counts_to_fraction/Cast_1:y:0*
T0*
_output_shapes
: 2*
(zero_fraction/counts_to_fraction/truediv
zero_fraction/fractionIdentity,zero_fraction/counts_to_fraction/truediv:z:0*
T0*
_output_shapes
: 2
zero_fraction/fractionk
CastCast%cast_zero_fraction_total_size_size_16*

DstT0*

SrcT0	*
_output_shapes
: 2
CastG
mul_0Mulzero_fraction/fraction:output:0Cast:y:0*
T02
mul"
mul	mul_0:z:0*
_input_shapes
:: :

_output_shapes
: 
ń
a
zero_fraction_cond_true_34957
3count_nonzero_notequal_zero_fraction_readvariableop
cast	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0*
_output_shapes
: 2
count_nonzero/nonzero_countj
CastCast$count_nonzero/nonzero_count:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast"
castCast:y:0*
_input_shapes

::$  

_output_shapes

:
ń
a
zero_fraction_cond_true_33667
3count_nonzero_notequal_zero_fraction_readvariableop
cast	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0*
_output_shapes
: 2
count_nonzero/nonzero_countj
CastCast$count_nonzero/nonzero_count:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast"
castCast:y:0*
_input_shapes

::$  

_output_shapes

:

ł
1zero_fraction_total_zero_zero_count_13_false_3701G
Czero_fraction_readvariableop_linear_linear_model_feature_21_weights)
%cast_zero_fraction_total_size_size_13	
mulŔ
zero_fraction/ReadVariableOpReadVariableOpCzero_fraction_readvariableop_linear_linear_model_feature_21_weights*
_output_shapes

:*
dtype02
zero_fraction/ReadVariableOpj
zero_fraction/SizeConst*
_output_shapes
: *
dtype0	*
value	B	 R2
zero_fraction/Size|
zero_fraction/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R˙˙˙˙2
zero_fraction/LessEqual/yĄ
zero_fraction/LessEqual	LessEqualzero_fraction/Size:output:0"zero_fraction/LessEqual/y:output:0*
T0	*
_output_shapes
: 2
zero_fraction/LessEqualů
zero_fraction/condStatelessIfzero_fraction/LessEqual:z:0$zero_fraction/ReadVariableOp:value:0*
Tcond0
*
Tin
2*
Tout
2	*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
else_branch!R
zero_fraction_cond_false_3711*
output_shapes
: */
then_branch R
zero_fraction_cond_true_37102
zero_fraction/cond
zero_fraction/cond/IdentityIdentityzero_fraction/cond:output:0*
T0	*
_output_shapes
: 2
zero_fraction/cond/Identityˇ
$zero_fraction/counts_to_fraction/subSubzero_fraction/Size:output:0$zero_fraction/cond/Identity:output:0*
T0	*
_output_shapes
: 2&
$zero_fraction/counts_to_fraction/sub°
%zero_fraction/counts_to_fraction/CastCast(zero_fraction/counts_to_fraction/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2'
%zero_fraction/counts_to_fraction/Cast§
'zero_fraction/counts_to_fraction/Cast_1Castzero_fraction/Size:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2)
'zero_fraction/counts_to_fraction/Cast_1Ř
(zero_fraction/counts_to_fraction/truedivRealDiv)zero_fraction/counts_to_fraction/Cast:y:0+zero_fraction/counts_to_fraction/Cast_1:y:0*
T0*
_output_shapes
: 2*
(zero_fraction/counts_to_fraction/truediv
zero_fraction/fractionIdentity,zero_fraction/counts_to_fraction/truediv:z:0*
T0*
_output_shapes
: 2
zero_fraction/fractionk
CastCast%cast_zero_fraction_total_size_size_13*

DstT0*

SrcT0	*
_output_shapes
: 2
CastG
mul_0Mulzero_fraction/fraction:output:0Cast:y:0*
T02
mul"
mul	mul_0:z:0*
_input_shapes
:: :

_output_shapes
: 
Đ
y
zero_fraction_cond_false_36257
3count_nonzero_notequal_zero_fraction_readvariableop
count_nonzero_nonzero_count	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0	*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0	*
_output_shapes
: 2
count_nonzero/nonzero_count"C
count_nonzero_nonzero_count$count_nonzero/nonzero_count:output:0*
_input_shapes

::$  

_output_shapes

:

`
/zero_fraction_total_zero_zero_count_3_true_3270
placeholder
placeholder_1		
constS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Const"
constConst:output:0*
_input_shapes
:: :

_output_shapes
: 

`
/zero_fraction_total_zero_zero_count_4_true_3313
placeholder
placeholder_1		
constS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Const"
constConst:output:0*
_input_shapes
:: :

_output_shapes
: 
ń
a
zero_fraction_cond_true_36677
3count_nonzero_notequal_zero_fraction_readvariableop
cast	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0*
_output_shapes
: 2
count_nonzero/nonzero_countj
CastCast$count_nonzero/nonzero_count:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast"
castCast:y:0*
_input_shapes

::$  

_output_shapes

:
Đ
y
zero_fraction_cond_false_31957
3count_nonzero_notequal_zero_fraction_readvariableop
count_nonzero_nonzero_count	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0	*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0	*
_output_shapes
: 2
count_nonzero/nonzero_count"C
count_nonzero_nonzero_count$count_nonzero/nonzero_count:output:0*
_input_shapes

::$  

_output_shapes

:

`
/zero_fraction_total_zero_zero_count_1_true_3184
placeholder
placeholder_1		
constS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Const"
constConst:output:0*
_input_shapes
:: :

_output_shapes
: 

ą
0zero_fraction_total_zero_zero_count_5_false_3357G
Czero_fraction_readvariableop_linear_linear_model_feature_14_weights(
$cast_zero_fraction_total_size_size_5	
mulŔ
zero_fraction/ReadVariableOpReadVariableOpCzero_fraction_readvariableop_linear_linear_model_feature_14_weights*
_output_shapes

:*
dtype02
zero_fraction/ReadVariableOpj
zero_fraction/SizeConst*
_output_shapes
: *
dtype0	*
value	B	 R2
zero_fraction/Size|
zero_fraction/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R˙˙˙˙2
zero_fraction/LessEqual/yĄ
zero_fraction/LessEqual	LessEqualzero_fraction/Size:output:0"zero_fraction/LessEqual/y:output:0*
T0	*
_output_shapes
: 2
zero_fraction/LessEqualů
zero_fraction/condStatelessIfzero_fraction/LessEqual:z:0$zero_fraction/ReadVariableOp:value:0*
Tcond0
*
Tin
2*
Tout
2	*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
else_branch!R
zero_fraction_cond_false_3367*
output_shapes
: */
then_branch R
zero_fraction_cond_true_33662
zero_fraction/cond
zero_fraction/cond/IdentityIdentityzero_fraction/cond:output:0*
T0	*
_output_shapes
: 2
zero_fraction/cond/Identityˇ
$zero_fraction/counts_to_fraction/subSubzero_fraction/Size:output:0$zero_fraction/cond/Identity:output:0*
T0	*
_output_shapes
: 2&
$zero_fraction/counts_to_fraction/sub°
%zero_fraction/counts_to_fraction/CastCast(zero_fraction/counts_to_fraction/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2'
%zero_fraction/counts_to_fraction/Cast§
'zero_fraction/counts_to_fraction/Cast_1Castzero_fraction/Size:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2)
'zero_fraction/counts_to_fraction/Cast_1Ř
(zero_fraction/counts_to_fraction/truedivRealDiv)zero_fraction/counts_to_fraction/Cast:y:0+zero_fraction/counts_to_fraction/Cast_1:y:0*
T0*
_output_shapes
: 2*
(zero_fraction/counts_to_fraction/truediv
zero_fraction/fractionIdentity,zero_fraction/counts_to_fraction/truediv:z:0*
T0*
_output_shapes
: 2
zero_fraction/fractionj
CastCast$cast_zero_fraction_total_size_size_5*

DstT0*

SrcT0	*
_output_shapes
: 2
CastG
mul_0Mulzero_fraction/fraction:output:0Cast:y:0*
T02
mul"
mul	mul_0:z:0*
_input_shapes
:: :

_output_shapes
: 
Đ
y
zero_fraction_cond_false_32387
3count_nonzero_notequal_zero_fraction_readvariableop
count_nonzero_nonzero_count	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0	*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0	*
_output_shapes
: 2
count_nonzero/nonzero_count"C
count_nonzero_nonzero_count$count_nonzero/nonzero_count:output:0*
_input_shapes

::$  

_output_shapes

:

a
0zero_fraction_total_zero_zero_count_16_true_3829
placeholder
placeholder_1		
constS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Const"
constConst:output:0*
_input_shapes
:: :

_output_shapes
: 

ą
0zero_fraction_total_zero_zero_count_8_false_3486G
Czero_fraction_readvariableop_linear_linear_model_feature_17_weights(
$cast_zero_fraction_total_size_size_8	
mulŔ
zero_fraction/ReadVariableOpReadVariableOpCzero_fraction_readvariableop_linear_linear_model_feature_17_weights*
_output_shapes

:*
dtype02
zero_fraction/ReadVariableOpj
zero_fraction/SizeConst*
_output_shapes
: *
dtype0	*
value	B	 R2
zero_fraction/Size|
zero_fraction/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R˙˙˙˙2
zero_fraction/LessEqual/yĄ
zero_fraction/LessEqual	LessEqualzero_fraction/Size:output:0"zero_fraction/LessEqual/y:output:0*
T0	*
_output_shapes
: 2
zero_fraction/LessEqualů
zero_fraction/condStatelessIfzero_fraction/LessEqual:z:0$zero_fraction/ReadVariableOp:value:0*
Tcond0
*
Tin
2*
Tout
2	*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
else_branch!R
zero_fraction_cond_false_3496*
output_shapes
: */
then_branch R
zero_fraction_cond_true_34952
zero_fraction/cond
zero_fraction/cond/IdentityIdentityzero_fraction/cond:output:0*
T0	*
_output_shapes
: 2
zero_fraction/cond/Identityˇ
$zero_fraction/counts_to_fraction/subSubzero_fraction/Size:output:0$zero_fraction/cond/Identity:output:0*
T0	*
_output_shapes
: 2&
$zero_fraction/counts_to_fraction/sub°
%zero_fraction/counts_to_fraction/CastCast(zero_fraction/counts_to_fraction/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2'
%zero_fraction/counts_to_fraction/Cast§
'zero_fraction/counts_to_fraction/Cast_1Castzero_fraction/Size:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2)
'zero_fraction/counts_to_fraction/Cast_1Ř
(zero_fraction/counts_to_fraction/truedivRealDiv)zero_fraction/counts_to_fraction/Cast:y:0+zero_fraction/counts_to_fraction/Cast_1:y:0*
T0*
_output_shapes
: 2*
(zero_fraction/counts_to_fraction/truediv
zero_fraction/fractionIdentity,zero_fraction/counts_to_fraction/truediv:z:0*
T0*
_output_shapes
: 2
zero_fraction/fractionj
CastCast$cast_zero_fraction_total_size_size_8*

DstT0*

SrcT0	*
_output_shapes
: 2
CastG
mul_0Mulzero_fraction/fraction:output:0Cast:y:0*
T02
mul"
mul	mul_0:z:0*
_input_shapes
:: :

_output_shapes
: 
Đ
y
zero_fraction_cond_false_35827
3count_nonzero_notequal_zero_fraction_readvariableop
count_nonzero_nonzero_count	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0	*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0	*
_output_shapes
: 2
count_nonzero/nonzero_count"C
count_nonzero_nonzero_count$count_nonzero/nonzero_count:output:0*
_input_shapes

::$  

_output_shapes

:
Đ
y
zero_fraction_cond_false_36687
3count_nonzero_notequal_zero_fraction_readvariableop
count_nonzero_nonzero_count	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0	*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0	*
_output_shapes
: 2
count_nonzero/nonzero_count"C
count_nonzero_nonzero_count$count_nonzero/nonzero_count:output:0*
_input_shapes

::$  

_output_shapes

:
Đ
y
zero_fraction_cond_false_34967
3count_nonzero_notequal_zero_fraction_readvariableop
count_nonzero_nonzero_count	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0	*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0	*
_output_shapes
: 2
count_nonzero/nonzero_count"C
count_nonzero_nonzero_count$count_nonzero/nonzero_count:output:0*
_input_shapes

::$  

_output_shapes

:
Đ
y
zero_fraction_cond_false_39267
3count_nonzero_notequal_zero_fraction_readvariableop
count_nonzero_nonzero_count	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0	*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0	*
_output_shapes
: 2
count_nonzero/nonzero_count"C
count_nonzero_nonzero_count$count_nonzero/nonzero_count:output:0*
_input_shapes

::$  

_output_shapes

:
ń
a
zero_fraction_cond_true_37967
3count_nonzero_notequal_zero_fraction_readvariableop
cast	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0*
_output_shapes
: 2
count_nonzero/nonzero_countj
CastCast$count_nonzero/nonzero_count:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast"
castCast:y:0*
_input_shapes

::$  

_output_shapes

:

ą
0zero_fraction_total_zero_zero_count_9_false_3529G
Czero_fraction_readvariableop_linear_linear_model_feature_18_weights(
$cast_zero_fraction_total_size_size_9	
mulŔ
zero_fraction/ReadVariableOpReadVariableOpCzero_fraction_readvariableop_linear_linear_model_feature_18_weights*
_output_shapes

:*
dtype02
zero_fraction/ReadVariableOpj
zero_fraction/SizeConst*
_output_shapes
: *
dtype0	*
value	B	 R2
zero_fraction/Size|
zero_fraction/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R˙˙˙˙2
zero_fraction/LessEqual/yĄ
zero_fraction/LessEqual	LessEqualzero_fraction/Size:output:0"zero_fraction/LessEqual/y:output:0*
T0	*
_output_shapes
: 2
zero_fraction/LessEqualů
zero_fraction/condStatelessIfzero_fraction/LessEqual:z:0$zero_fraction/ReadVariableOp:value:0*
Tcond0
*
Tin
2*
Tout
2	*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
else_branch!R
zero_fraction_cond_false_3539*
output_shapes
: */
then_branch R
zero_fraction_cond_true_35382
zero_fraction/cond
zero_fraction/cond/IdentityIdentityzero_fraction/cond:output:0*
T0	*
_output_shapes
: 2
zero_fraction/cond/Identityˇ
$zero_fraction/counts_to_fraction/subSubzero_fraction/Size:output:0$zero_fraction/cond/Identity:output:0*
T0	*
_output_shapes
: 2&
$zero_fraction/counts_to_fraction/sub°
%zero_fraction/counts_to_fraction/CastCast(zero_fraction/counts_to_fraction/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2'
%zero_fraction/counts_to_fraction/Cast§
'zero_fraction/counts_to_fraction/Cast_1Castzero_fraction/Size:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2)
'zero_fraction/counts_to_fraction/Cast_1Ř
(zero_fraction/counts_to_fraction/truedivRealDiv)zero_fraction/counts_to_fraction/Cast:y:0+zero_fraction/counts_to_fraction/Cast_1:y:0*
T0*
_output_shapes
: 2*
(zero_fraction/counts_to_fraction/truediv
zero_fraction/fractionIdentity,zero_fraction/counts_to_fraction/truediv:z:0*
T0*
_output_shapes
: 2
zero_fraction/fractionj
CastCast$cast_zero_fraction_total_size_size_9*

DstT0*

SrcT0	*
_output_shapes
: 2
CastG
mul_0Mulzero_fraction/fraction:output:0Cast:y:0*
T02
mul"
mul	mul_0:z:0*
_input_shapes
:: :

_output_shapes
: 

ą
0zero_fraction_total_zero_zero_count_3_false_3271G
Czero_fraction_readvariableop_linear_linear_model_feature_12_weights(
$cast_zero_fraction_total_size_size_3	
mulŔ
zero_fraction/ReadVariableOpReadVariableOpCzero_fraction_readvariableop_linear_linear_model_feature_12_weights*
_output_shapes

:*
dtype02
zero_fraction/ReadVariableOpj
zero_fraction/SizeConst*
_output_shapes
: *
dtype0	*
value	B	 R2
zero_fraction/Size|
zero_fraction/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R˙˙˙˙2
zero_fraction/LessEqual/yĄ
zero_fraction/LessEqual	LessEqualzero_fraction/Size:output:0"zero_fraction/LessEqual/y:output:0*
T0	*
_output_shapes
: 2
zero_fraction/LessEqualů
zero_fraction/condStatelessIfzero_fraction/LessEqual:z:0$zero_fraction/ReadVariableOp:value:0*
Tcond0
*
Tin
2*
Tout
2	*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
else_branch!R
zero_fraction_cond_false_3281*
output_shapes
: */
then_branch R
zero_fraction_cond_true_32802
zero_fraction/cond
zero_fraction/cond/IdentityIdentityzero_fraction/cond:output:0*
T0	*
_output_shapes
: 2
zero_fraction/cond/Identityˇ
$zero_fraction/counts_to_fraction/subSubzero_fraction/Size:output:0$zero_fraction/cond/Identity:output:0*
T0	*
_output_shapes
: 2&
$zero_fraction/counts_to_fraction/sub°
%zero_fraction/counts_to_fraction/CastCast(zero_fraction/counts_to_fraction/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2'
%zero_fraction/counts_to_fraction/Cast§
'zero_fraction/counts_to_fraction/Cast_1Castzero_fraction/Size:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2)
'zero_fraction/counts_to_fraction/Cast_1Ř
(zero_fraction/counts_to_fraction/truedivRealDiv)zero_fraction/counts_to_fraction/Cast:y:0+zero_fraction/counts_to_fraction/Cast_1:y:0*
T0*
_output_shapes
: 2*
(zero_fraction/counts_to_fraction/truediv
zero_fraction/fractionIdentity,zero_fraction/counts_to_fraction/truediv:z:0*
T0*
_output_shapes
: 2
zero_fraction/fractionj
CastCast$cast_zero_fraction_total_size_size_3*

DstT0*

SrcT0	*
_output_shapes
: 2
CastG
mul_0Mulzero_fraction/fraction:output:0Cast:y:0*
T02
mul"
mul	mul_0:z:0*
_input_shapes
:: :

_output_shapes
: 

ą
0zero_fraction_total_zero_zero_count_6_false_3400G
Czero_fraction_readvariableop_linear_linear_model_feature_15_weights(
$cast_zero_fraction_total_size_size_6	
mulŔ
zero_fraction/ReadVariableOpReadVariableOpCzero_fraction_readvariableop_linear_linear_model_feature_15_weights*
_output_shapes

:*
dtype02
zero_fraction/ReadVariableOpj
zero_fraction/SizeConst*
_output_shapes
: *
dtype0	*
value	B	 R2
zero_fraction/Size|
zero_fraction/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R˙˙˙˙2
zero_fraction/LessEqual/yĄ
zero_fraction/LessEqual	LessEqualzero_fraction/Size:output:0"zero_fraction/LessEqual/y:output:0*
T0	*
_output_shapes
: 2
zero_fraction/LessEqualů
zero_fraction/condStatelessIfzero_fraction/LessEqual:z:0$zero_fraction/ReadVariableOp:value:0*
Tcond0
*
Tin
2*
Tout
2	*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
else_branch!R
zero_fraction_cond_false_3410*
output_shapes
: */
then_branch R
zero_fraction_cond_true_34092
zero_fraction/cond
zero_fraction/cond/IdentityIdentityzero_fraction/cond:output:0*
T0	*
_output_shapes
: 2
zero_fraction/cond/Identityˇ
$zero_fraction/counts_to_fraction/subSubzero_fraction/Size:output:0$zero_fraction/cond/Identity:output:0*
T0	*
_output_shapes
: 2&
$zero_fraction/counts_to_fraction/sub°
%zero_fraction/counts_to_fraction/CastCast(zero_fraction/counts_to_fraction/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2'
%zero_fraction/counts_to_fraction/Cast§
'zero_fraction/counts_to_fraction/Cast_1Castzero_fraction/Size:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2)
'zero_fraction/counts_to_fraction/Cast_1Ř
(zero_fraction/counts_to_fraction/truedivRealDiv)zero_fraction/counts_to_fraction/Cast:y:0+zero_fraction/counts_to_fraction/Cast_1:y:0*
T0*
_output_shapes
: 2*
(zero_fraction/counts_to_fraction/truediv
zero_fraction/fractionIdentity,zero_fraction/counts_to_fraction/truediv:z:0*
T0*
_output_shapes
: 2
zero_fraction/fractionj
CastCast$cast_zero_fraction_total_size_size_6*

DstT0*

SrcT0	*
_output_shapes
: 2
CastG
mul_0Mulzero_fraction/fraction:output:0Cast:y:0*
T02
mul"
mul	mul_0:z:0*
_input_shapes
:: :

_output_shapes
: 

ą
0zero_fraction_total_zero_zero_count_7_false_3443G
Czero_fraction_readvariableop_linear_linear_model_feature_16_weights(
$cast_zero_fraction_total_size_size_7	
mulŔ
zero_fraction/ReadVariableOpReadVariableOpCzero_fraction_readvariableop_linear_linear_model_feature_16_weights*
_output_shapes

:*
dtype02
zero_fraction/ReadVariableOpj
zero_fraction/SizeConst*
_output_shapes
: *
dtype0	*
value	B	 R2
zero_fraction/Size|
zero_fraction/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R˙˙˙˙2
zero_fraction/LessEqual/yĄ
zero_fraction/LessEqual	LessEqualzero_fraction/Size:output:0"zero_fraction/LessEqual/y:output:0*
T0	*
_output_shapes
: 2
zero_fraction/LessEqualů
zero_fraction/condStatelessIfzero_fraction/LessEqual:z:0$zero_fraction/ReadVariableOp:value:0*
Tcond0
*
Tin
2*
Tout
2	*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
else_branch!R
zero_fraction_cond_false_3453*
output_shapes
: */
then_branch R
zero_fraction_cond_true_34522
zero_fraction/cond
zero_fraction/cond/IdentityIdentityzero_fraction/cond:output:0*
T0	*
_output_shapes
: 2
zero_fraction/cond/Identityˇ
$zero_fraction/counts_to_fraction/subSubzero_fraction/Size:output:0$zero_fraction/cond/Identity:output:0*
T0	*
_output_shapes
: 2&
$zero_fraction/counts_to_fraction/sub°
%zero_fraction/counts_to_fraction/CastCast(zero_fraction/counts_to_fraction/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2'
%zero_fraction/counts_to_fraction/Cast§
'zero_fraction/counts_to_fraction/Cast_1Castzero_fraction/Size:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2)
'zero_fraction/counts_to_fraction/Cast_1Ř
(zero_fraction/counts_to_fraction/truedivRealDiv)zero_fraction/counts_to_fraction/Cast:y:0+zero_fraction/counts_to_fraction/Cast_1:y:0*
T0*
_output_shapes
: 2*
(zero_fraction/counts_to_fraction/truediv
zero_fraction/fractionIdentity,zero_fraction/counts_to_fraction/truediv:z:0*
T0*
_output_shapes
: 2
zero_fraction/fractionj
CastCast$cast_zero_fraction_total_size_size_7*

DstT0*

SrcT0	*
_output_shapes
: 2
CastG
mul_0Mulzero_fraction/fraction:output:0Cast:y:0*
T02
mul"
mul	mul_0:z:0*
_input_shapes
:: :

_output_shapes
: 
Đ
y
zero_fraction_cond_false_39697
3count_nonzero_notequal_zero_fraction_readvariableop
count_nonzero_nonzero_count	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0	*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0	*
_output_shapes
: 2
count_nonzero/nonzero_count"C
count_nonzero_nonzero_count$count_nonzero/nonzero_count:output:0*
_input_shapes

::$  

_output_shapes

:
Đ
y
zero_fraction_cond_false_38837
3count_nonzero_notequal_zero_fraction_readvariableop
count_nonzero_nonzero_count	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0	*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0	*
_output_shapes
: 2
count_nonzero/nonzero_count"C
count_nonzero_nonzero_count$count_nonzero/nonzero_count:output:0*
_input_shapes

::$  

_output_shapes

:

a
0zero_fraction_total_zero_zero_count_17_true_3872
placeholder
placeholder_1		
constS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Const"
constConst:output:0*
_input_shapes
:: :

_output_shapes
: 
ń
a
zero_fraction_cond_true_35817
3count_nonzero_notequal_zero_fraction_readvariableop
cast	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0*
_output_shapes
: 2
count_nonzero/nonzero_countj
CastCast$count_nonzero/nonzero_count:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast"
castCast:y:0*
_input_shapes

::$  

_output_shapes

:
Đ
y
zero_fraction_cond_false_33677
3count_nonzero_notequal_zero_fraction_readvariableop
count_nonzero_nonzero_count	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0	*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0	*
_output_shapes
: 2
count_nonzero/nonzero_count"C
count_nonzero_nonzero_count$count_nonzero/nonzero_count:output:0*
_input_shapes

::$  

_output_shapes

:
Đ
y
zero_fraction_cond_false_37977
3count_nonzero_notequal_zero_fraction_readvariableop
count_nonzero_nonzero_count	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0	*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0	*
_output_shapes
: 2
count_nonzero/nonzero_count"C
count_nonzero_nonzero_count$count_nonzero/nonzero_count:output:0*
_input_shapes

::$  

_output_shapes

:

a
0zero_fraction_total_zero_zero_count_19_true_3958
placeholder
placeholder_1		
constS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Const"
constConst:output:0*
_input_shapes
:: :

_output_shapes
: 
ń
a
zero_fraction_cond_true_40117
3count_nonzero_notequal_zero_fraction_readvariableop
cast	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0*
_output_shapes
: 2
count_nonzero/nonzero_countj
CastCast$count_nonzero/nonzero_count:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast"
castCast:y:0*
_input_shapes

::$  

_output_shapes

:

a
0zero_fraction_total_zero_zero_count_21_true_4044
placeholder
placeholder_1		
constS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Const"
constConst:output:0*
_input_shapes
:: :

_output_shapes
: 
ń
a
zero_fraction_cond_true_38827
3count_nonzero_notequal_zero_fraction_readvariableop
cast	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0*
_output_shapes
: 2
count_nonzero/nonzero_countj
CastCast$count_nonzero/nonzero_count:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast"
castCast:y:0*
_input_shapes

::$  

_output_shapes

:

a
0zero_fraction_total_zero_zero_count_18_true_3915
placeholder
placeholder_1		
constS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Const"
constConst:output:0*
_input_shapes
:: :

_output_shapes
: 

˛
1zero_fraction_total_zero_zero_count_17_false_3873F
Bzero_fraction_readvariableop_linear_linear_model_feature_3_weights)
%cast_zero_fraction_total_size_size_17	
mulż
zero_fraction/ReadVariableOpReadVariableOpBzero_fraction_readvariableop_linear_linear_model_feature_3_weights*
_output_shapes

:*
dtype02
zero_fraction/ReadVariableOpj
zero_fraction/SizeConst*
_output_shapes
: *
dtype0	*
value	B	 R2
zero_fraction/Size|
zero_fraction/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R˙˙˙˙2
zero_fraction/LessEqual/yĄ
zero_fraction/LessEqual	LessEqualzero_fraction/Size:output:0"zero_fraction/LessEqual/y:output:0*
T0	*
_output_shapes
: 2
zero_fraction/LessEqualů
zero_fraction/condStatelessIfzero_fraction/LessEqual:z:0$zero_fraction/ReadVariableOp:value:0*
Tcond0
*
Tin
2*
Tout
2	*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
else_branch!R
zero_fraction_cond_false_3883*
output_shapes
: */
then_branch R
zero_fraction_cond_true_38822
zero_fraction/cond
zero_fraction/cond/IdentityIdentityzero_fraction/cond:output:0*
T0	*
_output_shapes
: 2
zero_fraction/cond/Identityˇ
$zero_fraction/counts_to_fraction/subSubzero_fraction/Size:output:0$zero_fraction/cond/Identity:output:0*
T0	*
_output_shapes
: 2&
$zero_fraction/counts_to_fraction/sub°
%zero_fraction/counts_to_fraction/CastCast(zero_fraction/counts_to_fraction/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2'
%zero_fraction/counts_to_fraction/Cast§
'zero_fraction/counts_to_fraction/Cast_1Castzero_fraction/Size:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2)
'zero_fraction/counts_to_fraction/Cast_1Ř
(zero_fraction/counts_to_fraction/truedivRealDiv)zero_fraction/counts_to_fraction/Cast:y:0+zero_fraction/counts_to_fraction/Cast_1:y:0*
T0*
_output_shapes
: 2*
(zero_fraction/counts_to_fraction/truediv
zero_fraction/fractionIdentity,zero_fraction/counts_to_fraction/truediv:z:0*
T0*
_output_shapes
: 2
zero_fraction/fractionk
CastCast%cast_zero_fraction_total_size_size_17*

DstT0*

SrcT0	*
_output_shapes
: 2
CastG
mul_0Mulzero_fraction/fraction:output:0Cast:y:0*
T02
mul"
mul	mul_0:z:0*
_input_shapes
:: :

_output_shapes
: 

˛
1zero_fraction_total_zero_zero_count_11_false_3615F
Bzero_fraction_readvariableop_linear_linear_model_feature_2_weights)
%cast_zero_fraction_total_size_size_11	
mulż
zero_fraction/ReadVariableOpReadVariableOpBzero_fraction_readvariableop_linear_linear_model_feature_2_weights*
_output_shapes

:*
dtype02
zero_fraction/ReadVariableOpj
zero_fraction/SizeConst*
_output_shapes
: *
dtype0	*
value	B	 R2
zero_fraction/Size|
zero_fraction/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R˙˙˙˙2
zero_fraction/LessEqual/yĄ
zero_fraction/LessEqual	LessEqualzero_fraction/Size:output:0"zero_fraction/LessEqual/y:output:0*
T0	*
_output_shapes
: 2
zero_fraction/LessEqualů
zero_fraction/condStatelessIfzero_fraction/LessEqual:z:0$zero_fraction/ReadVariableOp:value:0*
Tcond0
*
Tin
2*
Tout
2	*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
else_branch!R
zero_fraction_cond_false_3625*
output_shapes
: */
then_branch R
zero_fraction_cond_true_36242
zero_fraction/cond
zero_fraction/cond/IdentityIdentityzero_fraction/cond:output:0*
T0	*
_output_shapes
: 2
zero_fraction/cond/Identityˇ
$zero_fraction/counts_to_fraction/subSubzero_fraction/Size:output:0$zero_fraction/cond/Identity:output:0*
T0	*
_output_shapes
: 2&
$zero_fraction/counts_to_fraction/sub°
%zero_fraction/counts_to_fraction/CastCast(zero_fraction/counts_to_fraction/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2'
%zero_fraction/counts_to_fraction/Cast§
'zero_fraction/counts_to_fraction/Cast_1Castzero_fraction/Size:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2)
'zero_fraction/counts_to_fraction/Cast_1Ř
(zero_fraction/counts_to_fraction/truedivRealDiv)zero_fraction/counts_to_fraction/Cast:y:0+zero_fraction/counts_to_fraction/Cast_1:y:0*
T0*
_output_shapes
: 2*
(zero_fraction/counts_to_fraction/truediv
zero_fraction/fractionIdentity,zero_fraction/counts_to_fraction/truediv:z:0*
T0*
_output_shapes
: 2
zero_fraction/fractionk
CastCast%cast_zero_fraction_total_size_size_11*

DstT0*

SrcT0	*
_output_shapes
: 2
CastG
mul_0Mulzero_fraction/fraction:output:0Cast:y:0*
T02
mul"
mul	mul_0:z:0*
_input_shapes
:: :

_output_shapes
: 
ń
a
zero_fraction_cond_true_32807
3count_nonzero_notequal_zero_fraction_readvariableop
cast	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0*
_output_shapes
: 2
count_nonzero/nonzero_countj
CastCast$count_nonzero/nonzero_count:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast"
castCast:y:0*
_input_shapes

::$  

_output_shapes

:

ą
0zero_fraction_total_zero_zero_count_2_false_3228G
Czero_fraction_readvariableop_linear_linear_model_feature_11_weights(
$cast_zero_fraction_total_size_size_2	
mulŔ
zero_fraction/ReadVariableOpReadVariableOpCzero_fraction_readvariableop_linear_linear_model_feature_11_weights*
_output_shapes

:*
dtype02
zero_fraction/ReadVariableOpj
zero_fraction/SizeConst*
_output_shapes
: *
dtype0	*
value	B	 R2
zero_fraction/Size|
zero_fraction/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R˙˙˙˙2
zero_fraction/LessEqual/yĄ
zero_fraction/LessEqual	LessEqualzero_fraction/Size:output:0"zero_fraction/LessEqual/y:output:0*
T0	*
_output_shapes
: 2
zero_fraction/LessEqualů
zero_fraction/condStatelessIfzero_fraction/LessEqual:z:0$zero_fraction/ReadVariableOp:value:0*
Tcond0
*
Tin
2*
Tout
2	*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
else_branch!R
zero_fraction_cond_false_3238*
output_shapes
: */
then_branch R
zero_fraction_cond_true_32372
zero_fraction/cond
zero_fraction/cond/IdentityIdentityzero_fraction/cond:output:0*
T0	*
_output_shapes
: 2
zero_fraction/cond/Identityˇ
$zero_fraction/counts_to_fraction/subSubzero_fraction/Size:output:0$zero_fraction/cond/Identity:output:0*
T0	*
_output_shapes
: 2&
$zero_fraction/counts_to_fraction/sub°
%zero_fraction/counts_to_fraction/CastCast(zero_fraction/counts_to_fraction/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2'
%zero_fraction/counts_to_fraction/Cast§
'zero_fraction/counts_to_fraction/Cast_1Castzero_fraction/Size:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2)
'zero_fraction/counts_to_fraction/Cast_1Ř
(zero_fraction/counts_to_fraction/truedivRealDiv)zero_fraction/counts_to_fraction/Cast:y:0+zero_fraction/counts_to_fraction/Cast_1:y:0*
T0*
_output_shapes
: 2*
(zero_fraction/counts_to_fraction/truediv
zero_fraction/fractionIdentity,zero_fraction/counts_to_fraction/truediv:z:0*
T0*
_output_shapes
: 2
zero_fraction/fractionj
CastCast$cast_zero_fraction_total_size_size_2*

DstT0*

SrcT0	*
_output_shapes
: 2
CastG
mul_0Mulzero_fraction/fraction:output:0Cast:y:0*
T02
mul"
mul	mul_0:z:0*
_input_shapes
:: :

_output_shapes
: 

`
/zero_fraction_total_zero_zero_count_8_true_3485
placeholder
placeholder_1		
constS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Const"
constConst:output:0*
_input_shapes
:: :

_output_shapes
: 
ń
a
zero_fraction_cond_true_40547
3count_nonzero_notequal_zero_fraction_readvariableop
cast	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0*
_output_shapes
: 2
count_nonzero/nonzero_countj
CastCast$count_nonzero/nonzero_count:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast"
castCast:y:0*
_input_shapes

::$  

_output_shapes

:

ł
1zero_fraction_total_zero_zero_count_14_false_3744G
Czero_fraction_readvariableop_linear_linear_model_feature_22_weights)
%cast_zero_fraction_total_size_size_14	
mulŔ
zero_fraction/ReadVariableOpReadVariableOpCzero_fraction_readvariableop_linear_linear_model_feature_22_weights*
_output_shapes

:*
dtype02
zero_fraction/ReadVariableOpj
zero_fraction/SizeConst*
_output_shapes
: *
dtype0	*
value	B	 R2
zero_fraction/Size|
zero_fraction/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R˙˙˙˙2
zero_fraction/LessEqual/yĄ
zero_fraction/LessEqual	LessEqualzero_fraction/Size:output:0"zero_fraction/LessEqual/y:output:0*
T0	*
_output_shapes
: 2
zero_fraction/LessEqualů
zero_fraction/condStatelessIfzero_fraction/LessEqual:z:0$zero_fraction/ReadVariableOp:value:0*
Tcond0
*
Tin
2*
Tout
2	*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
else_branch!R
zero_fraction_cond_false_3754*
output_shapes
: */
then_branch R
zero_fraction_cond_true_37532
zero_fraction/cond
zero_fraction/cond/IdentityIdentityzero_fraction/cond:output:0*
T0	*
_output_shapes
: 2
zero_fraction/cond/Identityˇ
$zero_fraction/counts_to_fraction/subSubzero_fraction/Size:output:0$zero_fraction/cond/Identity:output:0*
T0	*
_output_shapes
: 2&
$zero_fraction/counts_to_fraction/sub°
%zero_fraction/counts_to_fraction/CastCast(zero_fraction/counts_to_fraction/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2'
%zero_fraction/counts_to_fraction/Cast§
'zero_fraction/counts_to_fraction/Cast_1Castzero_fraction/Size:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2)
'zero_fraction/counts_to_fraction/Cast_1Ř
(zero_fraction/counts_to_fraction/truedivRealDiv)zero_fraction/counts_to_fraction/Cast:y:0+zero_fraction/counts_to_fraction/Cast_1:y:0*
T0*
_output_shapes
: 2*
(zero_fraction/counts_to_fraction/truediv
zero_fraction/fractionIdentity,zero_fraction/counts_to_fraction/truediv:z:0*
T0*
_output_shapes
: 2
zero_fraction/fractionk
CastCast%cast_zero_fraction_total_size_size_14*

DstT0*

SrcT0	*
_output_shapes
: 2
CastG
mul_0Mulzero_fraction/fraction:output:0Cast:y:0*
T02
mul"
mul	mul_0:z:0*
_input_shapes
:: :

_output_shapes
: 

a
0zero_fraction_total_zero_zero_count_20_true_4001
placeholder
placeholder_1		
constS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Const"
constConst:output:0*
_input_shapes
:: :

_output_shapes
: 
ń
a
zero_fraction_cond_true_32377
3count_nonzero_notequal_zero_fraction_readvariableop
cast	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0*
_output_shapes
: 2
count_nonzero/nonzero_countj
CastCast$count_nonzero/nonzero_count:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast"
castCast:y:0*
_input_shapes

::$  

_output_shapes

:
ń
a
zero_fraction_cond_true_37537
3count_nonzero_notequal_zero_fraction_readvariableop
cast	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0*
_output_shapes
: 2
count_nonzero/nonzero_countj
CastCast$count_nonzero/nonzero_count:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast"
castCast:y:0*
_input_shapes

::$  

_output_shapes

:
Đ
y
zero_fraction_cond_false_38407
3count_nonzero_notequal_zero_fraction_readvariableop
count_nonzero_nonzero_count	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0	*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0	*
_output_shapes
: 2
count_nonzero/nonzero_count"C
count_nonzero_nonzero_count$count_nonzero/nonzero_count:output:0*
_input_shapes

::$  

_output_shapes

:

ą
0zero_fraction_total_zero_zero_count_4_false_3314G
Czero_fraction_readvariableop_linear_linear_model_feature_13_weights(
$cast_zero_fraction_total_size_size_4	
mulŔ
zero_fraction/ReadVariableOpReadVariableOpCzero_fraction_readvariableop_linear_linear_model_feature_13_weights*
_output_shapes

:*
dtype02
zero_fraction/ReadVariableOpj
zero_fraction/SizeConst*
_output_shapes
: *
dtype0	*
value	B	 R2
zero_fraction/Size|
zero_fraction/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R˙˙˙˙2
zero_fraction/LessEqual/yĄ
zero_fraction/LessEqual	LessEqualzero_fraction/Size:output:0"zero_fraction/LessEqual/y:output:0*
T0	*
_output_shapes
: 2
zero_fraction/LessEqualů
zero_fraction/condStatelessIfzero_fraction/LessEqual:z:0$zero_fraction/ReadVariableOp:value:0*
Tcond0
*
Tin
2*
Tout
2	*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
else_branch!R
zero_fraction_cond_false_3324*
output_shapes
: */
then_branch R
zero_fraction_cond_true_33232
zero_fraction/cond
zero_fraction/cond/IdentityIdentityzero_fraction/cond:output:0*
T0	*
_output_shapes
: 2
zero_fraction/cond/Identityˇ
$zero_fraction/counts_to_fraction/subSubzero_fraction/Size:output:0$zero_fraction/cond/Identity:output:0*
T0	*
_output_shapes
: 2&
$zero_fraction/counts_to_fraction/sub°
%zero_fraction/counts_to_fraction/CastCast(zero_fraction/counts_to_fraction/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2'
%zero_fraction/counts_to_fraction/Cast§
'zero_fraction/counts_to_fraction/Cast_1Castzero_fraction/Size:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2)
'zero_fraction/counts_to_fraction/Cast_1Ř
(zero_fraction/counts_to_fraction/truedivRealDiv)zero_fraction/counts_to_fraction/Cast:y:0+zero_fraction/counts_to_fraction/Cast_1:y:0*
T0*
_output_shapes
: 2*
(zero_fraction/counts_to_fraction/truediv
zero_fraction/fractionIdentity,zero_fraction/counts_to_fraction/truediv:z:0*
T0*
_output_shapes
: 2
zero_fraction/fractionj
CastCast$cast_zero_fraction_total_size_size_4*

DstT0*

SrcT0	*
_output_shapes
: 2
CastG
mul_0Mulzero_fraction/fraction:output:0Cast:y:0*
T02
mul"
mul	mul_0:z:0*
_input_shapes
:: :

_output_shapes
: 
ń
a
zero_fraction_cond_true_38397
3count_nonzero_notequal_zero_fraction_readvariableop
cast	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0*
_output_shapes
: 2
count_nonzero/nonzero_countj
CastCast$count_nonzero/nonzero_count:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast"
castCast:y:0*
_input_shapes

::$  

_output_shapes

:

˛
1zero_fraction_total_zero_zero_count_22_false_4088F
Bzero_fraction_readvariableop_linear_linear_model_feature_8_weights)
%cast_zero_fraction_total_size_size_22	
mulż
zero_fraction/ReadVariableOpReadVariableOpBzero_fraction_readvariableop_linear_linear_model_feature_8_weights*
_output_shapes

:*
dtype02
zero_fraction/ReadVariableOpj
zero_fraction/SizeConst*
_output_shapes
: *
dtype0	*
value	B	 R2
zero_fraction/Size|
zero_fraction/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R˙˙˙˙2
zero_fraction/LessEqual/yĄ
zero_fraction/LessEqual	LessEqualzero_fraction/Size:output:0"zero_fraction/LessEqual/y:output:0*
T0	*
_output_shapes
: 2
zero_fraction/LessEqualů
zero_fraction/condStatelessIfzero_fraction/LessEqual:z:0$zero_fraction/ReadVariableOp:value:0*
Tcond0
*
Tin
2*
Tout
2	*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
else_branch!R
zero_fraction_cond_false_4098*
output_shapes
: */
then_branch R
zero_fraction_cond_true_40972
zero_fraction/cond
zero_fraction/cond/IdentityIdentityzero_fraction/cond:output:0*
T0	*
_output_shapes
: 2
zero_fraction/cond/Identityˇ
$zero_fraction/counts_to_fraction/subSubzero_fraction/Size:output:0$zero_fraction/cond/Identity:output:0*
T0	*
_output_shapes
: 2&
$zero_fraction/counts_to_fraction/sub°
%zero_fraction/counts_to_fraction/CastCast(zero_fraction/counts_to_fraction/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2'
%zero_fraction/counts_to_fraction/Cast§
'zero_fraction/counts_to_fraction/Cast_1Castzero_fraction/Size:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2)
'zero_fraction/counts_to_fraction/Cast_1Ř
(zero_fraction/counts_to_fraction/truedivRealDiv)zero_fraction/counts_to_fraction/Cast:y:0+zero_fraction/counts_to_fraction/Cast_1:y:0*
T0*
_output_shapes
: 2*
(zero_fraction/counts_to_fraction/truediv
zero_fraction/fractionIdentity,zero_fraction/counts_to_fraction/truediv:z:0*
T0*
_output_shapes
: 2
zero_fraction/fractionk
CastCast%cast_zero_fraction_total_size_size_22*

DstT0*

SrcT0	*
_output_shapes
: 2
CastG
mul_0Mulzero_fraction/fraction:output:0Cast:y:0*
T02
mul"
mul	mul_0:z:0*
_input_shapes
:: :

_output_shapes
: 

a
0zero_fraction_total_zero_zero_count_22_true_4087
placeholder
placeholder_1		
constS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Const"
constConst:output:0*
_input_shapes
:: :

_output_shapes
: 
Đ
y
zero_fraction_cond_false_41417
3count_nonzero_notequal_zero_fraction_readvariableop
count_nonzero_nonzero_count	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0	*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0	*
_output_shapes
: 2
count_nonzero/nonzero_count"C
count_nonzero_nonzero_count$count_nonzero/nonzero_count:output:0*
_input_shapes

::$  

_output_shapes

:

˛
1zero_fraction_total_zero_zero_count_18_false_3916F
Bzero_fraction_readvariableop_linear_linear_model_feature_4_weights)
%cast_zero_fraction_total_size_size_18	
mulż
zero_fraction/ReadVariableOpReadVariableOpBzero_fraction_readvariableop_linear_linear_model_feature_4_weights*
_output_shapes

:*
dtype02
zero_fraction/ReadVariableOpj
zero_fraction/SizeConst*
_output_shapes
: *
dtype0	*
value	B	 R2
zero_fraction/Size|
zero_fraction/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R˙˙˙˙2
zero_fraction/LessEqual/yĄ
zero_fraction/LessEqual	LessEqualzero_fraction/Size:output:0"zero_fraction/LessEqual/y:output:0*
T0	*
_output_shapes
: 2
zero_fraction/LessEqualů
zero_fraction/condStatelessIfzero_fraction/LessEqual:z:0$zero_fraction/ReadVariableOp:value:0*
Tcond0
*
Tin
2*
Tout
2	*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
else_branch!R
zero_fraction_cond_false_3926*
output_shapes
: */
then_branch R
zero_fraction_cond_true_39252
zero_fraction/cond
zero_fraction/cond/IdentityIdentityzero_fraction/cond:output:0*
T0	*
_output_shapes
: 2
zero_fraction/cond/Identityˇ
$zero_fraction/counts_to_fraction/subSubzero_fraction/Size:output:0$zero_fraction/cond/Identity:output:0*
T0	*
_output_shapes
: 2&
$zero_fraction/counts_to_fraction/sub°
%zero_fraction/counts_to_fraction/CastCast(zero_fraction/counts_to_fraction/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2'
%zero_fraction/counts_to_fraction/Cast§
'zero_fraction/counts_to_fraction/Cast_1Castzero_fraction/Size:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2)
'zero_fraction/counts_to_fraction/Cast_1Ř
(zero_fraction/counts_to_fraction/truedivRealDiv)zero_fraction/counts_to_fraction/Cast:y:0+zero_fraction/counts_to_fraction/Cast_1:y:0*
T0*
_output_shapes
: 2*
(zero_fraction/counts_to_fraction/truediv
zero_fraction/fractionIdentity,zero_fraction/counts_to_fraction/truediv:z:0*
T0*
_output_shapes
: 2
zero_fraction/fractionk
CastCast%cast_zero_fraction_total_size_size_18*

DstT0*

SrcT0	*
_output_shapes
: 2
CastG
mul_0Mulzero_fraction/fraction:output:0Cast:y:0*
T02
mul"
mul	mul_0:z:0*
_input_shapes
:: :

_output_shapes
: 
ń
a
zero_fraction_cond_true_41407
3count_nonzero_notequal_zero_fraction_readvariableop
cast	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0*
_output_shapes
: 2
count_nonzero/nonzero_countj
CastCast$count_nonzero/nonzero_count:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast"
castCast:y:0*
_input_shapes

::$  

_output_shapes

:

Ź
.zero_fraction_total_zero_zero_count_false_3142F
Bzero_fraction_readvariableop_linear_linear_model_feature_1_weights&
"cast_zero_fraction_total_size_size	
mulż
zero_fraction/ReadVariableOpReadVariableOpBzero_fraction_readvariableop_linear_linear_model_feature_1_weights*
_output_shapes

:*
dtype02
zero_fraction/ReadVariableOpj
zero_fraction/SizeConst*
_output_shapes
: *
dtype0	*
value	B	 R2
zero_fraction/Size|
zero_fraction/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R˙˙˙˙2
zero_fraction/LessEqual/yĄ
zero_fraction/LessEqual	LessEqualzero_fraction/Size:output:0"zero_fraction/LessEqual/y:output:0*
T0	*
_output_shapes
: 2
zero_fraction/LessEqualů
zero_fraction/condStatelessIfzero_fraction/LessEqual:z:0$zero_fraction/ReadVariableOp:value:0*
Tcond0
*
Tin
2*
Tout
2	*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
else_branch!R
zero_fraction_cond_false_3152*
output_shapes
: */
then_branch R
zero_fraction_cond_true_31512
zero_fraction/cond
zero_fraction/cond/IdentityIdentityzero_fraction/cond:output:0*
T0	*
_output_shapes
: 2
zero_fraction/cond/Identityˇ
$zero_fraction/counts_to_fraction/subSubzero_fraction/Size:output:0$zero_fraction/cond/Identity:output:0*
T0	*
_output_shapes
: 2&
$zero_fraction/counts_to_fraction/sub°
%zero_fraction/counts_to_fraction/CastCast(zero_fraction/counts_to_fraction/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2'
%zero_fraction/counts_to_fraction/Cast§
'zero_fraction/counts_to_fraction/Cast_1Castzero_fraction/Size:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2)
'zero_fraction/counts_to_fraction/Cast_1Ř
(zero_fraction/counts_to_fraction/truedivRealDiv)zero_fraction/counts_to_fraction/Cast:y:0+zero_fraction/counts_to_fraction/Cast_1:y:0*
T0*
_output_shapes
: 2*
(zero_fraction/counts_to_fraction/truediv
zero_fraction/fractionIdentity,zero_fraction/counts_to_fraction/truediv:z:0*
T0*
_output_shapes
: 2
zero_fraction/fractionh
CastCast"cast_zero_fraction_total_size_size*

DstT0*

SrcT0	*
_output_shapes
: 2
CastG
mul_0Mulzero_fraction/fraction:output:0Cast:y:0*
T02
mul"
mul	mul_0:z:0*
_input_shapes
:: :

_output_shapes
: 
ń
a
zero_fraction_cond_true_40977
3count_nonzero_notequal_zero_fraction_readvariableop
cast	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0*
_output_shapes
: 2
count_nonzero/nonzero_countj
CastCast$count_nonzero/nonzero_count:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast"
castCast:y:0*
_input_shapes

::$  

_output_shapes

:

ą
0zero_fraction_total_zero_zero_count_1_false_3185G
Czero_fraction_readvariableop_linear_linear_model_feature_10_weights(
$cast_zero_fraction_total_size_size_1	
mulŔ
zero_fraction/ReadVariableOpReadVariableOpCzero_fraction_readvariableop_linear_linear_model_feature_10_weights*
_output_shapes

:*
dtype02
zero_fraction/ReadVariableOpj
zero_fraction/SizeConst*
_output_shapes
: *
dtype0	*
value	B	 R2
zero_fraction/Size|
zero_fraction/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R˙˙˙˙2
zero_fraction/LessEqual/yĄ
zero_fraction/LessEqual	LessEqualzero_fraction/Size:output:0"zero_fraction/LessEqual/y:output:0*
T0	*
_output_shapes
: 2
zero_fraction/LessEqualů
zero_fraction/condStatelessIfzero_fraction/LessEqual:z:0$zero_fraction/ReadVariableOp:value:0*
Tcond0
*
Tin
2*
Tout
2	*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
else_branch!R
zero_fraction_cond_false_3195*
output_shapes
: */
then_branch R
zero_fraction_cond_true_31942
zero_fraction/cond
zero_fraction/cond/IdentityIdentityzero_fraction/cond:output:0*
T0	*
_output_shapes
: 2
zero_fraction/cond/Identityˇ
$zero_fraction/counts_to_fraction/subSubzero_fraction/Size:output:0$zero_fraction/cond/Identity:output:0*
T0	*
_output_shapes
: 2&
$zero_fraction/counts_to_fraction/sub°
%zero_fraction/counts_to_fraction/CastCast(zero_fraction/counts_to_fraction/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2'
%zero_fraction/counts_to_fraction/Cast§
'zero_fraction/counts_to_fraction/Cast_1Castzero_fraction/Size:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2)
'zero_fraction/counts_to_fraction/Cast_1Ř
(zero_fraction/counts_to_fraction/truedivRealDiv)zero_fraction/counts_to_fraction/Cast:y:0+zero_fraction/counts_to_fraction/Cast_1:y:0*
T0*
_output_shapes
: 2*
(zero_fraction/counts_to_fraction/truediv
zero_fraction/fractionIdentity,zero_fraction/counts_to_fraction/truediv:z:0*
T0*
_output_shapes
: 2
zero_fraction/fractionj
CastCast$cast_zero_fraction_total_size_size_1*

DstT0*

SrcT0	*
_output_shapes
: 2
CastG
mul_0Mulzero_fraction/fraction:output:0Cast:y:0*
T02
mul"
mul	mul_0:z:0*
_input_shapes
:: :

_output_shapes
: 
ń
a
zero_fraction_cond_true_35387
3count_nonzero_notequal_zero_fraction_readvariableop
cast	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0*
_output_shapes
: 2
count_nonzero/nonzero_countj
CastCast$count_nonzero/nonzero_count:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast"
castCast:y:0*
_input_shapes

::$  

_output_shapes

:

`
/zero_fraction_total_zero_zero_count_2_true_3227
placeholder
placeholder_1		
constS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Const"
constConst:output:0*
_input_shapes
:: :

_output_shapes
: 

˛
1zero_fraction_total_zero_zero_count_23_false_4131F
Bzero_fraction_readvariableop_linear_linear_model_feature_9_weights)
%cast_zero_fraction_total_size_size_23	
mulż
zero_fraction/ReadVariableOpReadVariableOpBzero_fraction_readvariableop_linear_linear_model_feature_9_weights*
_output_shapes

:*
dtype02
zero_fraction/ReadVariableOpj
zero_fraction/SizeConst*
_output_shapes
: *
dtype0	*
value	B	 R2
zero_fraction/Size|
zero_fraction/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R˙˙˙˙2
zero_fraction/LessEqual/yĄ
zero_fraction/LessEqual	LessEqualzero_fraction/Size:output:0"zero_fraction/LessEqual/y:output:0*
T0	*
_output_shapes
: 2
zero_fraction/LessEqualů
zero_fraction/condStatelessIfzero_fraction/LessEqual:z:0$zero_fraction/ReadVariableOp:value:0*
Tcond0
*
Tin
2*
Tout
2	*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
else_branch!R
zero_fraction_cond_false_4141*
output_shapes
: */
then_branch R
zero_fraction_cond_true_41402
zero_fraction/cond
zero_fraction/cond/IdentityIdentityzero_fraction/cond:output:0*
T0	*
_output_shapes
: 2
zero_fraction/cond/Identityˇ
$zero_fraction/counts_to_fraction/subSubzero_fraction/Size:output:0$zero_fraction/cond/Identity:output:0*
T0	*
_output_shapes
: 2&
$zero_fraction/counts_to_fraction/sub°
%zero_fraction/counts_to_fraction/CastCast(zero_fraction/counts_to_fraction/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2'
%zero_fraction/counts_to_fraction/Cast§
'zero_fraction/counts_to_fraction/Cast_1Castzero_fraction/Size:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2)
'zero_fraction/counts_to_fraction/Cast_1Ř
(zero_fraction/counts_to_fraction/truedivRealDiv)zero_fraction/counts_to_fraction/Cast:y:0+zero_fraction/counts_to_fraction/Cast_1:y:0*
T0*
_output_shapes
: 2*
(zero_fraction/counts_to_fraction/truediv
zero_fraction/fractionIdentity,zero_fraction/counts_to_fraction/truediv:z:0*
T0*
_output_shapes
: 2
zero_fraction/fractionk
CastCast%cast_zero_fraction_total_size_size_23*

DstT0*

SrcT0	*
_output_shapes
: 2
CastG
mul_0Mulzero_fraction/fraction:output:0Cast:y:0*
T02
mul"
mul	mul_0:z:0*
_input_shapes
:: :

_output_shapes
: 
ń
a
zero_fraction_cond_true_31517
3count_nonzero_notequal_zero_fraction_readvariableop
cast	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0*
_output_shapes
: 2
count_nonzero/nonzero_countj
CastCast$count_nonzero/nonzero_count:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast"
castCast:y:0*
_input_shapes

::$  

_output_shapes

:

a
0zero_fraction_total_zero_zero_count_13_true_3700
placeholder
placeholder_1		
constS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Const"
constConst:output:0*
_input_shapes
:: :

_output_shapes
: 
ń
a
zero_fraction_cond_true_39257
3count_nonzero_notequal_zero_fraction_readvariableop
cast	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0*
_output_shapes
: 2
count_nonzero/nonzero_countj
CastCast$count_nonzero/nonzero_count:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast"
castCast:y:0*
_input_shapes

::$  

_output_shapes

:
Đ
y
zero_fraction_cond_false_33247
3count_nonzero_notequal_zero_fraction_readvariableop
count_nonzero_nonzero_count	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0	*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0	*
_output_shapes
: 2
count_nonzero/nonzero_count"C
count_nonzero_nonzero_count$count_nonzero/nonzero_count:output:0*
_input_shapes

::$  

_output_shapes

:
ń
a
zero_fraction_cond_true_34097
3count_nonzero_notequal_zero_fraction_readvariableop
cast	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0*
_output_shapes
: 2
count_nonzero/nonzero_countj
CastCast$count_nonzero/nonzero_count:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast"
castCast:y:0*
_input_shapes

::$  

_output_shapes

:
Đ
y
zero_fraction_cond_false_34107
3count_nonzero_notequal_zero_fraction_readvariableop
count_nonzero_nonzero_count	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0	*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0	*
_output_shapes
: 2
count_nonzero/nonzero_count"C
count_nonzero_nonzero_count$count_nonzero/nonzero_count:output:0*
_input_shapes

::$  

_output_shapes

:
ń
a
zero_fraction_cond_true_34527
3count_nonzero_notequal_zero_fraction_readvariableop
cast	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0*
_output_shapes
: 2
count_nonzero/nonzero_countj
CastCast$count_nonzero/nonzero_count:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast"
castCast:y:0*
_input_shapes

::$  

_output_shapes

:

`
/zero_fraction_total_zero_zero_count_6_true_3399
placeholder
placeholder_1		
constS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Const"
constConst:output:0*
_input_shapes
:: :

_output_shapes
: 

˛
1zero_fraction_total_zero_zero_count_19_false_3959F
Bzero_fraction_readvariableop_linear_linear_model_feature_5_weights)
%cast_zero_fraction_total_size_size_19	
mulż
zero_fraction/ReadVariableOpReadVariableOpBzero_fraction_readvariableop_linear_linear_model_feature_5_weights*
_output_shapes

:*
dtype02
zero_fraction/ReadVariableOpj
zero_fraction/SizeConst*
_output_shapes
: *
dtype0	*
value	B	 R2
zero_fraction/Size|
zero_fraction/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R˙˙˙˙2
zero_fraction/LessEqual/yĄ
zero_fraction/LessEqual	LessEqualzero_fraction/Size:output:0"zero_fraction/LessEqual/y:output:0*
T0	*
_output_shapes
: 2
zero_fraction/LessEqualů
zero_fraction/condStatelessIfzero_fraction/LessEqual:z:0$zero_fraction/ReadVariableOp:value:0*
Tcond0
*
Tin
2*
Tout
2	*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
else_branch!R
zero_fraction_cond_false_3969*
output_shapes
: */
then_branch R
zero_fraction_cond_true_39682
zero_fraction/cond
zero_fraction/cond/IdentityIdentityzero_fraction/cond:output:0*
T0	*
_output_shapes
: 2
zero_fraction/cond/Identityˇ
$zero_fraction/counts_to_fraction/subSubzero_fraction/Size:output:0$zero_fraction/cond/Identity:output:0*
T0	*
_output_shapes
: 2&
$zero_fraction/counts_to_fraction/sub°
%zero_fraction/counts_to_fraction/CastCast(zero_fraction/counts_to_fraction/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2'
%zero_fraction/counts_to_fraction/Cast§
'zero_fraction/counts_to_fraction/Cast_1Castzero_fraction/Size:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2)
'zero_fraction/counts_to_fraction/Cast_1Ř
(zero_fraction/counts_to_fraction/truedivRealDiv)zero_fraction/counts_to_fraction/Cast:y:0+zero_fraction/counts_to_fraction/Cast_1:y:0*
T0*
_output_shapes
: 2*
(zero_fraction/counts_to_fraction/truediv
zero_fraction/fractionIdentity,zero_fraction/counts_to_fraction/truediv:z:0*
T0*
_output_shapes
: 2
zero_fraction/fractionk
CastCast%cast_zero_fraction_total_size_size_19*

DstT0*

SrcT0	*
_output_shapes
: 2
CastG
mul_0Mulzero_fraction/fraction:output:0Cast:y:0*
T02
mul"
mul	mul_0:z:0*
_input_shapes
:: :

_output_shapes
: 

a
0zero_fraction_total_zero_zero_count_10_true_3571
placeholder
placeholder_1		
constS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Const"
constConst:output:0*
_input_shapes
:: :

_output_shapes
: 
Đ
y
zero_fraction_cond_false_40987
3count_nonzero_notequal_zero_fraction_readvariableop
count_nonzero_nonzero_count	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0	*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0	*
_output_shapes
: 2
count_nonzero/nonzero_count"C
count_nonzero_nonzero_count$count_nonzero/nonzero_count:output:0*
_input_shapes

::$  

_output_shapes

:

a
0zero_fraction_total_zero_zero_count_23_true_4130
placeholder
placeholder_1		
constS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Const"
constConst:output:0*
_input_shapes
:: :

_output_shapes
: 
Đ
y
zero_fraction_cond_false_37117
3count_nonzero_notequal_zero_fraction_readvariableop
count_nonzero_nonzero_count	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0	*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0	*
_output_shapes
: 2
count_nonzero/nonzero_count"C
count_nonzero_nonzero_count$count_nonzero/nonzero_count:output:0*
_input_shapes

::$  

_output_shapes

:
Đ
y
zero_fraction_cond_false_37547
3count_nonzero_notequal_zero_fraction_readvariableop
count_nonzero_nonzero_count	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0	*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0	*
_output_shapes
: 2
count_nonzero/nonzero_count"C
count_nonzero_nonzero_count$count_nonzero/nonzero_count:output:0*
_input_shapes

::$  

_output_shapes

:
Đ
y
zero_fraction_cond_false_32817
3count_nonzero_notequal_zero_fraction_readvariableop
count_nonzero_nonzero_count	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0	*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0	*
_output_shapes
: 2
count_nonzero/nonzero_count"C
count_nonzero_nonzero_count$count_nonzero/nonzero_count:output:0*
_input_shapes

::$  

_output_shapes

:

˛
1zero_fraction_total_zero_zero_count_21_false_4045F
Bzero_fraction_readvariableop_linear_linear_model_feature_7_weights)
%cast_zero_fraction_total_size_size_21	
mulż
zero_fraction/ReadVariableOpReadVariableOpBzero_fraction_readvariableop_linear_linear_model_feature_7_weights*
_output_shapes

:*
dtype02
zero_fraction/ReadVariableOpj
zero_fraction/SizeConst*
_output_shapes
: *
dtype0	*
value	B	 R2
zero_fraction/Size|
zero_fraction/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R˙˙˙˙2
zero_fraction/LessEqual/yĄ
zero_fraction/LessEqual	LessEqualzero_fraction/Size:output:0"zero_fraction/LessEqual/y:output:0*
T0	*
_output_shapes
: 2
zero_fraction/LessEqualů
zero_fraction/condStatelessIfzero_fraction/LessEqual:z:0$zero_fraction/ReadVariableOp:value:0*
Tcond0
*
Tin
2*
Tout
2	*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
else_branch!R
zero_fraction_cond_false_4055*
output_shapes
: */
then_branch R
zero_fraction_cond_true_40542
zero_fraction/cond
zero_fraction/cond/IdentityIdentityzero_fraction/cond:output:0*
T0	*
_output_shapes
: 2
zero_fraction/cond/Identityˇ
$zero_fraction/counts_to_fraction/subSubzero_fraction/Size:output:0$zero_fraction/cond/Identity:output:0*
T0	*
_output_shapes
: 2&
$zero_fraction/counts_to_fraction/sub°
%zero_fraction/counts_to_fraction/CastCast(zero_fraction/counts_to_fraction/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2'
%zero_fraction/counts_to_fraction/Cast§
'zero_fraction/counts_to_fraction/Cast_1Castzero_fraction/Size:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2)
'zero_fraction/counts_to_fraction/Cast_1Ř
(zero_fraction/counts_to_fraction/truedivRealDiv)zero_fraction/counts_to_fraction/Cast:y:0+zero_fraction/counts_to_fraction/Cast_1:y:0*
T0*
_output_shapes
: 2*
(zero_fraction/counts_to_fraction/truediv
zero_fraction/fractionIdentity,zero_fraction/counts_to_fraction/truediv:z:0*
T0*
_output_shapes
: 2
zero_fraction/fractionk
CastCast%cast_zero_fraction_total_size_size_21*

DstT0*

SrcT0	*
_output_shapes
: 2
CastG
mul_0Mulzero_fraction/fraction:output:0Cast:y:0*
T02
mul"
mul	mul_0:z:0*
_input_shapes
:: :

_output_shapes
: 
Đ
y
zero_fraction_cond_false_40557
3count_nonzero_notequal_zero_fraction_readvariableop
count_nonzero_nonzero_count	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0	*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0	*
_output_shapes
: 2
count_nonzero/nonzero_count"C
count_nonzero_nonzero_count$count_nonzero/nonzero_count:output:0*
_input_shapes

::$  

_output_shapes

:

ł
1zero_fraction_total_zero_zero_count_15_false_3787G
Czero_fraction_readvariableop_linear_linear_model_feature_23_weights)
%cast_zero_fraction_total_size_size_15	
mulŔ
zero_fraction/ReadVariableOpReadVariableOpCzero_fraction_readvariableop_linear_linear_model_feature_23_weights*
_output_shapes

:*
dtype02
zero_fraction/ReadVariableOpj
zero_fraction/SizeConst*
_output_shapes
: *
dtype0	*
value	B	 R2
zero_fraction/Size|
zero_fraction/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R˙˙˙˙2
zero_fraction/LessEqual/yĄ
zero_fraction/LessEqual	LessEqualzero_fraction/Size:output:0"zero_fraction/LessEqual/y:output:0*
T0	*
_output_shapes
: 2
zero_fraction/LessEqualů
zero_fraction/condStatelessIfzero_fraction/LessEqual:z:0$zero_fraction/ReadVariableOp:value:0*
Tcond0
*
Tin
2*
Tout
2	*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
else_branch!R
zero_fraction_cond_false_3797*
output_shapes
: */
then_branch R
zero_fraction_cond_true_37962
zero_fraction/cond
zero_fraction/cond/IdentityIdentityzero_fraction/cond:output:0*
T0	*
_output_shapes
: 2
zero_fraction/cond/Identityˇ
$zero_fraction/counts_to_fraction/subSubzero_fraction/Size:output:0$zero_fraction/cond/Identity:output:0*
T0	*
_output_shapes
: 2&
$zero_fraction/counts_to_fraction/sub°
%zero_fraction/counts_to_fraction/CastCast(zero_fraction/counts_to_fraction/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2'
%zero_fraction/counts_to_fraction/Cast§
'zero_fraction/counts_to_fraction/Cast_1Castzero_fraction/Size:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2)
'zero_fraction/counts_to_fraction/Cast_1Ř
(zero_fraction/counts_to_fraction/truedivRealDiv)zero_fraction/counts_to_fraction/Cast:y:0+zero_fraction/counts_to_fraction/Cast_1:y:0*
T0*
_output_shapes
: 2*
(zero_fraction/counts_to_fraction/truediv
zero_fraction/fractionIdentity,zero_fraction/counts_to_fraction/truediv:z:0*
T0*
_output_shapes
: 2
zero_fraction/fractionk
CastCast%cast_zero_fraction_total_size_size_15*

DstT0*

SrcT0	*
_output_shapes
: 2
CastG
mul_0Mulzero_fraction/fraction:output:0Cast:y:0*
T02
mul"
mul	mul_0:z:0*
_input_shapes
:: :

_output_shapes
: 

`
/zero_fraction_total_zero_zero_count_5_true_3356
placeholder
placeholder_1		
constS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Const"
constConst:output:0*
_input_shapes
:: :

_output_shapes
: 

^
-zero_fraction_total_zero_zero_count_true_3141
placeholder
placeholder_1		
constS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Const"
constConst:output:0*
_input_shapes
:: :

_output_shapes
: 

a
0zero_fraction_total_zero_zero_count_14_true_3743
placeholder
placeholder_1		
constS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Const"
constConst:output:0*
_input_shapes
:: :

_output_shapes
: 

a
0zero_fraction_total_zero_zero_count_12_true_3657
placeholder
placeholder_1		
constS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Const"
constConst:output:0*
_input_shapes
:: :

_output_shapes
: 
Đ
y
zero_fraction_cond_false_31527
3count_nonzero_notequal_zero_fraction_readvariableop
count_nonzero_nonzero_count	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0	*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0	*
_output_shapes
: 2
count_nonzero/nonzero_count"C
count_nonzero_nonzero_count$count_nonzero/nonzero_count:output:0*
_input_shapes

::$  

_output_shapes

:
ń
a
zero_fraction_cond_true_39687
3count_nonzero_notequal_zero_fraction_readvariableop
cast	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0*
_output_shapes
: 2
count_nonzero/nonzero_countj
CastCast$count_nonzero/nonzero_count:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast"
castCast:y:0*
_input_shapes

::$  

_output_shapes

:
Đ
y
zero_fraction_cond_false_40127
3count_nonzero_notequal_zero_fraction_readvariableop
count_nonzero_nonzero_count	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0	*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0	*
_output_shapes
: 2
count_nonzero/nonzero_count"C
count_nonzero_nonzero_count$count_nonzero/nonzero_count:output:0*
_input_shapes

::$  

_output_shapes

:

˛
1zero_fraction_total_zero_zero_count_20_false_4002F
Bzero_fraction_readvariableop_linear_linear_model_feature_6_weights)
%cast_zero_fraction_total_size_size_20	
mulż
zero_fraction/ReadVariableOpReadVariableOpBzero_fraction_readvariableop_linear_linear_model_feature_6_weights*
_output_shapes

:*
dtype02
zero_fraction/ReadVariableOpj
zero_fraction/SizeConst*
_output_shapes
: *
dtype0	*
value	B	 R2
zero_fraction/Size|
zero_fraction/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R˙˙˙˙2
zero_fraction/LessEqual/yĄ
zero_fraction/LessEqual	LessEqualzero_fraction/Size:output:0"zero_fraction/LessEqual/y:output:0*
T0	*
_output_shapes
: 2
zero_fraction/LessEqualů
zero_fraction/condStatelessIfzero_fraction/LessEqual:z:0$zero_fraction/ReadVariableOp:value:0*
Tcond0
*
Tin
2*
Tout
2	*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
else_branch!R
zero_fraction_cond_false_4012*
output_shapes
: */
then_branch R
zero_fraction_cond_true_40112
zero_fraction/cond
zero_fraction/cond/IdentityIdentityzero_fraction/cond:output:0*
T0	*
_output_shapes
: 2
zero_fraction/cond/Identityˇ
$zero_fraction/counts_to_fraction/subSubzero_fraction/Size:output:0$zero_fraction/cond/Identity:output:0*
T0	*
_output_shapes
: 2&
$zero_fraction/counts_to_fraction/sub°
%zero_fraction/counts_to_fraction/CastCast(zero_fraction/counts_to_fraction/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2'
%zero_fraction/counts_to_fraction/Cast§
'zero_fraction/counts_to_fraction/Cast_1Castzero_fraction/Size:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2)
'zero_fraction/counts_to_fraction/Cast_1Ř
(zero_fraction/counts_to_fraction/truedivRealDiv)zero_fraction/counts_to_fraction/Cast:y:0+zero_fraction/counts_to_fraction/Cast_1:y:0*
T0*
_output_shapes
: 2*
(zero_fraction/counts_to_fraction/truediv
zero_fraction/fractionIdentity,zero_fraction/counts_to_fraction/truediv:z:0*
T0*
_output_shapes
: 2
zero_fraction/fractionk
CastCast%cast_zero_fraction_total_size_size_20*

DstT0*

SrcT0	*
_output_shapes
: 2
CastG
mul_0Mulzero_fraction/fraction:output:0Cast:y:0*
T02
mul"
mul	mul_0:z:0*
_input_shapes
:: :

_output_shapes
: 

ł
1zero_fraction_total_zero_zero_count_10_false_3572G
Czero_fraction_readvariableop_linear_linear_model_feature_19_weights)
%cast_zero_fraction_total_size_size_10	
mulŔ
zero_fraction/ReadVariableOpReadVariableOpCzero_fraction_readvariableop_linear_linear_model_feature_19_weights*
_output_shapes

:*
dtype02
zero_fraction/ReadVariableOpj
zero_fraction/SizeConst*
_output_shapes
: *
dtype0	*
value	B	 R2
zero_fraction/Size|
zero_fraction/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R˙˙˙˙2
zero_fraction/LessEqual/yĄ
zero_fraction/LessEqual	LessEqualzero_fraction/Size:output:0"zero_fraction/LessEqual/y:output:0*
T0	*
_output_shapes
: 2
zero_fraction/LessEqualů
zero_fraction/condStatelessIfzero_fraction/LessEqual:z:0$zero_fraction/ReadVariableOp:value:0*
Tcond0
*
Tin
2*
Tout
2	*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
else_branch!R
zero_fraction_cond_false_3582*
output_shapes
: */
then_branch R
zero_fraction_cond_true_35812
zero_fraction/cond
zero_fraction/cond/IdentityIdentityzero_fraction/cond:output:0*
T0	*
_output_shapes
: 2
zero_fraction/cond/Identityˇ
$zero_fraction/counts_to_fraction/subSubzero_fraction/Size:output:0$zero_fraction/cond/Identity:output:0*
T0	*
_output_shapes
: 2&
$zero_fraction/counts_to_fraction/sub°
%zero_fraction/counts_to_fraction/CastCast(zero_fraction/counts_to_fraction/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2'
%zero_fraction/counts_to_fraction/Cast§
'zero_fraction/counts_to_fraction/Cast_1Castzero_fraction/Size:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2)
'zero_fraction/counts_to_fraction/Cast_1Ř
(zero_fraction/counts_to_fraction/truedivRealDiv)zero_fraction/counts_to_fraction/Cast:y:0+zero_fraction/counts_to_fraction/Cast_1:y:0*
T0*
_output_shapes
: 2*
(zero_fraction/counts_to_fraction/truediv
zero_fraction/fractionIdentity,zero_fraction/counts_to_fraction/truediv:z:0*
T0*
_output_shapes
: 2
zero_fraction/fractionk
CastCast%cast_zero_fraction_total_size_size_10*

DstT0*

SrcT0	*
_output_shapes
: 2
CastG
mul_0Mulzero_fraction/fraction:output:0Cast:y:0*
T02
mul"
mul	mul_0:z:0*
_input_shapes
:: :

_output_shapes
: 

`
/zero_fraction_total_zero_zero_count_7_true_3442
placeholder
placeholder_1		
constS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Const"
constConst:output:0*
_input_shapes
:: :

_output_shapes
: 
ń
a
zero_fraction_cond_true_33237
3count_nonzero_notequal_zero_fraction_readvariableop
cast	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0*
_output_shapes
: 2
count_nonzero/nonzero_countj
CastCast$count_nonzero/nonzero_count:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast"
castCast:y:0*
_input_shapes

::$  

_output_shapes

:
Đ
y
zero_fraction_cond_false_34537
3count_nonzero_notequal_zero_fraction_readvariableop
count_nonzero_nonzero_count	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0	*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0	*
_output_shapes
: 2
count_nonzero/nonzero_count"C
count_nonzero_nonzero_count$count_nonzero/nonzero_count:output:0*
_input_shapes

::$  

_output_shapes

:
ń
a
zero_fraction_cond_true_31947
3count_nonzero_notequal_zero_fraction_readvariableop
cast	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0*
_output_shapes
: 2
count_nonzero/nonzero_countj
CastCast$count_nonzero/nonzero_count:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast"
castCast:y:0*
_input_shapes

::$  

_output_shapes

:

`
/zero_fraction_total_zero_zero_count_9_true_3528
placeholder
placeholder_1		
constS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Const"
constConst:output:0*
_input_shapes
:: :

_output_shapes
: 

a
0zero_fraction_total_zero_zero_count_11_true_3614
placeholder
placeholder_1		
constS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Const"
constConst:output:0*
_input_shapes
:: :

_output_shapes
: 
ń
a
zero_fraction_cond_true_36247
3count_nonzero_notequal_zero_fraction_readvariableop
cast	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0*
_output_shapes
: 2
count_nonzero/nonzero_countj
CastCast$count_nonzero/nonzero_count:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast"
castCast:y:0*
_input_shapes

::$  

_output_shapes

:

a
0zero_fraction_total_zero_zero_count_15_true_3786
placeholder
placeholder_1		
constS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Const"
constConst:output:0*
_input_shapes
:: :

_output_shapes
: 
ń
a
zero_fraction_cond_true_37107
3count_nonzero_notequal_zero_fraction_readvariableop
cast	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0*
_output_shapes
: 2
count_nonzero/nonzero_countj
CastCast$count_nonzero/nonzero_count:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast"
castCast:y:0*
_input_shapes

::$  

_output_shapes

:"ą<
save/Const:0save/Identity:0save/restore_all (5 @F8"~
global_stepom
k
global_step:0global_step/Assign!global_step/Read/ReadVariableOp:0(2global_step/Initializer/zeros:0H"%
saved_model_main_op


group_deps"3
	summaries&
$
bias:0
fraction_of_zero_weights:0"Š*
trainable_variables**
Ó
'linear/linear_model/Feature_1/weights:0,linear/linear_model/Feature_1/weights/Assign;linear/linear_model/Feature_1/weights/Read/ReadVariableOp:0(29linear/linear_model/Feature_1/weights/Initializer/zeros:08
×
(linear/linear_model/Feature_10/weights:0-linear/linear_model/Feature_10/weights/Assign<linear/linear_model/Feature_10/weights/Read/ReadVariableOp:0(2:linear/linear_model/Feature_10/weights/Initializer/zeros:08
×
(linear/linear_model/Feature_11/weights:0-linear/linear_model/Feature_11/weights/Assign<linear/linear_model/Feature_11/weights/Read/ReadVariableOp:0(2:linear/linear_model/Feature_11/weights/Initializer/zeros:08
×
(linear/linear_model/Feature_12/weights:0-linear/linear_model/Feature_12/weights/Assign<linear/linear_model/Feature_12/weights/Read/ReadVariableOp:0(2:linear/linear_model/Feature_12/weights/Initializer/zeros:08
×
(linear/linear_model/Feature_13/weights:0-linear/linear_model/Feature_13/weights/Assign<linear/linear_model/Feature_13/weights/Read/ReadVariableOp:0(2:linear/linear_model/Feature_13/weights/Initializer/zeros:08
×
(linear/linear_model/Feature_14/weights:0-linear/linear_model/Feature_14/weights/Assign<linear/linear_model/Feature_14/weights/Read/ReadVariableOp:0(2:linear/linear_model/Feature_14/weights/Initializer/zeros:08
×
(linear/linear_model/Feature_15/weights:0-linear/linear_model/Feature_15/weights/Assign<linear/linear_model/Feature_15/weights/Read/ReadVariableOp:0(2:linear/linear_model/Feature_15/weights/Initializer/zeros:08
×
(linear/linear_model/Feature_16/weights:0-linear/linear_model/Feature_16/weights/Assign<linear/linear_model/Feature_16/weights/Read/ReadVariableOp:0(2:linear/linear_model/Feature_16/weights/Initializer/zeros:08
×
(linear/linear_model/Feature_17/weights:0-linear/linear_model/Feature_17/weights/Assign<linear/linear_model/Feature_17/weights/Read/ReadVariableOp:0(2:linear/linear_model/Feature_17/weights/Initializer/zeros:08
×
(linear/linear_model/Feature_18/weights:0-linear/linear_model/Feature_18/weights/Assign<linear/linear_model/Feature_18/weights/Read/ReadVariableOp:0(2:linear/linear_model/Feature_18/weights/Initializer/zeros:08
×
(linear/linear_model/Feature_19/weights:0-linear/linear_model/Feature_19/weights/Assign<linear/linear_model/Feature_19/weights/Read/ReadVariableOp:0(2:linear/linear_model/Feature_19/weights/Initializer/zeros:08
Ó
'linear/linear_model/Feature_2/weights:0,linear/linear_model/Feature_2/weights/Assign;linear/linear_model/Feature_2/weights/Read/ReadVariableOp:0(29linear/linear_model/Feature_2/weights/Initializer/zeros:08
×
(linear/linear_model/Feature_20/weights:0-linear/linear_model/Feature_20/weights/Assign<linear/linear_model/Feature_20/weights/Read/ReadVariableOp:0(2:linear/linear_model/Feature_20/weights/Initializer/zeros:08
×
(linear/linear_model/Feature_21/weights:0-linear/linear_model/Feature_21/weights/Assign<linear/linear_model/Feature_21/weights/Read/ReadVariableOp:0(2:linear/linear_model/Feature_21/weights/Initializer/zeros:08
×
(linear/linear_model/Feature_22/weights:0-linear/linear_model/Feature_22/weights/Assign<linear/linear_model/Feature_22/weights/Read/ReadVariableOp:0(2:linear/linear_model/Feature_22/weights/Initializer/zeros:08
×
(linear/linear_model/Feature_23/weights:0-linear/linear_model/Feature_23/weights/Assign<linear/linear_model/Feature_23/weights/Read/ReadVariableOp:0(2:linear/linear_model/Feature_23/weights/Initializer/zeros:08
×
(linear/linear_model/Feature_24/weights:0-linear/linear_model/Feature_24/weights/Assign<linear/linear_model/Feature_24/weights/Read/ReadVariableOp:0(2:linear/linear_model/Feature_24/weights/Initializer/zeros:08
Ó
'linear/linear_model/Feature_3/weights:0,linear/linear_model/Feature_3/weights/Assign;linear/linear_model/Feature_3/weights/Read/ReadVariableOp:0(29linear/linear_model/Feature_3/weights/Initializer/zeros:08
Ó
'linear/linear_model/Feature_4/weights:0,linear/linear_model/Feature_4/weights/Assign;linear/linear_model/Feature_4/weights/Read/ReadVariableOp:0(29linear/linear_model/Feature_4/weights/Initializer/zeros:08
Ó
'linear/linear_model/Feature_5/weights:0,linear/linear_model/Feature_5/weights/Assign;linear/linear_model/Feature_5/weights/Read/ReadVariableOp:0(29linear/linear_model/Feature_5/weights/Initializer/zeros:08
Ó
'linear/linear_model/Feature_6/weights:0,linear/linear_model/Feature_6/weights/Assign;linear/linear_model/Feature_6/weights/Read/ReadVariableOp:0(29linear/linear_model/Feature_6/weights/Initializer/zeros:08
Ó
'linear/linear_model/Feature_7/weights:0,linear/linear_model/Feature_7/weights/Assign;linear/linear_model/Feature_7/weights/Read/ReadVariableOp:0(29linear/linear_model/Feature_7/weights/Initializer/zeros:08
Ó
'linear/linear_model/Feature_8/weights:0,linear/linear_model/Feature_8/weights/Assign;linear/linear_model/Feature_8/weights/Read/ReadVariableOp:0(29linear/linear_model/Feature_8/weights/Initializer/zeros:08
Ó
'linear/linear_model/Feature_9/weights:0,linear/linear_model/Feature_9/weights/Assign;linear/linear_model/Feature_9/weights/Read/ReadVariableOp:0(29linear/linear_model/Feature_9/weights/Initializer/zeros:08
ż
"linear/linear_model/bias_weights:0'linear/linear_model/bias_weights/Assign6linear/linear_model/bias_weights/Read/ReadVariableOp:0(24linear/linear_model/bias_weights/Initializer/zeros:08"+
	variablesţ*ű*
k
global_step:0global_step/Assign!global_step/Read/ReadVariableOp:0(2global_step/Initializer/zeros:0H
Ó
'linear/linear_model/Feature_1/weights:0,linear/linear_model/Feature_1/weights/Assign;linear/linear_model/Feature_1/weights/Read/ReadVariableOp:0(29linear/linear_model/Feature_1/weights/Initializer/zeros:08
×
(linear/linear_model/Feature_10/weights:0-linear/linear_model/Feature_10/weights/Assign<linear/linear_model/Feature_10/weights/Read/ReadVariableOp:0(2:linear/linear_model/Feature_10/weights/Initializer/zeros:08
×
(linear/linear_model/Feature_11/weights:0-linear/linear_model/Feature_11/weights/Assign<linear/linear_model/Feature_11/weights/Read/ReadVariableOp:0(2:linear/linear_model/Feature_11/weights/Initializer/zeros:08
×
(linear/linear_model/Feature_12/weights:0-linear/linear_model/Feature_12/weights/Assign<linear/linear_model/Feature_12/weights/Read/ReadVariableOp:0(2:linear/linear_model/Feature_12/weights/Initializer/zeros:08
×
(linear/linear_model/Feature_13/weights:0-linear/linear_model/Feature_13/weights/Assign<linear/linear_model/Feature_13/weights/Read/ReadVariableOp:0(2:linear/linear_model/Feature_13/weights/Initializer/zeros:08
×
(linear/linear_model/Feature_14/weights:0-linear/linear_model/Feature_14/weights/Assign<linear/linear_model/Feature_14/weights/Read/ReadVariableOp:0(2:linear/linear_model/Feature_14/weights/Initializer/zeros:08
×
(linear/linear_model/Feature_15/weights:0-linear/linear_model/Feature_15/weights/Assign<linear/linear_model/Feature_15/weights/Read/ReadVariableOp:0(2:linear/linear_model/Feature_15/weights/Initializer/zeros:08
×
(linear/linear_model/Feature_16/weights:0-linear/linear_model/Feature_16/weights/Assign<linear/linear_model/Feature_16/weights/Read/ReadVariableOp:0(2:linear/linear_model/Feature_16/weights/Initializer/zeros:08
×
(linear/linear_model/Feature_17/weights:0-linear/linear_model/Feature_17/weights/Assign<linear/linear_model/Feature_17/weights/Read/ReadVariableOp:0(2:linear/linear_model/Feature_17/weights/Initializer/zeros:08
×
(linear/linear_model/Feature_18/weights:0-linear/linear_model/Feature_18/weights/Assign<linear/linear_model/Feature_18/weights/Read/ReadVariableOp:0(2:linear/linear_model/Feature_18/weights/Initializer/zeros:08
×
(linear/linear_model/Feature_19/weights:0-linear/linear_model/Feature_19/weights/Assign<linear/linear_model/Feature_19/weights/Read/ReadVariableOp:0(2:linear/linear_model/Feature_19/weights/Initializer/zeros:08
Ó
'linear/linear_model/Feature_2/weights:0,linear/linear_model/Feature_2/weights/Assign;linear/linear_model/Feature_2/weights/Read/ReadVariableOp:0(29linear/linear_model/Feature_2/weights/Initializer/zeros:08
×
(linear/linear_model/Feature_20/weights:0-linear/linear_model/Feature_20/weights/Assign<linear/linear_model/Feature_20/weights/Read/ReadVariableOp:0(2:linear/linear_model/Feature_20/weights/Initializer/zeros:08
×
(linear/linear_model/Feature_21/weights:0-linear/linear_model/Feature_21/weights/Assign<linear/linear_model/Feature_21/weights/Read/ReadVariableOp:0(2:linear/linear_model/Feature_21/weights/Initializer/zeros:08
×
(linear/linear_model/Feature_22/weights:0-linear/linear_model/Feature_22/weights/Assign<linear/linear_model/Feature_22/weights/Read/ReadVariableOp:0(2:linear/linear_model/Feature_22/weights/Initializer/zeros:08
×
(linear/linear_model/Feature_23/weights:0-linear/linear_model/Feature_23/weights/Assign<linear/linear_model/Feature_23/weights/Read/ReadVariableOp:0(2:linear/linear_model/Feature_23/weights/Initializer/zeros:08
×
(linear/linear_model/Feature_24/weights:0-linear/linear_model/Feature_24/weights/Assign<linear/linear_model/Feature_24/weights/Read/ReadVariableOp:0(2:linear/linear_model/Feature_24/weights/Initializer/zeros:08
Ó
'linear/linear_model/Feature_3/weights:0,linear/linear_model/Feature_3/weights/Assign;linear/linear_model/Feature_3/weights/Read/ReadVariableOp:0(29linear/linear_model/Feature_3/weights/Initializer/zeros:08
Ó
'linear/linear_model/Feature_4/weights:0,linear/linear_model/Feature_4/weights/Assign;linear/linear_model/Feature_4/weights/Read/ReadVariableOp:0(29linear/linear_model/Feature_4/weights/Initializer/zeros:08
Ó
'linear/linear_model/Feature_5/weights:0,linear/linear_model/Feature_5/weights/Assign;linear/linear_model/Feature_5/weights/Read/ReadVariableOp:0(29linear/linear_model/Feature_5/weights/Initializer/zeros:08
Ó
'linear/linear_model/Feature_6/weights:0,linear/linear_model/Feature_6/weights/Assign;linear/linear_model/Feature_6/weights/Read/ReadVariableOp:0(29linear/linear_model/Feature_6/weights/Initializer/zeros:08
Ó
'linear/linear_model/Feature_7/weights:0,linear/linear_model/Feature_7/weights/Assign;linear/linear_model/Feature_7/weights/Read/ReadVariableOp:0(29linear/linear_model/Feature_7/weights/Initializer/zeros:08
Ó
'linear/linear_model/Feature_8/weights:0,linear/linear_model/Feature_8/weights/Assign;linear/linear_model/Feature_8/weights/Read/ReadVariableOp:0(29linear/linear_model/Feature_8/weights/Initializer/zeros:08
Ó
'linear/linear_model/Feature_9/weights:0,linear/linear_model/Feature_9/weights/Assign;linear/linear_model/Feature_9/weights/Read/ReadVariableOp:0(29linear/linear_model/Feature_9/weights/Initializer/zeros:08
ż
"linear/linear_model/bias_weights:0'linear/linear_model/bias_weights/Assign6linear/linear_model/bias_weights/Read/ReadVariableOp:0(24linear/linear_model/bias_weights/Initializer/zeros:08*×
classificationÄ
3
inputs)
input_example_tensor:0˙˙˙˙˙˙˙˙˙-
classes"
head/Tile:0˙˙˙˙˙˙˙˙˙A
scores7
 head/predictions/probabilities:0˙˙˙˙˙˙˙˙˙tensorflow/serving/classify*Ý
predictŃ
5
examples)
input_example_tensor:0˙˙˙˙˙˙˙˙˙?
all_class_ids.
head/predictions/Tile:0˙˙˙˙˙˙˙˙˙?
all_classes0
head/predictions/Tile_1:0˙˙˙˙˙˙˙˙˙A
	class_ids4
head/predictions/ExpandDims:0	˙˙˙˙˙˙˙˙˙@
classes5
head/predictions/str_classes:0˙˙˙˙˙˙˙˙˙>
logistic2
head/predictions/logistic:0˙˙˙˙˙˙˙˙˙k
logitsa
Jlinear/linear_model/linear/linear_model/linear/linear_model/weighted_sum:0˙˙˙˙˙˙˙˙˙H
probabilities7
 head/predictions/probabilities:0˙˙˙˙˙˙˙˙˙tensorflow/serving/predict*

regression
3
inputs)
input_example_tensor:0˙˙˙˙˙˙˙˙˙=
outputs2
head/predictions/logistic:0˙˙˙˙˙˙˙˙˙tensorflow/serving/regress*Ř
serving_defaultÄ
3
inputs)
input_example_tensor:0˙˙˙˙˙˙˙˙˙-
classes"
head/Tile:0˙˙˙˙˙˙˙˙˙A
scores7
 head/predictions/probabilities:0˙˙˙˙˙˙˙˙˙tensorflow/serving/classify