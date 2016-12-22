--[[
Due to interest of time, please prepared the data before-hand into a 4D torch
ByteTensor of size 50000x3x32x32 (training) and 10000x3x32x32 (testing) 

mkdir t5
cd t5/
git clone https://github.com/soumith/cifar.torch.git
cd cifar.torch/
th Cifar10BinToTensor.lua

]]

require 'torch'
require 'image'
require 'nn'
require 'cunn'
require 'cudnn'
require 'hzproc'

cudnn.benchmark = true
cudnn.fastest = true

function saveTensorAsGrid(tensor,fileName) 
	local padding = 1
	local grid = image.toDisplayTensor(tensor/255.0, padding)
	image.save(fileName,grid)
end

local trainset = torch.load('cifar.torch/cifar10-train.t7')
local testset = torch.load('cifar.torch/cifar10-test.t7')

local classes = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

local trainData = trainset.data:float() -- convert the data from a ByteTensor to a float Tensor.
local trainLabels = trainset.label:float():add(1)
local testData = testset.data:float()
local testLabels = testset.label:float():add(1)


--  ****************************************************************
--  Data Preprocessing
--  ****************************************************************


local function normalize(train_data, test_data, i)
	local train_mean = train_data[{ {}, i, {}, {}}]:mean()
	local train_std = train_data[{ {}, i, {}, {}}]:std()
	train_data[{ {}, i, {}, {}}]:add(-train_mean)
	train_data[{ {}, i, {}, {}}]:div(train_std)
	test_data[{ {}, i, {}, {}}]:add(-train_mean)
	test_data[{ {}, i, {}, {}}]:div(train_std)
end

print("Normalizing data...")
for i = 1, 3 do
	normalize(trainData, testData, i)
end
print("Done")
-- convert to YUV and normalize


--  ****************************************************************
--  Define our neural network
--  ****************************************************************

local function He_init(net)
  local function init(name)
    for k,v in pairs(net:findModules(name)) do
      print(k)
      local n = v.kW*v.kH*v.nInputPlane
      v.weight:normal(0, math.sqrt(2/n))
	  if v.bias then v.bias:zero() end
    end
  end
  init('nn.SpatialConvolution')
end

--[[
local function addConvBlockFMP(model, nInput, nFilter, size, poolingRatio)
	model:add(nn.SpatialConvolution(nInput, nFilter, size, size))
	model:add(nn.SpatialBatchNormalization(nFilter, 1e-3))
	model:add(nn.SpatialFractionalMaxPooling(2, 2, poolingRatio, poolingRatio))
	model:add(nn.LeakyReLU(0.3, true))
end
]]

local function calcNumberOfFilters(nLayer)
	local filterBase = 8
	local filterAdd = 5
	return filterBase + filterAdd * (nLayer - 1)
end

function forwardNet(model, data, labels, train)
    --another helpful function of optim is ConfusionMatrix
    local confusion = optim.ConfusionMatrix(classes)
    local lossAcc = 0
    local numBatches = 0
    if train then
        --set network into training mode
        model:training()
    else
        model:evaluate() -- turn of drop-out
    end
    for i = 1, data:size(1) - batchSize, batchSize do
        numBatches = numBatches + 1
        local x = data:narrow(1, i, batchSize):cuda()
        local yt = labels:narrow(1, i, batchSize):cuda()
        local y = model:forward(x)
        local err = criterion:forward(y, yt)
        lossAcc = lossAcc + err
        confusion:batchAdd(y,yt)
        
        if train then
            function feval()
                model:zeroGradParameters() --zero grads
                local dE_dy = criterion:backward(y,yt)
                model:backward(x, dE_dy) -- backpropagation
            
                return err, dE_dw
            end
        
            optim.adam(feval, w, optimState)
        end
    end
    
    confusion:updateValids()
    local avgLoss = lossAcc / numBatches
    local avgError = 1 - confusion.totalValid
    
    return avgLoss, avgError, tostring(confusion)
end

function plotError(trainError, testError, title)
	require 'gnuplot'
	local range = torch.range(1, trainError:size(1))
	gnuplot.pngfigure('testVsTrainError.png')
	gnuplot.plot({'trainError',trainError},{'testError',testError})
	gnuplot.xlabel('epochs')
	gnuplot.ylabel('Error')
	gnuplot.plotflush()
end

model = nn.Sequential()
-- building block
local function ConvBNReLU(...)
	local arg = {...}
	model:add(nn.SpatialConvolution(...))
	model:add(nn.SpatialBatchNormalization(arg[2], 1e-3))
	model:add(nn.ReLU(true))
	return model
end

smallLayer = 16
mediumLayer = 32
largeLayer = 64
XlargeLayer = 128

------------------------------------------------------
ConvBNReLU(3, XlargeLayer, 3, 3, 1, 1, 1, 1)
--model:add(nn.Dropout(0.1))

ConvBNReLU(XlargeLayer, largeLayer, 1, 1)
ConvBNReLU(largeLayer, smallLayer, 1, 1)
model:add(nn.Dropout(0.2))

model:add(nn.SpatialAveragePooling(3, 3, 2, 2))
--------------------------------------------------
ConvBNReLU(smallLayer, XlargeLayer, 2, 2, 1, 1, 1, 1)
--model:add(nn.Dropout(0.2))

ConvBNReLU(XlargeLayer, largeLayer, 1, 1)
ConvBNReLU(largeLayer, mediumLayer, 1, 1)
model:add(nn.Dropout(0.3))

model:add(nn.SpatialAveragePooling(3,3,2,2))
----------------------------------------------------
ConvBNReLU(mediumLayer, largeLayer, 2, 2, 1, 1, 1, 1)
--model:add(nn.Dropout(0.3))

ConvBNReLU(largeLayer, mediumLayer, 1, 1)
model:add(nn.Dropout(0.4))

model:add(nn.SpatialAveragePooling(3,3,2,2))
----------------------------------------------------
ConvBNReLU(mediumLayer, mediumLayer, 2, 2, 1, 1, 1, 1)
--model:add(nn.Dropout(0.4))

ConvBNReLU(mediumLayer, smallLayer, 1, 1)
model:add(nn.Dropout(0.5))

--model:add(nn.SpatialAveragePooling(2,2,2,2))
-----------------------------------------------------
print(model:forward(torch.rand(1, 3, 32, 32)):size())

ConvBNReLU(smallLayer, 8, 1, 1)
model:add(nn.View(8*4*4))
model:add(nn.Linear(8*4*4, 10))

He_init(model)

cudnn.convert(model, cudnn):cuda()
criterion = nn.CrossEntropyCriterion():cuda()


w, dE_dw = model:getParameters()
print('Number of parameters:', w:nElement())
print(model)


function shuffle(data,ydata) --shuffle data function
    local RandOrder = torch.randperm(data:size(1)):long()
    return data:index(1,RandOrder), ydata:index(1,RandOrder)
end

--  ****************************************************************
--  Training the network
--  ****************************************************************
require 'optim'

batchSize = 128
optimState = {}



---------------------------------------------------------------------

epochs = 500
trainLoss = torch.Tensor(epochs)
testLoss = torch.Tensor(epochs)
trainError = torch.Tensor(epochs)
testError = torch.Tensor(epochs)

for e = 1, epochs do
    trainData, trainLabels = shuffle(trainData, trainLabels) --shuffle training data
    trainLoss[e], trainError[e] = forwardNet(model, trainData, trainLabels, true)
    testLoss[e], testError[e], confusion = forwardNet(model, testData, testLabels, false)
    
    if e % 2 == 0 then
        print('Epoch ' .. e .. ':')
        print('Training error: ' .. trainError[e], 'Training Loss: ' .. trainLoss[e])
        print('Test error: ' .. testError[e], 'Test Loss: ' .. testLoss[e])
        print(confusion)
    end
end

plotError(trainError, testError, 'Classification Error')



