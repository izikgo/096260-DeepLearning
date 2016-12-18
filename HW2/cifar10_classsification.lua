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

-- convert to YUV and normalize


--  ****************************************************************
--  Define our neural network
--  ****************************************************************

local function addConvBlockFMP(model, nInput, nFilter, size, poolingRatio)
	model:add(nn.SpatialConvolution(nInput, nFilter, size, size))
	model:add(nn.SpatialFractionalMaxPooling(2, 2, poolingRatio, poolingRatio))
	model:add(nn.LeakyReLU(0.3, true))
end

local function calcNumberOfFilters(nLayer)
	local filterBase = 8
	local filterAdd = 5
	return filterBase + filterAdd * (nLayer - 1)
end

inputSize = 128
nConvLayers = 9
-- filtersBase = 10
-- filterAdd = 5
poolingRatio = 1 / math.pow(2, 1/3)

local model = nn.Sequential()

addConvBlockFMP(model, 3, calcNumberOfFilters(1), 2, poolingRatio)
for l = 2, nConvLayers do
	addConvBlockFMP(model, calcNumberOfFilters(l - 1), calcNumberOfFilters(l), 2, poolingRatio)
end

model:add(nn.SpatialConvolution(calcNumberOfFilters(nConvLayers), calcNumberOfFilters(nConvLayers), 2, 2))
model:add(nn.SpatialFractionalMaxPooling(2, 2, 1 / math.pow(2, 0.5), 1 / math.pow(2, 0.5)))
model:add(nn.SpatialConvolution(calcNumberOfFilters(nConvLayers), calcNumberOfFilters(nConvLayers), 1, 1))
model:add(nn.SpatialFractionalMaxPooling(2, 2, 1 / math.pow(2, 0.5), 1 / math.pow(2, 0.5)))
model:add(nn.View(-1):setNumInputDims(3))

nElementsConv = model:forward(torch.rand(3, inputSize, inputSize)):nElement()

model:add(nn.Linear(nElementsConv, #classes))

model:cuda()

cudnn.convert(model, cudnn)
criterion = nn.CrossEntropyCriterion():cuda()


w, dE_dw = model:getParameters()
print('Number of parameters:', w:nElement())
print(model)

--[[
function shuffle(data,ydata) --shuffle data function
    local RandOrder = torch.randperm(data:size(1)):long()
    return data:index(1,RandOrder), ydata:index(1,RandOrder)
end

--  ****************************************************************
--  Training the network
--  ****************************************************************
require 'optim'

local batchSize = 128
local optimState = {}

function forwardNet(data,labels, train)
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

---------------------------------------------------------------------

epochs = 25
trainLoss = torch.Tensor(epochs)
testLoss = torch.Tensor(epochs)
trainError = torch.Tensor(epochs)
testError = torch.Tensor(epochs)

--reset net weights
model:apply(function(l) l:reset() end)

timer = torch.Timer()

for e = 1, epochs do
    trainData, trainLabels = shuffle(trainData, trainLabels) --shuffle training data
    trainLoss[e], trainError[e] = forwardNet(trainData, trainLabels, true)
    testLoss[e], testError[e], confusion = forwardNet(testData, testLabels, false)
    
    if e % 5 == 0 then
        print('Epoch ' .. e .. ':')
        print('Training error: ' .. trainError[e], 'Training Loss: ' .. trainLoss[e])
        print('Test error: ' .. testError[e], 'Test Loss: ' .. testLoss[e])
        print(confusion)
    end
end

plotError(trainError, testError, 'Classification Error')


--  ****************************************************************
--  Network predictions
--  ****************************************************************


model:evaluate()   --turn off dropout

print(classes[testLabels[10] ])
print(testData[10]:size())
saveTensorAsGrid(testData[10],'testImg10.jpg')
local predicted = model:forward(testData[10]:view(1,3,32,32):cuda())
print(predicted:exp()) -- the output of the network is Log-Probabilities. To convert them to probabilities, you have to take e^x 

-- assigned a probability to each classes
for i=1,predicted:size(2) do
    print(classes[i],predicted[1][i])
end


]]

