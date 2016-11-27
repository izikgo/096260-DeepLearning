USE_CUDA = true;
USE_CUDNN = true;

local mnist = require 'mnist'
require 'nn'
require 'optim'

-- helpers so we can run our code under CPU as well
if USE_CUDA then
  require 'cunn'; 
  Tensor = torch.CudaTensor
  if USE_CUDNN then
    require 'cudnn'
    cudnn.benchmark = true
    cudnn.fastest = true
  end
else
  Tensor = torch.Tensor
end

local localize = function(thing)
  if USE_CUDA then
    return thing:cuda()
  end
  return thing
end

---------------------------------------------------

local trainData = localize(mnist.traindataset().data:double() / 255.)
local trainLabels = mnist.traindataset().label:add(1)
local testData = localize(mnist.testdataset().data:double() / 255.)
local testLabels = mnist.testdataset().label:add(1)


----- ### Shuffling data

function shuffle(data, labels) --shuffle data function
    local randomIndexes = torch.randperm(data:size(1)):long() 
    return data:index(1,randomIndexes), labels:index(1,randomIndexes)
end

------   ### Define model and criterion

local model = nn.Sequential()
model:add(nn.View(1, 28, 28))

-- Conv block I
model:add(nn.SpatialConvolution(1, 32, 5, 5, 1, 1, 2, 2)) -- 1x28x28 in, 32x28x28 out
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2, 2, 2, 2)) -- 32x28x28 in, 32x14x14 out

-- Conv block II
model:add(nn.SpatialConvolution(32, 64, 3, 3, 1, 1, 1, 1)) -- 32x14x14 in, 64x14x14 out
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2, 2, 2, 2)) -- 64x14x14 in, 64x7x7 out

model:add(nn.View(64 * 7 * 7)) -- reshapes the image into a vector without copy

-- Softmax layer
model:add(nn.Linear(64 * 7 * 7, 10))
model:add(nn.LogSoftMax())

localize(model)

if USE_CUDNN then
  cudnn.convert(model, cudnn)
end

print(tostring(model))

local w, dE_dw = model:getParameters()
print('Number of parameters:', w:nElement()) --over-specified model


---- ### Classification criterion

criterion = localize(nn.ClassNLLCriterion())

---	 ### predefined constants


batchSize = 128

optimState = {
    learningRate = 0.1
    
}

--- ### Main evaluation + training function

function forwardNet(data, labels, train)
	timer = torch.Timer()

    --another helpful function of optim is ConfusionMatrix
    local confusion = optim.ConfusionMatrix(torch.range(0,9):totable())
    local lossAcc = 0
    local numBatches = 0
    if train then
        --set network into training mode
        model:training()
    end
    for i = 1, data:size(1) - batchSize, batchSize do
        numBatches = numBatches + 1
        local x = data:narrow(1, i, batchSize)
        local yt = labels:narrow(1, i, batchSize)
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
        
            optim.sgd(feval, w, optimState)
        end
    end
    
    confusion:updateValids()
    local avgLoss = lossAcc / numBatches
    local avgError = 1 - confusion.totalValid
	print(timer:time().real .. ' seconds')

    return avgLoss, avgError, tostring(confusion)
end


--- ### Train the network on training set, evaluate on separate set


epochs = 50

trainLoss = torch.Tensor(epochs)
testLoss = torch.Tensor(epochs)
trainError = torch.Tensor(epochs)
testError = torch.Tensor(epochs)

--reset net weights
model:apply(function(l) l:reset() end)

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



---		### Introduce momentum, L2 regularization
--reset net weights
--[[
model:apply(function(l) l:reset() end)

optimState = {
    learningRate = 0.1,
    momentum = 0.9,
    weightDecay = 1e-3
    
}
for e = 1, epochs do
    trainData, trainLabels = shuffle(trainData, trainLabels) --shuffle training data
    trainLoss[e], trainError[e] = forwardNet(trainData, trainLabels, true)
    testLoss[e], testError[e], confusion = forwardNet(testData, testLabels, false)
end

print('Training error: ' .. trainError[epochs], 'Training Loss: ' .. trainLoss[epochs])
print('Test error: ' .. testError[epochs], 'Test Loss: ' .. testLoss[epochs])
]]




--- ### Insert a Dropout layer
--[[
model:insert(nn.Dropout(0.9):cuda(), 8)


]]




-- ********************* Plots *********************
--[[
require 'gnuplot'
local range = torch.range(1, epochs)
gnuplot.pngfigure('test.png')
gnuplot.plot({'trainLoss',trainLoss},{'testLoss',testLoss})
gnuplot.xlabel('epochs')
gnuplot.ylabel('Loss')
gnuplot.plotflush()
]]














