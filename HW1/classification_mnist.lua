USE_CUDA = true;
USE_CUDNN = true;

local mnist = require 'mnist'
require 'nn'
require 'optim'
require 'nn_utils'

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

-- Dropout layer
model:add(nn.Dropout(0.4))

-- FC layer I
model:add(nn.Linear(64 * 7 * 7, 14))

-- Dropout layer
model:add(nn.Dropout(0.4))

-- FC layer II
model:add(nn.Linear(14, 10))

localize(model)

if USE_CUDNN then
  cudnn.convert(model, cudnn)
end

print(tostring(model))

local w, dE_dw = model:getParameters()
print('Number of parameters:', w:nElement())


criterion = localize(nn.CrossEntropyCriterion())


batchSize = 128
optimState = {
    learningRate = 0.01,
}


--- ### Train the network on training set, evaluate on separate set

epochs = 600

trainLoss = torch.Tensor(epochs)
testLoss = torch.Tensor(epochs)
trainError = torch.Tensor(epochs)
testError = torch.Tensor(epochs)

for e = 1, epochs do
    trainData, trainLabels = shuffle(trainData, trainLabels) --shuffle training data
    forwardNet(model, trainData, trainLabels, true, batchSize, optimState, criterion) -- train the network
    trainLoss[e], trainError[e] = forwardNet(model, trainData, trainLabels, false, batchSize, optimState, criterion) -- evaluate on train
    testLoss[e], testError[e], confusion = forwardNet(model, testData, testLabels, false, batchSize, optimState, criterion) -- evaluate on test
    
    if e % 10 == 0 then
        print('Epoch ' .. e .. ':')
        print('Training error: ' .. trainError[e], 'Training Loss: ' .. trainLoss[e])
        print('Test error: ' .. testError[e], 'Test Loss: ' .. testLoss[e])
        print(confusion)
        print('########################################################################')
   end
end

-- Save the model
torch.save('mnist_model.dat', model)

-- ********************* Plots *********************

require 'gnuplot'
gnuplot.pngfigure('Loss.png')
gnuplot.logscale(true)
gnuplot.plot({'trainLoss',trainLoss},{'testLoss',testLoss})
gnuplot.xlabel('epochs')
gnuplot.ylabel('Loss')
gnuplot.plotflush()

gnuplot.pngfigure('Error.png')
gnuplot.logscale(true)
gnuplot.plot({'trainError',trainError},{'testError',testError})
gnuplot.xlabel('epochs')
gnuplot.ylabel('Error')
gnuplot.plotflush()

