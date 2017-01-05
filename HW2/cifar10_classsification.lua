require 'torch'
require 'image'
require 'nn'
require 'cunn'
require 'cudnn'

opt = lapp[[
   -b,--batchSize             (default 128)          batch size
   --optimizerSGD                                    optimizer sgd (or adam) 
   -m,--momentum              (default 0.9)          momentum
   --nesterov                                        use nesterov momentum
   --epoch_step               (default 25)           epoch step (only relevant with SGD)
   --max_epoch                (default 300)          maximum number of iterations
   --hflip                                           use hflip
   --random_crop                                     use random cropping
   --random_rotate                                   use random rotations
]]

cudnn.benchmark = true
cudnn.fastest = true

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

--  ****************************************************************
--  Data Augmetation
--  ****************************************************************
do -- data augmentation module
  local DataAugmentation,parent = torch.class('nn.DataAugmentation', 'nn.Module')

  function DataAugmentation:__init(hflip, randcrop, randrot)
    parent.__init(self)
    self.train = true
	self.hflip = hflip
	self.randcrop = randcrop
	self.randrot = randrot
  end

  function DataAugmentation:updateOutput(input)
    if self.train then
		local img_size = input[1]:size(2)  -- assume square inputs
		for i=1,input:size(1) do
			if self.hflip and torch.bernoulli() == 1 then  -- horizontal flip
				input[i] = image.hflip(input[i])
			end
			if self.randcrop and torch.bernoulli() == 1 then  -- random cropping
				local pad = 2
				local padded = image.scale(input[i], img_size + 2*pad, img_size + 2*pad)
				local x = torch.random(1,pad*2 + 1)
				local y = torch.random(1,pad*2 + 1)
				input[i] = padded:narrow(3,x,img_size):narrow(2,y,img_size)
			end
			if self.randrot and torch.bernoulli() == 1 then  -- random rotation
				local theta = (2*torch.rand(1)[{1}] - 1) * math.pi/8
				input[i] = image.rotate(input[i], theta, 'bilinear')
			end
		end		
    end
    self.output:set(input)
    return self.output
  end
end

--  ****************************************************************
--  Define our neural network
--  ****************************************************************

local function He_init(net)
  local function init(name)
    for k,v in pairs(net:findModules(name)) do
      local fan_in = v.kW*v.kH*v.nInputPlane
	  local fan_out = v.kW*v.kH*v.nOutputPlane
	  local n = fan_in + fan_out
      v.weight:normal(0, math.sqrt(2/n))
	  if v.bias then v.bias:zero() end
    end
  end
  init('nn.SpatialConvolution')
end

function forwardNet(model, data, labels, train)
    --another helpful function of optim is ConfusionMatrix
    local confusion = optim.ConfusionMatrix(classes)
    local lossAcc = 0
    local numBatches = 0
    local data_aug = nn.DataAugmentation(opt.hflip, opt.random_crop, opt.random_rotate):float()
	if train then
        --set network into training mode
        model:training()
		data_aug.train = true
    else
        model:evaluate() -- turn of drop-out
		data_aug.train = false
    end
    for i = 1, data:size(1) - batchSize, batchSize do
        numBatches = numBatches + 1
        local x = data_aug:forward(data:narrow(1, i, batchSize))
		x = x:cuda()
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
			
            if opt.optimizerSGD then
				optim.sgd(feval, w, optimState)
			else
				optim.adam(feval, w, optimState)
			end
        end
    end
    
    confusion:updateValids()
    local avgLoss = lossAcc / numBatches
    local avgError = 1 - confusion.totalValid
    
    return avgLoss, avgError, tostring(confusion)
end

function plotError(trainError, testError, title, ylabel)
	require 'gnuplot'
	local range = torch.range(1, trainError:size(1))
	gnuplot.pngfigure(title .. '.png')
	gnuplot.title(title)
	gnuplot.plot({'train',trainError},{'test',testError})
	gnuplot.xlabel('epochs')
	gnuplot.ylabel(ylabel)
	gnuplot.plotflush()
end

model = nn.Sequential()

-- building block
local function ConvBNReLU(...)
	local arg = {...}
	model:add(nn.SpatialConvolution(...))
	model:add(nn.SpatialBatchNormalization(arg[2], 1e-3))
	model:add(nn.LeakyReLU(0.3, true))
	return model
end

main_depth = 42
medium_depth = main_depth * 5 / 6
small_depth = medium_depth * 3 / 5

ConvBNReLU(3,main_depth,5,5,1,1,2,2)
ConvBNReLU(main_depth,medium_depth,1,1)
ConvBNReLU(medium_depth,small_depth,1,1)
model:add(nn.SpatialMaxPooling(3,3,2,2):ceil())
model:add(nn.Dropout(0.2))
ConvBNReLU(small_depth,main_depth,5,5,1,1,2,2)
ConvBNReLU(main_depth,main_depth,1,1)
ConvBNReLU(main_depth,main_depth,1,1)
model:add(nn.SpatialAveragePooling(3,3,2,2):ceil())
model:add(nn.Dropout(0.2))
ConvBNReLU(main_depth,main_depth,3,3,1,1,1,1)
ConvBNReLU(main_depth,main_depth,1,1)
ConvBNReLU(main_depth,10,1,1)
model:add(nn.SpatialAveragePooling(8,8,1,1):ceil())
model:add(nn.View(10))

He_init(model)

cudnn.convert(model, cudnn):cuda()
criterion = nn.CrossEntropyCriterion():cuda()


w, dE_dw = model:getParameters()
print('Number of parameters:', w:nElement())
--print(model)


function shuffle(data,ydata) --shuffle data function
    local RandOrder = torch.randperm(data:size(1)):long()
    return data:index(1,RandOrder), ydata:index(1,RandOrder)
end

--  ****************************************************************
--  Training the network
--  ****************************************************************
require 'optim'

batchSize = opt.batchSize
if opt.optimizerSGD then
	optimState = {learningRate = 0.1, nesterov = opt.nesterov, dampening = 0, momentum = opt.momentum}
else
	optimState = {}
end

epochs = opt.max_epoch
trainLoss = torch.Tensor(epochs)
testLoss = torch.Tensor(epochs)
trainError = torch.Tensor(epochs)
testError = torch.Tensor(epochs)

best_error = 1
best_epoch = 0

for e = 1, epochs do
    trainData, trainLabels = shuffle(trainData, trainLabels) --shuffle training data
    
	if opt.optimizerSGD and e % opt.epoch_step == 0 then
		optimState.learningRate = optimState.learningRate / 2
	end
	
	forwardNet(model, trainData, trainLabels, true)
    trainLoss[e], trainError[e] = forwardNet(model, trainData, trainLabels, false)
	testLoss[e], testError[e], confusion = forwardNet(model, testData, testLabels, false)
    
    if e % 2 == 0 then
        print('Epoch ' .. e .. ':')
        print('Training error: ' .. trainError[e], 'Training Loss: ' .. trainLoss[e])
        print('Test error: ' .. testError[e], 'Test Loss: ' .. testLoss[e])
        print(confusion)
    end
	
	if testError[e] < best_error then
		best_error = testError[e]
		best_epoch = e
		-- Save the model
		model:clearState()
		torch.save('cifar10_model.dat', model)
	end
end

optimizer_str = opt.optimizerSGD and 'sgd' or 'adam'
plotError(trainError, testError, 'Classification Error (' .. optimizer_str .. ', best epoch - ' .. best_epoch .. ', best accuracy - ' .. string.format('%.2f', (1-best_error)*100) .. ')', 'Error')
plotError(trainLoss, testLoss, 'Classification Loss (' .. optimizer_str .. ')', 'Loss')
