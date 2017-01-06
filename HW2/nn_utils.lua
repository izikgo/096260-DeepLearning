local classes = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

function forwardNet(model, data, labels, batchSize)
    --another helpful function of optim is ConfusionMatrix
    local confusion = optim.ConfusionMatrix(classes)
    local lossAcc = 0
    local numBatches = 0
    
	model:evaluate()
    
	for i = 1, data:size(1) - batchSize, batchSize do
        numBatches = numBatches + 1
        local x = data:narrow(1, i, batchSize):cuda()
        local yt = labels:narrow(1, i, batchSize):cuda()
        local y = model:forward(x)
        local err = nn.CrossEntropyCriterion():cuda():forward(y, yt)
        lossAcc = lossAcc + err
        confusion:batchAdd(y,yt)
    end
    
    confusion:updateValids()
    local avgLoss = lossAcc / numBatches
    local avgError = 1 - confusion.totalValid
    
    return avgLoss, avgError, tostring(confusion)
end


-- The file 'mnist_model.dat' should be present in the directory
function getAverageErrorOnTest()
    require 'nn'
    require 'optim'
    require 'cunn'
    require 'cudnn'
    cudnn.benchmark = true
	cudnn.fastest = true

	local trainset = torch.load('cifar.torch/cifar10-train.t7')
	local testset = torch.load('cifar.torch/cifar10-test.t7')

	local classes = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

	local trainData = trainset.data:float() -- convert the data from a ByteTensor to a float Tensor.
	local trainLabels = trainset.label:float():add(1)
	local testData = testset.data:float()
	local testLabels = testset.label:float():add(1)

	local function normalize(train_data, test_data, i)
		local train_mean = train_data[{ {}, i, {}, {}}]:mean()
		local train_std = train_data[{ {}, i, {}, {}}]:std()
		train_data[{ {}, i, {}, {}}]:add(-train_mean)
		train_data[{ {}, i, {}, {}}]:div(train_std)
		test_data[{ {}, i, {}, {}}]:add(-train_mean)
		test_data[{ {}, i, {}, {}}]:div(train_std)
	end

	for i = 1, 3 do
		normalize(trainData, testData, i)
	end
	
    local model = torch.load('cifar10_model_85.5.dat')
    local batchSize = 128

    testLoss, testError = forwardNet(model, testData, testLabels, batchSize, optimState, criterion) -- evaluate on test
    return testError
end


