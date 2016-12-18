require 'optim'

function forwardNet(model, data, labels, train, batchSize, optimizer, optimState, criterion)
    --another helpful function of optim is ConfusionMatrix
    local confusion = optim.ConfusionMatrix(classes)
    local lossAcc = 0
    local numBatches = 0
    if train then
        --set network into training mode
        model:training()
    else
        model:evaluate()
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
        
            optimizer(feval, w, optimState)
        end
    end
    
    confusion:updateValids()
    local avgLoss = lossAcc / numBatches
    local avgError = 1 - confusion.totalValid
    
    return avgLoss, avgError, tostring(confusion)
end


-- The file 'mnist_model.dat' should be present in the directory
function getAverageErrorOnTest()
    local mnist = require 'mnist'
    require 'nn'
    require 'optim'
    require 'cunn'
    require 'cudnn'
    cudnn.benchmark = true
    cudnn.fastest = true

    local testData = (mnist.testdataset().data:double() / 255.):cuda()
    local testLabels = mnist.testdataset().label:add(1)
    local model = torch.load('mnist_model.dat')
    local batchSize = 128
    local optimState = {learningRate = 0.01}
    local criterion = nn.CrossEntropyCriterion():cuda()

    testLoss, testError = forwardNet(model, testData, testLabels, false, batchSize, optimState, criterion) -- evaluate on test
    return testError
end

