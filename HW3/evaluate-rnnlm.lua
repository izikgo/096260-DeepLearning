require 'nngraph'
require 'rnn'
local dl = require 'dataload'


--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Evaluate a RNNLM')
cmd:text('Options:')
cmd:option('--xplogpath', '', 'path to a previously saved xplog containing model')
cmd:option('--cuda', false, 'model was saved with cuda')
cmd:option('--device', 1, 'which GPU device to use')
cmd:text()
local opt = cmd:parse(arg or {})

-- check that saved model exists
assert(paths.filep(opt.xplogpath), opt.xplogpath..' does not exist')

if opt.cuda then
   require 'cunn'
   cutorch.setDevice(opt.device)
end

local xplog = torch.load(opt.xplogpath)
local lm = xplog.model
local criterion = xplog.criterion
local targetmodule = xplog.targetmodule
local trainset, validset, testset = dl.loadPTB({xplog.opt.batchsize,1,1})

print("Hyper-parameters (xplog.opt):")
print(xplog.opt)

-- ###############################################
--                Plots
-- ###############################################

local function plot_error(trainError, valError, testError, title, ylabel)
	require 'gnuplot'
	local range = torch.range(1, trainError:size(1))
	gnuplot.pngfigure(title .. '.png')
	gnuplot.title(title)
	gnuplot.plot({'train',trainError},{'validation',valError},{'test',testError})
	gnuplot.xlabel('epochs')
	gnuplot.ylabel(ylabel)
	gnuplot.plotflush()
end

plot_error(torch.FloatTensor(xplog.trainppl), torch.FloatTensor(xplog.valppl), torch.FloatTensor(xplog.testppl), 'PerplexityConvergenceGraph', 'Perplexity')
-- ###############################################
--                Generate Sentences
-- ###############################################
lm:evaluate()

local nsentences = 5
local sentence_beginning = {'buy', 'low', 'sell', 'high', 'in', 'the'}

for i=1, nsentences do
	lm:forget()
	local sampletext = {table.unpack(sentence_beginning)}
	local inputs = torch.LongTensor(1,1) -- seqlen x batchsize
	if opt.cuda then inputs = inputs:cuda() end
	local buffer = torch.FloatTensor()
	local sentence_pos = 1
	local prevword = trainset.vocab[sampletext[sentence_pos]]
	assert(prevword)
	while prevword ~= trainset.vocab['<eos>'] do
		inputs:fill(prevword)
		local output = lm:forward(inputs)[1][1]
		if sentence_pos >= #sentence_beginning then
			buffer:resize(output:size()):copy(output)
			buffer:exp()
			local sample = torch.multinomial(buffer, 1, true)
			local currentword = trainset.ivocab[sample[1]]
			table.insert(sampletext, currentword)
			prevword = sample[1]
		else
			prevword = trainset.vocab[sampletext[sentence_pos+1]]
		end
		sentence_pos = sentence_pos + 1
	end
	print(table.concat(sampletext, ' '))
end