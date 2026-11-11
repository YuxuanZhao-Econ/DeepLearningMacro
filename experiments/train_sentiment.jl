const SENTIMENT_STARTED=time()
using LinearAlgebra, Printf
BLAS.set_num_threads(1)
include(joinpath(@__DIR__,"..","src","SentimentLab.jl"))
root=normpath(joinpath(@__DIR__,".."))
data=SentimentLab.prepare_data(joinpath(root,"reference","datasets","imdb"))
println("Prepared splits: ",data.metadata)
flush(stdout)
run=SentimentLab.fit(data)
SentimentLab.save_run(run,joinpath(root,"output","sentiment_imdb"))
@printf("Test accuracy %.2f%% | total process %.1f s | fit %.1f s\n",
    100run.test.accuracy,time()-SENTIMENT_STARTED,run.elapsed_seconds)
write(joinpath(root,"output","sentiment_imdb","process_time.txt"),string(time()-SENTIMENT_STARTED))
