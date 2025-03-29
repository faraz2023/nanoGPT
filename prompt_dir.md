


Consider our gpt implementation for large langugage processing. take a look at the code in @PG_model_parcellation.py @model.py to see how the gpt model is impllemented. You can also look at @z.txt  to review the outputs of running:

python PG_model_parcellation.py --init_from=gpt2


we want to do the following:

1. we want to create a edge list of gelu activation neurons. 
you see that the output of GELU activation is typically of:

[batch_size, num_tokens, embd_dim]. 

For ease of use, we consider the last set of neuron embeddings as the set of interest

[batch_size, [-1], embd_dim]. 

for further ease of use, assume abtch_size == 1
naturally, all of the embd_dim GELO neurons are connected to their previous GELU neorons and their next GELU neurons layers. 

Your output should be .el file where nodes have integer ids (0,1,2...) and a csv file where for each node i we have attributes (e.g., transformer layer, neuron dim number, any other thing of signifcance). 

2. Subsequently (do not implement this yet because I want to make sure you do 1 first correctly but I want you to reflect and keep this in mind), for examples of inputs, we want to see which activation neurons get activate and match them to the graph, therefore trying to predict neuron activatio jbase don input mood. 


-----


Ok this is a good start with @trace_gelu_activations.py . Now we want to get into seriously generating datasets. 

We want a script that recives a input csv path and export_path directory. if the export path does not exists it creates the folder and:
Given inputfiles in csv format such as @test_1.csv (where each item is a prompt and max number of expected tokens) we want to create a dataset of neuron activations. the output dataset should be a csv file where rows are:

sample_id,input_text,output(next token in str), input_text_ids,output_id,number_of_activate_neorons,layer_0_num_active_neurons,...,level_n_active_neurons

and a data file that holds the list of activate neurons (sorted) for each of the outputs. Note that for input cases, we have a numb_max_tokens, so for each row in the input file, there will be multiple samples as the model goes through it's response process. Choose a dataformat that is compact and gives fast read access for downstream analytics