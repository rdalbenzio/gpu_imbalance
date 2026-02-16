# prompt.md

This file contains the prompt used to generate a performance test script able to create unbalancing between different GPUs.

Create a script used to create imbalance between n GPUs when a round robin load balancing is used in front of them.

Use the following algorithm

- memorize current time
- Generate and send a prompt and send it of length 2^n words for the GPU0, 2^(n-1) for GPU1 up to 2^0 for the GPUn. 
After M loops start generating the prompts as per the following.
   - retrieve the current concurrency of the GPU that is supposed to be selected by the load balancer
   - generate and send a prompt of lenght concurrency^2 with a maximum value of max_prompt, default 65536 words.
   - store the prompt generated in a txt file called prompts.txt
- continue the loop until T number of seconds passed, where T by default equal to 3600.
- the loop above needs to generate prompts with a predefined max_concurrency, with a default value of 512.
- at the end of the loop print a table with the following numbers:
   - total amounts of prompt sent
   - total amount of prompts received
   - total amount of prompts failed
   - total amount of tokens generated
   - total amount ot tokens received
   - avg tokens/sec generated
   - avg tokens/sec received

Generate a second scripts which, instead of generating the prompts, it uses the content of the file prompts.txt that has been generated before. Make the concurrency a variable with same default as the previous script, print the same table at the end. 
 




     