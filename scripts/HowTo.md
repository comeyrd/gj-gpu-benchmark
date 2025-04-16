# How to use the python scripts in this repo

## Custom Hipify

The goal is to overload Hipify with the custom Error Checking and Performance Analaysis tools.
So when a bug is finished in CUDA, calling "python custom_hipify.py" will search for bugs not 
transformed in HIP in the kernels/flawed/src directory, and hipify them.
Then will register it to the kernel registration function

It is a bit hacky but was a quick way to improve the development speed.

## New Bugg

The goal of this script is to create all the boiler plate to add a new buggy implementation of Gauss Jordan.

You need to call it with "python new_bugg.py [BIGRAM] [DESCRIPTION]"
The BIGRAM needs to be a two letter word, and DESCRIPTION could be whatever length you want (even empty).

It will create the gj-BIGRAM.cu with the reference implementation, create the gj-BIGRAM.hpp with the needed boilerplate
Inserts the header in gj-flawed.hpp and register the kernel in cuda-utils.cu

This is a bit hacky still, some things could be done for kernel registration with C++ and Static auto registring in the factory, but i don't have (yet) the developpement time to solve this hacky-ness