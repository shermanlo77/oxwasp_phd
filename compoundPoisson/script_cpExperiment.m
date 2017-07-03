clc;
close all;
clearvars;

rng(uint32(353759542),'twister');

n = 1000;
cpQQDensity(n, 1, 1, 1);
cpQQDensity(n, 1, 100, 1);
cpQQDensity(n, 100, 100, 1);

n = 100;
n_repeat = 10;
n_step = 5;
cpConvergence(n, 1, 1, 1, n_repeat, n_step);
cpConvergence(n, 1, 100, 1, n_repeat, n_step);
cpConvergence(n, 100, 100, 1, n_repeat, n_step);